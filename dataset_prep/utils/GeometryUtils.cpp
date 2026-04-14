#include "dataset_prep/utils/GeometryUtils.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace dataset_prep {
namespace {

constexpr float kTwoPi = 2.0f * static_cast<float>(CV_PI);

bool IsFinitePoint3(const cv::Point3f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

}  // namespace

cv::Point2f Rotate2D(const cv::Point2f& point, float rot_rad) {
    const float sn = std::sin(rot_rad);
    const float cs = std::cos(rot_rad);
    return cv::Point2f(point.x * cs - point.y * sn,
                       point.x * sn + point.y * cs);
}

cv::Matx23f GenTransFromPatchCv(float c_x,
                                float c_y,
                                float src_width,
                                float src_height,
                                int dst_width,
                                int dst_height,
                                float scale,
                                float rot_deg,
                                bool inv) {
    const float src_w = src_width * scale;
    const float src_h = src_height * scale;
    const cv::Point2f src_center(c_x, c_y);
    const float rot_rad = static_cast<float>(CV_PI) * rot_deg / 180.0f;
    const cv::Point2f src_downdir = Rotate2D(cv::Point2f(0.0f, src_h * 0.5f), rot_rad);
    const cv::Point2f src_rightdir = Rotate2D(cv::Point2f(src_w * 0.5f, 0.0f), rot_rad);

    const cv::Point2f dst_center(dst_width * 0.5f, dst_height * 0.5f);
    const cv::Point2f dst_downdir(0.0f, dst_height * 0.5f);
    const cv::Point2f dst_rightdir(dst_width * 0.5f, 0.0f);

    std::array<cv::Point2f, 3> src = {
        src_center,
        src_center + src_downdir,
        src_center + src_rightdir,
    };
    std::array<cv::Point2f, 3> dst = {
        dst_center,
        dst_center + dst_downdir,
        dst_center + dst_rightdir,
    };

    const cv::Mat affine = inv
        ? cv::getAffineTransform(dst.data(), src.data())
        : cv::getAffineTransform(src.data(), dst.data());
    cv::Matx23d affine_d;
    affine.copyTo(affine_d);
    return cv::Matx23f(static_cast<float>(affine_d(0, 0)),
                       static_cast<float>(affine_d(0, 1)),
                       static_cast<float>(affine_d(0, 2)),
                       static_cast<float>(affine_d(1, 0)),
                       static_cast<float>(affine_d(1, 1)),
                       static_cast<float>(affine_d(1, 2)));
}

cv::Matx34f BuildProjectionMatrix(const CameraCalibration& calibration) {
    cv::Matx34f rt(calibration.R(0, 0), calibration.R(0, 1), calibration.R(0, 2), calibration.t[0],
                   calibration.R(1, 0), calibration.R(1, 1), calibration.R(1, 2), calibration.t[1],
                   calibration.R(2, 0), calibration.R(2, 1), calibration.R(2, 2), calibration.t[2]);
    return calibration.K * rt;
}

bool TriangulatePointDLT(const std::vector<cv::Matx34f>& projections,
                         const std::vector<cv::Point2f>& points,
                         cv::Point3f* out_point) {
    if (out_point == nullptr || projections.size() < 2u || projections.size() != points.size()) {
        return false;
    }

    cv::Mat A(static_cast<int>(projections.size()) * 2, 4, CV_32F);
    for (size_t i = 0; i < projections.size(); ++i) {
        const cv::Matx34f& P = projections[i];
        const float u = points[i].x;
        const float v = points[i].y;
        for (int c = 0; c < 4; ++c) {
            A.at<float>(static_cast<int>(2 * i), c) = u * P(2, c) - P(0, c);
            A.at<float>(static_cast<int>(2 * i + 1), c) = v * P(2, c) - P(1, c);
        }
    }

    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    const cv::Mat homog = svd.vt.row(3).t();
    const float w = homog.at<float>(3, 0);
    if (!std::isfinite(w) || std::abs(w) <= 1e-8f) {
        return false;
    }

    out_point->x = homog.at<float>(0, 0) / w;
    out_point->y = homog.at<float>(1, 0) / w;
    out_point->z = homog.at<float>(2, 0) / w;
    return IsFinitePoint3(*out_point);
}

bool ProjectWorldPointToImage(const CameraCalibration& calibration,
                              const cv::Point3f& point_world,
                              cv::Point2f* out_pixel) {
    if (out_pixel == nullptr) {
        return false;
    }

    const cv::Vec3f Xw(point_world.x, point_world.y, point_world.z);
    const cv::Vec3f Xc = calibration.R * Xw + calibration.t;
    if (!std::isfinite(Xc[2]) || Xc[2] <= 1e-6f) {
        return false;
    }
    const cv::Vec3f uvw = calibration.K * Xc;
    if (!std::isfinite(uvw[2]) || std::abs(uvw[2]) <= 1e-8f) {
        return false;
    }
    *out_pixel = cv::Point2f(uvw[0] / uvw[2], uvw[1] / uvw[2]);
    return true;
}

cv::Vec3f SmoothAxisAngleRotation(const cv::Vec3f& previous_axis_angle,
                                  const cv::Vec3f& current_axis_angle,
                                  float alpha,
                                  float max_step_radians) {
    cv::Matx33f previous_rotation;
    cv::Matx33f current_rotation;
    cv::Rodrigues(previous_axis_angle, previous_rotation);
    cv::Rodrigues(current_axis_angle, current_rotation);

    const cv::Matx33f relative_rotation = current_rotation * previous_rotation.t();
    cv::Vec3f relative_axis_angle;
    cv::Rodrigues(cv::Mat(relative_rotation), relative_axis_angle);

    const float relative_angle = static_cast<float>(cv::norm(relative_axis_angle));
    if (relative_angle <= 1e-8f) {
        return previous_axis_angle;
    }

    float step_scale = std::clamp(alpha, 0.0f, 1.0f);
    if (max_step_radians > 0.0f) {
        step_scale = std::min(step_scale, max_step_radians / relative_angle);
    }

    const cv::Vec3f step_axis_angle = relative_axis_angle * step_scale;
    cv::Matx33f step_rotation;
    cv::Rodrigues(step_axis_angle, step_rotation);

    const cv::Matx33f smoothed_rotation = step_rotation * previous_rotation;
    cv::Vec3f smoothed_axis_angle;
    cv::Rodrigues(cv::Mat(smoothed_rotation), smoothed_axis_angle);

    const float smoothed_angle = static_cast<float>(cv::norm(smoothed_axis_angle));
    if (smoothed_angle <= 1e-8f) {
        return smoothed_axis_angle;
    }

    const cv::Vec3f axis = smoothed_axis_angle * (1.0f / smoothed_angle);
    const cv::Vec3f alt_axis = -axis;
    std::array<cv::Vec3f, 4> candidates = {
        smoothed_axis_angle,
        axis * (smoothed_angle - kTwoPi),
        axis * (smoothed_angle + kTwoPi),
        alt_axis * (kTwoPi - smoothed_angle),
    };

    cv::Vec3f best_axis_angle = candidates[0];
    float best_distance = static_cast<float>(cv::norm(candidates[0] - previous_axis_angle));
    for (size_t candidate_index = 1; candidate_index < candidates.size(); ++candidate_index) {
        const float distance =
            static_cast<float>(cv::norm(candidates[candidate_index] - previous_axis_angle));
        if (distance < best_distance) {
            best_distance = distance;
            best_axis_angle = candidates[candidate_index];
        }
    }
    return best_axis_angle;
}

void SmoothAxisAngleBlocks(const std::vector<float>& previous_values,
                           std::vector<float>* current_values,
                           float alpha,
                           float max_step_radians) {
    if (current_values == nullptr || current_values->size() < 3u ||
        previous_values.size() < current_values->size()) {
        return;
    }

    for (size_t offset = 0; offset + 2u < current_values->size(); offset += 3u) {
        const cv::Vec3f previous_axis_angle(previous_values[offset + 0u],
                                            previous_values[offset + 1u],
                                            previous_values[offset + 2u]);
        const cv::Vec3f current_axis_angle((*current_values)[offset + 0u],
                                           (*current_values)[offset + 1u],
                                           (*current_values)[offset + 2u]);
        const cv::Vec3f smoothed_axis_angle =
            SmoothAxisAngleRotation(previous_axis_angle,
                                    current_axis_angle,
                                    alpha,
                                    max_step_radians);
        (*current_values)[offset + 0u] = smoothed_axis_angle[0];
        (*current_values)[offset + 1u] = smoothed_axis_angle[1];
        (*current_values)[offset + 2u] = smoothed_axis_angle[2];
    }
}

cv::Vec3f SmoothTranslationStep(const cv::Vec3f& previous_translation,
                                const cv::Vec3f& current_translation,
                                float alpha,
                                float max_step) {
    cv::Vec3f delta = current_translation - previous_translation;
    const float delta_norm = static_cast<float>(cv::norm(delta));
    if (delta_norm <= 1e-8f) {
        return previous_translation;
    }

    float step_scale = std::clamp(alpha, 0.0f, 1.0f);
    if (max_step > 0.0f) {
        step_scale = std::min(step_scale, max_step / delta_norm);
    }
    return previous_translation + delta * step_scale;
}

}  // namespace dataset_prep
