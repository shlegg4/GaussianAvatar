#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "dataset_prep/ingestion/CameraCalibrationLoader.h"
#include "dataset_prep/ingestion/VideoSynchronizer.h"
#include "dataset_prep/processing/PearEstimator.h"
#include "dataset_prep/processing/BackgroundExtractor.h" 
#include "utils/RtmPoseDetector.h"
#include "utils/YoloPersonDetector.h"
#if DATASET_PREP_HAS_SMPL
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace fs = std::filesystem;
using namespace dataset_prep;

// --- UTILITY FUNCTIONS ---

namespace {

constexpr int kPearInputRes = 256;
constexpr float kPearCameraFocal = 24.0f;
constexpr int64_t kSmplxShapeParamCount = 10;
constexpr int64_t kSmplxExpressionParamCount = 240;
constexpr int64_t kSmplxGlobalOrientParamCount = 3;
constexpr int64_t kSmplxBodyPoseParamCount = 63;
constexpr int64_t kSmplxJawPoseParamCount = 3;
constexpr int64_t kSmplxEyePoseParamCount = 6;
constexpr int64_t kSmplxHandPoseParamCount = 45;
constexpr int kSmplxRootJointIndex = 0;
constexpr float kTriangulationMinScore = 0.25f;

struct JointMapEntry {
    int mocap_index;
    int smpl_index;
};

const std::array<JointMapEntry, 20> kMocapToSmplJointMap = {{
    {11, 1}, {12, 2}, {13, 4}, {14, 5}, {15, 7}, {16, 8}, {18, 12},
    {17, 15}, {5, 16}, {6, 17}, {7, 18}, {8, 19}, {9, 20}, {10, 21},
    {20, 10}, {22, 10}, {24, 10}, {21, 11}, {23, 11}, {25, 11},
}};

std::vector<float> PadFloatVector(const std::vector<float>& values, size_t expected_count) {
    std::vector<float> padded(expected_count, 0.0f);
    const size_t copy_count = std::min(values.size(), expected_count);
    std::copy_n(values.begin(), copy_count, padded.begin());
    return padded;
}

#if DATASET_PREP_HAS_SMPL
torch::Tensor MakeTorchInputTensor(const std::vector<float>& values,
                                   int64_t expected_count,
                                   const torch::Device& device) {
    std::vector<float> padded = PadFloatVector(values, static_cast<size_t>(expected_count));
    auto tensor = torch::from_blob(
        padded.data(),
        {1, expected_count},
        torch::TensorOptions().dtype(torch::kFloat32));
    return tensor.clone().to(device);
}
#endif

bool SanitizeBBox(const cv::Rect2f& bbox, int img_width, int img_height, cv::Rect2f* out_bbox) {
    if (out_bbox == nullptr) {
        return false;
    }

    const float x1 = std::max(0.0f, bbox.x);
    const float y1 = std::max(0.0f, bbox.y);
    const float x2 = std::min(static_cast<float>(img_width - 1),
                              x1 + std::max(0.0f, bbox.width - 1.0f));
    const float y2 = std::min(static_cast<float>(img_height - 1),
                              y1 + std::max(0.0f, bbox.height - 1.0f));
    if (bbox.width * bbox.height <= 0.0f || x2 <= x1 || y2 <= y1) {
        return false;
    }

    *out_bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    return true;
}

bool ProcessBBox(const cv::Rect2f& bbox,
                 int img_width,
                 int img_height,
                 const cv::Size& input_shape,
                 float ratio,
                 cv::Rect2f* out_bbox) {
    if (out_bbox == nullptr) {
        return false;
    }

    cv::Rect2f clipped_bbox;
    if (!SanitizeBBox(bbox, img_width, img_height, &clipped_bbox)) {
        return false;
    }

    float w = clipped_bbox.width;
    float h = clipped_bbox.height;
    const float c_x = clipped_bbox.x + w * 0.5f;
    const float c_y = clipped_bbox.y + h * 0.5f;
    const float aspect_ratio = static_cast<float>(input_shape.width) /
                               static_cast<float>(input_shape.height);
    if (w > aspect_ratio * h) {
        h = w / aspect_ratio;
    } else if (w < aspect_ratio * h) {
        w = h * aspect_ratio;
    }

    const float expanded_w = w * ratio;
    const float expanded_h = h * ratio;
    *out_bbox = cv::Rect2f(c_x - expanded_w * 0.5f,
                           c_y - expanded_h * 0.5f,
                           expanded_w,
                           expanded_h);
    return true;
}

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
                                bool inv = false) {
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

bool GeneratePatchImage(const cv::Mat& image,
                        const cv::Rect2f& bbox,
                        float scale,
                        float rot_deg,
                        bool do_flip,
                        const cv::Size& out_shape,
                        cv::Mat* out_patch,
                        cv::Matx23f* out_trans,
                        cv::Matx23f* out_inv_trans) {
    if (out_patch == nullptr || image.empty()) {
        return false;
    }

    cv::Mat src = image;
    float bb_c_x = bbox.x + bbox.width * 0.5f;
    const float bb_c_y = bbox.y + bbox.height * 0.5f;
    const float bb_width = bbox.width;
    const float bb_height = bbox.height;

    if (do_flip) {
        cv::flip(image, src, 1);
        bb_c_x = static_cast<float>(image.cols) - bb_c_x - 1.0f;
    }

    const cv::Matx23f trans = GenTransFromPatchCv(
        bb_c_x, bb_c_y, bb_width, bb_height,
        out_shape.width, out_shape.height, scale, rot_deg, false);
    cv::warpAffine(src,
                   *out_patch,
                   cv::Mat(trans),
                   out_shape,
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    if (out_trans != nullptr) {
        *out_trans = trans;
    }
    if (out_inv_trans != nullptr) {
        *out_inv_trans = GenTransFromPatchCv(
            bb_c_x, bb_c_y, bb_width, bb_height,
            out_shape.width, out_shape.height, scale, rot_deg, true);
    }
    return true;
}

cv::Matx33f DefaultPearCameraRotation() {
    return cv::Matx33f(-1.0f, 0.0f, 0.0f,
                       0.0f, -1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Matx33f ProjectionSignFlip() {
    return cv::Matx33f(-1.0f, 0.0f, 0.0f,
                       0.0f, -1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Matx33f ExtractPearCameraRotation(const SmplxResult& pear_result) {
    if (pear_result.camera_rt.size() >= 16u) {
        return cv::Matx33f(pear_result.camera_rt[0], pear_result.camera_rt[1], pear_result.camera_rt[2],
                           pear_result.camera_rt[4], pear_result.camera_rt[5], pear_result.camera_rt[6],
                           pear_result.camera_rt[8], pear_result.camera_rt[9], pear_result.camera_rt[10]);
    }
    return DefaultPearCameraRotation();
}

cv::Vec3f ExtractPearCameraTranslation(const SmplxResult& pear_result) {
    if (pear_result.camera_rt.size() >= 16u) {
        return cv::Vec3f(pear_result.camera_rt[3],
                         pear_result.camera_rt[7],
                         pear_result.camera_rt[11]);
    }
    if (pear_result.camera_translation.size() >= 3u) {
        return cv::Vec3f(pear_result.camera_translation[0],
                         pear_result.camera_translation[1],
                         pear_result.camera_translation[2]);
    }
    return cv::Vec3f(0.0f, 0.0f, 0.0f);
}

cv::Point2f ProjectPearCameraPoint(const cv::Vec3f& point_cam, int width, int height) {
    const float half_w = static_cast<float>(width) * 0.5f;
    const float half_h = static_cast<float>(height) * 0.5f;
    const float focal_x = half_w * kPearCameraFocal;
    const float focal_y = half_h * kPearCameraFocal;
    return cv::Point2f(half_w - focal_x * (point_cam[0] / point_cam[2]),
                       half_h - focal_y * (point_cam[1] / point_cam[2]));
}

void ApplyExtraRootRotation(const cv::Matx33f& extra_rotation, std::vector<float>* pose_axis_angle) {
    if (pose_axis_angle == nullptr || pose_axis_angle->size() < 3u) {
        return;
    }

    cv::Vec3f root_aa((*pose_axis_angle)[0], (*pose_axis_angle)[1], (*pose_axis_angle)[2]);
    cv::Matx33f root_rot;
    cv::Rodrigues(root_aa, root_rot);

    const cv::Matx33f rotated_root = extra_rotation * root_rot;
    cv::Vec3f rotated_root_aa;
    cv::Rodrigues(cv::Mat(rotated_root), rotated_root_aa);

    (*pose_axis_angle)[0] = rotated_root_aa[0];
    (*pose_axis_angle)[1] = rotated_root_aa[1];
    (*pose_axis_angle)[2] = rotated_root_aa[2];
}

struct TrainingCropMetadata {
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    float crop_x0 = 0.0f;
    float crop_y0 = 0.0f;
    float crop_w = 0.0f;
    float crop_h = 0.0f;
    float focal_length = 0.0f;
};

TrainingCropMetadata BuildTrainingCropMetadata(const cv::Rect2f& crop_bbox,
                                               int img_width,
                                               int img_height,
                                               int target_crop_res) {
    TrainingCropMetadata metadata;

    if (crop_bbox.width <= 1e-6f || crop_bbox.height <= 1e-6f || target_crop_res <= 0) {
        return metadata;
    }

    const float full_cx = static_cast<float>(img_width) * 0.5f;
    const float full_cy = static_cast<float>(img_height) * 0.5f;
    const float crop_half = static_cast<float>(target_crop_res) * 0.5f;

    // The saved PEAR crop is an affine-normalized patch with a centered virtual
    // camera, not a literal window cut from the full-frame camera. Export the
    // crop metadata in that centered virtual-camera convention so the training
    // renderer uses principal point (target_crop_res / 2, target_crop_res / 2).
    metadata.crop_cx = full_cx;
    metadata.crop_cy = full_cy;
    metadata.crop_size = static_cast<float>(target_crop_res);
    metadata.crop_x0 = full_cx - crop_half;
    metadata.crop_y0 = full_cy - crop_half;
    metadata.crop_w = static_cast<float>(target_crop_res);
    metadata.crop_h = static_cast<float>(target_crop_res);
    metadata.focal_length = 0.5f * static_cast<float>(target_crop_res) * kPearCameraFocal;
    return metadata;
}

std::array<float, 3> BuildTrainingCameraParams(const cv::Vec3f& translation,
                                               const TrainingCropMetadata& crop,
                                               int img_width,
                                               int img_height) {
    const float tz = std::max(translation[2], 1e-6f);
    const float s = (2.0f * crop.focal_length) / (std::max(crop.crop_size, 1e-6f) * tz);
    const float full_cx = static_cast<float>(img_width) * 0.5f;
    const float full_cy = static_cast<float>(img_height) * 0.5f;
    const float u_screen = (crop.focal_length * translation[0] / tz) + full_cx;
    const float v_screen = (crop.focal_length * translation[1] / tz) + full_cy;
    const float denom = crop.crop_size * s * 0.5f + 1e-9f;

    return {
        s,
        (u_screen - crop.crop_cx) / denom,
        (v_screen - crop.crop_cy) / denom,
    };
}

cv::Mat ApplyCropMatte(const cv::Mat& crop_image, const cv::Mat& crop_matte) {
    if (crop_image.empty() || crop_matte.empty()) {
        return crop_image.clone();
    }

    cv::Mat matte_resized;
    if (crop_matte.size() != crop_image.size()) {
        cv::resize(crop_matte, matte_resized, crop_image.size(), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        matte_resized = crop_matte;
    }

    cv::Mat matte_f32;
    if (matte_resized.type() == CV_32F) {
        matte_f32 = matte_resized;
    } else {
        matte_resized.convertTo(matte_f32, CV_32F, 1.0 / 255.0);
    }

    cv::Mat matte_bgr;
    if (matte_f32.channels() == 1) {
        cv::cvtColor(matte_f32, matte_bgr, cv::COLOR_GRAY2BGR);
    } else {
        matte_bgr = matte_f32;
    }

    cv::Mat crop_f32;
    crop_image.convertTo(crop_f32, CV_32F, 1.0 / 255.0);

    cv::Mat matted_f32 = crop_f32.mul(matte_bgr);
    cv::Mat matted_crop;
    matted_f32.convertTo(matted_crop, crop_image.type(), 255.0);
    return matted_crop;
}

cv::Point2f ApplyAffinePoint(const cv::Matx23f& affine, const cv::Point2f& point) {
    return cv::Point2f(affine(0, 0) * point.x + affine(0, 1) * point.y + affine(0, 2),
                       affine(1, 0) * point.x + affine(1, 1) * point.y + affine(1, 2));
}

bool IsFinitePoint3(const cv::Point3f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
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

std::map<std::string, CameraCalibration> BuildCalibrationByCameraId(
    const std::vector<CameraCalibration>& calibrations) {
    std::map<std::string, CameraCalibration> by_id;
    for (const auto& calibration : calibrations) {
        by_id[calibration.camera_id] = calibration;
    }
    return by_id;
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

#if DATASET_PREP_HAS_SMPL
void BuildSmplOptimizationTargets(const std::vector<cv::Point3f>& triangulated_joints,
                                  const std::vector<float>& triangulated_scores,
                                  std::vector<int64_t>* out_smpl_indices,
                                  std::vector<float>* out_positions,
                                  std::vector<float>* out_weights) {
    if (out_smpl_indices == nullptr || out_positions == nullptr || out_weights == nullptr) {
        return;
    }

    out_smpl_indices->clear();
    out_positions->clear();
    out_weights->clear();

    const auto add_joint = [&](int mocap_index, int smpl_index) {
        if (mocap_index < 0 || mocap_index >= static_cast<int>(triangulated_joints.size())) {
            return;
        }
        if (mocap_index >= static_cast<int>(triangulated_scores.size())) {
            return;
        }
        if (triangulated_scores[mocap_index] < kTriangulationMinScore) {
            return;
        }
        const cv::Point3f& p = triangulated_joints[mocap_index];
        if (!IsFinitePoint3(p)) {
            return;
        }

        out_smpl_indices->push_back(static_cast<int64_t>(smpl_index));
        out_positions->push_back(p.x);
        out_positions->push_back(p.y);
        out_positions->push_back(p.z);
        out_weights->push_back(std::max(triangulated_scores[mocap_index], 1e-3f));
    };

    add_joint(19, 0);
    for (const auto& mapping : kMocapToSmplJointMap) {
        add_joint(mapping.mocap_index, mapping.smpl_index);
    }
}

bool ExtractPelvisWorld(const std::vector<cv::Point3f>& triangulated_joints,
                        const std::vector<float>& triangulated_scores,
                        cv::Point3f* out_pelvis) {
    if (out_pelvis == nullptr) {
        return false;
    }

    if (triangulated_joints.size() > 19u && triangulated_scores.size() > 19u &&
        triangulated_scores[19] >= kTriangulationMinScore && IsFinitePoint3(triangulated_joints[19])) {
        *out_pelvis = triangulated_joints[19];
        return true;
    }

    if (triangulated_joints.size() > 12u && triangulated_scores.size() > 12u &&
        triangulated_scores[11] >= kTriangulationMinScore &&
        triangulated_scores[12] >= kTriangulationMinScore &&
        IsFinitePoint3(triangulated_joints[11]) &&
        IsFinitePoint3(triangulated_joints[12])) {
        *out_pelvis = (triangulated_joints[11] + triangulated_joints[12]) * 0.5f;
        return true;
    }

    return false;
}

bool OptimizeSmplxPoseFromTriangulatedJoints(
    torch::jit::script::Module* smplx_layer,
    const std::vector<float>& fixed_betas,
    const std::vector<float>& fixed_expression,
    const std::vector<float>& jaw_pose,
    const std::vector<float>& eye_pose,
    const std::vector<float>& left_hand_pose,
    const std::vector<float>& right_hand_pose,
    const std::vector<float>& init_global_orient,
    const std::vector<float>& init_body_pose,
    const std::vector<cv::Point3f>& triangulated_joints,
    const std::vector<float>& triangulated_scores,
    std::vector<float>* out_global_orient,
    std::vector<float>* out_body_pose,
    cv::Vec3f* out_translation_world) {
    if (smplx_layer == nullptr || out_global_orient == nullptr || out_body_pose == nullptr ||
        out_translation_world == nullptr) {
        return false;
    }

    std::vector<int64_t> smpl_indices;
    std::vector<float> target_positions;
    std::vector<float> target_weights;
    BuildSmplOptimizationTargets(
        triangulated_joints,
        triangulated_scores,
        &smpl_indices,
        &target_positions,
        &target_weights);
    if (smpl_indices.empty()) {
        return false;
    }

    cv::Point3f pelvis_world;
    if (!ExtractPelvisWorld(triangulated_joints, triangulated_scores, &pelvis_world)) {
        return false;
    }

    const torch::Device device(torch::kCPU);
    torch::Tensor betas = MakeTorchInputTensor(fixed_betas, kSmplxShapeParamCount, device).detach();
    torch::Tensor expression = MakeTorchInputTensor(fixed_expression, kSmplxExpressionParamCount, device).detach();
    torch::Tensor jaw = MakeTorchInputTensor(jaw_pose, kSmplxJawPoseParamCount, device).detach();
    torch::Tensor eye = MakeTorchInputTensor(eye_pose, kSmplxEyePoseParamCount, device).detach();
    torch::Tensor left_hand = MakeTorchInputTensor(left_hand_pose, kSmplxHandPoseParamCount, device).detach();
    torch::Tensor right_hand = MakeTorchInputTensor(right_hand_pose, kSmplxHandPoseParamCount, device).detach();

    auto global_orient =
        MakeTorchInputTensor(init_global_orient, kSmplxGlobalOrientParamCount, device)
            .detach()
            .clone()
            .set_requires_grad(true);
    auto body_pose =
        MakeTorchInputTensor(init_body_pose, kSmplxBodyPoseParamCount, device)
            .detach()
            .clone()
            .set_requires_grad(true);

    torch::Tensor translation = torch::zeros({1, 3}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> init_inputs;
        init_inputs.emplace_back(betas);
        init_inputs.emplace_back(expression);
        init_inputs.emplace_back(global_orient.detach());
        init_inputs.emplace_back(body_pose.detach());
        init_inputs.emplace_back(jaw);
        init_inputs.emplace_back(eye);
        init_inputs.emplace_back(left_hand);
        init_inputs.emplace_back(right_hand);

        const auto out = smplx_layer->forward(init_inputs).toTuple();
        if (!out || out->elements().size() < 3u) {
            return false;
        }
        const auto joints = out->elements()[2].toTensor().squeeze(0).to(torch::kCPU).contiguous();
        if (joints.size(0) <= kSmplxRootJointIndex) {
            return false;
        }
        auto root = joints[kSmplxRootJointIndex];
        translation = torch::zeros({1, 3}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
        translation[0][0] = pelvis_world.x - root[0].item<float>();
        translation[0][1] = pelvis_world.y - root[1].item<float>();
        translation[0][2] = pelvis_world.z - root[2].item<float>();
    }
    translation = translation.detach().clone().set_requires_grad(true);

    auto idx_tensor = torch::from_blob(
                          smpl_indices.data(),
                          {static_cast<int64_t>(smpl_indices.size())},
                          torch::TensorOptions().dtype(torch::kInt64))
                          .clone()
                          .to(device);
    auto target_tensor = torch::from_blob(
                             target_positions.data(),
                             {static_cast<int64_t>(smpl_indices.size()), 3},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone()
                             .to(device);
    auto weight_tensor = torch::from_blob(
                             target_weights.data(),
                             {static_cast<int64_t>(target_weights.size())},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone()
                             .to(device);
    weight_tensor = weight_tensor / (weight_tensor.mean() + 1e-8f);

    auto global_orient_init = global_orient.detach().clone();
    auto body_pose_init = body_pose.detach().clone();

    torch::optim::Adam optimizer({global_orient, body_pose, translation},
                                 torch::optim::AdamOptions(5e-3));

    for (int iter = 0; iter < 60; ++iter) {
        optimizer.zero_grad();

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(betas);
        inputs.emplace_back(expression);
        inputs.emplace_back(global_orient);
        inputs.emplace_back(body_pose);
        inputs.emplace_back(jaw);
        inputs.emplace_back(eye);
        inputs.emplace_back(left_hand);
        inputs.emplace_back(right_hand);

        const auto out = smplx_layer->forward(inputs).toTuple();
        if (!out || out->elements().size() < 3u) {
            return false;
        }

        auto joints = out->elements()[2].toTensor().squeeze(0);
        joints = joints.index_select(0, idx_tensor) + translation.squeeze(0);

        auto diff = joints - target_tensor;
        auto data_loss = ((diff * diff).sum(1) * weight_tensor).mean();
        auto orient_reg = (global_orient - global_orient_init).pow(2).mean() * 1e-2;
        auto body_reg = (body_pose - body_pose_init).pow(2).mean() * 1e-2;
        auto total_loss = data_loss + orient_reg + body_reg;
        total_loss.backward();
        optimizer.step();
    }

    auto go_cpu = global_orient.detach().to(torch::kCPU).contiguous();
    auto bp_cpu = body_pose.detach().to(torch::kCPU).contiguous();
    auto tr_cpu = translation.detach().to(torch::kCPU).contiguous();

    out_global_orient->assign(go_cpu.data_ptr<float>(), go_cpu.data_ptr<float>() + go_cpu.numel());
    out_body_pose->assign(bp_cpu.data_ptr<float>(), bp_cpu.data_ptr<float>() + bp_cpu.numel());
    *out_translation_world = cv::Vec3f(
        tr_cpu[0][0].item<float>(),
        tr_cpu[0][1].item<float>(),
        tr_cpu[0][2].item<float>());
    return true;
}
#endif

}  // namespace

cv::Mat CropSquareWithPaddingAndResize(const cv::Mat& src, int x, int y, int size, int target_res) {
    if (src.empty() || size <= 0 || target_res <= 0) return {};
    const cv::Rect roi(x, y, size, size);
    const cv::Rect image_bounds(0, 0, src.cols, src.rows);
    const cv::Rect valid_roi = roi & image_bounds;

    cv::Mat square = cv::Mat::zeros(size, size, src.type());
    if (valid_roi.area() > 0) {
        const cv::Rect dst_roi(valid_roi.x - roi.x, valid_roi.y - roi.y, valid_roi.width, valid_roi.height);
        src(valid_roi).copyTo(square(dst_roi));
    }
    
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(target_res, target_res), 0.0, 0.0, 
               size > target_res ? cv::INTER_AREA : cv::INTER_CUBIC);
    return resized;
}

std::vector<float> ConvertPearPoseToSmplAxisAngle(const SmplxResult& pear_result) {
    std::vector<float> pose_values(72, 0.0f);
    if (pear_result.global_orient.size() == 3u && pear_result.body_pose.size() >= 63u) {
        std::copy_n(pear_result.global_orient.begin(), 3u, pose_values.begin());
        std::copy_n(pear_result.body_pose.begin(), 63u, pose_values.begin() + 3);
    }
    return pose_values;
}

std::string JsonEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8u);
    for (char c : input) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

void WriteFloatArray(std::ostream& out, const std::vector<float>& values, size_t expected_count) {
    out << "[";
    for (size_t index = 0; index < expected_count; ++index) {
        if (index > 0u) out << ", ";
        out << (index < values.size() ? values[index] : 0.0f);
    }
    out << "]";
}

void WritePoint3Array(std::ostream& out,
                      const std::vector<cv::Point3f>& points,
                      size_t expected_count) {
    out << "[";
    for (size_t i = 0; i < expected_count; ++i) {
        if (i > 0u) {
            out << ", ";
        }
        if (i < points.size() && IsFinitePoint3(points[i])) {
            out << points[i].x << ", " << points[i].y << ", " << points[i].z;
        } else {
            out << "0.0, 0.0, 0.0";
        }
    }
    out << "]";
}

struct DatasetPrepOptions {
    fs::path output_dir;
    std::string target_camera_id;
    std::vector<VideoSourceConfig> sources;
    int max_frames = -1;
    int frame_stride = 1;
    double sync_tolerance_ms = 8.0;
};

struct PreparedViewSample {
    const SyncedView* view = nullptr;
    float detection_score = 0.0f;
    const CameraCalibration* calibration = nullptr;
    TrainingCropMetadata training_crop;
    cv::Mat crop_image;
    cv::Mat crop_image_to_save;
    cv::Mat crop_matte;
    cv::Matx23f pear_crop_inv_trans = cv::Matx23f::eye();
    cv::Matx23f crop_inv_trans = cv::Matx23f::eye();
    std::vector<cv::Point2f> rtmpose_keypoints;
    std::vector<float> rtmpose_scores;
    SmplxResult pear_result;
};

struct TriangulationViewSample {
    const CameraCalibration* calibration = nullptr;
    std::vector<cv::Point2f> rtmpose_keypoints;
    std::vector<float> rtmpose_scores;
};

void PrintUsage(std::ostream& out) {
    out << "Usage:\n"
        << "  dataset_prep <video.mp4> <output_dir> <camera_id> [max_frames] [frame_stride]\n"
        << "  dataset_prep --output-dir <dir> --target-camera <camera_id>\n"
        << "               --feed <camera_id=video.mp4> [--feed <camera_id=video.mp4> ...]\n"
        << "               [--max-frames N] [--frame-stride N] [--sync-tolerance-ms MS]\n";
}

bool ParseFeedSpec(const std::string& feed_spec, VideoSourceConfig* out_source) {
    if (out_source == nullptr) {
        return false;
    }

    const size_t delimiter = feed_spec.find('=');
    if (delimiter == std::string::npos || delimiter == 0u || delimiter + 1u >= feed_spec.size()) {
        return false;
    }

    out_source->camera_id = feed_spec.substr(0, delimiter);
    out_source->video_path = feed_spec.substr(delimiter + 1u);
    return !out_source->camera_id.empty() && !out_source->video_path.empty();
}

bool ParseCommandLine(int argc, char* argv[], DatasetPrepOptions* out_options) {
    if (out_options == nullptr) {
        return false;
    }

    DatasetPrepOptions options;
    if (argc >= 4 && argc <= 6 && std::string(argv[1]).rfind("--", 0) != 0) {
        options.output_dir = argv[2];
        options.target_camera_id = argv[3];
        options.max_frames = (argc > 4) ? std::stoi(argv[4]) : -1;
        options.frame_stride = (argc > 5) ? std::stoi(argv[5]) : 1;

        VideoSourceConfig source;
        source.camera_id = options.target_camera_id;
        source.video_path = argv[1];
        options.sources.push_back(std::move(source));
    } else {
        for (int arg_index = 1; arg_index < argc; ++arg_index) {
            const std::string arg = argv[arg_index];
            if (arg == "--output-dir") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --output-dir.\n";
                    return false;
                }
                options.output_dir = argv[++arg_index];
            } else if (arg == "--target-camera") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --target-camera.\n";
                    return false;
                }
                options.target_camera_id = argv[++arg_index];
            } else if (arg == "--feed") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --feed.\n";
                    return false;
                }

                VideoSourceConfig source;
                if (!ParseFeedSpec(argv[++arg_index], &source)) {
                    std::cerr << "Invalid --feed value. Expected <camera_id=video_path>.\n";
                    return false;
                }
                options.sources.push_back(std::move(source));
            } else if (arg == "--max-frames") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --max-frames.\n";
                    return false;
                }
                options.max_frames = std::stoi(argv[++arg_index]);
            } else if (arg == "--frame-stride") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --frame-stride.\n";
                    return false;
                }
                options.frame_stride = std::stoi(argv[++arg_index]);
            } else if (arg == "--sync-tolerance-ms") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --sync-tolerance-ms.\n";
                    return false;
                }
                options.sync_tolerance_ms = std::stod(argv[++arg_index]);
            } else if (!arg.empty() && arg[0] != '-') {
                VideoSourceConfig source;
                if (!ParseFeedSpec(arg, &source)) {
                    std::cerr << "Unknown positional argument: " << arg << "\n";
                    return false;
                }
                options.sources.push_back(std::move(source));
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return false;
            }
        }
    }

    if (options.output_dir.empty()) {
        std::cerr << "output_dir is required.\n";
        return false;
    }
    if (options.frame_stride <= 0) {
        std::cerr << "frame_stride must be >= 1.\n";
        return false;
    }
    if (options.sources.empty()) {
        std::cerr << "At least one feed is required.\n";
        return false;
    }
    if (options.target_camera_id.empty()) {
        options.target_camera_id = options.sources.front().camera_id;
    }

    bool found_target_camera = false;
    for (size_t source_index = 0; source_index < options.sources.size(); ++source_index) {
        auto& source = options.sources[source_index];
        source.source_camera_index = static_cast<int>(source_index);
        source.frame_stride = options.frame_stride;
        if (source.camera_id == options.target_camera_id) {
            found_target_camera = true;
        }
    }

    if (!found_target_camera) {
        std::cerr << "target_camera_id must match one of the provided feeds.\n";
        return false;
    }

    *out_options = std::move(options);
    return true;
}

bool ResolveCalibrationDirectory(fs::path* out_dir) {
    if (out_dir == nullptr) {
        return false;
    }

    std::error_code ec;
    fs::path cursor = fs::current_path(ec);
    if (ec) {
        return false;
    }

    // Support running from repo root, build_dataset_prep, and nested binary dirs.
    for (int depth = 0; depth < 6; ++depth) {
        const fs::path candidate = cursor / "data";
        if (fs::exists(candidate / "extrinsics.json", ec) && !ec) {
            *out_dir = fs::absolute(candidate).lexically_normal();
            return true;
        }

        if (!cursor.has_parent_path()) {
            break;
        }
        const fs::path parent = cursor.parent_path();
        if (parent == cursor) {
            break;
        }
        cursor = parent;
    }

    return false;
}

void AccumulateVectorPrefix(const std::vector<float>& source, std::vector<float>* accumulator) {
    if (accumulator == nullptr) {
        return;
    }

    const size_t value_count = std::min(source.size(), accumulator->size());
    for (size_t index = 0; index < value_count; ++index) {
        (*accumulator)[index] += source[index];
    }
}

void NormalizeVectorInPlace(float denominator, std::vector<float>* values) {
    if (values == nullptr || denominator <= 0.0f) {
        return;
    }

    for (float& value : *values) {
        value /= denominator;
    }
}

void StabilizeInternalParameters(const std::vector<PreparedViewSample>& view_results,
                                 SmplxResult* target_result) {
    if (target_result == nullptr || view_results.empty()) {
        return;
    }

    std::vector<float> avg_betas(target_result->betas.size(), 0.0f);
    std::vector<float> avg_body_pose(target_result->body_pose.size(), 0.0f);
    std::vector<float> avg_expression(target_result->expression.size(), 0.0f);

    for (const auto& sample : view_results) {
        AccumulateVectorPrefix(sample.pear_result.betas, &avg_betas);
        AccumulateVectorPrefix(sample.pear_result.body_pose, &avg_body_pose);
        AccumulateVectorPrefix(sample.pear_result.expression, &avg_expression);
    }

    const float sample_count = static_cast<float>(view_results.size());
    NormalizeVectorInPlace(sample_count, &avg_betas);
    NormalizeVectorInPlace(sample_count, &avg_body_pose);
    NormalizeVectorInPlace(sample_count, &avg_expression);

    target_result->betas = std::move(avg_betas);
    target_result->body_pose = std::move(avg_body_pose);
    target_result->expression = std::move(avg_expression);
}

// --- MAIN PIPELINE ---

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        const std::string first_arg = argv[1];
        if (first_arg == "--help" || first_arg == "-h") {
            PrintUsage(std::cout);
            return 0;
        }
    }

    DatasetPrepOptions options;
    if (!ParseCommandLine(argc, argv, &options)) {
        PrintUsage(std::cerr);
        return 1;
    }

    const int target_crop_res = 1024;
    const float crop_margin = 1.25f;

    CameraCalibrationLoader calibration_loader;
    std::vector<CameraCalibration> calibrations;
    fs::path calibration_dir;
    if (!ResolveCalibrationDirectory(&calibration_dir)) {
        std::cerr << "Failed to locate calibration directory containing data/extrinsics.json from "
                  << fs::current_path().string() << "\n";
        return 1;
    }
    if (!calibration_loader.Load(calibration_dir, options.sources, &calibrations)) {
        std::cerr << "Failed to load camera calibrations from "
                  << calibration_dir.string() << "\n";
        return 1;
    }
    const auto calibration_by_camera_id = BuildCalibrationByCameraId(calibrations);

    fs::create_directories(options.output_dir / "crops" / options.target_camera_id);
    fs::create_directories(options.output_dir / "overlays" / options.target_camera_id);
    fs::create_directories(options.output_dir / "overlays_full" / options.target_camera_id);
    fs::create_directories(options.output_dir / "mattes" / options.target_camera_id);

    std::ofstream jsonl_out(options.output_dir / "frames.jsonl", std::ios::app);
    if (!jsonl_out.is_open()) {
        std::cerr << "Failed to open frames.jsonl for writing.\n";
        return 1;
    }
    jsonl_out << std::fixed << std::setprecision(6);

    std::ofstream rtmpose_keypoints_out(options.output_dir / "rtmpose_keypoints3d.jsonl", std::ios::app);
    if (!rtmpose_keypoints_out.is_open()) {
        std::cerr << "Failed to open rtmpose_keypoints3d.jsonl for writing.\n";
        return 1;
    }
    rtmpose_keypoints_out << std::fixed << std::setprecision(6);

    // Initialize Models
    YoloPersonDetectorOptions yolo_opts;
    yolo_opts.conf_threshold = 0.25f;
    yolo_opts.use_cuda = true;
    YoloPersonDetector yolo(yolo_opts);
    if (!yolo.Load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\yolov8n.onnx")) {
        std::cerr << "Failed to load YOLO model.\n";
        return 1;
    }

    PearEstimator::Options pear_opts;
    pear_opts.model_path = "C:\\Users\\Sam\\Documents\\GaussianAvatar\\pear_ehm.onnx";
    pear_opts.use_cuda = true;
    PearEstimator pear(pear_opts);
    if (!pear.Initialize()) {
        std::cerr << "Failed to load PEAR model.\n";
        return 1;
    }

    BackgroundExtractor::Options modnet_opts;
    modnet_opts.model_path = "C:\\Users\\Sam\\Documents\\GaussianAvatar\\modnet.onnx";
    modnet_opts.use_cuda = true;
    modnet_opts.input_size = target_crop_res;
    modnet_opts.binary_threshold = -1.0f;
    BackgroundExtractor modnet(modnet_opts);
    if (!modnet.Initialize()) {
        std::cerr << "Failed to load MODNet model.\n";
        return 1;
    }

    RtmPoseDetectorOptions rtmpose_opts;
    rtmpose_opts.use_cuda = true;
    rtmpose_opts.conf_threshold = 0.05f;
    RtmPoseDetector rtmpose(rtmpose_opts);
    if (!rtmpose.Load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\rtmpose26.onnx")) {
        std::cerr << "Failed to load RTMPose model.\n";
        return 1;
    }

#if DATASET_PREP_HAS_SMPL
    torch::jit::script::Module smplx_layer;
    try {
        smplx_layer = torch::jit::load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\smplx_libtorch.pt");
    } catch (const c10::Error& error) {
        std::cerr << "Failed to load SMPL-X TorchScript module.\n"
                  << error.what() << std::endl;
        return 1;
    }
    smplx_layer.to(torch::kCPU);
    smplx_layer.eval();
#endif

    VideoSynchronizer synchronizer({options.sync_tolerance_ms});
    if (!synchronizer.Open(options.sources)) {
        std::cerr << "Failed to open one or more input feeds.\n";
        return 1;
    }

    int selected_frame_count = 0;
    int exported_frame_count = 0;
    SyncedFrameCollection synced_frames;

    while (synchronizer.GetNextSyncedViews(&synced_frames)) {
        if (options.max_frames >= 0 && selected_frame_count >= options.max_frames) {
            break;
        }
        ++selected_frame_count;

        std::cout << "Processing synced frame " << synced_frames.sync_index << "..." << std::endl;

        int frame_yolo_failures = 0;
        int frame_calibration_misses = 0;
        int frame_rtmpose_failures = 0;
        int frame_pear_failures = 0;

        std::vector<PreparedViewSample> view_results;
        std::vector<TriangulationViewSample> triangulation_views;
        view_results.reserve(synced_frames.views.size());

        for (const auto& view : synced_frames.views) {
            cv::Rect2f best_bbox;
            float best_score = 0.0f;
            yolo.DetectPerson(view.image, &best_bbox, &best_score);
            if (best_score <= 0.0f || best_bbox.area() <= 0.0f) {
                ++frame_yolo_failures;
                continue;
            }

            auto calibration_it = calibration_by_camera_id.find(view.camera_id);
            if (calibration_it == calibration_by_camera_id.end() || !calibration_it->second.valid) {
                ++frame_calibration_misses;
                continue;
            }

            std::vector<cv::Point2f> rtmpose_keypoints;
            std::vector<float> rtmpose_scores;
            if (!rtmpose.DetectPose(view.image,
                                    &rtmpose_keypoints,
                                    &rtmpose_scores,
                                    synced_frames.sync_index)) {
                ++frame_rtmpose_failures;
            }

            if (!rtmpose_keypoints.empty() &&
                rtmpose_keypoints.size() == rtmpose_scores.size()) {
                TriangulationViewSample triangulation_sample;
                triangulation_sample.calibration = &calibration_it->second;
                triangulation_sample.rtmpose_keypoints = rtmpose_keypoints;
                triangulation_sample.rtmpose_scores = rtmpose_scores;
                triangulation_views.push_back(std::move(triangulation_sample));
            }

            cv::Rect2f crop_bbox;
            if (!ProcessBBox(best_bbox,
                             view.image.cols,
                             view.image.rows,
                             cv::Size(kPearInputRes, kPearInputRes),
                             crop_margin,
                             &crop_bbox)) {
                continue;
            }

            PreparedViewSample sample;
            sample.view = &view;
            sample.detection_score = best_score;
            sample.calibration = &calibration_it->second;
            sample.rtmpose_keypoints = std::move(rtmpose_keypoints);
            sample.rtmpose_scores = std::move(rtmpose_scores);
            sample.training_crop = BuildTrainingCropMetadata(
                crop_bbox, view.image.cols, view.image.rows, target_crop_res);

            cv::Mat pear_crop_image;
            if (!GeneratePatchImage(view.image,
                                    crop_bbox,
                                    1.0f,
                                    0.0f,
                                    false,
                                    cv::Size(kPearInputRes, kPearInputRes),
                                    &pear_crop_image,
                                    nullptr,
                                    &sample.pear_crop_inv_trans)) {
                continue;
            }

            if (!GeneratePatchImage(view.image,
                                    crop_bbox,
                                    1.0f,
                                    0.0f,
                                    false,
                                    cv::Size(target_crop_res, target_crop_res),
                                    &sample.crop_image,
                                    nullptr,
                                    &sample.crop_inv_trans)) {
                continue;
            }

            if (!modnet.ProcessImage(sample.crop_image, &sample.crop_matte)) {
                std::cerr << "Warning: Failed to extract matte on synced frame "
                          << synced_frames.sync_index << " for camera "
                          << view.camera_id << "\n";
            }
            sample.crop_image_to_save = ApplyCropMatte(sample.crop_image, sample.crop_matte);
            cv::Mat pear_input_image = ApplyCropMatte(pear_crop_image, sample.crop_matte);

            if (!pear.Estimate(pear_input_image, &sample.pear_result)) {
                ++frame_pear_failures;
                continue;
            }

            view_results.push_back(std::move(sample));
        }

        auto target_view_it = std::find_if(
            view_results.begin(),
            view_results.end(),
            [&](const PreparedViewSample& sample) {
                return sample.view != nullptr &&
                       sample.view->camera_id == options.target_camera_id;
            });
        if (target_view_it == view_results.end()) {
            std::cout << "Skipping synced frame " << synced_frames.sync_index
                      << ": no valid target view."
                      << " yolo_failures=" << frame_yolo_failures
                      << " calibration_misses=" << frame_calibration_misses
                      << " rtmpose_failures=" << frame_rtmpose_failures
                      << " pear_failures=" << frame_pear_failures
                      << " successful_views=" << view_results.size()
                      << std::endl;
            continue;
        }

        auto best_pose_it = std::max_element(
            view_results.begin(),
            view_results.end(),
            [](const PreparedViewSample& lhs, const PreparedViewSample& rhs) {
                return lhs.detection_score < rhs.detection_score;
            });
        if (best_pose_it == view_results.end()) {
            std::cout << "Skipping synced frame " << synced_frames.sync_index
                      << ": failed to pick best pose view." << std::endl;
            continue;
        }

        // Use RTMPose full-frame detections for geometric triangulation.
        std::vector<cv::Point3f> triangulated_joints(
            kMocapJointCount,
            cv::Point3f(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()));
        std::vector<float> triangulated_scores(kMocapJointCount, 0.0f);
        for (size_t joint_index = 0; joint_index < kMocapJointCount; ++joint_index) {
            std::vector<cv::Matx34f> projections;
            std::vector<cv::Point2f> observations;
            std::vector<float> scores;

            for (const auto& sample : triangulation_views) {
                if (sample.calibration == nullptr ||
                    sample.rtmpose_keypoints.size() <= joint_index ||
                    sample.rtmpose_scores.size() <= joint_index) {
                    continue;
                }
                const float score = sample.rtmpose_scores[joint_index];
                if (score < kTriangulationMinScore) {
                    continue;
                }

                projections.push_back(BuildProjectionMatrix(*sample.calibration));
                observations.push_back(sample.rtmpose_keypoints[joint_index]);
                scores.push_back(score);
            }

            cv::Point3f point_world;
            if (TriangulatePointDLT(projections, observations, &point_world)) {
                triangulated_joints[joint_index] = point_world;
                float avg_score = 0.0f;
                for (float score : scores) {
                    avg_score += score;
                }
                triangulated_scores[joint_index] =
                    scores.empty() ? 0.0f : (avg_score / static_cast<float>(scores.size()));
            }
        }

        // Keep PEAR shape/expression priors stable across views and use the best
        // detection as the initial pose seed.
        SmplxResult fused_result = best_pose_it->pear_result;
        StabilizeInternalParameters(view_results, &fused_result);
        cv::Matx33f camera_rotation = ExtractPearCameraRotation(fused_result);
        cv::Vec3f camera_translation = ExtractPearCameraTranslation(fused_result);

        std::vector<float> optimized_global_orient =
            PadFloatVector(fused_result.global_orient, static_cast<size_t>(kSmplxGlobalOrientParamCount));
        std::vector<float> optimized_body_pose =
            PadFloatVector(fused_result.body_pose, static_cast<size_t>(kSmplxBodyPoseParamCount));
        cv::Vec3f optimized_translation_world(0.0f, 0.0f, 0.0f);
        bool has_optimized_pose = false;

        const std::vector<float> smplx_global_orient =
            PadFloatVector(fused_result.global_orient, static_cast<size_t>(kSmplxGlobalOrientParamCount));
        const std::vector<float> smplx_body_pose =
            PadFloatVector(fused_result.body_pose, static_cast<size_t>(kSmplxBodyPoseParamCount));
        const std::vector<float> smplx_jaw_pose =
            PadFloatVector(fused_result.jaw_pose, static_cast<size_t>(kSmplxJawPoseParamCount));
        const std::vector<float> smplx_eye_pose(
            static_cast<size_t>(kSmplxEyePoseParamCount), 0.0f);
        const std::vector<float> smplx_left_hand_pose =
            PadFloatVector(fused_result.left_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
        const std::vector<float> smplx_right_hand_pose =
            PadFloatVector(fused_result.right_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
        const std::vector<float> smplx_expression =
            PadFloatVector(fused_result.expression, static_cast<size_t>(kSmplxExpressionParamCount));

        std::vector<float> train_pose_local = ConvertPearPoseToSmplAxisAngle(fused_result);
        std::vector<float> export_global_orient = smplx_global_orient;
        std::vector<float> export_body_pose = smplx_body_pose;
        cv::Vec3f training_camera_translation = camera_translation;

        cv::Mat crop_overlay = target_view_it->crop_image.clone();
        cv::Mat full_overlay = target_view_it->view->image.clone();
        cv::Mat crop_mesh_overlay = cv::Mat::zeros(crop_overlay.size(), crop_overlay.type());

#if DATASET_PREP_HAS_SMPL
        has_optimized_pose = OptimizeSmplxPoseFromTriangulatedJoints(
            &smplx_layer,
            fused_result.betas,
            smplx_expression,
            smplx_jaw_pose,
            smplx_eye_pose,
            smplx_left_hand_pose,
            smplx_right_hand_pose,
            smplx_global_orient,
            smplx_body_pose,
            triangulated_joints,
            triangulated_scores,
            &optimized_global_orient,
            &optimized_body_pose,
            &optimized_translation_world);

        if (has_optimized_pose) {
            for (size_t i = 0; i < 3u; ++i) {
                train_pose_local[i] = optimized_global_orient[i];
            }
            for (size_t i = 0; i < 63u; ++i) {
                train_pose_local[i + 3u] = optimized_body_pose[i];
            }
            export_global_orient = optimized_global_orient;
            export_body_pose = optimized_body_pose;

            if (target_view_it->calibration != nullptr) {
                camera_rotation = target_view_it->calibration->R;
                const cv::Vec3f translation_world = optimized_translation_world;
                camera_translation = target_view_it->calibration->R * translation_world +
                                     target_view_it->calibration->t;
                training_camera_translation = camera_translation;
            }
        } else {
            const cv::Matx33f train_rotation = ProjectionSignFlip() * camera_rotation;
            ApplyExtraRootRotation(train_rotation, &train_pose_local);
            training_camera_translation = ProjectionSignFlip() * camera_translation;
        }
#else
        const cv::Matx33f train_rotation = ProjectionSignFlip() * camera_rotation;
        ApplyExtraRootRotation(train_rotation, &train_pose_local);
        training_camera_translation = ProjectionSignFlip() * camera_translation;
#endif

        const std::array<float, 3> train_camera = BuildTrainingCameraParams(
            training_camera_translation,
            target_view_it->training_crop,
            target_view_it->view->image.cols,
            target_view_it->view->image.rows);

#if DATASET_PREP_HAS_SMPL
        torch::NoGradGuard no_grad;

        // The exported SMPL-X TorchScript wrapper was traced on CPU tensors.
        // Keep dataset_prep inference on CPU here to avoid device-mismatch issues.
        const torch::Device smplx_device(torch::kCPU);

        if (!fused_result.betas.empty()) {
            std::cout << "PEAR betas: [";
            for (size_t beta_index = 0; beta_index < fused_result.betas.size(); ++beta_index) {
                if (beta_index > 0u) {
                    std::cout << ", ";
                }
                std::cout << fused_result.betas[beta_index];
            }
            std::cout << "]\n";
        }

        std::vector<torch::jit::IValue> smplx_inputs;
        smplx_inputs.emplace_back(MakeTorchInputTensor(fused_result.betas,
                                                       kSmplxShapeParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_expression,
                                                       kSmplxExpressionParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(has_optimized_pose ? optimized_global_orient : smplx_global_orient,
                                                       kSmplxGlobalOrientParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(has_optimized_pose ? optimized_body_pose : smplx_body_pose,
                                                       kSmplxBodyPoseParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_jaw_pose,
                                                       kSmplxJawPoseParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_eye_pose,
                                                       kSmplxEyePoseParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_left_hand_pose,
                                                       kSmplxHandPoseParamCount,
                                                       smplx_device));
        smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_right_hand_pose,
                                                       kSmplxHandPoseParamCount,
                                                       smplx_device));

        const auto smplx_out = smplx_layer.forward(smplx_inputs).toTuple();
        if (!smplx_out || smplx_out->elements().size() < 3u) {
            std::cerr << "Warning: SMPL-X forward returned an unexpected output tuple on synced frame "
                      << synced_frames.sync_index << "\n";
            continue;
        }

        auto verts_tensor = smplx_out->elements()[0].toTensor().squeeze(0).to(torch::kCPU).contiguous();
        auto joints_tensor = smplx_out->elements()[2].toTensor().squeeze(0).to(torch::kCPU).contiguous();
        auto joints_acc = joints_tensor.accessor<float, 2>();
        if (joints_acc.size(0) > 0) {
            std::cout << "SMPL-X pelvis root joint 3D before verts_acc: ("
                      << joints_acc[0][0] << ", "
                      << joints_acc[0][1] << ", "
                      << joints_acc[0][2] << ")\n";
        }
        auto verts_acc = verts_tensor.accessor<float, 2>();

        for (int i = 0; i < verts_acc.size(0); ++i) {
            const cv::Vec3f point_model(verts_acc[i][0], verts_acc[i][1], verts_acc[i][2]);
            const cv::Vec3f point_cam = camera_rotation * point_model + camera_translation;
            if (point_cam[2] <= 1e-6f) {
                continue;
            }

            if (has_optimized_pose && target_view_it->calibration != nullptr) {
                const cv::Vec3f uvw = target_view_it->calibration->K * point_cam;
                if (std::abs(uvw[2]) <= 1e-8f) {
                    continue;
                }

                const cv::Point2f full_frame_pt(uvw[0] / uvw[2], uvw[1] / uvw[2]);
                if (full_frame_pt.x >= 0.0f && full_frame_pt.y >= 0.0f &&
                    full_frame_pt.x < static_cast<float>(full_overlay.cols) &&
                    full_frame_pt.y < static_cast<float>(full_overlay.rows)) {
                    cv::circle(full_overlay,
                               cv::Point(static_cast<int>(std::lround(full_frame_pt.x)),
                                         static_cast<int>(std::lround(full_frame_pt.y))),
                               1,
                               cv::Scalar(0, 255, 0),
                               -1,
                               cv::LINE_AA);
                }
            } else {
                const cv::Point2f projected = ProjectPearCameraPoint(
                    point_cam, crop_overlay.cols, crop_overlay.rows);
                if (projected.x >= 0.0f && projected.y >= 0.0f &&
                    projected.x < static_cast<float>(crop_overlay.cols) &&
                    projected.y < static_cast<float>(crop_overlay.rows)) {
                    const cv::Point crop_pt(static_cast<int>(std::lround(projected.x)),
                                            static_cast<int>(std::lround(projected.y)));
                    cv::circle(crop_overlay, crop_pt, 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                    cv::circle(crop_mesh_overlay, crop_pt, 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                }
            }
        }

        // cv::Mat warped_overlay = cv::Mat::zeros(full_overlay.size(), full_overlay.type());
        // cv::warpAffine(crop_mesh_overlay,
        //                warped_overlay,
        //                cv::Mat(target_view_it->crop_inv_trans),
        //                full_overlay.size(),
        //                cv::INTER_LINEAR,
        //                cv::BORDER_CONSTANT,
        //                cv::Scalar(0, 0, 0));
        // cv::Mat mask;
        // cv::cvtColor(warped_overlay, mask, cv::COLOR_BGR2GRAY);
        // warped_overlay.copyTo(full_overlay, mask);
#endif

        std::ostringstream stem;
        stem << "frame_" << std::setw(6) << std::setfill('0') << synced_frames.sync_index;

        const std::string crop_path = (options.output_dir / "crops" / options.target_camera_id /
                                       ("crop_" + stem.str() + ".png")).string();
        const std::string overlay_path = (options.output_dir / "overlays" / options.target_camera_id /
                                          ("overlay_" + stem.str() + ".png")).string();
        const std::string full_path = (options.output_dir / "overlays_full" / options.target_camera_id /
                                       (stem.str() + ".jpg")).string();
        const std::string matte_path = (options.output_dir / "mattes" / options.target_camera_id /
                                        ("matte_" + stem.str() + ".png")).string();

        cv::imwrite(crop_path, target_view_it->crop_image_to_save);
        cv::imwrite(overlay_path, crop_overlay);
        cv::imwrite(full_path, full_overlay);
        if (!target_view_it->crop_matte.empty()) {
            cv::imwrite(matte_path, target_view_it->crop_matte);
        }

        jsonl_out << "{\"frame\":" << synced_frames.sync_index
                  << ",\"sync_index\":" << synced_frames.sync_index
                  << ",\"camera_id\":\"" << JsonEscape(options.target_camera_id) << "\""
                  << ",\"person_id\":0"
                  << ",\"video_frame_index\":" << target_view_it->view->video_frame_index
                  << ",\"crop\":\"" << JsonEscape(crop_path) << "\""
                  << ",\"mask\":\"" << JsonEscape(matte_path) << "\""
                  << ",\"overlay\":\"" << JsonEscape(overlay_path) << "\""
                  << ",\"img_w\":" << target_view_it->view->image.cols
                  << ",\"img_h\":" << target_view_it->view->image.rows
                  << ",\"crop_cx\":" << target_view_it->training_crop.crop_cx
                  << ",\"crop_cy\":" << target_view_it->training_crop.crop_cy
                  << ",\"crop_size\":" << target_view_it->training_crop.crop_size
                  << ",\"crop_x0\":" << target_view_it->training_crop.crop_x0
                  << ",\"crop_y0\":" << target_view_it->training_crop.crop_y0
                  << ",\"crop_w\":" << target_view_it->training_crop.crop_w
                  << ",\"crop_h\":" << target_view_it->training_crop.crop_h
                  << ",\"focal_length\":" << target_view_it->training_crop.focal_length
                  << ",\"y_sign\": 1.0"
                  << ",\"pose\":";
        WriteFloatArray(jsonl_out, train_pose_local, 72u);
        jsonl_out << ",\"betas\":";
        WriteFloatArray(jsonl_out, fused_result.betas, 10u);
        jsonl_out << ",\"cam\":["
                  << train_camera[0] << ", "
                  << train_camera[1] << ", "
                  << train_camera[2] << "]";
        jsonl_out << ",\"body_model\":\"smplx\"";
        jsonl_out << ",\"smplx_shape\":";
        WriteFloatArray(jsonl_out, fused_result.betas, static_cast<size_t>(kSmplxShapeParamCount));
        jsonl_out << ",\"smplx_expression\":";
        WriteFloatArray(jsonl_out, smplx_expression, static_cast<size_t>(kSmplxExpressionParamCount));
        jsonl_out << ",\"smplx_global_orient\":";
        WriteFloatArray(jsonl_out, export_global_orient, static_cast<size_t>(kSmplxGlobalOrientParamCount));
        jsonl_out << ",\"smplx_body_pose\":";
        WriteFloatArray(jsonl_out, export_body_pose, static_cast<size_t>(kSmplxBodyPoseParamCount));
        jsonl_out << ",\"smplx_jaw_pose\":";
        WriteFloatArray(jsonl_out, smplx_jaw_pose, static_cast<size_t>(kSmplxJawPoseParamCount));
        jsonl_out << ",\"smplx_eye_pose\":";
        WriteFloatArray(jsonl_out, smplx_eye_pose, static_cast<size_t>(kSmplxEyePoseParamCount));
        jsonl_out << ",\"smplx_left_hand_pose\":";
        WriteFloatArray(jsonl_out, smplx_left_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
        jsonl_out << ",\"smplx_right_hand_pose\":";
        WriteFloatArray(jsonl_out, smplx_right_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
        if (!fused_result.camera_rt.empty()) {
            jsonl_out << ",\"camera_rt\":";
            WriteFloatArray(jsonl_out, fused_result.camera_rt, 16u);
        }
        jsonl_out << "}\n";

        rtmpose_keypoints_out << "{\"frame\":" << synced_frames.sync_index
                              << ",\"sync_index\":" << synced_frames.sync_index
                              << ",\"camera_id\":\"" << JsonEscape(options.target_camera_id) << "\""
                              << ",\"joint_count\":" << kMocapJointCount
                              << ",\"rtmpose_keypoints_3d\":";
        WritePoint3Array(rtmpose_keypoints_out, triangulated_joints, kMocapJointCount);
        rtmpose_keypoints_out << ",\"rtmpose_scores\":";
        WriteFloatArray(rtmpose_keypoints_out, triangulated_scores, kMocapJointCount);
        rtmpose_keypoints_out << "}\n";

        ++exported_frame_count;
    }

    std::cout << "Finished exporting " << exported_frame_count
              << " frames from " << selected_frame_count
              << " selected synchronized frames (stride " << options.frame_stride
              << ", feeds " << options.sources.size() << ")." << std::endl;
    return 0;
}
