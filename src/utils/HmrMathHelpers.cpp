#include "HmrMathHelpers.h"

#include <algorithm>
#include <stdexcept>

#include "HmrInferenceConstants.h"
#include "HmrInferenceUtils.h"

cv::Mat Rot6dToRotMatSingle(const float* feat) {
    const float eps = 1e-8f;

    // Read strided values to match Python's 'reshape(-1, 3, 2)' behavior.
    cv::Vec3f a1(feat[0], feat[2], feat[4]);
    cv::Vec3f a2(feat[1], feat[3], feat[5]);

    float n1 = std::max(static_cast<float>(cv::norm(a1)), eps);
    cv::Vec3f b1 = a1 / n1;

    float dot = b1.dot(a2);
    cv::Vec3f b2 = a2 - dot * b1;
    float n2 = std::max(static_cast<float>(cv::norm(b2)), eps);
    b2 = b2 / n2;

    cv::Vec3f b3 = b1.cross(b2);

    cv::Mat rot_mat(3, 3, CV_32F);
    rot_mat.at<float>(0, 0) = b1[0]; rot_mat.at<float>(0, 1) = b2[0]; rot_mat.at<float>(0, 2) = b3[0];
    rot_mat.at<float>(1, 0) = b1[1]; rot_mat.at<float>(1, 1) = b2[1]; rot_mat.at<float>(1, 2) = b3[1];
    rot_mat.at<float>(2, 0) = b1[2]; rot_mat.at<float>(2, 1) = b2[2]; rot_mat.at<float>(2, 2) = b3[2];
    return rot_mat;
}

std::vector<float> ConvertPose6dToAxisAngle(const std::vector<float>& pose_6d) {
    if (pose_6d.size() != 144) {
        throw std::runtime_error("Expected 144 pose values (6D), got " + std::to_string(pose_6d.size()));
    }
    std::vector<float> pose_aa;
    pose_aa.reserve(72);
    for (int i = 0; i < 24; ++i) {
        const float* current_6d = &pose_6d[i * 6];
        cv::Mat rmat = Rot6dToRotMatSingle(current_6d);
        cv::Mat rvec;
        cv::Rodrigues(rmat, rvec);
        pose_aa.push_back(rvec.at<float>(0));
        pose_aa.push_back(rvec.at<float>(1));
        pose_aa.push_back(rvec.at<float>(2));
    }
    return pose_aa;
}

torch::Tensor PoseToAxisAngle(const SmplResult& res) {
    const int pose_size = static_cast<int>(res.pose.size());
    if (pose_size == 72) {
        auto pose = torch::from_blob(const_cast<float*>(res.pose.data()), {1, 24, 3}, torch::kFloat).clone();
        return pose;
    }
    if (pose_size == 144) {
        const auto pose_aa = ConvertPose6dToAxisAngle(res.pose);
        auto pose = torch::from_blob(const_cast<float*>(pose_aa.data()), {1, 24, 3}, torch::kFloat).clone();
        return pose;
    }
    if (pose_size % 3 == 0 && (pose_size / 3) == 24) {
        auto pose = torch::from_blob(const_cast<float*>(res.pose.data()), {1, 24, 3}, torch::kFloat).clone();
        return pose;
    }
    throw std::runtime_error("Unsupported pose size: " + std::to_string(pose_size));
}

int CountProjectedInFrame(const torch::Tensor& verts_cpu, float s, float tx, float ty, float y_sign,
                          int width, int height) {
    const float img_size = static_cast<float>(kInputW);
    const float scale_x = static_cast<float>(width) / img_size;
    const float scale_y = static_cast<float>(height) / img_size;

    auto verts_acc = verts_cpu.accessor<float, 2>();
    int count = 0;
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0];
        const float Y = verts_acc[i][1] * y_sign;
        const float u = (s * (X + tx) + img_size * 0.5f) * scale_x;
        const float v = (s * (Y + ty) + img_size * 0.5f) * scale_y;
        if (u >= 0.0f && v >= 0.0f && u < width && v < height) {
            count++;
        }
    }
    return count;
}

int CountProjectedInFramePinhole(const torch::Tensor& verts_cpu, const cv::Vec3f& t, float y_sign,
                                 float focal_length, float cx, float cy,
                                 int width, int height) {
    auto verts_acc = verts_cpu.accessor<float, 2>();
    int count = 0;
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0] + t[0];
        const float Y = verts_acc[i][1] * y_sign + t[1];
        const float Z = verts_acc[i][2] + t[2];
        const float u = (focal_length * X / (Z + 1e-9f)) + cx;
        const float v = (focal_length * Y / (Z + 1e-9f)) + cy;
        if (u >= 0.0f && v >= 0.0f && u < width && v < height) {
            count++;
        }
    }
    return count;
}

cv::Vec3f EstimateTranslation(const std::vector<float>& cam, float crop_cx, float crop_cy, float crop_size,
                              float focal_length, float img_w, float img_h) {
    const float s = cam[0];
    const float tx = cam[1];
    const float ty = cam[2];

    const float tz = 2.0f * focal_length / (crop_size * s + 1e-9f);
    const float tx_screen = (crop_cx - img_w * 0.5f) + (tx * crop_size * s * 0.5f);
    const float ty_screen = (crop_cy - img_h * 0.5f) + (ty * crop_size * s * 0.5f);

    const float tx_3d = tx_screen * tz / focal_length;
    const float ty_3d = ty_screen * tz / focal_length;

    return cv::Vec3f(tx_3d, ty_3d, tz);
}
