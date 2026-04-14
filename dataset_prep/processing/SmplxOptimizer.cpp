#include "dataset_prep/processing/SmplxOptimizer.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <opencv2/calib3d.hpp>

#if __has_include(<opencv2/viz.hpp>)
#include <opencv2/viz.hpp>
#define DATASET_PREP_HAS_OPENCV_VIZ 1
#else
#define DATASET_PREP_HAS_OPENCV_VIZ 0
#endif

#include "dataset_prep/DatasetPrepTypes.h"

#if DATASET_PREP_HAS_SMPL
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace dataset_prep {
namespace {

constexpr int kSmplxRootJointIndex = 0;
constexpr float kTriangulationMinScore = 0.000025f;
constexpr int kVizRenderStride = 2;
constexpr int kVizRenderDelayMs = 80;

struct JointMapEntry {
    int mocap_index;
    int smpl_index;
};

// In dataset_prep/processing/SmplxOptimizer.cpp (around line 35)
const std::array<JointMapEntry, 20> kMocapToSmplJointMap = {{
    {12, 1}, {11, 2}, {14, 4}, {13, 5}, {16, 7}, {15, 8}, {18, 12}, // Swapped 11/12, 13/14, 15/16
    {17, 15}, 
    {6, 16}, {5, 17}, {8, 18}, {7, 19}, {10, 20}, {9, 21},          // Swapped 5/6, 7/8, 9/10
    {21, 10}, {23, 10}, {25, 10},                                  // R-Toes/Heel -> SMPL L-Foot
    {20, 11}, {22, 11}, {24, 11},                                  // L-Toes/Heel -> SMPL R-Foot
}};

bool IsFinitePoint3(const cv::Point3f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool HasReliableJoint(const std::vector<cv::Point3f>& triangulated_joints,
                      const std::vector<float>& triangulated_scores,
                      int joint_index,
                      cv::Vec3f* out_point) {
    if (out_point == nullptr) {
        return false;
    }
    if (joint_index < 0 || joint_index >= static_cast<int>(triangulated_joints.size()) ||
        joint_index >= static_cast<int>(triangulated_scores.size())) {
        return false;
    }
    if (triangulated_scores[joint_index] < kTriangulationMinScore ||
        !IsFinitePoint3(triangulated_joints[joint_index])) {
        return false;
    }

    const auto& p = triangulated_joints[joint_index];
    *out_point = cv::Vec3f(p.x, p.y, p.z);
    return true;
}

bool EstimateRootOrientationFromHips(const std::vector<cv::Point3f>& triangulated_joints,
                                     const std::vector<float>& triangulated_scores,
                                     cv::Vec3f* out_axis_angle) {
    if (out_axis_angle == nullptr) {
        return false;
    }

    cv::Vec3f left_hip;
    cv::Vec3f right_hip;
    if (!HasReliableJoint(triangulated_joints, triangulated_scores, 11, &left_hip) ||
        !HasReliableJoint(triangulated_joints, triangulated_scores, 12, &right_hip)) {
        return false;
    }

    cv::Vec3f right_axis = right_hip - left_hip;
    const float right_norm = cv::norm(right_axis);
    if (right_norm < 1e-5f) {
        return false;
    }
    right_axis *= (1.0f / right_norm);

    cv::Vec3f pelvis = 0.5f * (left_hip + right_hip);
    cv::Vec3f left_shoulder;
    cv::Vec3f right_shoulder;
    if (!HasReliableJoint(triangulated_joints, triangulated_scores, 5, &left_shoulder) ||
        !HasReliableJoint(triangulated_joints, triangulated_scores, 6, &right_shoulder)) {
        return false;
    }

    cv::Vec3f up_axis = 0.5f * (left_shoulder + right_shoulder) - pelvis;
    up_axis -= right_axis * (right_axis.dot(up_axis));
    const float up_norm = cv::norm(up_axis);
    if (up_norm < 1e-5f) {
        return false;
    }
    up_axis *= (1.0f / up_norm);

    cv::Vec3f forward_axis = right_axis.cross(up_axis);
    const float forward_norm = cv::norm(forward_axis);
    if (forward_norm < 1e-5f) {
        return false;
    }
    forward_axis *= (1.0f / forward_norm);

    up_axis = forward_axis.cross(right_axis);
    const float up_renorm = cv::norm(up_axis);
    if (up_renorm < 1e-5f) {
        return false;
    }
    up_axis *= (1.0f / up_renorm);

    cv::Matx33d rotation(
        static_cast<double>(right_axis[0]), static_cast<double>(up_axis[0]), static_cast<double>(forward_axis[0]),
        static_cast<double>(right_axis[1]), static_cast<double>(up_axis[1]), static_cast<double>(forward_axis[1]),
        static_cast<double>(right_axis[2]), static_cast<double>(up_axis[2]), static_cast<double>(forward_axis[2]));

    cv::Mat axis_angle;
    cv::Rodrigues(cv::Mat(rotation), axis_angle);
    if (axis_angle.rows != 3 || axis_angle.cols != 1) {
        return false;
    }

    *out_axis_angle = cv::Vec3f(
        static_cast<float>(axis_angle.at<double>(0, 0)),
        static_cast<float>(axis_angle.at<double>(1, 0)),
        static_cast<float>(axis_angle.at<double>(2, 0)));
    return true;
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
#endif

}  // namespace

std::vector<float> PadFloatVector(const std::vector<float>& values, size_t expected_count) {
    std::vector<float> padded(expected_count, 0.0f);
    const size_t copy_count = std::min(values.size(), expected_count);
    std::copy_n(values.begin(), copy_count, padded.begin());
    return padded;
}

bool OptimizeSmplxPoseFromTriangulatedJoints(SmplxTorchModule* smplx_layer,
                                             const SmplxOptimizationInputs& inputs,
                                             SmplxOptimizationResult* out_result) {
#if !DATASET_PREP_HAS_SMPL
    (void)smplx_layer;
    (void)inputs;
    (void)out_result;
    return false;
#else
    if (smplx_layer == nullptr || out_result == nullptr) {
        return false;
    }

    std::vector<int64_t> smpl_indices;
    std::vector<float> target_positions;
    std::vector<float> target_weights;
    BuildSmplOptimizationTargets(
        inputs.triangulated_joints,
        inputs.triangulated_scores,
        &smpl_indices,
        &target_positions,
        &target_weights);
    if (smpl_indices.empty()) {
        return false;
    }

    cv::Point3f pelvis_world;
    if (!ExtractPelvisWorld(inputs.triangulated_joints, inputs.triangulated_scores, &pelvis_world)) {
        return false;
    }

    const torch::Device device(torch::kCPU);
    torch::Tensor betas = MakeTorchInputTensor(inputs.fixed_betas, kSmplxShapeParamCount, device).detach();
    torch::Tensor expression =
        MakeTorchInputTensor(inputs.fixed_expression, kSmplxExpressionParamCount, device).detach();
    torch::Tensor jaw = MakeTorchInputTensor(inputs.jaw_pose, kSmplxJawPoseParamCount, device).detach();
    torch::Tensor eye = MakeTorchInputTensor(inputs.eye_pose, kSmplxEyePoseParamCount, device).detach();
    torch::Tensor left_hand = MakeTorchInputTensor(inputs.left_hand_pose, kSmplxHandPoseParamCount, device).detach();
    torch::Tensor right_hand = MakeTorchInputTensor(inputs.right_hand_pose, kSmplxHandPoseParamCount, device).detach();

    std::vector<float> init_global_orient = inputs.init_global_orient;
    cv::Vec3f hips_axis_angle;
    if (EstimateRootOrientationFromHips(
            inputs.triangulated_joints,
            inputs.triangulated_scores,
            &hips_axis_angle)) {
        if (init_global_orient.size() < static_cast<size_t>(kSmplxGlobalOrientParamCount)) {
            init_global_orient.resize(static_cast<size_t>(kSmplxGlobalOrientParamCount), 0.0f);
        }
        init_global_orient[0] = hips_axis_angle[0];
        init_global_orient[1] = hips_axis_angle[1];
        init_global_orient[2] = hips_axis_angle[2];
    }

    auto global_orient =
        MakeTorchInputTensor(init_global_orient, kSmplxGlobalOrientParamCount, device)
            .detach()
            .clone()
            .set_requires_grad(true);
    auto body_pose =
        MakeTorchInputTensor(inputs.init_body_pose, kSmplxBodyPoseParamCount, device)
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

// #if DATASET_PREP_HAS_OPENCV_VIZ
//     cv::viz::Viz3d viewer("SMPL-X 3D Optimization View");
//     viewer.showWidget("Coordinate System", cv::viz::WCoordinateSystem(0.5));

//     // Change this: Use inputs.triangulated_joints to see all 26 joints
//     std::vector<cv::Point3f> target_pts_viz;
//     for (size_t i = 0; i < inputs.triangulated_joints.size(); ++i) {
//         if (inputs.triangulated_scores[i] >= kTriangulationMinScore && 
//             IsFinitePoint3(inputs.triangulated_joints[i])) {
//             target_pts_viz.push_back(inputs.triangulated_joints[i]);
//         }
//     }
    
//     if (!target_pts_viz.empty()) {
//         cv::viz::WCloud target_cloud(target_pts_viz, cv::viz::Color::green());
//         target_cloud.setRenderingProperty(cv::viz::POINT_SIZE, 8.0);
//         viewer.showWidget("RTMPose Targets", target_cloud);
//     }
// #endif

    for (int iter = 0; iter < 60; ++iter) {
        optimizer.zero_grad();

        std::vector<torch::jit::IValue> fwd_inputs;
        fwd_inputs.emplace_back(betas);
        fwd_inputs.emplace_back(expression);
        fwd_inputs.emplace_back(global_orient);
        fwd_inputs.emplace_back(body_pose);
        fwd_inputs.emplace_back(jaw);
        fwd_inputs.emplace_back(eye);
        fwd_inputs.emplace_back(left_hand);
        fwd_inputs.emplace_back(right_hand);

        const auto out = smplx_layer->forward(fwd_inputs).toTuple();
        if (!out || out->elements().size() < 3u) {
            return false;
        }

        auto joints_world = out->elements()[2].toTensor().squeeze(0) + translation.squeeze(0);
        auto joints = joints_world.index_select(0, idx_tensor);

        auto diff = joints - target_tensor;
        auto data_loss = ((diff * diff).sum(1) * weight_tensor).mean();
        auto orient_reg = (global_orient - global_orient_init).pow(2).mean() * 1e-2;
        auto body_reg = (body_pose - body_pose_init).pow(2).mean() * 1e-2;
        auto total_loss = data_loss + orient_reg + body_reg; 
        total_loss.backward();
        optimizer.step();

// #if DATASET_PREP_HAS_OPENCV_VIZ
//         if ((iter % kVizRenderStride == 0) && !viewer.wasStopped()) {
//             auto joints_cpu = joints_world.detach().to(torch::kCPU).contiguous();
//             std::vector<cv::Point3f> smpl_pts_viz;
//             smpl_pts_viz.reserve(static_cast<size_t>(joints_cpu.size(0)));
//             for (int64_t i = 0; i < joints_cpu.size(0); ++i) {
//                 smpl_pts_viz.emplace_back(
//                     joints_cpu[i][0].item<float>(),
//                     joints_cpu[i][1].item<float>(),
//                     joints_cpu[i][2].item<float>());
//             }

//             cv::viz::WCloud smpl_cloud(smpl_pts_viz, cv::viz::Color::red());
//             smpl_cloud.setRenderingProperty(cv::viz::POINT_SIZE, 6.0);
//             viewer.showWidget("SMPL Joints", smpl_cloud);
//             viewer.spinOnce(kVizRenderDelayMs, true);
//         }
// #endif
    }

    auto go_cpu = global_orient.detach().to(torch::kCPU).contiguous();
    auto bp_cpu = body_pose.detach().to(torch::kCPU).contiguous();
    auto tr_cpu = translation.detach().to(torch::kCPU).contiguous();

    out_result->global_orient.assign(go_cpu.data_ptr<float>(), go_cpu.data_ptr<float>() + go_cpu.numel());
    out_result->body_pose.assign(bp_cpu.data_ptr<float>(), bp_cpu.data_ptr<float>() + bp_cpu.numel());
    out_result->translation_world = cv::Vec3f(
        tr_cpu[0][0].item<float>(),
        tr_cpu[0][1].item<float>(),
        tr_cpu[0][2].item<float>());

    out_result->optimized_joints.clear();
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> fwd_inputs;
        fwd_inputs.emplace_back(betas);
        fwd_inputs.emplace_back(expression);
        fwd_inputs.emplace_back(global_orient.detach());
        fwd_inputs.emplace_back(body_pose.detach());
        fwd_inputs.emplace_back(jaw);
        fwd_inputs.emplace_back(eye);
        fwd_inputs.emplace_back(left_hand);
        fwd_inputs.emplace_back(right_hand);

        const auto out = smplx_layer->forward(fwd_inputs).toTuple();
        if (!out || out->elements().size() < 3u) {
            return false;
        }

        auto joints_world = out->elements()[2].toTensor().squeeze(0) + translation.detach().squeeze(0);
        auto joints_final_cpu = joints_world.to(torch::kCPU).contiguous();
        out_result->optimized_joints.reserve(static_cast<size_t>(joints_final_cpu.size(0)));
        for (int64_t i = 0; i < joints_final_cpu.size(0); ++i) {
            out_result->optimized_joints.emplace_back(
                joints_final_cpu[i][0].item<float>(),
                joints_final_cpu[i][1].item<float>(),
                joints_final_cpu[i][2].item<float>());
        }
    }
    return true;
#endif
}

}  // namespace dataset_prep
