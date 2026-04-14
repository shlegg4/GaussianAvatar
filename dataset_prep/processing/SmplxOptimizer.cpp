#include "dataset_prep/processing/SmplxOptimizer.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "dataset_prep/DatasetPrepTypes.h"

#if DATASET_PREP_HAS_SMPL
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace dataset_prep {
namespace {

constexpr int kSmplxRootJointIndex = 0;
constexpr float kTriangulationMinScore = 0.000025f;

struct JointMapEntry {
    int mocap_index;
    int smpl_index;
};

const std::array<JointMapEntry, 20> kMocapToSmplJointMap = {{
    {11, 1}, {12, 2}, {13, 4}, {14, 5}, {15, 7}, {16, 8}, {18, 12},
    {17, 15}, {5, 16}, {6, 17}, {7, 18}, {8, 19}, {9, 20}, {10, 21},
    {20, 10}, {22, 10}, {24, 10}, {21, 11}, {23, 11}, {25, 11},
}};

bool IsFinitePoint3(const cv::Point3f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
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

    auto global_orient =
        MakeTorchInputTensor(inputs.init_global_orient, kSmplxGlobalOrientParamCount, device)
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

    out_result->global_orient.assign(go_cpu.data_ptr<float>(), go_cpu.data_ptr<float>() + go_cpu.numel());
    out_result->body_pose.assign(bp_cpu.data_ptr<float>(), bp_cpu.data_ptr<float>() + bp_cpu.numel());
    out_result->translation_world = cv::Vec3f(
        tr_cpu[0][0].item<float>(),
        tr_cpu[0][1].item<float>(),
        tr_cpu[0][2].item<float>());
    return true;
#endif
}

}  // namespace dataset_prep
