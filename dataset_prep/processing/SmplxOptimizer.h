#pragma once

#include <vector>

#include <opencv2/core.hpp>

#if DATASET_PREP_HAS_SMPL
#include <torch/script.h>
#endif

namespace dataset_prep {

#if DATASET_PREP_HAS_SMPL
using SmplxTorchModule = torch::jit::script::Module;
#else
struct SmplxTorchModule;
#endif

constexpr int64_t kSmplxShapeParamCount = 10;
constexpr int64_t kSmplxExpressionParamCount = 240;
constexpr int64_t kSmplxGlobalOrientParamCount = 3;
constexpr int64_t kSmplxBodyPoseParamCount = 63;
constexpr int64_t kSmplxJawPoseParamCount = 3;
constexpr int64_t kSmplxEyePoseParamCount = 6;
constexpr int64_t kSmplxHandPoseParamCount = 45;

std::vector<float> PadFloatVector(const std::vector<float>& values, size_t expected_count);

struct SmplxOptimizationInputs {
    std::vector<float> fixed_betas;
    std::vector<float> fixed_expression;
    std::vector<float> jaw_pose;
    std::vector<float> eye_pose;
    std::vector<float> left_hand_pose;
    std::vector<float> right_hand_pose;
    std::vector<float> init_global_orient;
    std::vector<float> init_body_pose;
    std::vector<cv::Point3f> triangulated_joints;
    std::vector<float> triangulated_scores;
};

struct SmplxOptimizationResult {
    std::vector<float> global_orient;
    std::vector<float> body_pose;
    cv::Vec3f translation_world{0.0f, 0.0f, 0.0f};
    std::vector<cv::Point3f> optimized_joints;
    std::vector<float> optimized_betas;
};

bool OptimizeSmplxPoseFromTriangulatedJoints(SmplxTorchModule* smplx_layer,
                                             const SmplxOptimizationInputs& inputs,
                                             SmplxOptimizationResult* out_result);

}  // namespace dataset_prep
