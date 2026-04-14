#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include "utils/HmrMathHelpers.h"
#include "utils/train/TrainTypes.h"

struct TrainDataGPU
{
    torch::Tensor all_poses; // (N, 72)
    torch::Tensor all_trans; // (N, 3)
    torch::Tensor all_expression; // (N, 250)
    torch::Tensor all_jaw_pose; // (N, 3)
    torch::Tensor all_eye_pose; // (N, 6)
    torch::Tensor all_left_hand_pose; // (N, 45)
    torch::Tensor all_right_hand_pose; // (N, 45)
    torch::Tensor all_time;  // (N, 1)
    torch::Tensor all_crops; // (N, 3) -> [cx, cy, size] normalized

    TrainDataGPU(const std::vector<TrainSample> &samples, torch::Device device)
    {
        constexpr int64_t kSmplPoseParamCount = 72;
        constexpr int64_t kSmplxExpressionParamCount = 250;
        constexpr int64_t kSmplxJawPoseParamCount = 3;
        constexpr int64_t kSmplxEyePoseParamCount = 6;
        constexpr int64_t kSmplxHandPoseParamCount = 45;

        auto append_padded_vector = [](const std::vector<float> &source, size_t target_count, std::vector<float> *dest)
        {
            if (dest == nullptr)
            {
                return;
            }
            const size_t copy_count = std::min(source.size(), target_count);
            dest->insert(dest->end(), source.begin(), source.begin() + static_cast<std::ptrdiff_t>(copy_count));
            dest->insert(dest->end(), target_count - copy_count, 0.0f);
        };

        const int64_t N = static_cast<int64_t>(samples.size());
        std::vector<float> flat_poses;
        std::vector<float> flat_trans;
        std::vector<float> flat_expression;
        std::vector<float> flat_jaw_pose;
        std::vector<float> flat_eye_pose;
        std::vector<float> flat_left_hand_pose;
        std::vector<float> flat_right_hand_pose;
        std::vector<float> flat_time;
        std::vector<float> flat_crops;
        flat_poses.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplPoseParamCount));
        flat_trans.reserve(static_cast<size_t>(N) * 3u);
        flat_expression.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxExpressionParamCount));
        flat_jaw_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxJawPoseParamCount));
        flat_eye_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxEyePoseParamCount));
        flat_left_hand_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxHandPoseParamCount));
        flat_right_hand_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxHandPoseParamCount));
        flat_time.reserve(static_cast<size_t>(N));
        flat_crops.reserve(static_cast<size_t>(N) * 3u);

        for (int64_t i = 0; i < N; ++i)
        {
            const auto &s = samples[static_cast<size_t>(i)];
            if (s.pose.size() == 72)
            {
                flat_poses.insert(flat_poses.end(), s.pose.begin(), s.pose.end());
            }
            else if (s.pose.size() == 144)
            {
                const auto pose_aa = ConvertPose6dToAxisAngle(s.pose);
                flat_poses.insert(flat_poses.end(), pose_aa.begin(), pose_aa.end());
            }
            else if (s.pose.size() % 3 == 0 && (s.pose.size() / 3) == 24)
            {
                flat_poses.insert(flat_poses.end(), s.pose.begin(), s.pose.end());
            }
            else
            {
                throw std::runtime_error("Unsupported pose size: " + std::to_string(s.pose.size()));
            }

            append_padded_vector(s.smplx_expression, static_cast<size_t>(kSmplxExpressionParamCount), &flat_expression);
            append_padded_vector(s.smplx_jaw_pose, static_cast<size_t>(kSmplxJawPoseParamCount), &flat_jaw_pose);
            append_padded_vector(s.smplx_eye_pose, static_cast<size_t>(kSmplxEyePoseParamCount), &flat_eye_pose);
            append_padded_vector(s.smplx_left_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount), &flat_left_hand_pose);
            append_padded_vector(s.smplx_right_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount), &flat_right_hand_pose);

            cv::Vec3f t;
            if (s.has_translation)
            {
                t = cv::Vec3f(s.translation[0], s.translation[1], s.translation[2]);
            }
            else
            {
                t = EstimateTranslation(s.cam, s.crop_cx, s.crop_cy,
                                        s.crop_size, s.focal_length,
                                        static_cast<float>(s.img_w), static_cast<float>(s.img_h));
            }
            flat_trans.push_back(t[0]);
            flat_trans.push_back(t[1]);
            flat_trans.push_back(t[2]);

            const float img_w = std::max(1.0f, static_cast<float>(s.img_w));
            const float img_h = std::max(1.0f, static_cast<float>(s.img_h));
            const float norm_cx = s.crop_cx / img_w;
            const float norm_cy = s.crop_cy / img_h;
            const float norm_size = s.crop_size / img_w;
            flat_crops.push_back(norm_cx);
            flat_crops.push_back(norm_cy);
            flat_crops.push_back(norm_size);

            const float time_val = static_cast<float>(i) /
                                   static_cast<float>(std::max<int64_t>(1, N - 1));
            flat_time.push_back(time_val);
        }

        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        all_poses = torch::from_blob(flat_poses.data(), {N, kSmplPoseParamCount}, opts).clone().to(device);
        all_trans = torch::from_blob(flat_trans.data(), {N, 3}, opts).clone().to(device);
        all_expression = torch::from_blob(flat_expression.data(), {N, kSmplxExpressionParamCount}, opts).clone().to(device);
        all_jaw_pose = torch::from_blob(flat_jaw_pose.data(), {N, kSmplxJawPoseParamCount}, opts).clone().to(device);
        all_eye_pose = torch::from_blob(flat_eye_pose.data(), {N, kSmplxEyePoseParamCount}, opts).clone().to(device);
        all_left_hand_pose = torch::from_blob(flat_left_hand_pose.data(), {N, kSmplxHandPoseParamCount}, opts).clone().to(device);
        all_right_hand_pose = torch::from_blob(flat_right_hand_pose.data(), {N, kSmplxHandPoseParamCount}, opts).clone().to(device);
        all_time = torch::from_blob(flat_time.data(), {N, 1}, opts).clone().to(device);
        all_crops = torch::from_blob(flat_crops.data(), {N, 3}, opts).clone().to(device);
    }
};
