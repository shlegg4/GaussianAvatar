#include "SmplifyLite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>

#include "HmrInferenceUtils.h"
#include "HmrMathHelpers.h"
#include "HmrOverlayHelpers.h"
#include "SmplLBS.h"

namespace
{

    struct KptMap
    {
        int kpt_idx;
        int smpl_idx;
    };

    struct VertMap
    {
        int kpt_idx;
        int vert_idx;
    };

    const KptMap kRtmPoseToSmpl[] = {
        // RTMPose Halpe26 order.
        {5, 16},  // left_shoulder
        {6, 17},  // right_shoulder
        {7, 18},  // left_elbow
        {8, 19},  // right_elbow
        {9, 20},  // left_wrist
        {10, 21}, // right_wrist
        {11, 1},  // left_hip
        {12, 2},  // right_hip
        {13, 4},  // left_knee
        {14, 5},  // right_knee
        {15, 7},  // left_ankle
        {16, 8},  // right_ankle
        {20, 10}, // left_big_toe -> left_foot
        {22, 10}, // left_small_toe -> left_foot
        {24, 10}, // left_heel -> left_foot
        {21, 11}, // right_big_toe -> right_foot
        {23, 11}, // right_small_toe -> right_foot
        {25, 11}, // right_heel -> right_foot
    };

    const VertMap kRtmPoseFaceToSmplVerts[] = {
        // RTMPose Halpe26 face -> SMPL-H vertex ids.
        {0, 332},  // nose
        {1, 2800}, // left_eye
        {2, 6260}, // right_eye
        {3, 583},  // left_ear
        {4, 4071}, // right_ear
    };

    torch::Tensor BuildTargetTensor(const std::vector<cv::Point2f> &keypoints,
                                    const std::vector<float> &scores,
                                    float min_score,
                                    float weight_floor,
                                    float weight_pow,
                                    std::vector<int> *out_smpl_indices,
                                    std::vector<float> *out_weights)
    {
        out_smpl_indices->clear();
        out_weights->clear();
        std::vector<float> target_xy;
        for (const auto &mapping : kRtmPoseToSmpl)
        {
            const int kp_idx = mapping.kpt_idx;
            if (kp_idx < 0 || kp_idx >= static_cast<int>(keypoints.size()))
            {
                continue;
            }
            const float score = scores[kp_idx]; 
            const cv::Point2f &p = keypoints[kp_idx];
            target_xy.push_back(p.x);
            target_xy.push_back(p.y);
            const float w = std::pow(std::max(score, weight_floor), weight_pow);
            out_smpl_indices->push_back(mapping.smpl_idx);
            out_weights->push_back(w);
        }
        if (target_xy.empty())
        {
            return torch::Tensor();
        }
        return torch::from_blob(target_xy.data(), {static_cast<int64_t>(target_xy.size() / 2), 2},
                                torch::kFloat)
            .clone();
    }

    torch::Tensor BuildVertexTargetTensor(const std::vector<cv::Point2f> &keypoints,
                                          const std::vector<float> &scores,
                                          float min_score,
                                          float weight_floor,
                                          float weight_pow,
                                          std::vector<int> *out_vert_indices,
                                          std::vector<float> *out_weights)
    {
        out_vert_indices->clear();
        out_weights->clear();
        std::vector<float> target_xy;
        for (const auto &mapping : kRtmPoseFaceToSmplVerts)
        {
            const int kp_idx = mapping.kpt_idx;
            if (kp_idx < 0 || kp_idx >= static_cast<int>(keypoints.size()))
            {
                continue;
            }
            const float score = scores[kp_idx];
            if (score < min_score)
            {
                continue;
            }
            const cv::Point2f &p = keypoints[kp_idx];
            target_xy.push_back(p.x);
            target_xy.push_back(p.y);
            const float w = std::pow(std::max(score, weight_floor), weight_pow);
            out_vert_indices->push_back(mapping.vert_idx);
            out_weights->push_back(w);
        }
        if (target_xy.empty())
        {
            return torch::Tensor();
        }
        return torch::from_blob(target_xy.data(), {static_cast<int64_t>(target_xy.size() / 2), 2},
                                torch::kFloat)
            .clone();
    }

    torch::Tensor Matx33fToTensor(const cv::Matx33f& m, torch::Device device)
    {
        return torch::from_blob((void*)m.val, {3, 3}, torch::kFloat).clone().to(device);
    }

    torch::Tensor Vec3fToTensor(const cv::Vec3f& v, torch::Device device)
    {
        std::array<float, 3> data = {v[0], v[1], v[2]};
        return torch::from_blob(data.data(), {3}, torch::kFloat).clone().to(device);
    }

    void DrawJointsOverlayMetric(cv::Mat &frame,
                                 const torch::Tensor &joints,
                                 float tx,
                                 float ty,
                                 float tz,
                                 float y_sign,
                                 float f,
                                 float cx,
                                 float cy)
    {
        if (std::abs(tz) < 1e-9f)
            return;
        auto joints_cpu = joints.squeeze(0).to(torch::kCPU).contiguous();
        const int width = frame.cols;
        const int height = frame.rows;
        auto joints_acc = joints_cpu.accessor<float, 2>();
        for (int i = 0; i < joints_acc.size(0); ++i)
        {
            const float X = joints_acc[i][0] + tx;
            const float Y = joints_acc[i][1] * y_sign + ty;
            const float Z = joints_acc[i][2] + tz;
            const float u = (f * X / (Z + 1e-9f)) + cx;
            const float v = (f * Y / (Z + 1e-9f)) + cy;
            const int ui = static_cast<int>(u);
            const int vi = static_cast<int>(v);
            if (ui < 0 || vi < 0 || ui >= width || vi >= height)
                continue;
            cv::circle(frame, cv::Point(ui, vi), 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        }
    }

    void DrawKeypointsOverlay(cv::Mat &frame,
                              const std::vector<cv::Point2f> &keypoints,
                              const std::vector<float> &scores,
                              float min_score)
    {
        const int width = frame.cols;
        const int height = frame.rows;
        const int count = std::min(keypoints.size(), scores.size());
        for (int i = 0; i < count; ++i)
        {
            if (scores[i] < min_score)
            {
                continue;
            }
            const int ui = static_cast<int>(keypoints[i].x + 0.5f);
            const int vi = static_cast<int>(keypoints[i].y + 0.5f);
            if (ui < 0 || vi < 0 || ui >= width || vi >= height)
            {
                continue;
            }
            cv::circle(frame, cv::Point(ui, vi), 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        }
    }

    void DrawVerticesOverlayMetric(cv::Mat &frame,
                                   const torch::Tensor &verts,
                                   float tx,
                                   float ty,
                                   float tz,
                                   float y_sign,
                                   float f,
                                   float cx,
                                   float cy)
    {
        if (std::abs(tz) < 1e-9f)
            return;
        auto verts_cpu = verts.squeeze(0).to(torch::kCPU).contiguous();
        const int width = frame.cols;
        const int height = frame.rows;
        auto verts_acc = verts_cpu.accessor<float, 2>();
        for (int i = 0; i < verts_acc.size(0); ++i)
        {
            const float X = verts_acc[i][0] + tx;
            const float Y = verts_acc[i][1] * y_sign + ty;
            const float Z = verts_acc[i][2] + tz;
            const float u = (f * X / (Z + 1e-9f)) + cx;
            const float v = (f * Y / (Z + 1e-9f)) + cy;
            const int ui = static_cast<int>(u);
            const int vi = static_cast<int>(v);
            if (ui < 0 || vi < 0 || ui >= width || vi >= height)
                continue;
            cv::circle(frame, cv::Point(ui, vi), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }
    }

} // namespace

bool SmplifyLite(SMPLLayer &smpl_layer,
                 const std::vector<cv::Point2f> &keypoints,
                 const std::vector<float> &keypoint_scores,
                 float crop_cx, float crop_cy, float crop_size,
                 float f_geo, float f_render, float img_w, float img_h,
                 SmplResult *io_res,
                 const SmplifyLiteOptions &options,
                 float *out_y_sign,
                 const cv::Mat *render_frame,
                 const std::string *render_dir,
                 int frame_idx)
{
    if (!io_res || keypoints.empty() || keypoints.size() != keypoint_scores.size())
    {
        std::cout << "[SmplifyLite] Skipping: invalid keypoints." << std::endl;
        return false;
    }

    const auto device = smpl_layer.v_template.device();

    std::vector<int> smpl_indices;
    std::vector<float> weights_vec;
    auto target_xy = BuildTargetTensor(keypoints, keypoint_scores,
                                       options.keypoint_threshold,
                                       options.keypoint_weight_floor,
                                       options.keypoint_weight_pow,
                                       &smpl_indices, &weights_vec);
    if (!target_xy.defined() || smpl_indices.empty())
    {
        std::cout << "[SmplifyLite] Skipping: no keypoints above threshold." << std::endl;
        return false;
    }

    auto weights = torch::from_blob(weights_vec.data(),
                                    {static_cast<int64_t>(weights_vec.size())},
                                    torch::kFloat)
                       .clone();
    weights = weights / (weights.mean() + 1e-8f);

    std::vector<int> face_vert_indices;
    std::vector<float> face_weights_vec;
    auto face_target_xy = BuildVertexTargetTensor(keypoints, keypoint_scores,
                                                  options.keypoint_threshold,
                                                  options.keypoint_weight_floor,
                                                  options.keypoint_weight_pow,
                                                  &face_vert_indices, &face_weights_vec);
    torch::Tensor face_weights;
    if (face_target_xy.defined() && !face_weights_vec.empty())
    {
        face_weights = torch::from_blob(face_weights_vec.data(),
                                        {static_cast<int64_t>(face_weights_vec.size())},
                                        torch::kFloat)
                           .clone();
        face_weights = face_weights / (face_weights.mean() + 1e-8f);
    }

    target_xy = target_xy.to(device);
    weights = weights.to(device);
    if (face_target_xy.defined())
    {
        face_target_xy = face_target_xy.to(device);
    }
    if (face_weights.defined())
    {
        face_weights = face_weights.to(device);
    }

    auto pose_init = PoseToAxisAngle(*io_res).clone().to(device);
    auto betas_init = torch::from_blob(io_res->shape.data(),
                                       {1, static_cast<int64_t>(io_res->shape.size())},
                                       torch::kFloat)
                          .clone()
                          .to(device);

    auto pose = pose_init.clone().detach().set_requires_grad(true);
    auto betas = betas_init.clone().detach().set_requires_grad(true); 

    // FIXED: Make Translation a proper Tensor with Gradients
    const cv::Vec3f t_init_val = EstimateTranslation(io_res->camera, crop_cx, crop_cy, crop_size,
                                                     f_geo, img_w, img_h);
    auto t_tensor = torch::tensor({t_init_val[0], t_init_val[1], t_init_val[2]}, torch::kFloat)
                        .reshape({1, 3})
                        .to(device)
                        .detach()
                        .requires_grad_(options.optimize_translation);

    const float cx = img_w * 0.5f;
    const float cy = img_h * 0.5f;
    const float y_sign = out_y_sign ? *out_y_sign : -1.0f;
 

    std::vector<torch::optim::OptimizerParamGroup> param_groups; 
    param_groups.emplace_back(
        std::vector<torch::Tensor>{pose},
        std::make_unique<torch::optim::AdamOptions>(options.lr) 
    );

    std::vector<torch::Tensor> robust_params = {betas}; 
    robust_params.push_back(t_tensor);
        
    param_groups.emplace_back(
        robust_params,
        std::make_unique<torch::optim::AdamOptions>(options.lr * 100.0)
    );

    torch::optim::Adam optimizer(param_groups, torch::optim::AdamOptions(options.lr));
   
    const auto trans_zeros = torch::zeros({1, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device));
    float prev_loss_val = std::numeric_limits<float>::infinity();
    for (int iter = 0; iter < options.num_iters; ++iter)
    {
        optimizer.zero_grad();

        // Forward Pass (SMPL) 
        auto pose_eff = pose;
        auto smpl_out = smpl_layer.forward(betas, pose_eff, trans_zeros);
        auto joints = smpl_out.joints.squeeze(0);

        // --- Data Loss (Reprojection) ---
        auto idx_tensor = torch::from_blob(smpl_indices.data(),
                                           {static_cast<int64_t>(smpl_indices.size())},
                                           torch::kInt)
                              .to(torch::kLong);
        auto selected = joints.index_select(0, idx_tensor);

        // Apply optimized translation
        auto X = selected.index({torch::indexing::Slice(), 0}) + t_tensor[0][0];
        auto Y = selected.index({torch::indexing::Slice(), 1}) * y_sign + t_tensor[0][1];
        auto Z = selected.index({torch::indexing::Slice(), 2}) + t_tensor[0][2];

        // Project
        auto u = (f_geo * X / (Z + 1e-9f)) + cx;
        auto v = (f_geo * Y / (Z + 1e-9f)) + cy;
        auto proj = torch::stack({u, v}, 1); 
 
        auto diff = proj - target_xy;
        auto loss_sq = (diff * diff).sum(1);

        // Normalize Data Loss (Mean instead of Sum) 
        auto data_loss = (loss_sq * weights).mean();

        torch::Tensor face_loss = torch::zeros({}, torch::kFloat);
        if (options.face_weight > 0.0f && face_target_xy.defined() && !face_vert_indices.empty())
        {
            auto face_idx_tensor = torch::from_blob(face_vert_indices.data(),
                                                    {static_cast<int64_t>(face_vert_indices.size())},
                                                    torch::kInt)
                                       .to(torch::kLong);
            auto verts = smpl_out.vertices.squeeze(0);
            auto face_selected = verts.index_select(0, face_idx_tensor);

            auto Xf = face_selected.index({torch::indexing::Slice(), 0}) + t_tensor[0][0];
            auto Yf = face_selected.index({torch::indexing::Slice(), 1}) * y_sign + t_tensor[0][1];
            auto Zf = face_selected.index({torch::indexing::Slice(), 2}) + t_tensor[0][2];

            auto uf = (f_geo * Xf / (Zf + 1e-9f)) + cx;
            auto vf = (f_geo * Yf / (Zf + 1e-9f)) + cy;
            auto face_proj = torch::stack({uf, vf}, 1);

            auto face_diff = face_proj - face_target_xy;
            auto face_loss_sq = (face_diff * face_diff).sum(1);
            face_loss = (face_loss_sq * face_weights).mean();
        }

        // --- Height Loss --- 
        auto y_coords = joints.index({torch::indexing::Slice(), 1});
        auto height_est = y_coords.max() - y_coords.min();
        auto height_loss = (height_est - options.target_height).pow(2) * options.height_weight;

        // --- Regularizers ---
        auto pose_reg = ((pose - pose_init)).pow(2).mean() * options.pose_reg;
        auto betas_reg = (betas - betas_init).pow(2).mean() * options.betas_reg;

        auto total = data_loss + (face_loss * options.face_weight) + pose_reg + betas_reg + height_loss;
        const float curr_loss_val = total.item<float>();
        if (iter >= options.min_iters && std::abs(prev_loss_val - curr_loss_val) < options.loss_tol)
        {
            break;
        }
        prev_loss_val = curr_loss_val;
        total.backward();
        optimizer.step(); 
    }

    // Copy results back to CPU
    auto pose_cpu = pose.detach().cpu();
    auto betas_cpu = betas.detach().cpu();
    auto t_cpu = t_tensor.detach().cpu();

    io_res->pose.assign(pose_cpu.data_ptr<float>(), pose_cpu.data_ptr<float>() + pose_cpu.numel());
    io_res->shape.assign(betas_cpu.data_ptr<float>(), betas_cpu.data_ptr<float>() + betas_cpu.numel());

    if (out_y_sign)
    {
        *out_y_sign = y_sign;
    } 


    if (io_res->camera.size() >= 3)
    {
        const float tz_final = std::max(t_cpu[0][2].item<float>(), 1e-6f); 
        const float s_new = 2.0f * f_geo / (crop_size * tz_final);
 
        const float tx_metric = t_cpu[0][0].item<float>();
        const float ty_metric = t_cpu[0][1].item<float>();
 
        const float u_screen = (f_geo * tx_metric / tz_final) + cx;
        const float v_screen = (f_geo * ty_metric / tz_final) + cy;
 
        const float scale_pixels = crop_size * s_new;
        const float denom = scale_pixels * 0.5f + 1e-9f;

        const float tx_param = (u_screen - crop_cx) / denom;
        const float ty_param = (v_screen - crop_cy) / denom;

        io_res->camera[0] = s_new;
        io_res->camera[1] = tx_param;
        io_res->camera[2] = ty_param;
    }

    std::cout << "[SmplifyLite] Done." << std::endl;
    return true;
}

bool SmplifyLiteMultiView(SMPLLayer& smpl_layer,
                          const std::vector<SmplifyMultiViewObservation>& views,
                          SmplResult* io_res,
                          const SmplifyLiteOptions& options,
                          float* out_y_sign)
{
    if (!io_res || views.empty())
    {
        std::cout << "[SmplifyLiteMultiView] Skipping: invalid inputs." << std::endl;
        return false;
    }

    const auto device = smpl_layer.v_template.device();

    struct ViewData
    {
        torch::Tensor target_xy;
        torch::Tensor weights;
        torch::Tensor idx_tensor;
        torch::Tensor R;
        torch::Tensor t;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    std::vector<ViewData> valid_views;
    valid_views.reserve(views.size());

    for (const auto& view : views)
    {
        if (view.keypoints.empty() || view.keypoints.size() != view.keypoint_scores.size())
        {
            continue;
        }

        std::vector<int> smpl_indices;
        std::vector<float> weights_vec;
        auto target_xy = BuildTargetTensor(view.keypoints, view.keypoint_scores,
                                           options.keypoint_threshold,
                                           options.keypoint_weight_floor,
                                           options.keypoint_weight_pow,
                                           &smpl_indices, &weights_vec);
        if (!target_xy.defined() || smpl_indices.empty())
        {
            continue;
        }

        auto weights = torch::from_blob(weights_vec.data(),
                                        {static_cast<int64_t>(weights_vec.size())},
                                        torch::kFloat)
                           .clone();
        weights = weights / (weights.mean() + 1e-8f);

        auto idx_tensor = torch::from_blob(smpl_indices.data(),
                                           {static_cast<int64_t>(smpl_indices.size())},
                                           torch::kLong)
                              .clone();

        ViewData vd;
        vd.target_xy = target_xy.to(device);
        vd.weights = weights.to(device);
        vd.idx_tensor = idx_tensor.to(device);
        vd.R = Matx33fToTensor(view.R, device);
        vd.t = Vec3fToTensor(view.t, device);
        vd.fx = view.K(0, 0);
        vd.fy = view.K(1, 1);
        vd.cx = view.K(0, 2);
        vd.cy = view.K(1, 2);
        valid_views.push_back(std::move(vd));
    }

    if (valid_views.empty())
    {
        std::cout << "[SmplifyLiteMultiView] Skipping: no valid views." << std::endl;
        return false;
    }

    auto pose_init = PoseToAxisAngle(*io_res).clone().to(device);
    auto betas_init = torch::from_blob(io_res->shape.data(),
                                       {1, static_cast<int64_t>(io_res->shape.size())},
                                       torch::kFloat)
                          .clone()
                          .to(device);

    auto pose = pose_init.clone().detach().set_requires_grad(true);
    auto betas = betas_init.clone().detach().set_requires_grad(true);

    cv::Vec3f t_init_val(0.0f, 0.0f, 0.0f);
    if (io_res->camera.size() >= 3)
    {
        t_init_val = cv::Vec3f(io_res->camera[0], io_res->camera[1], io_res->camera[2]);
    }
    auto t_tensor = torch::tensor({t_init_val[0], t_init_val[1], t_init_val[2]}, torch::kFloat)
                        .reshape({1, 3})
                        .to(device)
                        .detach()
                        .requires_grad_(options.optimize_translation);

    std::vector<torch::optim::OptimizerParamGroup> param_groups;
    param_groups.emplace_back(
        std::vector<torch::Tensor>{pose},
        std::make_unique<torch::optim::AdamOptions>(options.lr)
    );

    std::vector<torch::Tensor> robust_params = {betas};
    robust_params.push_back(t_tensor);

    param_groups.emplace_back(
        robust_params,
        std::make_unique<torch::optim::AdamOptions>(options.lr * 100.0)
    );

    torch::optim::Adam optimizer(param_groups, torch::optim::AdamOptions(options.lr));

    float prev_loss_val = std::numeric_limits<float>::infinity();
    const float sigma2 = options.reproj_robust_sigma * options.reproj_robust_sigma;

    for (int iter = 0; iter < options.num_iters; ++iter)
    {
        optimizer.zero_grad();

        auto smpl_out = smpl_layer.forward(betas, pose, t_tensor);
        auto joints = smpl_out.joints.squeeze(0);

        torch::Tensor total_data = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        int view_count = 0;

        for (const auto& view : valid_views)
        {
            auto selected = joints.index_select(0, view.idx_tensor);
            auto Xc = torch::matmul(selected, view.R.t()) + view.t;
            auto X = Xc.index({torch::indexing::Slice(), 0});
            auto Y = Xc.index({torch::indexing::Slice(), 1});
            auto Z = Xc.index({torch::indexing::Slice(), 2});

            auto u = (view.fx * X / (Z + 1e-9f)) + view.cx;
            auto v = (view.fy * Y / (Z + 1e-9f)) + view.cy;
            auto proj = torch::stack({u, v}, 1);

            auto diff = proj - view.target_xy;
            auto dist2 = (diff * diff).sum(1);
            if (sigma2 > 0.0f)
            {
                dist2 = dist2 / (dist2 + sigma2);
            }
            auto data_loss = (dist2 * view.weights).mean();
            total_data = total_data + data_loss;
            ++view_count;
        }

        if (view_count > 0)
        {
            total_data = total_data / static_cast<float>(view_count);
        }

        auto pose_reg = (pose - pose_init).pow(2).mean() * options.pose_reg;
        auto betas_reg = (betas - betas_init).pow(2).mean() * options.betas_reg;

        auto total = total_data + pose_reg + betas_reg;
        const float curr_loss_val = total.item<float>();
        if (iter >= options.min_iters && std::abs(prev_loss_val - curr_loss_val) < options.loss_tol)
        {
            break;
        }
        prev_loss_val = curr_loss_val;
        total.backward();
        optimizer.step();
    }

    auto pose_cpu = pose.detach().cpu();
    auto betas_cpu = betas.detach().cpu();
    auto t_cpu = t_tensor.detach().cpu();

    io_res->pose.assign(pose_cpu.data_ptr<float>(), pose_cpu.data_ptr<float>() + pose_cpu.numel());
    io_res->shape.assign(betas_cpu.data_ptr<float>(), betas_cpu.data_ptr<float>() + betas_cpu.numel());
    io_res->camera.assign({t_cpu[0][0].item<float>(), t_cpu[0][1].item<float>(), t_cpu[0][2].item<float>()});

    if (out_y_sign)
    {
        *out_y_sign = 1.0f;
    }

    std::cout << "[SmplifyLiteMultiView] Done." << std::endl;
    return true;
}

namespace
{

struct MocapJointMap
{
    int mocap_idx;
    int smpl_idx;
};

constexpr int kMocapJointCount3D = 26;
constexpr int kSmplJointCount3D = 24;
constexpr int kSmplPoseParamCount3D = 72;
constexpr int kSmplShapeParamCount3D = 10;

const MocapJointMap kMocapToSmplTargetMap[] = {
    {11, 1},  {12, 2},  {13, 4},  {14, 5},  {15, 7},  {16, 8},  {18, 12},
    {17, 15}, {5, 16},  {6, 17},  {7, 18},  {8, 19},  {9, 20},  {10, 21},
    {20, 10}, {22, 10}, {24, 10}, {21, 11}, {23, 11}, {25, 11},
};

bool IsFinitePoint(const cv::Point3f& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool NormalizeQuaternion(cv::Vec4f* quaternion)
{
    if (quaternion == nullptr)
    {
        return false;
    }
    for (int i = 0; i < 4; ++i)
    {
        if (!std::isfinite((*quaternion)[i]))
        {
            return false;
        }
    }

    const float norm = std::sqrt(quaternion->dot(*quaternion));
    if (norm <= 1e-8f)
    {
        return false;
    }

    *quaternion *= (1.0f / norm);
    if ((*quaternion)[0] < 0.0f)
    {
        *quaternion *= -1.0f;
    }
    return true;
}

cv::Vec3f QuaternionToAxisAngle(cv::Vec4f quaternion)
{
    if (!NormalizeQuaternion(&quaternion))
    {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }

    const cv::Vec3f axis_part(quaternion[1], quaternion[2], quaternion[3]);
    const float sin_half_angle = static_cast<float>(cv::norm(axis_part));
    if (sin_half_angle <= 1e-8f)
    {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }

    const float angle = 2.0f * std::atan2(sin_half_angle, quaternion[0]);
    return axis_part * (angle / sin_half_angle);
}

float JointConfidenceAt(const std::vector<float>& joint_confidences, int index)
{
    if (index < 0 || index >= static_cast<int>(joint_confidences.size()))
    {
        return 1.0f;
    }
    return joint_confidences[index];
}

bool TryGetNormalizedQuaternion(const std::vector<cv::Vec4f>& joint_rotations,
                                int index,
                                cv::Vec4f* out_quaternion)
{
    if (out_quaternion == nullptr || index < 0 || index >= static_cast<int>(joint_rotations.size()))
    {
        return false;
    }

    *out_quaternion = joint_rotations[index];
    return NormalizeQuaternion(out_quaternion);
}

bool AverageQuaternions(const std::vector<cv::Vec4f>& joint_rotations,
                        const std::vector<int>& joint_indices,
                        cv::Vec4f* out_quaternion)
{
    if (out_quaternion == nullptr)
    {
        return false;
    }

    cv::Vec4f accumulated(0.0f, 0.0f, 0.0f, 0.0f);
    bool has_value = false;
    for (const int index : joint_indices)
    {
        cv::Vec4f quaternion;
        if (!TryGetNormalizedQuaternion(joint_rotations, index, &quaternion))
        {
            continue;
        }

        if (has_value && accumulated.dot(quaternion) < 0.0f)
        {
            quaternion *= -1.0f;
        }
        accumulated += quaternion;
        has_value = true;
    }

    if (!has_value)
    {
        return false;
    }

    *out_quaternion = accumulated;
    return NormalizeQuaternion(out_quaternion);
}

void SetPoseJointFromQuaternion(int smpl_joint_index,
                                const cv::Vec4f& quaternion,
                                std::vector<float>* pose_values)
{
    if (pose_values == nullptr || smpl_joint_index < 0 || smpl_joint_index >= kSmplJointCount3D)
    {
        return;
    }

    const cv::Vec3f axis_angle = QuaternionToAxisAngle(quaternion);
    const size_t base_index = static_cast<size_t>(smpl_joint_index) * 3u;
    (*pose_values)[base_index + 0u] = axis_angle[0];
    (*pose_values)[base_index + 1u] = axis_angle[1];
    (*pose_values)[base_index + 2u] = axis_angle[2];
}

void InitializePoseFromMocap(const std::vector<cv::Vec4f>& joint_rotations,
                             std::vector<float>* pose_values)
{
    if (pose_values == nullptr)
    {
        return;
    }

    pose_values->assign(kSmplPoseParamCount3D, 0.0f);

    const auto set_single = [&](int mocap_index, int smpl_index) {
        cv::Vec4f quaternion;
        if (TryGetNormalizedQuaternion(joint_rotations, mocap_index, &quaternion))
        {
            SetPoseJointFromQuaternion(smpl_index, quaternion, pose_values);
        }
    };

    set_single(19, 0);
    set_single(11, 1);
    set_single(12, 2);
    set_single(13, 4);
    set_single(14, 5);
    set_single(15, 7);
    set_single(16, 8);
    set_single(18, 12);
    set_single(17, 15);
    set_single(5, 16);
    set_single(6, 17);
    set_single(7, 18);
    set_single(8, 19);
    set_single(9, 20);
    set_single(10, 21);

    cv::Vec4f quaternion;
    if (AverageQuaternions(joint_rotations, {20, 22, 24}, &quaternion))
    {
        SetPoseJointFromQuaternion(10, quaternion, pose_values);
    }
    if (AverageQuaternions(joint_rotations, {21, 23, 25}, &quaternion))
    {
        SetPoseJointFromQuaternion(11, quaternion, pose_values);
    }
}

void AddTargetJoint(int mocap_index,
                    int smpl_index,
                    const std::vector<cv::Point3f>& joint_centers,
                    const std::vector<float>& joint_confidences,
                    float min_joint_confidence,
                    std::vector<int>* out_smpl_indices,
                    std::vector<float>* out_positions,
                    std::vector<float>* out_weights)
{
    if (out_smpl_indices == nullptr || out_positions == nullptr || out_weights == nullptr ||
        mocap_index < 0 || mocap_index >= static_cast<int>(joint_centers.size()))
    {
        return;
    }

    const float confidence = JointConfidenceAt(joint_confidences, mocap_index);
    if (confidence < min_joint_confidence || !IsFinitePoint(joint_centers[mocap_index]))
    {
        return;
    }

    out_smpl_indices->push_back(smpl_index);
    out_positions->push_back(joint_centers[mocap_index].x);
    out_positions->push_back(joint_centers[mocap_index].y);
    out_positions->push_back(joint_centers[mocap_index].z);
    out_weights->push_back(std::max(confidence, min_joint_confidence));
}

void BuildTargets(const std::vector<cv::Point3f>& joint_centers,
                  const std::vector<float>& joint_confidences,
                  float min_joint_confidence,
                  std::vector<int>* out_smpl_indices,
                  std::vector<float>* out_positions,
                  std::vector<float>* out_weights)
{
    if (out_smpl_indices == nullptr || out_positions == nullptr || out_weights == nullptr)
    {
        return;
    }

    out_smpl_indices->clear();
    out_positions->clear();
    out_weights->clear();

    if (joint_centers.size() < kMocapJointCount3D)
    {
        return;
    }

    AddTargetJoint(19, 0, joint_centers, joint_confidences, min_joint_confidence,
                   out_smpl_indices, out_positions, out_weights);

    if (out_smpl_indices->empty() || out_smpl_indices->front() != 0)
    {
        const bool has_left_hip =
            JointConfidenceAt(joint_confidences, 11) >= min_joint_confidence && IsFinitePoint(joint_centers[11]);
        const bool has_right_hip =
            JointConfidenceAt(joint_confidences, 12) >= min_joint_confidence && IsFinitePoint(joint_centers[12]);
        if (has_left_hip && has_right_hip)
        {
            const cv::Point3f pelvis = (joint_centers[11] + joint_centers[12]) * 0.5f;
            out_smpl_indices->push_back(0);
            out_positions->push_back(pelvis.x);
            out_positions->push_back(pelvis.y);
            out_positions->push_back(pelvis.z);
            out_weights->push_back(std::max(
                0.5f * (JointConfidenceAt(joint_confidences, 11) + JointConfidenceAt(joint_confidences, 12)),
                min_joint_confidence));
        }
    }

    for (const auto& mapping : kMocapToSmplTargetMap)
    {
        AddTargetJoint(mapping.mocap_idx, mapping.smpl_idx, joint_centers, joint_confidences,
                       min_joint_confidence, out_smpl_indices, out_positions, out_weights);
    }
}

}  // namespace

SmplifyLiteMocapSolver::SmplifyLiteMocapSolver(const std::string& model_path,
                                               const SmplMocapFitOptions& options)
    : options_(options)
{
    try
    {
        smpl_layer_ = std::make_shared<SMPLLayer>(model_path);
        torch::Device device(torch::kCPU);
        if (options_.use_cuda && torch::cuda::is_available())
        {
            device = torch::Device(torch::kCUDA);
        }
        smpl_layer_->to(device);
        smpl_layer_->eval();
    }
    catch (const std::exception& e)
    {
        std::cerr << "SmplifyLiteMocapSolver: failed to load " << model_path
                  << ": " << e.what() << std::endl;
        smpl_layer_.reset();
    }
}

bool SmplifyLiteMocapSolver::IsReady() const
{
    return smpl_layer_ != nullptr;
}

bool SmplifyLiteMocapSolver::FitToMocap(const std::vector<cv::Point3f>& joint_centers,
                                        const std::vector<cv::Vec4f>& joint_rotations,
                                        const std::vector<float>& joint_confidences,
                                        SmplParameters* out_parameters)
{
    if (!IsReady() || out_parameters == nullptr || joint_centers.size() < kMocapJointCount3D)
    {
        return false;
    }

    std::vector<cv::Vec4f> normalized_rotations = joint_rotations;
    normalized_rotations.resize(kMocapJointCount3D, cv::Vec4f(1.0f, 0.0f, 0.0f, 0.0f));

    std::vector<float> pose_init_values;
    InitializePoseFromMocap(normalized_rotations, &pose_init_values);

    out_parameters->thetas = pose_init_values;
    out_parameters->betas.assign(kSmplShapeParamCount3D, 0.0f);

    std::vector<int> smpl_indices;
    std::vector<float> target_positions;
    std::vector<float> target_weights;
    BuildTargets(joint_centers, joint_confidences, options_.min_joint_confidence,
                 &smpl_indices, &target_positions, &target_weights);
    if (smpl_indices.empty() || target_positions.empty() || target_weights.empty())
    {
        return false;
    }

    const auto device = smpl_layer_->v_template.device();
    const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat).device(device);

    auto pose_init = torch::from_blob(pose_init_values.data(),
                                      {1, kSmplJointCount3D, 3},
                                      torch::kFloat)
                         .clone()
                         .to(device);
    auto betas_init = torch::zeros({1, kSmplShapeParamCount3D}, tensor_options);
    auto trans_zeros = torch::zeros({1, 3}, tensor_options);

    auto idx_tensor = torch::from_blob(smpl_indices.data(),
                                       {static_cast<int64_t>(smpl_indices.size())},
                                       torch::kLong)
                          .clone()
                          .to(device);
    auto target_tensor = torch::from_blob(target_positions.data(),
                                          {static_cast<int64_t>(smpl_indices.size()), 3},
                                          torch::kFloat)
                             .clone()
                             .to(device);
    auto weight_tensor = torch::from_blob(target_weights.data(),
                                          {static_cast<int64_t>(target_weights.size())},
                                          torch::kFloat)
                             .clone()
                             .to(device);
    weight_tensor = weight_tensor / (weight_tensor.mean() + 1e-8f);

    auto trans_init = torch::zeros({1, 3}, tensor_options);
    {
        torch::NoGradGuard no_grad;
        auto smpl_out = smpl_layer_->forward(betas_init, pose_init, trans_zeros);
        auto selected = smpl_out.joints.squeeze(0).index_select(0, idx_tensor);
        trans_init = (target_tensor.mean(0) - selected.mean(0)).reshape({1, 3}).clone();
    }

    auto pose = pose_init.clone().detach().set_requires_grad(true);
    auto betas = betas_init.clone().detach().set_requires_grad(true);
    auto trans = trans_init.clone().detach().set_requires_grad(true);

    torch::optim::Adam optimizer({pose, betas, trans}, torch::optim::AdamOptions(options_.lr));

    for (int iter = 0; iter < options_.num_iters; ++iter)
    {
        optimizer.zero_grad();

        auto smpl_out = smpl_layer_->forward(betas, pose, trans_zeros);
        auto selected = smpl_out.joints.squeeze(0).index_select(0, idx_tensor) + trans.squeeze(0);
        auto diff = selected - target_tensor;

        auto data_loss = ((diff * diff).sum(1) * weight_tensor).mean() * options_.position_weight;
        auto pose_reg = (pose - pose_init).pow(2).mean() * options_.pose_reg;
        auto betas_reg = betas.pow(2).mean() * options_.betas_reg;
        auto trans_reg = (trans - trans_init).pow(2).mean() * options_.translation_reg;

        auto total_loss = data_loss + pose_reg + betas_reg + trans_reg;
        total_loss.backward();
        optimizer.step();
    }

    auto pose_cpu = pose.detach().cpu().contiguous().reshape({-1});
    auto betas_cpu = betas.detach().cpu().contiguous().reshape({-1});

    out_parameters->thetas.assign(pose_cpu.data_ptr<float>(),
                                  pose_cpu.data_ptr<float>() + pose_cpu.numel());
    out_parameters->betas.assign(betas_cpu.data_ptr<float>(),
                                 betas_cpu.data_ptr<float>() + betas_cpu.numel());
    return true;
}
