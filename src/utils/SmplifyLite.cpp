#include "SmplifyLite.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>

#include "HmrInferenceUtils.h"
#include "HmrMathHelpers.h"
#include "HmrOverlayHelpers.h"
#include "SmplLBS.h"

namespace {

struct KptMap {
    int yolo_idx;
    int smpl_idx;
};
 
const KptMap kYoloToSmpl[] = { 
    {5, 16},   // left_shoulder
    {6, 17},   // right_shoulder
    {7, 18},   // left_elbow
    {8, 19},   // right_elbow
    {9, 20},   // left_wrist
    {10, 21},  // right_wrist
    {11, 1},   // left_hip
    {12, 2},   // right_hip
    {13, 4},   // left_knee
    {14, 5},   // right_knee
    {15, 7},   // left_ankle
    {16, 8},   // right_ankle
};

torch::Tensor BuildTargetTensor(const std::vector<cv::Point2f>& keypoints,
                                const std::vector<float>& scores,
                                float min_score,
                                std::vector<int>* out_smpl_indices,
                                std::vector<float>* out_weights) {
    out_smpl_indices->clear();
    out_weights->clear();
    std::vector<float> target_xy;
    for (const auto& mapping : kYoloToSmpl) {
        const int kp_idx = mapping.yolo_idx;
        if (kp_idx < 0 || kp_idx >= static_cast<int>(keypoints.size())) {
            continue;
        }
        const float score = scores[kp_idx];
        if (score < min_score) {
            continue;
        }
        const cv::Point2f& p = keypoints[kp_idx];
        target_xy.push_back(p.x);
        target_xy.push_back(p.y);
        out_smpl_indices->push_back(mapping.smpl_idx);
        out_weights->push_back(score);
    }
    if (target_xy.empty()) {
        return torch::Tensor();
    }
    return torch::from_blob(target_xy.data(), {static_cast<int64_t>(target_xy.size() / 2), 2},
                            torch::kFloat).clone();
}


void DrawJointsOverlayMetric(cv::Mat& frame,
                             const torch::Tensor& joints,
                             float tx,
                             float ty,
                             float tz,
                             float y_sign,
                             float f,
                             float cx,
                             float cy) {
    if (std::abs(tz) < 1e-9f) return;
    auto joints_cpu = joints.squeeze(0).to(torch::kCPU).contiguous();
    const int width = frame.cols;
    const int height = frame.rows;
    auto joints_acc = joints_cpu.accessor<float, 2>();
    for (int i = 0; i < joints_acc.size(0); ++i) {
        const float X = joints_acc[i][0] + tx;
        const float Y = joints_acc[i][1] * y_sign + ty;
        const float Z = joints_acc[i][2] + tz;
        const float u = (f * X / (Z + 1e-9f)) + cx;
        const float v = (f * Y / (Z + 1e-9f)) + cy;
        const int ui = static_cast<int>(u);
        const int vi = static_cast<int>(v);
        if (ui < 0 || vi < 0 || ui >= width || vi >= height) continue;
        cv::circle(frame, cv::Point(ui, vi), 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
}

void DrawYoloKeypointsOverlay(cv::Mat& frame,
                              const std::vector<cv::Point2f>& keypoints,
                              const std::vector<float>& scores,
                              float min_score) {
    const int width = frame.cols;
    const int height = frame.rows;
    const int count = std::min(keypoints.size(), scores.size());
    for (int i = 0; i < count; ++i) {
        if (scores[i] < min_score) {
            continue;
        }
        const int ui = static_cast<int>(keypoints[i].x + 0.5f);
        const int vi = static_cast<int>(keypoints[i].y + 0.5f);
        if (ui < 0 || vi < 0 || ui >= width || vi >= height) {
            continue;
        }
        cv::circle(frame, cv::Point(ui, vi), 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }
}

void DrawVerticesOverlayMetric(cv::Mat& frame,
                               const torch::Tensor& verts,
                               float tx,
                               float ty,
                               float tz,
                               float y_sign,
                               float f,
                               float cx,
                               float cy) {
    if (std::abs(tz) < 1e-9f) return;
    auto verts_cpu = verts.squeeze(0).to(torch::kCPU).contiguous();
    const int width = frame.cols;
    const int height = frame.rows;
    auto verts_acc = verts_cpu.accessor<float, 2>();
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0] + tx;
        const float Y = verts_acc[i][1] * y_sign + ty;
        const float Z = verts_acc[i][2] + tz;
        const float u = (f * X / (Z + 1e-9f)) + cx;
        const float v = (f * Y / (Z + 1e-9f)) + cy;
        const int ui = static_cast<int>(u);
        const int vi = static_cast<int>(v);
        if (ui < 0 || vi < 0 || ui >= width || vi >= height) continue;
        cv::circle(frame, cv::Point(ui, vi), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
}

} // namespace

bool SmplifyLite(SMPLLayer& smpl_layer,
                 const std::vector<cv::Point2f>& keypoints,
                 const std::vector<float>& keypoint_scores,
                 float crop_cx, float crop_cy, float crop_size,
                 float f_geo, float f_render, float img_w, float img_h,
                 SmplResult* io_res,
                 const SmplifyLiteOptions& options,
                 float* out_y_sign,
                 const cv::Mat* render_frame,
                 const std::string* render_dir,
                 int frame_idx) {
    if (!io_res || keypoints.empty() || keypoints.size() != keypoint_scores.size()) {
        std::cout << "[SmplifyLite] Skipping: invalid keypoints." << std::endl;
        return false;
    }

    std::vector<int> smpl_indices;
    std::vector<float> weights_vec;
    auto target_xy = BuildTargetTensor(keypoints, keypoint_scores, options.keypoint_threshold,
                                       &smpl_indices, &weights_vec);
    if (!target_xy.defined() || smpl_indices.empty()) {
        std::cout << "[SmplifyLite] Skipping: no keypoints above threshold." << std::endl;
        return false;
    }

    auto weights = torch::from_blob(weights_vec.data(),
                                    {static_cast<int64_t>(weights_vec.size())},
                                    torch::kFloat).clone();
    weights = weights / (weights.mean() + 1e-8f);

    auto pose_init = PoseToAxisAngle(*io_res).clone();
    auto betas_init = torch::from_blob(io_res->shape.data(),
                                       {1, static_cast<int64_t>(io_res->shape.size())},
                                       torch::kFloat).clone();

    auto pose = pose_init.clone().detach().set_requires_grad(true);
    auto betas = betas_init.clone().detach().set_requires_grad(true);
    auto pose_mask = torch::ones_like(pose);
    if (options.lock_wrist_ankle) {
        auto lock_idx = torch::tensor({7, 8, 20, 21}, torch::kLong);
        pose_mask.index_put_({0, lock_idx, torch::indexing::Slice()}, 0);
    }

    // FIXED: Make Translation a proper Tensor with Gradients
    const cv::Vec3f t_init_val = EstimateTranslation(io_res->camera, crop_cx, crop_cy, crop_size,
                                                     f_geo, img_w, img_h);
    auto t_tensor = torch::tensor({t_init_val[0], t_init_val[1], t_init_val[2]}, torch::kFloat)
                        .reshape({1, 3})
                        .detach()
                        .requires_grad_(options.optimize_translation);
                        
    const float cx = img_w * 0.5f;
    const float cy = img_h * 0.5f;
    const float y_sign = out_y_sign ? *out_y_sign : -1.0f;
  
    // Add t_tensor to optimizer
    std::vector<torch::Tensor> optim_params = {pose, betas};
    if (options.optimize_translation) {
        optim_params.push_back(t_tensor);
    }
    torch::optim::Adam optimizer(optim_params, torch::optim::AdamOptions(options.lr));

    // Tuning Constants
    // We heavily upweight height to compete with pixel error
    // Typical Data Loss: ~1000-5000 (after normalization)
    // Typical Height Loss: ~0.01 (meters squared) -> Needs ~1e5 multiplier
    const float w_height_internal = 100000.0f; 
    
    auto render_step = [&](int iter, const torch::Tensor& verts, const torch::Tensor& joints) {
        if (!render_frame || !render_dir || render_dir->empty() || options.render_every <= 0) {
            return;
        }
        if ((iter) % options.render_every != 0) {
            return;
        }
        std::filesystem::path out_dir = std::filesystem::path(*render_dir) / "smplify_steps";
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            return;
        }

        const float tz = std::max(t_tensor[0][2].item<float>(), 1e-6f);
        const float tx_val = t_tensor[0][0].item<float>();
        const float ty_val = t_tensor[0][1].item<float>();

        cv::Mat overlay = render_frame->clone();
        DrawVerticesOverlayMetric(overlay, verts, tx_val, ty_val, tz, y_sign, f_geo, cx, cy);
        DrawJointsOverlayMetric(overlay, joints, tx_val, ty_val, tz, y_sign, f_geo, cx, cy);
        DrawYoloKeypointsOverlay(overlay, keypoints, keypoint_scores, options.keypoint_threshold);

        std::ostringstream name;
        name << "step_" << std::setw(6) << std::setfill('0') << frame_idx
             << "_iter_" << std::setw(3) << std::setfill('0') << (iter + 1) << ".png";
        const auto out_path = out_dir / name.str();
        cv::imwrite(out_path.string(), overlay);
    };

    for (int iter = 0; iter < options.num_iters; ++iter) {
        optimizer.zero_grad();
        
        // Forward Pass (SMPL)
        // We pass zero translation here because we apply t_tensor manually in projection
        const auto trans_zeros = torch::zeros({1, 3}, torch::kFloat);
        auto pose_eff = pose * pose_mask + pose_init * (1 - pose_mask);
        auto smpl_out = smpl_layer.forward(betas, pose_eff, trans_zeros);
        auto joints = smpl_out.joints.squeeze(0);

        // --- Data Loss (Reprojection) ---
        auto idx_tensor = torch::from_blob(smpl_indices.data(),
                                           {static_cast<int64_t>(smpl_indices.size())},
                                           torch::kInt).to(torch::kLong);
        auto selected = joints.index_select(0, idx_tensor);
        
        // Apply optimized translation
        auto X = selected.index({torch::indexing::Slice(), 0}) + t_tensor[0][0];
        auto Y = selected.index({torch::indexing::Slice(), 1}) * y_sign + t_tensor[0][1];
        auto Z = selected.index({torch::indexing::Slice(), 2}) + t_tensor[0][2];
        
        // Project
        auto u = (f_geo * X / (Z + 1e-9f)) + cx;
        auto v = (f_geo * Y / (Z + 1e-9f)) + cy;
        auto proj = torch::stack({u, v}, 1);


        // Render Mesh
        render_step(iter, smpl_out.vertices, smpl_out.joints);
  
        auto diff = proj - target_xy;
        auto loss_sq = (diff * diff).sum(1);
        
        // Normalize Data Loss (Mean instead of Sum)
        // This keeps loss magnitude stable regardless of keypoint count
        auto data_loss = (loss_sq * weights).mean(); 

        // --- Height Loss ---
        // Estimate height as (Max Y - Min Y) of all joints
        // This is robust to pose variations (like T-pose vs A-pose)
        auto y_coords = joints.index({torch::indexing::Slice(), 1});
        auto height_est = y_coords.max() - y_coords.min();
        auto height_loss = (height_est - options.target_height).pow(2) * w_height_internal;

        // --- Regularizers ---
        auto pose_reg = ((pose - pose_init) * pose_mask).pow(2).mean() * options.pose_reg;
        auto betas_reg = (betas - betas_init).pow(2).mean() * options.betas_reg;

        auto total = data_loss + pose_reg + betas_reg + height_loss;
        total.backward();
        optimizer.step();
        if (options.lock_wrist_ankle) {
            torch::NoGradGuard no_grad;
            auto clamped = pose * pose_mask + pose_init * (1 - pose_mask);
            pose.copy_(clamped);
        }

        if (options.log_every > 0 &&
            (iter == 0 || (iter + 1) % options.log_every == 0 || iter + 1 == options.num_iters)) {
            std::cout << "[SmplifyLite] iter " << (iter + 1)
                      << " total=" << total.item<float>()
                      << " data=" << data_loss.item<float>()
                      << " height=" << height_loss.item<float>()
                      << " h_est=" << height_est.item<float>() 
                      << " t_z=" << t_tensor[0][2].item<float>()
                      << std::endl;
        }
    }

    // Copy results back to CPU
    auto pose_cpu = pose.detach().cpu();
    auto betas_cpu = betas.detach().cpu();
    auto t_cpu = t_tensor.detach().cpu();

    io_res->pose.assign(pose_cpu.data_ptr<float>(), pose_cpu.data_ptr<float>() + pose_cpu.numel());
    io_res->shape.assign(betas_cpu.data_ptr<float>(), betas_cpu.data_ptr<float>() + betas_cpu.numel());

    if (out_y_sign) {
        *out_y_sign = y_sign;
    }
    
    // Update Camera (Reverse Engineering Scale 's')
    // We need to convert the new metric depth (t_z) back into the weak-perspective scale 's'
    // expected by the renderer, preserving the new 3D position.
    if (io_res->camera.size() >= 3) {
        const float tz_final = std::max(t_cpu[0][2].item<float>(), 1e-6f);
        // From Eq: tz = 2 * f_geo / (crop_size * s)
        // Therefore: s = 2 * f_geo / (crop_size * tz)
        const float s_new = 2.0f * f_geo / (crop_size * tz_final);
        
        // Update Translation (tx, ty) in weak-perspective screen space
        const float tx_metric = t_cpu[0][0].item<float>();
        const float ty_metric = t_cpu[0][1].item<float>();
        
        // Project center (0,0,0) + t to screen
        // u_center = f * tx / z + cx
        const float u_screen = (f_geo * tx_metric / tz_final) + cx;
        const float v_screen = (f_geo * ty_metric / tz_final) + cy;
        
        // Convert screen coords back to normalized crop coords [-1, 1]
        // u = crop_cx + tx_param * (crop_size * s / 2)
        // scale_pixels = crop_size * s
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
