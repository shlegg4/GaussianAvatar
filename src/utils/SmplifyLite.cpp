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

#include <fstream>
#include <torch/torch.h>

void WritePlyPointCloud(const torch::Tensor &vertices, const std::string &filename)
{
    // Ensure we are working with a 2D CPU tensor: [Num_Vertices, 3]
    auto verts_cpu = vertices.squeeze(0).to(torch::kCPU).contiguous();
    auto verts_acc = verts_cpu.accessor<float, 2>();
    int num_verts = verts_acc.size(0);

    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // PLY Header
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << num_verts << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";

    // Write vertices
    for (int i = 0; i < num_verts; ++i)
    {
        out << verts_acc[i][0] << " " << verts_acc[i][1] << " " << verts_acc[i][2] << "\n";
    }
    out.close();
}
 
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

    torch::Tensor Matx33fToTensor(const cv::Matx33f &m, torch::Device device)
    {
        return torch::from_blob((void *)m.val, {3, 3}, torch::kFloat).clone().to(device);
    }

    torch::Tensor Vec3fToTensor(const cv::Vec3f &v, torch::Device device)
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
        std::make_unique<torch::optim::AdamOptions>(options.lr));

    std::vector<torch::Tensor> robust_params = {betas};
    robust_params.push_back(t_tensor);

    param_groups.emplace_back(
        robust_params,
        std::make_unique<torch::optim::AdamOptions>(options.lr * 100.0));

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

bool SmplifyLiteMultiView(SMPLLayer &smpl_layer,
                          const std::vector<SmplifyMultiViewObservation> &views,
                          SmplResult *io_res,
                          const SmplifyLiteOptions &options,
                          float *out_y_sign,
                          const std::vector<cv::Mat> *render_frames,
                          const std::string *render_dir,
                          int frame_idx)
{
    if (!io_res || views.empty())
    {
        std::cout << "[SmplifyLiteMultiView] Skipping: invalid inputs." << std::endl;
        return false;
    }

    const auto device = smpl_layer.v_template.device();

    struct ViewData
    {
        int original_view_idx = 0;
        torch::Tensor target_xy;
        torch::Tensor weights;
        torch::Tensor idx_tensor;

        // Face tracking tensors
        torch::Tensor face_target_xy;
        torch::Tensor face_weights;
        torch::Tensor face_idx_tensor;

        torch::Tensor halpe_idx_tensor;

        torch::Tensor R;
        torch::Tensor t;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    std::vector<ViewData> valid_views;
    valid_views.reserve(views.size());

    // 1. Strict 1-to-1 Halpe26 to SMPL24 mapping. 
    // Skips heels/extra toes to prevent triplicate joint fighting.
    const std::vector<std::pair<int, int>> halpe26_to_smpl = {
        // {11, 1},  // L Hip
        // {12, 2},  // R Hip
        // {13, 4},  // L Knee
        // {14, 5},  // R Knee
        // {15, 7},  // L Ankle
        // {16, 8},  // R Ankle
        // {20, 10}, // L Big Toe -> SMPL L Foot
        // {21, 11}, // R Big Toe -> SMPL R Foot
        // {18, 12}, // Neck
        // {5,  16}, // L Shoulder
        // {6,  17}, // R Shoulder
        {7,  18}, // L Elbow
        // {8,  19}, // R Elbow
        {9,  20}, // L Wrist
        // {10, 21}  // R Wrist
    };

    int v_idx = 0;
    for (const auto &view : views)
    {
        if (view.keypoints.empty() || view.keypoints.size() != view.keypoint_scores.size())
        {
            v_idx++;
            continue;
        }

        std::vector<int> halpe_indices;
        std::vector<int> smpl_indices;
        std::vector<float> weights_vec;
        std::vector<float> target_vec;

        // Map RTMPose (Halpe26) specifically to SMPL
        for (const auto& mapping : halpe26_to_smpl) {
            int halpe_idx = mapping.first;
            int smpl_idx = mapping.second;
            
            if (halpe_idx < view.keypoints.size()) {
                float score = view.keypoint_scores[halpe_idx];
                
                // Keep the raw score as long as it passes the floor threshold
                if (score > options.keypoint_threshold) {
                    halpe_indices.push_back(halpe_idx);
                    smpl_indices.push_back(smpl_idx);
                    target_vec.push_back(view.keypoints[halpe_idx].x);
                    target_vec.push_back(view.keypoints[halpe_idx].y);
                    
                    // Apply pow weighting if configured, otherwise use raw probability
                    float final_weight = std::pow(score, options.keypoint_weight_pow);
                    
                    // Optional: Boost arm weights slightly to help overcome the pose prior
                    if (smpl_idx >= 16 && smpl_idx <= 21) {
                        final_weight *= 2.0f; 
                    }
                    
                    weights_vec.push_back(final_weight);
                }
            }
        }

        if (smpl_indices.empty())
        {
            v_idx++;
            continue;
        }

        auto target_xy = torch::from_blob(target_vec.data(), {static_cast<int64_t>(smpl_indices.size()), 2}, torch::kFloat).clone();
        
        auto weights = torch::from_blob(weights_vec.data(), {static_cast<int64_t>(weights_vec.size())}, torch::kFloat).clone();
        // REMOVED: weights = weights / (weights.mean() + 1e-8f); // This was destroying the relative confidence scores!

        auto idx_tensor = torch::from_blob(smpl_indices.data(), {static_cast<int64_t>(smpl_indices.size())}, torch::kInt).to(torch::kLong);
        auto halpe_idx_tensor = torch::from_blob(halpe_indices.data(), {static_cast<int64_t>(halpe_indices.size())}, torch::kInt).to(torch::kLong);

        ViewData vd;
        vd.original_view_idx = v_idx;
        vd.target_xy = target_xy.to(device);
        vd.weights = weights.to(device);
        vd.idx_tensor = idx_tensor.to(device);
        vd.R = Matx33fToTensor(view.R, device);
        vd.t = Vec3fToTensor(view.t, device);
        vd.fx = view.K(0, 0);
        vd.fy = view.K(1, 1);
        vd.cx = view.K(0, 2);
        vd.cy = view.K(1, 2);
        vd.halpe_idx_tensor = halpe_idx_tensor.to(device);

        // --- Extract Face Keypoints ---
        if (options.face_weight > 0.0f)
        {
            std::vector<int> face_vert_indices;
            std::vector<float> face_weights_vec;
            auto face_target_xy = BuildVertexTargetTensor(view.keypoints, view.keypoint_scores,
                                                          options.keypoint_threshold,
                                                          options.keypoint_weight_floor,
                                                          options.keypoint_weight_pow,
                                                          &face_vert_indices, &face_weights_vec);
            if (face_target_xy.defined() && !face_weights_vec.empty())
            {
                auto face_weights = torch::from_blob(face_weights_vec.data(), {static_cast<int64_t>(face_weights_vec.size())}, torch::kFloat).clone();
                // Face vertices can still be normalized if needed, but safer to keep raw too
                // face_weights = face_weights / (face_weights.mean() + 1e-8f); 
                auto face_idx_tensor = torch::from_blob(face_vert_indices.data(), {static_cast<int64_t>(face_vert_indices.size())}, torch::kInt).to(torch::kLong);

                vd.face_target_xy = face_target_xy.to(device);
                vd.face_weights = face_weights.to(device);
                vd.face_idx_tensor = face_idx_tensor.to(device);
            }
        }

        valid_views.push_back(std::move(vd));
        v_idx++;
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
        std::make_unique<torch::optim::AdamOptions>(options.lr));

    std::vector<torch::Tensor> robust_params = {betas};
    robust_params.push_back(t_tensor);

    param_groups.emplace_back(
        robust_params,
        std::make_unique<torch::optim::AdamOptions>(options.lr * 0.1));

    torch::optim::Adam optimizer(param_groups, torch::optim::AdamOptions(options.lr));

    float prev_loss_val = std::numeric_limits<float>::infinity();
    const float sigma2 = options.reproj_robust_sigma * options.reproj_robust_sigma;

    for (int iter = 0; iter < options.num_iters; ++iter)
    {
        // Debug log cur pose every 50 iters
        if (iter % 50 == 0)
        {
            std::cout << "\n========== [SmplifyLiteMultiView] Iter " << iter << " ==========\n";
            std::cout << "--> SMPL Trans (Global):\n" << t_tensor.squeeze() << "\n";
            std::cout << "--> SMPL Betas (Shape):\n" << betas.squeeze() << "\n";
            std::cout << "--> SMPL Pose (Axis-Angle):\n" << pose.squeeze() << "\n";
            std::cout << "========================================================\n";
        }

        optimizer.zero_grad();

        auto smpl_out = smpl_layer.forward(betas, pose, t_tensor);
        auto joints = smpl_out.joints.squeeze(0);
        auto verts = smpl_out.vertices.squeeze(0);

        torch::Tensor total_data = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        torch::Tensor total_face_data = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat).device(device));

        int view_count = 0;
        int face_view_count = 0;

        bool do_render = (render_frames && render_dir && options.render_every > 0 &&
                          (iter % options.render_every == 0 || iter == options.num_iters - 1));

        for (const auto &view : valid_views)
        {
            cv::Mat canvas;
            if (do_render && view.original_view_idx < render_frames->size())
            {
                canvas = (*render_frames)[view.original_view_idx].clone();
            }

            // --- Body Loss ---
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

            // --- DEBUG LOGGING ---
            if (iter % 50 == 0)
            {
                std::cout << "\n  >>> View " << view_count << " Per-Joint Breakdown (Iter " << iter << ") <<<\n";
                auto debug_dist2_weighted = dist2 * view.weights;
                auto w_cpu = view.weights.to(torch::kCPU).contiguous();
                auto l_cpu = debug_dist2_weighted.to(torch::kCPU).contiguous();
                auto idx_cpu = view.idx_tensor.to(torch::kCPU).toType(torch::kInt64).contiguous();
                auto halpe_cpu = view.halpe_idx_tensor.to(torch::kCPU).toType(torch::kInt64).contiguous();

                auto w_acc = w_cpu.accessor<float, 1>();
                auto l_acc = l_cpu.accessor<float, 1>();
                auto idx_acc = idx_cpu.accessor<int64_t, 1>();
                auto halpe_acc = halpe_cpu.accessor<int64_t, 1>();

                for (int k = 0; k < w_acc.size(0); ++k)
                {
                    std::cout << "    Joint " << std::setw(2) << idx_acc[k]
                              << " | Halpe ID: " << halpe_acc[k]
                              << " | Conf: " << std::fixed << std::setprecision(4) << w_acc[k]
                              << " | Loss: " << l_acc[k] << "\n";
                }
            }
            // ---------------------

            ++view_count;

            if (do_render && !canvas.empty())
            {
                auto proj_cpu = proj.detach().cpu();
                auto proj_acc = proj_cpu.accessor<float, 2>();
                auto target_cpu = view.target_xy.detach().cpu();
                auto target_acc = target_cpu.accessor<float, 2>();
                for (int k = 0; k < proj_acc.size(0); ++k)
                {
                    cv::Point2f p_est(proj_acc[k][0], proj_acc[k][1]);
                    cv::Point2f p_tgt(target_acc[k][0], target_acc[k][1]);
                    cv::line(canvas, p_est, p_tgt, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                    cv::circle(canvas, p_tgt, 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                    cv::circle(canvas, p_est, 4, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                }
            }

            // --- Face Loss ---
            if (options.face_weight > 0.0f && view.face_target_xy.defined())
            {
                auto face_selected = verts.index_select(0, view.face_idx_tensor);
                auto Xf_c = torch::matmul(face_selected, view.R.t()) + view.t;
                auto Xf = Xf_c.index({torch::indexing::Slice(), 0});
                auto Yf = Xf_c.index({torch::indexing::Slice(), 1});
                auto Zf = Xf_c.index({torch::indexing::Slice(), 2});

                auto uf = (view.fx * Xf / (Zf + 1e-9f)) + view.cx;
                auto vf = (view.fy * Yf / (Zf + 1e-9f)) + view.cy;
                auto face_proj = torch::stack({uf, vf}, 1);

                auto face_diff = face_proj - view.face_target_xy;
                auto face_dist2 = (face_diff * face_diff).sum(1);
                if (sigma2 > 0.0f)
                {
                    face_dist2 = face_dist2 / (face_dist2 + sigma2);
                }
                auto face_loss = (face_dist2 * view.face_weights).mean();
                total_face_data = total_face_data + face_loss;
                face_view_count++;

                // Render Face points
                if (do_render && !canvas.empty())
                {
                    auto proj_cpu = face_proj.detach().cpu();
                    auto proj_acc = proj_cpu.accessor<float, 2>();
                    auto target_cpu = view.face_target_xy.detach().cpu();
                    auto target_acc = target_cpu.accessor<float, 2>();
                    for (int k = 0; k < proj_acc.size(0); ++k)
                    {
                        cv::Point2f p_est(proj_acc[k][0], proj_acc[k][1]);
                        cv::Point2f p_tgt(target_acc[k][0], target_acc[k][1]);
                        cv::line(canvas, p_est, p_tgt, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
                        cv::circle(canvas, p_tgt, 3, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
                        cv::circle(canvas, p_est, 3, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
                    }
                }
            }

            if (do_render && !canvas.empty())
            {
                char filename[128];
                snprintf(filename, sizeof(filename), "opt_f%05d_iter%03d_v%d.jpg", frame_idx, iter, view.original_view_idx);
                std::filesystem::path out_path = std::filesystem::path(*render_dir) / filename;
                cv::imwrite(out_path.string(), canvas);
            }
        }

        if (view_count > 0)
            total_data = total_data / static_cast<float>(view_count);
        if (face_view_count > 0)
            total_face_data = total_face_data / static_cast<float>(face_view_count);

        auto pose_reg = (pose - pose_init).pow(2).mean() * options.pose_reg;
        auto betas_reg = (betas - betas_init).pow(2).mean() * options.betas_reg;

        auto total = total_data + (total_face_data * options.face_weight) + pose_reg + betas_reg;
        const float curr_loss_val = total.item<float>();

        if (iter % options.log_every == 0 || iter == options.num_iters - 1)
        {
            std::cout << "  Iter " << std::setw(3) << iter
                      << " | Loss: " << std::fixed << std::setprecision(4) << curr_loss_val
                      << "  [Body: " << total_data.item<float>()
                      << ", Face: " << (total_face_data * options.face_weight).item<float>()
                      << ", Reg: " << (pose_reg + betas_reg).item<float>() << "]\n";
        }

        if (iter >= options.min_iters && std::abs(prev_loss_val - curr_loss_val) < options.loss_tol)
        {
            std::cout << "  Converged at iteration " << iter << "\n";
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
        *out_y_sign = 1.0f;
    std::cout << "[SmplifyLiteMultiView] Done." << std::endl;
    return true;
}

