#include "TextureOptimizer.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "HmrMathHelpers.h"
#include "SmplLBS.h"

namespace {

torch::Tensor ImagesToTensor(const std::vector<cv::Mat>& images, torch::Device device) {
    if (images.empty()) return torch::Tensor();

    std::vector<torch::Tensor> batch_tensors;
    batch_tensors.reserve(images.size());

    for (const auto& img_bgr : images) {
        cv::Mat float_img;
        img_bgr.convertTo(float_img, CV_32F, 1.0 / 255.0);
        auto tensor = torch::from_blob(float_img.data, {img_bgr.rows, img_bgr.cols, 3}, torch::kFloat).clone();
        tensor = tensor.permute({2, 0, 1});
        batch_tensors.push_back(tensor);
    }

    return torch::stack(batch_tensors).to(device);
}

torch::Tensor ComputeTranslationDiff(const torch::Tensor& cam_params,
                                     const torch::Tensor& crop_info,
                                     float focal, float img_w, float img_h) {
    auto s = cam_params.index({"...", 0});
    auto tx = cam_params.index({"...", 1});
    auto ty = cam_params.index({"...", 2});

    auto cx = crop_info.index({"...", 0});
    auto cy = crop_info.index({"...", 1});
    auto sz = crop_info.index({"...", 2});

    auto tz_3d = (2.0f * focal) / (sz * s + 1e-8f);
    auto tx_screen = (cx - img_w * 0.5f) + (tx * sz * s * 0.5f);
    auto ty_screen = (cy - img_h * 0.5f) + (ty * sz * s * 0.5f);
    auto tx_3d = tx_screen * tz_3d / focal;
    auto ty_3d = ty_screen * tz_3d / focal;

    return torch::stack({tx_3d, ty_3d, tz_3d}, 1);
}

torch::Tensor ProjectVerticesDiff(const torch::Tensor& vertices,
                                  const torch::Tensor& trans,
                                  float focal, int W, int H) {
    auto X = vertices.index({"...", 0}) + trans.index({"...", 0}).unsqueeze(1);
    auto Y = vertices.index({"...", 1}) + trans.index({"...", 1}).unsqueeze(1);
    auto Z = vertices.index({"...", 2}) + trans.index({"...", 2}).unsqueeze(1);

    auto u = (focal * X) / (Z + 1e-6f) + static_cast<float>(W) * 0.5f;
    auto v = (focal * Y) / (Z + 1e-6f) + static_cast<float>(H) * 0.5f;

    auto u_norm = 2.0f * (u / static_cast<float>(W)) - 1.0f;
    auto v_norm = 2.0f * (v / static_cast<float>(H)) - 1.0f;
    return torch::stack({u_norm, v_norm}, 2);
}

std::vector<float> PoseAxisAngleTo6d(const torch::Tensor& pose_axis_angle_cpu) {
    const int batch = static_cast<int>(pose_axis_angle_cpu.size(0));
    auto rot_mats = batch_rodrigues(pose_axis_angle_cpu); // [B, 24, 3, 3]

    auto c1 = rot_mats.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto c2 = rot_mats.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1});

    auto c1x = c1.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto c1y = c1.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    auto c1z = c1.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});

    auto c2x = c2.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto c2y = c2.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    auto c2z = c2.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});

    auto six_d = torch::stack({c1x, c2x, c1y, c2y, c1z, c2z}, -1); // [B, 24, 6]
    auto flat = six_d.reshape({batch, -1}).contiguous();

    std::vector<float> out(static_cast<size_t>(flat.numel()));
    std::memcpy(out.data(), flat.data_ptr<float>(), out.size() * sizeof(float));
    return out;
}

torch::Tensor LoadExtraData(const std::string& path) {
    if (!std::filesystem::exists(path)) return torch::Tensor();
    try {
        std::ifstream input(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
        input.close();
        auto data = torch::pickle_load(bytes);
        if (data.isTensor()) return data.toTensor();
    } catch (...) {
    }
    return torch::Tensor();
}

struct BarycentricCoords {
    float u;
    float v;
    float w;
    bool valid;
};

BarycentricCoords ComputeBarycentric(float px, float py,
                                     float ax, float ay,
                                     float bx, float by,
                                     float cx, float cy) {
    float v0x = bx - ax, v0y = by - ay;
    float v1x = cx - ax, v1y = cy - ay;
    float v2x = px - ax, v2y = py - ay;
    float d00 = v0x * v0x + v0y * v0y;
    float d01 = v0x * v1x + v0y * v1y;
    float d11 = v1x * v1x + v1y * v1y;
    float d20 = v2x * v0x + v2y * v0y;
    float d21 = v2x * v1x + v2y * v1y;
    float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-6f) return {0.0f, 0.0f, 0.0f, false};
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    return {u, v, w, (v >= 0.0f) && (w >= 0.0f) && (u >= 0.0f)};
}

cv::Mat RasterizeUV(const torch::Tensor& vertex_values,
                    const torch::Tensor& faces,
                    const torch::Tensor& uvs,
                    int size) {
    cv::Mat map = cv::Mat::zeros(size, size, CV_32FC3);
    if (!faces.defined() || !uvs.defined()) return map;
    if (vertex_values.dim() != 2 || vertex_values.size(0) == 0) return map;

    auto faces_acc = faces.accessor<int64_t, 2>();
    auto uvs_acc = uvs.accessor<float, 2>();
    auto vals_cpu = vertex_values.cpu();
    auto val_acc = vals_cpu.accessor<float, 2>();

    const int num_faces = static_cast<int>(faces.size(0));
    const bool is_scalar = (vertex_values.size(1) == 1);

    for (int f = 0; f < num_faces; ++f) {
        int idx0 = static_cast<int>(faces_acc[f][0]);
        int idx1 = static_cast<int>(faces_acc[f][1]);
        int idx2 = static_cast<int>(faces_acc[f][2]);

        float u0 = uvs_acc[idx0][0] * size;
        float v0 = (1.0f - uvs_acc[idx0][1]) * size;
        float u1 = uvs_acc[idx1][0] * size;
        float v1 = (1.0f - uvs_acc[idx1][1]) * size;
        float u2 = uvs_acc[idx2][0] * size;
        float v2 = (1.0f - uvs_acc[idx2][1]) * size;

        int min_x = std::max(0, static_cast<int>(std::floor(std::min({u0, u1, u2}))));
        int max_x = std::min(size - 1, static_cast<int>(std::ceil(std::max({u0, u1, u2}))));
        int min_y = std::max(0, static_cast<int>(std::floor(std::min({v0, v1, v2}))));
        int max_y = std::min(size - 1, static_cast<int>(std::ceil(std::max({v0, v1, v2}))));

        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                auto b = ComputeBarycentric(x + 0.5f, y + 0.5f, u0, v0, u1, v1, u2, v2);
                if (!b.valid) continue;

                float val0_r = val_acc[idx0][0];
                float val1_r = val_acc[idx1][0];
                float val2_r = val_acc[idx2][0];
                float r = val0_r * b.u + val1_r * b.v + val2_r * b.w;

                float g = r;
                float b_val = r;
                if (!is_scalar) {
                    float val0_g = val_acc[idx0][1];
                    float val0_b = val_acc[idx0][2];
                    float val1_g = val_acc[idx1][1];
                    float val1_b = val_acc[idx1][2];
                    float val2_g = val_acc[idx2][1];
                    float val2_b = val_acc[idx2][2];
                    g = val0_g * b.u + val1_g * b.v + val2_g * b.w;
                    b_val = val0_b * b.u + val1_b * b.v + val2_b * b.w;
                }

                map.at<cv::Vec3f>(y, x) = cv::Vec3f(b_val, g, r);
            }
        }
    }

    return map;
}

} // namespace

bool OptimizeTextureConsistency(SMPLLayer& smpl_layer,
                                const std::vector<cv::Mat>& frames,
                                std::vector<SmplResult>* results,
                                const std::vector<float>& crop_info_flat,
                                float focal_length,
                                const HmrOutputOptions& opts) {
    if (!results || frames.empty() || results->empty()) return false;
    if (frames.size() != results->size()) return false;
    if (crop_info_flat.size() != frames.size() * 3) return false;

    const auto device = smpl_layer.v_template.device();
    const int num_frames = static_cast<int>(frames.size());
    const int W = frames[0].cols;
    const int H = frames[0].rows;

    if (W <= 0 || H <= 0) return false;

    static torch::Tensor faces_tensor;
    static torch::Tensor uvs_tensor;
    static bool uv_loaded = false;
    static bool uv_warned = false;

    if (!uv_loaded) {
        std::filesystem::path model_dir = std::filesystem::path(opts.smpl_model_path).parent_path();
        faces_tensor = LoadExtraData((model_dir / "smpl_faces.pt").string());
        if (faces_tensor.defined()) {
            faces_tensor = faces_tensor.to(torch::kCPU).to(torch::kLong);
        }
        uvs_tensor = LoadExtraData((model_dir / "smpl_uv.pt").string());
        if (uvs_tensor.defined()) {
            uvs_tensor = uvs_tensor.to(torch::kCPU).to(torch::kFloat);
        }
        uv_loaded = true;
    }

    // Create debug output directory if saving is enabled
    std::filesystem::path debug_dir;
    if (opts.save_outputs && !opts.output_dir.empty()) {
        debug_dir = std::filesystem::path(opts.output_dir) / "texture_opt_debug";
        std::filesystem::create_directories(debug_dir);
    }

    int step = opts.texture_batch - 5;
    if (step < 1) step = 1;

    std::cout << "[TextureOpt] Optimizing " << num_frames << " frames..." << std::endl;

    for (int start = 0; start < num_frames; start += step) {
        int end = std::min(start + opts.texture_batch, num_frames);
        int batch_size = end - start;
        if (batch_size < 2) continue;

        std::vector<cv::Mat> batch_imgs;
        std::vector<float> pose_vec;
        std::vector<float> cam_vec;
        std::vector<float> crop_vec;
        std::vector<float> shape_vec;
        batch_imgs.reserve(batch_size);

        const int pose_size = static_cast<int>((*results)[start].pose.size());
        if (!(pose_size == 72 || pose_size == 144)) {
            continue;
        }

        shape_vec = (*results)[start].shape;
        
        for (int i = start; i < end; ++i) {
            const auto& res = (*results)[i];
            batch_imgs.push_back(frames[i]);

            if (pose_size == 72) {
                pose_vec.insert(pose_vec.end(), res.pose.begin(), res.pose.end());
            } else {
                const auto pose_aa = ConvertPose6dToAxisAngle(res.pose);
                pose_vec.insert(pose_vec.end(), pose_aa.begin(), pose_aa.end());
            }

            cam_vec.insert(cam_vec.end(), res.camera.begin(), res.camera.end());
            crop_vec.push_back(crop_info_flat[i * 3 + 0]);
            crop_vec.push_back(crop_info_flat[i * 3 + 1]);
            crop_vec.push_back(crop_info_flat[i * 3 + 2]);
        }
        if (batch_imgs.empty()) continue;

        auto img_tensor = ImagesToTensor(batch_imgs, device);
        auto pose_tensor = torch::from_blob(pose_vec.data(), {batch_size, 24, 3}, torch::kFloat).to(device).clone();
        auto cam_tensor = torch::from_blob(cam_vec.data(), {batch_size, 3}, torch::kFloat).to(device).clone();
        auto shape_tensor = torch::from_blob(shape_vec.data(), {1, 10}, torch::kFloat).to(device).clone();
        auto crop_tensor = torch::from_blob(crop_vec.data(), {batch_size, 3}, torch::kFloat).to(device).clone();

        pose_tensor.set_requires_grad(true);
        cam_tensor.set_requires_grad(true);

        auto pose_init = pose_tensor.detach().clone();
        auto cam_init = cam_tensor.detach().clone();

        torch::optim::Adam optimizer({pose_tensor, cam_tensor}, torch::optim::AdamOptions(0.01));

        for (int iter = 0; iter < opts.texture_iters; ++iter) {
            optimizer.zero_grad();

            auto batch_shape = shape_tensor.expand({batch_size, 10});
            auto zero_trans = torch::zeros({batch_size, 3}, device);
            auto smpl_out = smpl_layer.forward(batch_shape, pose_tensor, zero_trans);

            auto T = ComputeTranslationDiff(cam_tensor, crop_tensor, focal_length,
                                            static_cast<float>(W), static_cast<float>(H));
            auto grid = ProjectVerticesDiff(smpl_out.vertices, T, focal_length, W, H);

            auto grid_expanded = grid.unsqueeze(1); // [B, 1, V, 2]
            auto sampled = torch::nn::functional::grid_sample(
                img_tensor, grid_expanded,
                torch::nn::functional::GridSampleFuncOptions()
                    .padding_mode(torch::kBorder)
                    .align_corners(false));
            auto colors = sampled.squeeze(2); // [B, 3, V]

            // Calculate Mean & Variance for Optimization AND Visualization
            auto mean_color = colors.mean(0); // [3, V]
            auto diff = colors - mean_color.unsqueeze(0);
            auto sq_err = diff.pow(2).sum(1); // [B, V] sum over RGB
            auto variance_per_vert = sq_err.mean(0); // [V] mean over Batch

            // Robust Loss
            const float k = 0.1f;
            auto robust = sq_err / (sq_err + k * k);
            auto loss_tex = robust.mean();

            auto loss_reg = (pose_tensor - pose_init).pow(2).mean() +
                            (cam_tensor - cam_init).pow(2).mean();

            auto loss = loss_tex + 0.01f * loss_reg;
            loss.backward();
            optimizer.step();

            // --- UV Map Saving (every iteration) ---
            if (opts.save_outputs && !debug_dir.empty()) {
                if (!faces_tensor.defined() || !uvs_tensor.defined()) {
                    if (!uv_warned) {
                        std::cout << "[TextureOpt] UV export skipped: missing smpl_faces.pt or smpl_uv.pt." << std::endl;
                        uv_warned = true;
                    }
                } else {
                    const int uv_size = 1024;
                    std::string batch_prefix = "batch_" + std::to_string(start) + "_epoch_" + std::to_string(iter);

                    auto mean_vis = mean_color.permute({1, 0}).detach().clamp(0.0f, 1.0f); // [V, 3]
                    cv::Mat map_mean = RasterizeUV(mean_vis, faces_tensor, uvs_tensor, uv_size);
                    cv::Mat map_mean_u8;
                    map_mean.convertTo(map_mean_u8, CV_8UC3, 255.0);
                    cv::imwrite((debug_dir / (batch_prefix + "_mean.png")).string(), map_mean_u8);

                    auto var_min = variance_per_vert.min();
                    auto var_max = variance_per_vert.max();
                    auto var_norm = (variance_per_vert - var_min) / (var_max - var_min + 1e-8f);
                    auto var_vis = var_norm.unsqueeze(1).detach().clamp(0.0f, 1.0f); // [V, 1]
                    cv::Mat map_var = RasterizeUV(var_vis, faces_tensor, uvs_tensor, uv_size);
                    cv::Mat map_var_u8;
                    map_var.convertTo(map_var_u8, CV_8UC3, 255.0);
                    cv::imwrite((debug_dir / (batch_prefix + "_variance.png")).string(), map_var_u8);
                }
            }
        }

        auto pose_cpu = pose_tensor.detach().cpu();
        auto cam_cpu = cam_tensor.detach().cpu();

        if (pose_size == 72) {
            auto p_acc = pose_cpu.accessor<float, 3>();
            auto c_acc = cam_cpu.accessor<float, 2>();
            for (int i = 0; i < batch_size; ++i) {
                int global_idx = start + i;
                std::vector<float> new_pose;
                new_pose.reserve(72);
                for (int j = 0; j < 24; ++j) {
                    new_pose.push_back(p_acc[i][j][0]);
                    new_pose.push_back(p_acc[i][j][1]);
                    new_pose.push_back(p_acc[i][j][2]);
                }
                (*results)[global_idx].pose = std::move(new_pose);
                (*results)[global_idx].camera = {c_acc[i][0], c_acc[i][1], c_acc[i][2]};
            }
        } else {
            auto c_acc = cam_cpu.accessor<float, 2>();
            auto pose_6d = PoseAxisAngleTo6d(pose_cpu);
            for (int i = 0; i < batch_size; ++i) {
                int global_idx = start + i;
                const int offset = i * 144;
                (*results)[global_idx].pose.assign(pose_6d.begin() + offset, pose_6d.begin() + offset + 144);
                (*results)[global_idx].camera = {c_acc[i][0], c_acc[i][1], c_acc[i][2]};
            }
        }
    }

    return true;
} 