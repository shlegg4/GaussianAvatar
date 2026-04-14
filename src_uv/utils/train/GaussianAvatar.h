#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "utils/SmplxLBS.h"
#include "utils/render/GaussianUtils.h"
#include "utils/render/RenderMathUtils.h"
#include "utils/train/ViewerExport.h"

struct GaussianAvatar : torch::nn::Module
{
    static constexpr int kUvSize = 512;

    std::shared_ptr<SMPLXLayer> smplx;
    torch::Tensor g_scales, g_rots, g_opacities, g_colors, g_offsets, g_sh;
    torch::Tensor g_bary_coords, g_face_indices, faces_buffer;
    torch::Tensor valid_uv_mask, valid_uv_indices;
    torch::Tensor current_flat_offsets, current_flat_scales, current_flat_rots;
    torch::Tensor current_flat_opacities, current_flat_colors, current_flat_sh;
    torch::Tensor v_template_cached, faces_cached;
    int sh_degree_ = 0;

    explicit GaussianAvatar(const std::string &model_path)
    {
        smplx = std::make_shared<SMPLXLayer>(model_path);
        register_module("smplx", smplx);
    }

    void init_uv_mapping(const std::string &smpl_uv_obj_path)
    {
        auto smpl_face_uvs = LoadSmplUVsFromOBJ(smpl_uv_obj_path);
        if (!smpl_face_uvs.defined() || smpl_face_uvs.dim() != 3 || smpl_face_uvs.size(1) != 3 || smpl_face_uvs.size(2) != 2)
        {
            throw std::runtime_error("Failed to load SMPL UV mapping from: " + smpl_uv_obj_path);
        }

        auto uvs_cpu = smpl_face_uvs.to(torch::kCPU).to(torch::kFloat32).contiguous();
        const float *uv_ptr = uvs_cpu.data_ptr<float>();

        const int64_t uv_face_count = uvs_cpu.size(0);
        const int64_t mesh_face_count = faces_cached.size(0);
        const int64_t face_count = std::min(uv_face_count, mesh_face_count);
        if (face_count <= 0)
        {
            throw std::runtime_error("UV mapping has no faces.");
        }

        std::vector<uint8_t> mask_values(static_cast<size_t>(kUvSize) * static_cast<size_t>(kUvSize), 0u);
        std::vector<int64_t> flat_indices;
        std::vector<int64_t> face_ids;
        std::vector<float> bary_values;
        flat_indices.reserve(mask_values.size() / 2);
        face_ids.reserve(mask_values.size() / 2);
        bary_values.reserve((mask_values.size() / 2) * 3u);

        auto edge_fn = [](float ax, float ay, float bx, float by, float px, float py)
        {
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
        };

        constexpr float kInsideEps = 1e-6f;
        for (int64_t face_idx = 0; face_idx < face_count; ++face_idx)
        {
            const float *tri = uv_ptr + (face_idx * 3 * 2);
            const float x0 = tri[0] * static_cast<float>(kUvSize - 1);
            const float y0 = (1.0f - tri[1]) * static_cast<float>(kUvSize - 1);
            const float x1 = tri[2] * static_cast<float>(kUvSize - 1);
            const float y1 = (1.0f - tri[3]) * static_cast<float>(kUvSize - 1);
            const float x2 = tri[4] * static_cast<float>(kUvSize - 1);
            const float y2 = (1.0f - tri[5]) * static_cast<float>(kUvSize - 1);

            const float denom = edge_fn(x0, y0, x1, y1, x2, y2);
            if (std::abs(denom) < 1e-8f)
            {
                continue;
            }

            const int min_x = std::clamp(static_cast<int>(std::floor(std::min({x0, x1, x2}))), 0, kUvSize - 1);
            const int max_x = std::clamp(static_cast<int>(std::ceil(std::max({x0, x1, x2}))), 0, kUvSize - 1);
            const int min_y = std::clamp(static_cast<int>(std::floor(std::min({y0, y1, y2}))), 0, kUvSize - 1);
            const int max_y = std::clamp(static_cast<int>(std::ceil(std::max({y0, y1, y2}))), 0, kUvSize - 1);

            for (int py = min_y; py <= max_y; ++py)
            {
                for (int px = min_x; px <= max_x; ++px)
                {
                    const float sx = static_cast<float>(px) + 0.5f;
                    const float sy = static_cast<float>(py) + 0.5f;

                    const float w0 = edge_fn(x1, y1, x2, y2, sx, sy) / denom;
                    const float w1 = edge_fn(x2, y2, x0, y0, sx, sy) / denom;
                    const float w2 = 1.0f - w0 - w1;

                    if (w0 < -kInsideEps || w1 < -kInsideEps || w2 < -kInsideEps)
                    {
                        continue;
                    }

                    const int64_t linear_idx = static_cast<int64_t>(py) * static_cast<int64_t>(kUvSize) + static_cast<int64_t>(px);
                    if (mask_values[static_cast<size_t>(linear_idx)] != 0u)
                    {
                        continue;
                    }

                    mask_values[static_cast<size_t>(linear_idx)] = 1u;
                    flat_indices.push_back(linear_idx);
                    face_ids.push_back(face_idx);
                    bary_values.push_back(w0);
                    bary_values.push_back(w1);
                    bary_values.push_back(w2);
                }
            }
        }

        if (face_ids.empty())
        {
            throw std::runtime_error("UV mapping rasterization produced zero valid UV pixels.");
        }

        auto device = smplx->v_template.device();
        valid_uv_mask = torch::from_blob(mask_values.data(), {kUvSize, kUvSize}, torch::TensorOptions().dtype(torch::kUInt8))
                            .clone()
                            .to(device)
                            .to(torch::kBool);
        valid_uv_indices = torch::from_blob(flat_indices.data(), {static_cast<int64_t>(flat_indices.size())}, torch::TensorOptions().dtype(torch::kLong))
                               .clone()
                               .to(device);
        g_face_indices = torch::from_blob(face_ids.data(), {static_cast<int64_t>(face_ids.size())}, torch::TensorOptions().dtype(torch::kLong))
                             .clone()
                             .to(device);
        g_bary_coords = torch::from_blob(bary_values.data(), {static_cast<int64_t>(face_ids.size()), 3}, torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(device);

        register_buffer("valid_uv_mask", valid_uv_mask);
        register_buffer("valid_uv_indices", valid_uv_indices);
        register_buffer("g_face_indices", g_face_indices);
        register_buffer("g_bary_coords", g_bary_coords);

        std::cout << "UV mapping initialized with " << face_ids.size()
                  << " valid UV pixels at " << kUvSize << "x" << kUvSize << "." << std::endl;
    }

    void init_gaussians(torch::Tensor faces_idx,
                        int sh_degree,
                        const std::string &smpl_uv_obj_path)
    {
        auto device = smplx->v_template.device();
        sh_degree_ = std::max(0, sh_degree);
        v_template_cached = smplx->v_template.clone().detach();
        faces_cached = faces_idx.clone().detach();
        faces_buffer = faces_cached.to(device);

        init_uv_mapping(smpl_uv_obj_path);

        auto uv_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        g_offsets = torch::zeros({3, kUvSize, kUvSize}, uv_opts).set_requires_grad(true);
        g_scales = torch::full({3, kUvSize, kUvSize}, std::log(0.002f), uv_opts).set_requires_grad(true);
        g_rots = torch::zeros({4, kUvSize, kUvSize}, uv_opts).set_requires_grad(true);
        g_opacities = torch::full({1, kUvSize, kUvSize}, 0.95f, uv_opts).set_requires_grad(true);
        g_colors = torch::full({3, kUvSize, kUvSize}, 0.5f, uv_opts).set_requires_grad(true);
        {
            torch::NoGradGuard no_grad;
            g_rots.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0f);
        }

        if (sh_degree > 0)
        {
            const int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
            g_sh = torch::zeros({sh_coeffs * 3, kUvSize, kUvSize}, uv_opts).set_requires_grad(true);
        }
        else
        {
            g_sh = torch::zeros({0}, torch::TensorOptions().device(device));
        }

        register_parameter("g_scales", g_scales);
        register_parameter("g_rots", g_rots);
        register_parameter("g_opacities", g_opacities);
        register_parameter("g_colors", g_colors);
        register_parameter("g_offsets", g_offsets);
        register_parameter("g_sh", g_sh);
        register_buffer("v_template_cached", v_template_cached);
        register_buffer("faces_cached", faces_cached);
    }

    int64_t NumGaussians() const
    {
        return g_face_indices.defined() ? g_face_indices.size(0) : 0;
    }

    void UpdateFlatParams(int sh_degree)
    {
        if (!valid_uv_indices.defined())
        {
            return;
        }

        current_flat_offsets = g_offsets.view({3, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_scales = g_scales.view({3, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_rots = g_rots.view({4, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_opacities = g_opacities.view({1, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_colors = g_colors.view({3, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();

        if (sh_degree > 0 && g_sh.defined() && g_sh.numel() > 0)
        {
            const int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
            current_flat_sh = g_sh.view({sh_coeffs, 3, -1})
                                  .index_select(2, valid_uv_indices)
                                  .permute({2, 0, 1})
                                  .contiguous();
        }
        else
        {
            current_flat_sh = torch::zeros({0}, g_colors.options());
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> get_bone_data()
    {
        auto device = g_bary_coords.device();

        auto v_weights = smplx->weights.to(device);

        auto selected_faces = faces_buffer.index_select(0, g_face_indices);

        auto v0 = selected_faces.select(1, 0);
        auto v1 = selected_faces.select(1, 1);
        auto v2 = selected_faces.select(1, 2);

        auto w0 = v_weights.index_select(0, v0);
        auto w1 = v_weights.index_select(0, v1);
        auto w2 = v_weights.index_select(0, v2);

        auto u = g_bary_coords.select(1, 0).unsqueeze(1);
        auto v = g_bary_coords.select(1, 1).unsqueeze(1);
        auto w = g_bary_coords.select(1, 2).unsqueeze(1);

        auto splat_weights = u * w0 + v * w1 + w * w2;

        auto topk = torch::topk(splat_weights, 4, 1);
        auto top_weights = std::get<0>(topk);
        auto top_indices = std::get<1>(topk).to(torch::kFloat32);

        auto sum_weights = top_weights.sum(1, true);
        top_weights = top_weights / torch::clamp_min(sum_weights, 1e-6f);

        return {top_indices, top_weights};
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor betas,
                                                     torch::Tensor expression,
                                                     torch::Tensor pose,
                                                     torch::Tensor jaw_pose,
                                                     torch::Tensor eye_pose,
                                                     torch::Tensor left_hand_pose,
                                                     torch::Tensor right_hand_pose,
                                                     torch::Tensor trans)
    {
        using torch::indexing::Slice;

        auto smplx_out = smplx->forward(betas, expression, pose, jaw_pose, eye_pose, left_hand_pose, right_hand_pose, trans);
        auto verts_posed = smplx_out.vertices;
        const int64_t batch_count = verts_posed.size(0);
        const int64_t gaussian_count = g_face_indices.size(0);

        UpdateFlatParams(sh_degree_);

        auto selected_faces = faces_buffer.index_select(0, g_face_indices);
        auto face_i0 = selected_faces.index({Slice(), 0});
        auto face_i1 = selected_faces.index({Slice(), 1});
        auto face_i2 = selected_faces.index({Slice(), 2});

        auto A = verts_posed.index_select(1, face_i0);
        auto B = verts_posed.index_select(1, face_i1);
        auto C = verts_posed.index_select(1, face_i2);

        auto verts_canon = v_template_cached.to(verts_posed.device());
        auto A_can = verts_canon.index_select(0, face_i0);
        auto B_can = verts_canon.index_select(0, face_i1);
        auto C_can = verts_canon.index_select(0, face_i2);

        auto R_canon = ComputeTriFrames(A_can, B_can, C_can);
        auto X = torch::nn::functional::normalize(B - A, torch::nn::functional::NormalizeFuncOptions().dim(2));
        auto N = torch::cross(B - A, C - A, 2);
        N = torch::nn::functional::normalize(N, torch::nn::functional::NormalizeFuncOptions().dim(2));
        auto Y = torch::cross(N, X, 2);
        auto R_posed = torch::stack({X, Y, N}, 3);

        auto R_posed_flat = R_posed.reshape({-1, 3, 3});
        auto offset_x = current_flat_offsets.index({Slice(), 0}).view({1, gaussian_count, 1});
        auto offset_y = current_flat_offsets.index({Slice(), 1}).view({1, gaussian_count, 1});
        auto offset_z = current_flat_offsets.index({Slice(), 2}).view({1, gaussian_count, 1});
        auto posed_offsets = (X * offset_x) + (Y * offset_y) + (N * offset_z);

        auto R_canon_batch = R_canon.unsqueeze(0).expand({batch_count, gaussian_count, 3, 3});
        auto R_canon_flat = R_canon_batch.reshape({-1, 3, 3});
        auto R_skin_flat = torch::bmm(R_posed_flat, R_canon_flat.transpose(1, 2));

        auto R_skin = R_skin_flat.view({batch_count, gaussian_count, 3, 3});

        auto u = g_bary_coords.index({Slice(), 0}).view({1, gaussian_count, 1});
        auto v = g_bary_coords.index({Slice(), 1}).view({1, gaussian_count, 1});
        auto w = g_bary_coords.index({Slice(), 2}).view({1, gaussian_count, 1});

        auto skinned_pos = u * A + v * B + w * C;
        auto final_pos = skinned_pos + posed_offsets;

        auto q_skin_flat = MatrixToQuat(R_skin_flat);
        auto base_rots = current_flat_rots.unsqueeze(0).expand({batch_count, gaussian_count, 4}).reshape({-1, 4});
        auto final_rot = QuatMultiply(q_skin_flat, base_rots).view({batch_count, gaussian_count, 4});

        if (batch_count == 1)
        {
            return {final_pos.squeeze(0), final_rot.squeeze(0)};
        }

        return {final_pos, final_rot};
    }
};
