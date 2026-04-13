#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/SmplLBS.h"

struct SmplxOutput
{
    torch::Tensor vertices;          // [B, 10475, 3]
    torch::Tensor joints;            // [B, 55, 3]
    torch::Tensor transforms;        // [B, 55, 4, 4]
    torch::Tensor vertex_transforms; // [B, 10475, 4, 4]
};

struct SMPLXLayer : torch::nn::Module
{
    torch::Tensor v_template;  // [10475, 3]
    torch::Tensor shapedirs;   // [10475, 3, 10]
    torch::Tensor exprdirs;    // [10475, 3, E]
    torch::Tensor blenddirs;   // [10475, 3, 10 + E]
    torch::Tensor posedirs;    // [10475, 3, 486]
    torch::Tensor J_regressor; // [55, 10475]
    torch::Tensor parents;     // [55]
    torch::Tensor weights;     // [10475, 55]
    torch::Tensor faces;       // [F, 3]
    torch::Tensor parents_cpu;
    std::vector<int64_t> parents_host;

    SMPLXLayer(const std::string &model_path)
    {
        std::cout << "Loading SMPL-X parameters from " << model_path << "..." << std::endl;

        c10::IValue data;
        try
        {
            std::ifstream input(model_path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("Could not open file: " + model_path +
                                         " (cwd: " + std::filesystem::current_path().string() + ")");
            }

            std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                    (std::istreambuf_iterator<char>()));
            input.close();
            data = torch::pickle_load(bytes);
        }
        catch (const c10::Error &e)
        {
            throw std::runtime_error("Failed to load SMPL-X model from '" + model_path + "': " + e.msg());
        }

        if (!data.isGenericDict())
        {
            throw std::runtime_error("SMPL-X file '" + model_path + "' does not contain a tensor dictionary.");
        }

        auto dict = data.toGenericDict();
        auto get_tensor = [&](const char *key) -> torch::Tensor
        {
            const c10::IValue name{std::string(key)};
            if (!dict.contains(name))
            {
                throw std::runtime_error("Missing key in SMPL-X model: " + std::string(key));
            }
            return dict.at(name).toTensor().clone();
        };

        v_template = register_buffer("v_template", get_tensor("v_template").to(torch::kFloat32));
        shapedirs = register_buffer("shapedirs", get_tensor("shapedirs").to(torch::kFloat32));
        exprdirs = register_buffer("exprdirs", get_tensor("exprdirs").to(torch::kFloat32));
        blenddirs = register_buffer("blenddirs", torch::cat({shapedirs, exprdirs}, 2).contiguous());
        posedirs = register_buffer("posedirs", get_tensor("posedirs").to(torch::kFloat32));
        J_regressor = register_buffer("J_regressor", get_tensor("J_regressor").to(torch::kFloat32));
        weights = register_buffer("weights", get_tensor("weights").to(torch::kFloat32));
        faces = register_buffer("faces", get_tensor("faces").to(torch::kLong));

        auto raw_parents = get_tensor("parents").to(torch::kLong).clone();
        if (raw_parents.numel() == 0)
        {
            throw std::runtime_error("SMPL-X parents array is empty.");
        }
        raw_parents[0] = 0;
        parents = register_buffer("parents", raw_parents);
        parents_cpu = raw_parents.to(torch::kCPU).to(torch::kLong).contiguous();
        parents_host.resize(static_cast<size_t>(parents_cpu.numel()));
        auto parents_acc = parents_cpu.accessor<int64_t, 1>();
        for (int64_t i = 0; i < parents_cpu.numel(); ++i)
        {
            parents_host[static_cast<size_t>(i)] = parents_acc[i];
        }

        std::cout << "SMPL-X model loaded successfully." << std::endl;
    }

    SmplxOutput forward(torch::Tensor betas,
                        torch::Tensor expression,
                        torch::Tensor pose_axis_angle,
                        torch::Tensor jaw_pose,
                        torch::Tensor eye_pose,
                        torch::Tensor left_hand_pose,
                        torch::Tensor right_hand_pose,
                        torch::Tensor trans)
    {
        const auto batch_size = betas.size(0);
        const auto device = betas.device();
        const int64_t joint_count = J_regressor.size(0);
        const int64_t expr_count = exprdirs.size(2);

        auto expand_param = [&](torch::Tensor value, int64_t width) -> torch::Tensor
        {
            if (!value.defined() || value.numel() == 0)
            {
                return torch::zeros({batch_size, width}, betas.options());
            }
            if (value.dim() == 1)
            {
                value = value.unsqueeze(0);
            }
            if (value.dim() != 2)
            {
                value = value.view({value.size(0), -1});
            }
            if (value.size(0) == 1 && batch_size > 1)
            {
                value = value.expand({batch_size, value.size(1)});
            }
            value = value.to(device);
            if (value.size(1) < width)
            {
                auto pad = torch::zeros({batch_size, width - value.size(1)}, value.options());
                value = torch::cat({value, pad}, 1);
            }
            else if (value.size(1) > width)
            {
                value = value.slice(1, 0, width);
            }
            return value.contiguous();
        };

        if (trans.dim() == 1)
        {
            trans = trans.unsqueeze(0);
        }
        if (trans.dim() == 2 && trans.size(0) == 1 && batch_size > 1)
        {
            trans = trans.expand({batch_size, trans.size(1)});
        }
        trans = trans.to(device);

        expression = expand_param(expression, expr_count);
        jaw_pose = expand_param(jaw_pose, 3);
        eye_pose = expand_param(eye_pose, 6);
        left_hand_pose = expand_param(left_hand_pose, 45);
        right_hand_pose = expand_param(right_hand_pose, 45);

        if (pose_axis_angle.dim() == 2)
        {
            pose_axis_angle = pose_axis_angle.view({batch_size, -1, 3});
        }
        else if (pose_axis_angle.dim() == 3 && pose_axis_angle.size(0) == 1 && batch_size > 1)
        {
            pose_axis_angle = pose_axis_angle.expand({batch_size, pose_axis_angle.size(1), pose_axis_angle.size(2)});
        }
        pose_axis_angle = pose_axis_angle.to(device).contiguous();

        auto global_pose = pose_axis_angle.index({torch::indexing::Slice(), 0}).contiguous();
        auto body_pose = pose_axis_angle.index({torch::indexing::Slice(), torch::indexing::Slice(1, 22)}).contiguous();
        auto pose_all = torch::cat({
                                      global_pose.view({batch_size, 3}),
                                      body_pose.view({batch_size, 63}),
                                      jaw_pose,
                                      eye_pose,
                                      left_hand_pose,
                                      right_hand_pose,
                                  },
                                  1)
                            .view({batch_size, joint_count, 3});

        auto betas_full = torch::cat({betas.to(device), expression}, 1);
        auto v_shaped = v_template.unsqueeze(0) + torch::einsum("bl,mkl->bmk", {betas_full, blenddirs});
        auto J = torch::einsum("jv, bvi -> bji", {J_regressor, v_shaped});

        auto rot_mats = batch_rodrigues(pose_all);
        auto pose_feature = (rot_mats.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) -
                             torch::eye(3, device))
                                .view({batch_size, -1});
        auto v_posed = v_shaped + torch::einsum("bl,mkl->bmk", {pose_feature, posedirs});

        std::vector<torch::Tensor> G_vec(static_cast<size_t>(joint_count));
        auto J_parents = J.index_select(1, parents);
        auto J_offsets = J - J_parents;

        for (int64_t i = 0; i < joint_count; ++i)
        {
            auto mat_rot = rot_mats.index({torch::indexing::Slice(), i});
            auto val_trans = (i == 0)
                                 ? J.index({torch::indexing::Slice(), 0})
                                 : J_offsets.index({torch::indexing::Slice(), i});

            auto local_transform = transform_mat(mat_rot, val_trans);
            if (i == 0)
            {
                G_vec[static_cast<size_t>(i)] = local_transform;
            }
            else
            {
                const int64_t parent_idx = parents_host[static_cast<size_t>(i)];
                if (parent_idx < 0 || parent_idx >= i)
                {
                    G_vec[static_cast<size_t>(i)] = local_transform;
                }
                else
                {
                    G_vec[static_cast<size_t>(i)] = torch::matmul(G_vec[static_cast<size_t>(parent_idx)], local_transform);
                }
            }
        }

        auto results_G = torch::stack(G_vec, 1);

        auto neg_J = -J;
        auto opts = neg_J.options();
        auto zeros = torch::zeros({batch_size, joint_count, 1}, opts);
        auto ones = torch::ones({batch_size, joint_count, 1}, opts);

        auto r0 = torch::cat({ones, zeros, zeros, neg_J.slice(2, 0, 1)}, 2).unsqueeze(2);
        auto r1 = torch::cat({zeros, ones, zeros, neg_J.slice(2, 1, 2)}, 2).unsqueeze(2);
        auto r2 = torch::cat({zeros, zeros, ones, neg_J.slice(2, 2, 3)}, 2).unsqueeze(2);
        auto r3 = torch::cat({zeros, zeros, zeros, ones}, 2).unsqueeze(2);

        auto T_rest_inv = torch::cat({r0, r1, r2, r3}, 2);
        auto skinning_transforms = torch::matmul(results_G, T_rest_inv);
        auto vertex_transforms = torch::einsum("bkij,vk->bvij", {skinning_transforms, weights});

        auto v_posed_homo = torch::cat({v_posed, torch::ones({batch_size, v_posed.size(1), 1}, device)}, 2);
        auto v_homo = torch::matmul(vertex_transforms, v_posed_homo.unsqueeze(-1)).squeeze(-1);

        auto verts = v_homo.index({torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(0, 3)});
        verts = verts + trans.unsqueeze(1);

        auto J_posed = results_G.index({torch::indexing::Slice(),
                                        torch::indexing::Slice(),
                                        torch::indexing::Slice(0, 3),
                                        3});
        J_posed = J_posed + trans.unsqueeze(1);

        return {verts, J_posed, results_G, vertex_transforms};
    }
};
