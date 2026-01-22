#pragma once
#include <torch/torch.h>
#include <torch/script.h> // Required for pickle_load
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

// ==========================================
// Struct: SMPL Output Container
// ==========================================
struct SmplOutput {
    torch::Tensor vertices;          // [B, 6890, 3]
    torch::Tensor joints;            // [B, 24, 3]
    torch::Tensor transforms;        // [B, 24, 4, 4] World transforms of joints
    torch::Tensor vertex_transforms; // [B, 6890, 4, 4] Skinning transforms per vertex
};

// ==========================================
// Helper: Batch Rodrigues (Axis-Angle to Rotation Matrix)
// ==========================================
inline torch::Tensor batch_rodrigues(torch::Tensor theta) {
    auto batch_size = theta.sizes()[0];
    auto num_rotations = (theta.dim() == 2) ? 1 : theta.size(1);

    // Flatten to [BxN, 3]
    theta = theta.view({-1, 3});

    auto angle = torch::norm(theta, 2, 1, true) + 1e-8;
    auto r = theta / angle;

    auto angle_cos = torch::cos(angle);
    auto angle_sin = torch::sin(angle);

    auto x = r.index({torch::indexing::Slice(), 0});
    auto y = r.index({torch::indexing::Slice(), 1});
    auto z = r.index({torch::indexing::Slice(), 2});

    // Cross product matrix
    auto K = torch::zeros({theta.size(0), 3, 3}, theta.options());
    K.index_put_({torch::indexing::Slice(), 0, 1}, -z);
    K.index_put_({torch::indexing::Slice(), 0, 2}, y);
    K.index_put_({torch::indexing::Slice(), 1, 0}, z);
    K.index_put_({torch::indexing::Slice(), 1, 2}, -x);
    K.index_put_({torch::indexing::Slice(), 2, 0}, -y);
    K.index_put_({torch::indexing::Slice(), 2, 1}, x);

    auto I = torch::eye(3, theta.options()).unsqueeze(0);

    // Rodrigues formula: I + sin(a)*K + (1-cos(a))*K^2
    auto R = I + angle_sin.unsqueeze(-1) * K + (1 - angle_cos.unsqueeze(-1)) * torch::matmul(K, K);

    if (num_rotations > 1) {
        return R.view({batch_size, num_rotations, 3, 3});
    }
    return R.view({batch_size, 3, 3});
}

// ==========================================
// Helper: Transform Matrix Construction
// ==========================================
inline torch::Tensor transform_mat(torch::Tensor R, torch::Tensor t) {
    auto batch_size = R.size(0);
    auto mat = torch::eye(4, R.options()).unsqueeze(0).repeat({batch_size, 1, 1});

    mat.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
    mat.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 3}, t);

    return mat;
}

// ==========================================
// SMPL Layer Module
// ==========================================
struct SMPLLayer : torch::nn::Module {
    // Model Constants (Buffers)
    torch::Tensor v_template;   // [6890, 3]
    torch::Tensor shapedirs;    // [6890, 3, 10]
    torch::Tensor posedirs;     // [6890, 3, 207]
    torch::Tensor J_regressor;  // [24, 6890]
    torch::Tensor parents;      // [24] (Long)
    torch::Tensor weights;      // [6890, 24]

    // Constructor: Loads parameters from .pt file
    SMPLLayer(const std::string& model_path) {
        std::cout << "Loading SMPL parameters from " << model_path << "..." << std::endl;

        c10::IValue data;
        try {
            // Read the binary file into a vector
            std::ifstream input(model_path, std::ios::binary);
            if (!input) throw std::runtime_error("Could not open file: " + model_path);

            std::vector<char> bytes(
                (std::istreambuf_iterator<char>(input)),
                (std::istreambuf_iterator<char>()));
            input.close();

            // Deserialize the pickle archive
            data = torch::pickle_load(bytes);
        } catch (const std::exception& e) {
            std::cerr << "CRITICAL ERROR: Failed to load " << model_path << "\n" << e.what() << std::endl;
            throw;
        }

        // Convert IValue to GenericDict
        auto dict = data.toGenericDict();

        // Helper to extract tensors by string key
        auto get_tensor = [&](const std::string& key) {
            // Keys in pickle_load are IValues, so we must construct an IValue string key to search
            if (!dict.contains(c10::IValue(key))) {
                throw std::runtime_error("Key '" + key + "' not found in " + model_path);
            }
            return dict.at(c10::IValue(key)).toTensor();
        };

        // 1. Load and Register Buffers
        v_template  = register_buffer("v_template", get_tensor("v_template"));
        auto raw_shapedirs = get_tensor("shapedirs");
        if (raw_shapedirs.size(2) > 10) {
            std::cout << "Notice: Trimming shapedirs from " << raw_shapedirs.size(2) << " to 10 components." << std::endl;
            raw_shapedirs = raw_shapedirs.slice(2, 0, 10).clone(); // Slice dim 2, from 0 to 10
        }
        shapedirs = register_buffer("shapedirs", raw_shapedirs);
        posedirs    = register_buffer("posedirs", get_tensor("posedirs"));
        J_regressor = register_buffer("J_regressor", get_tensor("J_regressor"));
        weights     = register_buffer("weights", get_tensor("weights"));

        // 2. Load Parents (Safety Fix)
        auto raw_parents = get_tensor("parents");

        raw_parents = raw_parents.clone();
        raw_parents[0] = 0;

        parents = register_buffer("parents", raw_parents);

        std::cout << "SMPL Model loaded successfully." << std::endl;
    }

    // Forward Pass
    SmplOutput forward(torch::Tensor betas, torch::Tensor pose_axis_angle, torch::Tensor trans) {
        auto batch_size = betas.size(0);
        auto device = betas.device();

        // 1. Shape Blending
        // v_shaped = v_template + shapedirs * betas
        auto v_shaped = v_template.unsqueeze(0) + torch::einsum("bl,mkl->bmk", {betas, shapedirs});

        // 2. Joint Regression
        // Corrected Einsum: "jv, bvi -> bji"
        auto J = torch::einsum("jv, bvi -> bji", {J_regressor, v_shaped});

        // 3. Pose Blending
        if (pose_axis_angle.dim() == 2) pose_axis_angle = pose_axis_angle.view({batch_size, -1, 3});
        auto rot_mats = batch_rodrigues(pose_axis_angle);

        // Pose Feature: R_n - I
        auto pose_feature = (rot_mats.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)})
                             - torch::eye(3, device)).view({batch_size, -1}); // [B, 207]

        // v_posed = v_shaped + posedirs * pose_feature
        auto v_posed = v_shaped + torch::einsum("bl,mkl->bmk", {pose_feature, posedirs});

        // 4. Forward Kinematics (FK)
        // Using vector/stack to avoid in-place version errors
        std::vector<torch::Tensor> G_vec(24);

        // Move parents to CPU for loop logic
        auto parents_cpu = parents.to(torch::kCPU);
        auto parents_acc = parents_cpu.accessor<int64_t, 1>();

        // Prepare relative offsets safely
        auto J_parents = J.index_select(1, parents);
        auto J_offsets = J - J_parents;

        // Construct offsets list for the loop
        std::vector<torch::Tensor> offsets_list;
        offsets_list.reserve(24);
        for (int i = 0; i < 24; ++i) {
            if (i == 0) offsets_list.push_back(J.index({torch::indexing::Slice(), 0}));
            else offsets_list.push_back(J_offsets.index({torch::indexing::Slice(), i}));
        }

        // FK Loop
        for (int i = 0; i < 24; ++i) {
            auto mat_rot = rot_mats.index({torch::indexing::Slice(), i});
            auto val_trans = offsets_list[i];

            auto local_transform = transform_mat(mat_rot, val_trans);

            if (i == 0) {
                // Root
                G_vec[i] = local_transform;
            } else {
                // Child: Global_parent * Local_child
                int64_t parent_idx = parents_acc[i];
                G_vec[i] = torch::matmul(G_vec[parent_idx], local_transform);
            }
        }

        // Stack results [B, 24, 4, 4]
        auto results_G = torch::stack(G_vec, 1);

        // 5. Linear Blend Skinning
        // T_k = G_k * Translate(-J_rest)
        auto neg_J = -J;
        auto T_rest_inv = torch::eye(4, device).unsqueeze(0).unsqueeze(0).repeat({batch_size, 24, 1, 1});
        T_rest_inv.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3), 3}, neg_J);

        auto skinnning_transforms = torch::matmul(results_G, T_rest_inv);

        // Blend transforms per vertex
        auto vertex_transforms = torch::einsum("bkij,vk->bvij", {skinnning_transforms, weights});

        // Apply skinning
        auto v_posed_homo = torch::cat({v_posed, torch::ones({batch_size, v_posed.size(1), 1}, device)}, 2);
        auto v_homo = torch::matmul(vertex_transforms, v_posed_homo.unsqueeze(-1)).squeeze(-1);

        // Extract verts and add global translation
        auto verts = v_homo.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
        verts = verts + trans.unsqueeze(1);

        // Extract posed joint positions from global transforms and add translation
        auto J_posed = results_G.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(0, 3),
            3
        });
        J_posed = J_posed + trans.unsqueeze(1);

        return {verts, J_posed, results_G, vertex_transforms};
    }
};
