#pragma once

#include <torch/torch.h>

struct PoseRefiner : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    const float ROT_SCALE = 0.5f;
    const float TRANS_XY_SCALE = 0.25f;
    const float TRANS_Z_SCALE = 0.01f;

    PoseRefiner(int input_dim = 72 + 3 + 3)
    {
        const int embed_dim = 4;
        const int total_input = input_dim + embed_dim;
        const int hidden_dim = 128;

        fc1 = register_module("fc1", torch::nn::Linear(total_input, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, input_dim - 3));

        torch::nn::init::xavier_uniform_(fc1->weight);
        torch::nn::init::zeros_(fc1->bias);
        torch::nn::init::zeros_(fc3->weight);
        torch::nn::init::zeros_(fc3->bias);
    }

    torch::Tensor forward(torch::Tensor noisy_pose_trans, torch::Tensor crop_params, torch::Tensor time_norm)
    {
        auto device = time_norm.device();
        auto freqs = torch::pow(2.0, torch::arange(0, 2, torch::TensorOptions().device(device)).to(torch::kFloat));
        auto time_projected = time_norm * freqs.unsqueeze(0) * (2.0f * 3.14159f);
        auto time_embed = torch::cat({torch::sin(time_projected), torch::cos(time_projected)}, 1);

        auto x = torch::cat({noisy_pose_trans, crop_params, time_embed}, 1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto delta = fc3->forward(x);

        auto delta_pose = torch::tanh(delta.slice(1, 0, 72)) * ROT_SCALE;

        auto delta_trans_raw = delta.slice(1, 72, 75);

        auto trans_x = delta_trans_raw.slice(1, 0, 1) * TRANS_XY_SCALE;
        auto trans_y = delta_trans_raw.slice(1, 1, 2) * TRANS_XY_SCALE;
        auto trans_z = delta_trans_raw.slice(1, 2, 3) * TRANS_Z_SCALE;

        auto delta_trans = torch::cat({trans_x, trans_y, trans_z}, 1);
        return torch::cat({delta_pose, delta_trans}, 1);
    }
};
