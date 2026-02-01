#pragma once

#include <tuple>

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height,
                                                                       torch::Device device);
torch::Tensor ComputeTriFrames(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C);
torch::Tensor MatrixToQuat(const torch::Tensor &rot_mat);
torch::Tensor QuatMultiply(const torch::Tensor &p, const torch::Tensor &q);
torch::Tensor QuatToMat3(const torch::Tensor &q, torch::Device device);
torch::Tensor RotateSH(const torch::Tensor &sh, const torch::Tensor &rotations);
