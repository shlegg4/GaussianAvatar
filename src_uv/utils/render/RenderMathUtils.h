#pragma once

#include <tuple>

#include <torch/torch.h>

struct CameraProjectionInput
{
    float focal = 0.0f;
    int full_width = 0;
    int full_height = 0;
    int render_width = 0;
    int render_height = 0;
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_x0 = 0.0f;
    float crop_y0 = 0.0f;
    float crop_w = 0.0f;
    float crop_h = 0.0f;
};

struct CameraProjectionOutput
{
    torch::Tensor view_mat;
    torch::Tensor proj_mat;
    float tan_fovx = 0.0f;
    float tan_fovy = 0.0f;
    float principal_x = 0.0f;
    float principal_y = 0.0f;
};

CameraProjectionOutput BuildCameraProjection(const CameraProjectionInput &input, torch::Device device);

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height,
                                                                       torch::Device device);
std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height,
                                                                       float cx, float cy, torch::Device device);
torch::Tensor ComputeTriFrames(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C);
torch::Tensor MatrixToQuat(const torch::Tensor &rot_mat);
torch::Tensor QuatMultiply(const torch::Tensor &p, const torch::Tensor &q);
torch::Tensor QuatToMat3(const torch::Tensor &q, torch::Device device);
torch::Tensor RotateSH(const torch::Tensor &sh, const torch::Tensor &rotations);
