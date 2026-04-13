#pragma once

#include <torch/torch.h>

std::pair<torch::Tensor, torch::Tensor> create_separable_windows(int window_size, torch::Device device);

torch::Tensor ssim_fast(const torch::Tensor &img1,
                        const torch::Tensor &img2,
                        const torch::Tensor &window_v,
                        const torch::Tensor &window_h,
                        const torch::Tensor &matte_mask,
                        int window_size = 11);

torch::Tensor Downsample(const torch::Tensor &img);
