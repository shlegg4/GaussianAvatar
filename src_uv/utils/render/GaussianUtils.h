#pragma once

#include <torch/torch.h>

#include <vector>

torch::Tensor BakeSHToRGB(const torch::Tensor &sh,
                          const torch::Tensor &rotations,
                          int sh_degree);

bool BuildSharedGaussianBuffer(const torch::Tensor &positions,
                               const torch::Tensor &colors,
                               const torch::Tensor &opacities,
                               const torch::Tensor &scales,
                               const torch::Tensor &rotations,
                               const torch::Tensor &sh,
                               int sh_degree,
                               std::vector<float> *out);
