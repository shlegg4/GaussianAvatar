#pragma once

#include <filesystem>

#include <torch/torch.h>

bool SaveViewerData(const std::filesystem::path &output_dir, int epoch,
                    const torch::Tensor &positions,
                    const torch::Tensor &colors,
                    const torch::Tensor &opacities,
                    const torch::Tensor &scales,
                    const torch::Tensor &rotations,
                    const torch::Tensor &sh,
                    int sh_degree);

bool SaveViewerDataOverwrite(const std::filesystem::path &output_dir,
                             const torch::Tensor &positions,
                             const torch::Tensor &colors,
                             const torch::Tensor &opacities,
                             const torch::Tensor &scales,
                             const torch::Tensor &rotations,
                             const torch::Tensor &sh,
                             int sh_degree);
