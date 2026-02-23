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

bool ExportOrientedPointCloudPly(const std::filesystem::path &path,
                                 const torch::Tensor &positions,
                                 const torch::Tensor &rotations,
                                 const torch::Tensor &scales,
                                 const torch::Tensor &opacities,
                                 const torch::Tensor &colors,
                                 float opacity_threshold,
                                 int samples_per_gaussian);

bool ExtractMeshTSDF_Open3D(const std::filesystem::path &output_path,
                            const torch::Tensor &means3D,
                            const torch::Tensor &colors_or_sh,
                            const torch::Tensor &opacities,
                            const torch::Tensor &scales,
                            const torch::Tensor &rotations,
                            int sh_degree,
                            int H = 1024,
                            int W = 1024,
                            bool save_debug_frames = true);
