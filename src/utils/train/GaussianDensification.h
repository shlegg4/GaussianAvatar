#pragma once

#include <torch/torch.h>

struct DensificationConfig
{
    int every = 500;
    int max_splits = 10000;
    int max_clones = 10000;
    float scale_threshold = 0.001f;
    float split_scale_factor = 0.7f;
    float split_offset_scale = 0.5f;
    float min_grad_norm = 0.0f;
    float grow_grad_threshold = 1e-4f;
    float prune_opacity_threshold = 0.01f;
    int prune_max = 256;
    float reset_opacity = 0.001f;
};

struct DensificationState
{
    torch::Tensor grad_offsets_accum;
    torch::Tensor grad_scales_accum;
    int64_t steps = 0;

    void EnsureLike(const torch::Tensor &offsets, const torch::Tensor &scales);
    void Accumulate(const torch::Tensor &offsets, const torch::Tensor &scales);
    void Reset();
};

struct DensifyStats
{
    int64_t splits = 0;
    int64_t clones = 0;
    int64_t pruned = 0;
};

DensifyStats DensifyGaussians(torch::Tensor &g_scales,
                              torch::Tensor &g_rots,
                              torch::Tensor &g_opacities,
                              torch::Tensor &g_colors,
                              torch::Tensor &g_offsets,
                              torch::Tensor &g_sh,
                              torch::Tensor &g_bary_coords,
                              torch::Tensor &g_face_indices,
                              float render_scale_modifier,
                              const DensificationConfig &cfg,
                              DensificationState *state);
