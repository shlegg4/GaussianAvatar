#include "utils/train/GaussianDensification.h"

#include <algorithm>
#include <cmath>
#include <iostream>

void DensificationState::EnsureLike(const torch::Tensor &offsets, const torch::Tensor &scales)
{
    if (!grad_offsets_accum.defined() || grad_offsets_accum.sizes() != offsets.sizes())
    {
        grad_offsets_accum = torch::zeros_like(offsets);
    }
    if (!grad_scales_accum.defined() || grad_scales_accum.sizes() != scales.sizes())
    {
        grad_scales_accum = torch::zeros_like(scales);
    }
}

void DensificationState::Accumulate(const torch::Tensor &offsets, const torch::Tensor &scales)
{
    auto off_grad = offsets.grad();
    auto sca_grad = scales.grad();
    if (!off_grad.defined() || !sca_grad.defined())
    {
        std::cout << "Warning: Attempted to accumulate densification gradients, but gradients are undefined." << std::endl;
        return;
    }
    EnsureLike(offsets, scales);
    grad_offsets_accum = grad_offsets_accum + off_grad.detach();
    grad_scales_accum = grad_scales_accum + sca_grad.detach();
    steps++;
}

void DensificationState::Reset()
{
    if (grad_offsets_accum.defined())
    {
        grad_offsets_accum.zero_();
    }
    if (grad_scales_accum.defined())
    {
        grad_scales_accum.zero_();
    }
    steps = 0;
}

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
                              DensificationState *state)
{
    DensifyStats stats;
    torch::NoGradGuard no_grad;
    if (!state || state->steps <= 0)
    {
        return stats;
    }
    if (!g_scales.defined() || g_scales.numel() == 0)
    {
        state->Reset();
        return stats;
    }

    const float steps = static_cast<float>(std::max<int64_t>(1, state->steps));
    auto grad_offsets = state->grad_offsets_accum / steps;
    auto grad_scales = state->grad_scales_accum / steps;
    auto grad_norm = torch::norm(grad_offsets, 2, 1) + torch::norm(grad_scales, 2, 1);
    auto grad_scale_mean = grad_scales.mean(1);

    int64_t total = g_scales.size(0);
    auto scale_vals = torch::exp(g_scales) * render_scale_modifier;
    auto scale_mean = scale_vals.mean(1);

    auto opacities = g_opacities.squeeze(1);
    const float grad_norm_min = grad_norm.min().item<float>();
    const float grad_norm_mean = grad_norm.mean().item<float>();
    const float grad_norm_max = grad_norm.max().item<float>();
    const float grad_scale_min = grad_scale_mean.min().item<float>();
    const float grad_scale_mean_val = grad_scale_mean.mean().item<float>();
    const float grad_scale_max = grad_scale_mean.max().item<float>();
    std::cout << "Densify grads (norm min/mean/max): "
              << grad_norm_min << " / " << grad_norm_mean << " / " << grad_norm_max
              << " | scale-grad mean min/mean/max: "
              << grad_scale_min << " / " << grad_scale_mean_val << " / " << grad_scale_max
              << std::endl;
    auto keep_mask = torch::ones({total}, torch::TensorOptions().device(g_scales.device()).dtype(torch::kBool));
    if (cfg.prune_opacity_threshold > 0.0f && cfg.prune_max > 0)
    {
        auto prune_mask = opacities < cfg.prune_opacity_threshold;
        auto prune_indices = torch::nonzero(prune_mask).squeeze(1);
        if (prune_indices.defined() && prune_indices.numel() > 0)
        {
            int64_t k_prune = std::min<int64_t>(cfg.prune_max, prune_indices.size(0));
            if (k_prune > 0)
            {
                prune_indices = prune_indices.index({torch::indexing::Slice(0, k_prune)});
                keep_mask.index_put_({prune_indices}, false);
                stats.pruned = k_prune;
            }
        }
    }

    auto keep_indices = torch::nonzero(keep_mask).squeeze(1);
    if (!keep_indices.defined() || keep_indices.numel() == 0)
    {
        state->Reset();
        return stats;
    }

    auto kept_scales = g_scales.index_select(0, keep_indices);
    auto kept_rots = g_rots.index_select(0, keep_indices);
    auto kept_opacities = g_opacities.index_select(0, keep_indices);
    auto kept_colors = g_colors.index_select(0, keep_indices);
    auto kept_offsets = g_offsets.index_select(0, keep_indices);
    auto kept_bary = g_bary_coords.index_select(0, keep_indices);
    auto kept_faces = g_face_indices.index_select(0, keep_indices);
    torch::Tensor kept_sh;
    if (g_sh.defined() && g_sh.numel() > 0)
    {
        kept_sh = g_sh.index_select(0, keep_indices);
    }

    auto kept_grad_offsets = grad_offsets.index_select(0, keep_indices);
    auto kept_grad_norm = grad_norm.index_select(0, keep_indices);
    auto kept_scale_mean = scale_mean.index_select(0, keep_indices);
    auto kept_opacity_flat = opacities.index_select(0, keep_indices);

    auto split_mask = kept_grad_norm > cfg.min_grad_norm;
    if (cfg.prune_opacity_threshold > 0.0f)
    {
        split_mask = split_mask & (kept_opacity_flat >= cfg.prune_opacity_threshold);
    }
    auto split_indices = torch::nonzero(split_mask).squeeze(1);

    int64_t k = 0;
    if (split_indices.defined() && split_indices.numel() > 0)
    {
        k = std::min<int64_t>(cfg.max_splits, split_indices.size(0));
    }

    if (k > 0)
    {
        auto cand_scores = kept_grad_norm.index_select(0, split_indices);
        auto topk = torch::topk(cand_scores, k);
        auto parent_local = split_indices.index_select(0, std::get<1>(topk));

        const float safe_scale_factor = std::max(cfg.split_scale_factor, 1e-6f);
        auto delta_log = torch::tensor(std::log(safe_scale_factor), g_scales.options());

        auto parent_scales = kept_scales.index_select(0, parent_local);
        auto parent_offsets = kept_offsets.index_select(0, parent_local);
        auto parent_rots = kept_rots.index_select(0, parent_local);
        auto parent_colors = kept_colors.index_select(0, parent_local);
        auto parent_opacities = kept_opacities.index_select(0, parent_local);
        auto parent_bary = kept_bary.index_select(0, parent_local);
        auto parent_faces = kept_faces.index_select(0, parent_local);
        torch::Tensor parent_sh;
        if (kept_sh.defined() && kept_sh.numel() > 0)
        {
            parent_sh = kept_sh.index_select(0, parent_local);
        }

        auto parent_grad = kept_grad_offsets.index_select(0, parent_local);
        auto grad_len = torch::norm(parent_grad, 2, 1, true);
        auto rand_dir = torch::randn_like(parent_grad);
        rand_dir = rand_dir / (torch::norm(rand_dir, 2, 1, true) + 1e-8f);
        auto dir = torch::where(grad_len > 1e-8f, parent_grad / grad_len, rand_dir);

        auto scale_mag = kept_scale_mean.index_select(0, parent_local).unsqueeze(1);
        auto delta = dir * (scale_mag * cfg.split_offset_scale);

        auto new_scale = parent_scales + delta_log;
        auto new_opacity = parent_opacities * 0.5f;

        kept_scales.index_put_({parent_local}, new_scale);
        kept_offsets.index_put_({parent_local}, parent_offsets + delta);
        kept_opacities.index_put_({parent_local}, new_opacity);

        auto child_scales = new_scale;
        auto child_offsets = parent_offsets - delta;
        auto child_rots = parent_rots;
        auto child_colors = parent_colors;
        auto child_opacities = new_opacity;
        auto child_bary = parent_bary;
        auto child_faces = parent_faces;

        kept_scales = torch::cat({kept_scales, child_scales}, 0);
        kept_offsets = torch::cat({kept_offsets, child_offsets}, 0);
        kept_rots = torch::cat({kept_rots, child_rots}, 0);
        kept_colors = torch::cat({kept_colors, child_colors}, 0);
        kept_opacities = torch::cat({kept_opacities, child_opacities}, 0);
        kept_bary = torch::cat({kept_bary, child_bary}, 0);
        kept_faces = torch::cat({kept_faces, child_faces}, 0);
        if (kept_sh.defined() && kept_sh.numel() > 0)
        {
            kept_sh = torch::cat({kept_sh, parent_sh}, 0);
        }
    }

    g_scales = kept_scales.detach().clone().set_requires_grad(true);
    g_offsets = kept_offsets.detach().clone().set_requires_grad(true);
    g_rots = kept_rots.detach().clone().set_requires_grad(true);
    g_colors = kept_colors.detach().clone().set_requires_grad(true);
    g_opacities = kept_opacities.detach().clone().set_requires_grad(true);
    g_bary_coords = kept_bary;
    g_face_indices = kept_faces;
    if (g_sh.defined() && g_sh.numel() > 0)
    {
        g_sh = kept_sh.detach().clone().set_requires_grad(true);
    }

    if (cfg.reset_opacity > 0.0f)
    {
        auto new_opacity = torch::full_like(g_opacities, cfg.reset_opacity);
        g_opacities.index_put_({torch::indexing::Slice()}, new_opacity);
    }

    stats.splits = k;
    state->Reset();
    return stats;
}
