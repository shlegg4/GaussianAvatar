#include "utils/train/GaussianDensification.h"

#include <algorithm>
#include <cmath>

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

DensifyStats DensifyGaussians(torch::Tensor g_scales,
                              torch::Tensor g_rots,
                              torch::Tensor g_opacities,
                              torch::Tensor g_colors,
                              torch::Tensor g_offsets,
                              torch::Tensor g_sh,
                              torch::Tensor g_bary_coords,
                              torch::Tensor g_face_indices,
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

    const int64_t total = g_scales.size(0);
    auto scale_vals = torch::exp(g_scales) * render_scale_modifier;
    auto scale_mean = scale_vals.mean(1);

    auto opacities = g_opacities.squeeze(1);
    auto pruned_mask = torch::zeros({total}, torch::TensorOptions().device(g_scales.device()).dtype(torch::kBool));
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
                const float min_scale = std::max(render_scale_modifier * 1e-3f, 1e-8f);
                const float log_min_scale = std::log(min_scale / std::max(render_scale_modifier, 1e-8f));
                auto log_min = torch::full({k_prune, 3}, log_min_scale, g_scales.options());
                g_opacities.index_put_({prune_indices, torch::indexing::Slice()}, 0.0f);
                g_scales.index_put_({prune_indices}, log_min);
                g_offsets.index_put_({prune_indices}, 0.0f);
                pruned_mask.index_put_({prune_indices}, true);
                stats.pruned = k_prune;
            }
        }
    }

    auto candidate_mask = scale_mean > cfg.scale_threshold;
    if (cfg.grow_grad_threshold > 0.0f)
    {
        candidate_mask = candidate_mask | (grad_scale_mean > cfg.grow_grad_threshold);
    }
    if (cfg.prune_opacity_threshold > 0.0f)
    {
        candidate_mask = candidate_mask & (opacities >= cfg.prune_opacity_threshold);
    }
    auto candidate_indices = torch::nonzero(candidate_mask).squeeze(1);
    if (!candidate_indices.defined() || candidate_indices.numel() == 0)
    {
        state->Reset();
        return stats;
    }

    int64_t k = std::min<int64_t>(cfg.max_splits, candidate_indices.size(0));
    if (cfg.min_grad_norm > 0.0f)
    {
        auto cand_grad = grad_norm.index_select(0, candidate_indices);
        auto grad_mask = cand_grad > cfg.min_grad_norm;
        auto keep = torch::nonzero(grad_mask).squeeze(1);
        if (keep.numel() == 0)
        {
            state->Reset();
            return stats;
        }
        candidate_indices = candidate_indices.index_select(0, keep);
        k = std::min<int64_t>(k, candidate_indices.size(0));
    }

    auto cand_scores = grad_norm.index_select(0, candidate_indices);
    auto topk = torch::topk(cand_scores, k);
    auto parent_indices = candidate_indices.index_select(0, std::get<1>(topk));

    auto available_mask = torch::ones({total}, torch::TensorOptions().device(g_scales.device()).dtype(torch::kBool));
    available_mask.index_put_({parent_indices}, false);
    if (stats.pruned > 0)
    {
        available_mask.index_put_({pruned_mask}, false);
    }
    auto available_indices = torch::nonzero(available_mask).squeeze(1);
    if (!available_indices.defined() || available_indices.numel() == 0)
    {
        state->Reset();
        return stats;
    }

    k = std::min<int64_t>(k, available_indices.size(0));
    auto available_op = opacities.index_select(0, available_indices);
    auto lowest = torch::topk(-available_op, k);
    auto child_indices = available_indices.index_select(0, std::get<1>(lowest));

    const float safe_scale_factor = std::max(cfg.split_scale_factor, 1e-6f);
    auto delta_log = torch::tensor(std::log(safe_scale_factor), g_scales.options());
    for (int64_t i = 0; i < k; ++i)
    {
        const int64_t p = parent_indices[i].item<int64_t>();
        const int64_t c = child_indices[i].item<int64_t>();

        auto parent_scale = g_scales.index({p}).clone();
        auto new_scale = parent_scale + delta_log;
        g_scales.index_put_({p}, new_scale);
        g_scales.index_put_({c}, new_scale);

        auto parent_offset = g_offsets.index({p}).clone();
        auto grad = grad_offsets.index({p});
        auto grad_len = torch::norm(grad).item<float>();
        torch::Tensor dir;
        if (grad_len > 1e-8f)
        {
            dir = grad / grad_len;
        }
        else
        {
            dir = torch::randn_like(grad);
            dir = dir / (torch::norm(dir).item<float>() + 1e-8f);
        }
        const float scale_mag = scale_mean.index({p}).item<float>();
        auto delta = dir * (scale_mag * cfg.split_offset_scale);

        g_offsets.index_put_({p}, parent_offset + delta);
        g_offsets.index_put_({c}, parent_offset - delta);

        g_rots.index_put_({c}, g_rots.index({p}));
        g_colors.index_put_({c}, g_colors.index({p}));
        auto parent_opacity = g_opacities.index({p}).clone();
        auto new_opacity = parent_opacity * 0.5f;
        g_opacities.index_put_({p}, new_opacity);
        g_opacities.index_put_({c}, new_opacity);
        g_bary_coords.index_put_({c}, g_bary_coords.index({p}));
        g_face_indices.index_put_({c}, g_face_indices.index({p}));
        if (g_sh.defined() && g_sh.numel() > 0)
        {
            g_sh.index_put_({c}, g_sh.index({p}));
        }
    }

    if (cfg.reset_opacity > 0.0f)
    {
        auto new_opacity = torch::full_like(g_opacities, cfg.reset_opacity);
        g_opacities.index_put_({parent_indices}, new_opacity.index({torch::indexing::Slice(0, k)}));
        g_opacities.index_put_({child_indices}, new_opacity.index({torch::indexing::Slice(0, k)}));
    }

    stats.splits = k;
    state->Reset();
    return stats;
}
