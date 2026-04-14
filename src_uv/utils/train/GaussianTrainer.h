#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "GaussianRasterizer.h"
#include "utils/math/LossFunctions.h"
#include "utils/render/RenderMathUtils.h"
#include "utils/train/GaussianAvatar.h"
#include "utils/train/GaussianDataLoader.h"
#include "utils/train/PoseRefiner.h"

struct TrainStepResult
{
    bool stepped = false;
    torch::Tensor loss;
    torch::Tensor valid_sum;
    int total_samples = 0;
    torch::Tensor rot_delta_mean;
    torch::Tensor trans_delta_max;
    bool verbose_captured = false;
    torch::Tensor final_recon;
    torch::Tensor final_ssim;
    torch::Tensor final_sobel;
    torch::Tensor final_outside;
    torch::Tensor final_alpha;
    torch::Tensor offset_reg;
    torch::Tensor scale_reg;
    torch::Tensor sh_reg_loss;
    torch::Tensor sugar_reg;
    torch::Tensor knn_reg;
    torch::Tensor opacity_reg;
    torch::Tensor anisotropic_reg;
    torch::Tensor grad_rot_norm;
    torch::Tensor grad_offset_norm;
    torch::Tensor grad_scale_norm;
    torch::Tensor grad_color_norm;
    torch::Tensor grad_opacity_norm;
    torch::Tensor grad_pose_norm;
    torch::Tensor scale_mean;
    torch::Tensor scale_min;
    torch::Tensor scale_max;
    torch::Tensor scale_cap_hit_ratio;
    torch::Tensor opacity_low_ratio;
    torch::Tensor opacity_high_ratio;
    torch::Tensor std_rot;
    torch::Tensor std_offset;
    torch::Tensor std_scale;
    torch::Tensor std_color;
    torch::Tensor std_opacity;
};

class GaussianTrainer
{
public:
    GaussianTrainer(GaussianAvatar &avatar,
                    PoseRefiner &pose_refiner,
                    torch::optim::Adam &optimizer,
                    const torch::Tensor &canonical_betas,
                    const torch::Tensor &window_v,
                    const torch::Tensor &window_h,
                    int ssim_window_size,
                    const torch::Tensor &cam_pos,
                    float render_scale_modifier,
                    float scale_cap,
                    int graph_batch_size,
                    bool use_sh,
                    float render_threshold,
                    float outside_mask_weight,
                    float alpha_loss_weight,
                    float lambda_dssim,
                    float offset_reg_weight,
                    float scale_reg_weight,
                    float sh_reg_weight,
                    float sugar_weight,
                    float opacity_reg_weight,
                    float anisotropic_reg_weight,
                    int opacity_reg_start_epoch,
                    float betas_reg_weight,
                    const torch::Tensor &canonical_betas_anchor,
                    const torch::Tensor &global_expression_bias,
                    float expr_bias_reg_weight)
        : avatar_(avatar),
          pose_refiner_(pose_refiner),
          optimizer_(optimizer),
          canonical_betas_(canonical_betas),
          window_v_(window_v),
          window_h_(window_h),
          ssim_window_size_(ssim_window_size),
          cam_pos_(cam_pos),
          render_scale_modifier_(render_scale_modifier),
          scale_cap_(scale_cap),
          graph_batch_size_(graph_batch_size),
          use_sh_(use_sh),
          render_threshold_(render_threshold),
          outside_mask_weight_(outside_mask_weight),
          alpha_loss_weight_(alpha_loss_weight),
          lambda_dssim_(lambda_dssim),
          offset_reg_weight_(offset_reg_weight),
          scale_reg_weight_(scale_reg_weight),
          sh_reg_weight_(sh_reg_weight),
          sugar_weight_(sugar_weight),
          opacity_reg_weight_(opacity_reg_weight),
          anisotropic_reg_weight_(anisotropic_reg_weight),
          opacity_reg_start_epoch_(opacity_reg_start_epoch),
          betas_reg_weight_(betas_reg_weight),
          canonical_betas_anchor_(canonical_betas_anchor),
          global_expression_bias_(global_expression_bias),
          expr_bias_reg_weight_(expr_bias_reg_weight)
    {
    }

    TrainStepResult TrainStep(const TrainingBatch &batch,
                              int epoch,
                              int batch_step,
                              int sh_degree_eff,
                              bool capture_verbose_metrics)
    {
        TrainStepResult result;
        if (batch.sample_indices.empty())
        {
            return result;
        }

        optimizer_.zero_grad();

        std::vector<torch::Tensor> images_list;
        std::vector<torch::Tensor> alphas_list;
        std::vector<torch::Tensor> targets_list;
        std::vector<torch::Tensor> masks_list;
        std::vector<torch::Tensor> valids_list;

        images_list.reserve(batch.sample_indices.size());
        alphas_list.reserve(batch.sample_indices.size());
        targets_list.reserve(batch.sample_indices.size());
        masks_list.reserve(batch.sample_indices.size());
        valids_list.reserve(batch.sample_indices.size());

        auto pose_flat = batch.pose_base_batch.view({-1, 72});
        auto net_input = torch::cat({pose_flat, batch.trans_base_batch}, 1);
        auto deltas = pose_refiner_.forward(net_input, batch.crop_params, batch.time_tensor);

        auto pose_deltas = deltas.slice(1, 0, 72).view({-1, 24, 3});
        auto trans_deltas = deltas.slice(1, 72, 75).view({-1, 3});

        if ((batch_step % 50 == 0) || capture_verbose_metrics)
        {
            result.rot_delta_mean = pose_deltas.detach().abs().mean();
            result.trans_delta_max = trans_deltas.detach().abs().max();
        }

        auto pose_total = batch.pose_base_batch + pose_deltas;
        auto trans_total = batch.trans_base_batch + trans_deltas;
        const int64_t batch_count = pose_total.size(0);
        auto betas_batch = canonical_betas_.expand({batch_count, canonical_betas_.size(1)});
        auto zero_trans = torch::zeros_like(trans_total);

        torch::Tensor batch_means3D;
        torch::Tensor batch_rots;

        {
            auto expression_batch_corr = batch.expression_batch +
                                         global_expression_bias_.expand({batch_count, global_expression_bias_.size(1)});
            std::tie(batch_means3D, batch_rots) = avatar_.forward(betas_batch,
                                                                  expression_batch_corr,
                                                                  pose_total,
                                                                  batch.jaw_pose_batch,
                                                                  batch.eye_pose_batch,
                                                                  batch.left_hand_pose_batch,
                                                                  batch.right_hand_pose_batch,
                                                                  zero_trans);
        }

        if (batch_means3D.dim() == 2)
        {
            batch_means3D = batch_means3D.unsqueeze(0);
        }
        if (batch_rots.dim() == 2)
        {
            batch_rots = batch_rots.unsqueeze(0);
        }

        auto ones = torch::ones_like(batch.y_signs);
        auto y_scales = torch::stack({ones, batch.y_signs, ones}, 1).unsqueeze(1);
        batch_means3D = batch_means3D * y_scales + trans_total.unsqueeze(1);

        for (size_t k = 0; k < batch.sample_indices.size(); ++k)
        {
            const auto &image_slice = batch.image_slices[k];
            const auto &mask_slice = batch.mask_slices[k];

            auto target_u8 = batch.packed_images_u8
                                 .narrow(0, image_slice.offset, image_slice.length)
                                 .view({image_slice.shape[0], image_slice.shape[1], image_slice.shape[2]});
            auto matte_mask_raw = batch.packed_masks
                                      .narrow(0, mask_slice.offset, mask_slice.length)
                                      .view({mask_slice.shape[0], mask_slice.shape[1], mask_slice.shape[2]});

            const int H = static_cast<int>(image_slice.shape[0]);
            const int W = static_cast<int>(image_slice.shape[1]);
            const int C = static_cast<int>(image_slice.shape[2]);
            if (H <= 0 || W <= 0)
            {
                continue;
            }
            if (C != 3)
            {
                continue;
            }

            auto target = target_u8.to(torch::kFloat32).div(255.0f);
            target = target.flip({2}).permute({2, 0, 1}).contiguous();
            auto matte_mask = matte_mask_raw.to(torch::kFloat32);
            if (matte_mask.dim() == 2)
            {
                matte_mask = matte_mask.unsqueeze(0);
            }
            else if (matte_mask.dim() == 3 && matte_mask.size(0) == 1)
            {
            }
            else if (matte_mask.dim() == 3 && matte_mask.size(2) == 1)
            {
                matte_mask = matte_mask.permute({2, 0, 1});
            }
            else if (matte_mask.dim() == 3 && matte_mask.size(1) == 1)
            {
                matte_mask = matte_mask.permute({1, 0, 2});
            }
            else
            {
                continue;
            }
            matte_mask = matte_mask.contiguous();
            if (matte_mask.size(1) != H || matte_mask.size(2) != W)
            {
                continue;
            }

            auto means3D = batch_means3D.index({static_cast<int64_t>(k)});
            auto current_rots = batch_rots.index({static_cast<int64_t>(k)});

            torch::Tensor current_sh = use_sh_ ? avatar_.current_flat_sh : torch::zeros({0}, avatar_.g_colors.options());
            if (use_sh_ && sh_degree_eff > 0)
            {
                current_sh = RotateSH(current_sh, current_rots);
            }

            auto outputs = GaussianRasterizer::apply(
                means3D,
                (use_sh_ ? torch::zeros({0}, avatar_.g_colors.options()) : avatar_.current_flat_colors),
                avatar_.current_flat_opacities,
                CappedScales(avatar_.current_flat_scales),
                current_rots,
                render_scale_modifier_,
                batch.view_mats.index({static_cast<int64_t>(k)}),
                batch.proj_mats.index({static_cast<int64_t>(k)}),
                batch.tan_fovx[k],
                batch.tan_fovy[k],
                H,
                W,
                current_sh,
                (use_sh_ ? sh_degree_eff : 0),
                cam_pos_,
                false);

            auto image = outputs[0];
            auto alpha = outputs[1];
            if (!image.defined() || image.dim() != 3 || image.size(0) != 3 || image.size(1) != H || image.size(2) != W)
            {
                continue;
            }

            if (alpha.dim() == 2)
            {
                alpha = alpha.unsqueeze(0);
            }
            else if (alpha.dim() == 3 && alpha.size(0) == 1)
            {
            }
            else if (alpha.dim() == 3 && alpha.size(2) == 1)
            {
                alpha = alpha.permute({2, 0, 1});
            }
            else if (alpha.dim() == 3 && alpha.size(1) == 1)
            {
                alpha = alpha.permute({1, 0, 2});
            }
            else
            {
                continue;
            }
            alpha = alpha.contiguous();
            if (alpha.size(1) != H || alpha.size(2) != W)
            {
                continue;
            }

            if (!images_list.empty())
            {
                if (image.size(1) != images_list.front().size(1) ||
                    image.size(2) != images_list.front().size(2))
                {
                    continue;
                }
            }

            auto valid_pixels = (image > render_threshold_).to(torch::kFloat32) * matte_mask;
            auto coverage_ratio = valid_pixels.sum() / (matte_mask.sum() + 1e-6f);
            auto is_valid_sample = (coverage_ratio > 0.3f).to(torch::kFloat32).detach();

            images_list.push_back(image);
            alphas_list.push_back(alpha);
            targets_list.push_back(target);
            masks_list.push_back(matte_mask);
            valids_list.push_back(is_valid_sample.view({1}));
        }

        if (images_list.empty())
        {
            return result;
        }

        auto b_images = torch::stack(images_list);
        auto b_targets = torch::stack(targets_list);
        auto b_masks = torch::stack(masks_list);
        auto b_alphas = torch::stack(alphas_list);
        auto b_valids = torch::cat(valids_list).to(b_images.options().dtype());

        const int64_t B = b_images.size(0);

        auto bg_color = torch::rand({B, 3, 1, 1}, b_images.options());
        auto comp_images = b_images + bg_color * (1.0f - b_alphas);
        auto comp_targets = (b_targets * b_masks) + bg_color * (1.0f - b_masks);

        auto diff = torch::abs(comp_images - comp_targets);
        auto recon_losses = diff.sum({1, 2, 3}) /
                            torch::clamp_min(b_masks.sum({1, 2, 3}) * 3.0f, 1e-6f);
        auto outside_losses = (b_images * (1.0f - b_masks)).mean({1, 2, 3});
        auto alpha_losses = torch::abs(b_alphas - b_masks).mean({1, 2, 3});

        auto b_images_down = Downsample(b_images);
        auto b_targets_down = Downsample(b_targets);
        auto b_masks_down = Downsample(b_masks);

        auto ssim_values = ssim_fast(b_images_down, b_targets_down, window_v_, window_h_, b_masks_down, ssim_window_size_);
        auto ssim_losses = 1.0f - ssim_values;

        auto sobel_options = b_images.options().dtype(torch::kFloat32);
        auto sobel_x = torch::tensor({{-1.0f, 0.0f, 1.0f},
                                      {-2.0f, 0.0f, 2.0f},
                                      {-1.0f, 0.0f, 1.0f}},
                                     sobel_options)
                           .view({1, 1, 3, 3})
                           .repeat({3, 1, 1, 1});
        auto sobel_y = torch::tensor({{-1.0f, -2.0f, -1.0f},
                                      {0.0f, 0.0f, 0.0f},
                                      {1.0f, 2.0f, 1.0f}},
                                     sobel_options)
                           .view({1, 1, 3, 3})
                           .repeat({3, 1, 1, 1});

        auto conv_opts = torch::nn::functional::Conv2dFuncOptions().padding(1).groups(3);
        auto pred_gx = torch::nn::functional::conv2d(comp_images, sobel_x, conv_opts);
        auto pred_gy = torch::nn::functional::conv2d(comp_images, sobel_y, conv_opts);
        auto targ_gx = torch::nn::functional::conv2d(comp_targets, sobel_x, conv_opts);
        auto targ_gy = torch::nn::functional::conv2d(comp_targets, sobel_y, conv_opts);

        auto pred_mag = torch::sqrt(pred_gx.pow(2) + pred_gy.pow(2) + 1e-6f);
        auto targ_mag = torch::sqrt(targ_gx.pow(2) + targ_gy.pow(2) + 1e-6f);
        auto sobel_diff = torch::abs(pred_mag - targ_mag);
        auto sobel_losses = sobel_diff.sum({1, 2, 3}) /
                            torch::clamp_min(b_masks.sum({1, 2, 3}) * 3.0f, 1e-6f);

        recon_losses = recon_losses * b_valids;
        outside_losses = outside_losses * b_valids;
        alpha_losses = alpha_losses * b_valids;
        ssim_losses = ssim_losses * b_valids;
        sobel_losses = sobel_losses * b_valids;

        auto total_losses = recon_losses + outside_mask_weight_ * outside_losses + alpha_loss_weight_ * alpha_losses;
        total_losses = torch::where(torch::isfinite(total_losses), total_losses, torch::zeros_like(total_losses));

        const float outlier_percentile = 1.0f;
        auto outlier_threshold = torch::quantile(total_losses, outlier_percentile);
        auto inlier_mask = (total_losses <= outlier_threshold).to(torch::kFloat32);

        auto final_weight = inlier_mask * b_valids;
        auto valid_sum = final_weight.sum();
        auto denom = torch::clamp_min(valid_sum, 1e-6f);

        auto final_recon = (recon_losses * final_weight).sum() / denom;
        auto final_ssim = (ssim_losses * final_weight).sum() / denom;
        auto final_outside = (outside_losses * final_weight).sum() / denom;
        auto final_alpha = (alpha_losses * final_weight).sum() / denom;
        auto final_sobel = (sobel_losses * final_weight).sum() / denom;

        auto current_scales = CappedScales(avatar_.current_flat_scales) * render_scale_modifier_;
        auto scale_reg = torch::mean(current_scales.pow(2).sum(1));
        auto offset_xy_reg = torch::mean(avatar_.current_flat_offsets.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).pow(2).sum(1));
        auto offset_z = avatar_.current_flat_offsets.index({torch::indexing::Slice(), 2});
        auto offset_z_reg = torch::mean(offset_z.pow(2));
        auto offset_center_reg = offset_z.mean().pow(2);
        auto offset_reg = offset_z_reg + (0.02f * offset_xy_reg) + (0.1f * offset_center_reg);
        auto x_scale = current_scales.index({torch::indexing::Slice(), 0});
        auto y_scale = current_scales.index({torch::indexing::Slice(), 1});
        auto xy_max = torch::max(x_scale, y_scale);
        auto xy_min = torch::min(x_scale, y_scale);
        auto anisotropic_loss = torch::mean(xy_max / torch::clamp_min(xy_min, 1e-5f));

        torch::Tensor cap_penalty = torch::zeros({1}, current_scales.options());
        if (scale_cap_ > 0.0f)
        {
            auto excess = torch::relu(current_scales - (scale_cap_ * render_scale_modifier_));
            cap_penalty = excess.pow(2).mean();
        }

        using torch::indexing::Slice;

        auto z_scale = current_scales.index({Slice(), 2});
        auto skin_thinness_loss = z_scale.pow(2).mean();

        auto calc_masked_tv = [&](const torch::Tensor &map)
        {
            auto mask_f = avatar_.valid_uv_mask.to(torch::kFloat32).unsqueeze(0);

            auto mask_y = mask_f.slice(1, 1) * mask_f.slice(1, 0, -1);
            auto mask_x = mask_f.slice(2, 1) * mask_f.slice(2, 0, -1);

            auto diff_y = torch::abs(map.slice(1, 1) - map.slice(1, 0, -1));
            auto diff_x = torch::abs(map.slice(2, 1) - map.slice(2, 0, -1));

            auto tv_y = (diff_y * mask_y).sum() /
                        torch::clamp_min(mask_y.sum() * map.size(0), 1.0f);
            auto tv_x = (diff_x * mask_x).sum() /
                        torch::clamp_min(mask_x.sum() * map.size(0), 1.0f);

            return tv_y + tv_x;
        };

        auto tv_offset = calc_masked_tv(avatar_.g_offsets);
        auto tv_scale = calc_masked_tv(torch::exp(avatar_.g_scales));
        auto tv_loss = (tv_offset * 0.1f) + (tv_scale * 0.05f);
        auto sugar_loss = skin_thinness_loss + tv_loss;

        auto clamped_opacities = torch::clamp(avatar_.g_opacities, 0.0f, 1.0f);
        auto opacity_binarization_loss = (clamped_opacities * (1.0f - clamped_opacities)).mean();
        const float opacity_reg_scale = (epoch >= opacity_reg_start_epoch_) ? 1.0f : 0.0f;
        auto opacity_reg_loss = opacity_binarization_loss * opacity_reg_scale;
        auto betas_reg_loss = (canonical_betas_ - canonical_betas_anchor_).pow(2).mean();
        auto expr_bias_reg_loss = global_expression_bias_.pow(2).mean();

        torch::Tensor sh_reg_loss = torch::zeros({1}, avatar_.g_sh.options());
        if (use_sh_ && avatar_.g_sh.defined() && avatar_.g_sh.numel() > 3)
        {
            auto higher_orders = avatar_.g_sh.index({Slice(3, torch::indexing::None), Slice(), Slice()});
            sh_reg_loss = higher_orders.pow(2).mean();
        }

        const float sobel_weight = 0.5f;
        auto loss = (1.0f - lambda_dssim_) * final_recon +
                    lambda_dssim_ * final_ssim +
                    sobel_weight * final_sobel +
                    outside_mask_weight_ * final_outside +
                    alpha_loss_weight_ * final_alpha +
                    offset_reg_weight_ * offset_reg +
                    scale_reg_weight_ * scale_reg +
                    sh_reg_weight_ * sh_reg_loss +
                    sugar_weight_ * sugar_loss +
                    opacity_reg_weight_ * opacity_reg_loss +
                    anisotropic_reg_weight_ * anisotropic_loss +
                    betas_reg_weight_ * betas_reg_loss +
                    expr_bias_reg_weight_ * expr_bias_reg_loss +
                    100.0f * cap_penalty;

        if (capture_verbose_metrics)
        {
            result.verbose_captured = true;
            result.final_recon = final_recon.detach();
            result.final_ssim = final_ssim.detach();
            result.final_sobel = final_sobel.detach();
            result.final_outside = final_outside.detach();
            result.final_alpha = final_alpha.detach();
            result.offset_reg = offset_reg.detach();
            result.scale_reg = scale_reg.detach();
            result.sh_reg_loss = sh_reg_loss.detach();
            result.sugar_reg = sugar_loss.detach();
            result.knn_reg = tv_loss.detach();
            result.opacity_reg = opacity_reg_loss.detach();
            result.anisotropic_reg = anisotropic_loss.detach();
            result.scale_mean = current_scales.mean().detach();
            result.scale_min = current_scales.min().detach();
            result.scale_max = current_scales.max().detach();

            auto cap_hit_ratio = torch::zeros({1}, current_scales.options());
            if (scale_cap_ > 0.0f)
            {
                const float final_scale_cap = scale_cap_ * render_scale_modifier_;
                cap_hit_ratio =
                    (current_scales >= (final_scale_cap * 0.999f))
                        .to(current_scales.options().dtype())
                        .mean();
            }
            result.scale_cap_hit_ratio = cap_hit_ratio.detach();

            auto opacity_values = torch::clamp(avatar_.current_flat_opacities.detach(), 0.0f, 1.0f);
            result.opacity_low_ratio =
                (opacity_values <= 0.001f).to(opacity_values.options().dtype()).mean().detach();
            result.opacity_high_ratio =
                (opacity_values >= 0.999f).to(opacity_values.options().dtype()).mean().detach();
        }

        loss.backward();

        if (capture_verbose_metrics)
        {
            auto grad_norm_from_param = [](const torch::Tensor &param) -> torch::Tensor
            {
                auto grad = param.grad();
                if (!grad.defined())
                {
                    return torch::Tensor();
                }
                return grad.detach().pow(2).sum().sqrt();
            };

            result.grad_rot_norm = grad_norm_from_param(avatar_.g_rots);
            result.grad_offset_norm = grad_norm_from_param(avatar_.g_offsets);
            result.grad_scale_norm = grad_norm_from_param(avatar_.g_scales);
            result.grad_color_norm = grad_norm_from_param(use_sh_ ? avatar_.g_sh : avatar_.g_colors);
            result.grad_opacity_norm = grad_norm_from_param(avatar_.g_opacities);

            const auto color_param = use_sh_ ? avatar_.g_sh : avatar_.g_colors;
            result.std_rot = avatar_.g_rots.detach().std();
            result.std_offset = avatar_.current_flat_offsets.index({Slice(), 2}).detach().std();
            result.std_scale = avatar_.g_scales.detach().std();
            result.std_color = color_param.defined() ? color_param.detach().std() : torch::Tensor();
            result.std_opacity = avatar_.g_opacities.detach().std();

            torch::Tensor pose_grad_sq = torch::zeros({1}, avatar_.g_offsets.options());
            bool has_pose_grad = false;
            for (const auto &param : pose_refiner_.parameters())
            {
                auto grad = param.grad();
                if (!grad.defined())
                {
                    continue;
                }
                pose_grad_sq = pose_grad_sq + grad.detach().pow(2).sum();
                has_pose_grad = true;
            }
            if (has_pose_grad)
            {
                result.grad_pose_norm = pose_grad_sq.sqrt();
            }
        }

        if (pose_refiner_.parameters().size() > 0)
        {
            torch::nn::utils::clip_grad_norm_(pose_refiner_.parameters(), 1.0f);
        }

        optimizer_.step();

        {
            torch::NoGradGuard no_grad;

            auto rot_norm = avatar_.g_rots.norm(2, 0, true);
            rot_norm = torch::clamp_min(rot_norm, 1e-9);
            avatar_.g_rots.div_(rot_norm);
        }

        result.stepped = true;
        result.loss = loss.detach();
        result.valid_sum = valid_sum.detach();
        result.total_samples = static_cast<int>(total_losses.size(0));
        return result;
    }

private:
    torch::Tensor CappedScales(const torch::Tensor &log_scales) const
    {
        return torch::exp(log_scales);
    }

    GaussianAvatar &avatar_;
    PoseRefiner &pose_refiner_;
    torch::optim::Adam &optimizer_;
    const torch::Tensor &canonical_betas_;
    const torch::Tensor &window_v_;
    const torch::Tensor &window_h_;
    int ssim_window_size_ = 11;
    const torch::Tensor &cam_pos_;
    float render_scale_modifier_ = 1.0f;
    float scale_cap_ = -1.0f;
    int graph_batch_size_ = 0;
    bool use_sh_ = false;
    float render_threshold_ = 0.0f;
    float outside_mask_weight_ = 0.0f;
    float alpha_loss_weight_ = 0.0f;
    float lambda_dssim_ = 0.0f;
    float offset_reg_weight_ = 0.0f;
    float scale_reg_weight_ = 0.0f;
    float sh_reg_weight_ = 0.0f;
    float sugar_weight_ = 0.0f;
    float opacity_reg_weight_ = 0.0f;
    float anisotropic_reg_weight_ = 0.0f;
    int opacity_reg_start_epoch_ = 0;
    float betas_reg_weight_ = 0.0f;
    torch::Tensor canonical_betas_anchor_;
    torch::Tensor global_expression_bias_;
    float expr_bias_reg_weight_ = 0.0f;
    bool graph_captured_ = false;
    int64_t captured_gaussians_ = -1;
#if GAUSS_HAS_CUDA_GRAPH
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
#endif
    torch::Tensor s_betas_;
    torch::Tensor s_pose_;
    torch::Tensor s_trans_;
    torch::Tensor s_means_;
    torch::Tensor s_rots_;
};
