#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "GaussianRasterizer.h"
#include "SharedGaussian.h"
#include "utils/HmrInferenceUtils.h"
#include "utils/HmrMathHelpers.h"
#include "utils/SmplLBS.h"
#include "utils/image/MaskUtils.h"
#include "utils/image/TensorCvUtils.h"
#include "utils/io/PathUtils.h"
#include "utils/math/StatsUtils.h"
#include "utils/render/RenderMathUtils.h"
#include "utils/train/TrainCache.h"
#include "utils/train/GaussianDensification.h"
#include "utils/train/TrainImageSaver.h"
#include "utils/train/TrainJsonl.h"
#include "utils/train/TrainTypes.h"
#include "utils/train/ViewerExport.h"

bool ParseTrainArgs(int argc, char *argv[], TrainOptions *options)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0)
        {
            std::string key = arg;
            std::string value;
            const size_t eq = arg.find('=');
            if (eq != std::string::npos)
            {
                key = arg.substr(0, eq);
                value = arg.substr(eq + 1);
            }
            else
            {
                if (key != "--viewer" && key != "--headless" && key != "--viewer-stream-poses")
                {
                    if (i + 1 >= argc)
                    {
                        std::cerr << "Missing value for " << key << std::endl;
                        return false;
                    }
                    value = argv[++i];
                }
            }
            if (key == "--jsonl")
                options->jsonl_path = value;
            else if (key == "--smpl")
                options->smpl_model_path = value;
            else if (key == "--num-gaussians")
                options->num_gaussians = std::stoi(value);
            else if (key == "--epochs")
                options->epochs = std::stoi(value);
            else if (key == "--lr")
                options->lr = std::stof(value);
            else if (key == "--lr-decay-epoch")
                options->lr_decay_epoch = std::stoi(value);
            else if (key == "--lr-decay-multiplier")
                options->lr_decay_multiplier = std::stof(value);
            else if (key == "--lr-min-multiplier")
                options->lr_min_multiplier = std::stof(value);
            else if (key == "--output-dir")
                options->output_dir = value;
            else if (key == "--train-dir")
                options->output_dir = value;
            else if (key == "--viewer-export-dir")
                options->viewer_export_dir = value;
            else if (key == "--scale-reg")
                options->scale_reg_weight = std::stof(value);
            else if (key == "--scale-lr")
                options->scale_lr = std::stof(value);
            else if (key == "--scale-max")
                options->scale_max_value = std::stof(value);
            else if (key == "--rot-lr")
                options->rot_lr = std::stof(value);
            else if (key == "--offset-lr")
                options->offset_lr = std::stof(value);
            else if (key == "--offset-reg")
                options->offset_reg_weight = std::stof(value);
            else if (key == "--pose-reg")
                options->pose_reg_weight = std::stof(value);
            else if (key == "--pose-lr")
                options->pose_lr = std::stof(value);
            else if (key == "--alpha-loss")
                options->alpha_loss_weight = std::stof(value);
            else if (key == "--lambda-dssim")
                options->lambda_dssim = std::stof(value);
            else if (key == "--color-lr")
                options->color_lr = std::stof(value);
            else if (key == "--opacity-lr")
                options->opacity_lr = std::stof(value);
            else if (key == "--sh-degree")
                options->sh_degree = std::stoi(value);
            else if (key == "--densify-every")
                options->densify_every = std::stoi(value);
            else if (key == "--densify-max")
                options->densify_max_splits = std::stoi(value);
            else if (key == "--densify-max-clones")
                options->densify_max_clones = std::stoi(value);
            else if (key == "--densify-scale")
                options->densify_scale_threshold = std::stof(value);
            else if (key == "--densify-split-scale")
                options->densify_split_scale = std::stof(value);
            else if (key == "--densify-split-offset")
                options->densify_split_offset = std::stof(value);
            else if (key == "--densify-min-grad")
                options->densify_min_grad = std::stof(value);
            else if (key == "--densify-grow-grad")
                options->densify_grow_grad = std::stof(value);
            else if (key == "--densify-prune-opacity")
                options->densify_prune_opacity = std::stof(value);
            else if (key == "--densify-prune-max")
                options->densify_prune_max = std::stoi(value);
            else if (key == "--densify-reset-opacity")
                options->densify_reset_opacity = std::stof(value);
            else if (key == "--densify-stop-epoch")
                options->densify_stop_epoch = std::stoi(value);
            else if (key == "--viewer")
                options->enable_viewer = true;
            else if (key == "--headless")
                options->enable_viewer = false;
            else if (key == "--viewer-stream-poses")
                options->viewer_stream_poses = true;
            else if (key == "--viewer-every")
                options->viewer_every = std::stoi(value);
            else if (key == "--viewer-shm")
                options->viewer_shm_name = value;
            else if (key == "--viewer-pose-shm")
                options->viewer_pose_shm_name = value;
            else if (key == "--viewer-bind-shm")
                options->viewer_bind_shm_name = value;
            else
            {
                std::cerr << "Unknown argument: " << key << std::endl;
                return false;
            }
        }
        else
        {
            if (options->jsonl_path.empty())
            {
                options->jsonl_path = arg;
            }
            else
            {
                std::cerr << "Unexpected argument: " << arg << std::endl;
                return false;
            }
        }
    }
    if (options->jsonl_path.empty())
    {
        std::cerr << "Missing required --jsonl argument." << std::endl;
        return false;
    }
    if (options->color_lr < 0.0f)
    {
        options->color_lr = options->lr * 10.0f;
    }
    if (options->sh_degree < 0 || options->sh_degree > 3)
    {
        std::cerr << "Invalid --sh-degree (supported: 0-3)." << std::endl;
        return false;
    }
    return true;
}

bool BuildSharedGaussianBuffer(const torch::Tensor &positions, const torch::Tensor &colors,
                               const torch::Tensor &opacities, const torch::Tensor &scales,
                               const torch::Tensor &rotations, const torch::Tensor &sh, int sh_degree,
                               std::vector<float> *out)
{
    if (!positions.defined() || !colors.defined() || !opacities.defined() || !scales.defined() ||
        !rotations.defined())
        return false;
    if (positions.dim() != 2 || positions.size(1) != 3)
        return false;
    if (colors.dim() != 2 || colors.size(1) != 3)
        return false;
    if (opacities.dim() != 2 || opacities.size(1) != 1)
        return false;
    if (scales.dim() != 2 || scales.size(1) != 3)
        return false;
    if (rotations.dim() != 2 || rotations.size(1) != 4)
        return false;

    const int64_t count = positions.size(0);
    if (colors.size(0) != count || opacities.size(0) != count || scales.size(0) != count ||
        rotations.size(0) != count)
        return false;

    auto pos_cpu = positions.to(torch::kCPU).contiguous();
    auto col_cpu = colors.to(torch::kCPU).contiguous();
    auto opa_cpu = opacities.to(torch::kCPU).contiguous();
    auto sca_cpu = scales.to(torch::kCPU).contiguous();
    auto rot_cpu = rotations.to(torch::kCPU).contiguous();
    torch::Tensor sh_cpu;
    const int sh_coeffs = (sh_degree > 0) ? (sh_degree + 1) * (sh_degree + 1) : 0;
    if (sh_coeffs > 0)
    {
        if (!sh.defined() || sh.dim() != 3 || sh.size(1) != sh_coeffs || sh.size(2) != 3)
            return false;
        if (sh.size(0) != count)
            return false;
        sh_cpu = sh.to(torch::kCPU).contiguous();
    }

    const float *pos_ptr = pos_cpu.data_ptr<float>();
    const float *col_ptr = col_cpu.data_ptr<float>();
    const float *opa_ptr = opa_cpu.data_ptr<float>();
    const float *sca_ptr = sca_cpu.data_ptr<float>();
    const float *rot_ptr = rot_cpu.data_ptr<float>();

    const size_t stride = static_cast<size_t>(shared_gaussian::kSharedStrideFloats) + 7u +
                          static_cast<size_t>(sh_coeffs) * 3u;
    out->resize(static_cast<size_t>(count) * stride);
    float *dst = out->data();
    for (int64_t i = 0; i < count; ++i)
    {
        const size_t base = static_cast<size_t>(i) * stride;
        const size_t pos_base = static_cast<size_t>(i) * 3;
        dst[base + 0] = pos_ptr[pos_base + 0];
        dst[base + 1] = pos_ptr[pos_base + 1];
        dst[base + 2] = pos_ptr[pos_base + 2];
        dst[base + 3] = std::clamp(col_ptr[pos_base + 0], 0.0f, 1.0f);
        dst[base + 4] = std::clamp(col_ptr[pos_base + 1], 0.0f, 1.0f);
        dst[base + 5] = std::clamp(col_ptr[pos_base + 2], 0.0f, 1.0f);
        dst[base + 6] = std::clamp(opa_ptr[i], 0.0f, 1.0f);
        dst[base + 7] = (sca_ptr[pos_base + 0] + sca_ptr[pos_base + 1] + sca_ptr[pos_base + 2]) / 3.0f;
        dst[base + 8] = sca_ptr[pos_base + 0];
        dst[base + 9] = sca_ptr[pos_base + 1];
        dst[base + 10] = sca_ptr[pos_base + 2];
        const size_t rot_base = static_cast<size_t>(i) * 4;
        dst[base + 11] = rot_ptr[rot_base + 0];
        dst[base + 12] = rot_ptr[rot_base + 1];
        dst[base + 13] = rot_ptr[rot_base + 2];
        dst[base + 14] = rot_ptr[rot_base + 3];

        if (sh_coeffs > 0)
        {
            const float *sh_ptr = sh_cpu.data_ptr<float>() + (static_cast<size_t>(i) * sh_coeffs * 3u);
            float *dst_sh = dst + base + shared_gaussian::kSharedStrideFloats + 7u;
            std::memcpy(dst_sh, sh_ptr, static_cast<size_t>(sh_coeffs) * 3u * sizeof(float));
        }
    }
    return true;
}

// --- SSIM Implementation Start ---
torch::Tensor create_window(int window_size, int channel)
{
    auto t = torch::arange(window_size, torch::kFloat32) - window_size / 2;
    auto gauss = torch::exp(-(t * t) / (2 * 1.5 * 1.5));
    gauss = gauss / gauss.sum();

    auto window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0);
    return window_2d.expand({channel, 1, window_size, window_size}).contiguous();
}

struct SsimWindowCache
{
    int window_size = 0;
    int channel = 0;
    c10::Device device = c10::Device(c10::kCPU);
    torch::Tensor window;
};

static SsimWindowCache g_ssim_window_cache;

torch::Tensor ssim(const torch::Tensor &img1, const torch::Tensor &img2, int window_size = 5)
{
    const bool return_scalar = (img1.dim() == 3 && img2.dim() == 3);
    auto inp1 = (img1.dim() == 3) ? img1.unsqueeze(0) : img1;
    auto inp2 = (img2.dim() == 3) ? img2.unsqueeze(0) : img2;

    const int channel = static_cast<int>(inp1.size(1));
    if (!g_ssim_window_cache.window.defined() ||
        g_ssim_window_cache.window_size != window_size ||
        g_ssim_window_cache.channel != channel ||
        g_ssim_window_cache.device != inp1.device())
    {
        g_ssim_window_cache.window_size = window_size;
        g_ssim_window_cache.channel = channel;
        g_ssim_window_cache.device = inp1.device();
        g_ssim_window_cache.window = create_window(window_size, channel).to(inp1.device());
    }
    auto window = g_ssim_window_cache.window;

    auto mu1 = torch::conv2d(inp1, window, {}, 1, window_size / 2, 1, channel);
    auto mu2 = torch::conv2d(inp2, window, {}, 1, window_size / 2, 1, channel);

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = torch::conv2d(inp1 * inp1, window, {}, 1, window_size / 2, 1, channel) - mu1_sq;
    auto sigma2_sq = torch::conv2d(inp2 * inp2, window, {}, 1, window_size / 2, 1, channel) - mu2_sq;
    auto sigma12 = torch::conv2d(inp1 * inp2, window, {}, 1, window_size / 2, 1, channel) - mu1_mu2;

    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (return_scalar)
    {
        return ssim_map.mean();
    }
    return ssim_map.mean({1, 2, 3});
}

struct GaussianAvatar : torch::nn::Module
{
    std::shared_ptr<SMPLLayer> smpl;
    torch::Tensor g_scales, g_rots, g_opacities, g_colors, g_offsets, g_sh;
    torch::Tensor g_bary_coords, g_face_indices, faces_buffer;
    torch::Tensor v_template_cached, faces_cached;

    GaussianAvatar(const std::string &model_path)
    {
        smpl = std::make_shared<SMPLLayer>(model_path);
        register_module("smpl", smpl);
    }

    void init_gaussians(int num_gaussians, torch::Tensor faces_idx, float render_scale_modifier, int sh_degree)
    {
        auto device = smpl->v_template.device();
        v_template_cached = smpl->v_template.clone().detach();
        faces_cached = faces_idx.clone().detach();
        faces_buffer = faces_cached.to(device);
        int num_faces = faces_buffer.size(0);

        auto r1 = torch::rand({num_gaussians, 1}, device);
        auto r2 = torch::rand({num_gaussians, 1}, device);
        auto mask = (r1 + r2) > 1.0;
        r1.index_put_({mask}, 1.0 - r1.index({mask}));
        r2.index_put_({mask}, 1.0 - r2.index({mask}));

        auto w = 1.0 - r1 - r2;
        g_bary_coords = torch::cat({r1, r2, w}, 1);

        auto verts = smpl->v_template.to(device);
        auto face_a = verts.index_select(0, faces_buffer.index({torch::indexing::Slice(), 0}));
        auto face_b = verts.index_select(0, faces_buffer.index({torch::indexing::Slice(), 1}));
        auto face_c = verts.index_select(0, faces_buffer.index({torch::indexing::Slice(), 2}));
        auto ab = face_b - face_a;
        auto ac = face_c - face_a;
        auto cross = torch::cross(ab, ac, 1);
        auto face_area = 0.5f * torch::norm(cross, 2, 1);
        auto face_prob = face_area + 1e-12f;
        face_prob = face_prob / face_prob.sum();
        g_face_indices = torch::multinomial(face_prob, num_gaussians, true);

        register_buffer("g_face_indices", g_face_indices);
        register_buffer("g_bary_coords", g_bary_coords);
        auto face_scale = torch::sqrt(torch::clamp_min(face_area, 1e-12f));
        const float safe_scale_mod = std::max(render_scale_modifier, 1e-6f);
        const float normal_scale = 0.05f;

        auto z_axis = torch::tensor({0.0f, 0.0f, 1.0f}, torch::TensorOptions().device(device));
        auto z_expand = z_axis.unsqueeze(0).expand({num_faces, 3});
        auto face_normals = cross / torch::clamp_min(torch::norm(cross, 2, 1, true), 1e-8f);
        auto dot = torch::sum(face_normals * z_expand, 1).clamp(-1.0f, 1.0f);
        auto axis = torch::cross(z_expand, face_normals, 1);
        auto axis_norm = torch::norm(axis, 2, 1, true);
        auto axis_norm_clamped = torch::clamp_min(axis_norm, 1e-6f);
        auto axis_unit = axis / axis_norm_clamped;
        auto angle = torch::acos(dot);
        auto half = 0.5f * angle;
        auto qw = torch::cos(half);
        auto qxyz = axis_unit * torch::sin(half).unsqueeze(1);
        auto face_quats = torch::cat({qw.unsqueeze(1), qxyz}, 1);

        auto face_scale_sel = face_scale.index_select(0, g_face_indices);
        auto log_tan = torch::log(face_scale_sel / safe_scale_mod);
        auto log_norm = torch::log((face_scale_sel * normal_scale) / safe_scale_mod);
        g_scales = torch::stack({log_tan, log_tan, log_norm}, 1).clone().set_requires_grad(true);

        auto g_rots_init = face_quats.index_select(0, g_face_indices);
        g_rots = g_rots_init.detach().clone().set_requires_grad(true);
        g_opacities = torch::full({num_gaussians, 1}, 0.1, torch::requires_grad().device(device));
        g_colors = torch::full({num_gaussians, 3}, 0.5, torch::requires_grad().device(device));
        g_offsets = torch::zeros({num_gaussians, 3}, torch::requires_grad().device(device));
        if (sh_degree > 0)
        {
            const int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
            g_sh = torch::zeros({num_gaussians, sh_coeffs, 3}, torch::requires_grad().device(device));
        }
        else
        {
            g_sh = torch::zeros({0}, torch::TensorOptions().device(device));
        }

        register_parameter("g_scales", g_scales);
        register_parameter("g_rots", g_rots);
        register_parameter("g_opacities", g_opacities);
        register_parameter("g_colors", g_colors);
        register_parameter("g_offsets", g_offsets);
        register_parameter("g_sh", g_sh);
        register_buffer("v_template_cached", v_template_cached);
        register_buffer("faces_cached", faces_cached);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor betas, torch::Tensor pose, torch::Tensor trans)
    {
        auto smpl_out = smpl->forward(betas, pose, trans);
        auto verts_posed = smpl_out.vertices[0];
        auto selected_faces = faces_buffer.index_select(0, g_face_indices);

        auto A = verts_posed.index_select(0, selected_faces.index({torch::indexing::Slice(), 0}));
        auto B = verts_posed.index_select(0, selected_faces.index({torch::indexing::Slice(), 1}));
        auto C = verts_posed.index_select(0, selected_faces.index({torch::indexing::Slice(), 2}));

        auto verts_canon = v_template_cached.to(verts_posed.device());
        auto A_can = verts_canon.index_select(0, selected_faces.index({torch::indexing::Slice(), 0}));
        auto B_can = verts_canon.index_select(0, selected_faces.index({torch::indexing::Slice(), 1}));
        auto C_can = verts_canon.index_select(0, selected_faces.index({torch::indexing::Slice(), 2}));

        auto R_posed = ComputeTriFrames(A, B, C);
        auto R_canon = ComputeTriFrames(A_can, B_can, C_can);
        auto R_skin = torch::bmm(R_posed, R_canon.transpose(1, 2));
        auto posed_offsets = torch::bmm(R_skin, g_offsets.unsqueeze(2)).squeeze(2);

        auto u = g_bary_coords.index({torch::indexing::Slice(), 0}).unsqueeze(1);
        auto v = g_bary_coords.index({torch::indexing::Slice(), 1}).unsqueeze(1);
        auto w = g_bary_coords.index({torch::indexing::Slice(), 2}).unsqueeze(1);

        auto skinned_pos = u * A + v * B + w * C;
        auto final_pos = skinned_pos + posed_offsets;

        auto q_skin = MatrixToQuat(R_skin);
        auto final_rot = QuatMultiply(q_skin, g_rots);

        return {final_pos, final_rot};
    }
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: gaussian_train --jsonl <path> [--smpl <path>] [--num-gaussians <int>]"
                     " [--epochs <int>] [--lr <float>] [--output-dir <path>]"
                     " [--lr-decay-epoch <int>] [--lr-decay-multiplier <float>]"
                     " [--lr-min-multiplier <float>]"
                     " [--train-dir <path>] [--viewer-export-dir <path>]"
                     " [--scale-reg <float>] [--scale-max <float>]"
                     " [--rot-lr <float>] [--offset-lr <float>]"
                     " [--offset-reg <float>] [--pose-reg <float>] [--pose-lr <float>] [--alpha-loss <float>]"
                     " [--lambda-dssim <float>] [--gan-weight <float>]"
                     " [--color-lr <float>] [--opacity-lr <float>]"
                     " [--sh-degree <int>]"
                     " [--densify-every <int>] [--densify-max <int>] [--densify-max-clones <int>]"
                     " [--densify-scale <float>]"
                     " [--densify-split-scale <float>] [--densify-split-offset <float>]"
                     " [--densify-min-grad <float>] [--densify-grow-grad <float>]"
                     " [--densify-prune-opacity <float>]"
                     " [--densify-prune-max <int>] [--densify-reset-opacity <float>]"
                     " [--densify-stop-epoch <int>]"
                     " [--viewer|--headless]"
                     " [--viewer-every <int>] [--viewer-shm <name>] [--viewer-pose-shm <name>]"
                     " [--viewer-bind-shm <name>] [--viewer-stream-poses]\n";
        return -1;
    }
    if (!torch::cuda::is_available())
    {
        std::cerr << "CUDA Required for Rasterizer!" << std::endl;
        return -1;
    }

    TrainOptions options;
    if (!ParseTrainArgs(argc, argv, &options))
    {
        return -1;
    }
    const std::string &jsonl_path = options.jsonl_path;
    const std::string &smpl_model_path = options.smpl_model_path;
    const int num_gaussians = options.num_gaussians;
    const int epochs = options.epochs;
    const float lr = options.lr;
    const int lr_decay_epoch = options.lr_decay_epoch;
    const float lr_decay_multiplier = options.lr_decay_multiplier;
    const float lr_min_multiplier = options.lr_min_multiplier;
    const float gan_weight = options.gan_weight;
    const std::string &output_dir = options.output_dir;
    const std::string &viewer_export_dir = options.viewer_export_dir;
    const float scale_reg_weight = options.scale_reg_weight;
    const float scale_lr = (options.scale_lr < 0.0f) ? lr : options.scale_lr;
    const float scale_max_value = options.scale_max_value;
    const float rot_lr = (options.rot_lr < 0.0f) ? lr : options.rot_lr;
    const float offset_lr = (options.offset_lr < 0.0f) ? lr : options.offset_lr;
    const float offset_reg_weight = options.offset_reg_weight;
    const float pose_reg_weight = options.pose_reg_weight;
    const float pose_lr = options.pose_lr;
    const float color_lr = options.color_lr;
    const float opacity_lr = (options.opacity_lr < 0.0f) ? lr : options.opacity_lr;
    const int sh_degree = options.sh_degree;
    const int viewer_every = std::max(1, options.viewer_every);
    const bool viewer_stream_poses = options.viewer_stream_poses;
    const int densify_every = std::max(1, options.densify_every);
    const float outside_mask_weight = 0.1f;
    const float alpha_loss_weight = options.alpha_loss_weight;
    const float lambda_dssim = options.lambda_dssim;
    const float render_scale_modifier = 1.0f;
    const float render_threshold = 3.0f / 255.0f;
    const int tile_size = 64;
    const float tile_outlier_mult = 3.0f;
    const float tile_mask_min = 0.01f;
    const float tile_median_min = 1e-4f;
    const int batch_size = 8;
    const float safe_scale_mod = std::max(render_scale_modifier, 1e-6f);
    const float scale_cap = (scale_max_value > 0.0f) ? (scale_max_value / safe_scale_mod) : -1.0f;
    auto capped_scales = [&](const torch::Tensor &log_scales)
    {
        auto scales = torch::exp(log_scales);
        if (scale_cap > 0.0f)
        {
            scales = torch::clamp(scales, 0.0f, scale_cap);
        }
        return scales;
    };

    std::ifstream input(jsonl_path);
    if (!input.is_open())
    {
        std::cerr << "Failed to open " << jsonl_path << std::endl;
        return -1;
    }

    std::vector<TrainSample> samples;
    std::string line;
    while (std::getline(input, line))
    {
        TrainSample sample;
        if (ParseTrainSample(line, &sample))
        {
            samples.push_back(std::move(sample));
        }
    }
    if (samples.empty())
    {
        std::cerr << "No training samples found in " << jsonl_path << std::endl;
        return -1;
    }

    auto device = torch::kCUDA;
    auto storage_device = torch::kCPU;
    std::vector<CachedSampleData> cached;
    cached.resize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto &sample = samples[i];
        CachedSampleData entry;
        cv::Mat crop = cv::imread(sample.crop_path);
        if (crop.empty())
        {
            cached[i] = std::move(entry);
            continue;
        }
        const std::string matte_path = DeriveMattePath(sample.crop_path);
        cv::Mat matte = cv::imread(matte_path, cv::IMREAD_UNCHANGED);
        if (matte.empty())
        {
            cached[i] = std::move(entry);
            continue;
        }

        auto target = LoadImageTensor(crop, storage_device);
        if (!target.defined() || target.dim() != 3 || target.size(0) != 3)
        {
            cached[i] = std::move(entry);
            continue;
        }
        const int H = static_cast<int>(target.size(1));
        const int W = static_cast<int>(target.size(2));
        if (H <= 0 || W <= 0)
        {
            cached[i] = std::move(entry);
            continue;
        }

        auto matte_mask = LoadMatteMaskTensor(matte, W, H, storage_device);
        if (!matte_mask.defined())
        {
            cached[i] = std::move(entry);
            continue;
        }

        entry.target = target;
        entry.matte_mask = matte_mask;
        entry.crop_bgr = crop;
        entry.valid = true;
        cached[i] = std::move(entry);
    }

    GaussianAvatar avatar(smpl_model_path);
    avatar.to(device);

    std::ifstream smpl_in(smpl_model_path, std::ios::binary);
    std::vector<char> f_bytes((std::istreambuf_iterator<char>(smpl_in)), (std::istreambuf_iterator<char>()));
    auto dict = torch::pickle_load(f_bytes).toGenericDict();
    torch::Tensor faces = dict.at("faces").toTensor().to(torch::kLong).to(device);
    avatar.init_gaussians(num_gaussians, faces, render_scale_modifier, options.sh_degree);
    auto pose_offsets = torch::zeros({static_cast<int64_t>(samples.size()), 24, 3},
                                     torch::TensorOptions().device(device).dtype(torch::kFloat).requires_grad(true));
    auto global_rot_offsets = torch::zeros({static_cast<int64_t>(samples.size()), 3},
                                           torch::TensorOptions().device(device).dtype(torch::kFloat).requires_grad(true));
    auto global_trans_offsets = torch::zeros({static_cast<int64_t>(samples.size()), 3},
                                             torch::TensorOptions().device(device).dtype(torch::kFloat).requires_grad(true));
    const auto avg_betas = ComputeAverageBetas(samples);
    torch::Tensor canonical_betas;
    if (!avg_betas.empty())
    {
        canonical_betas = torch::from_blob(const_cast<float *>(avg_betas.data()),
                                           {1, static_cast<int64_t>(avg_betas.size())},
                                           torch::kFloat)
                              .clone()
                              .to(device);
    }
    if (!canonical_betas.defined())
    {
        std::cerr << "Failed to initialize canonical betas." << std::endl;
        return -1;
    }
    auto canonical_pose = torch::zeros({1, 24, 3}, canonical_betas.options());
    auto canonical_trans = torch::zeros({1, 3}, canonical_betas.options());

    const bool use_sh = sh_degree > 0;
    auto sh = use_sh ? avatar.g_sh : torch::zeros({0}, torch::TensorOptions().device(device));
    auto cam_pos = torch::zeros({3}, torch::TensorOptions().device(device));

    const uint32_t shared_stride = shared_gaussian::kSharedStrideFloats + 7u +
                                   (use_sh ? static_cast<uint32_t>((sh_degree + 1) * (sh_degree + 1) * 3) : 0u);
    shared_gaussian::SharedGaussianWriter shared_writer;
    bool publish_viewer = options.enable_viewer;
    const std::string bind_shm_name = options.viewer_bind_shm_name.empty()
                                          ? (options.viewer_shm_name + "_bind")
                                          : options.viewer_bind_shm_name;
    shared_gaussian::SharedBindWriter bind_writer;
    if (publish_viewer)
    {
        if (!shared_writer.Init(options.viewer_shm_name, static_cast<uint32_t>(num_gaussians),
                                shared_stride, static_cast<uint32_t>(sh_degree),
                                render_scale_modifier))
        {
            std::cerr << "Failed to open shared memory mapping: " << options.viewer_shm_name << std::endl;
            publish_viewer = false;
        }
        if (publish_viewer)
        {
            const uint32_t betas_count = static_cast<uint32_t>(canonical_betas.numel());
            if (!bind_writer.Init(bind_shm_name, static_cast<uint32_t>(num_gaussians),
                                  shared_gaussian::kSharedBindStrideFloats, betas_count))
            {
                std::cerr << "Failed to open bind shared memory mapping: " << bind_shm_name << std::endl;
            }
        }
    }
    std::vector<float> shared_buffer;
    uint64_t shared_frame = 0;
    std::vector<float> bind_buffer;
    std::vector<float> bind_betas_buffer;

    DensificationConfig densify_cfg;
    densify_cfg.every = densify_every;
    densify_cfg.max_splits = std::max(0, options.densify_max_splits);
    densify_cfg.max_clones = std::max(0, options.densify_max_clones);
    densify_cfg.scale_threshold = (options.densify_scale_threshold > 0.0f)
                                      ? options.densify_scale_threshold
                                      : 0.0f;
    densify_cfg.split_scale_factor = options.densify_split_scale;
    densify_cfg.split_offset_scale = options.densify_split_offset;
    densify_cfg.min_grad_norm = options.densify_min_grad;
    densify_cfg.grow_grad_threshold = options.densify_grow_grad;
    densify_cfg.prune_opacity_threshold = options.densify_prune_opacity;
    densify_cfg.prune_max = std::max(0, options.densify_prune_max);
    densify_cfg.reset_opacity = options.densify_reset_opacity;
    const int densify_stop_epoch = options.densify_stop_epoch;
    DensificationState densify_state;
    int64_t global_step = 0;

    std::vector<torch::Tensor> rot_params = {avatar.g_rots};
    std::vector<torch::Tensor> offset_params = {avatar.g_offsets};
    std::vector<torch::Tensor> scale_params = {avatar.g_scales};
    std::vector<torch::Tensor> color_params = {use_sh ? avatar.g_sh : avatar.g_colors};
    std::vector<torch::Tensor> opacity_params = {avatar.g_opacities};
    std::vector<torch::Tensor> pose_params = {pose_offsets, global_rot_offsets, global_trans_offsets};
    float lr_multiplier = 1.0f;
    float pose_lr_multiplier = 1.0f;
    bool lr_decay_applied = false;
    torch::optim::Adam optimizer(rot_params, torch::optim::AdamOptions(rot_lr * lr_multiplier));
    optimizer.add_param_group({offset_params});
    auto &offset_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(offset_group.options()).lr(offset_lr * lr_multiplier);
    optimizer.add_param_group({scale_params});
    auto &scale_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(scale_group.options()).lr(scale_lr * lr_multiplier);
    optimizer.add_param_group({color_params});
    auto &color_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(color_group.options()).lr(color_lr * lr_multiplier);
    optimizer.add_param_group({opacity_params});
    auto &opacity_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(opacity_group.options()).lr(opacity_lr * lr_multiplier);
    optimizer.add_param_group({pose_params});
    auto &pose_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(pose_group.options()).lr(pose_lr * lr_multiplier);

    auto apply_lr_multiplier = [&](float pose_multiplier)
    {
        pose_lr_multiplier = pose_multiplier;
        // static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options()).lr(rot_lr * lr_multiplier);
        // static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[1].options()).lr(offset_lr * lr_multiplier);
        // static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[2].options()).lr(scale_lr * lr_multiplier);
        // static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[3].options()).lr(color_lr * lr_multiplier);
        // static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[4].options()).lr(opacity_lr * lr_multiplier);
        static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[5].options()).lr(pose_lr * pose_lr_multiplier);
    };

    auto rebuild_optimizer = [&]()
    {
        rot_params = {avatar.g_rots};
        offset_params = {avatar.g_offsets};
        scale_params = {avatar.g_scales};
        color_params = {use_sh ? avatar.g_sh : avatar.g_colors};
        opacity_params = {avatar.g_opacities};
        pose_params = {pose_offsets, global_rot_offsets, global_trans_offsets};
        optimizer = torch::optim::Adam(rot_params, torch::optim::AdamOptions(rot_lr * lr_multiplier));
        optimizer.add_param_group({offset_params});
        auto &new_offset_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_offset_group.options()).lr(offset_lr * lr_multiplier);
        optimizer.add_param_group({scale_params});
        auto &new_scale_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_scale_group.options()).lr(scale_lr * lr_multiplier);
        optimizer.add_param_group({color_params});
        auto &new_color_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_color_group.options()).lr(color_lr * lr_multiplier);
        optimizer.add_param_group({opacity_params});
        auto &new_opacity_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_opacity_group.options()).lr(opacity_lr * lr_multiplier);
        optimizer.add_param_group({pose_params});
        auto &new_pose_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_pose_group.options()).lr(pose_lr * pose_lr_multiplier);
    };

    std::filesystem::path out_dir_path(output_dir);
    std::error_code out_ec;
    std::filesystem::create_directories(out_dir_path, out_ec);
    if (out_ec)
    {
        std::cerr << "Failed to create output dir: " << output_dir << std::endl;
        return -1;
    }

    std::filesystem::path viewer_out_path;
    if (!viewer_export_dir.empty())
    {
        viewer_out_path = std::filesystem::path(viewer_export_dir);
        std::error_code viewer_ec;
        std::filesystem::create_directories(viewer_out_path, viewer_ec);
        if (viewer_ec)
        {
            std::cerr << "Failed to create viewer export dir: " << viewer_export_dir << std::endl;
            return -1;
        }
    }

    std::mt19937 rng(static_cast<unsigned int>(std::random_device{}()));
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (lr_decay_epoch >= 0 && epoch >= lr_decay_epoch)
        {
            if (!lr_decay_applied)
            {
                lr_decay_applied = true;
            }
            const float next_pose_multiplier = std::max(pose_lr_multiplier * lr_decay_multiplier, lr_min_multiplier);
            apply_lr_multiplier(next_pose_multiplier);
            std::cout << "Applied pose LR decay at epoch " << epoch
                      << ": pose multiplier " << pose_lr_multiplier << std::endl;
        }
        {
            const auto rot_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options()).lr();
            const auto offset_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[1].options()).lr();
            const auto scale_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[2].options()).lr();
            const auto color_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[3].options()).lr();
            const auto opacity_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[4].options()).lr();
            const auto pose_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[5].options()).lr();
            std::cout << "Epoch " << epoch << " LRs: "
                      << "rot=" << rot_lr_eff
                      << " offset=" << offset_lr_eff
                      << " scale=" << scale_lr_eff
                      << " color=" << color_lr_eff
                      << " opacity=" << opacity_lr_eff
                      << " pose=" << pose_lr_eff
                      << std::endl;
        }
        std::vector<size_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        int dropped_since_log = 0;
        for (size_t batch_start = 0; batch_start < indices.size(); batch_start += batch_size)
        {

            const size_t batch_end = std::min(indices.size(), batch_start + batch_size);
            int batch_step = static_cast<int>(batch_start / batch_size);
            optimizer.zero_grad();

            std::vector<torch::Tensor> recon_losses;
            std::vector<torch::Tensor> outside_losses;
            std::vector<torch::Tensor> alpha_losses;
            std::vector<torch::Tensor> ssim_losses;
            std::vector<float> recon_values;
            std::vector<float> total_values;
            std::vector<torch::Tensor> batch_images_comp;
            std::vector<torch::Tensor> batch_targets_comp;
            std::vector<torch::Tensor> batch_masks_comp;
            std::vector<torch::Tensor> batch_alphas_comp;
            int skipped_malformed = 0;
            int skipped_mask = 0;
            int64_t tiles_total = 0;
            int64_t tiles_mask_valid = 0;
            int64_t tiles_outlier_valid = 0;
            int64_t tiles_final = 0;
            float tiles_median_sum = 0.0f;
            int tiles_median_count = 0;

            bool has_batched_transfers = false;
            torch::Tensor gpu_targets_block;
            torch::Tensor gpu_mattes_block;
            {
                bool can_stack = true;
                int target_h = -1;
                int target_w = -1;
                std::vector<torch::Tensor> batch_targets_cpu;
                std::vector<torch::Tensor> batch_mattes_cpu;
                batch_targets_cpu.reserve(batch_end - batch_start);
                batch_mattes_cpu.reserve(batch_end - batch_start);
                for (size_t idx = batch_start; idx < batch_end; ++idx)
                {
                    const auto &cached_entry = cached[indices[idx]];
                    if (!cached_entry.valid || !cached_entry.target.defined() || !cached_entry.matte_mask.defined())
                    {
                        can_stack = false;
                        break;
                    }
                    if (cached_entry.target.dim() != 3 || cached_entry.matte_mask.dim() != 3)
                    {
                        can_stack = false;
                        break;
                    }
                    const int H = static_cast<int>(cached_entry.target.size(1));
                    const int W = static_cast<int>(cached_entry.target.size(2));
                    if (target_h < 0)
                    {
                        target_h = H;
                        target_w = W;
                    }
                    else if (target_h != H || target_w != W)
                    {
                        can_stack = false;
                        break;
                    }
                    if (cached_entry.matte_mask.size(1) != H || cached_entry.matte_mask.size(2) != W)
                    {
                        can_stack = false;
                        break;
                    }
                    batch_targets_cpu.push_back(cached_entry.target);
                    batch_mattes_cpu.push_back(cached_entry.matte_mask);
                }
                if (can_stack && !batch_targets_cpu.empty())
                {
                    gpu_targets_block = torch::stack(batch_targets_cpu).to(device, true);
                    gpu_mattes_block = torch::stack(batch_mattes_cpu).to(device, true);
                    has_batched_transfers = true;
                }
            }

            for (size_t idx = batch_start; idx < batch_end; ++idx)
            {
                const auto &sample = samples[indices[idx]];
                const auto &cached_entry = cached[indices[idx]];
                if (!cached_entry.valid)
                {
                    skipped_malformed++;
                    continue;
                }
                torch::Tensor target;
                torch::Tensor matte_mask;
                if (has_batched_transfers)
                {
                    const size_t batch_offset = idx - batch_start;
                    target = gpu_targets_block[batch_offset];
                    matte_mask = gpu_mattes_block[batch_offset];
                }
                else
                {
                    target = cached_entry.target.to(device, true);
                    matte_mask = cached_entry.matte_mask.to(device, true);
                }
                const int H = static_cast<int>(target.size(1));
                const int W = static_cast<int>(target.size(2));
                if (H <= 0 || W <= 0)
                {
                    skipped_malformed++;
                    continue;
                }

                SmplResult res;
                res.pose = sample.pose;
                res.shape = sample.betas;
                res.camera = sample.cam;

                auto pose = PoseToAxisAngle(res).to(device);
                auto pose_offset = pose_offsets.index({static_cast<int64_t>(indices[idx])}).unsqueeze(0);
                pose = pose + pose_offset;
                auto global_rot = global_rot_offsets.index({static_cast<int64_t>(indices[idx])}).unsqueeze(0);
                pose.index_put_({0, 0, torch::indexing::Slice()},
                                pose.index({0, 0, torch::indexing::Slice()}) + global_rot.squeeze(0));
                cv::Vec3f trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                         sample.crop_size, sample.focal_length,
                                                         static_cast<float>(sample.img_w),
                                                         static_cast<float>(sample.img_h));
                auto trans = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                           torch::TensorOptions().device(device).dtype(torch::kFloat));
                auto trans_offset = global_trans_offsets.index({static_cast<int64_t>(indices[idx])});
                trans = trans + trans_offset;

                torch::Tensor means3D;
                torch::Tensor current_rots;
                std::tie(means3D, current_rots) =
                    avatar.forward(canonical_betas, pose, torch::zeros({1, 3}, canonical_betas.options()));
                torch::Tensor current_sh = use_sh ? avatar.g_sh : torch::zeros({0}, avatar.g_colors.options());
                if (use_sh && sh_degree > 0)
                {
                    current_sh = RotateSH(current_sh, current_rots);
                }
                auto y_scale = torch::tensor({1.0f, sample.y_sign, 1.0f},
                                             torch::TensorOptions().device(device).dtype(torch::kFloat));
                means3D = means3D * y_scale;
                means3D = means3D + trans;

                const int full_w = sample.img_w;
                const int full_h = sample.img_h;
                float f_render = sample.focal_length;
                torch::Tensor view_mat;
                torch::Tensor proj_mat;
                float tan_fovx = 0.0f;
                float tan_fovy = 0.0f;
                const int render_w = W;
                const int render_h = H;
                const float full_cx = static_cast<float>(full_w) * 0.5f;
                const float full_cy = static_cast<float>(full_h) * 0.5f;
                float x0 = sample.crop_x0;
                float y0 = sample.crop_y0;
                if (sample.crop_w <= 0.0f || sample.crop_h <= 0.0f)
                {
                    x0 = sample.crop_cx - static_cast<float>(render_w) * 0.5f;
                    y0 = sample.crop_cy - static_cast<float>(render_h) * 0.5f;
                }
                const float cx_crop = full_cx - x0;
                const float cy_crop = full_cy - y0;
                std::tie(view_mat, proj_mat, tan_fovx, tan_fovy) =
                    BuildProjection(f_render, render_w, render_h, cx_crop, cy_crop, device);

                auto colors = use_sh ? torch::zeros({0}, avatar.g_colors.options()) : avatar.g_colors;
                // Supersample then resolve to preserve high-frequency edges.
                const float ss_factor = 2.0f;
                const int ss_h = static_cast<int>(render_h * ss_factor);
                const int ss_w = static_cast<int>(render_w * ss_factor);

                auto outputs = GaussianRasterizer::apply(
                    means3D,
                    colors,
                    avatar.g_opacities,
                    capped_scales(avatar.g_scales),
                    current_rots,
                    render_scale_modifier,
                    view_mat,
                    proj_mat,
                    tan_fovx,
                    tan_fovy,
                    ss_h,
                    ss_w,
                    current_sh,
                    use_sh ? sh_degree : 0,
                    cam_pos,
                    false);
                auto raw_image = outputs[0];
                auto raw_alpha = outputs[1];

                namespace F = torch::nn::functional;
                auto image = F::avg_pool2d(
                                 raw_image.unsqueeze(0),
                                 F::AvgPool2dFuncOptions(2).stride(2))
                                 .squeeze(0);
                auto alpha = F::avg_pool2d(
                                 raw_alpha.unsqueeze(0),
                                 F::AvgPool2dFuncOptions(2).stride(2))
                                 .squeeze(0);
                if (!image.defined() || image.dim() != 3 || image.size(0) != 3 ||
                    image.size(1) != H || image.size(2) != W ||
                    !alpha.defined() || alpha.dim() != 3 || alpha.size(0) != 1 ||
                    alpha.size(1) != H || alpha.size(2) != W)
                {
                    skipped_malformed++;
                    continue;
                }
                if (!IsMaskCoverageValidTensor(image, matte_mask, 0.3f, render_threshold))
                {
                    skipped_mask++;
                    continue;
                }

                using torch::indexing::Slice;
                const int tile_h = (H / tile_size) * tile_size;
                const int tile_w = (W / tile_size) * tile_size;
                if (tile_h < tile_size || tile_w < tile_size)
                {
                    skipped_malformed++;
                    continue;
                }

                auto image_crop = image.index({Slice(), Slice(0, tile_h), Slice(0, tile_w)});
                auto target_crop = target.index({Slice(), Slice(0, tile_h), Slice(0, tile_w)});
                auto matte_crop = matte_mask.index({Slice(), Slice(0, tile_h), Slice(0, tile_w)});
                auto alpha_crop = alpha.index({Slice(), Slice(0, tile_h), Slice(0, tile_w)});

                auto bg_color = torch::rand({3, 1, 1}, image_crop.options());
                auto image_comp = image_crop + bg_color * (1.0f - alpha_crop);
                auto target_comp = target_crop * matte_crop + bg_color * (1.0f - matte_crop);

                // Use avg_pool2d to compute mean loss per tile without materializing tiles.
                auto diff_map = torch::abs(image_comp - target_comp);
                auto diff_map_mean = diff_map.mean(0, true);
                auto loss_per_tile_2d = F::avg_pool2d(
                    diff_map_mean,
                    F::AvgPool2dFuncOptions(tile_size).stride(tile_size));
                auto loss_per_tile = loss_per_tile_2d.view({-1});

                auto mask_avg_2d = F::avg_pool2d(
                    matte_crop,
                    F::AvgPool2dFuncOptions(tile_size).stride(tile_size));
                auto mask_sums = mask_avg_2d.view({-1}) * static_cast<float>(tile_size * tile_size);
                auto valid_tile_mask = mask_sums > tile_mask_min;
                auto valid_losses = loss_per_tile.index({valid_tile_mask});
                float median_tile_loss = 0.0f;
                if (valid_losses.defined() && valid_losses.numel() > 0)
                {
                    median_tile_loss = torch::median(valid_losses).item<float>();
                }
                auto outlier_tile_mask = torch::ones_like(valid_tile_mask);
                if (median_tile_loss >= tile_median_min)
                {
                    const float outlier_thresh = median_tile_loss * tile_outlier_mult;
                    outlier_tile_mask = loss_per_tile <= outlier_thresh;
                }
                auto final_mask = valid_tile_mask & outlier_tile_mask;
                const int64_t valid_tiles = final_mask.sum().item<int64_t>();
                tiles_total += loss_per_tile.size(0);
                tiles_mask_valid += valid_tile_mask.sum().item<int64_t>();
                tiles_outlier_valid += outlier_tile_mask.sum().item<int64_t>();
                tiles_final += valid_tiles;
                tiles_median_sum += median_tile_loss;
                tiles_median_count++;
                if (valid_tiles <= 0)
                {
                    skipped_malformed++;
                    continue;
                }

                auto recon_loss = loss_per_tile.index({final_mask}).mean();
                auto loss_value = recon_loss.item<float>();
                if (!std::isfinite(loss_value))
                {
                    skipped_malformed++;
                    continue;
                }

                recon_losses.push_back(recon_loss);
                recon_values.push_back(loss_value);
                batch_images_comp.push_back(image_comp);
                batch_targets_comp.push_back(target_comp);
                batch_masks_comp.push_back(matte_crop);
                batch_alphas_comp.push_back(alpha_crop);
            }

            if (!batch_images_comp.empty())
            {
                bool can_stack_comp = true;
                int comp_h = -1;
                int comp_w = -1;
                for (const auto &comp : batch_images_comp)
                {
                    if (!comp.defined() || comp.dim() != 3)
                    {
                        can_stack_comp = false;
                        break;
                    }
                    const int h = static_cast<int>(comp.size(1));
                    const int w = static_cast<int>(comp.size(2));
                    if (comp_h < 0)
                    {
                        comp_h = h;
                        comp_w = w;
                    }
                    else if (comp_h != h || comp_w != w)
                    {
                        can_stack_comp = false;
                        break;
                    }
                }
                if (can_stack_comp)
                {
                    auto img_batch = torch::stack(batch_images_comp);
                    auto tgt_batch = torch::stack(batch_targets_comp);
                    auto msk_batch = torch::stack(batch_masks_comp);
                    auto alp_batch = torch::stack(batch_alphas_comp);

                    auto ssim_vals = ssim(img_batch, tgt_batch);
                    auto d_ssim_losses_t = 1.0f - ssim_vals;
                    auto outside_losses_t = (img_batch * (1.0f - msk_batch)).mean({1, 2, 3});
                    auto alpha_losses_t = torch::l1_loss(alp_batch, msk_batch, torch::Reduction::None)
                                              .mean({1, 2, 3});

                    std::vector<torch::Tensor> filtered_recon_losses;
                    std::vector<torch::Tensor> filtered_outside_losses;
                    std::vector<torch::Tensor> filtered_alpha_losses;
                    std::vector<torch::Tensor> filtered_ssim_losses;
                    std::vector<float> filtered_recon_values;
                    std::vector<float> filtered_total_values;

                    const size_t batch_count = batch_images_comp.size();
                    filtered_recon_losses.reserve(batch_count);
                    filtered_outside_losses.reserve(batch_count);
                    filtered_alpha_losses.reserve(batch_count);
                    filtered_ssim_losses.reserve(batch_count);
                    filtered_recon_values.reserve(batch_count);
                    filtered_total_values.reserve(batch_count);

                    for (size_t i = 0; i < batch_count; ++i)
                    {
                        const float outside_value = outside_losses_t[i].item<float>();
                        const float alpha_value = alpha_losses_t[i].item<float>();
                        const float ssim_value = d_ssim_losses_t[i].item<float>();
                        if (!std::isfinite(outside_value) || !std::isfinite(alpha_value) || !std::isfinite(ssim_value))
                        {
                            skipped_malformed++;
                            continue;
                        }

                        filtered_recon_losses.push_back(recon_losses[i]);
                        filtered_outside_losses.push_back(outside_losses_t[i]);
                        filtered_alpha_losses.push_back(alpha_losses_t[i]);
                        filtered_ssim_losses.push_back(d_ssim_losses_t[i]);
                        filtered_recon_values.push_back(recon_values[i]);
                        filtered_total_values.push_back(recon_values[i] +
                                                        outside_mask_weight * outside_value +
                                                        alpha_loss_weight * alpha_value);
                    }

                    recon_losses = std::move(filtered_recon_losses);
                    outside_losses = std::move(filtered_outside_losses);
                    alpha_losses = std::move(filtered_alpha_losses);
                    ssim_losses = std::move(filtered_ssim_losses);
                    recon_values = std::move(filtered_recon_values);
                    total_values = std::move(filtered_total_values);
                }
                else
                {
                    std::vector<torch::Tensor> filtered_recon_losses;
                    std::vector<torch::Tensor> filtered_outside_losses;
                    std::vector<torch::Tensor> filtered_alpha_losses;
                    std::vector<torch::Tensor> filtered_ssim_losses;
                    std::vector<float> filtered_recon_values;
                    std::vector<float> filtered_total_values;

                    const size_t batch_count = batch_images_comp.size();
                    filtered_recon_losses.reserve(batch_count);
                    filtered_outside_losses.reserve(batch_count);
                    filtered_alpha_losses.reserve(batch_count);
                    filtered_ssim_losses.reserve(batch_count);
                    filtered_recon_values.reserve(batch_count);
                    filtered_total_values.reserve(batch_count);

                    for (size_t i = 0; i < batch_count; ++i)
                    {
                        auto ssim_value = ssim(batch_images_comp[i], batch_targets_comp[i]);
                        auto d_ssim_loss = 1.0f - ssim_value;
                        auto outside_loss = torch::mean(batch_images_comp[i] * (1.0f - batch_masks_comp[i]));
                        auto alpha_loss = torch::l1_loss(batch_alphas_comp[i], batch_masks_comp[i]);

                        const float outside_value = outside_loss.item<float>();
                        const float alpha_value = alpha_loss.item<float>();
                        const float ssim_value_f = d_ssim_loss.item<float>();
                        if (!std::isfinite(outside_value) || !std::isfinite(alpha_value) || !std::isfinite(ssim_value_f))
                        {
                            skipped_malformed++;
                            continue;
                        }

                        filtered_recon_losses.push_back(recon_losses[i]);
                        filtered_outside_losses.push_back(outside_loss);
                        filtered_alpha_losses.push_back(alpha_loss);
                        filtered_ssim_losses.push_back(d_ssim_loss);
                        filtered_recon_values.push_back(recon_values[i]);
                        filtered_total_values.push_back(recon_values[i] +
                                                        outside_mask_weight * outside_value +
                                                        alpha_loss_weight * alpha_value);
                    }

                    recon_losses = std::move(filtered_recon_losses);
                    outside_losses = std::move(filtered_outside_losses);
                    alpha_losses = std::move(filtered_alpha_losses);
                    ssim_losses = std::move(filtered_ssim_losses);
                    recon_values = std::move(filtered_recon_values);
                    total_values = std::move(filtered_total_values);
                }
            }

            if (recon_losses.empty())
            {
                dropped_since_log += skipped_malformed + skipped_mask;
                if ((batch_step + 1) % 10 == 0)
                {
                    std::cout << "Dropped samples in last 10 batches: "
                              << dropped_since_log << std::endl;
                    if (tiles_total > 0)
                    {
                        const float avg_median = (tiles_median_count > 0)
                                                     ? (tiles_median_sum / static_cast<float>(tiles_median_count))
                                                     : 0.0f;
                        std::cout << "Tile stats: total=" << tiles_total
                                  << " mask_valid=" << tiles_mask_valid
                                  << " outlier_valid=" << tiles_outlier_valid
                                  << " final=" << tiles_final
                                  << " avg_median=" << avg_median << std::endl;
                    }
                    dropped_since_log = 0;
                }
                continue;
            }

            const float outlier_percentile = 1.0f;
            float outlier_threshold = std::numeric_limits<float>::infinity();
            if (!total_values.empty())
            {
                std::vector<float> sorted_values = total_values;
                std::sort(sorted_values.begin(), sorted_values.end());
                const size_t n = sorted_values.size();
                const size_t idx = std::min(n - 1, static_cast<size_t>(std::floor(outlier_percentile * (n - 1))));
                outlier_threshold = sorted_values[idx];
            }

            torch::Tensor recon_sum = torch::zeros({}, torch::TensorOptions().device(device));
            torch::Tensor ssim_sum = torch::zeros({}, torch::TensorOptions().device(device));
            int inlier_count = 0;
            int skipped_outlier = 0;
            int last_inlier_idx = -1;
            int best_inlier_idx = -1;
            float best_inlier_loss = std::numeric_limits<float>::infinity();
            std::vector<int> inlier_indices;
            inlier_indices.reserve(recon_losses.size());
            for (size_t i = 0; i < recon_losses.size(); ++i)
            {
                if (total_values[i] > outlier_threshold)
                {
                    skipped_outlier++;
                    continue;
                }
                recon_sum = recon_sum + recon_losses[i] +
                            outside_mask_weight * outside_losses[i] +
                            alpha_loss_weight * alpha_losses[i];
                ssim_sum = ssim_sum + ssim_losses[i];
                inlier_count++;
                inlier_indices.push_back(static_cast<int>(i));
                last_inlier_idx = static_cast<int>(i);
                if (total_values[i] < best_inlier_loss)
                {
                    best_inlier_loss = total_values[i];
                    best_inlier_idx = static_cast<int>(i);
                }
            }
            if (inlier_count == 0)
            {
                dropped_since_log += skipped_malformed + skipped_mask + skipped_outlier;
                if ((batch_step + 1) % 10 == 0)
                {
                    std::cout << "Dropped samples in last 10 batches: "
                              << dropped_since_log << std::endl;
                    dropped_since_log = 0;
                }
                continue;
            }

            auto recon_loss = recon_sum / static_cast<float>(inlier_count);
            auto avg_ssim_loss = ssim_sum / static_cast<float>(inlier_count);

            auto current_scales = torch::exp(avatar.g_scales) * render_scale_modifier;
            auto all_scales_sq = current_scales.pow(2).sum(1);
            auto scale_reg = torch::mean(all_scales_sq);
            auto offset_reg = torch::mean(torch::abs(avatar.g_offsets));

            auto loss = (1.0f - lambda_dssim) * recon_loss +
                        lambda_dssim * avg_ssim_loss +
                        offset_reg_weight * offset_reg +
                        scale_reg_weight * scale_reg;

            loss.backward();

            if (avatar.g_scales.grad().defined())
            {
                torch::NoGradGuard no_grad;
                auto mask = torch::tensor({1.0f, 1.0f, 0.0f}, avatar.g_scales.options());
                avatar.g_scales.grad().mul_(mask);
            }

            densify_state.Accumulate(avatar.g_offsets, avatar.g_scales);
            optimizer.step();

            {
                torch::NoGradGuard no_grad;

                auto rot_norm = avatar.g_rots.norm(2, 1, true);
                rot_norm = torch::clamp_min(rot_norm, 1e-9);
                avatar.g_rots.div_(rot_norm);

                float target_thickness = 0.001f;
                float target_log_scale = std::log(target_thickness);

                using torch::indexing::Slice;
                // Set all Z-scales (index 2) to the target thickness
                avatar.g_scales.index_put_({Slice(), 2}, target_log_scale);
            }

            global_step++;

            if (
                densify_cfg.max_splits > 0 && densify_cfg.every > 0 &&
                (global_step % densify_cfg.every) == 0)
            {
                const auto prev_count = avatar.g_scales.size(0);
                auto stats = DensifyGaussians(avatar.g_scales,
                                              avatar.g_rots,
                                              avatar.g_opacities,
                                              avatar.g_colors,
                                              avatar.g_offsets,
                                              avatar.g_sh,
                                              avatar.g_bary_coords,
                                              avatar.g_face_indices,
                                              render_scale_modifier,
                                              densify_cfg,
                                              &densify_state);
                if (avatar.g_scales.size(0) != prev_count || stats.splits > 0 || stats.pruned > 0)
                {
                    rebuild_optimizer();
                }
                if (stats.splits > 0 || stats.clones > 0)
                {
                    std::cout << "Densify step " << global_step
                              << ": split " << stats.splits
                              << ", cloned " << stats.clones << " gaussians." << std::endl;
                    std::cout << "Densify total gaussians: " << avatar.g_scales.size(0)
                              << " (added " << (stats.splits + stats.clones) << ")" << std::endl;
                }
                if (stats.pruned > 0)
                {
                    std::cout << "Densify step " << global_step
                              << ": pruned " << stats.pruned << " gaussians." << std::endl;
                    std::cout << "Densify total gaussians: " << avatar.g_scales.size(0)
                              << " (pruned " << stats.pruned << ")" << std::endl;
                }
            }

            if (best_inlier_idx >= 0)
            {
                const size_t sample_idx = static_cast<size_t>(best_inlier_idx);

                if (batch_step % 10 == 0)
                {
                    std::cout << "Epoch " << epoch << " Batch " << batch_step
                              << " Loss: " << loss.item<float>() << " Recon loss: "
                              << recon_loss.item<float>() << std::endl;

                    // Calculate components for the BEST inlier sample (or average them if you prefer)
                    float l1_val = recon_losses[best_inlier_idx].item<float>();
                    float alpha_val = alpha_losses[best_inlier_idx].item<float>();
                    float mask_val = outside_losses[best_inlier_idx].item<float>();

                    std::cout << "Epoch " << epoch << " Batch " << batch_step
                              << " Total: " << total_values[best_inlier_idx]
                              << " | RGB L1: " << l1_val                       // <--- The true "visual" loss
                              << " | Alpha: " << alpha_val * alpha_loss_weight // <--- Likely the culprit
                              << " | Mask: " << mask_val * outside_mask_weight
                              << std::endl;
                    auto scale_vals_log = (capped_scales(avatar.g_scales) * render_scale_modifier).detach();
                    const float scale_min = scale_vals_log.min().item<float>();
                    const float scale_max = scale_vals_log.max().item<float>();
                    const float scale_mean = scale_vals_log.mean().item<float>();
                    std::cout << "Scale stats (min/mean/max): "
                              << scale_min << " / " << scale_mean << " / " << scale_max << std::endl;
                    std::cout << "Global step: " << global_step << std::endl;
                    if (tiles_total > 0)
                    {
                        const float avg_median = (tiles_median_count > 0)
                                                     ? (tiles_median_sum / static_cast<float>(tiles_median_count))
                                                     : 0.0f;
                        std::cout << "Tile stats: total=" << tiles_total
                                  << " mask_valid=" << tiles_mask_valid
                                  << " outlier_valid=" << tiles_outlier_valid
                                  << " final=" << tiles_final
                                  << " avg_median=" << avg_median << std::endl;
                    }
                }

                dropped_since_log += skipped_malformed + skipped_mask + skipped_outlier;
                if ((batch_step + 1) % 10 == 0)
                {
                    std::cout << "Dropped samples in last 10 batches: "
                              << dropped_since_log << std::endl;
                    dropped_since_log = 0;
                }
            }
        }

        // --- MODIFIED SECTION START ---

        // 1. Always calculate canonical geometry for saving/viewing
        torch::Tensor positions, rotations;
        std::tie(positions, rotations) = avatar.forward(canonical_betas, canonical_pose, canonical_trans);
 

        torch::Tensor colors;
        if (use_sh)
        {
            using torch::indexing::Slice;
            colors = avatar.g_sh.index({Slice(), 0, Slice()});
        }
        else
        {
            colors = avatar.g_colors;
        }
        auto opacities = avatar.g_opacities;
        auto scales = capped_scales(avatar.g_scales);

        using torch::indexing::Slice;

        torch::Tensor sh_to_send = sh;
        if (use_sh && sh_degree > 0)
        {
            sh_to_send = RotateSH(avatar.g_sh, rotations);  
        }

        // 2. Handle Shared Memory Viewer (Only if enabled)
        if (publish_viewer)
        {
            const int64_t bind_count = avatar.g_offsets.size(0);
            if (bind_count > 0)
            {
                auto bary_cpu = avatar.g_bary_coords.to(torch::kCPU).contiguous();
                auto offsets_cpu = avatar.g_offsets.to(torch::kCPU).contiguous();
                auto rots_cpu = avatar.g_rots.to(torch::kCPU).contiguous();
                auto faces_cpu = avatar.g_face_indices.to(torch::kCPU).contiguous();

                const float *bary_ptr = bary_cpu.data_ptr<float>();
                const float *off_ptr = offsets_cpu.data_ptr<float>();
                const float *rot_ptr = rots_cpu.data_ptr<float>();
                const int64_t *face_ptr = faces_cpu.data_ptr<int64_t>();

                const uint32_t stride = shared_gaussian::kSharedBindStrideFloats;
                bind_buffer.resize(static_cast<size_t>(bind_count) * stride);
                for (int64_t i = 0; i < bind_count; ++i)
                {
                    const size_t base = static_cast<size_t>(i) * stride;
                    const size_t base3 = static_cast<size_t>(i) * 3u;
                    const size_t base4 = static_cast<size_t>(i) * 4u;
                    bind_buffer[base + 0] = bary_ptr[base3 + 0];
                    bind_buffer[base + 1] = bary_ptr[base3 + 1];
                    bind_buffer[base + 2] = bary_ptr[base3 + 2];
                    bind_buffer[base + 3] = off_ptr[base3 + 0];
                    bind_buffer[base + 4] = off_ptr[base3 + 1];
                    bind_buffer[base + 5] = off_ptr[base3 + 2];
                    bind_buffer[base + 6] = rot_ptr[base4 + 0];
                    bind_buffer[base + 7] = rot_ptr[base4 + 1];
                    bind_buffer[base + 8] = rot_ptr[base4 + 2];
                    bind_buffer[base + 9] = rot_ptr[base4 + 3];
                    bind_buffer[base + 10] = static_cast<float>(face_ptr[i]);
                }

                auto betas_cpu = canonical_betas.to(torch::kCPU).contiguous();
                const int64_t betas_count = betas_cpu.numel();
                bind_betas_buffer.resize(static_cast<size_t>(betas_count));
                std::memcpy(bind_betas_buffer.data(), betas_cpu.data_ptr<float>(),
                            static_cast<size_t>(betas_count) * sizeof(float));

                bind_writer.Write(bind_betas_buffer.data(), static_cast<uint32_t>(betas_count),
                                  bind_buffer.data(), static_cast<uint32_t>(bind_count));
            }
            if (BuildSharedGaussianBuffer(positions, colors, opacities, scales, rotations, sh_to_send, sh_degree,
                                          &shared_buffer))
            {
                shared_writer.Write(shared_buffer.data(), static_cast<uint32_t>(positions.size(0)),
                                    shared_frame++);
            }
        }

        // 3. Save to Disk (Every epoch, Overwrite)
        {
            std::filesystem::path save_dir = viewer_export_dir.empty() ? out_dir_path : viewer_out_path;

            SaveViewerDataOverwrite(save_dir, positions, colors, opacities, scales, rotations,
                                    sh_to_send, sh_degree);
        }

        auto render_view = [&](size_t sample_index,
                               const TrainSample &sample,
                               const CachedSampleData &cached_entry) -> torch::Tensor
        {
            if (!cached_entry.valid)
            {
                return torch::Tensor();
            }
            auto target = cached_entry.target;
            const int H = static_cast<int>(target.size(1));
            const int W = static_cast<int>(target.size(2));
            if (H <= 0 || W <= 0)
            {
                return torch::Tensor();
            }

            SmplResult res;
            res.pose = sample.pose;
            res.shape = sample.betas;
            res.camera = sample.cam;

            auto pose = PoseToAxisAngle(res).to(device);
            auto pose_offset = pose_offsets.index({static_cast<int64_t>(sample_index)}).unsqueeze(0);
            pose = pose + pose_offset;
            auto global_rot = global_rot_offsets.index({static_cast<int64_t>(sample_index)}).unsqueeze(0);
            pose.index_put_({0, 0, torch::indexing::Slice()},
                            pose.index({0, 0, torch::indexing::Slice()}) + global_rot.squeeze(0));
            cv::Vec3f trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                     sample.crop_size, sample.focal_length,
                                                     static_cast<float>(sample.img_w),
                                                     static_cast<float>(sample.img_h));
            auto trans = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                       torch::TensorOptions().device(device).dtype(torch::kFloat));
            auto trans_offset = global_trans_offsets.index({static_cast<int64_t>(sample_index)});
            trans = trans + trans_offset;

            if (viewer_stream_poses)
            {
                auto pose_cpu = pose.detach().to(torch::kCPU).contiguous();
                const float *pose_ptr = pose_cpu.data_ptr<float>();
                std::ostringstream pose_line;
                pose_line.setf(std::ios::fixed);
                pose_line << std::setprecision(6);
                pose_line << "smpl_pose: [";
                for (int i = 0; i < 72; ++i)
                {
                    if (i > 0)
                        pose_line << ", ";
                    pose_line << pose_ptr[i];
                }
                pose_line << "]";
                std::cout << pose_line.str() << "\n";

                auto trans_cpu = trans.detach().to(torch::kCPU).contiguous();
                const float tx = trans_cpu[0].item<float>();
                const float ty = trans_cpu[1].item<float>();
                const float tz = trans_cpu[2].item<float>();
                std::ostringstream trans_line;
                trans_line.setf(std::ios::fixed);
                trans_line << std::setprecision(6);
                trans_line << "smpl_trans: [" << tx << ", " << ty << ", " << tz << "]";
                std::cout << trans_line.str() << "\n";
                std::cout.flush();
            }

            torch::Tensor means3D;
            torch::Tensor current_rots;
            std::tie(means3D, current_rots) =
                avatar.forward(canonical_betas, pose, torch::zeros({1, 3}, canonical_betas.options()));
            torch::Tensor current_sh = use_sh ? avatar.g_sh : torch::zeros({0}, avatar.g_colors.options());
            if (use_sh && sh_degree > 0)
            {
                current_sh = RotateSH(current_sh, current_rots);
            }
            auto y_scale = torch::tensor({1.0f, sample.y_sign, 1.0f},
                                         torch::TensorOptions().device(device).dtype(torch::kFloat));
            means3D = means3D * y_scale;
            means3D = means3D + trans;

            const int full_w = sample.img_w;
            const int full_h = sample.img_h;
            float f_render = sample.focal_length;
            torch::Tensor view_mat;
            torch::Tensor proj_mat;
            float tan_fovx = 0.0f;
            float tan_fovy = 0.0f;
            const int render_w = W;
            const int render_h = H;
            const float full_cx = static_cast<float>(full_w) * 0.5f;
            const float full_cy = static_cast<float>(full_h) * 0.5f;
            float x0 = sample.crop_x0;
            float y0 = sample.crop_y0;
            if (sample.crop_w <= 0.0f || sample.crop_h <= 0.0f)
            {
                x0 = sample.crop_cx - static_cast<float>(render_w) * 0.5f;
                y0 = sample.crop_cy - static_cast<float>(render_h) * 0.5f;
            }
            const float cx_crop = full_cx - x0;
            const float cy_crop = full_cy - y0;
            std::tie(view_mat, proj_mat, tan_fovx, tan_fovy) =
                BuildProjection(f_render, render_w, render_h, cx_crop, cy_crop, device);

            auto colors = use_sh ? torch::zeros({0}, avatar.g_colors.options()) : avatar.g_colors;
            auto outputs = GaussianRasterizer::apply(
                means3D,
                colors,
                avatar.g_opacities,
                capped_scales(avatar.g_scales),
                current_rots,
                render_scale_modifier,
                view_mat,
                proj_mat,
                tan_fovx,
                tan_fovy,
                render_h,
                render_w,
                current_sh,
                use_sh ? sh_degree : 0,
                cam_pos,
                false);
            auto image = outputs[0];
            if (!image.defined() || image.dim() != 3 || image.size(0) != 3 ||
                image.size(1) != H || image.size(2) != W)
            {
                return torch::Tensor();
            }

            return image.detach();
        };

        if ((epoch + 1) % 5 == 0 || epoch == (epochs - 1))
        {
            const int saved_pairs = SaveEpochViewPairs(samples, cached, out_dir_path, epoch, render_view);
            std::cout << "Epoch " << epoch << " saved " << saved_pairs << " view pairs." << std::endl;
        }
    }

    return 0;
}
