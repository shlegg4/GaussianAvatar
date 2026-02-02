#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
                if (key != "--viewer" && key != "--headless")
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
            else if (key == "--output-dir")
                options->output_dir = value;
            else if (key == "--scale-reg")
                options->scale_reg_weight = std::stof(value);
            else if (key == "--scale-lr")
                options->scale_lr = std::stof(value);
            else if (key == "--scale-max")
                options->scale_max_value = std::stof(value);
            else if (key == "--offset-reg")
                options->offset_reg_weight = std::stof(value);
            else if (key == "--mesh-reg")
                options->mesh_reg_weight = std::stof(value);
            else if (key == "--mesh-max-dist")
                options->mesh_reg_max_dist = std::stof(value);
            else if (key == "--alpha-loss")
                options->alpha_loss_weight = std::stof(value);
            else if (key == "--opacity-binarize")
                options->opacity_binarize_weight = std::stof(value);
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
            else if (key == "--viewer-width")
                options->viewer_width = std::stoi(value);
            else if (key == "--viewer-height")
                options->viewer_height = std::stoi(value);
            else if (key == "--viewer-every")
                options->viewer_every = std::stoi(value);
            else if (key == "--viewer-shm")
                options->viewer_shm_name = value;
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

torch::Tensor CropRenderToTarget(const torch::Tensor &full_render, int crop_w, int crop_h,
                                 float crop_cx, float crop_cy)
{
    if (!full_render.defined() || full_render.dim() != 3)
    {
        return torch::Tensor();
    }
    const int channels = static_cast<int>(full_render.size(0));
    auto output = torch::zeros({channels, crop_h, crop_w}, full_render.options());

    const int full_h = static_cast<int>(full_render.size(1));
    const int full_w = static_cast<int>(full_render.size(2));
    const int x0 = static_cast<int>(std::lround(crop_cx - 0.5f * static_cast<float>(crop_w)));
    const int y0 = static_cast<int>(std::lround(crop_cy - 0.5f * static_cast<float>(crop_h)));
    const int x1 = x0 + crop_w;
    const int y1 = y0 + crop_h;

    const int src_x0 = std::max(0, x0);
    const int src_y0 = std::max(0, y0);
    const int src_x1 = std::min(full_w, x1);
    const int src_y1 = std::min(full_h, y1);

    const int src_w = src_x1 - src_x0;
    const int src_h = src_y1 - src_y0;
    if (src_w <= 0 || src_h <= 0)
    {
        return output;
    }

    const int dst_x0 = src_x0 - x0;
    const int dst_y0 = src_y0 - y0;

    using torch::indexing::Slice;
    output.index_put_({Slice(), Slice(dst_y0, dst_y0 + src_h), Slice(dst_x0, dst_x0 + src_w)},
                      full_render.index({Slice(), Slice(src_y0, src_y0 + src_h), Slice(src_x0, src_x0 + src_w)}));
    return output;
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
        auto face_area = 0.5f * torch::norm(cross, 2, 1) * 0.01f;
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
                     " [--scale-reg <float>] [--scale-max <float>]"
                     " [--offset-reg <float>] [--mesh-reg <float>]"
                     " [--mesh-max-dist <float>] [--alpha-loss <float>] [--opacity-binarize <float>]"
                     " [--color-lr <float>] [--opacity-lr <float>]"
                     " [--sh-degree <int>]"
                     " [--densify-every <int>] [--densify-max <int>] [--densify-max-clones <int>]"
                     " [--densify-scale <float>]"
                     " [--densify-split-scale <float>] [--densify-split-offset <float>]"
                     " [--densify-min-grad <float>] [--densify-grow-grad <float>]"
                     " [--densify-prune-opacity <float>]"
                     " [--densify-prune-max <int>] [--densify-reset-opacity <float>]"
                     " [--densify-stop-epoch <int>]"
                     " [--viewer|--headless] [--viewer-width <int>] [--viewer-height <int>]"
                     " [--viewer-every <int>] [--viewer-shm <name>]\n";
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
    const std::string &output_dir = options.output_dir;
    const float scale_reg_weight = options.scale_reg_weight;
    const float scale_lr = (options.scale_lr < 0.0f) ? lr : options.scale_lr;
    const float scale_max_value = options.scale_max_value;
    const float offset_reg_weight = options.offset_reg_weight;
    const float mesh_reg_weight = options.mesh_reg_weight;
    const float mesh_reg_max_dist = options.mesh_reg_max_dist;
    const float opacity_binarize_weight = options.opacity_binarize_weight;
    const float color_lr = options.color_lr;
    const float opacity_lr = (options.opacity_lr < 0.0f) ? lr : options.opacity_lr;
    const int sh_degree = options.sh_degree;
    const int viewer_every = std::max(1, options.viewer_every);
    const int densify_every = std::max(1, options.densify_every);
    const float pose_lr = 1e-4f;
    const float outside_mask_weight = 0.1f;
    const float alpha_loss_weight = options.alpha_loss_weight;
    const float render_scale_modifier = 1.0f;
    const float render_threshold = 3.0f / 255.0f;
    const int batch_size = 4;
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

        auto target = LoadImageTensor(crop, device);
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

        auto matte_mask = LoadMatteMaskTensor(matte, W, H, device);
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
    if (publish_viewer)
    {
        if (!shared_writer.Init(options.viewer_shm_name, static_cast<uint32_t>(num_gaussians),
                                shared_stride, static_cast<uint32_t>(sh_degree),
                                render_scale_modifier))
        {
            std::cerr << "Failed to open shared memory mapping: " << options.viewer_shm_name << std::endl;
            publish_viewer = false;
        }
    }
    std::vector<float> shared_buffer;
    uint64_t shared_frame = 0;

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

    std::vector<torch::Tensor> base_params = {
        avatar.g_rots,
        avatar.g_offsets};
    std::vector<torch::Tensor> scale_params = {avatar.g_scales};
    std::vector<torch::Tensor> color_params = {use_sh ? avatar.g_sh : avatar.g_colors};
    std::vector<torch::Tensor> opacity_params = {avatar.g_opacities};
    std::vector<torch::Tensor> pose_params = {pose_offsets};
    torch::optim::Adam optimizer(base_params, torch::optim::AdamOptions(lr));
    optimizer.add_param_group({scale_params});
    auto &scale_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(scale_group.options()).lr(scale_lr);
    optimizer.add_param_group({color_params});
    auto &color_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(color_group.options()).lr(color_lr);
    optimizer.add_param_group({opacity_params});
    auto &opacity_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(opacity_group.options()).lr(opacity_lr);
    optimizer.add_param_group({pose_params});
    auto &pose_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(pose_group.options()).lr(pose_lr);

    auto rebuild_optimizer = [&]()
    {
        base_params = {avatar.g_rots, avatar.g_offsets};
        scale_params = {avatar.g_scales};
        color_params = {use_sh ? avatar.g_sh : avatar.g_colors};
        opacity_params = {avatar.g_opacities};
        optimizer = torch::optim::Adam(base_params, torch::optim::AdamOptions(lr));
        optimizer.add_param_group({scale_params});
        auto &new_scale_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_scale_group.options()).lr(scale_lr);
        optimizer.add_param_group({color_params});
        auto &new_color_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_color_group.options()).lr(color_lr);
        optimizer.add_param_group({opacity_params});
        auto &new_opacity_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_opacity_group.options()).lr(opacity_lr);
        optimizer.add_param_group({pose_params});
        auto &new_pose_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_pose_group.options()).lr(pose_lr);
    };

    torch::Tensor last_render;
    std::filesystem::path out_dir_path(output_dir);
    std::error_code out_ec;
    std::filesystem::create_directories(out_dir_path, out_ec);
    if (out_ec)
    {
        std::cerr << "Failed to create output dir: " << output_dir << std::endl;
        return -1;
    }

    std::mt19937 rng(static_cast<unsigned int>(std::random_device{}()));
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
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
            std::vector<float> recon_values;
            std::vector<float> total_values;
            std::vector<TrainSample> batch_samples;
            std::vector<cv::Mat> batch_crops;
            std::vector<torch::Tensor> batch_renders;
            std::vector<torch::Tensor> batch_means3d;
            std::vector<torch::Tensor> batch_betas;
            std::vector<torch::Tensor> batch_pose;
            std::vector<torch::Tensor> batch_trans;
            int skipped_malformed = 0;
            int skipped_mask = 0;

            for (size_t idx = batch_start; idx < batch_end; ++idx)
            {
                const auto &sample = samples[indices[idx]];
                const auto &cached_entry = cached[indices[idx]];
                if (!cached_entry.valid)
                {
                    skipped_malformed++;
                    continue;
                }
                auto target = cached_entry.target;
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
                cv::Vec3f trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                         sample.crop_size, sample.focal_length,
                                                         static_cast<float>(sample.img_w),
                                                         static_cast<float>(sample.img_h));
                auto trans = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                           torch::TensorOptions().device(device).dtype(torch::kFloat));

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
                const float x0 = sample.crop_cx - static_cast<float>(render_w) * 0.5f;
                const float y0 = sample.crop_cy - static_cast<float>(render_h) * 0.5f;
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
                auto alpha = outputs[1];
                if (!image.defined() || image.dim() != 3 || image.size(0) != 3 ||
                    image.size(1) != H || image.size(2) != W ||
                    !alpha.defined() || alpha.dim() != 3 || alpha.size(0) != 1 ||
                    alpha.size(1) != H || alpha.size(2) != W)
                {
                    skipped_malformed++;
                    continue;
                }
                if (!IsMaskCoverageValidTensor(image, cached_entry.matte_mask, 0.3f, render_threshold))
                {
                    skipped_mask++;
                    continue;
                }

                auto outside_mask = 1.0f - cached_entry.matte_mask;
                auto outside_loss = torch::mean(image * outside_mask);
                auto alpha_loss = torch::l1_loss(alpha, cached_entry.matte_mask);

                auto recon_loss = torch::l1_loss(image, target);
                auto loss_value = recon_loss.item<float>();
                auto outside_value = outside_loss.item<float>();
                auto alpha_value = alpha_loss.item<float>();
                if (!std::isfinite(loss_value) || !std::isfinite(outside_value) || !std::isfinite(alpha_value))
                {
                    skipped_malformed++;
                    continue;
                }

                recon_losses.push_back(recon_loss);
                outside_losses.push_back(outside_loss);
                alpha_losses.push_back(alpha_loss);
                recon_values.push_back(loss_value);
                total_values.push_back(loss_value + outside_mask_weight * outside_value + alpha_loss_weight * alpha_value);
                batch_samples.push_back(sample);
                batch_crops.push_back(cached_entry.crop_bgr);
                batch_renders.push_back(image.detach());
                batch_means3d.push_back(means3D.detach());
                batch_betas.push_back(canonical_betas.detach());
                batch_pose.push_back(pose.detach());
                batch_trans.push_back(trans.detach());
            }

            if (recon_losses.empty())
            {
                dropped_since_log += skipped_malformed + skipped_mask;
                if ((batch_step + 1) % 10 == 0)
                {
                    std::cout << "Dropped samples in last 10 batches: "
                              << dropped_since_log << std::endl;
                    dropped_since_log = 0;
                }
                continue;
            }

            const float outlier_percentile = 0.8f;
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
            int inlier_count = 0;
            int skipped_outlier = 0;
            int last_inlier_idx = -1;
            int best_inlier_idx = -1;
            float best_inlier_loss = std::numeric_limits<float>::infinity();
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
                inlier_count++;
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
            // --- Top-K Scale Regularization (Safety Valve) ---
            auto current_scales = torch::exp(avatar.g_scales) * render_scale_modifier;
            using torch::indexing::Slice;
            auto tan_scales_sq = current_scales.index({Slice(), Slice(0, 2)}).pow(2).sum(1);
            const int64_t num_gaussians = tan_scales_sq.size(0);
            const int64_t k = std::min<int64_t>(num_gaussians, std::max<int64_t>(64, num_gaussians / 200));
            auto topk_result = torch::topk(tan_scales_sq, k, /*dim=*/0, /*largest=*/true, /*sorted=*/false);
            auto top_k_values = std::get<0>(topk_result);
            auto scale_reg = torch::mean(top_k_values);
            auto offset_reg = torch::mean(avatar.g_offsets.pow(2));
            auto loss = recon_loss +
                        offset_reg_weight * offset_reg +
                        scale_reg_weight * scale_reg;
            loss.backward();
            densify_state.Accumulate(avatar.g_offsets, avatar.g_scales);
            optimizer.step();
            global_step++;

            const bool densify_enabled = (densify_stop_epoch <= 0) || (epoch < densify_stop_epoch);
            if (densify_enabled &&
                global_step <= 1000 &&
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
                last_render = batch_renders[sample_idx];

                if (batch_step % 10 == 0)
                {
                    std::cout << "Epoch " << epoch << " Batch " << batch_step
                              << " Loss: " << loss.item<float>() << " Offset Reg: "
                              << offset_reg.item<float>() * offset_reg_weight << " Scale Reg: "
                              << scale_reg.item<float>() * scale_reg_weight << " Recon loss: " << recon_loss.item<float>() << std::endl;
                    auto scale_vals_log = (capped_scales(avatar.g_scales) * render_scale_modifier).detach();
                    const float scale_min = scale_vals_log.min().item<float>();
                    const float scale_max = scale_vals_log.max().item<float>();
                    const float scale_mean = scale_vals_log.mean().item<float>();
                    std::cout << "Scale stats (min/mean/max): "
                              << scale_min << " / " << scale_mean << " / " << scale_max << std::endl;
                }

                if (publish_viewer && (batch_step % viewer_every == 0))
                {
                    torch::Tensor positions;
                    torch::Tensor rotations;
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
                    torch::Tensor sh_to_send = sh;
                    if (use_sh && sh_degree > 0)
                    {
                        sh_to_send = RotateSH(avatar.g_sh, rotations);
                    }
                    if (BuildSharedGaussianBuffer(positions, colors, opacities, scales, rotations, sh_to_send, sh_degree,
                                                  &shared_buffer))
                    {
                        shared_writer.Write(shared_buffer.data(), static_cast<uint32_t>(positions.size(0)),
                                            shared_frame++);
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
            cv::Vec3f trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                     sample.crop_size, sample.focal_length,
                                                     static_cast<float>(sample.img_w),
                                                     static_cast<float>(sample.img_h));
            auto trans = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                       torch::TensorOptions().device(device).dtype(torch::kFloat));

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
            const float x0 = sample.crop_cx - static_cast<float>(render_w) * 0.5f;
            const float y0 = sample.crop_cy - static_cast<float>(render_h) * 0.5f;
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

    std::filesystem::path out_path = out_dir_path / "final_render.png";
    if (!SaveImageTensorPng(out_path.string(), last_render))
    {
        std::cerr << "Failed to save final render to " << out_path.string() << std::endl;
    }
    else
    {
        std::cout << "Saved final render to " << out_path.string() << std::endl;
    }

    return 0;
}
