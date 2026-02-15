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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <torch/csrc/autograd/profiler.h>

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

// --- SSIM Implementation Start ---
torch::Tensor create_window(int window_size, int channel)
{
    auto t = torch::arange(window_size, torch::kFloat32) - window_size / 2;
    auto gauss = torch::exp(-(t * t) / (2 * 1.5 * 1.5));
    gauss = gauss / gauss.sum();

    auto window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0);
    return window_2d.expand({channel, 1, window_size, window_size}).contiguous();
}

torch::Tensor ssim(const torch::Tensor &img1, const torch::Tensor &img2, const torch::Tensor &window,
                   int window_size = 21)
{
    auto inp1 = (img1.dim() == 3) ? img1.unsqueeze(0) : img1;
    auto inp2 = (img2.dim() == 3) ? img2.unsqueeze(0) : img2;

    const int channel = static_cast<int>(inp1.size(1));
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

    return ssim_map.mean();
} 

struct TrainDataGPU
{
    torch::Tensor all_poses;   // (N, 72)
    torch::Tensor all_trans;   // (N, 3)
    torch::Tensor all_time;    // (N, 1)

    TrainDataGPU(const std::vector<TrainSample> &samples, torch::Device device)
    {
        const int64_t N = static_cast<int64_t>(samples.size());
        std::vector<float> flat_poses;
        std::vector<float> flat_trans;
        std::vector<float> flat_time;
        flat_poses.reserve(static_cast<size_t>(N) * 72u);
        flat_trans.reserve(static_cast<size_t>(N) * 3u);
        flat_time.reserve(static_cast<size_t>(N));

        for (int64_t i = 0; i < N; ++i)
        {
            const auto &s = samples[static_cast<size_t>(i)];
            if (s.pose.size() == 72)
            {
                flat_poses.insert(flat_poses.end(), s.pose.begin(), s.pose.end());
            }
            else if (s.pose.size() == 144)
            {
                const auto pose_aa = ConvertPose6dToAxisAngle(s.pose);
                flat_poses.insert(flat_poses.end(), pose_aa.begin(), pose_aa.end());
            }
            else if (s.pose.size() % 3 == 0 && (s.pose.size() / 3) == 24)
            {
                flat_poses.insert(flat_poses.end(), s.pose.begin(), s.pose.end());
            }
            else
            {
                throw std::runtime_error("Unsupported pose size: " + std::to_string(s.pose.size()));
            }

            cv::Vec3f t = EstimateTranslation(s.cam, s.crop_cx, s.crop_cy,
                                              s.crop_size, s.focal_length,
                                              static_cast<float>(s.img_w), static_cast<float>(s.img_h));
            flat_trans.push_back(t[0]);
            flat_trans.push_back(t[1]);
            flat_trans.push_back(t[2]);

            const float time_val = static_cast<float>(i) /
                                   static_cast<float>(std::max<int64_t>(1, N - 1));
            flat_time.push_back(time_val);
        }

        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        all_poses = torch::from_blob(flat_poses.data(), {N, 72}, opts).clone().to(device);
        all_trans = torch::from_blob(flat_trans.data(), {N, 3}, opts).clone().to(device);
        all_time = torch::from_blob(flat_time.data(), {N, 1}, opts).clone().to(device);
    }
};

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

struct PoseRefiner : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    const float ROT_SCALE = 0.05f;
    const float TRANS_XY_SCALE = 0.2f;
    const float TRANS_Z_SCALE = 0.2f;

    PoseRefiner(int input_dim = 72 + 3)
    {
        const int embed_dim = 4;
        const int total_input = input_dim + embed_dim;
        const int hidden_dim = 24;

        fc1 = register_module("fc1", torch::nn::Linear(total_input, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, input_dim));

        torch::nn::init::zeros_(fc3->weight);
        torch::nn::init::zeros_(fc3->bias);
    }

    torch::Tensor forward(torch::Tensor noisy_pose_trans, torch::Tensor time_norm)
    {
        auto device = time_norm.device();
        auto freqs = torch::pow(2.0, torch::arange(0, 2, torch::TensorOptions().device(device)).to(torch::kFloat));
        auto time_projected = time_norm * freqs.unsqueeze(0) * 3.14159f;
        auto time_embed = torch::cat({torch::sin(time_projected), torch::cos(time_projected)}, 1);

        auto x = torch::cat({noisy_pose_trans, time_embed}, 1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto delta = fc3->forward(x);

        auto delta_pose = torch::tanh(delta.slice(1, 0, 72)) * ROT_SCALE;
        auto delta_trans_raw = delta.slice(1, 72, 75);

        auto trans_x = torch::tanh(delta_trans_raw.slice(1, 0, 1)) * TRANS_XY_SCALE;
        auto trans_y = torch::tanh(delta_trans_raw.slice(1, 1, 2)) * TRANS_XY_SCALE;
        auto trans_z = torch::tanh(delta_trans_raw.slice(1, 2, 3)) * TRANS_Z_SCALE;

        auto delta_trans = torch::cat({trans_x, trans_y, trans_z}, 1);
        return torch::cat({delta_pose, delta_trans}, 1);
    }
};



int run_real_training(int argc, char *argv[])
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
                     " [--lambda-dssim <float>]"
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

    at::globalContext().setBenchmarkCuDNN(false);

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
    const int ssim_window_size = 21;
    torch::Tensor ssim_window = create_window(ssim_window_size, 3).to(device);
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

        if (storage_device == torch::kCPU)
        {
            target = target.pin_memory();
            matte_mask = matte_mask.pin_memory();
        }

        entry.target = target;
        entry.matte_mask = matte_mask;
        entry.crop_bgr = crop;
        entry.valid = true;
        cached[i] = std::move(entry);
    }

    TrainDataGPU gpu_data(samples, device);

    GaussianAvatar avatar(smpl_model_path);
    avatar.to(device);

    std::ifstream smpl_in(smpl_model_path, std::ios::binary);
    std::vector<char> f_bytes((std::istreambuf_iterator<char>(smpl_in)), (std::istreambuf_iterator<char>()));
    auto dict = torch::pickle_load(f_bytes).toGenericDict();
    torch::Tensor faces = dict.at("faces").toTensor().to(torch::kLong).to(device);
    avatar.init_gaussians(num_gaussians, faces, render_scale_modifier, options.sh_degree);
    PoseRefiner pose_refiner;
    pose_refiner.to(device);
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
    const int warmup_epochs = 0;

    std::vector<torch::Tensor> rot_params = {avatar.g_rots};
    std::vector<torch::Tensor> offset_params = {avatar.g_offsets};
    std::vector<torch::Tensor> scale_params = {avatar.g_scales};
    std::vector<torch::Tensor> color_params = {use_sh ? avatar.g_sh : avatar.g_colors};
    std::vector<torch::Tensor> opacity_params = {avatar.g_opacities};
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
    optimizer.add_param_group({pose_refiner.parameters()});
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
        optimizer.add_param_group({pose_refiner.parameters()});
        auto &new_pose_group = optimizer.param_groups().back();
        static_cast<torch::optim::AdamOptions &>(new_pose_group.options()).lr(pose_lr * pose_lr_multiplier);
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
        const int sh_degree_eff = (epoch < warmup_epochs) ? 0 : sh_degree;
        if (epoch < warmup_epochs)
        {
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options()).lr(0.0f);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[1].options()).lr(0.0f);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[2].options()).lr(0.0f);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[3].options()).lr(color_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[4].options()).lr(opacity_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[5].options()).lr(pose_lr * pose_lr_multiplier);
        }
        else if (epoch == warmup_epochs)
        {
            std::cout << "Warmup complete. Unfreezing Gaussian geometry..." << std::endl;
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options()).lr(rot_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[1].options()).lr(offset_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[2].options()).lr(scale_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[3].options()).lr(color_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[4].options()).lr(opacity_lr * lr_multiplier);
        }
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
            std::vector<torch::Tensor> total_losses;
            
            // Stats accumulation (GPU tensors only, no floats!)
            std::vector<torch::Tensor> debug_rot_deltas;
            std::vector<torch::Tensor> debug_trans_deltas;

            std::vector<torch::Tensor> batch_renders;
            std::vector<torch::Tensor> batch_betas; // Stores validity weights

            std::vector<int64_t> valid_batch_indices;
            valid_batch_indices.reserve(batch_end - batch_start);
            for (size_t idx = batch_start; idx < batch_end; ++idx)
            {
                const int64_t sample_idx = static_cast<int64_t>(indices[idx]);
                if (!cached[sample_idx].valid)
                {
                    continue;
                }
                valid_batch_indices.push_back(sample_idx);
            }

            if (valid_batch_indices.empty())
            {
                continue;
            }

            std::sort(valid_batch_indices.begin(), valid_batch_indices.end(),
                [&](int64_t a, int64_t b) {
                    return cached[static_cast<size_t>(a)].target.numel() > cached[static_cast<size_t>(b)].target.numel();
                }
            );

            int64_t total_target_floats = 0;
            int64_t total_mask_floats = 0;
            std::vector<int64_t> target_offsets;
            std::vector<int64_t> mask_offsets;
            target_offsets.reserve(valid_batch_indices.size());
            mask_offsets.reserve(valid_batch_indices.size());

            for (const auto& idx : valid_batch_indices)
            {
                const auto& entry = cached[static_cast<size_t>(idx)];
                
                target_offsets.push_back(total_target_floats);
                total_target_floats += entry.target.numel();
                
                mask_offsets.push_back(total_mask_floats);
                total_mask_floats += entry.matte_mask.numel();
            }

            // 2. Allocate ONE pinned buffer for the whole batch
            auto packed_targets_cpu = torch::empty(
                {total_target_floats}, 
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true)
            );
            auto packed_masks_cpu = torch::empty(
                {total_mask_floats}, 
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true)
            );

            // 3. Copy data into the pinned buffer (Flattening implicitly)
            float* t_ptr = packed_targets_cpu.data_ptr<float>();
            float* m_ptr = packed_masks_cpu.data_ptr<float>();

            for (size_t k = 0; k < valid_batch_indices.size(); ++k)
            {
                const auto& entry = cached[static_cast<size_t>(valid_batch_indices[k])];
                
                // Fast copy using raw pointers
                const int64_t t_size = entry.target.numel();
                const int64_t m_size = entry.matte_mask.numel();
                
                std::memcpy(t_ptr + target_offsets[k], entry.target.data_ptr<float>(), t_size * sizeof(float));
                std::memcpy(m_ptr + mask_offsets[k], entry.matte_mask.data_ptr<float>(), m_size * sizeof(float));
            }

            // 4. Async Upload (One big chunk per type)
            auto packed_targets_gpu = packed_targets_cpu.to(device, /*non_blocking=*/true);
            auto packed_masks_gpu = packed_masks_cpu.to(device, /*non_blocking=*/true);

            auto idx_tensor = torch::from_blob(valid_batch_indices.data(),
                                               {static_cast<int64_t>(valid_batch_indices.size())},
                                               torch::kLong).to(device, /*non_blocking=*/true);
            auto pose_base_batch = gpu_data.all_poses.index_select(0, idx_tensor).view({-1, 24, 3});
            auto trans_base_batch = gpu_data.all_trans.index_select(0, idx_tensor).view({-1, 3});
            auto time_tensor = gpu_data.all_time.index_select(0, idx_tensor);

            auto pose_flat = pose_base_batch.view({-1, 72});
            auto net_input = torch::cat({pose_flat, trans_base_batch}, 1);
            auto deltas = pose_refiner.forward(net_input, time_tensor);

            auto pose_deltas = deltas.slice(1, 0, 72).view({-1, 24, 3});
            auto trans_deltas = deltas.slice(1, 72, 75).view({-1, 3});

            if (batch_step % 50 == 0)
            {
                debug_rot_deltas.push_back(pose_deltas.detach().abs().mean());
                debug_trans_deltas.push_back(trans_deltas.detach().abs());
            }

            for (size_t k = 0; k < valid_batch_indices.size(); ++k)
            {
                const int64_t sample_idx = valid_batch_indices[k];
                const auto &sample = samples[static_cast<size_t>(sample_idx)]; 
                const auto& entry = cached[static_cast<size_t>(sample_idx)]; // Need metadata from CPU cache

                int64_t t_len = entry.target.numel();
                int64_t m_len = entry.matte_mask.numel();
                
                auto target = packed_targets_gpu.narrow(0, target_offsets[k], t_len).view(entry.target.sizes());
                auto matte_mask = packed_masks_gpu.narrow(0, mask_offsets[k], m_len).view(entry.matte_mask.sizes());

                const int H = static_cast<int>(target.size(1));
                const int W = static_cast<int>(target.size(2));
                if (H <= 0 || W <= 0) { continue; } 

                auto pose_base = pose_base_batch.index({static_cast<int64_t>(k)}).unsqueeze(0);
                auto trans_base = trans_base_batch.index({static_cast<int64_t>(k)}).unsqueeze(0);
                auto pose_delta = pose_deltas.index({static_cast<int64_t>(k)}).unsqueeze(0);
                auto trans_delta = trans_deltas.index({static_cast<int64_t>(k)}).unsqueeze(0);

                auto pose = pose_base + pose_delta;
                auto trans = (trans_base + trans_delta).squeeze(0);

                torch::Tensor means3D, current_rots;
                std::tie(means3D, current_rots) = avatar.forward(canonical_betas, pose,
                                                                torch::zeros({1, 3}, canonical_betas.options()));

                torch::Tensor current_sh = use_sh ? avatar.g_sh : torch::zeros({0}, avatar.g_colors.options());
                if (use_sh && sh_degree_eff > 0)
                {
                    current_sh = RotateSH(current_sh, current_rots);
                }

                auto one = torch::ones({1, 1}, torch::TensorOptions().device(device));
                auto y_sign_tensor = torch::full({1, 1}, sample.y_sign, torch::TensorOptions().device(device));
                auto y_scale = torch::cat({one, y_sign_tensor, one}, 1);
                means3D = means3D * y_scale + trans;

                const float full_cx = static_cast<float>(sample.img_w) * 0.5f;
                const float full_cy = static_cast<float>(sample.img_h) * 0.5f;
                float x0 = (sample.crop_w > 0) ? sample.crop_x0 : (sample.crop_cx - W * 0.5f);
                float y0 = (sample.crop_h > 0) ? sample.crop_y0 : (sample.crop_cy - H * 0.5f);

                torch::Tensor view_mat, proj_mat;
                float tan_fovx, tan_fovy;
                std::tie(view_mat, proj_mat, tan_fovx, tan_fovy) =
                    BuildProjection(sample.focal_length, W, H, full_cx - x0, full_cy - y0, device);

                auto outputs = GaussianRasterizer::apply(
                    means3D,
                    (use_sh ? torch::zeros({0}, avatar.g_colors.options()) : avatar.g_colors),
                    avatar.g_opacities,
                    capped_scales(avatar.g_scales),
                    current_rots,
                    render_scale_modifier,
                    view_mat, proj_mat, tan_fovx, tan_fovy, H, W,
                    current_sh, (use_sh ? sh_degree_eff : 0), cam_pos, false);

                auto image = outputs[0];
                auto alpha = outputs[1]; 

                // --- 1. Pure GPU Validity Check ---
                // Calculate coverage without pulling bool to CPU
                auto valid_pixels = (image > render_threshold).to(torch::kFloat32) * matte_mask;
                auto coverage_ratio = valid_pixels.sum() / (matte_mask.sum() + 1e-6f);
                auto is_valid_sample = (coverage_ratio > 0.3f).to(torch::kFloat32).detach(); 

                // --- 2. Calculate Losses ---
                auto ssim_value = ssim(image, target, ssim_window, ssim_window_size);
                auto d_ssim_loss = (1.0f - ssim_value);
                auto outside_loss = torch::mean(image * (1.0f - matte_mask));
                auto alpha_loss = torch::l1_loss(alpha, matte_mask);

                auto diff = torch::abs(image - target) * matte_mask;
                auto recon_loss = diff.sum() / torch::clamp_min(matte_mask.sum() * 3.0f, 1e-6);

                // --- 3. Mask Losses (Multiply by 0.0 if invalid) ---
                recon_loss = recon_loss * is_valid_sample;
                outside_loss = outside_loss * is_valid_sample;
                alpha_loss = alpha_loss * is_valid_sample;
                d_ssim_loss = d_ssim_loss * is_valid_sample;
                
                auto total_loss = recon_loss + outside_mask_weight * outside_loss + alpha_loss_weight * alpha_loss;
                
                // Guard NaNs
                total_loss = torch::where(torch::isfinite(total_loss), total_loss, torch::zeros_like(total_loss));

                // Accumulate
                recon_losses.push_back(recon_loss);
                outside_losses.push_back(outside_loss);
                alpha_losses.push_back(alpha_loss);
                ssim_losses.push_back(d_ssim_loss);
                total_losses.push_back(total_loss);
                
                // Store validity weight for batch reduction
                batch_betas.push_back(is_valid_sample);  
            } // End Inner Loop

            if (recon_losses.empty()) continue;

            // --- BATCH REDUCTION (PURE GPU) ---
            auto total_tensor = torch::stack(total_losses);
            auto validity_weights = torch::stack(batch_betas).squeeze(); 
            if (validity_weights.dim() == 0) validity_weights = validity_weights.unsqueeze(0);

            // Outlier Rejection
            const float outlier_percentile = 1.0f; // 1.0 = Keep all
            auto outlier_threshold = torch::quantile(total_tensor, outlier_percentile);
            auto inlier_mask = (total_tensor <= outlier_threshold).to(torch::kFloat32);
            
            // Combine inlier logic with validity logic
            auto final_weight = inlier_mask * validity_weights;
            auto valid_sum = final_weight.sum();
            auto denom = torch::clamp_min(valid_sum, 1e-6f);

            // Weighted Sum
            auto recon_loss = (torch::stack(recon_losses) * final_weight).sum() / denom;
            auto avg_ssim_loss = (torch::stack(ssim_losses) * final_weight).sum() / denom;
            auto outside_term = (torch::stack(outside_losses) * final_weight).sum() / denom;
            auto alpha_term = (torch::stack(alpha_losses) * final_weight).sum() / denom;

            // Regularization
            auto current_scales = torch::exp(avatar.g_scales) * render_scale_modifier;
            auto scale_reg = torch::mean(current_scales.pow(2).sum(1));
            auto offset_reg = torch::mean(torch::abs(avatar.g_offsets));

            auto loss = (1.0f - lambda_dssim) * recon_loss +
                        lambda_dssim * avg_ssim_loss +
                        outside_mask_weight * outside_term +
                        alpha_loss_weight * alpha_term + 
                        offset_reg_weight * offset_reg +
                        scale_reg_weight * scale_reg;

            loss.backward();

            if (avatar.g_scales.grad().defined()) {
                torch::NoGradGuard no_grad;
                avatar.g_scales.grad().mul_(torch::tensor({1.0f, 1.0f, 0.0f}, avatar.g_scales.options()));
            }
            
            if (epoch % 3 == 0) {
                 densify_state.Accumulate(avatar.g_offsets, avatar.g_scales);
            }
            optimizer.step();

            // --- DEFERRED LOGGING (Sync happens HERE, once per batch) ---
            if (batch_step % 10 == 0)
            {
                // It is safe to sync here because we are printing to console anyway.
                float current_loss_val = loss.item<float>();
                float valid_count = valid_sum.item<float>();
                
                std::cout << "Epoch " << epoch << " Batch " << batch_step
                          << " Loss: " << current_loss_val
                          << " Valid: " << valid_count << "/" << total_losses.size() << std::endl;

                // Print deferred MLP stats if available
                if (!debug_rot_deltas.empty()) {
                    auto rot_ten = torch::stack(debug_rot_deltas).mean();
                    auto trans_ten = torch::stack(debug_trans_deltas).max();
                     std::cout << "[MLP Stats] Rot Delta: " << rot_ten.item<float>()
                               << " | Max Shift: " << trans_ten.item<float>() << std::endl;
                } 
            }

            global_step++;

            // const bool densify_enabled = (densify_stop_epoch <= 0) || (epoch < densify_stop_epoch);
            if (
                epoch >= warmup_epochs &&
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

        torch::Tensor sh_to_send = sh;
        if (use_sh && sh_degree_eff > 0)
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
            // Use viewer_export_dir if provided, otherwise default to output_dir
            std::filesystem::path save_dir = viewer_export_dir.empty() ? out_dir_path : viewer_out_path;

            // Overwrite a constant epoch index on each save.
            SaveViewerDataOverwrite(save_dir, positions, colors, opacities, scales, rotations,
                                    sh_to_send, sh_degree);
        }

        // --- MODIFIED SECTION END ---

        auto render_view = [&](size_t sample_index,
                               const TrainSample &sample,
                               const CachedSampleData &cached_entry) -> torch::Tensor
        {
            torch::NoGradGuard no_grad;
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

            auto pose_base = PoseToAxisAngle(res).to(device);
            cv::Vec3f trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                     sample.crop_size, sample.focal_length,
                                                     static_cast<float>(sample.img_w),
                                                     static_cast<float>(sample.img_h));
            auto trans_base = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                            torch::TensorOptions().device(device).dtype(torch::kFloat)).unsqueeze(0);

            auto pose_flat = pose_base.view({1, 72});
            auto net_input = torch::cat({pose_flat, trans_base}, 1);
            const float time_val = static_cast<float>(sample_index) /
                                   static_cast<float>(std::max<size_t>(1, samples.size() - 1));
            auto time_tensor = torch::tensor({time_val}, torch::TensorOptions().device(device).dtype(torch::kFloat)).unsqueeze(0);

            auto deltas = pose_refiner.forward(net_input, time_tensor);
            auto pose_delta = deltas.slice(1, 0, 72).view({1, 24, 3});
            auto trans_delta = deltas.slice(1, 72, 75).view({1, 3});

            auto pose = pose_base + pose_delta;
            auto trans = (trans_base + trans_delta).squeeze(0);

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

int main(int argc, char *argv[]) {
    std::cout << "Starting Full Program Profiling..." << std::endl;

    // The RecordProfile guard automatically starts profiling.
    // When this 'guard' variable goes out of scope (at the closing brace),
    // it automatically stops profiling and saves the file to "trace.json".
    {
        // Use default config (CPU + CUDA)
        torch::autograd::profiler::RecordProfile guard("full_profile.json");
        
        try {
            run_real_training(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "Exception during profiling: " << e.what() << std::endl;
        }
        
        // 'guard' is destroyed here -> "full_profile.json" is written to disk.
    }

    std::cout << "Profiling complete. Saved to full_profile.json" << std::endl;
 
    return 0;
}