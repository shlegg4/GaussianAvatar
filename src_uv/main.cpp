#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/csrc/autograd/profiler.h>

#include "GaussianRasterizer.h"
#include "SharedGaussian.h"
#include "utils/HmrInferenceUtils.h"
#include "utils/HmrMathHelpers.h"
#include "utils/SmplxLBS.h"
#include "utils/image/MaskUtils.h"
#include "utils/image/TensorCvUtils.h"
#include "utils/io/PathUtils.h"
#include "utils/math/LossFunctions.h"
#include "utils/math/StatsUtils.h"
#include "utils/render/GaussianUtils.h"
#include "utils/render/RenderMathUtils.h"
#include "utils/train/TrainCache.h"
#include "utils/train/GaussianDataLoader.h"
#include "utils/train/TrainImageSaver.h"
#include "utils/train/TrainJsonl.h"
#include "utils/train/TrainTypes.h"
#include "utils/train/ViewerExport.h"

namespace
{

constexpr int64_t kSmplPoseParamCount = 72;
constexpr int64_t kSmplPoseJointCount = 24;
constexpr int64_t kSmplxExpressionParamCount = 250;
constexpr int64_t kSmplxJawPoseParamCount = 3;
constexpr int64_t kSmplxEyePoseParamCount = 6;
constexpr int64_t kSmplxHandPoseParamCount = 45;

void AppendPaddedVector(const std::vector<float> &source, size_t target_count, std::vector<float> *dest)
{
    if (dest == nullptr)
    {
        return;
    }
    const size_t copy_count = std::min(source.size(), target_count);
    dest->insert(dest->end(), source.begin(), source.begin() + static_cast<std::ptrdiff_t>(copy_count));
    dest->insert(dest->end(), target_count - copy_count, 0.0f);
}

torch::Tensor TensorFromPaddedVector(const std::vector<float> &source,
                                     int64_t target_count,
                                     torch::Device device)
{
    std::vector<float> padded;
    padded.reserve(static_cast<size_t>(target_count));
    AppendPaddedVector(source, static_cast<size_t>(target_count), &padded);
    return torch::from_blob(padded.data(),
                            {1, target_count},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

bool SamplesUseSmplx(const std::vector<TrainSample> &samples)
{
    for (const auto &sample : samples)
    {
        if (sample.uses_smplx)
        {
            return true;
        }
    }
    return false;
}

std::filesystem::path FindAncestorFile(std::filesystem::path start_dir,
                                       const std::string &filename,
                                       int max_levels = 6)
{
    if (filename.empty())
    {
        return {};
    }

    if (start_dir.empty())
    {
        start_dir = std::filesystem::current_path();
    }

    std::error_code ec;
    start_dir = std::filesystem::absolute(start_dir, ec);
    if (ec)
    {
        ec.clear();
        start_dir = std::filesystem::current_path();
    }

    for (int level = 0; level <= max_levels; ++level)
    {
        const std::filesystem::path candidate = start_dir / filename;
        if (std::filesystem::exists(candidate, ec))
        {
            return candidate;
        }
        ec.clear();

        const std::filesystem::path parent = start_dir.parent_path();
        if (parent.empty() || parent == start_dir)
        {
            break;
        }
        start_dir = parent;
    }

    return {};
}

std::string ResolveAvatarModelPath(const std::string &requested_path)
{
    if (requested_path.empty())
    {
        return requested_path;
    }

    std::error_code ec;
    std::filesystem::path path(requested_path);
    if (std::filesystem::exists(path, ec))
    {
        return std::filesystem::absolute(path, ec).string();
    }
    ec.clear();

    if (path.is_relative())
    {
        const std::filesystem::path relative_candidate = std::filesystem::current_path() / path;
        if (std::filesystem::exists(relative_candidate, ec))
        {
            return std::filesystem::absolute(relative_candidate, ec).string();
        }
        ec.clear();
    }

    const std::filesystem::path ancestor_candidate =
        FindAncestorFile(std::filesystem::current_path(), path.filename().string());
    if (!ancestor_candidate.empty())
    {
        return ancestor_candidate.string();
    }

    return requested_path;
}

} // namespace

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
                if (key != "--viewer" && key != "--headless" &&
                    key != "--viewer-stream-poses" &&
                    key != "--verbose" && key != "--verbose-diagnostics")
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
            else if (key == "--max-gaussians" || key == "--viewer-max-gaussians")
                options->max_gaussians = std::stoi(value);
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
            else if (key == "--sh-reg" || key == "--sh-reg-weight" || key == "--shregweight")
                options->sh_reg_weight = std::stof(value);
            else if (key == "--sugar-reg" || key == "--sugar-weight" || key == "--sugar-reg-weight" || key == "--sugarweight")
                options->sugar_weight = std::stof(value);
            else if (key == "--anisotropic-reg")
                options->anisotropic_reg_weight = std::stof(value);
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
            else if (key == "--betas-lr")
                options->betas_lr = std::stof(value);
            else if (key == "--betas-reg" || key == "--betas-reg-weight")
                options->betas_reg_weight = std::stof(value);
            else if (key == "--expr-bias-lr")
                options->expr_bias_lr = std::stof(value);
            else if (key == "--expr-bias-reg" || key == "--expr-bias-reg-weight")
                options->expr_bias_reg_weight = std::stof(value);
            else if (key == "--alpha-loss")
                options->alpha_loss_weight = std::stof(value);
            else if (key == "--opacity-reg" || key == "--opacity-reg-weight")
                options->opacity_reg_weight = std::stof(value);
            else if (key == "--lambda-dssim")
                options->lambda_dssim = std::stof(value);
            else if (key == "--color-lr")
                options->color_lr = std::stof(value);
            else if (key == "--opacity-lr")
                options->opacity_lr = std::stof(value);
            else if (key == "--psr-opacity-threshold")
                options->psr_opacity_threshold = std::stof(value);
            else if (key == "--psr-samples-per-gaussian")
                options->psr_samples_per_gaussian = std::stoi(value);
            else if (key == "--sh-degree")
                options->sh_degree = std::stoi(value);
            else if (key == "--viewer")
                options->enable_viewer = true;
            else if (key == "--headless")
                options->enable_viewer = false;
            else if (key == "--viewer-stream-poses")
                options->viewer_stream_poses = true;
            else if (key == "--verbose" || key == "--verbose-diagnostics")
                options->verbose_diagnostics = true;
            else if (key == "--verbose-every")
                options->verbose_every = std::stoi(value);
            else if (key == "--viewer-every")
                options->viewer_every = std::stoi(value);
            else if (key == "--viewer-shm")
                options->viewer_shm_name = value;
            else if (key == "--viewer-pose-shm")
                options->viewer_pose_shm_name = value;
            else if (key == "--viewer-bind-shm")
                options->viewer_bind_shm_name = value;
            else if (key == "--mesh-method")
                options->mesh_method = value;
            else if (key == "--loader-workers")
                options->loader_workers = std::stoi(value);
            else if (key == "--loader-prefetch")
                options->loader_prefetch_batches = std::stoi(value);
            else if (key == "--torch-cpu-threads")
                options->torch_cpu_threads = std::stoi(value);
            else if (key == "--torch-interop-threads")
                options->torch_interop_threads = std::stoi(value);
            else if (key == "--opencv-threads")
                options->opencv_threads = std::stoi(value);
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
    if (options->psr_opacity_threshold < 0.0f || options->psr_opacity_threshold > 1.0f)
    {
        std::cerr << "Invalid --psr-opacity-threshold (supported: 0.0-1.0)." << std::endl;
        return false;
    }
    if (options->psr_samples_per_gaussian < 0)
    {
        std::cerr << "Invalid --psr-samples-per-gaussian (must be >= 0)." << std::endl;
        return false;
    }
    if (options->mesh_method != "tsdf" &&
        options->mesh_method != "poisson" &&
        options->mesh_method != "uv")
    {
        std::cerr << "Invalid --mesh-method (supported: tsdf|poisson|uv)." << std::endl;
        return false;
    }
    if (options->loader_workers < 1)
    {
        std::cerr << "Invalid --loader-workers (must be >= 1)." << std::endl;
        return false;
    }
    if (options->loader_prefetch_batches < 1)
    {
        std::cerr << "Invalid --loader-prefetch (must be >= 1)." << std::endl;
        return false;
    }
    if (options->torch_cpu_threads < 1)
    {
        std::cerr << "Invalid --torch-cpu-threads (must be >= 1)." << std::endl;
        return false;
    }
    if (options->torch_interop_threads < 1)
    {
        std::cerr << "Invalid --torch-interop-threads (must be >= 1)." << std::endl;
        return false;
    }
    if (options->opencv_threads < 0)
    {
        std::cerr << "Invalid --opencv-threads (must be >= 0)." << std::endl;
        return false;
    }
    if (options->betas_lr < 0.0f)
    {
        std::cerr << "Invalid --betas-lr (must be >= 0)." << std::endl;
        return false;
    }
    if (options->betas_reg_weight < 0.0f)
    {
        std::cerr << "Invalid --betas-reg (must be >= 0)." << std::endl;
        return false;
    }
        if (options->expr_bias_lr < 0.0f)
        {
            std::cerr << "Invalid --expr-bias-lr (must be >= 0)." << std::endl;
            return false;
        }
        if (options->expr_bias_reg_weight < 0.0f)
        {
            std::cerr << "Invalid --expr-bias-reg (must be >= 0)." << std::endl;
            return false;
        }
    return true;
}

struct TrainDataGPU
{
    torch::Tensor all_poses; // (N, 72)
    torch::Tensor all_trans; // (N, 3)
    torch::Tensor all_expression; // (N, 250)
    torch::Tensor all_jaw_pose; // (N, 3)
    torch::Tensor all_eye_pose; // (N, 6)
    torch::Tensor all_left_hand_pose; // (N, 45)
    torch::Tensor all_right_hand_pose; // (N, 45)
    torch::Tensor all_time;  // (N, 1)
    torch::Tensor all_crops; // (N, 3) -> [cx, cy, size] normalized

    TrainDataGPU(const std::vector<TrainSample> &samples, torch::Device device)
    {
        const int64_t N = static_cast<int64_t>(samples.size());
        std::vector<float> flat_poses;
        std::vector<float> flat_trans;
        std::vector<float> flat_expression;
        std::vector<float> flat_jaw_pose;
        std::vector<float> flat_eye_pose;
        std::vector<float> flat_left_hand_pose;
        std::vector<float> flat_right_hand_pose;
        std::vector<float> flat_time;
        std::vector<float> flat_crops;
        flat_poses.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplPoseParamCount));
        flat_trans.reserve(static_cast<size_t>(N) * 3u);
        flat_expression.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxExpressionParamCount));
        flat_jaw_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxJawPoseParamCount));
        flat_eye_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxEyePoseParamCount));
        flat_left_hand_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxHandPoseParamCount));
        flat_right_hand_pose.reserve(static_cast<size_t>(N) * static_cast<size_t>(kSmplxHandPoseParamCount));
        flat_time.reserve(static_cast<size_t>(N));
        flat_crops.reserve(static_cast<size_t>(N) * 3u);

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

            AppendPaddedVector(s.smplx_expression, static_cast<size_t>(kSmplxExpressionParamCount), &flat_expression);
            AppendPaddedVector(s.smplx_jaw_pose, static_cast<size_t>(kSmplxJawPoseParamCount), &flat_jaw_pose);
            AppendPaddedVector(s.smplx_eye_pose, static_cast<size_t>(kSmplxEyePoseParamCount), &flat_eye_pose);
            AppendPaddedVector(s.smplx_left_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount), &flat_left_hand_pose);
            AppendPaddedVector(s.smplx_right_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount), &flat_right_hand_pose);

            cv::Vec3f t;
            if (s.has_translation)
            {
                t = cv::Vec3f(s.translation[0], s.translation[1], s.translation[2]);
            }
            else
            {
                t = EstimateTranslation(s.cam, s.crop_cx, s.crop_cy,
                                        s.crop_size, s.focal_length,
                                        static_cast<float>(s.img_w), static_cast<float>(s.img_h));
            }
            flat_trans.push_back(t[0]);
            flat_trans.push_back(t[1]);
            flat_trans.push_back(t[2]);

            const float img_w = std::max(1.0f, static_cast<float>(s.img_w));
            const float img_h = std::max(1.0f, static_cast<float>(s.img_h));
            const float norm_cx = s.crop_cx / img_w;
            const float norm_cy = s.crop_cy / img_h;
            const float norm_size = s.crop_size / img_w;
            flat_crops.push_back(norm_cx);
            flat_crops.push_back(norm_cy);
            flat_crops.push_back(norm_size);

            const float time_val = static_cast<float>(i) /
                                   static_cast<float>(std::max<int64_t>(1, N - 1));
            flat_time.push_back(time_val);
        }

        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        all_poses = torch::from_blob(flat_poses.data(), {N, kSmplPoseParamCount}, opts).clone().to(device);
        all_trans = torch::from_blob(flat_trans.data(), {N, 3}, opts).clone().to(device);
        all_expression = torch::from_blob(flat_expression.data(), {N, kSmplxExpressionParamCount}, opts).clone().to(device);
        all_jaw_pose = torch::from_blob(flat_jaw_pose.data(), {N, kSmplxJawPoseParamCount}, opts).clone().to(device);
        all_eye_pose = torch::from_blob(flat_eye_pose.data(), {N, kSmplxEyePoseParamCount}, opts).clone().to(device);
        all_left_hand_pose = torch::from_blob(flat_left_hand_pose.data(), {N, kSmplxHandPoseParamCount}, opts).clone().to(device);
        all_right_hand_pose = torch::from_blob(flat_right_hand_pose.data(), {N, kSmplxHandPoseParamCount}, opts).clone().to(device);
        all_time = torch::from_blob(flat_time.data(), {N, 1}, opts).clone().to(device);
        all_crops = torch::from_blob(flat_crops.data(), {N, 3}, opts).clone().to(device);
    }
};

struct GaussianAvatar : torch::nn::Module
{
    static constexpr int kUvSize = 512;

    std::shared_ptr<SMPLXLayer> smplx;
    torch::Tensor g_scales, g_rots, g_opacities, g_colors, g_offsets, g_sh;
    torch::Tensor g_bary_coords, g_face_indices, faces_buffer;
    torch::Tensor valid_uv_mask, valid_uv_indices;
    torch::Tensor current_flat_offsets, current_flat_scales, current_flat_rots;
    torch::Tensor current_flat_opacities, current_flat_colors, current_flat_sh;
    torch::Tensor v_template_cached, faces_cached;
    int sh_degree_ = 0;

    GaussianAvatar(const std::string &model_path)
    {
        smplx = std::make_shared<SMPLXLayer>(model_path);
        register_module("smplx", smplx);
    }

    void init_uv_mapping(const std::string &smpl_uv_obj_path)
    {
        auto smpl_face_uvs = LoadSmplUVsFromOBJ(smpl_uv_obj_path);
        if (!smpl_face_uvs.defined() || smpl_face_uvs.dim() != 3 || smpl_face_uvs.size(1) != 3 || smpl_face_uvs.size(2) != 2)
        {
            throw std::runtime_error("Failed to load SMPL UV mapping from: " + smpl_uv_obj_path);
        }

        auto uvs_cpu = smpl_face_uvs.to(torch::kCPU).to(torch::kFloat32).contiguous();
        const float *uv_ptr = uvs_cpu.data_ptr<float>();

        const int64_t uv_face_count = uvs_cpu.size(0);
        const int64_t mesh_face_count = faces_cached.size(0);
        const int64_t face_count = std::min(uv_face_count, mesh_face_count);
        if (face_count <= 0)
        {
            throw std::runtime_error("UV mapping has no faces.");
        }

        std::vector<uint8_t> mask_values(static_cast<size_t>(kUvSize) * static_cast<size_t>(kUvSize), 0u);
        std::vector<int64_t> flat_indices;
        std::vector<int64_t> face_ids;
        std::vector<float> bary_values;
        flat_indices.reserve(mask_values.size() / 2);
        face_ids.reserve(mask_values.size() / 2);
        bary_values.reserve((mask_values.size() / 2) * 3u);

        auto edge_fn = [](float ax, float ay, float bx, float by, float px, float py)
        {
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
        };

        constexpr float kInsideEps = 1e-6f;
        for (int64_t face_idx = 0; face_idx < face_count; ++face_idx)
        {
            const float *tri = uv_ptr + (face_idx * 3 * 2);
            const float x0 = tri[0] * static_cast<float>(kUvSize - 1);
            const float y0 = (1.0f - tri[1]) * static_cast<float>(kUvSize - 1);
            const float x1 = tri[2] * static_cast<float>(kUvSize - 1);
            const float y1 = (1.0f - tri[3]) * static_cast<float>(kUvSize - 1);
            const float x2 = tri[4] * static_cast<float>(kUvSize - 1);
            const float y2 = (1.0f - tri[5]) * static_cast<float>(kUvSize - 1);

            const float denom = edge_fn(x0, y0, x1, y1, x2, y2);
            if (std::abs(denom) < 1e-8f)
            {
                continue;
            }

            const int min_x = std::clamp(static_cast<int>(std::floor(std::min({x0, x1, x2}))), 0, kUvSize - 1);
            const int max_x = std::clamp(static_cast<int>(std::ceil(std::max({x0, x1, x2}))), 0, kUvSize - 1);
            const int min_y = std::clamp(static_cast<int>(std::floor(std::min({y0, y1, y2}))), 0, kUvSize - 1);
            const int max_y = std::clamp(static_cast<int>(std::ceil(std::max({y0, y1, y2}))), 0, kUvSize - 1);

            for (int py = min_y; py <= max_y; ++py)
            {
                for (int px = min_x; px <= max_x; ++px)
                {
                    const float sx = static_cast<float>(px) + 0.5f;
                    const float sy = static_cast<float>(py) + 0.5f;

                    const float w0 = edge_fn(x1, y1, x2, y2, sx, sy) / denom;
                    const float w1 = edge_fn(x2, y2, x0, y0, sx, sy) / denom;
                    const float w2 = 1.0f - w0 - w1;

                    if (w0 < -kInsideEps || w1 < -kInsideEps || w2 < -kInsideEps)
                    {
                        continue;
                    }

                    const int64_t linear_idx = static_cast<int64_t>(py) * static_cast<int64_t>(kUvSize) + static_cast<int64_t>(px);
                    if (mask_values[static_cast<size_t>(linear_idx)] != 0u)
                    {
                        continue;
                    }

                    mask_values[static_cast<size_t>(linear_idx)] = 1u;
                    flat_indices.push_back(linear_idx);
                    face_ids.push_back(face_idx);
                    bary_values.push_back(w0);
                    bary_values.push_back(w1);
                    bary_values.push_back(w2);
                }
            }
        }

        if (face_ids.empty())
        {
            throw std::runtime_error("UV mapping rasterization produced zero valid UV pixels.");
        }

        auto device = smplx->v_template.device();
        valid_uv_mask = torch::from_blob(mask_values.data(), {kUvSize, kUvSize}, torch::TensorOptions().dtype(torch::kUInt8))
                            .clone()
                            .to(device)
                            .to(torch::kBool);
        valid_uv_indices = torch::from_blob(flat_indices.data(), {static_cast<int64_t>(flat_indices.size())}, torch::TensorOptions().dtype(torch::kLong))
                               .clone()
                               .to(device);
        g_face_indices = torch::from_blob(face_ids.data(), {static_cast<int64_t>(face_ids.size())}, torch::TensorOptions().dtype(torch::kLong))
                             .clone()
                             .to(device);
        g_bary_coords = torch::from_blob(bary_values.data(), {static_cast<int64_t>(face_ids.size()), 3}, torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(device);

        register_buffer("valid_uv_mask", valid_uv_mask);
        register_buffer("valid_uv_indices", valid_uv_indices);
        register_buffer("g_face_indices", g_face_indices);
        register_buffer("g_bary_coords", g_bary_coords);

        std::cout << "UV mapping initialized with " << face_ids.size()
                  << " valid UV pixels at " << kUvSize << "x" << kUvSize << "." << std::endl;
    }

    void init_gaussians(torch::Tensor faces_idx,
                        int sh_degree,
                        const std::string &smpl_uv_obj_path)
    {
        auto device = smplx->v_template.device();
        sh_degree_ = std::max(0, sh_degree);
        v_template_cached = smplx->v_template.clone().detach();
        faces_cached = faces_idx.clone().detach();
        faces_buffer = faces_cached.to(device);

        init_uv_mapping(smpl_uv_obj_path);

        auto uv_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        g_offsets = torch::zeros({3, kUvSize, kUvSize}, uv_opts).set_requires_grad(true);
        g_scales = torch::full({3, kUvSize, kUvSize}, std::log(0.002f), uv_opts).set_requires_grad(true);
        g_rots = torch::zeros({4, kUvSize, kUvSize}, uv_opts).set_requires_grad(true);
        g_opacities = torch::full({1, kUvSize, kUvSize}, 0.95f, uv_opts).set_requires_grad(true);
        g_colors = torch::full({3, kUvSize, kUvSize}, 0.5f, uv_opts).set_requires_grad(true);
        {
            torch::NoGradGuard no_grad;
            g_rots.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0f);
        }

        if (sh_degree > 0)
        {
            const int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
            g_sh = torch::zeros({sh_coeffs * 3, kUvSize, kUvSize}, uv_opts).set_requires_grad(true);
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

    int64_t NumGaussians() const
    {
        return g_face_indices.defined() ? g_face_indices.size(0) : 0;
    }

    void UpdateFlatParams(int sh_degree)
    {
        if (!valid_uv_indices.defined())
        {
            return;
        }

        current_flat_offsets = g_offsets.view({3, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_scales = g_scales.view({3, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_rots = g_rots.view({4, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_opacities = g_opacities.view({1, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();
        current_flat_colors = g_colors.view({3, -1}).index_select(1, valid_uv_indices).transpose(0, 1).contiguous();

        if (sh_degree > 0 && g_sh.defined() && g_sh.numel() > 0)
        {
            const int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
            current_flat_sh = g_sh.view({sh_coeffs, 3, -1})
                                  .index_select(2, valid_uv_indices)
                                  .permute({2, 0, 1})
                                  .contiguous();
        }
        else
        {
            current_flat_sh = torch::zeros({0}, g_colors.options());
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> get_bone_data()
    {
        auto device = g_bary_coords.device();
         
        auto v_weights = smplx->weights.to(device); 
        
        auto selected_faces = faces_buffer.index_select(0, g_face_indices);
        
        auto v0 = selected_faces.select(1, 0);
        auto v1 = selected_faces.select(1, 1);
        auto v2 = selected_faces.select(1, 2);
        
        auto w0 = v_weights.index_select(0, v0);
        auto w1 = v_weights.index_select(0, v1);
        auto w2 = v_weights.index_select(0, v2);
        
        auto u = g_bary_coords.select(1, 0).unsqueeze(1);
        auto v = g_bary_coords.select(1, 1).unsqueeze(1);
        auto w = g_bary_coords.select(1, 2).unsqueeze(1);
        
        // Interpolate the bone weights for the splat's exact position on the face
        auto splat_weights = u * w0 + v * w1 + w * w2;
        
        // Extract the Top 4 most influential bones
        auto topk = torch::topk(splat_weights, 4, 1);
        auto top_weights = std::get<0>(topk);
        auto top_indices = std::get<1>(topk).to(torch::kFloat32); // Must be float for the .bin buffer
        
        // Normalize so the 4 weights always sum exactly to 1.0
        auto sum_weights = top_weights.sum(1, true);
        top_weights = top_weights / torch::clamp_min(sum_weights, 1e-6f);
        
        return {top_indices, top_weights};
    }


    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor betas,
                                                     torch::Tensor expression,
                                                     torch::Tensor pose,
                                                     torch::Tensor jaw_pose,
                                                     torch::Tensor eye_pose,
                                                     torch::Tensor left_hand_pose,
                                                     torch::Tensor right_hand_pose,
                                                     torch::Tensor trans)
    {
        using torch::indexing::Slice;

        auto smplx_out = smplx->forward(betas, expression, pose, jaw_pose, eye_pose, left_hand_pose, right_hand_pose, trans);
        auto verts_posed = smplx_out.vertices;
        const int64_t batch_count = verts_posed.size(0);
        const int64_t gaussian_count = g_face_indices.size(0);

        UpdateFlatParams(sh_degree_);

        auto selected_faces = faces_buffer.index_select(0, g_face_indices);
        auto face_i0 = selected_faces.index({Slice(), 0});
        auto face_i1 = selected_faces.index({Slice(), 1});
        auto face_i2 = selected_faces.index({Slice(), 2});

        auto A = verts_posed.index_select(1, face_i0);
        auto B = verts_posed.index_select(1, face_i1);
        auto C = verts_posed.index_select(1, face_i2);

        auto verts_canon = v_template_cached.to(verts_posed.device());
        auto A_can = verts_canon.index_select(0, face_i0);
        auto B_can = verts_canon.index_select(0, face_i1);
        auto C_can = verts_canon.index_select(0, face_i2);

        auto R_canon = ComputeTriFrames(A_can, B_can, C_can);
        auto X = torch::nn::functional::normalize(B - A, torch::nn::functional::NormalizeFuncOptions().dim(2));
        auto N = torch::cross(B - A, C - A, 2);
        N = torch::nn::functional::normalize(N, torch::nn::functional::NormalizeFuncOptions().dim(2));
        auto Y = torch::cross(N, X, 2);
        auto R_posed = torch::stack({X, Y, N}, 3);

        auto R_posed_flat = R_posed.reshape({-1, 3, 3});
        auto offset_x = current_flat_offsets.index({Slice(), 0}).view({1, gaussian_count, 1});
        auto offset_y = current_flat_offsets.index({Slice(), 1}).view({1, gaussian_count, 1});
        auto offset_z = current_flat_offsets.index({Slice(), 2}).view({1, gaussian_count, 1});
        auto posed_offsets = (X * offset_x) + (Y * offset_y) + (N * offset_z);

        // Re-calculate Skinning Rotation for the Gaussian Orientation (Quaternions)
        // We still need R_skin for the ROTATION of the Gaussian splat itself
        auto R_canon_batch = R_canon.unsqueeze(0).expand({batch_count, gaussian_count, 3, 3});
        auto R_canon_flat = R_canon_batch.reshape({-1, 3, 3});
        auto R_skin_flat = torch::bmm(R_posed_flat, R_canon_flat.transpose(1, 2));

        auto R_skin = R_skin_flat.view({batch_count, gaussian_count, 3, 3});

        auto u = g_bary_coords.index({Slice(), 0}).view({1, gaussian_count, 1});
        auto v = g_bary_coords.index({Slice(), 1}).view({1, gaussian_count, 1});
        auto w = g_bary_coords.index({Slice(), 2}).view({1, gaussian_count, 1});

        auto skinned_pos = u * A + v * B + w * C;
        auto final_pos = skinned_pos + posed_offsets;

        auto q_skin_flat = MatrixToQuat(R_skin_flat);
        auto base_rots = current_flat_rots.unsqueeze(0).expand({batch_count, gaussian_count, 4}).reshape({-1, 4});
        auto final_rot = QuatMultiply(q_skin_flat, base_rots).view({batch_count, gaussian_count, 4});

        if (batch_count == 1)
        {
            return {final_pos.squeeze(0), final_rot.squeeze(0)};
        }

        return {final_pos, final_rot};
    }
};

struct PoseRefiner : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    const float ROT_SCALE = 0.5f;
    const float TRANS_XY_SCALE = 0.25f;
    const float TRANS_Z_SCALE = 0.01f;

    PoseRefiner(int input_dim = 72 + 3 + 3)
    {
        const int embed_dim = 4;
        const int total_input = input_dim + embed_dim;
        const int hidden_dim = 128;

        fc1 = register_module("fc1", torch::nn::Linear(total_input, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, input_dim - 3));

        torch::nn::init::xavier_uniform_(fc1->weight);
        torch::nn::init::zeros_(fc1->bias);
        torch::nn::init::zeros_(fc3->weight);
        torch::nn::init::zeros_(fc3->bias);
    }

    torch::Tensor forward(torch::Tensor noisy_pose_trans, torch::Tensor crop_params, torch::Tensor time_norm)
    {
        auto device = time_norm.device();
        auto freqs = torch::pow(2.0, torch::arange(0, 2, torch::TensorOptions().device(device)).to(torch::kFloat));
        auto time_projected = time_norm * freqs.unsqueeze(0) * (2.0f * 3.14159f);
        auto time_embed = torch::cat({torch::sin(time_projected), torch::cos(time_projected)}, 1);

        auto x = torch::cat({noisy_pose_trans, crop_params, time_embed}, 1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto delta = fc3->forward(x);

        auto delta_pose = torch::tanh(delta.slice(1, 0, 72)) * ROT_SCALE;

        auto delta_trans_raw = delta.slice(1, 72, 75);

        auto trans_x = delta_trans_raw.slice(1, 0, 1) * TRANS_XY_SCALE;
        auto trans_y = delta_trans_raw.slice(1, 1, 2) * TRANS_XY_SCALE;
        auto trans_z = delta_trans_raw.slice(1, 2, 3) * TRANS_Z_SCALE;

        auto delta_trans = torch::cat({trans_x, trans_y, trans_z}, 1);
        return torch::cat({delta_pose, delta_trans}, 1);
    }
};

class AsyncMetricLogger
{
public:
    AsyncMetricLogger()
    {
        worker_ = std::thread(&AsyncMetricLogger::WorkerLoop, this);
    }

    ~AsyncMetricLogger()
    {
        Stop();
    }

    AsyncMetricLogger(const AsyncMetricLogger &) = delete;
    AsyncMetricLogger &operator=(const AsyncMetricLogger &) = delete;

    void Log(std::string line)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            lines_.push(std::move(line));
        }
        cv_.notify_one();
    }

private:
    void Stop()
    {
        stop_requested_.store(true);
        cv_.notify_all();
        if (worker_.joinable())
        {
            worker_.join();
        }
    }

    void WorkerLoop()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]()
                     { return stop_requested_.load() || !lines_.empty(); });
            if (lines_.empty())
            {
                if (stop_requested_.load())
                {
                    break;
                }
                continue;
            }
            std::string line = std::move(lines_.front());
            lines_.pop();
            lock.unlock();
            std::cout << line << std::endl;
        }
    }

    std::atomic<bool> stop_requested_{false};
    std::queue<std::string> lines_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
};

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
            auto y_sign = batch.y_signs.index({static_cast<int64_t>(k)});
 

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

        // Sobel edge consistency on composited images/targets (RGB depthwise).
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
        // Keep normal offsets bounded, lightly discourage tangent drift, and prevent global bias drift.
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
            // ReLU ensures we ONLY penalize scales that are larger than scale_cap_
            // Squaring it makes the penalty grow exponentially the further it exceeds the cap
            auto excess = torch::relu(current_scales - (scale_cap_ * render_scale_modifier_));
            cap_penalty = excess.pow(2).mean();
        }

        using torch::indexing::Slice;

        // 1) Skin thinness: keep one axis small, but allow orientation freedom.
        auto z_scale = current_scales.index({Slice(), 2});
        auto skin_thinness_loss = z_scale.pow(2).mean();

        // Compute TV only on edges fully inside valid UV islands.
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
        // Regularize actual scales (exp-domain) with a lighter weight to avoid freezing.
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
            using torch::indexing::Slice;
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

        // if (avatar_.g_scales.grad().defined())
        // {
        //     torch::NoGradGuard no_grad;
        //     avatar_.g_scales.grad().mul_(torch::tensor({1.0f, 1.0f, 0.0f}, avatar_.g_scales.options()));
        // }

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

            // float target_thickness = 0.001f;
            // float target_log_scale = std::log(target_thickness);

            // using torch::indexing::Slice;
            // // Set all Z-scales (index 2) to the target thickness
            // avatar_.g_scales.index_put_({Slice(), 2}, target_log_scale);
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
        // Remove the torch::clamp so gradients can always flow back!
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

int run_real_training(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: gaussian_train --jsonl <path> [--smpl <path>] [--num-gaussians <int>]"
                     " [--max-gaussians <int>]"
                     " [--epochs <int>] [--lr <float>] [--output-dir <path>]"
                     " [--lr-decay-epoch <int>] [--lr-decay-multiplier <float>]"
                     " [--lr-min-multiplier <float>]"
                     " [--train-dir <path>] [--viewer-export-dir <path>]"
                     " [--scale-reg <float>] [--sh-reg <float>] [--sugar-reg <float>] [--anisotropic-reg <float>] [--scale-max <float>]"
                     " [--rot-lr <float>] [--offset-lr <float>]"
                     " [--offset-reg <float>] [--pose-reg <float>] [--pose-lr <float>] [--betas-lr <float>] [--betas-reg <float>]"
                     " [--expr-bias-lr <float>] [--expr-bias-reg <float>] [--alpha-loss <float>] [--opacity-reg <float>]"
                     " [--lambda-dssim <float>]"
                     " [--color-lr <float>] [--opacity-lr <float>]"
                     " [--psr-opacity-threshold <float>]"
                     " [--psr-samples-per-gaussian <int>]"
                     " [--sh-degree <int>]"
                     " [--loader-workers <int>] [--loader-prefetch <int>]"
                     " [--torch-cpu-threads <int>] [--torch-interop-threads <int>] [--opencv-threads <int>]"
                     " [--verbose|--verbose-diagnostics] [--verbose-every <int>]"
                     " [--mesh-method <tsdf|poisson|uv>]"
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
    std::string avatar_model_path = options.smpl_model_path;
    const int num_gaussians = options.num_gaussians;
    const int max_gaussians = options.max_gaussians;
    const int epochs = options.epochs;
    const float lr = options.lr;
    const int lr_decay_epoch = options.lr_decay_epoch;
    const float lr_decay_multiplier = options.lr_decay_multiplier;
    const float lr_min_multiplier = options.lr_min_multiplier;
    const std::string &output_dir = options.output_dir;
    const std::string &viewer_export_dir = options.viewer_export_dir;
    const float scale_reg_weight = options.scale_reg_weight;
    const float sh_reg_weight = options.sh_reg_weight;
    const float sugar_weight = options.sugar_weight;
    const float anisotropic_reg_weight = options.anisotropic_reg_weight;
    const float scale_lr = (options.scale_lr < 0.0f) ? lr : options.scale_lr;
    const float scale_max_value = options.scale_max_value;
    const float rot_lr = (options.rot_lr < 0.0f) ? lr : options.rot_lr;
    const float offset_lr = (options.offset_lr < 0.0f) ? lr : options.offset_lr;
    const float offset_reg_weight = options.offset_reg_weight;
    const float pose_reg_weight = options.pose_reg_weight;
    const float pose_lr = options.pose_lr;
    const float betas_lr = options.betas_lr;
    const float betas_reg_weight = options.betas_reg_weight;
    const float expr_bias_lr = options.expr_bias_lr;
    const float expr_bias_reg_weight = options.expr_bias_reg_weight;
    const float color_lr = options.color_lr;
    const float opacity_lr = (options.opacity_lr < 0.0f) ? lr : options.opacity_lr;
    const float psr_opacity_threshold = options.psr_opacity_threshold;
    const int psr_samples_per_gaussian = options.psr_samples_per_gaussian;
    const int sh_degree = options.sh_degree;
    const int viewer_every = std::max(1, options.viewer_every);
    const bool viewer_stream_poses = options.viewer_stream_poses;
    const bool verbose_diagnostics = options.verbose_diagnostics;
    const int verbose_log_every = std::max(1, options.verbose_every);
    const float outside_mask_weight = 0.1f;
    const float alpha_loss_weight = options.alpha_loss_weight;
    const float opacity_reg_weight = options.opacity_reg_weight;
    const float lambda_dssim = options.lambda_dssim;
    const float render_scale_modifier = 1.0f;
    const float render_threshold = 3.0f / 255.0f;
    const int batch_size = 4;
    const int loader_workers = std::max(1, options.loader_workers);
    const size_t loader_prefetch_batches = static_cast<size_t>(std::max(1, options.loader_prefetch_batches));
    const int metric_log_every = 100;
    const float safe_scale_mod = std::max(render_scale_modifier, 1e-6f);
    const float scale_cap = (scale_max_value > 0.0f) ? (scale_max_value / safe_scale_mod) : -1.0f;
    if (verbose_diagnostics)
    {
        std::cout << "Verbose diagnostics enabled (every " << verbose_log_every << " steps)." << std::endl;
    }
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

    at::globalContext().setBenchmarkCuDNN(true);
    at::set_num_threads(std::max(1, options.torch_cpu_threads));
    at::set_num_interop_threads(std::max(1, options.torch_interop_threads));
    cv::setNumThreads(std::max(0, options.opencv_threads));
    std::cout << "CPU thread settings: torch=" << std::max(1, options.torch_cpu_threads)
              << " interop=" << std::max(1, options.torch_interop_threads)
              << " opencv=" << std::max(0, options.opencv_threads)
              << " loader_workers=" << loader_workers
              << " loader_prefetch=" << loader_prefetch_batches
              << std::endl;

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
    const bool dataset_uses_smplx = SamplesUseSmplx(samples);
    const std::filesystem::path avatar_model_input_path(avatar_model_path);
    const std::string avatar_model_name = avatar_model_input_path.filename().string();
    if (avatar_model_name == "smpl_data.pt" || avatar_model_name == "smplx_libtorch.pt")
    {
        const std::filesystem::path replacement = avatar_model_input_path.has_parent_path()
                                                      ? (avatar_model_input_path.parent_path() / "smplx_data.pt")
                                                      : std::filesystem::path("smplx_data.pt");
        avatar_model_path = replacement.string();
        if (dataset_uses_smplx)
        {
            std::cout << "Detected SMPL-X dataset. Using " << avatar_model_path << " for the Gaussian avatar stage." << std::endl;
        }
        else
        {
            std::cout << "Using " << avatar_model_path << " for the Gaussian avatar stage." << std::endl;
        }
    }

    const std::string resolved_avatar_model_path = ResolveAvatarModelPath(avatar_model_path);
    if (resolved_avatar_model_path != avatar_model_path)
    {
        std::cout << "Resolved avatar model path to " << resolved_avatar_model_path << std::endl;
        avatar_model_path = resolved_avatar_model_path;
    }

    torch::Device device(torch::kCUDA);
    const int ssim_window_size = 11;
    torch::Tensor window_v, window_h;
    std::tie(window_v, window_h) = create_separable_windows(ssim_window_size, device);
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

        const int H = crop.rows;
        const int W = crop.cols;
        if (H <= 0 || W <= 0)
        {
            cached[i] = std::move(entry);
            continue;
        }
        if (crop.channels() != 3)
        {
            cached[i] = std::move(entry);
            continue;
        }

        auto matte_mask = LoadMatteMaskTensor(matte, W, H, torch::kCPU);
        if (!matte_mask.defined())
        {
            cached[i] = std::move(entry);
            continue;
        }

        matte_mask = matte_mask.pin_memory();

        entry.matte_mask = matte_mask;
        entry.crop_bgr = crop;
        entry.valid = true;
        cached[i] = std::move(entry);
    }

    std::filesystem::path smpl_uv_obj_path =
        std::filesystem::path(avatar_model_path).parent_path() /
        (dataset_uses_smplx ? "smplx_uv.obj" : "smpl_uv.obj");
    if (!std::filesystem::exists(smpl_uv_obj_path))
    {
        smpl_uv_obj_path = dataset_uses_smplx ? "smplx_uv.obj" : "smpl_uv.obj";
    }

    TrainDataGPU gpu_data(samples, device);

    GaussianAvatar avatar(avatar_model_path);
    avatar.to(device);
    avatar.init_gaussians(avatar.smplx->faces.to(device), options.sh_degree, smpl_uv_obj_path.string());
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
    canonical_betas.set_requires_grad(true);
    auto canonical_betas_anchor = canonical_betas.detach().clone();
    auto canonical_pose = torch::zeros({1, kSmplPoseJointCount, 3}, canonical_betas.options());
    auto canonical_trans = torch::zeros({1, 3}, canonical_betas.options());
    auto canonical_expression = torch::zeros({1, kSmplxExpressionParamCount}, canonical_betas.options());
    canonical_expression.set_requires_grad(true);
    auto canonical_jaw_pose = torch::zeros({1, kSmplxJawPoseParamCount}, canonical_betas.options());
    auto canonical_eye_pose = torch::zeros({1, kSmplxEyePoseParamCount}, canonical_betas.options());
    auto canonical_left_hand_pose = torch::zeros({1, kSmplxHandPoseParamCount}, canonical_betas.options());
    auto canonical_right_hand_pose = torch::zeros({1, kSmplxHandPoseParamCount}, canonical_betas.options());

    const bool use_sh = sh_degree > 0;
    auto cam_pos = torch::zeros({3}, torch::TensorOptions().device(device));

    const uint32_t shared_stride = shared_gaussian::kSharedStrideFloats + 7u +
                                   (use_sh ? static_cast<uint32_t>((sh_degree + 1) * (sh_degree + 1) * 3) : 0u);
    const int64_t max_u32 = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    const int64_t initial_gaussian_count = std::max<int64_t>(1, avatar.NumGaussians());
    int64_t requested_viewer_capacity = (max_gaussians > 0)
                                            ? static_cast<int64_t>(max_gaussians)
                                            : (initial_gaussian_count * 2);
    if (max_gaussians > 0 && max_gaussians < initial_gaussian_count)
    {
        std::cerr << "Warning: --max-gaussians (" << max_gaussians
                  << ") is smaller than UV Gaussian count (" << initial_gaussian_count
                  << "); clamping to start count." << std::endl;
    }
    requested_viewer_capacity = std::max<int64_t>(requested_viewer_capacity, initial_gaussian_count);
    requested_viewer_capacity = std::min<int64_t>(requested_viewer_capacity, max_u32);
    const uint32_t viewer_capacity = static_cast<uint32_t>(requested_viewer_capacity);
    shared_gaussian::SharedGaussianWriter shared_writer;
    bool publish_viewer = options.enable_viewer;
    const std::string bind_shm_name = options.viewer_bind_shm_name.empty()
                                          ? (options.viewer_shm_name + "_bind")
                                          : options.viewer_bind_shm_name;
    shared_gaussian::SharedBindWriter bind_writer;
    if (publish_viewer)
    {
        if (!shared_writer.Init(options.viewer_shm_name, viewer_capacity,
                                shared_stride, static_cast<uint32_t>(sh_degree),
                                render_scale_modifier))
        {
            std::cerr << "Failed to open shared memory mapping: " << options.viewer_shm_name << std::endl;
            publish_viewer = false;
        }
        if (publish_viewer)
        {
            std::cout << "Viewer shared memory capacity: " << viewer_capacity
                      << " gaussians (start: " << initial_gaussian_count << ")" << std::endl;
            const uint32_t betas_count = static_cast<uint32_t>(canonical_betas.numel());
            if (!bind_writer.Init(bind_shm_name, viewer_capacity,
                                  shared_gaussian::kSharedBindStrideFloats, betas_count))
            {
                std::cerr << "Failed to open bind shared memory mapping: " << bind_shm_name << std::endl;
            }
        }
    }
    std::vector<float> shared_buffer;
    uint64_t shared_frame = 0;
    bool shared_capacity_warned = false;
    bool bind_capacity_warned = false;
    std::vector<float> bind_buffer;
    std::vector<float> bind_betas_buffer;

    const int warmup_epochs = 20;

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
    optimizer.add_param_group({std::vector<torch::Tensor>{canonical_betas}});
    auto &betas_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(betas_group.options()).lr(betas_lr * lr_multiplier);
    optimizer.add_param_group({std::vector<torch::Tensor>{canonical_expression}});
    auto &expr_bias_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(expr_bias_group.options()).lr(expr_bias_lr * lr_multiplier);
    optimizer.add_param_group({pose_refiner.parameters()});
    auto &pose_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(pose_group.options()).lr(pose_lr * lr_multiplier);

    auto apply_lr_multiplier = [&](float pose_multiplier)
    {
        pose_lr_multiplier = pose_multiplier;
        static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[7].options()).lr(pose_lr * pose_lr_multiplier);
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
    AsyncMetricLogger metric_logger;
    GaussianTrainer trainer(avatar,
                            pose_refiner,
                            optimizer,
                            canonical_betas,
                            window_v,
                            window_h,
                            ssim_window_size,
                            cam_pos,
                            render_scale_modifier,
                            scale_cap,
                            batch_size,
                            use_sh,
                            render_threshold,
                            outside_mask_weight,
                            alpha_loss_weight,
                            lambda_dssim,
                            offset_reg_weight,
                            scale_reg_weight,
                            sh_reg_weight,
                            sugar_weight,
                            opacity_reg_weight,
                            anisotropic_reg_weight,
                            warmup_epochs,
                            betas_reg_weight,
                            canonical_betas_anchor,
                            canonical_expression,
                            expr_bias_reg_weight);
    GaussianDataLoader loader(samples,
                              cached,
                              gpu_data.all_poses,
                              gpu_data.all_trans,
                              gpu_data.all_expression,
                              gpu_data.all_jaw_pose,
                              gpu_data.all_eye_pose,
                              gpu_data.all_left_hand_pose,
                              gpu_data.all_right_hand_pose,
                              gpu_data.all_time,
                              gpu_data.all_crops,
                              std::vector<int64_t>{},
                              batch_size,
                              device,
                              loader_workers,
                              loader_prefetch_batches);

    auto save_uv_debug_maps = [&](int epoch_idx) -> int
    {
        using torch::indexing::Slice;

        std::error_code uv_ec;
        const std::filesystem::path uv_dir =
            out_dir_path / "pairs" / ("epoch_" + std::to_string(epoch_idx)) / "uv_maps";
        std::filesystem::create_directories(uv_dir, uv_ec);
        if (uv_ec)
        {
            std::cerr << "Failed to create UV debug directory: " << uv_dir.string() << std::endl;
            return 0;
        }

        auto save_map = [&](const std::string &filename, const torch::Tensor &input, bool assume_unit) -> bool
        {
            if (!input.defined() || input.numel() == 0)
            {
                return false;
            }

            torch::Tensor img = input.detach().to(torch::kFloat32).cpu();
            if (img.dim() == 2)
            {
                img = img.unsqueeze(0);
            }
            if (img.dim() != 3)
            {
                return false;
            }

            const int64_t c = img.size(0);
            if (c == 1)
            {
                img = img.repeat({3, 1, 1});
            }
            else if (c == 2)
            {
                auto z = torch::zeros({1, img.size(1), img.size(2)}, img.options());
                img = torch::cat({img, z}, 0);
            }
            else if (c > 3)
            {
                img = img.index({Slice(0, 3), Slice(), Slice()});
            }

            if (assume_unit)
            {
                img = img.clamp(0.0f, 1.0f);
            }
            else
            {
                const auto min_val = img.min().item<float>();
                const auto max_val = img.max().item<float>();
                const float span = max_val - min_val;
                if (span > 1e-8f)
                {
                    img = (img - min_val) / span;
                }
                else
                {
                    img = torch::zeros_like(img);
                }
            }

            const cv::Mat bgr = TensorToBgr(img);
            if (bgr.empty())
            {
                return false;
            }
            return cv::imwrite((uv_dir / filename).string(), bgr);
        };

        int saved_count = 0;
        if (save_map("valid_mask.png", avatar.valid_uv_mask.to(torch::kFloat32), true))
            ++saved_count;
        {
            auto offsets_vis = avatar.g_offsets.detach().to(torch::kFloat32);
            auto robust = torch::clamp_min(offsets_vis.abs().quantile(0.98f), 1e-6f);
            offsets_vis = ((offsets_vis / robust) * 0.5f) + 0.5f;
            offsets_vis = offsets_vis.clamp(0.0f, 1.0f);
            if (save_map("offsets.png", offsets_vis, true))
                ++saved_count;
        }
        if (save_map("scales_exp.png", torch::exp(avatar.g_scales), false))
            ++saved_count;
        if (save_map("opacities.png", avatar.g_opacities, true))
            ++saved_count;
        if (save_map("colors.png", avatar.g_colors, true))
            ++saved_count;

        if (use_sh && avatar.g_sh.defined() && avatar.g_sh.dim() == 3 && avatar.g_sh.numel() > 0)
        {
            if (avatar.g_sh.size(0) >= 3)
            {
                if (save_map("sh_dc.png", avatar.g_sh.index({Slice(0, 3), Slice(), Slice()}), false))
                    ++saved_count;
            }
            if (avatar.g_sh.size(0) > 3)
            {
                auto sh_rest = avatar.g_sh.index({Slice(3, torch::indexing::None), Slice(), Slice()});
                auto sh_rest_energy = torch::sqrt(sh_rest.pow(2).mean(0, true) + 1e-8f);
                if (save_map("sh_rest_energy.png", sh_rest_energy, false))
                    ++saved_count;
            }
        }

        return saved_count;
    };

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
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[5].options()).lr(betas_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[6].options()).lr(expr_bias_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[7].options()).lr(pose_lr * pose_lr_multiplier);
        }
        else if (epoch == warmup_epochs)
        {
            std::cout << "Warmup complete. Unfreezing Gaussian geometry..." << std::endl;
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options()).lr(rot_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[1].options()).lr(offset_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[2].options()).lr(scale_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[3].options()).lr(color_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[4].options()).lr(opacity_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[5].options()).lr(betas_lr * lr_multiplier);
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[6].options()).lr(expr_bias_lr * lr_multiplier);
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
            const auto betas_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[5].options()).lr();
            const auto expr_bias_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[6].options()).lr();
            const auto pose_lr_eff = static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[7].options()).lr();
            std::cout << "Epoch " << epoch << " LRs: "
                      << "rot=" << rot_lr_eff
                      << " offset=" << offset_lr_eff
                      << " scale=" << scale_lr_eff
                      << " color=" << color_lr_eff
                      << " opacity=" << opacity_lr_eff
                      << " betas=" << betas_lr_eff
                      << " expr_bias=" << expr_bias_lr_eff
                      << " pose=" << pose_lr_eff
                      << std::endl;
        }
        std::vector<size_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        std::vector<int64_t> ordered_indices(indices.begin(), indices.end());
        loader.Reset(std::move(ordered_indices));

        TrainingBatch batch;
        int batch_step = 0;
        while (loader.Next(&batch))
        {
            const bool should_log_metrics = (batch_step % metric_log_every) == 0;
            const bool should_log_verbose = verbose_diagnostics && ((batch_step % verbose_log_every) == 0);
            auto step_result = trainer.TrainStep(batch, epoch, batch_step, sh_degree_eff, should_log_verbose);
            if (!step_result.stepped)
            {
                batch_step++;
                continue;
            }

            if (should_log_metrics)
            {
                const float current_loss_val = step_result.loss.item<float>();
                const float valid_count = step_result.valid_sum.item<float>();

                std::ostringstream line;
                line.setf(std::ios::fixed);
                line << std::setprecision(6)
                     << "Epoch " << epoch
                     << " Batch " << batch_step
                     << " Loss: " << current_loss_val
                     << " Valid: " << valid_count << "/" << step_result.total_samples;
                metric_logger.Log(line.str());

                if (step_result.rot_delta_mean.defined() && step_result.trans_delta_max.defined())
                {
                    const float rot_val = step_result.rot_delta_mean.item<float>();
                    const float trans_val = step_result.trans_delta_max.item<float>();
                    std::ostringstream mlp_line;
                    mlp_line.setf(std::ios::fixed);
                    mlp_line << std::setprecision(6)
                             << "[MLP Stats] Rot Delta: " << rot_val
                             << " | Max Shift: " << trans_val;
                    metric_logger.Log(mlp_line.str());
                }
            }

            if (should_log_verbose && step_result.verbose_captured)
            {
                auto tensor_to_float = [](const torch::Tensor &value) -> float
                {
                    if (!value.defined())
                    {
                        return std::numeric_limits<float>::quiet_NaN();
                    }
                    return value.item<float>();
                };

                const float recon_val = tensor_to_float(step_result.final_recon);
                const float ssim_val = tensor_to_float(step_result.final_ssim);
                const float sobel_val = tensor_to_float(step_result.final_sobel);
                const float outside_val = tensor_to_float(step_result.final_outside);
                const float alpha_val = tensor_to_float(step_result.final_alpha);
                const float offset_reg_val = tensor_to_float(step_result.offset_reg);
                const float scale_reg_val = tensor_to_float(step_result.scale_reg);
                const float sh_reg_val = tensor_to_float(step_result.sh_reg_loss);
                const float sugar_reg_val = tensor_to_float(step_result.sugar_reg);
                const float knn_reg_val = tensor_to_float(step_result.knn_reg);
                const float opacity_reg_val = tensor_to_float(step_result.opacity_reg);
                const float anisotropic_reg_val = tensor_to_float(step_result.anisotropic_reg);

                const float grad_rot = tensor_to_float(step_result.grad_rot_norm);
                const float grad_offset = tensor_to_float(step_result.grad_offset_norm);
                const float grad_scale = tensor_to_float(step_result.grad_scale_norm);
                const float grad_color = tensor_to_float(step_result.grad_color_norm);
                const float grad_opacity = tensor_to_float(step_result.grad_opacity_norm);
                const float grad_pose = tensor_to_float(step_result.grad_pose_norm);
                const float std_rot = tensor_to_float(step_result.std_rot);
                const float std_offset = tensor_to_float(step_result.std_offset);
                const float std_scale = tensor_to_float(step_result.std_scale);
                const float std_color = tensor_to_float(step_result.std_color);
                const float std_opacity = tensor_to_float(step_result.std_opacity);

                const float scale_mean_val = tensor_to_float(step_result.scale_mean);
                const float scale_min_val = tensor_to_float(step_result.scale_min);
                const float scale_max_val = tensor_to_float(step_result.scale_max);
                const float scale_cap_hit_ratio = tensor_to_float(step_result.scale_cap_hit_ratio);
                const float opacity_low_ratio = tensor_to_float(step_result.opacity_low_ratio);
                const float opacity_high_ratio = tensor_to_float(step_result.opacity_high_ratio);

                const float rot_delta_val = tensor_to_float(step_result.rot_delta_mean);
                const float trans_delta_val = tensor_to_float(step_result.trans_delta_max);
                const float pose_offset_grad_ratio = (std::isfinite(grad_offset) && std::abs(grad_offset) > 1e-12f)
                                                         ? (grad_pose / grad_offset)
                                                         : 0.0f;

                std::ostringstream loss_line;
                loss_line.setf(std::ios::fixed);
                loss_line << std::setprecision(6)
                          << "[Verbose/Loss] recon=" << recon_val
                          << " ssim=" << ssim_val
                          << " sobel=" << sobel_val
                          << " outside=" << outside_val
                          << " alpha=" << alpha_val
                          << " offset_reg=" << offset_reg_val
                          << " scale_reg=" << scale_reg_val
                          << " sh_reg=" << sh_reg_val
                          << " sugar_reg=" << sugar_reg_val
                          << " knn_reg=" << knn_reg_val
                          << " opacity_reg=" << opacity_reg_val
                          << " anisotropic_reg=" << anisotropic_reg_val;
                metric_logger.Log(loss_line.str());

                std::ostringstream grad_line;
                grad_line.setf(std::ios::fixed);
                grad_line << std::setprecision(6)
                          << "[Verbose/Grad] rot=" << grad_rot
                          << " offset=" << grad_offset
                          << " scale=" << grad_scale
                          << " color=" << grad_color
                          << " opacity=" << grad_opacity
                          << " pose_mlp=" << grad_pose;
                metric_logger.Log(grad_line.str());

                std::ostringstream std_line;
                std_line.setf(std::ios::fixed);
                std_line << std::setprecision(6)
                         << "[Verbose/Std] rot=" << std_rot
                         << " offset=" << std_offset
                         << " scale=" << std_scale
                         << " color=" << std_color
                         << " opacity=" << std_opacity;
                metric_logger.Log(std_line.str());

                std::ostringstream clamp_line;
                clamp_line.setf(std::ios::fixed);
                clamp_line << std::setprecision(6)
                           << "[Verbose/Clamp] scale_mean=" << scale_mean_val
                           << " scale_min=" << scale_min_val
                           << " scale_max=" << scale_max_val
                           << " scale_cap_hit_ratio=" << scale_cap_hit_ratio
                           << " opacity_low_ratio=" << opacity_low_ratio
                           << " opacity_high_ratio=" << opacity_high_ratio;
                metric_logger.Log(clamp_line.str());

                std::ostringstream conflict_line;
                conflict_line.setf(std::ios::fixed);
                conflict_line << std::setprecision(6)
                              << "[Verbose/Conflict] rot_delta_mean=" << rot_delta_val
                              << " trans_delta_max=" << trans_delta_val
                              << " pose_to_offset_grad_ratio=" << pose_offset_grad_ratio;
                metric_logger.Log(conflict_line.str());

            }

            batch_step++;
        }

        // --- MODIFIED SECTION START ---
        {
            torch::NoGradGuard no_grad;
            torch::Tensor positions, rotations, colors, opacities, scales;
            torch::Tensor sh_to_send_shared = torch::zeros({0}, torch::TensorOptions().device(device));
            torch::Tensor sh_to_save = torch::zeros({0}, torch::TensorOptions().device(device));

            // 1. Always calculate canonical geometry for saving/viewing
            std::tie(positions, rotations) = avatar.forward(canonical_betas,
                                                            canonical_expression,
                                                            canonical_pose,
                                                            canonical_jaw_pose,
                                                            canonical_eye_pose,
                                                            canonical_left_hand_pose,
                                                            canonical_right_hand_pose,
                                                            canonical_trans);

            if (use_sh)
            {
                using torch::indexing::Slice;
                colors = avatar.current_flat_sh.index({Slice(), 0, Slice()});
            }
            else
            {
                colors = avatar.current_flat_colors;
            }
            opacities = avatar.current_flat_opacities;
            scales = capped_scales(avatar.current_flat_scales);

            if (use_sh)
            {
                // Shared-memory viewer uses fixed startup stride (configured sh_degree).
                // Keep SH payload shape stable across warmup epochs.
                if (sh_degree > 0)
                {
                    sh_to_send_shared = RotateSH(avatar.current_flat_sh, rotations);
                }
                // Disk export uses effective degree per epoch.
                if (sh_degree_eff > 0)
                {
                    sh_to_save = RotateSH(avatar.current_flat_sh, rotations);
                }
            }

            // 2. Handle Shared Memory Viewer (Only if enabled)
            if (publish_viewer)
            {
                const int64_t bind_count = avatar.current_flat_offsets.size(0);
                if (bind_count > 0)
                {
                    if (!bind_capacity_warned &&
                        bind_count > static_cast<int64_t>(viewer_capacity))
                    {
                        std::cerr << "Warning: bind shared memory capacity exceeded ("
                                  << bind_count << " > " << viewer_capacity
                                  << "). Viewer bind data will be truncated." << std::endl;
                        bind_capacity_warned = true;
                    }
                    auto bary_cpu = avatar.g_bary_coords.to(torch::kCPU).contiguous();
                    auto offsets_cpu = avatar.current_flat_offsets.to(torch::kCPU).contiguous();
                    auto rots_cpu = avatar.current_flat_rots.to(torch::kCPU).contiguous();
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
                if (BuildSharedGaussianBuffer(positions, colors, opacities, scales, rotations, sh_to_send_shared, sh_degree,
                                              &shared_buffer))
                {
                    const int64_t point_count = positions.size(0);
                    if (!shared_capacity_warned &&
                        point_count > static_cast<int64_t>(viewer_capacity))
                    {
                        std::cerr << "Warning: shared memory capacity exceeded ("
                                  << point_count << " > " << viewer_capacity
                                  << "). Viewer output will be truncated." << std::endl;
                        shared_capacity_warned = true;
                    }
                    shared_writer.Write(shared_buffer.data(), static_cast<uint32_t>(positions.size(0)),
                                        shared_frame++);
                }
            }

            // 3. Save to Disk (Every epoch, Overwrite)
            {
                std::filesystem::path save_dir = viewer_export_dir.empty() ? out_dir_path : viewer_out_path;
                torch::Tensor bone_indices, bone_weights;
                std::tie(bone_indices, bone_weights) = avatar.get_bone_data();
                
                SaveViewerDataOverwrite(save_dir, positions, colors, opacities, scales, rotations,
                                        sh_to_save, sh_degree_eff, bone_indices, bone_weights);
            }

            // --- MODIFIED SECTION END ---

            auto render_view = [&](size_t sample_index,
                                   const TrainSample &sample,
                                   const CachedSampleData &cached_entry) -> RenderViewResult
            {
                RenderViewResult render_result;
                torch::NoGradGuard no_grad;
                if (!cached_entry.valid)
                {
                    return render_result;
                }
                if (cached_entry.crop_bgr.empty())
                {
                    return render_result;
                }
                const int H = cached_entry.crop_bgr.rows;
                const int W = cached_entry.crop_bgr.cols;
                if (H <= 0 || W <= 0)
                {
                    return render_result;
                }

                SmplResult res;
                res.pose = sample.pose;
                res.shape = sample.betas;
                res.camera = sample.cam;

                auto pose_base = PoseToAxisAngle(res).to(device);
                cv::Vec3f trans_cv;
                if (sample.has_translation)
                {
                    trans_cv = cv::Vec3f(sample.translation[0], sample.translation[1], sample.translation[2]);
                }
                else
                {
                    trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                   sample.crop_size, sample.focal_length,
                                                   static_cast<float>(sample.img_w),
                                                   static_cast<float>(sample.img_h));
                }
                auto trans_base = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                                torch::TensorOptions().device(device).dtype(torch::kFloat))
                                      .unsqueeze(0);

                auto pose_flat = pose_base.view({1, 72});
                auto net_input = torch::cat({pose_flat, trans_base}, 1);
                const float img_w = std::max(1.0f, static_cast<float>(sample.img_w));
                const float img_h = std::max(1.0f, static_cast<float>(sample.img_h));
                auto crop_params = torch::tensor(
                                       {sample.crop_cx / img_w, sample.crop_cy / img_h, sample.crop_size / img_w},
                                       torch::TensorOptions().device(device).dtype(torch::kFloat))
                                       .unsqueeze(0);
                const float time_val = static_cast<float>(sample_index) /
                                       static_cast<float>(std::max<size_t>(1, samples.size() - 1));
                auto time_tensor = torch::tensor({time_val}, torch::TensorOptions().device(device).dtype(torch::kFloat)).unsqueeze(0);

                auto deltas = pose_refiner.forward(net_input, crop_params, time_tensor);
                auto pose_delta = deltas.slice(1, 0, 72).view({1, 24, 3});
                auto trans_delta = deltas.slice(1, 72, 75).view({1, 3});

                auto pose = pose_base + pose_delta;
                auto trans = (trans_base + trans_delta).squeeze(0);
                auto effective_expression =
                    TensorFromPaddedVector(sample.smplx_expression, kSmplxExpressionParamCount, device) +
                    canonical_expression;
                auto effective_jaw_pose =
                    TensorFromPaddedVector(sample.smplx_jaw_pose, kSmplxJawPoseParamCount, device);
                auto effective_eye_pose =
                    TensorFromPaddedVector(sample.smplx_eye_pose, kSmplxEyePoseParamCount, device);
                auto effective_left_hand_pose =
                    TensorFromPaddedVector(sample.smplx_left_hand_pose, kSmplxHandPoseParamCount, device);
                auto effective_right_hand_pose =
                    TensorFromPaddedVector(sample.smplx_right_hand_pose, kSmplxHandPoseParamCount, device);

                PoseSampleExport pose_export;
                auto pose_base_cpu = pose_base.detach().to(torch::kCPU).contiguous().view({24, 3});
                auto pose_delta_cpu = pose_delta.detach().to(torch::kCPU).contiguous().view({24, 3});
                auto pose_refined_cpu = pose.detach().to(torch::kCPU).contiguous().view({24, 3});
                auto trans_base_cpu = trans_base.detach().to(torch::kCPU).contiguous().view({3});
                auto trans_delta_cpu = trans_delta.detach().to(torch::kCPU).contiguous().view({3});
                auto trans_cpu = trans.detach().to(torch::kCPU).contiguous().view({3});

                const float *pose_base_ptr = pose_base_cpu.data_ptr<float>();
                const float *pose_delta_ptr = pose_delta_cpu.data_ptr<float>();
                const float *pose_refined_ptr = pose_refined_cpu.data_ptr<float>();
                const float *trans_base_ptr = trans_base_cpu.data_ptr<float>();
                const float *trans_delta_ptr = trans_delta_cpu.data_ptr<float>();
                const float *trans_ptr = trans_cpu.data_ptr<float>();

                pose_export.original_transl = {trans_base_ptr[0], trans_base_ptr[1], trans_base_ptr[2]};
                pose_export.delta_transl = {trans_delta_ptr[0], trans_delta_ptr[1], trans_delta_ptr[2]};
                pose_export.refined_transl = {trans_ptr[0], trans_ptr[1], trans_ptr[2]};

                for (int joint_idx = 0; joint_idx < 24; ++joint_idx)
                {
                    const size_t offset = static_cast<size_t>(joint_idx) * 3u;
                    pose_export.original_pose[joint_idx].rot = {
                        pose_base_ptr[offset + 0],
                        pose_base_ptr[offset + 1],
                        pose_base_ptr[offset + 2]};
                    pose_export.pose_delta[joint_idx].rot = {
                        pose_delta_ptr[offset + 0],
                        pose_delta_ptr[offset + 1],
                        pose_delta_ptr[offset + 2]};
                    pose_export.refined_pose[joint_idx].rot = {
                        pose_refined_ptr[offset + 0],
                        pose_refined_ptr[offset + 1],
                        pose_refined_ptr[offset + 2]};

                    pose_export.original_pose[joint_idx].transl = pose_export.original_transl;
                    pose_export.pose_delta[joint_idx].transl = pose_export.delta_transl;
                    pose_export.refined_pose[joint_idx].transl = pose_export.refined_transl;
                }
                pose_export.valid = true;
                render_result.pose_export = pose_export;

                SmplxParamsExport smplx_export;
                auto betas_cpu = canonical_betas.detach().to(torch::kCPU).contiguous().view({-1});
                auto expression_cpu = effective_expression.detach().to(torch::kCPU).contiguous().view({-1});
                auto jaw_pose_cpu = effective_jaw_pose.detach().to(torch::kCPU).contiguous().view({-1});
                auto eye_pose_cpu = effective_eye_pose.detach().to(torch::kCPU).contiguous().view({-1});
                auto left_hand_pose_cpu = effective_left_hand_pose.detach().to(torch::kCPU).contiguous().view({-1});
                auto right_hand_pose_cpu = effective_right_hand_pose.detach().to(torch::kCPU).contiguous().view({-1});

                const float *betas_ptr = betas_cpu.data_ptr<float>();
                const float *expression_ptr = expression_cpu.data_ptr<float>();
                const float *jaw_pose_ptr = jaw_pose_cpu.data_ptr<float>();
                const float *eye_pose_ptr = eye_pose_cpu.data_ptr<float>();
                const float *left_hand_pose_ptr = left_hand_pose_cpu.data_ptr<float>();
                const float *right_hand_pose_ptr = right_hand_pose_cpu.data_ptr<float>();

                smplx_export.body_model = "smplx";
                smplx_export.y_sign = sample.y_sign;
                smplx_export.transl = {trans_ptr[0], trans_ptr[1], trans_ptr[2]};
                smplx_export.betas.assign(betas_ptr, betas_ptr + static_cast<size_t>(betas_cpu.numel()));
                smplx_export.pose_axis_angle.assign(
                    pose_refined_ptr, pose_refined_ptr + static_cast<size_t>(pose_refined_cpu.numel()));
                smplx_export.expression.assign(
                    expression_ptr, expression_ptr + static_cast<size_t>(expression_cpu.numel()));
                smplx_export.jaw_pose.assign(
                    jaw_pose_ptr, jaw_pose_ptr + static_cast<size_t>(jaw_pose_cpu.numel()));
                smplx_export.eye_pose.assign(
                    eye_pose_ptr, eye_pose_ptr + static_cast<size_t>(eye_pose_cpu.numel()));
                smplx_export.left_hand_pose.assign(
                    left_hand_pose_ptr,
                    left_hand_pose_ptr + static_cast<size_t>(left_hand_pose_cpu.numel()));
                smplx_export.right_hand_pose.assign(
                    right_hand_pose_ptr,
                    right_hand_pose_ptr + static_cast<size_t>(right_hand_pose_cpu.numel()));
                smplx_export.valid = true;
                render_result.smplx_export = std::move(smplx_export);

                if (viewer_stream_poses)
                {
                    auto pose_stream_cpu = pose_refined_cpu.contiguous();
                    const float *pose_ptr = pose_stream_cpu.data_ptr<float>();
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

                    const float tx = trans_ptr[0];
                    const float ty = trans_ptr[1];
                    const float tz = trans_ptr[2];
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
                    avatar.forward(canonical_betas,
                                   effective_expression,
                                   pose,
                                   effective_jaw_pose,
                                   effective_eye_pose,
                                   effective_left_hand_pose,
                                   effective_right_hand_pose,
                                   torch::zeros({1, 3}, canonical_betas.options()));
                torch::Tensor current_sh = use_sh ? avatar.current_flat_sh : torch::zeros({0}, avatar.g_colors.options());
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

                auto colors = use_sh ? torch::zeros({0}, avatar.g_colors.options()) : avatar.current_flat_colors;
                auto outputs = GaussianRasterizer::apply(
                    means3D,
                    colors,
                    avatar.current_flat_opacities,
                    capped_scales(avatar.current_flat_scales),
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
                    return render_result;
                }

                render_result.image = image.detach();
                return render_result;
            };

            if ((epoch + 1) % 10 == 0 || epoch == (epochs - 1) || epoch == 0)
            {
                const int saved_pairs = SaveEpochViewPairs(samples, cached, out_dir_path, epoch, render_view);
                std::cout << "Epoch " << epoch << " saved " << saved_pairs << " view pairs." << std::endl;
                const int saved_uv_maps = save_uv_debug_maps(epoch);
                std::cout << "Epoch " << epoch << " saved " << saved_uv_maps << " UV debug maps." << std::endl;
            }
        }
    }

    {
        torch::NoGradGuard no_grad;
        torch::Tensor positions, rotations, scales, colors, opacities;
        std::tie(positions, rotations) = avatar.forward(canonical_betas,
                                                        canonical_expression,
                                                        canonical_pose,
                                                        canonical_jaw_pose,
                                                        canonical_eye_pose,
                                                        canonical_left_hand_pose,
                                                        canonical_right_hand_pose,
                                                        canonical_trans);
        scales = capped_scales(avatar.current_flat_scales);
        opacities = avatar.current_flat_opacities;
        if (use_sh)
        {
            using torch::indexing::Slice;
            colors = avatar.current_flat_sh.index({Slice(), 0, Slice()});
        }
        else
        {
            colors = avatar.current_flat_colors;
        }

        
    }
    return 0;
}

int main(int argc, char *argv[])
{
    // std::cout << "Starting Full Program Profiling..." << std::endl;

    // // 1. Configure the profiler to use the Kineto engine
    // torch::autograd::profiler::ProfilerConfig cfg(
    //     torch::autograd::profiler::ProfilerState::KINETO,
    //     false, // report_input_shapes
    //     false, // profile_memory
    //     false, // with_stack
    //     false, // with_flops
    //     false  // with_modules
    // );

    // // 2. Explicitly request both CPU and CUDA activities
    // std::set<torch::autograd::profiler::ActivityType> activities = {
    //     torch::autograd::profiler::ActivityType::CPU,
    //     torch::autograd::profiler::ActivityType::CUDA};

    // // 3. Start tracing
    // torch::autograd::profiler::prepareProfiler(cfg, activities);
    // torch::autograd::profiler::enableProfiler(cfg, activities);

    // try
    // {
    //     run_real_training(argc, argv);
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "Exception during profiling: " << e.what() << std::endl;
    // }

    // // 4. Stop tracing and save to disk
    // auto profiler_result = torch::autograd::profiler::disableProfiler();
    // profiler_result->save("full_profile.json");

    // std::cout << "Profiling complete. Saved to full_profile.json" << std::endl;

    run_real_training(argc, argv);

    return 0;
}
