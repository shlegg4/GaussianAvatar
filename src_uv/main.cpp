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
#include "utils/train/AsyncMetricLogger.h"
#include "utils/train/GaussianAvatar.h"
#include "utils/train/GaussianDataLoader.h"
#include "utils/train/GaussianTrainer.h"
#include "utils/train/PoseRefiner.h"
#include "utils/train/TrainDataGPU.h"
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
#pragma optimize("", off)
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
        const std::string full_frame_path = DeriveFullFramePath(sample.crop_path);
        if (!full_frame_path.empty())
        {
            cv::Mat full_frame = cv::imread(full_frame_path);
            if (!full_frame.empty())
            {
                entry.target_bgr = std::move(full_frame);
            }
        }
        if (entry.target_bgr.empty())
        {
            entry.target_bgr = crop;
        }
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
                if (sample.img_h <= 0 || sample.img_w <= 0)
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

                const int render_w = std::max(1, sample.img_w);
                const int render_h = std::max(1, sample.img_h);
                const float default_cx = static_cast<float>(render_w) * 0.5f;
                const float default_cy = static_cast<float>(render_h) * 0.5f;
                const float full_fx = (sample.full_fx > 0.0f) ? sample.full_fx : sample.focal_length;
                const float full_fy = (sample.full_fy > 0.0f) ? sample.full_fy : sample.focal_length;
                const float full_cx = (sample.full_cx > 0.0f) ? sample.full_cx : default_cx;
                const float full_cy = (sample.full_cy > 0.0f) ? sample.full_cy : default_cy;
                const CameraProjectionInput projection_input{
                    sample.focal_length,
                    full_fx,
                    full_fy,
                    sample.img_w,
                    sample.img_h,
                    render_w,
                    render_h,
                    full_cx,
                    full_cy,
                    0.0f,
                    0.0f,
                    static_cast<float>(render_w),
                    static_cast<float>(render_h)};
                const CameraProjectionOutput projection =
                    BuildCameraProjection(projection_input, device); 

                // Temporary debug path: identity view with means3D + trans only.
                torch::Tensor view_mat = projection.view_mat;
                torch::Tensor cam_pos_render = cam_pos;

                auto colors = use_sh ? torch::zeros({0}, avatar.g_colors.options()) : avatar.current_flat_colors;
                
                auto outputs = GaussianRasterizer::apply(
                    means3D,
                    colors,
                    avatar.current_flat_opacities,
                    capped_scales(avatar.current_flat_scales),
                    current_rots,
                    render_scale_modifier,
                    view_mat,
                    projection.proj_mat,
                    projection.tan_fovx,
                    projection.tan_fovy,
                    render_h,
                    render_w,
                    current_sh,
                    use_sh ? sh_degree : 0,
                    cam_pos_render,
                    false);
                auto image = outputs[0];
                if (!image.defined() || image.dim() != 3 || image.size(0) != 3 ||
                    image.size(1) != render_h || image.size(2) != render_w)
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
#pragma optimize("", on)
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
