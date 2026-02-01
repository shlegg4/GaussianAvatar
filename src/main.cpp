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

struct TrainSample
{
    int frame = -1;
    std::string crop_path;
    int img_w = 0;
    int img_h = 0;
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    float focal_length = 0.0f;
    float y_sign = 1.0f;
    std::vector<float> pose;
    std::vector<float> betas;
    std::vector<float> cam;
};

struct TrainOptions
{
    std::string jsonl_path;
    std::string smpl_model_path = "smpl_data.pt";
    int num_gaussians = 5000;
    int epochs = 1;
    float lr = 0.01f;
    std::string output_dir = "outputs";
    float scale_reg_weight = 0.001f;
    float scale_max_reg_weight = 0.1f;
    float scale_max_value = 0.02f;
    float offset_reg_weight = 0.01f;
    float mesh_reg_weight = 0.1f;
    float mesh_reg_max_dist = 0.02f;
    float color_lr = -1.0f;
    int sh_degree = 3;
    bool enable_viewer = false;
    int viewer_width = 800;
    int viewer_height = 600;
    int viewer_every = 1;
    std::string viewer_shm_name = "GaussianAvatarShared";
};

cv::Mat TensorToBgr(const torch::Tensor &image);
bool IsMaskCoverageValidTensor(const torch::Tensor &render, const torch::Tensor &matte_mask,
                               float max_outside_ratio, float render_threshold);

bool ExtractStringField(const std::string &line, const std::string &key, std::string *out)
{
    const std::string tag = "\"" + key + "\":\"";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    size_t end = line.find('"', start);
    if (end == std::string::npos)
        return false;
    *out = line.substr(start, end - start);
    return true;
}

bool ExtractNumberField(const std::string &line, const std::string &key, double *out)
{
    const std::string tag = "\"" + key + "\":";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    const char *ptr = line.c_str() + start;
    char *end = nullptr;
    double val = std::strtod(ptr, &end);
    if (end == ptr)
        return false;
    *out = val;
    return true;
}

bool ExtractArrayField(const std::string &line, const std::string &key, std::vector<float> *out)
{
    const std::string tag = "\"" + key + "\":[";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    size_t end = line.find(']', start);
    if (end == std::string::npos)
        return false;
    std::string content = line.substr(start, end - start);
    out->clear();
    if (content.empty())
        return true;
    size_t pos = 0;
    while (pos < content.size())
    {
        size_t comma = content.find(',', pos);
        std::string token = (comma == std::string::npos) ? content.substr(pos) : content.substr(pos, comma - pos);
        if (!token.empty())
        {
            char *end_ptr = nullptr;
            float val = std::strtof(token.c_str(), &end_ptr);
            if (end_ptr != token.c_str())
            {
                out->push_back(val);
            }
        }
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return true;
}

bool ParseTrainSample(const std::string &line, TrainSample *out)
{
    TrainSample sample;
    double value = 0.0;
    if (ExtractNumberField(line, "frame", &value))
    {
        sample.frame = static_cast<int>(value);
    }
    if (!ExtractStringField(line, "crop", &sample.crop_path))
        return false;
    if (sample.crop_path.empty())
        return false;

    if (!ExtractNumberField(line, "img_w", &value))
        return false;
    sample.img_w = static_cast<int>(value);
    if (!ExtractNumberField(line, "img_h", &value))
        return false;
    sample.img_h = static_cast<int>(value);
    if (!ExtractNumberField(line, "crop_cx", &value))
        return false;
    sample.crop_cx = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_cy", &value))
        return false;
    sample.crop_cy = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_size", &value))
        return false;
    sample.crop_size = static_cast<float>(value);
    if (!ExtractNumberField(line, "focal_length", &value))
        return false;
    sample.focal_length = static_cast<float>(value);
    if (!ExtractNumberField(line, "y_sign", &value))
        return false;
    sample.y_sign = static_cast<float>(value);

    if (!ExtractArrayField(line, "pose", &sample.pose))
        return false;
    if (!ExtractArrayField(line, "betas", &sample.betas))
        return false;
    if (!ExtractArrayField(line, "cam", &sample.cam))
        return false;

    if (sample.pose.empty() || sample.betas.empty() || sample.cam.empty())
        return false;
    *out = std::move(sample);
    return true;
}

bool ReplaceFirst(std::string *value, const std::string &from, const std::string &to)
{
    size_t pos = value->find(from);
    if (pos == std::string::npos)
        return false;
    value->replace(pos, from.size(), to);
    return true;
}

std::string DeriveMattePath(const std::string &crop_path)
{
    std::string matte_path = crop_path;
    if (!ReplaceFirst(&matte_path, "\\crops\\", "\\mattes\\"))
    {
        ReplaceFirst(&matte_path, "/crops/", "/mattes/");
    }
    if (!ReplaceFirst(&matte_path, "crop_", "matte_"))
    {
        ReplaceFirst(&matte_path, "crop-", "matte-");
    }
    return matte_path;
}

float ComputeMedian(std::vector<float> values)
{
    if (values.empty())
        return 0.0f;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 1)
        return values[mid];
    return 0.5f * (values[mid - 1] + values[mid]);
}

float ComputeMad(const std::vector<float> &values, float median)
{
    std::vector<float> deviations;
    deviations.reserve(values.size());
    for (float v : values)
    {
        deviations.push_back(std::abs(v - median));
    }
    return ComputeMedian(std::move(deviations));
}

std::vector<float> ComputeAverageBetas(const std::vector<TrainSample> &samples)
{
    std::vector<float> sum;
    size_t count = 0;
    for (const auto &sample : samples)
    {
        if (sample.betas.empty())
            continue;
        if (sum.empty())
        {
            sum.assign(sample.betas.size(), 0.0f);
        }
        if (sample.betas.size() != sum.size())
            continue;
        for (size_t i = 0; i < sum.size(); ++i)
        {
            sum[i] += sample.betas[i];
        }
        count++;
    }
    if (count == 0)
        return {};
    for (float &v : sum)
    {
        v /= static_cast<float>(count);
    }
    return sum;
}

bool IsMaskCoverageValid(const torch::Tensor &render, const cv::Mat &matte, float max_outside_ratio)
{
    if (!render.defined() || matte.empty())
        return false;
    const int H = static_cast<int>(render.size(1));
    const int W = static_cast<int>(render.size(2));

    cv::Mat matte_gray;
    if (matte.channels() == 4)
    {
        std::vector<cv::Mat> channels;
        cv::split(matte, channels);
        matte_gray = channels[3];
    }
    else if (matte.channels() == 3)
    {
        cv::cvtColor(matte, matte_gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        matte_gray = matte;
    }

    if (matte_gray.size() != cv::Size(W, H))
    {
        cv::resize(matte_gray, matte_gray, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat matte_mask;
    cv::threshold(matte_gray, matte_mask, 127, 255, cv::THRESH_BINARY);

    cv::Mat render_bgr = TensorToBgr(render);
    if (render_bgr.empty())
        return false;
    cv::Mat render_gray;
    cv::cvtColor(render_bgr, render_gray, cv::COLOR_BGR2GRAY);
    cv::Mat render_mask;
    cv::threshold(render_gray, render_mask, 3, 255, cv::THRESH_BINARY);

    const int rendered_pixels = cv::countNonZero(render_mask);
    if (rendered_pixels == 0)
        return false;

    cv::Mat outside_mask;
    cv::bitwise_not(matte_mask, outside_mask);
    cv::Mat outside_render;
    cv::bitwise_and(render_mask, outside_mask, outside_render);
    const int outside_pixels = cv::countNonZero(outside_render);
    const float outside_ratio = static_cast<float>(outside_pixels) / static_cast<float>(rendered_pixels);
    return outside_ratio <= max_outside_ratio;
}

bool IsMaskCoverageValidTensor(const torch::Tensor &render, const torch::Tensor &matte_mask,
                               float max_outside_ratio, float render_threshold)
{
    if (!render.defined() || !matte_mask.defined())
        return false;
    if (render.dim() != 3 || render.size(0) != 3)
        return false;
    if (matte_mask.dim() != 3 || matte_mask.size(0) != 1)
        return false;
    if (render.size(1) != matte_mask.size(1) || render.size(2) != matte_mask.size(2))
        return false;

    auto render_gray = render.mean(0, true);
    auto render_mask = render_gray > render_threshold;
    auto matte_bin = matte_mask > 0.5f;
    auto rendered_pixels = render_mask.sum().item<float>();
    if (rendered_pixels <= 0.0f)
        return false;
    auto outside = render_mask.logical_and(matte_bin.logical_not());
    auto outside_pixels = outside.sum().item<float>();
    const float outside_ratio = outside_pixels / rendered_pixels;
    return outside_ratio <= max_outside_ratio;
}

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
            else if (key == "--scale-max-reg")
                options->scale_max_reg_weight = std::stof(value);
            else if (key == "--scale-max")
                options->scale_max_value = std::stof(value);
            else if (key == "--offset-reg")
                options->offset_reg_weight = std::stof(value);
            else if (key == "--mesh-reg")
                options->mesh_reg_weight = std::stof(value);
            else if (key == "--mesh-max-dist")
                options->mesh_reg_max_dist = std::stof(value);
            else if (key == "--color-lr")
                options->color_lr = std::stof(value);
            else if (key == "--sh-degree")
                options->sh_degree = std::stoi(value);
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

torch::Tensor LoadImageTensor(const cv::Mat &bgr, torch::Device device)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(
                      float_img.data,
                      {float_img.rows, float_img.cols, 3},
                      torch::kFloat)
                      .clone();
    tensor = tensor.permute({2, 0, 1}).contiguous().to(device);
    return tensor;
}

torch::Tensor LoadMatteMaskTensor(const cv::Mat &matte, int target_w, int target_h, torch::Device device)
{
    if (matte.empty())
        return torch::Tensor();

    cv::Mat matte_gray;
    if (matte.channels() == 4)
    {
        std::vector<cv::Mat> channels;
        cv::split(matte, channels);
        matte_gray = channels[3];
    }
    else if (matte.channels() == 3)
    {
        cv::cvtColor(matte, matte_gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        matte_gray = matte;
    }

    if (matte_gray.size() != cv::Size(target_w, target_h))
    {
        cv::resize(matte_gray, matte_gray, cv::Size(target_w, target_h), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat float_mask;
    matte_gray.convertTo(float_mask, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(float_mask.data, {float_mask.rows, float_mask.cols, 1}, torch::kFloat).clone();
    tensor = tensor.permute({2, 0, 1}).contiguous().to(device);
    return tensor;
}

bool SaveImageTensorPng(const std::string &path, const torch::Tensor &image)
{
    if (!image.defined())
        return false;
    auto img = image.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kU8).cpu();
    if (img.dim() != 3 || img.size(0) != 3)
        return false;
    img = img.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return cv::imwrite(path, bgr);
}

cv::Mat TensorToBgr(const torch::Tensor &image)
{
    if (!image.defined())
        return cv::Mat();
    auto img = image.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kU8).cpu();
    if (img.dim() != 3 || img.size(0) != 3)
        return cv::Mat();
    img = img.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height, torch::Device device)
{
    const float n = 0.01f;
    const float f = 100.0f;
    const float tan_fovx = (static_cast<float>(width) * 0.5f) / std::max(focal, 1e-6f);
    const float tan_fovy = (static_cast<float>(height) * 0.5f) / std::max(focal, 1e-6f);

    auto view = torch::eye(4, device);
    auto proj = torch::zeros({4, 4}, device);
    proj[0][0] = 1.0f / tan_fovx;
    proj[1][1] = 1.0f / tan_fovy;
    proj[2][2] = f / (f - n);
    proj[2][3] = -(f * n) / (f - n);
    proj[3][2] = 1.0f;

    return {view.transpose(0, 1).contiguous(), proj.transpose(0, 1).contiguous(), tan_fovx, tan_fovy};
}

torch::Tensor ComputeTriFrames(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C)
{
    auto X = torch::nn::functional::normalize(B - A, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto N = torch::cross(B - A, C - A, 1);
    N = torch::nn::functional::normalize(N, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto Y = torch::cross(N, X, 1);
    return torch::stack({X, Y, N}, 2);
}

torch::Tensor MatrixToQuat(const torch::Tensor &rot_mat)
{
    using torch::indexing::Slice;
    auto m00 = rot_mat.index({Slice(), 0, 0});
    auto m11 = rot_mat.index({Slice(), 1, 1});
    auto m22 = rot_mat.index({Slice(), 2, 2});
    auto tr = m00 + m11 + m22;

    const auto num = rot_mat.size(0);
    auto q = torch::zeros({num, 4}, rot_mat.options());

    auto mask1 = tr > 0;
    if (mask1.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(tr.index({mask1}) + 1.0f, 0.0f)) * 2.0f;
        q.index_put_({mask1, 0}, 0.25f * S);
        q.index_put_({mask1, 1}, (rot_mat.index({mask1, 2, 1}) - rot_mat.index({mask1, 1, 2})) / S);
        q.index_put_({mask1, 2}, (rot_mat.index({mask1, 0, 2}) - rot_mat.index({mask1, 2, 0})) / S);
        q.index_put_({mask1, 3}, (rot_mat.index({mask1, 1, 0}) - rot_mat.index({mask1, 0, 1})) / S);
    }

    auto mask2 = (~mask1) & (m00 > m11) & (m00 > m22);
    if (mask2.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m00.index({mask2}) - m11.index({mask2}) - m22.index({mask2}),
                                              0.0f)) *
                 2.0f;
        q.index_put_({mask2, 0}, (rot_mat.index({mask2, 2, 1}) - rot_mat.index({mask2, 1, 2})) / S);
        q.index_put_({mask2, 1}, 0.25f * S);
        q.index_put_({mask2, 2}, (rot_mat.index({mask2, 0, 1}) + rot_mat.index({mask2, 1, 0})) / S);
        q.index_put_({mask2, 3}, (rot_mat.index({mask2, 0, 2}) + rot_mat.index({mask2, 2, 0})) / S);
    }

    auto mask3 = (~mask1) & (~mask2) & (m11 > m22);
    if (mask3.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m11.index({mask3}) - m00.index({mask3}) - m22.index({mask3}),
                                              0.0f)) *
                 2.0f;
        q.index_put_({mask3, 0}, (rot_mat.index({mask3, 0, 2}) - rot_mat.index({mask3, 2, 0})) / S);
        q.index_put_({mask3, 1}, (rot_mat.index({mask3, 0, 1}) + rot_mat.index({mask3, 1, 0})) / S);
        q.index_put_({mask3, 2}, 0.25f * S);
        q.index_put_({mask3, 3}, (rot_mat.index({mask3, 1, 2}) + rot_mat.index({mask3, 2, 1})) / S);
    }

    auto mask4 = (~mask1) & (~mask2) & (~mask3);
    if (mask4.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m22.index({mask4}) - m00.index({mask4}) - m11.index({mask4}),
                                              0.0f)) *
                 2.0f;
        q.index_put_({mask4, 0}, (rot_mat.index({mask4, 1, 0}) - rot_mat.index({mask4, 0, 1})) / S);
        q.index_put_({mask4, 1}, (rot_mat.index({mask4, 0, 2}) + rot_mat.index({mask4, 2, 0})) / S);
        q.index_put_({mask4, 2}, (rot_mat.index({mask4, 1, 2}) + rot_mat.index({mask4, 2, 1})) / S);
        q.index_put_({mask4, 3}, 0.25f * S);
    }

    return torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().dim(1));
}

torch::Tensor QuatMultiply(const torch::Tensor &p, const torch::Tensor &q)
{
    auto pw = p.select(1, 0);
    auto px = p.select(1, 1);
    auto py = p.select(1, 2);
    auto pz = p.select(1, 3);
    auto qw = q.select(1, 0);
    auto qx = q.select(1, 1);
    auto qy = q.select(1, 2);
    auto qz = q.select(1, 3);

    auto w = pw * qw - px * qx - py * qy - pz * qz;
    auto x = pw * qx + px * qw + py * qz - pz * qy;
    auto y = pw * qy - px * qz + py * qw + pz * qx;
    auto z = pw * qz + px * qy - py * qx + pz * qw;

    return torch::stack({w, x, y, z}, 1);
}

torch::Tensor QuatToMat3(const torch::Tensor &q, torch::Device device)
{
    auto qn = torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto w = qn.select(1, 0);
    auto x = qn.select(1, 1);
    auto y = qn.select(1, 2);
    auto z = qn.select(1, 3);

    auto ww = w * w;
    auto xx = x * x;
    auto yy = y * y;
    auto zz = z * z;
    auto wx = w * x;
    auto wy = w * y;
    auto wz = w * z;
    auto xy = x * y;
    auto xz = x * z;
    auto yz = y * z;

    auto m00 = ww + xx - yy - zz;
    auto m01 = 2.0f * (xy - wz);
    auto m02 = 2.0f * (xz + wy);
    auto m10 = 2.0f * (xy + wz);
    auto m11 = ww - xx + yy - zz;
    auto m12 = 2.0f * (yz - wx);
    auto m20 = 2.0f * (xz - wy);
    auto m21 = 2.0f * (yz + wx);
    auto m22 = ww - xx - yy + zz;

    auto row0 = torch::stack({m00, m01, m02}, 1);
    auto row1 = torch::stack({m10, m11, m12}, 1);
    auto row2 = torch::stack({m20, m21, m22}, 1);
    return torch::stack({row0, row1, row2}, 1).to(device);
}

torch::Tensor RotateSH(const torch::Tensor &sh, const torch::Tensor &rotations)
{
    if (!sh.defined() || sh.dim() < 3 || sh.size(1) < 4)
        return sh;

    // 1. CRITICAL FIX: Normalize rotations to ensure valid rotation matrix
    auto rots_norm = torch::nn::functional::normalize(rotations, 
        torch::nn::functional::NormalizeFuncOptions().dim(1));

    // 2. Generate Rotation Matrix from unit quaternions
    auto rot_mats = QuatToMat3(rots_norm, sh.device());

    using torch::indexing::Slice;
    
    // 3. Extract Y, Z, X bands (Standard 3DGS order: 1=Y, 2=Z, 3=X)
    auto sh_d1 = sh.index({Slice(), Slice(1, 4), Slice()});
    auto sh_y = sh_d1.index({Slice(), 0, Slice()});
    auto sh_z = sh_d1.index({Slice(), 1, Slice()});
    auto sh_x = sh_d1.index({Slice(), 2, Slice()});

    // 4. Construct vector in (X, Y, Z) order to match Rotation Matrix
    auto sh_vec = torch::stack({sh_x, sh_y, sh_z}, 1);

    // 5. Apply Rotation (R * [x, y, z]^T)
    auto rotated_vec = torch::bmm(rot_mats, sh_vec);

    auto out_sh = sh.clone();

    // 6. Unpack back to SH order: 1->Y, 2->Z, 3->X
    // rotated_vec indices: 0->X', 1->Y', 2->Z'
    out_sh.index_put_({Slice(), 1, Slice()}, rotated_vec.index({Slice(), 1, Slice()})); // Y gets Y'
    out_sh.index_put_({Slice(), 2, Slice()}, rotated_vec.index({Slice(), 2, Slice()})); // Z gets Z' 
    out_sh.index_put_({Slice(), 3, Slice()}, -rotated_vec.index({Slice(), 0, Slice()})); // -X -> X (Inverted)
    // Zero out higher degrees (if any) as they aren't rotated here
    if (sh.size(1) > 4)
    {
        out_sh.index_put_({Slice(), Slice(4, torch::indexing::None), Slice()}, 0.0f);
    }

    return out_sh;
}

torch::Tensor CropRenderToTarget(const torch::Tensor &full_render, int crop_w, int crop_h,
                                 float crop_cx, float crop_cy)
{
    if (!full_render.defined() || full_render.dim() != 3)
    {
        return torch::Tensor();
    }
    auto output = torch::zeros({3, crop_h, crop_w}, full_render.options());

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

struct CachedSampleData
{
    torch::Tensor target;
    torch::Tensor matte_mask;
    cv::Mat crop_bgr;
    bool valid = false;
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: gaussian_train --jsonl <path> [--smpl <path>] [--num-gaussians <int>]"
                     " [--epochs <int>] [--lr <float>] [--output-dir <path>]"
                     " [--scale-reg <float>] [--scale-max-reg <float>] [--scale-max <float>]"
                     " [--offset-reg <float>] [--mesh-reg <float>]"
                     " [--mesh-max-dist <float>] [--color-lr <float>]"
                     " [--sh-degree <int>]"
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
    const float scale_max_reg_weight = options.scale_max_reg_weight;
    const float scale_max_value = options.scale_max_value;
    const float offset_reg_weight = options.offset_reg_weight;
    const float mesh_reg_weight = options.mesh_reg_weight;
    const float mesh_reg_max_dist = options.mesh_reg_max_dist;
    const float color_lr = options.color_lr;
    const int sh_degree = options.sh_degree;
    const int viewer_every = std::max(1, options.viewer_every);
    const float pose_lr = 1e-4f;
    const float outside_mask_weight = 0.1f;
    const float render_scale_modifier = 0.000001f;
    const float render_threshold = 3.0f / 255.0f;
    const int batch_size = 4;

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

    std::vector<torch::Tensor> base_params = {
        avatar.g_scales,
        avatar.g_rots,
        avatar.g_opacities,
        avatar.g_offsets};
    std::vector<torch::Tensor> color_params = {use_sh ? avatar.g_sh : avatar.g_colors};
    std::vector<torch::Tensor> pose_params = {pose_offsets};
    torch::optim::Adam optimizer(base_params, torch::optim::AdamOptions(lr));
    optimizer.add_param_group({color_params});
    auto &color_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(color_group.options()).lr(color_lr);
    optimizer.add_param_group({pose_params});
    auto &pose_group = optimizer.param_groups().back();
    static_cast<torch::optim::AdamOptions &>(pose_group.options()).lr(pose_lr);

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
            std::vector<float> recon_values;
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
                std::tie(view_mat, proj_mat, tan_fovx, tan_fovy) = BuildProjection(f_render, full_w, full_h, device);

                auto colors = use_sh ? torch::zeros({0}, avatar.g_colors.options()) : avatar.g_colors;
                auto image_full = GaussianRasterizer::apply(
                    means3D,
                    colors,
                    avatar.g_opacities,
                    torch::exp(avatar.g_scales),
                    current_rots,
                    render_scale_modifier,
                    view_mat,
                    proj_mat,
                    tan_fovx,
                    tan_fovy,
                    full_h,
                    full_w,
                    current_sh,
                    use_sh ? sh_degree : 0,
                    cam_pos,
                    false);
                auto image = CropRenderToTarget(image_full, W, H, sample.crop_cx, sample.crop_cy);
                if (!image.defined() || image.dim() != 3 || image.size(0) != 3 ||
                    image.size(1) != H || image.size(2) != W)
                {
                    skipped_malformed++;
                    continue;
                }
                // if (!IsMaskCoverageValidTensor(image, cached_entry.matte_mask, 0.3f, render_threshold))
                // {
                //     skipped_mask++;
                //     continue;
                // }

                auto outside_mask = 1.0f - cached_entry.matte_mask;
                auto outside_loss = torch::mean(image * outside_mask);

                auto recon_loss = torch::mse_loss(image, target);
                auto loss_value = recon_loss.item<float>();
                auto outside_value = outside_loss.item<float>();
                if (!std::isfinite(loss_value) || !std::isfinite(outside_value))
                {
                    skipped_malformed++;
                    continue;
                }

                recon_losses.push_back(recon_loss);
                outside_losses.push_back(outside_loss);
                recon_values.push_back(loss_value);
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

            float median = ComputeMedian(recon_values);
            float mad = ComputeMad(recon_values, median);
            float outlier_threshold = median + 3.0f * std::max(mad, 1e-6f);

            torch::Tensor recon_sum = torch::zeros({}, torch::TensorOptions().device(device));
            int inlier_count = 0;
            int skipped_outlier = 0;
            int last_inlier_idx = -1;
            int best_inlier_idx = -1;
            float best_inlier_loss = std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < recon_losses.size(); ++i)
            {
                if (recon_values[i] > outlier_threshold)
                {
                    skipped_outlier++;
                    continue;
                }
                recon_sum = recon_sum + recon_losses[i] + outside_mask_weight * outside_losses[i];
                inlier_count++;
                last_inlier_idx = static_cast<int>(i);
                if (recon_values[i] < best_inlier_loss)
                {
                    best_inlier_loss = recon_values[i];
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
            auto scale_vals = torch::exp(avatar.g_scales) * render_scale_modifier;
            auto scale_reg = torch::mean(scale_vals.pow(2));
            auto scale_max_reg = torch::mean(torch::relu(scale_vals - scale_max_value).pow(2));
            auto offset_reg = torch::mean(avatar.g_offsets.pow(2));
            auto offset_norm = torch::sqrt(avatar.g_offsets.pow(2).sum(1) + 1e-12f);
            auto mesh_reg = torch::mean(torch::relu(offset_norm - mesh_reg_max_dist).pow(2));
            auto loss = recon_loss + scale_reg_weight * scale_reg + offset_reg_weight * offset_reg +
                        mesh_reg_weight * mesh_reg + scale_max_reg_weight * scale_max_reg;
            loss.backward();
            optimizer.step();

            if (use_sh && (batch_step % 10 == 0))
            {
                auto sh_abs_mean = avatar.g_sh.abs().mean().item<float>();
                auto sh_std = avatar.g_sh.std().item<float>();
                auto sh_min = avatar.g_sh.min().item<float>();
                auto sh_max = avatar.g_sh.max().item<float>();
                std::cout << "SH stats batch " << batch_step
                          << " abs_mean=" << sh_abs_mean
                          << " std=" << sh_std
                          << " min=" << sh_min
                          << " max=" << sh_max
                          << std::endl;
            }

            if (best_inlier_idx >= 0)
            {
                const size_t sample_idx = static_cast<size_t>(best_inlier_idx);
                last_render = batch_renders[sample_idx];

                if (batch_step % 10 == 0)
                {
                    const auto &sample = batch_samples[sample_idx];
                    const int H = static_cast<int>(last_render.size(1));
                    const int W = static_cast<int>(last_render.size(2));
                    std::cout << "Debug batch " << batch_step
                              << " img_w=" << sample.img_w << " img_h=" << sample.img_h
                              << " crop_w=" << W << " crop_h=" << H
                              << " crop_cx=" << sample.crop_cx << " crop_cy=" << sample.crop_cy
                              << " crop_size=" << sample.crop_size
                              << " focal=" << sample.focal_length
                              << " f_render=" << sample.focal_length
                              << " y_sign=" << sample.y_sign
                              << " skipped_malformed=" << skipped_malformed
                              << " skipped_mask=" << skipped_mask
                              << " skipped_outlier=" << skipped_outlier
                              << std::endl;

                    cv::Mat render_bgr = TensorToBgr(last_render);
                    if (!render_bgr.empty())
                    {
                        cv::Mat target_bgr = batch_crops[sample_idx];
                        if (render_bgr.size() != target_bgr.size())
                        {
                            cv::resize(render_bgr, render_bgr, target_bgr.size(), 0, 0, cv::INTER_AREA);
                        }
                        cv::Mat side_by_side;
                        cv::hconcat(target_bgr, render_bgr, side_by_side);
                        const auto &sample = batch_samples[sample_idx];
                        const std::string label = "frame=" + std::to_string(sample.frame) +
                                                  " loss=" + std::to_string(best_inlier_loss);
                        const int left_x = 10;
                        const int right_x = target_bgr.cols + 10;
                        const int y = 30;
                        cv::putText(side_by_side, label, cv::Point(left_x, y),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                        cv::putText(side_by_side, label, cv::Point(left_x, y),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                        cv::putText(side_by_side, label, cv::Point(right_x, y),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                        cv::putText(side_by_side, label, cv::Point(right_x, y),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                        std::filesystem::path pair_path = out_dir_path /
                                                          ("pair_e" + std::to_string(epoch) + "_b" + std::to_string(batch_step) + ".png");
                        cv::imwrite(pair_path.string(), side_by_side);
                    }
                    std::cout << "Epoch " << epoch << " Batch " << batch_step
                              << " Loss: " << loss.item<float>() << std::endl;
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
                    auto scales = torch::exp(avatar.g_scales);
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
