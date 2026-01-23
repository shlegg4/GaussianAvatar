#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "GaussianRasterizer.h"
#include "utils/HmrInferenceUtils.h"
#include "utils/HmrMathHelpers.h"
#include "utils/SmplLBS.h"

struct TrainSample {
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

bool ExtractStringField(const std::string& line, const std::string& key, std::string* out) {
    const std::string tag = "\"" + key + "\":\"";
    size_t start = line.find(tag);
    if (start == std::string::npos) return false;
    start += tag.size();
    size_t end = line.find('"', start);
    if (end == std::string::npos) return false;
    *out = line.substr(start, end - start);
    return true;
}

bool ExtractNumberField(const std::string& line, const std::string& key, double* out) {
    const std::string tag = "\"" + key + "\":";
    size_t start = line.find(tag);
    if (start == std::string::npos) return false;
    start += tag.size();
    const char* ptr = line.c_str() + start;
    char* end = nullptr;
    double val = std::strtod(ptr, &end);
    if (end == ptr) return false;
    *out = val;
    return true;
}

bool ExtractArrayField(const std::string& line, const std::string& key, std::vector<float>* out) {
    const std::string tag = "\"" + key + "\":[";
    size_t start = line.find(tag);
    if (start == std::string::npos) return false;
    start += tag.size();
    size_t end = line.find(']', start);
    if (end == std::string::npos) return false;
    std::string content = line.substr(start, end - start);
    out->clear();
    if (content.empty()) return true;
    size_t pos = 0;
    while (pos < content.size()) {
        size_t comma = content.find(',', pos);
        std::string token = (comma == std::string::npos) ?
            content.substr(pos) : content.substr(pos, comma - pos);
        if (!token.empty()) {
            char* end_ptr = nullptr;
            float val = std::strtof(token.c_str(), &end_ptr);
            if (end_ptr != token.c_str()) {
                out->push_back(val);
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return true;
}

bool ParseTrainSample(const std::string& line, TrainSample* out) {
    TrainSample sample;
    if (!ExtractStringField(line, "crop", &sample.crop_path)) return false;
    if (sample.crop_path.empty()) return false;

    double value = 0.0;
    if (!ExtractNumberField(line, "img_w", &value)) return false;
    sample.img_w = static_cast<int>(value);
    if (!ExtractNumberField(line, "img_h", &value)) return false;
    sample.img_h = static_cast<int>(value);
    if (!ExtractNumberField(line, "crop_cx", &value)) return false;
    sample.crop_cx = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_cy", &value)) return false;
    sample.crop_cy = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_size", &value)) return false;
    sample.crop_size = static_cast<float>(value);
    if (!ExtractNumberField(line, "focal_length", &value)) return false;
    sample.focal_length = static_cast<float>(value);
    if (!ExtractNumberField(line, "y_sign", &value)) return false;
    sample.y_sign = static_cast<float>(value);

    if (!ExtractArrayField(line, "pose", &sample.pose)) return false;
    if (!ExtractArrayField(line, "betas", &sample.betas)) return false;
    if (!ExtractArrayField(line, "cam", &sample.cam)) return false;

    if (sample.pose.empty() || sample.betas.empty() || sample.cam.empty()) return false;
    *out = std::move(sample);
    return true;
}

torch::Tensor LoadImageTensor(const cv::Mat& bgr, torch::Device device) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(
        float_img.data,
        {float_img.rows, float_img.cols, 3},
        torch::kFloat).clone();
    tensor = tensor.permute({2, 0, 1}).contiguous().to(device);
    return tensor;
}

bool SaveImageTensorPng(const std::string& path, const torch::Tensor& image) {
    if (!image.defined()) return false;
    auto img = image.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kU8).cpu();
    if (img.dim() != 3 || img.size(0) != 3) return false;
    img = img.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return cv::imwrite(path, bgr);
}

cv::Mat TensorToBgr(const torch::Tensor& image) {
    if (!image.defined()) return cv::Mat();
    auto img = image.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kU8).cpu();
    if (img.dim() != 3 || img.size(0) != 3) return cv::Mat();
    img = img.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height, torch::Device device) {
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

torch::Tensor CropRenderToTarget(const torch::Tensor& full_render, int crop_w, int crop_h,
                                 float crop_cx, float crop_cy) {
    if (!full_render.defined() || full_render.dim() != 3) {
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
    if (src_w <= 0 || src_h <= 0) {
        return output;
    }

    const int dst_x0 = src_x0 - x0;
    const int dst_y0 = src_y0 - y0;

    using torch::indexing::Slice;
    output.index_put_({Slice(), Slice(dst_y0, dst_y0 + src_h), Slice(dst_x0, dst_x0 + src_w)},
                      full_render.index({Slice(), Slice(src_y0, src_y0 + src_h), Slice(src_x0, src_x0 + src_w)}));
    return output;
}

struct GaussianAvatar : torch::nn::Module {
    std::shared_ptr<SMPLLayer> smpl;
    torch::Tensor g_scales, g_rots, g_opacities, g_colors;
    torch::Tensor g_bary_coords, g_face_indices, faces_buffer;

    GaussianAvatar(const std::string& model_path) {
        smpl = std::make_shared<SMPLLayer>(model_path);
        register_module("smpl", smpl);
    }

    void init_gaussians(int num_gaussians, torch::Tensor faces_idx) {
        auto device = smpl->v_template.device();
        faces_buffer = faces_idx.to(device);
        int num_faces = faces_buffer.size(0);

        g_face_indices = torch::randint(0, num_faces, {num_gaussians}, torch::kLong).to(device);

        auto r1 = torch::rand({num_gaussians, 1}, device);
        auto r2 = torch::rand({num_gaussians, 1}, device);
        auto mask = (r1 + r2) > 1.0;
        r1.index_put_({mask}, 1.0 - r1.index({mask}));
        r2.index_put_({mask}, 1.0 - r2.index({mask}));

        auto w = 1.0 - r1 - r2;
        g_bary_coords = torch::cat({r1, r2, w}, 1);

        register_buffer("g_face_indices", g_face_indices);
        register_buffer("g_bary_coords", g_bary_coords);

        g_scales = torch::full({num_gaussians, 3}, -2.5, torch::requires_grad().device(device));
        auto g_rots_init = torch::zeros({num_gaussians, 4}, torch::TensorOptions().device(device));
        g_rots_init.index_put_({torch::indexing::Slice(), 0}, 1.0);
        g_rots = g_rots_init.detach().clone().set_requires_grad(true);
        g_opacities = torch::full({num_gaussians, 1}, 0.1, torch::requires_grad().device(device));
        g_colors = torch::full({num_gaussians, 3}, 0.5, torch::requires_grad().device(device));

        register_parameter("g_scales", g_scales);
        register_parameter("g_rots", g_rots);
        register_parameter("g_opacities", g_opacities);
        register_parameter("g_colors", g_colors);
    }

    torch::Tensor forward(torch::Tensor betas, torch::Tensor pose, torch::Tensor trans) {
        auto smpl_out = smpl->forward(betas, pose, trans);
        auto verts = smpl_out.vertices[0];
        auto selected_faces = faces_buffer.index_select(0, g_face_indices);

        auto A = verts.index_select(0, selected_faces.index({torch::indexing::Slice(), 0}));
        auto B = verts.index_select(0, selected_faces.index({torch::indexing::Slice(), 1}));
        auto C = verts.index_select(0, selected_faces.index({torch::indexing::Slice(), 2}));

        auto u = g_bary_coords.index({torch::indexing::Slice(), 0}).unsqueeze(1);
        auto v = g_bary_coords.index({torch::indexing::Slice(), 1}).unsqueeze(1);
        auto w = g_bary_coords.index({torch::indexing::Slice(), 2}).unsqueeze(1);

        return u * A + v * B + w * C;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: gaussian_train <gaussian_train.jsonl> [smpl_data.pt] [num_gaussians] [epochs] [lr] [output_dir]\n";
        return -1;
    }
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA Required for Rasterizer!" << std::endl;
        return -1;
    }

    std::string jsonl_path = argv[1];
    std::string smpl_model_path = (argc >= 3) ? argv[2] : "smpl_data.pt";
    int num_gaussians = (argc >= 4) ? std::stoi(argv[3]) : 5000;
    int epochs = (argc >= 5) ? std::stoi(argv[4]) : 1;
    float lr = (argc >= 6) ? std::stof(argv[5]) : 0.01f;
    std::string output_dir = (argc >= 7) ? argv[6] : "outputs";

    std::ifstream input(jsonl_path);
    if (!input.is_open()) {
        std::cerr << "Failed to open " << jsonl_path << std::endl;
        return -1;
    }

    std::vector<TrainSample> samples;
    std::string line;
    while (std::getline(input, line)) {
        TrainSample sample;
        if (ParseTrainSample(line, &sample)) {
            samples.push_back(std::move(sample));
        }
    }
    if (samples.empty()) {
        std::cerr << "No training samples found in " << jsonl_path << std::endl;
        return -1;
    }

    auto device = torch::kCUDA;
    GaussianAvatar avatar(smpl_model_path);
    avatar.to(device);

    std::ifstream smpl_in(smpl_model_path, std::ios::binary);
    std::vector<char> f_bytes((std::istreambuf_iterator<char>(smpl_in)), (std::istreambuf_iterator<char>()));
    auto dict = torch::pickle_load(f_bytes).toGenericDict();
    torch::Tensor faces = dict.at("faces").toTensor().to(torch::kLong).to(device);
    avatar.init_gaussians(num_gaussians, faces);

    auto sh = torch::zeros({0}, torch::TensorOptions().device(device));
    auto cam_pos = torch::zeros({3}, torch::TensorOptions().device(device));

    torch::optim::Adam optimizer(avatar.parameters(), torch::optim::AdamOptions(lr));

    torch::Tensor last_render;
    std::filesystem::path out_dir_path(output_dir);
    std::error_code out_ec;
    std::filesystem::create_directories(out_dir_path, out_ec);
    if (out_ec) {
        std::cerr << "Failed to create output dir: " << output_dir << std::endl;
        return -1;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        int step = 0;
        for (const auto& sample : samples) {
            cv::Mat crop = cv::imread(sample.crop_path);
            if (crop.empty()) {
                continue;
            }
            auto target = LoadImageTensor(crop, device);
            const int H = static_cast<int>(target.size(1));
            const int W = static_cast<int>(target.size(2));

            SmplResult res;
            res.pose = sample.pose;
            res.shape = sample.betas;
            res.camera = sample.cam;

            auto pose = PoseToAxisAngle(res).to(device);
            auto betas = torch::from_blob(res.shape.data(), {1, static_cast<int64_t>(res.shape.size())},
                                          torch::kFloat).clone().to(device);

            cv::Vec3f trans_cv = EstimateTranslation(res.camera, sample.crop_cx, sample.crop_cy,
                                                     sample.crop_size, sample.focal_length,
                                                     static_cast<float>(sample.img_w),
                                                     static_cast<float>(sample.img_h));
            auto trans = torch::tensor({trans_cv[0], trans_cv[1], trans_cv[2]},
                                       torch::TensorOptions().device(device).dtype(torch::kFloat));

            auto means3D = avatar.forward(betas, pose, torch::zeros({1, 3}, betas.options()));
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

            optimizer.zero_grad();
            auto image_full = GaussianRasterizer::apply(
                means3D,
                avatar.g_colors,
                avatar.g_opacities,
                avatar.g_scales,
                avatar.g_rots,
                0.005f,
                view_mat,
                proj_mat,
                tan_fovx,
                tan_fovy,
                full_h,
                full_w,
                sh,
                0,
                cam_pos,
                false);
            auto image = CropRenderToTarget(image_full, W, H, sample.crop_cx, sample.crop_cy);

            auto loss = torch::mse_loss(image, target);
            loss.backward();
            optimizer.step();

            last_render = image.detach();

            if (step % 10 == 0) {
                std::cout << "Debug step " << step
                          << " img_w=" << sample.img_w << " img_h=" << sample.img_h
                          << " crop_w=" << W << " crop_h=" << H
                          << " crop_cx=" << sample.crop_cx << " crop_cy=" << sample.crop_cy
                          << " crop_size=" << sample.crop_size
                          << " focal=" << sample.focal_length
                          << " f_render=" << f_render
                          << " trans=(" << trans_cv[0] << "," << trans_cv[1] << "," << trans_cv[2] << ")"
                          << " y_sign=" << sample.y_sign
                          << std::endl;
                std::filesystem::path render_path = out_dir_path /
                    ("render_e" + std::to_string(epoch) + "_s" + std::to_string(step) + ".png");
                SaveImageTensorPng(render_path.string(), last_render);

                cv::Mat render_bgr = TensorToBgr(last_render);
                if (!render_bgr.empty()) {
                    cv::Mat target_bgr = crop;
                    if (render_bgr.size() != target_bgr.size()) {
                        cv::resize(render_bgr, render_bgr, target_bgr.size(), 0, 0, cv::INTER_AREA);
                    }
                    cv::Mat side_by_side;
                    cv::hconcat(target_bgr, render_bgr, side_by_side);
                    std::filesystem::path pair_path = out_dir_path /
                        ("pair_e" + std::to_string(epoch) + "_s" + std::to_string(step) + ".png");
                    cv::imwrite(pair_path.string(), side_by_side);
                }
                std::cout << "Epoch " << epoch << " Step " << step
                          << " Loss: " << loss.item<float>() << std::endl;
            }
            step++;
        }
    }

    std::filesystem::path out_path = out_dir_path / "final_render.png";
    if (!SaveImageTensorPng(out_path.string(), last_render)) {
        std::cerr << "Failed to save final render to " << out_path.string() << std::endl;
    } else {
        std::cout << "Saved final render to " << out_path.string() << std::endl;
    }

    return 0;
}
