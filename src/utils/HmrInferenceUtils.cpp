#include "HmrInferenceUtils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include <torch/torch.h>
#include <onnxruntime_c_api.h>

#include "SmplLBS.h"

namespace {

constexpr int kInputW = 224;
constexpr int kInputH = 224;
const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
const std::vector<float> kStd  = {0.229f, 0.224f, 0.225f};

std::vector<float> PreprocessImage(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(kInputW, kInputH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    std::vector<float> input_data;
    input_data.reserve(1 * 3 * kInputH * kInputW);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < kInputH; ++h) {
            for (int w = 0; w < kInputW; ++w) {
                float val = channels[c].at<float>(h, w);
                input_data.push_back((val - kMean[c]) / kStd[c]);
            }
        }
    }
    return input_data;
}

torch::Tensor Rot6dToAxisAngle(const torch::Tensor& rot6d) {
    auto a1 = rot6d.slice(-1, 0, 3);
    auto a2 = rot6d.slice(-1, 3, 6);
    auto b1 = torch::nn::functional::normalize(a1, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
    auto dot = (b1 * a2).sum(-1, true);
    auto b2 = torch::nn::functional::normalize(a2 - dot * b1, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
    auto b3 = torch::cross(b1, b2, -1);
    auto rotmat = torch::stack({b1, b2, b3}, -1);

    auto trace = rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 0, 0})
               + rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 1, 1})
               + rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 2, 2});
    auto cos_angle = torch::clamp((trace - 1.0f) * 0.5f, -1.0f + 1e-6f, 1.0f - 1e-6f);
    auto angle = torch::acos(cos_angle);

    auto rx = rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 2, 1})
            - rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 1, 2});
    auto ry = rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 0, 2})
            - rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 2, 0});
    auto rz = rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 1, 0})
            - rotmat.index({torch::indexing::Slice(), torch::indexing::Slice(), 0, 1});
    auto axis = torch::stack({rx, ry, rz}, -1);

    auto sin_angle = torch::sin(angle);
    auto denom = 2.0f * sin_angle;
    auto safe = torch::where(denom.abs() < 1e-6f, torch::ones_like(denom), denom);
    axis = axis / safe.unsqueeze(-1);

    auto axis_angle = axis * angle.unsqueeze(-1);
    auto small = angle.abs() < 1e-6f;
    axis_angle = torch::where(small.unsqueeze(-1), torch::zeros_like(axis_angle), axis_angle);
    return axis_angle;
}

torch::Tensor PoseToAxisAngle(const SmplResult& res) {
    const int pose_size = static_cast<int>(res.pose.size());
    if (pose_size == 72) {
        auto pose = torch::from_blob(const_cast<float*>(res.pose.data()), {1, 24, 3}, torch::kFloat).clone();
        return pose;
    }
    if (pose_size == 144) {
        auto pose6d = torch::from_blob(const_cast<float*>(res.pose.data()), {1, 24, 6}, torch::kFloat).clone();
        return Rot6dToAxisAngle(pose6d);
    }
    if (pose_size % 3 == 0 && (pose_size / 3) == 24) {
        auto pose = torch::from_blob(const_cast<float*>(res.pose.data()), {1, 24, 3}, torch::kFloat).clone();
        return pose;
    }
    throw std::runtime_error("Unsupported pose size: " + std::to_string(pose_size));
}

bool EnsureOutputDir(const std::string& dir) {
    if (dir.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return !ec;
}

std::string MakeFrameName(int frame_idx) {
    std::ostringstream oss;
    oss << "overlay_" << std::setw(6) << std::setfill('0') << frame_idx << ".png";
    return oss.str();
}

void WriteResult(std::ofstream& out, int frame_idx, const SmplResult& res) {
    out << "Frame: " << frame_idx << "\n";
    out << "Pose: ";
    for (float v : res.pose) out << v << " ";
    out << "\n";
    out << "Shape: ";
    for (float v : res.shape) out << v << " ";
    out << "\n";
    out << "Camera: ";
    for (float v : res.camera) out << v << " ";
    out << "\n";
    out << "----------------------------------------\n";
}

int CountProjectedInFrame(const torch::Tensor& verts_cpu, float s, float tx, float ty, float y_sign,
                          int width, int height) {
    const float img_size = static_cast<float>(kInputW);
    const float scale_x = static_cast<float>(width) / img_size;
    const float scale_y = static_cast<float>(height) / img_size;

    auto verts_acc = verts_cpu.accessor<float, 2>();
    int count = 0;
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0];
        const float Y = verts_acc[i][1] * y_sign;
        const float u = (s * (X + tx) + img_size * 0.5f) * scale_x;
        const float v = (s * (Y + ty) + img_size * 0.5f) * scale_y;
        if (u >= 0.0f && v >= 0.0f && u < width && v < height) {
            count++;
        }
    }
    return count;
}

void DrawVerticesOverlay(cv::Mat& frame, const torch::Tensor& verts, const std::vector<float>& cam) {
    if (cam.size() < 3) return;
    const float s = cam[0];
    const float tx = cam[1];
    const float ty = cam[2];

    auto verts_cpu = verts.squeeze(0).to(torch::kCPU).contiguous();

    const int width = frame.cols;
    const int height = frame.rows;

    const int count_pos = CountProjectedInFrame(verts_cpu, s, tx, ty, 1.0f, width, height);
    const int count_neg = CountProjectedInFrame(verts_cpu, s, tx, ty, -1.0f, width, height);
    const float y_sign = (count_neg > count_pos) ? -1.0f : 1.0f;

    const float img_size = static_cast<float>(kInputW);
    const float scale_x = static_cast<float>(width) / img_size;
    const float scale_y = static_cast<float>(height) / img_size;

    auto verts_acc = verts_cpu.accessor<float, 2>();
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0];
        const float Y = verts_acc[i][1] * y_sign;
        const float u = (s * (X + tx) + img_size * 0.5f) * scale_x;
        const float v = (s * (Y + ty) + img_size * 0.5f) * scale_y;
        const int ui = static_cast<int>(u);
        const int vi = static_cast<int>(v);
        if (ui < 0 || vi < 0 || ui >= width || vi >= height) continue;
        cv::circle(frame, cv::Point(ui, vi), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
}

} // namespace

bool RunHmrInferenceOnVideo(const std::string& model_path,
                            const std::string& video_path,
                            const HmrOutputOptions& options,
                            ResultsDict* out_results) {
    if (out_results) out_results->clear();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CLIF_HMR");
    Ort::SessionOptions session_options;
    // try {
    //     Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    //     std::cout << "Using ONNX Runtime CUDA Execution Provider." << std::endl;
    // } catch (const Ort::Exception& e) {
    //     std::cout << "CUDA EP not available (" << e.what() << "). Falling back to CPU." << std::endl;
    // }

    std::wstring wide_model_path(model_path.begin(), model_path.end());
    std::unique_ptr<Ort::Session> session;

    try {
        session = std::make_unique<Ort::Session>(env, wide_model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "\n[ERROR] Failed to load model: " << e.what() << std::endl;
        std::cerr << "Ensure .onnx and .onnx.data are in the same folder." << std::endl;
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* input_names[] = {"image", "bbox"};
    const char* output_names[] = {"pose", "betas", "cam"};

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video." << std::endl;
        return false;
    }

    const bool save_outputs = options.save_outputs && !options.output_dir.empty();
    std::ofstream outfile;
    if (save_outputs) {
        if (!EnsureOutputDir(options.output_dir)) {
            std::cerr << "Failed to create output dir: " << options.output_dir << std::endl;
            return false;
        }
        outfile.open(std::filesystem::path(options.output_dir) / "output.txt");
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output.txt in " << options.output_dir << std::endl;
            return false;
        }
    }

    std::unique_ptr<SMPLLayer> smpl_layer;
    if (save_outputs) {
        smpl_layer = std::make_unique<SMPLLayer>(options.smpl_model_path);
    }

    std::vector<float> bbox_data = {112.0f, 112.0f, 224.0f / 200.0f};
    std::vector<int64_t> bbox_shape = {1, 3};

    std::cout << "Processing video..." << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    while (cap.read(frame)) {
        std::vector<float> img_data = PreprocessImage(frame);
        std::vector<int64_t> img_shape = {1, 3, kInputH, kInputW};

        Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
            mem_info, img_data.data(), img_data.size(), img_shape.data(), img_shape.size());
        Ort::Value bbox_tensor = Ort::Value::CreateTensor<float>(
            mem_info, bbox_data.data(), bbox_data.size(), bbox_shape.data(), bbox_shape.size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(img_tensor));
        input_tensors.push_back(std::move(bbox_tensor));

        try {
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names, input_tensors.data(), 2,
                output_names, 3);

            float* pose_ptr  = output_tensors[0].GetTensorMutableData<float>();
            float* shape_ptr = output_tensors[1].GetTensorMutableData<float>();
            float* cam_ptr   = output_tensors[2].GetTensorMutableData<float>();

            SmplResult res;
            auto pose_info = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            int pose_size = static_cast<int>(pose_info[1]);

            res.pose.assign(pose_ptr, pose_ptr + pose_size);
            res.shape.assign(shape_ptr, shape_ptr + 10);
            res.camera.assign(cam_ptr, cam_ptr + 3);

            if (out_results) {
                (*out_results)[frame_idx] = res;
            }

            if (save_outputs) {
                WriteResult(outfile, frame_idx, res);

                torch::NoGradGuard no_grad;
                auto betas = torch::from_blob(res.shape.data(), {1, 10}, torch::kFloat).clone();
                auto pose_axis = PoseToAxisAngle(res);
                auto trans = torch::zeros({1, 3}, torch::kFloat);

                auto smpl_out = smpl_layer->forward(betas, pose_axis, trans);
                cv::Mat overlay = frame.clone();
                DrawVerticesOverlay(overlay, smpl_out.vertices, res.camera);

                const auto out_path = std::filesystem::path(options.output_dir) / MakeFrameName(frame_idx);
                cv::imwrite(out_path.string(), overlay);
            }

        } catch (const Ort::Exception& e) {
            std::cerr << "Inference error frame " << frame_idx << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Postprocess error frame " << frame_idx << ": " << e.what() << std::endl;
        }

        if (frame_idx % 30 == 0) std::cout << "Frame: " << frame_idx << std::endl;
        frame_idx++;
    }

    if (outfile.is_open()) outfile.close();
    std::cout << "Finished." << std::endl;
    return true;
}
