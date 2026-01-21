#include "HmrInferenceUtils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "SmplLBS.h"

namespace {

constexpr int kInputW = 224;
constexpr int kInputH = 224;
const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
const std::vector<float> kStd  = {0.229f, 0.224f, 0.225f};

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

template <typename T>
struct TrtDeleter {
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

void CheckCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << msg << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

int64_t Volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

TrtUniquePtr<nvinfer1::ICudaEngine> BuildEngineFromOnnx(const std::string& model_path,
                                                        TrtLogger& logger) {
    const std::filesystem::path engine_path = std::filesystem::path(model_path).string() + ".engine";
    if (std::filesystem::exists(engine_path)) {
        std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
        if (!engine_file) {
            throw std::runtime_error("Failed to open cached engine: " + engine_path.string());
        }
        const std::streamsize size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        if (!engine_file.read(buffer.data(), size)) {
            throw std::runtime_error("Failed to read cached engine: " + engine_path.string());
        }

        TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime.");

        TrtUniquePtr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize cached engine: " + engine_path.string());
        }
        std::cout << "Loaded cached TensorRT engine: " << engine_path.string() << std::endl;
        return engine;
    }

    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("Failed to create TensorRT builder.");

    const uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) throw std::runtime_error("Failed to create TensorRT network.");

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) throw std::runtime_error("Failed to create TensorRT ONNX parser.");

    if (!parser->parseFromFile(model_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX model: " + model_path);
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("Failed to create TensorRT builder config.");

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    TrtUniquePtr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config));
    if (!engine) throw std::runtime_error("Failed to build TensorRT engine.");

    TrtUniquePtr<nvinfer1::IHostMemory> serialized(engine->serialize());
    if (!serialized) throw std::runtime_error("Failed to serialize TensorRT engine.");

    std::ofstream out(engine_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open engine output: " + engine_path.string());
    }
    out.write(static_cast<const char*>(serialized->data()), serialized->size());
    if (!out) {
        throw std::runtime_error("Failed to write engine output: " + engine_path.string());
    }
    std::cout << "Saved TensorRT engine: " << engine_path.string() << std::endl;

    return engine;
}

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

    TrtLogger logger;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    try {
        engine = BuildEngineFromOnnx(model_path, logger);
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Failed to build TensorRT engine: " << e.what() << std::endl;
        std::cerr << "Ensure .onnx and .onnx.data are in the same folder." << std::endl;
        return false;
    }

    TrtUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "\n[ERROR] Failed to create TensorRT execution context." << std::endl;
        return false;
    }

    const int idx_image = engine->getBindingIndex("image");
    const int idx_bbox = engine->getBindingIndex("bbox");
    const int idx_pose = engine->getBindingIndex("pose");
    const int idx_betas = engine->getBindingIndex("betas");
    const int idx_cam = engine->getBindingIndex("cam");
    if (idx_image < 0 || idx_bbox < 0 || idx_pose < 0 || idx_betas < 0 || idx_cam < 0) {
        std::cerr << "\n[ERROR] Failed to find required TensorRT bindings (image/bbox/pose/betas/cam)." << std::endl;
        return false;
    }

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

    const nvinfer1::Dims4 img_dims(1, 3, kInputH, kInputW);
    const nvinfer1::Dims2 bbox_dims(1, 3);

    if (!context->setBindingDimensions(idx_image, img_dims) ||
        !context->setBindingDimensions(idx_bbox, bbox_dims)) {
        std::cerr << "\n[ERROR] Failed to set TensorRT input dimensions." << std::endl;
        return false;
    }
    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "\n[ERROR] TensorRT input dimensions not fully specified." << std::endl;
        return false;
    }

    const auto img_out_dims = context->getBindingDimensions(idx_image);
    const auto bbox_out_dims = context->getBindingDimensions(idx_bbox);
    const auto pose_dims = context->getBindingDimensions(idx_pose);
    const auto betas_dims = context->getBindingDimensions(idx_betas);
    const auto cam_dims = context->getBindingDimensions(idx_cam);

    const size_t img_bytes = static_cast<size_t>(Volume(img_out_dims)) * sizeof(float);
    const size_t bbox_bytes = static_cast<size_t>(Volume(bbox_out_dims)) * sizeof(float);
    const size_t pose_bytes = static_cast<size_t>(Volume(pose_dims)) * sizeof(float);
    const size_t betas_bytes = static_cast<size_t>(Volume(betas_dims)) * sizeof(float);
    const size_t cam_bytes = static_cast<size_t>(Volume(cam_dims)) * sizeof(float);

    void* d_image = nullptr;
    void* d_bbox = nullptr;
    void* d_pose = nullptr;
    void* d_betas = nullptr;
    void* d_cam = nullptr;
    cudaStream_t stream = nullptr;

    try {
        CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");
        CheckCuda(cudaMalloc(&d_image, img_bytes), "cudaMalloc image failed");
        CheckCuda(cudaMalloc(&d_bbox, bbox_bytes), "cudaMalloc bbox failed");
        CheckCuda(cudaMalloc(&d_pose, pose_bytes), "cudaMalloc pose failed");
        CheckCuda(cudaMalloc(&d_betas, betas_bytes), "cudaMalloc betas failed");
        CheckCuda(cudaMalloc(&d_cam, cam_bytes), "cudaMalloc cam failed");
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] CUDA setup failed: " << e.what() << std::endl;
        if (d_image) cudaFree(d_image);
        if (d_bbox) cudaFree(d_bbox);
        if (d_pose) cudaFree(d_pose);
        if (d_betas) cudaFree(d_betas);
        if (d_cam) cudaFree(d_cam);
        if (stream) cudaStreamDestroy(stream);
        return false;
    }

    std::vector<void*> bindings(engine->getNbBindings(), nullptr);
    bindings[idx_image] = d_image;
    bindings[idx_bbox] = d_bbox;
    bindings[idx_pose] = d_pose;
    bindings[idx_betas] = d_betas;
    bindings[idx_cam] = d_cam;

    std::vector<float> pose_out(static_cast<size_t>(Volume(pose_dims)));
    std::vector<float> betas_out(static_cast<size_t>(Volume(betas_dims)));
    std::vector<float> cam_out(static_cast<size_t>(Volume(cam_dims)));

    std::cout << "Processing video..." << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    while (cap.read(frame)) {
        std::vector<float> img_data = PreprocessImage(frame);

        try {
            CheckCuda(cudaMemcpyAsync(d_image, img_data.data(), img_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync image failed");
            CheckCuda(cudaMemcpyAsync(d_bbox, bbox_data.data(), bbox_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync bbox failed");

            if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
                throw std::runtime_error("TensorRT enqueue failed");
            }

            CheckCuda(cudaMemcpyAsync(pose_out.data(), d_pose, pose_bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync pose failed");
            CheckCuda(cudaMemcpyAsync(betas_out.data(), d_betas, betas_bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync betas failed");
            CheckCuda(cudaMemcpyAsync(cam_out.data(), d_cam, cam_bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync cam failed");

            CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

            SmplResult res;
            int pose_size = static_cast<int>(pose_out.size());

            res.pose.assign(pose_out.begin(), pose_out.end());
            res.shape.assign(betas_out.begin(), betas_out.end());
            res.camera.assign(cam_out.begin(), cam_out.end());

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

        } catch (const std::exception& e) {
            std::cerr << "Inference/postprocess error frame " << frame_idx << ": " << e.what() << std::endl;
        }

        if (frame_idx % 30 == 0) std::cout << "Frame: " << frame_idx << std::endl;
        frame_idx++;
    }

    if (outfile.is_open()) outfile.close();
    if (d_image) cudaFree(d_image);
    if (d_bbox) cudaFree(d_bbox);
    if (d_pose) cudaFree(d_pose);
    if (d_betas) cudaFree(d_betas);
    if (d_cam) cudaFree(d_cam);
    if (stream) cudaStreamDestroy(stream);
    std::cout << "Finished." << std::endl;
    return true;
}
