#include "DepthAnything.h"

#include <algorithm>
#include <cstdint>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

} // namespace

DepthAnything::DepthAnything(const DepthAnythingOptions& options)
    : options_(options) {}

bool DepthAnything::Load(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "depth_anything");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (options_.use_cuda) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
            } catch (const Ort::Exception&) {
            }
        }

#ifdef _WIN32
        std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
#endif
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions allocator;
        input_names_.clear();
        output_names_.clear();

        const size_t input_count = session_->GetInputCount();
        for (size_t i = 0; i < input_count; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(name.get());
        }

        const size_t output_count = session_->GetOutputCount();
        for (size_t i = 0; i < output_count; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(name.get());
        }

        input_width_ = options_.input_size;
        input_height_ = options_.input_size;
        if (input_count > 0) {
            const auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() >= 4) {
                const int64_t h = shape[2];
                const int64_t w = shape[3];
                if (h > 0) input_height_ = static_cast<int>(h);
                if (w > 0) input_width_ = static_cast<int>(w);
            }
        }
    } catch (const Ort::Exception&) {
        env_.reset();
        session_.reset();
        input_names_.clear();
        output_names_.clear();
        input_width_ = 0;
        input_height_ = 0;
        return false;
    }

    return !input_names_.empty() && !output_names_.empty() && input_width_ > 0 && input_height_ > 0;
}

bool DepthAnything::ComputeDepth(const cv::Mat& input_bgr, cv::Mat* out_depth) {
    if (!out_depth || !session_ || input_bgr.empty()) {
        return false;
    }

    cv::Mat resized;
    cv::resize(input_bgr, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    std::vector<float> input_data;
    input_data.reserve(static_cast<size_t>(1 * 3 * input_height_ * input_width_));
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_height_; ++y) {
            const float* row_ptr = channels[c].ptr<float>(y);
            for (int x = 0; x < input_width_; ++x) {
                input_data.push_back((row_ptr[x] - kMean[c]) / kStd[c]);
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names;
    input_names.reserve(input_names_.size());
    for (const auto& name : input_names_) {
        input_names.push_back(name.c_str());
    }
    std::vector<const char*> output_names;
    output_names.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        output_names.push_back(name.c_str());
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), &input_tensor, 1,
                                 output_names.data(), output_names.size());
    if (outputs.empty() || !outputs[0].IsTensor()) {
        return false;
    }

    const auto& depth_tensor = outputs[0];
    const auto depth_shape = depth_tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (depth_shape.size() < 2) {
        return false;
    }

    const int64_t h = depth_shape[depth_shape.size() - 2];
    const int64_t w = depth_shape[depth_shape.size() - 1];
    if (h <= 0 || w <= 0) {
        return false;
    }
    const int depth_h = static_cast<int>(h);
    const int depth_w = static_cast<int>(w);
    const size_t pixel_count = static_cast<size_t>(depth_h) * static_cast<size_t>(depth_w);

    const float* depth_data = depth_tensor.GetTensorData<float>();
    if (!depth_data) {
        return false;
    }

    const auto minmax = std::minmax_element(depth_data, depth_data + pixel_count);
    const float min_depth = *minmax.first;
    const float max_depth = *minmax.second;
    const float range = std::max(max_depth - min_depth, 1e-6f);

    cv::Mat depth_u8(depth_h, depth_w, CV_8U);
    for (int y = 0; y < depth_h; ++y) {
        uint8_t* out_row = depth_u8.ptr<uint8_t>(y);
        for (int x = 0; x < depth_w; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(depth_w) + static_cast<size_t>(x);
            float norm = (depth_data[idx] - min_depth) / range;
            norm = std::clamp(norm, 0.0f, 1.0f);
            out_row[x] = static_cast<uint8_t>(std::lround(norm * 255.0f));
        }
    }

    // Improve local contrast (e.g. facial geometry) after global normalization.
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    cv::Mat enhanced_depth;
    clahe->apply(depth_u8, enhanced_depth);

    if (options_.invert_depth) {
        cv::bitwise_not(enhanced_depth, enhanced_depth);
    }

    cv::resize(enhanced_depth, *out_depth, input_bgr.size(), 0, 0, cv::INTER_CUBIC);
    return true;
}
