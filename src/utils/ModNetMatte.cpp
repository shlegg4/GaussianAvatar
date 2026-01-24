#include "ModNetMatte.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kNormMean = 0.5f;
constexpr float kNormStd = 0.5f;

} // namespace

ModNetMatte::ModNetMatte(const ModNetMatteOptions& options)
    : options_(options) {}

bool ModNetMatte::Load(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "modnet");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (options_.use_cuda) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
            } catch (const Ort::Exception&) {
            }
        }

        std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options);
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

        input_width_ = 0;
        input_height_ = 0;
        if (input_count > 0) {
            const auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() >= 4) {
                const int64_t h = shape[2];
                const int64_t w = shape[3];
                if (h > 0) input_height_ = static_cast<int>(h);
                if (w > 0) input_width_ = static_cast<int>(w);
            }
        }
        if (input_width_ <= 0) input_width_ = options_.input_size;
        if (input_height_ <= 0) input_height_ = options_.input_size;
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

cv::Mat ModNetMatte::Preprocess(const cv::Mat& bgr) const {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_AREA);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    if (options_.normalize_to_unit) {
        resized = (resized - kNormMean) / kNormStd;
    }
    return resized;
}

bool ModNetMatte::ComputeMatte(const cv::Mat& bgr, cv::Mat* out_matte) {
    if (!out_matte || !session_ || bgr.empty()) {
        return false;
    }

    cv::Mat input = Preprocess(bgr);
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);

    std::vector<float> input_data;
    input_data.reserve(static_cast<size_t>(1 * 3 * input_height_ * input_width_));
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height_; ++h) {
            const float* row_ptr = channels[c].ptr<float>(h);
            for (int w = 0; w < input_width_; ++w) {
                input_data.push_back(row_ptr[w]);
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

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

    const auto& output = outputs[0];
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    int out_h = 0;
    int out_w = 0;
    if (shape.size() == 4) {
        out_h = static_cast<int>(shape[2]);
        out_w = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        out_h = static_cast<int>(shape[1]);
        out_w = static_cast<int>(shape[2]);
    } else {
        return false;
    }
    if (out_h <= 0 || out_w <= 0) {
        return false;
    }

    const float* output_data = output.GetTensorData<float>();
    cv::Mat matte_small(out_h, out_w, CV_32F);
    std::memcpy(matte_small.data, output_data, sizeof(float) * static_cast<size_t>(out_h * out_w));
    cv::Mat matte;
    cv::resize(matte_small, matte, bgr.size(), 0, 0, cv::INTER_LINEAR);
    cv::threshold(matte, matte, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(matte, matte, 1.0, 1.0, cv::THRESH_TRUNC);

    *out_matte = matte;
    return true;
}

cv::Mat ModNetMatte::ApplyMatte(const cv::Mat& bgr, const cv::Mat& matte) const {
    if (bgr.empty() || matte.empty()) {
        return bgr.clone();
    }

    cv::Mat matte_resized;
    if (matte.size() != bgr.size()) {
        cv::resize(matte, matte_resized, bgr.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        matte_resized = matte;
    }

    cv::Mat matte_3;
    if (matte_resized.channels() == 1) {
        cv::cvtColor(matte_resized, matte_3, cv::COLOR_GRAY2BGR);
    } else {
        matte_3 = matte_resized;
    }

    cv::Mat bgr_f;
    bgr.convertTo(bgr_f, CV_32F, 1.0 / 255.0);
    cv::Mat result_f = bgr_f.mul(matte_3);
    cv::Mat result;
    result_f.convertTo(result, CV_8U, 255.0);
    return result;
}
