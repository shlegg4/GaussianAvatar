#include "dataset_prep/processing/CliffEstimator.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include "utils/HmrInferenceConstants.h"
#include "utils/OnnxRuntimeCudaProvider.h"

namespace dataset_prep {
namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int FindNameIndex(const std::vector<std::string>& names, const std::string& pattern) {
    const std::string pattern_lower = ToLower(pattern);
    for (size_t index = 0; index < names.size(); ++index) {
        if (ToLower(names[index]).find(pattern_lower) != std::string::npos) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

std::vector<float> BuildCliffBBox(float cx,
                                  float cy,
                                  float box_size,
                                  float focal_length,
                                  int img_w,
                                  int img_h) {
    const float x = cx - static_cast<float>(img_w) * 0.5f;
    const float y = cy - static_cast<float>(img_h) * 0.5f;

    std::vector<float> bbox(3u, 0.0f);
    bbox[0] = (x / focal_length) * 2.8f;
    bbox[1] = (y / focal_length) * 2.8f;
    bbox[2] = (box_size - 0.24f * focal_length) / (0.06f * focal_length);
    return bbox;
}

std::vector<float> PreprocessImage(const cv::Mat& image) {
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

    cv::Mat resized;
    if (image.cols == kInputW && image.rows == kInputH) {
        resized = image.clone();
    } else {
        cv::resize(image, resized, cv::Size(kInputW, kInputH), 0.0, 0.0, cv::INTER_AREA);
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input_data;
    input_data.reserve(static_cast<size_t>(3 * kInputH * kInputW));
    for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < kInputH; ++row) {
            const float* row_ptr = channels[channel].ptr<float>(row);
            for (int col = 0; col < kInputW; ++col) {
                input_data.push_back((row_ptr[col] - kMean[channel]) / kStd[channel]);
            }
        }
    }

    return input_data;
}

bool CopyTensorToVector(const Ort::Value& tensor, std::vector<float>* out_values) {
    if (out_values == nullptr || !tensor.IsTensor()) {
        return false;
    }

    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t element_count = 1u;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return false;
        }
        element_count *= static_cast<size_t>(dim);
    }

    const float* data = tensor.GetTensorData<float>();
    if (data == nullptr) {
        return false;
    }

    out_values->assign(data, data + element_count);
    return true;
}

}  // namespace

class CliffEstimator::Impl {
public:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    int image_input_index = -1;
    int bbox_input_index = -1;
    int pose_output_index = -1;
    int betas_output_index = -1;
    int cam_output_index = -1;
};

CliffEstimator::CliffEstimator(const Options& options)
    : options_(options) {}

CliffEstimator::~CliffEstimator() = default;

bool CliffEstimator::Initialize() {
    impl_ = std::make_unique<Impl>();

    try {
        impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dataset_prep_cliff");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (options_.use_cuda) {
            onnxruntime_utils::TryAppendCudaExecutionProvider(session_options);
        }

        const std::wstring model_path(options_.model_path.begin(), options_.model_path.end());
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, model_path.c_str(), session_options);
        impl_->memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = impl_->session->GetInputCount();
        impl_->input_names.clear();
        for (size_t index = 0; index < input_count; ++index) {
            auto name = impl_->session->GetInputNameAllocated(index, allocator);
            impl_->input_names.emplace_back(name.get());
        }

        const size_t output_count = impl_->session->GetOutputCount();
        impl_->output_names.clear();
        for (size_t index = 0; index < output_count; ++index) {
            auto name = impl_->session->GetOutputNameAllocated(index, allocator);
            impl_->output_names.emplace_back(name.get());
        }

        impl_->image_input_index = FindNameIndex(impl_->input_names, "image");
        impl_->bbox_input_index = FindNameIndex(impl_->input_names, "bbox");
        if (impl_->image_input_index < 0 && !impl_->input_names.empty()) {
            impl_->image_input_index = 0;
        }
        if (impl_->bbox_input_index < 0 && impl_->input_names.size() > 1u) {
            impl_->bbox_input_index = 1;
        }

        impl_->pose_output_index = FindNameIndex(impl_->output_names, "pose");
        impl_->betas_output_index = FindNameIndex(impl_->output_names, "betas");
        impl_->cam_output_index = FindNameIndex(impl_->output_names, "cam");
        if (impl_->pose_output_index < 0 && !impl_->output_names.empty()) {
            impl_->pose_output_index = 0;
        }
        if (impl_->betas_output_index < 0 && impl_->output_names.size() > 1u) {
            impl_->betas_output_index = 1;
        }
        if (impl_->cam_output_index < 0 && impl_->output_names.size() > 2u) {
            impl_->cam_output_index = 2;
        }
    } catch (const Ort::Exception&) {
        impl_.reset();
        return false;
    }

    return IsReady();
}

bool CliffEstimator::IsReady() const {
    return impl_ != nullptr &&
           impl_->session != nullptr &&
           impl_->image_input_index >= 0 &&
           impl_->bbox_input_index >= 0 &&
           impl_->pose_output_index >= 0 &&
           impl_->betas_output_index >= 0 &&
           impl_->cam_output_index >= 0;
}

bool CliffEstimator::Estimate(const cv::Mat& image,
                              float crop_cx,
                              float crop_cy,
                              float crop_size,
                              float focal_length,
                              int img_w,
                              int img_h,
                              SmplResult* out_result) const {
    if (!IsReady() || out_result == nullptr || image.empty() ||
        !(crop_size > 0.0f) || !(focal_length > 0.0f) ||
        img_w <= 0 || img_h <= 0) {
        return false;
    }

    std::vector<float> image_input = PreprocessImage(image);
    std::vector<float> bbox_input = BuildCliffBBox(
        crop_cx, crop_cy, crop_size, focal_length, img_w, img_h);

    std::vector<int64_t> image_shape = {1, 3, kInputH, kInputW};
    std::vector<int64_t> bbox_shape = {1, 3};

    std::vector<const char*> input_names;
    input_names.reserve(impl_->input_names.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(impl_->input_names.size());
    for (size_t index = 0; index < impl_->input_names.size(); ++index) {
        input_names.push_back(impl_->input_names[index].c_str());
        if (static_cast<int>(index) == impl_->image_input_index) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                impl_->memory_info,
                image_input.data(),
                image_input.size(),
                image_shape.data(),
                image_shape.size()));
        } else if (static_cast<int>(index) == impl_->bbox_input_index) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                impl_->memory_info,
                bbox_input.data(),
                bbox_input.size(),
                bbox_shape.data(),
                bbox_shape.size()));
        } else {
            return false;
        }
    }

    std::vector<const char*> output_names;
    output_names.reserve(impl_->output_names.size());
    for (const auto& name : impl_->output_names) {
        output_names.push_back(name.c_str());
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                      input_names.data(),
                                      input_tensors.data(),
                                      input_tensors.size(),
                                      output_names.data(),
                                      output_names.size());
    } catch (const Ort::Exception&) {
        return false;
    }

    if (impl_->pose_output_index >= static_cast<int>(outputs.size()) ||
        impl_->betas_output_index >= static_cast<int>(outputs.size()) ||
        impl_->cam_output_index >= static_cast<int>(outputs.size())) {
        return false;
    }

    SmplResult result;
    if (!CopyTensorToVector(outputs[static_cast<size_t>(impl_->pose_output_index)], &result.pose) ||
        !CopyTensorToVector(outputs[static_cast<size_t>(impl_->betas_output_index)], &result.shape) ||
        !CopyTensorToVector(outputs[static_cast<size_t>(impl_->cam_output_index)], &result.camera)) {
        return false;
    }

    *out_result = std::move(result);
    return true;
}

}  // namespace dataset_prep
