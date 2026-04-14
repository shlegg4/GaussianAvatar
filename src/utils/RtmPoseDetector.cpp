#include "RtmPoseDetector.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "OnnxRuntimeCudaProvider.h"

namespace {

const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

float DecodeSimccArgMaxConfidence(const float* data, int size, int* out_argmax) {
    if (!data || size <= 0) {
        if (out_argmax) *out_argmax = 0;
        return 0.0f;
    }

    int argmax = 0;
    float max_val = data[0];
    float second_val = -std::numeric_limits<float>::infinity();
    float min_val = data[0];
    float sum_val = data[0];
    for (int i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            second_val = max_val;
            max_val = data[i];
            argmax = i;
        } else if (data[i] > second_val) {
            second_val = data[i];
        }
        min_val = std::min(min_val, data[i]);
        sum_val += data[i];
    }
    if (!std::isfinite(second_val)) {
        second_val = max_val;
    }

    if (out_argmax) *out_argmax = argmax;

    float confidence = 0.0f;
    const char* decode_mode = "logits_softmax";

    // Some RTMPose exports already output probabilities for each SimCC axis.
    // In that case, applying softmax again collapses confidence to about 1/N.
    if (min_val >= -1e-6f && max_val <= 1.0f + 1e-3f &&
        sum_val >= 0.5f && sum_val <= 1.5f) {
        decode_mode = "probabilities";
        confidence = std::clamp(data[argmax], 0.0f, 1.0f);
    } else {
        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += std::exp(data[i] - max_val);
        }
        if (sum_exp <= 0.0f) {
            confidence = 0.0f;
        } else {
            const float logsumexp = max_val + std::log(sum_exp);

            // If values are log-probabilities (logsumexp ~= 0), exp(max_logp)
            // is the canonical confidence.
            if (max_val <= 1e-4f && std::abs(logsumexp) <= 0.2f) {
                decode_mode = "log_probabilities";
                confidence = std::exp(max_val);
            } else {
                // SimCC heads are often decoded from argmax directly, and the raw
                // maxima are better confidence proxies than softmax over long axes.
                decode_mode = "logits_peak_sigmoid";
                confidence = 1.0f / (1.0f + std::exp(-max_val));
            }
        }
    }

    static bool printed_decode_stats = false;
    if (!printed_decode_stats) {
        printed_decode_stats = true;
        std::cout << "RTMPose SimCC decode mode=" << decode_mode
                  << " size=" << size
                  << " min=" << min_val
                  << " max=" << max_val
                  << " second=" << second_val
                  << " sum=" << sum_val
                  << " conf=" << confidence
                  << std::endl;
    }

    return std::clamp(confidence, 0.0f, 1.0f);
}

} // namespace

RtmPoseDetector::RtmPoseDetector(const RtmPoseDetectorOptions& options)
    : options_(options) {}

bool RtmPoseDetector::Load(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rtmpose");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (options_.use_cuda) {
            onnxruntime_utils::TryAppendCudaExecutionProvider(session_options);
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
    } catch (const Ort::Exception&) {
        env_.reset();
        session_.reset();
        input_names_.clear();
        output_names_.clear();
        return false;
    }

    return !input_names_.empty() && output_names_.size() >= 2;
}

bool RtmPoseDetector::DetectPose(const cv::Mat& bgr,
                                 std::vector<cv::Point2f>* out_keypoints,
                                 std::vector<float>* out_keypoint_scores,
                                 int frame_idx) {
    if (bgr.empty() || !session_) {
        return false;
    }

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    cv::Mat padded = Letterbox(bgr, &scale, &pad_x, &pad_y);
    if (options_.save_debug_input && !options_.debug_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(options_.debug_dir, ec);
        if (!ec) {
            std::ostringstream name;
            const int safe_idx = frame_idx >= 0 ? frame_idx : 0;
            name << "rtmpose_input_" << std::setw(6) << std::setfill('0') << safe_idx << ".png";
            const auto out_path = std::filesystem::path(options_.debug_dir) / name.str();
            cv::imwrite(out_path.string(), padded);
        }
    }

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input_data;
    input_data.reserve(1 * 3 * options_.input_height * options_.input_width);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < options_.input_height; ++y) {
            const float* row_ptr = channels[c].ptr<float>(y);
            for (int x = 0; x < options_.input_width; ++x) {
                const float val = row_ptr[x];
                input_data.push_back((val - kMean[c]) / kStd[c]);
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, options_.input_height, options_.input_width};
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
    if (outputs.size() < 2) {
        return false;
    }

    const Ort::Value* simcc_x = nullptr;
    const Ort::Value* simcc_y = nullptr;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const std::string& name = output_names_[i];
        if (name.find("simcc_x") != std::string::npos) {
            simcc_x = &outputs[i];
        } else if (name.find("simcc_y") != std::string::npos) {
            simcc_y = &outputs[i];
        }
    }
    if (!simcc_x || !simcc_y) {
        simcc_x = &outputs[0];
        simcc_y = &outputs[1];
    }

    const bool success = DecodeSimcc(*simcc_x, *simcc_y,
                       scale, pad_x, pad_y,
                       bgr.cols, bgr.rows,
                       out_keypoints, out_keypoint_scores);
    return success;
}

cv::Mat RtmPoseDetector::Letterbox(const cv::Mat& src, float* out_scale, int* out_pad_x, int* out_pad_y) const {
    const int src_w = src.cols;
    const int src_h = src.rows;
    const float scale = std::min(static_cast<float>(options_.input_width) / src_w,
                                 static_cast<float>(options_.input_height) / src_h);
    const int new_w = static_cast<int>(std::round(src_w * scale));
    const int new_h = static_cast<int>(std::round(src_h * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    const int pad_w = options_.input_width - new_w;
    const int pad_h = options_.input_height - new_h;
    const int left = pad_w / 2;
    const int top = pad_h / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, pad_h - top, left, pad_w - left,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    if (out_scale) *out_scale = scale;
    if (out_pad_x) *out_pad_x = left;
    if (out_pad_y) *out_pad_y = top;
    return padded;
}

bool RtmPoseDetector::DecodeSimcc(const Ort::Value& simcc_x,
                                  const Ort::Value& simcc_y,
                                  float scale,
                                  int pad_x,
                                  int pad_y,
                                  int img_w,
                                  int img_h,
                                  std::vector<cv::Point2f>* out_keypoints,
                                  std::vector<float>* out_keypoint_scores) const {
    if (!simcc_x.IsTensor() || !simcc_y.IsTensor()) {
        std::cerr << "RTMPose outputs are not tensors." << std::endl;
        return false;
    }

    const auto shape_x = simcc_x.GetTensorTypeAndShapeInfo().GetShape();
    const auto shape_y = simcc_y.GetTensorTypeAndShapeInfo().GetShape();
    if (shape_x.size() != 3 || shape_y.size() != 3) {
        std::cerr << "RTMPose SimCC outputs have invalid shape." << std::endl;
        return false;
    }
    if (shape_x[0] != 1 || shape_y[0] != 1 || shape_x[1] != shape_y[1]) {
        std::cerr << "RTMPose SimCC outputs have incompatible shape." << std::endl;
        return false;
    }

    const int num_joints = static_cast<int>(shape_x[1]);
    const int simcc_w = static_cast<int>(shape_x[2]);
    const int simcc_h = static_cast<int>(shape_y[2]);

    const float* data_x = simcc_x.GetTensorData<float>();
    const float* data_y = simcc_y.GetTensorData<float>();
    if (!data_x || !data_y) {
        std::cerr << "RTMPose SimCC outputs have no data." << std::endl;
        return false;
    }

    if (out_keypoints) out_keypoints->clear();
    if (out_keypoint_scores) out_keypoint_scores->clear();
    if (out_keypoints) out_keypoints->reserve(num_joints);
    if (out_keypoint_scores) out_keypoint_scores->reserve(num_joints);

    for (int j = 0; j < num_joints; ++j) {
        const float* row_x = data_x + j * simcc_w;
        const float* row_y = data_y + j * simcc_h;
        int argmax_x = 0;
        int argmax_y = 0;
        const float prob_x = DecodeSimccArgMaxConfidence(row_x, simcc_w, &argmax_x);
        const float prob_y = DecodeSimccArgMaxConfidence(row_y, simcc_h, &argmax_y);

        float x = static_cast<float>(argmax_x) / options_.simcc_split_ratio;
        float y = static_cast<float>(argmax_y) / options_.simcc_split_ratio;
        x = (x - static_cast<float>(pad_x)) / scale;
        y = (y - static_cast<float>(pad_y)) / scale;
        x = std::max(0.0f, std::min(x, static_cast<float>(img_w - 1)));
        y = std::max(0.0f, std::min(y, static_cast<float>(img_h - 1)));
        const float score = std::sqrt(prob_x * prob_y);

        if (out_keypoints) out_keypoints->emplace_back(x, y);
        if (out_keypoint_scores) out_keypoint_scores->push_back(score);
    }

    if (out_keypoint_scores && options_.conf_threshold > 0.0f) {
        bool any_valid = false;
        for (float score : *out_keypoint_scores) {
            if (score >= options_.conf_threshold) {
                any_valid = true;
                break;
            }
        }
        if (!any_valid) {
            std::cerr << "RTMPose detection confidence scores are all below threshold." << std::endl;
            return false;
        }
    }

    return true;
}
