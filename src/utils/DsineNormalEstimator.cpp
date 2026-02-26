#include "DsineNormalEstimator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

inline uint8_t FloatToU8(float value) {
    const float clamped = std::clamp(value, 0.0f, 255.0f);
    return static_cast<uint8_t>(std::lround(clamped));
}

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

} // namespace

DsineNormalEstimator::DsineNormalEstimator(const DsineOptions& options)
    : options_(options) {}

bool DsineNormalEstimator::Load(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dsine");
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

        image_input_index_ = 0;
        intrinsics_input_index_ = std::numeric_limits<size_t>::max();
        intrinsics_input_shape_.clear();
        intrinsics_buffer_.clear();

        for (size_t i = 0; i < input_names_.size(); ++i) {
            const std::string lower = ToLower(input_names_[i]);
            if (lower.find("intrinsics") != std::string::npos ||
                lower.find("intrinsic") != std::string::npos) {
                intrinsics_input_index_ = i;
                break;
            }
        }

        for (size_t i = 0; i < input_count; ++i) {
            if (i == intrinsics_input_index_) {
                continue;
            }
            const auto shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() >= 4) {
                const int64_t c_nchw = shape[1];
                const int64_t c_nhwc = shape[3];
                if (c_nchw == 3 || c_nhwc == 3 || c_nchw == 1 || c_nhwc == 1) {
                    image_input_index_ = i;
                    break;
                }
            }
        }

        if (intrinsics_input_index_ == std::numeric_limits<size_t>::max() && input_count > 1) {
            intrinsics_input_index_ = (image_input_index_ == 0) ? 1 : 0;
        }
        if (intrinsics_input_index_ < input_count) {
            intrinsics_input_shape_ = session_->GetInputTypeInfo(intrinsics_input_index_)
                                          .GetTensorTypeAndShapeInfo().GetShape();
        }

        input_width_ = options_.input_size;
        input_height_ = options_.input_size;
        if (image_input_index_ < input_count) {
            const auto shape =
                session_->GetInputTypeInfo(image_input_index_).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() >= 4) {
                int64_t h = 0;
                int64_t w = 0;
                if (shape[1] == 3) {
                    // [N, C, H, W]
                    h = shape[2];
                    w = shape[3];
                } else if (shape[3] == 3) {
                    // [N, H, W, C]
                    h = shape[1];
                    w = shape[2];
                }
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

bool DsineNormalEstimator::ComputeNormals(const cv::Mat& input_bgr, cv::Mat* out_normals) {
    const float fallback_focal = static_cast<float>(std::max(input_bgr.cols, input_bgr.rows));
    const float cx = static_cast<float>(input_bgr.cols) * 0.5f;
    const float cy = static_cast<float>(input_bgr.rows) * 0.5f;
    return ComputeNormals(input_bgr, fallback_focal, cx, cy, out_normals);
}

bool DsineNormalEstimator::BuildIntrinsicsTensor(float focal_px,
                                                 float principal_x,
                                                 float principal_y,
                                                 Ort::Value* out_tensor) const {
    if (!out_tensor) {
        return false;
    }
    if (intrinsics_input_index_ == std::numeric_limits<size_t>::max()) {
        return false;
    }

    std::vector<int64_t> tensor_shape = intrinsics_input_shape_;
    if (tensor_shape.empty()) {
        return false;
    }
    for (auto& d : tensor_shape) {
        if (d <= 0) {
            d = 1;
        }
    }

    size_t elem_count = 1;
    for (int64_t d : tensor_shape) {
        elem_count *= static_cast<size_t>(d);
    }
    if (elem_count == 0) {
        return false;
    }

    intrinsics_buffer_.assign(elem_count, 0.0f);
    const bool matrix33 =
        tensor_shape.size() >= 2 &&
        tensor_shape[tensor_shape.size() - 1] == 3 &&
        tensor_shape[tensor_shape.size() - 2] == 3 &&
        elem_count >= 9;

    if (matrix33) {
        intrinsics_buffer_[0] = focal_px;
        intrinsics_buffer_[2] = principal_x;
        intrinsics_buffer_[4] = focal_px;
        intrinsics_buffer_[5] = principal_y;
        intrinsics_buffer_[8] = 1.0f;
    } else if (elem_count >= 4) {
        intrinsics_buffer_[0] = focal_px;
        intrinsics_buffer_[1] = focal_px;
        intrinsics_buffer_[2] = principal_x;
        intrinsics_buffer_[3] = principal_y;
    } else if (elem_count >= 3) {
        intrinsics_buffer_[0] = focal_px;
        intrinsics_buffer_[1] = principal_x;
        intrinsics_buffer_[2] = principal_y;
    } else {
        return false;
    }

    *out_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        intrinsics_buffer_.data(),
        intrinsics_buffer_.size(),
        tensor_shape.data(),
        tensor_shape.size());
    return true;
}

bool DsineNormalEstimator::ComputeNormals(const cv::Mat& input_bgr,
                                          float focal_px,
                                          float principal_x,
                                          float principal_y,
                                          cv::Mat* out_normals) {
    if (!out_normals || !session_ || input_bgr.empty()) {
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

    std::vector<int64_t> image_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(), image_shape.data(), image_shape.size());
    Ort::Value intrinsics_tensor{nullptr};
    if (intrinsics_input_index_ != std::numeric_limits<size_t>::max()) {
        const float safe_focal = std::max(1.0f, focal_px);
        if (!BuildIntrinsicsTensor(safe_focal, principal_x, principal_y, &intrinsics_tensor)) {
            return false;
        }
    }

    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_tensors;
    input_names.reserve(input_names_.size());
    input_tensors.reserve(input_names_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
        input_names.push_back(input_names_[i].c_str());
        if (i == image_input_index_) {
            input_tensors.emplace_back(std::move(input_tensor));
        } else if (i == intrinsics_input_index_) {
            input_tensors.emplace_back(std::move(intrinsics_tensor));
        } else {
            return false;
        }
    }
    std::vector<const char*> output_names;
    output_names.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        output_names.push_back(name.c_str());
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), input_tensors.data(), input_tensors.size(),
                                 output_names.data(), output_names.size());
    if (outputs.empty() || !outputs[0].IsTensor()) {
        return false;
    }

    const auto& normal_tensor = outputs[0];
    const auto shape = normal_tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || shape[0] != 1) {
        return false;
    }

    int out_h = 0;
    int out_w = 0;
    bool nchw = false;
    if (shape[1] == 3) {
        // [1, 3, H, W]
        nchw = true;
        out_h = static_cast<int>(shape[2]);
        out_w = static_cast<int>(shape[3]);
    } else if (shape[3] == 3) {
        // [1, H, W, 3]
        nchw = false;
        out_h = static_cast<int>(shape[1]);
        out_w = static_cast<int>(shape[2]);
    } else {
        return false;
    }
    if (out_h <= 0 || out_w <= 0) {
        return false;
    }

    const float* data = normal_tensor.GetTensorData<float>();
    if (!data) {
        return false;
    }

    cv::Mat normal_map(out_h, out_w, CV_8UC3);
    if (nchw) {
        const size_t hw = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        const float* x_ptr = data;
        const float* y_ptr = data + hw;
        const float* z_ptr = data + hw * 2;
        for (int y = 0; y < out_h; ++y) {
            cv::Vec3b* row = normal_map.ptr<cv::Vec3b>(y);
            for (int x = 0; x < out_w; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(out_w) + static_cast<size_t>(x);
                const float x01 = (x_ptr[idx] + 1.0f) * 0.5f * 255.0f;
                const float y01 = (y_ptr[idx] + 1.0f) * 0.5f * 255.0f;
                const float z01 = (z_ptr[idx] + 1.0f) * 0.5f * 255.0f;
                // OpenCV uses BGR. Standard mapping: X->R, Y->G, Z->B.
                row[x] = cv::Vec3b(FloatToU8(z01), FloatToU8(y01), FloatToU8(x01));
            }
        }
    } else {
        for (int y = 0; y < out_h; ++y) {
            cv::Vec3b* row = normal_map.ptr<cv::Vec3b>(y);
            for (int x = 0; x < out_w; ++x) {
                const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(out_w) + static_cast<size_t>(x)) * 3;
                const float x01 = (data[idx + 0] + 1.0f) * 0.5f * 255.0f;
                const float y01 = (data[idx + 1] + 1.0f) * 0.5f * 255.0f;
                const float z01 = (data[idx + 2] + 1.0f) * 0.5f * 255.0f;
                row[x] = cv::Vec3b(FloatToU8(z01), FloatToU8(y01), FloatToU8(x01));
            }
        }
    }

    cv::resize(normal_map, *out_normals, input_bgr.size(), 0, 0, cv::INTER_CUBIC);
    return true;
}
