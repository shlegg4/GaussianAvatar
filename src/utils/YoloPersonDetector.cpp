#include "YoloPersonDetector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "OnnxRuntimeCudaProvider.h"

YoloPersonDetector::YoloPersonDetector(const YoloPersonDetectorOptions& options)
    : options_(options) {}

bool YoloPersonDetector::Load(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yolo");
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

    return !input_names_.empty() && !output_names_.empty();
}

bool YoloPersonDetector::DetectPerson(const cv::Mat& bgr, cv::Rect2f* out_bbox, float* out_score,
                                      std::vector<cv::Point2f>* out_keypoints,
                                      std::vector<float>* out_keypoint_scores) {
    if (bgr.empty() || !session_) {
        return false;
    }

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    cv::Mat padded = Letterbox(bgr, &scale, &pad_x, &pad_y);

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0,
                           cv::Size(options_.input_width, options_.input_height),
                           cv::Scalar(), true, false);

    std::vector<int64_t> input_shape = {1, 3, options_.input_height, options_.input_width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

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
    if (outputs.empty()) {
        return false;
    }

    const auto& output = outputs[0];
    if (!output.IsTensor()) {
        return false;
    }
    const float* output_data = output.GetTensorData<float>();
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();

    last_detections_.clear();
    ParseDetections(output_data, shape, scale, pad_x, pad_y, bgr.cols, bgr.rows, &last_detections_);
    if (last_detections_.empty()) {
        return false;
    }

    const auto best_it = std::max_element(
        last_detections_.begin(), last_detections_.end(),
        [](const YoloDetection& a, const YoloDetection& b) { return a.score < b.score; });
    if (best_it == last_detections_.end()) {
        return false;
    }
    if (out_bbox) {
        *out_bbox = best_it->bbox;
    }
    if (out_score) {
        *out_score = best_it->score;
    }
    if (out_keypoints) {
        *out_keypoints = best_it->keypoints;
    }
    if (out_keypoint_scores) {
        *out_keypoint_scores = best_it->keypoint_scores;
    }
    return true;
}

cv::Mat YoloPersonDetector::Letterbox(const cv::Mat& src, float* out_scale, int* out_pad_x, int* out_pad_y) const {
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
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    if (out_scale) *out_scale = scale;
    if (out_pad_x) *out_pad_x = left;
    if (out_pad_y) *out_pad_y = top;
    return padded;
}

void YoloPersonDetector::ParseDetections(const float* data, const std::vector<int64_t>& shape,
                                         float scale, int pad_x, int pad_y,
                                         int img_w, int img_h, std::vector<YoloDetection>* out) const {
    if (!data || !out) {
        return;
    }
    if (shape.size() < 2 || shape.size() > 3) {
        return;
    }

    int64_t rows = 0;
    int64_t cols = 0;
    bool transposed = false;
    if (shape.size() == 2) {
        rows = shape[0];
        cols = shape[1];
    } else {
        const int64_t dim1 = shape[1];
        const int64_t dim2 = shape[2];
        if (dim1 < dim2) {
            rows = dim2;
            cols = dim1;
            transposed = true;
        } else {
            rows = dim1;
            cols = dim2;
        }
    }

    if (rows <= 0 || cols <= 0) {
        return;
    }

    const int cols_int = static_cast<int>(cols);
    int class_offset = -1;
    auto has_pose_layout_for_offset = [&](int offset, int classes) -> bool {
        const int rem = cols_int - offset - classes;
        return rem >= 0 && (rem % 3) == 0;
    };
    auto has_det_layout_for_offset = [&](int offset, int classes) -> bool {
        return (cols_int - offset) == classes;
    };

    if (cols_int == 4 + options_.num_classes) {
        class_offset = 4;
    } else if (cols_int == 5 + options_.num_classes) {
        class_offset = 5;
    } else {
        const bool offset5_ok = has_pose_layout_for_offset(5, options_.num_classes) ||
                                has_pose_layout_for_offset(5, 1) ||
                                has_det_layout_for_offset(5, options_.num_classes);
        const bool offset4_ok = has_pose_layout_for_offset(4, options_.num_classes) ||
                                has_pose_layout_for_offset(4, 1) ||
                                has_det_layout_for_offset(4, options_.num_classes);
        if (offset5_ok) {
            class_offset = 5;
        } else if (offset4_ok) {
            class_offset = 4;
        } else if (cols_int > 5) {
            class_offset = 5;
        } else if (cols_int > 4) {
            class_offset = 4;
        } else {
            return;
        }
    }
    const bool has_objectness = (class_offset == 5);
    int remaining = cols_int - class_offset;
    if (remaining <= 0) {
        return;
    }

    int num_classes = options_.num_classes;
    int num_keypoints = 0;
    auto has_pose_layout = [&](int classes) -> bool {
        const int rem = remaining - classes;
        return rem >= 0 && (rem % 3) == 0;
    };
    if (has_pose_layout(num_classes)) {
        num_keypoints = (remaining - num_classes) / 3;
    } else if (has_pose_layout(1)) {
        num_classes = 1;
        num_keypoints = (remaining - num_classes) / 3;
    } else {
        num_classes = remaining;
        num_keypoints = 0;
    }

    if (num_classes <= 0) {
        return;
    }

    auto value_at = [&](int64_t r, int64_t c) -> float {
        if (transposed) {
            return data[c * rows + r];
        }
        return data[r * cols + c];
    };

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<cv::Point2f>> all_keypoints;
    std::vector<std::vector<float>> all_keypoint_scores;

    const int keypoint_offset = class_offset + num_classes;
    for (int64_t i = 0; i < rows; ++i) {
        const float x = value_at(i, 0);
        const float y = value_at(i, 1);
        const float w = value_at(i, 2);
        const float h = value_at(i, 3);
        const float obj = has_objectness ? value_at(i, 4) : 1.0f;

        int best_class = -1;
        float best_class_score = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            const float score = value_at(i, class_offset + c);
            if (score > best_class_score) {
                best_class_score = score;
                best_class = c;
            }
        }

        if (best_class != 0) {
            continue;
        }

        const float score = obj * best_class_score;
        if (score < options_.conf_threshold) {
            continue;
        }

        const float x0 = (x - 0.5f * w - pad_x) / scale;
        const float y0 = (y - 0.5f * h - pad_y) / scale;
        const float x1 = (x + 0.5f * w - pad_x) / scale;
        const float y1 = (y + 0.5f * h - pad_y) / scale;

        const float clamped_x0 = std::max(0.0f, std::min(x0, static_cast<float>(img_w - 1)));
        const float clamped_y0 = std::max(0.0f, std::min(y0, static_cast<float>(img_h - 1)));
        const float clamped_x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w - 1)));
        const float clamped_y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h - 1)));
        const float bw = std::max(0.0f, clamped_x1 - clamped_x0);
        const float bh = std::max(0.0f, clamped_y1 - clamped_y0);

        boxes.emplace_back(clamped_x0, clamped_y0, bw, bh);
        scores.push_back(score);
        class_ids.push_back(best_class);

        if (num_keypoints > 0) {
            std::vector<cv::Point2f> kpts;
            std::vector<float> kpt_scores;
            kpts.reserve(static_cast<size_t>(num_keypoints));
            kpt_scores.reserve(static_cast<size_t>(num_keypoints));
            for (int k = 0; k < num_keypoints; ++k) {
                const float kx = value_at(i, keypoint_offset + k * 3 + 0);
                const float ky = value_at(i, keypoint_offset + k * 3 + 1);
                const float ks = value_at(i, keypoint_offset + k * 3 + 2);
                const float x_unpad = (kx - pad_x) / scale;
                const float y_unpad = (ky - pad_y) / scale;
                const float x_clamped = std::max(0.0f, std::min(x_unpad, static_cast<float>(img_w - 1)));
                const float y_clamped = std::max(0.0f, std::min(y_unpad, static_cast<float>(img_h - 1)));
                kpts.emplace_back(x_clamped, y_clamped);
                kpt_scores.push_back(ks);
            }
            all_keypoints.push_back(std::move(kpts));
            all_keypoint_scores.push_back(std::move(kpt_scores));
        } else {
            all_keypoints.emplace_back();
            all_keypoint_scores.emplace_back();
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, options_.conf_threshold, options_.nms_threshold, indices);

    out->clear();
    out->reserve(indices.size());
    for (int idx : indices) {
        YoloDetection det;
        const auto& box = boxes[idx];
        det.bbox = cv::Rect2f(static_cast<float>(box.x),
                              static_cast<float>(box.y),
                              static_cast<float>(box.width),
                              static_cast<float>(box.height));
        det.score = scores[idx];
        det.class_id = class_ids[idx];
        if (num_keypoints > 0 && idx < static_cast<int>(all_keypoints.size())) {
            det.keypoints = std::move(all_keypoints[idx]);
            det.keypoint_scores = std::move(all_keypoint_scores[idx]);
        }
        out->push_back(det);
    }
}
