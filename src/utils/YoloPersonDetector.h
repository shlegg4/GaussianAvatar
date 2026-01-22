#pragma once

#include <string>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include <onnxruntime_cxx_api.h>

struct YoloPersonDetectorOptions {
    int input_width = 640;
    int input_height = 640;
    int num_classes = 80;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool use_cuda = false;
};

struct YoloDetection {
    cv::Rect2f bbox;
    float score = 0.0f;
    int class_id = -1;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> keypoint_scores;
};

class YoloPersonDetector {
public:
    explicit YoloPersonDetector(const YoloPersonDetectorOptions& options = {});

    bool Load(const std::string& model_path);

    // Returns the best person bbox if found (COCO class 0). False if none detected.
    bool DetectPerson(const cv::Mat& bgr, cv::Rect2f* out_bbox, float* out_score = nullptr,
                      std::vector<cv::Point2f>* out_keypoints = nullptr,
                      std::vector<float>* out_keypoint_scores = nullptr);

    const std::vector<YoloDetection>& last_detections() const { return last_detections_; }

private:
    cv::Mat Letterbox(const cv::Mat& src, float* out_scale, int* out_pad_x, int* out_pad_y) const;
    void ParseDetections(const float* data, const std::vector<int64_t>& shape,
                         float scale, int pad_x, int pad_y,
                         int img_w, int img_h, std::vector<YoloDetection>* out) const;

    YoloPersonDetectorOptions options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<YoloDetection> last_detections_;
};
