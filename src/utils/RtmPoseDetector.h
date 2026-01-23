#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <onnxruntime_cxx_api.h>

struct RtmPoseDetectorOptions {
    int input_width = 192;
    int input_height = 256;
    float conf_threshold = 0.0001f;
    float simcc_split_ratio = 2.0f;
    bool use_cuda = false;
    bool save_debug_input = false;
    std::string debug_dir;
};

class RtmPoseDetector {
public:
    explicit RtmPoseDetector(const RtmPoseDetectorOptions& options = {});

    bool Load(const std::string& model_path);

    bool DetectPose(const cv::Mat& bgr,
                    std::vector<cv::Point2f>* out_keypoints,
                    std::vector<float>* out_keypoint_scores,
                    int frame_idx = -1);

private:
    cv::Mat Letterbox(const cv::Mat& src, float* out_scale, int* out_pad_x, int* out_pad_y) const;
    bool DecodeSimcc(const Ort::Value& simcc_x,
                     const Ort::Value& simcc_y,
                     float scale,
                     int pad_x,
                     int pad_y,
                     int img_w,
                     int img_h,
                     std::vector<cv::Point2f>* out_keypoints,
                     std::vector<float>* out_keypoint_scores) const;

    RtmPoseDetectorOptions options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};
