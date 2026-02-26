#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

struct DepthAnythingOptions {
    bool use_cuda = false;
    int input_size = 518;
    bool invert_depth = true;
};

class DepthAnything {
public:
    explicit DepthAnything(const DepthAnythingOptions& options = {});

    bool Load(const std::string& model_path);
    bool ComputeDepth(const cv::Mat& input_bgr, cv::Mat* out_depth);

private:
    DepthAnythingOptions options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int input_width_ = 0;
    int input_height_ = 0;
};
