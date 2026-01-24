#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

struct ModNetMatteOptions {
    int input_size = 512;
    bool use_cuda = false;
    bool normalize_to_unit = true;
};

class ModNetMatte {
public:
    explicit ModNetMatte(const ModNetMatteOptions& options = {});

    bool Load(const std::string& model_path);

    // Output matte is CV_32F in range [0,1], same size as input bgr.
    bool ComputeMatte(const cv::Mat& bgr, cv::Mat* out_matte);

    cv::Mat ApplyMatte(const cv::Mat& bgr, const cv::Mat& matte) const;

private:
    cv::Mat Preprocess(const cv::Mat& bgr) const;

    ModNetMatteOptions options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int input_width_ = 0;
    int input_height_ = 0;
};
