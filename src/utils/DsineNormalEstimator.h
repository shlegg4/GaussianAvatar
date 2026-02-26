#pragma once

#include <memory>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

struct DsineOptions {
    bool use_cuda = false;
    int input_size = 512;
};

class DsineNormalEstimator {
public:
    explicit DsineNormalEstimator(const DsineOptions& options = {});

    bool Load(const std::string& model_path);
    bool ComputeNormals(const cv::Mat& input_bgr, cv::Mat* out_normals);
    bool ComputeNormals(const cv::Mat& input_bgr,
                        float focal_px,
                        float principal_x,
                        float principal_y,
                        cv::Mat* out_normals);

private:
    bool BuildIntrinsicsTensor(float focal_px,
                               float principal_x,
                               float principal_y,
                               Ort::Value* out_tensor) const;

    DsineOptions options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> intrinsics_input_shape_;
    mutable std::vector<float> intrinsics_buffer_;
    int input_width_ = 0;
    int input_height_ = 0;
    size_t image_input_index_ = 0;
    size_t intrinsics_input_index_ = std::numeric_limits<size_t>::max();
};
