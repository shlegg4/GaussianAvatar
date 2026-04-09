#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "utils/HmrInferenceUtils.h"

namespace dataset_prep {

class CliffEstimator {
public:
    struct Options {
        std::string model_path;
        bool use_cuda = false;
    };

    explicit CliffEstimator(const Options& options = {});
    ~CliffEstimator();

    bool Initialize();
    bool IsReady() const;

    bool Estimate(const cv::Mat& image,
                  float crop_cx,
                  float crop_cy,
                  float crop_size,
                  float focal_length,
                  int img_w,
                  int img_h,
                  SmplResult* out_result) const;

private:
    Options options_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dataset_prep
