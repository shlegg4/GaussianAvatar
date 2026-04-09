#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace dataset_prep {

struct SmplxResult {
    std::vector<float> global_orient;
    std::vector<float> body_pose;
    std::vector<float> left_hand_pose;
    std::vector<float> right_hand_pose;
    std::vector<float> jaw_pose;
    std::vector<float> betas;
    std::vector<float> expression;
    std::vector<float> camera_translation;
};

class PearEstimator {
public:
    struct Options {
        std::string model_path;
        bool use_cuda = false;
    };

    explicit PearEstimator(const Options& options = {});
    ~PearEstimator();

    bool Initialize();
    bool IsReady() const;

    bool Estimate(const cv::Mat& crop_image, SmplxResult* out_result) const;

private:
    Options options_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dataset_prep
