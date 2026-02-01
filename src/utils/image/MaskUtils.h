#pragma once

#include <opencv2/core.hpp>
#include <torch/torch.h>

bool IsMaskCoverageValid(const torch::Tensor &render, const cv::Mat &matte, float max_outside_ratio);
bool IsMaskCoverageValidTensor(const torch::Tensor &render, const torch::Tensor &matte_mask,
                               float max_outside_ratio, float render_threshold);
