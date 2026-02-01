#pragma once

#include <opencv2/core.hpp>
#include <torch/torch.h>

struct CachedSampleData
{
    torch::Tensor target;
    torch::Tensor matte_mask;
    cv::Mat crop_bgr;
    bool valid = false;
};
