#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "HmrInferenceUtils.h"
#include "SmplLBS.h"

// Photometric texture consistency optimization over a batch of frames.
// Returns true if optimization was performed.
bool OptimizeTextureConsistency(SMPLLayer& smpl_layer,
                                const std::vector<cv::Mat>& frames,
                                std::vector<SmplResult>* results,
                                const std::vector<float>& crop_info_flat,
                                float focal_length,
                                const HmrOutputOptions& opts);
