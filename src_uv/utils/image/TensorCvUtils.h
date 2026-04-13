#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <torch/torch.h>

cv::Mat TensorToBgr(const torch::Tensor &image);
torch::Tensor LoadImageTensor(const cv::Mat &bgr, torch::Device device);
torch::Tensor LoadMatteMaskTensor(const cv::Mat &matte, int target_w, int target_h, torch::Device device);
bool SaveImageTensorPng(const std::string &path, const torch::Tensor &image);
