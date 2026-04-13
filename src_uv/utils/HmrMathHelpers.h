#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

struct SmplResult;

cv::Mat Rot6dToRotMatSingle(const float* feat);
std::vector<float> ConvertPose6dToAxisAngle(const std::vector<float>& pose_6d);
torch::Tensor PoseToAxisAngle(const SmplResult& res);

int CountProjectedInFrame(const torch::Tensor& verts_cpu, float s, float tx, float ty, float y_sign,
                          int width, int height);
int CountProjectedInFramePinhole(const torch::Tensor& verts_cpu, const cv::Vec3f& t, float y_sign,
                                 float focal_length, float cx, float cy,
                                 int width, int height);
cv::Vec3f EstimateTranslation(const std::vector<float>& cam, float crop_cx, float crop_cy, float crop_size,
                              float focal_length, float img_w, float img_h);
