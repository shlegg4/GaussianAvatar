#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

void DrawVerticesOverlayPinhole(cv::Mat& frame, const torch::Tensor& verts, const std::vector<float>& cam,
                                float crop_cx, float crop_cy, float crop_size,
                                float f_geo, float f_render,
                                float y_sign_override = 0.0f);
void DrawVerticesOverlay(cv::Mat& frame, const torch::Tensor& verts, const std::vector<float>& cam);
void DrawOverlayDebug(cv::Mat& frame, const std::vector<float>& cam, const std::vector<float>& bbox);
void DrawPoseKeypoints(cv::Mat& frame, const std::vector<cv::Point2f>& keypoints,
                       const std::vector<float>& keypoint_scores, float min_score);
void WriteObjVertices(const torch::Tensor& verts, const std::string& path);
