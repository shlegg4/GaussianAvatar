#pragma once

#include <vector>

#include <opencv2/core.hpp>

struct SmplResult;
struct SMPLLayer;

struct SmplifyLiteOptions {
    int num_iters = 50;
    float lr = 5e-4f;
    float keypoint_threshold = 1e-5f;
    float keypoint_weight_floor = 0.5f;
    float keypoint_weight_pow = 0.5f;
    float face_weight = 0.0f;
    float loss_tol = 1e-4f;
    int min_iters = 5;
    float pose_reg = 100.0f;
    float betas_reg = 5e-4f;
    int log_every = 5;
    int render_every = 5;
    bool optimize_translation = true;
    float target_height = 1.77f;
    float height_weight = 1000.0f; 
    bool lock_wrist_ankle = true;
};

bool SmplifyLite(SMPLLayer& smpl_layer,
                 const std::vector<cv::Point2f>& keypoints,
                 const std::vector<float>& keypoint_scores,
                 float crop_cx, float crop_cy, float crop_size,
                 float f_geo, float f_render, float img_w, float img_h,
                 SmplResult* io_res,
                 const SmplifyLiteOptions& options = {},
                 float* out_y_sign = nullptr,
                 const cv::Mat* render_frame = nullptr,
                 const std::string* render_dir = nullptr,
                 int frame_idx = 0);
