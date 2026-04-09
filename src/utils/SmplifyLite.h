#pragma once

#include <memory>
#include <string>
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
    float reproj_robust_sigma = 50.0f;
};

struct SmplifyMultiViewObservation {
    std::vector<cv::Point2f> keypoints;
    std::vector<float> keypoint_scores;
    cv::Matx33f K;
    cv::Matx33f R;
    cv::Vec3f t;
    float img_w = 0.0f;
    float img_h = 0.0f;
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

bool SmplifyLiteMultiView(SMPLLayer& smpl_layer,
                          const std::vector<SmplifyMultiViewObservation>& views,
                          SmplResult* io_res,
                          const SmplifyLiteOptions& options = {},
                          float* out_y_sign = nullptr);

struct SmplParameters {
    std::vector<float> thetas;
    std::vector<float> betas;
    std::vector<float> translation;
    float mocap_scale = 1.0f;
};

struct SmplMocapFitOptions {
    int num_iters = 20;
    float lr = 5e-2f;
    float min_joint_confidence = 1e-5f;
    float position_weight = 1.0f;
    float pose_reg = 5.0f;
    float betas_reg = 1e-3f;
    float translation_reg = 1e-3f;
    bool normalize_human_scale = true;
    float canonical_height_m = 1.77f;
    float min_scale = 1e-3f;
    float max_scale = 1000.0f;
    bool use_cuda = false;
};

class SmplifyLiteMocapSolver {
public:
    explicit SmplifyLiteMocapSolver(const std::string& model_path,
                                    const SmplMocapFitOptions& options = {});

    bool IsReady() const;
    bool FitToMocap(const std::vector<cv::Point3f>& joint_centers,
                    const std::vector<cv::Vec4f>& joint_rotations,
                    const std::vector<float>& joint_confidences,
                    SmplParameters* out_parameters);

private:
    SmplMocapFitOptions options_;
    std::shared_ptr<SMPLLayer> smpl_layer_;
};
