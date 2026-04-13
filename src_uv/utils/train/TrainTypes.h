#pragma once

#include <array>
#include <string>
#include <vector>

struct TrainSample
{
    int frame = -1;
    std::string body_model = "smpl";
    std::string crop_path;
    int img_w = 0;
    int img_h = 0;
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    float crop_x0 = 0.0f;
    float crop_y0 = 0.0f;
    float crop_w = 0.0f;
    float crop_h = 0.0f;
    float focal_length = 0.0f;
    float y_sign = 1.0f;
    std::vector<float> pose;
    std::vector<float> betas;
    std::vector<float> cam;
    std::vector<float> smplx_expression;
    std::vector<float> smplx_jaw_pose;
    std::vector<float> smplx_eye_pose;
    std::vector<float> smplx_left_hand_pose;
    std::vector<float> smplx_right_hand_pose;
    std::vector<float> camera_rt;
    std::array<float, 3> translation{0.0f, 0.0f, 0.0f};
    bool has_translation = false;
    bool uses_smplx = false;
};

struct TrainOptions
{
    std::string jsonl_path;
    std::string smpl_model_path = "smplx_data.pt";
    int num_gaussians = 5000;
    int max_gaussians = -1;
    int epochs = 1;
    float lr = 0.01f;
    int lr_decay_epoch = -1;
    float lr_decay_multiplier = 0.5f;
    float lr_min_multiplier = 0.1f;
    std::string output_dir = "outputs";
    std::string viewer_export_dir;
    float scale_reg_weight = 200.0f;
    float scale_lr = -1.0f;
    float scale_max_value = 0.02f;
    float rot_lr = -1.0f;
    float offset_lr = -1.0f;
    float offset_reg_weight = 0.01f;
    float pose_reg_weight = 0.01f;
    float pose_lr = 5e-3f;
    float betas_lr = 1e-2f;
    float betas_reg_weight = 1e-4f;
    float expr_bias_lr = 5e-3f;
    float expr_bias_reg_weight = 1e-4f;
    float alpha_loss_weight = 0.1f;
    float opacity_reg_weight = 0.0f;
    float sh_reg_weight = 5e-5f;
    float sugar_weight = 1e-2f;
    float anisotropic_reg_weight = 0.0f;
    float lambda_dssim = 0.2f;
    float color_lr = -1.0f;
    float opacity_lr = -1.0f;
    float psr_opacity_threshold = 0.3f;
    int psr_samples_per_gaussian = 8;
    std::string mesh_method = "poisson";
    int sh_degree = 3;
    bool enable_viewer = false;
    int viewer_every = 1;
    std::string viewer_shm_name = "GaussianAvatarShared";
    std::string viewer_pose_shm_name;
    std::string viewer_bind_shm_name;
    bool viewer_stream_poses = false;
    bool verbose_diagnostics = false;
    int verbose_every = 100;
    int loader_workers = 2;
    int loader_prefetch_batches = 2;
    int torch_cpu_threads = 1;
    int torch_interop_threads = 1;
    int opencv_threads = 1;
};
