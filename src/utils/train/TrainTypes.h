#pragma once

#include <string>
#include <vector>

struct TrainSample
{
    int frame = -1;
    std::string crop_path;
    int img_w = 0;
    int img_h = 0;
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    float focal_length = 0.0f;
    float y_sign = 1.0f;
    std::vector<float> pose;
    std::vector<float> betas;
    std::vector<float> cam;
};

struct TrainOptions
{
    std::string jsonl_path;
    std::string smpl_model_path = "smpl_data.pt";
    int num_gaussians = 5000;
    int epochs = 1;
    float lr = 0.01f;
    std::string output_dir = "outputs";
    float scale_reg_weight = 0.001f;
    float scale_max_reg_weight = 0.1f;
    float scale_max_value = 0.02f;
    float offset_reg_weight = 0.01f;
    float mesh_reg_weight = 0.1f;
    float mesh_reg_max_dist = 0.02f;
    float color_lr = -1.0f;
    float opacity_lr = -1.0f;
    int sh_degree = 3;
    int densify_every = 500;
    int densify_max_splits = 64;
    float densify_scale_threshold = -1.0f;
    float densify_split_scale = 0.7f;
    float densify_split_offset = 0.5f;
    float densify_min_grad = 0.0f;
    float densify_grow_grad = 1e-4f;
    float densify_prune_opacity = 0.01f;
    int densify_prune_max = 256;
    float densify_reset_opacity = 0.001f;
    int densify_stop_epoch = 15;
    bool enable_viewer = false;
    int viewer_width = 800;
    int viewer_height = 600;
    int viewer_every = 1;
    std::string viewer_shm_name = "GaussianAvatarShared";
};
