#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/core.hpp>

// Configuration for the folder-based optimizer
struct FolderOptimizerOptions {
    std::string input_folder;
    std::string output_dir;
    std::string smpl_model_path = "smpl_data.pt";
    
    bool use_modnet = false;
    std::string modnet_model_path;
    bool modnet_use_cuda = false;
    int modnet_input_size = 512;

    int frame_stride = 1;
    bool save_visualization = true;
    
    // Optimization params
    int smplify_iters = 150;
    float smplify_lr = 1e-2f;
    float smplify_pos_weight = 1.0f;
    float smplify_reg_weight = 0.05f;
};

// Main entry point
bool RunFolderOptimization(const FolderOptimizerOptions& options);