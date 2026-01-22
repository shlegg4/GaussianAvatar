#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

struct SmplResult {
    std::vector<float> pose;   // 72 or 144 floats
    std::vector<float> shape;  // 10 floats
    std::vector<float> camera; // 3 floats
};

using ResultsDict = std::map<int, SmplResult>;

struct HmrOutputOptions {
    std::string output_dir;
    std::string smpl_model_path = "smpl_data.pt";
    bool save_outputs = false;
    bool use_yolo = false;
    std::string yolo_model_path;
    float focal_length_scale = 1.2f;
};

bool RunHmrInferenceOnVideo(const std::string& model_path,
                            const std::string& video_path,
                            const HmrOutputOptions& options,
                            ResultsDict* out_results);
