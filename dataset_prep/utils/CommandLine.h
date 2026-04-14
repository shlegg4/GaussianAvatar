#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

struct DatasetPrepOptions {
    std::filesystem::path output_dir;
    std::string target_camera_id;
    std::vector<VideoSourceConfig> sources;
    int max_frames = -1;
    int start_frame_index = 0;
    int frame_stride = 1;
    double sync_tolerance_ms = 8.0;
    float temporal_smooth_alpha = 1.0f;
};

void PrintDatasetPrepUsage(std::ostream& out);
bool ParseDatasetPrepCommandLine(int argc, char* argv[], DatasetPrepOptions* out_options);
bool ResolveCalibrationDirectory(std::filesystem::path* out_dir);

}  // namespace dataset_prep
