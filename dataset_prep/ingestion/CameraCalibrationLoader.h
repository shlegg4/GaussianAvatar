#pragma once

#include <filesystem>
#include <vector>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

class CameraCalibrationLoader {
public:
    bool Load(const std::filesystem::path& calibration_dir,
              const std::vector<VideoSourceConfig>& video_sources,
              std::vector<CameraCalibration>* out_calibrations) const;
};

}  // namespace dataset_prep
