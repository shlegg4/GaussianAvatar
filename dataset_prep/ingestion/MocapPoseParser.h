#pragma once

#include <filesystem>
#include <string>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

class MocapPoseParser {
public:
    struct ParseOptions {
        double timestamp_scale = 1.0;
        bool skip_empty_people = false;
    };

    bool Parse(const std::filesystem::path& jsonl_path,
               MocapSequence3D* out_sequence,
               const ParseOptions& options = {}) const;

private:
    bool ParseLine(const std::string& line,
                   MocapFrame3D* out_frame,
                   double timestamp_scale,
                   size_t line_number = 0u) const;
};

}  // namespace dataset_prep
