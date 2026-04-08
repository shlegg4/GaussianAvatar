#pragma once

#include <filesystem>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

class DatasetExporter {
public:
    struct Options {
        std::filesystem::path output_dir;
        bool save_images = true;
        bool save_masks = true;
        bool save_pose_json = true;
        bool append_manifest = true;
        int png_compression = 3;
    };

    explicit DatasetExporter(const Options& options = {});

    bool Initialize();
    bool SaveFrame(const ExportFrameArtifacts& artifacts);

private:
    bool EnsureCameraDirectories(const ExportFrameArtifacts& artifacts) const;
    bool WritePoseJson(const ExportFrameArtifacts& artifacts,
                       const std::filesystem::path& pose_path) const;
    bool AppendManifestLine(const ExportFrameArtifacts& artifacts,
                            const std::filesystem::path& pose_path) const;
    bool UseTrainingExport(const ExportFrameArtifacts& artifacts) const;
    std::filesystem::path MakeTrainingCropPath(const ExportFrameArtifacts& artifacts,
                                               const ExportTrainingSample& sample) const;
    std::filesystem::path MakeTrainingMattePath(const ExportFrameArtifacts& artifacts,
                                                const ExportTrainingSample& sample) const;
    std::string MakeTrainingStem(const ExportFrameArtifacts& artifacts,
                                 const ExportTrainingSample& sample) const;
    std::string MakeFrameStem(int sync_index) const;

    Options options_;
};

}  // namespace dataset_prep
