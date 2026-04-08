#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

class BackgroundExtractor {
public:
    struct Options {
        std::string model_path;
        bool use_cuda = false;
        int input_size = 512;
        float binary_threshold = -1.0f;
    };

    explicit BackgroundExtractor(const Options& options = {});
    ~BackgroundExtractor();

    bool Initialize();
    bool Process(const SyncedFrameCollection& synced_frames,
                 std::vector<BackgroundExtractorResult>* out_results);
    bool ProcessImage(const cv::Mat& image, cv::Mat* out_matte);

private:
    Options options_;
    bool ready_ = false;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dataset_prep
