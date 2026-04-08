#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/videoio.hpp>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

class VideoSynchronizer {
public:
    struct Options {
        double sync_tolerance_ms = 8.0;
    };

    explicit VideoSynchronizer(const Options& options = {});

    bool Open(const std::vector<VideoSourceConfig>& sources);
    bool HasNextFrame() const;
    bool GetNextSyncedViews(SyncedFrameCollection* out_collection);

private:
    struct CaptureState {
        VideoSourceConfig config;
        std::filesystem::path resolved_video_path;
        cv::VideoCapture capture;
        SyncedView current_view;
        bool has_current = false;
        bool end_of_stream = false;
    };

    bool AdvanceState(CaptureState* state);
    bool AlignCurrentFrames();

    Options options_;
    int next_sync_index_ = 0;
    std::vector<CaptureState> states_;
};

}  // namespace dataset_prep
