#include "dataset_prep/ingestion/VideoSynchronizer.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <system_error>

namespace dataset_prep {
namespace {

const char* BackendName(int backend) {
    switch (backend) {
        case cv::CAP_FFMPEG: return "FFMPEG";
        case cv::CAP_MSMF: return "MSMF";
        case cv::CAP_DSHOW: return "DSHOW";
        case cv::CAP_GSTREAMER: return "GSTREAMER";
        case cv::CAP_ANY: return "ANY";
        default: return "UNKNOWN";
    }
}

bool TryOpenCapture(const std::filesystem::path& video_path,
                    cv::VideoCapture* capture,
                    std::string* out_error) {
    if (capture == nullptr) {
        return false;
    }

    const auto normalized_path = std::filesystem::absolute(video_path).lexically_normal();
    const std::string filename = normalized_path.generic_string();
    if (filename.empty()) {
        if (out_error != nullptr) {
            *out_error = "empty filename";
        }
        return false;
    }

    if (!std::filesystem::exists(normalized_path)) {
        if (out_error != nullptr) {
            *out_error = "file does not exist: " + filename;
        }
        return false;
    }

    const std::vector<int> backends = {
        cv::CAP_FFMPEG,
        cv::CAP_MSMF,
        cv::CAP_DSHOW,
        cv::CAP_ANY,
    };

    std::ostringstream error_stream;
    for (const int backend : backends) {
        capture->release();
        if (capture->open(filename, backend) && capture->isOpened()) {
            return true;
        }

        if (error_stream.tellp() > 0) {
            error_stream << ", ";
        }
        error_stream << BackendName(backend);
    }

    if (out_error != nullptr) {
        *out_error = "all backends failed for " + filename +
                     " (tried: " + error_stream.str() + ")";
    }
    return false;
}

std::filesystem::path MakeTranscodePath(const std::filesystem::path& video_path) {
    const auto normalized_path = std::filesystem::absolute(video_path).lexically_normal();
    const auto temp_root = std::filesystem::temp_directory_path() / "GaussianAvatar_dataset_prep";
    const auto hash_value = std::hash<std::string>{}(normalized_path.generic_string());
    std::ostringstream filename;
    filename << normalized_path.stem().string() << "_" << hash_value << ".avi";
    return temp_root / filename.str();
}

bool EnsureMjpegFallback(const std::filesystem::path& source_path,
                         std::filesystem::path* out_fallback_path,
                         std::string* out_error) {
    if (out_fallback_path == nullptr) {
        return false;
    }

    std::error_code error;
    const auto fallback_path = MakeTranscodePath(source_path);
    std::filesystem::create_directories(fallback_path.parent_path(), error);
    if (error) {
        if (out_error != nullptr) {
            *out_error = "failed to create temp directory for fallback AVI";
        }
        return false;
    }

    const bool needs_transcode =
        !std::filesystem::exists(fallback_path, error) ||
        (!error && std::filesystem::last_write_time(fallback_path, error) <
                        std::filesystem::last_write_time(source_path, error));
    error.clear();

    if (needs_transcode) {
        std::ostringstream command;
        command << "ffmpeg -y -hide_banner -loglevel error -i \""
                << std::filesystem::absolute(source_path).string()
                << "\" -an -c:v mjpeg -q:v 2 \"" << fallback_path.string() << "\"";
        const int exit_code = std::system(command.str().c_str());
        if (exit_code != 0 || !std::filesystem::exists(fallback_path)) {
            if (out_error != nullptr) {
                *out_error = "ffmpeg fallback transcode failed";
            }
            return false;
        }
    }

    *out_fallback_path = fallback_path;
    return true;
}

}  // namespace

VideoSynchronizer::VideoSynchronizer(const Options& options)
    : options_(options) {}

bool VideoSynchronizer::Open(const std::vector<VideoSourceConfig>& sources) {
    states_.clear();
    next_sync_index_ = 0;

    for (const auto& source : sources) {
        CaptureState state;
        state.config = source;
        state.resolved_video_path = source.video_path;
        std::string open_error;
        if (!TryOpenCapture(state.resolved_video_path, &state.capture, &open_error)) {
            std::filesystem::path fallback_path;
            std::string fallback_error;
            if (EnsureMjpegFallback(source.video_path, &fallback_path, &fallback_error) &&
                TryOpenCapture(fallback_path, &state.capture, &open_error)) {
                state.resolved_video_path = fallback_path;
            } else {
                std::cerr << "VideoSynchronizer: failed to open "
                          << source.video_path.string();
                if (!open_error.empty()) {
                    std::cerr << " (" << open_error << ")";
                }
                if (!fallback_error.empty()) {
                    std::cerr << "; ffmpeg fallback: " << fallback_error;
                }
                std::cerr << std::endl;
                states_.clear();
                return false;
            }
        }
        if (!state.capture.isOpened()) {
            std::cerr << "VideoSynchronizer: failed to open "
                      << source.video_path.string();
            if (!open_error.empty()) {
                std::cerr << " (" << open_error << ")";
            }
            std::cerr << std::endl;
            states_.clear();
            return false;
        }
        if (!AdvanceState(&state)) {
            std::cerr << "VideoSynchronizer: no readable frames in " << source.video_path.string() << std::endl;
            states_.clear();
            return false;
        }
        states_.push_back(std::move(state));
    }

    return !states_.empty();
}

bool VideoSynchronizer::HasNextFrame() const {
    if (states_.empty()) {
        return false;
    }
    for (const auto& state : states_) {
        if (!state.has_current || state.end_of_stream) {
            return false;
        }
    }
    return true;
}

bool VideoSynchronizer::GetNextSyncedViews(SyncedFrameCollection* out_collection) {
    if (out_collection == nullptr || !HasNextFrame()) {
        return false;
    }
    if (!AlignCurrentFrames()) {
        return false;
    }

    SyncedFrameCollection collection;
    collection.sync_index = next_sync_index_++;
    collection.views.reserve(states_.size());

    double timestamp_sum = 0.0;
    for (const auto& state : states_) {
        SyncedView view = state.current_view;
        view.image = state.current_view.image.clone();
        collection.views.push_back(std::move(view));
        timestamp_sum += state.current_view.video_timestamp_ms;
    }
    collection.sync_timestamp_ms = timestamp_sum / static_cast<double>(collection.views.size());

    for (auto& state : states_) {
        AdvanceState(&state);
    }

    *out_collection = std::move(collection);
    return true;
}

bool VideoSynchronizer::AdvanceState(CaptureState* state) {
    if (state == nullptr) {
        return false;
    }

    const int stride = std::max(1, state->config.frame_stride);
    cv::Mat frame;
    bool read_success = false;
    for (int i = 0; i < stride; ++i) {
        if (!state->capture.read(frame)) {
            state->end_of_stream = true;
            state->has_current = false;
            state->current_view.image.release();
            return false;
        }
        read_success = true;
    }

    if (!read_success) {
        state->end_of_stream = true;
        state->has_current = false;
        return false;
    }

    state->current_view.camera_id = state->config.camera_id;
    state->current_view.source_camera_index = state->config.source_camera_index;
    state->current_view.video_frame_index =
        std::max(0, static_cast<int>(state->capture.get(cv::CAP_PROP_POS_FRAMES)) - 1);
    state->current_view.video_timestamp_ms =
        state->capture.get(cv::CAP_PROP_POS_MSEC) + state->config.timestamp_offset_ms;
    state->current_view.image = frame;
    state->has_current = true;
    state->end_of_stream = false;
    return true;
}

bool VideoSynchronizer::AlignCurrentFrames() {
    if (!HasNextFrame()) {
        return false;
    }

    std::vector<double> timestamps;
    timestamps.reserve(states_.size());
    for (const auto& state : states_) {
        timestamps.push_back(state.current_view.video_timestamp_ms);
    }
    std::sort(timestamps.begin(), timestamps.end());
    const double target_timestamp = timestamps[timestamps.size() / 2u];

    for (auto& state : states_) {
        while (state.has_current &&
               state.current_view.video_timestamp_ms + options_.sync_tolerance_ms < target_timestamp) {
            if (!AdvanceState(&state)) {
                return false;
            }
        }
    }
    return HasNextFrame();
}

}  // namespace dataset_prep
