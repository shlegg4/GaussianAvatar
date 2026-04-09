#pragma once

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace dataset_prep {

constexpr size_t kMocapJointCount = 26u;
constexpr size_t kSmplPoseParamCount = 72u;
constexpr size_t kSmplShapeParamCount = 10u;

struct Joint3D {
    cv::Point3f xyz{0.0f, 0.0f, 0.0f};
    cv::Vec4f quaternion{1.0f, 0.0f, 0.0f, 0.0f};
    float confidence = 0.0f;

    bool IsValid(float min_confidence = 0.0f) const {
        return std::isfinite(xyz.x) && std::isfinite(xyz.y) &&
               std::isfinite(xyz.z) && confidence >= min_confidence;
    }
};

struct MocapPerson3D {
    int person_id = -1;
    float score = 0.0f;
    std::array<Joint3D, kMocapJointCount> joints{};
    std::vector<float> smpl_pose = std::vector<float>(kSmplPoseParamCount, 0.0f);
    std::vector<float> smpl_shape = std::vector<float>(kSmplShapeParamCount, 0.0f);
    cv::Point3f smpl_translation{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()};
    float smpl_scale = std::numeric_limits<float>::quiet_NaN();
    bool smpl_valid = false;

    float MeanConfidence() const {
        float sum = 0.0f;
        for (const auto& joint : joints) {
            sum += joint.confidence;
        }
        return sum / static_cast<float>(joints.size());
    }
};

struct MocapFrame3D {
    int frame_seq = -1;
    double timestamp_ms = -1.0;
    double slot_timestamp_ms = -1.0;
    std::vector<MocapPerson3D> people;

    double ReferenceTimestampMs() const {
        return slot_timestamp_ms >= 0.0 ? slot_timestamp_ms : timestamp_ms;
    }

    bool HasTimestamp() const {
        return ReferenceTimestampMs() >= 0.0;
    }
};

struct MocapSequence3D {
    std::filesystem::path source_path;
    std::map<int, MocapFrame3D> frames_by_seq;
    std::multimap<double, int> frame_seq_by_timestamp_ms;

    const MocapFrame3D* FindClosestFrameByTimestamp(double target_timestamp_ms,
                                                    double max_delta_ms) const {
        if (target_timestamp_ms < 0.0 || frame_seq_by_timestamp_ms.empty()) {
            return nullptr;
        }

        auto lower = frame_seq_by_timestamp_ms.lower_bound(target_timestamp_ms);
        double best_delta = std::numeric_limits<double>::max();
        int best_frame_seq = -1;

        auto consider = [&](const std::multimap<double, int>::const_iterator& iterator) {
            if (iterator == frame_seq_by_timestamp_ms.end()) {
                return;
            }
            const double delta = std::abs(iterator->first - target_timestamp_ms);
            if (delta < best_delta) {
                best_delta = delta;
                best_frame_seq = iterator->second;
            }
        };

        consider(lower);
        if (lower != frame_seq_by_timestamp_ms.begin()) {
            auto previous = lower;
            --previous;
            consider(previous);
        }

        if (best_frame_seq >= 0 && best_delta <= max_delta_ms) {
            const auto frame_it = frames_by_seq.find(best_frame_seq);
            if (frame_it != frames_by_seq.end()) {
                return &frame_it->second;
            }
        }
        return nullptr;
    }

    const MocapFrame3D* FindClosestFrameByFrameSeq(int target_frame_seq,
                                                   int max_delta) const {
        if (target_frame_seq < 0 || frames_by_seq.empty()) {
            return nullptr;
        }

        const auto exact_it = frames_by_seq.find(target_frame_seq);
        if (exact_it != frames_by_seq.end()) {
            return &exact_it->second;
        }

        const auto lower = frames_by_seq.lower_bound(target_frame_seq);
        const MocapFrame3D* best_frame = nullptr;
        int best_delta = std::numeric_limits<int>::max();

        auto consider = [&](const std::map<int, MocapFrame3D>::const_iterator& iterator) {
            if (iterator == frames_by_seq.end()) {
                return;
            }
            const int delta = std::abs(iterator->first - target_frame_seq);
            if (delta < best_delta) {
                best_delta = delta;
                best_frame = &iterator->second;
            }
        };

        consider(lower);
        if (lower != frames_by_seq.begin()) {
            auto previous = lower;
            --previous;
            consider(previous);
        }

        if (best_delta <= max_delta) {
            return best_frame;
        }
        return nullptr;
    }

    const MocapFrame3D* FindClosestFrame(double target_timestamp_ms,
                                         int target_frame_seq,
                                         double max_timestamp_delta_ms,
                                         int max_frame_delta,
                                         bool prefer_timestamps = true) const {
        if (prefer_timestamps) {
            if (const auto* frame = FindClosestFrameByTimestamp(target_timestamp_ms, max_timestamp_delta_ms)) {
                return frame;
            }
            return FindClosestFrameByFrameSeq(target_frame_seq, max_frame_delta);
        }

        if (const auto* frame = FindClosestFrameByFrameSeq(target_frame_seq, max_frame_delta)) {
            return frame;
        }
        return FindClosestFrameByTimestamp(target_timestamp_ms, max_timestamp_delta_ms);
    }
};

struct VideoSourceConfig {
    std::string camera_id;
    int source_camera_index = -1;
    std::filesystem::path video_path;
    double timestamp_offset_ms = 0.0;
    int frame_stride = 1;
};

struct SyncedView {
    std::string camera_id;
    int source_camera_index = -1;
    int video_frame_index = -1;
    double video_timestamp_ms = -1.0;
    cv::Mat image;
};

struct SyncedFrameCollection {
    int sync_index = -1;
    double sync_timestamp_ms = -1.0;
    std::vector<SyncedView> views;
};

struct PoseLookupResult {
    int sync_index = -1;
    double sync_timestamp_ms = -1.0;
    double pose_timestamp_ms = -1.0;
    double timestamp_delta_ms = -1.0;
    const MocapFrame3D* frame = nullptr;
};

struct BackgroundExtractorResult {
    std::string camera_id;
    int source_camera_index = -1;
    cv::Mat matte;
    cv::Mat foreground;
};

struct CameraCalibration {
    int source_camera_index = -1;
    std::string camera_id;
    cv::Matx33f K = cv::Matx33f::eye();
    cv::Matx33f R = cv::Matx33f::eye();
    cv::Vec3f t{0.0f, 0.0f, 0.0f};
    int image_width = 0;
    int image_height = 0;
    bool valid = false;
};

struct ExportTrainingSample {
    std::string camera_id;
    int source_camera_index = -1;
    int video_frame_index = -1;
    int person_index = -1;
    int person_id = -1;
    cv::Mat crop_image;
    cv::Mat crop_matte;
    cv::Mat crop_overlay;
    float img_w = 0.0f;
    float img_h = 0.0f;
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    float crop_x0 = 0.0f;
    float crop_y0 = 0.0f;
    float crop_w = 0.0f;
    float crop_h = 0.0f;
    float focal_length = 0.0f;
    float y_sign = 1.0f;
    float smpl_scale = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> cam = std::vector<float>(3u, 0.0f);
    std::vector<float> pose = std::vector<float>(kSmplPoseParamCount, 0.0f);
    std::vector<float> betas = std::vector<float>(kSmplShapeParamCount, 0.0f);
};

struct ExportFrameArtifacts {
    SyncedFrameCollection synced_frames;
    PoseLookupResult pose_lookup;
    std::vector<BackgroundExtractorResult> masks;
    MocapFrame3D pose3d;
    std::vector<ExportTrainingSample> training_samples;
    bool training_export_requested = false;
};

}  // namespace dataset_prep
