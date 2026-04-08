#include "dataset_prep/ingestion/CameraCalibrationLoader.h"

#include <iostream>
#include <map>
#include <string>

#include <opencv2/core/persistence.hpp>

namespace dataset_prep {
namespace {

bool ReadMatx33f(const cv::FileNode& node, cv::Matx33f* out_value) {
    if (out_value == nullptr || node.empty() || !node.isSeq() || node.size() != 9u) {
        return false;
    }

    cv::Matx33f value;
    size_t index = 0u;
    for (auto it = node.begin(); it != node.end(); ++it, ++index) {
        value.val[index] = static_cast<float>(static_cast<double>(*it));
    }
    *out_value = value;
    return true;
}

bool ReadVec3f(const cv::FileNode& node, cv::Vec3f* out_value) {
    if (out_value == nullptr || node.empty() || !node.isSeq() || node.size() != 3u) {
        return false;
    }

    cv::Vec3f value;
    size_t index = 0u;
    for (auto it = node.begin(); it != node.end(); ++it, ++index) {
        value[static_cast<int>(index)] = static_cast<float>(static_cast<double>(*it));
    }
    *out_value = value;
    return true;
}

bool LoadExtrinsics(const std::filesystem::path& extrinsics_path,
                    std::map<int, std::pair<cv::Matx33f, cv::Vec3f>>* out_extrinsics) {
    if (out_extrinsics == nullptr) {
        return false;
    }

    cv::FileStorage storage(extrinsics_path.string(),
                            cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        std::cerr << "CameraCalibrationLoader: failed to open "
                  << extrinsics_path.string() << std::endl;
        return false;
    }

    const cv::FileNode cameras = storage["cameras"];
    if (cameras.empty() || !cameras.isSeq()) {
        std::cerr << "CameraCalibrationLoader: missing cameras[] in "
                  << extrinsics_path.string() << std::endl;
        return false;
    }

    out_extrinsics->clear();
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        const cv::FileNode camera_node = *it;
        if (!camera_node["success"].empty() &&
            static_cast<int>(camera_node["success"]) == 0) {
            continue;
        }

        const int cam_id = static_cast<int>(camera_node["cam_id"]);
        const cv::FileNode extrinsics_node = camera_node["extrinsics"];
        cv::Matx33f rotation;
        cv::Vec3f translation;
        if (extrinsics_node.empty() ||
            !ReadMatx33f(extrinsics_node["R"], &rotation) ||
            !ReadVec3f(extrinsics_node["t"], &translation)) {
            std::cerr << "CameraCalibrationLoader: invalid extrinsics for cam"
                      << cam_id << " in " << extrinsics_path.string() << std::endl;
            return false;
        }

        (*out_extrinsics)[cam_id] = std::make_pair(rotation, translation);
    }

    return !out_extrinsics->empty();
}

bool LoadIntrinsics(const std::filesystem::path& intrinsics_path,
                    CameraCalibration* out_calibration) {
    if (out_calibration == nullptr) {
        return false;
    }

    cv::FileStorage storage(intrinsics_path.string(),
                            cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        std::cerr << "CameraCalibrationLoader: failed to open "
                  << intrinsics_path.string() << std::endl;
        return false;
    }

    CameraCalibration calibration;
    if (!ReadMatx33f(storage["K"], &calibration.K)) {
        std::cerr << "CameraCalibrationLoader: missing K in "
                  << intrinsics_path.string() << std::endl;
        return false;
    }

    calibration.image_width = static_cast<int>(storage["width"]);
    calibration.image_height = static_cast<int>(storage["height"]);
    if (calibration.image_width <= 0 || calibration.image_height <= 0) {
        std::cerr << "CameraCalibrationLoader: invalid image size in "
                  << intrinsics_path.string() << std::endl;
        return false;
    }

    *out_calibration = calibration;
    return true;
}

}  // namespace

bool CameraCalibrationLoader::Load(const std::filesystem::path& calibration_dir,
                                   const std::vector<VideoSourceConfig>& video_sources,
                                   std::vector<CameraCalibration>* out_calibrations) const {
    if (out_calibrations == nullptr) {
        return false;
    }

    const auto normalized_dir = std::filesystem::absolute(calibration_dir).lexically_normal();
    const auto extrinsics_path = normalized_dir / "extrinsics.json";

    std::map<int, std::pair<cv::Matx33f, cv::Vec3f>> extrinsics_by_index;
    if (!LoadExtrinsics(extrinsics_path, &extrinsics_by_index)) {
        return false;
    }

    std::vector<CameraCalibration> calibrations;
    calibrations.reserve(video_sources.size());
    for (const auto& source : video_sources) {
        if (source.source_camera_index < 0) {
            std::cerr << "CameraCalibrationLoader: invalid source camera index for "
                      << source.camera_id << std::endl;
            return false;
        }

        const auto intrinsics_path =
            normalized_dir / ("intrinsics_cam" + std::to_string(source.source_camera_index) + ".json");

        CameraCalibration calibration;
        if (!LoadIntrinsics(intrinsics_path, &calibration)) {
            return false;
        }

        const auto extrinsics_it = extrinsics_by_index.find(source.source_camera_index);
        if (extrinsics_it == extrinsics_by_index.end()) {
            std::cerr << "CameraCalibrationLoader: missing extrinsics for cam"
                      << source.source_camera_index << " in "
                      << extrinsics_path.string() << std::endl;
            return false;
        }

        calibration.source_camera_index = source.source_camera_index;
        calibration.camera_id = source.camera_id;
        calibration.R = extrinsics_it->second.first;
        calibration.t = extrinsics_it->second.second;
        calibration.valid = true;
        calibrations.push_back(std::move(calibration));
    }

    *out_calibrations = std::move(calibrations);
    return !out_calibrations->empty();
}

}  // namespace dataset_prep
