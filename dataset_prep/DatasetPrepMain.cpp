#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset_prep/export/DatasetExporter.h"
#include "dataset_prep/ingestion/CameraCalibrationLoader.h"
#include "dataset_prep/ingestion/MocapPoseParser.h"
#include "dataset_prep/ingestion/VideoSynchronizer.h"
#include "dataset_prep/processing/BackgroundExtractor.h"
#include "dataset_prep/processing/PearEstimator.h"
#if DATASET_PREP_HAS_SMPL
#include "utils/HmrInferenceUtils.h"
#include "utils/HmrMathHelpers.h"
#include "utils/RtmPoseDetector.h"
#include "utils/SmplLBS.h"
#include "utils/SmplifyLite.h"
#endif
#include "utils/YoloPersonDetector.h"

namespace dataset_prep {
namespace {

using SteadyClock = std::chrono::steady_clock;

struct CliOptions {
    std::vector<std::filesystem::path> video_paths;
    std::filesystem::path pose_3d_path;
    std::vector<std::string> camera_ids;
    std::filesystem::path output_dir;
    std::filesystem::path calibration_dir;
    double sync_tolerance_ms = 8.0;
    double pose_timestamp_scale = 1.0;
    double pose_timestamp_delta_ms = 8.0;
    int pose_frame_delta = 2;
    int frame_stride = 1;
    int max_frames = -1;
    int crop_resolution = 512;
    float crop_margin = 1.2f;
    bool prefer_pose_timestamps = true;
    bool save_images = true;
    bool save_masks = true;
    bool save_pose_json = true;
    std::string modnet_model_path;
    bool modnet_use_cuda = false;
    int modnet_input_size = 512;
    float modnet_binary_threshold = -1.0f;
    bool save_training_debug = true;
#if DATASET_PREP_HAS_SMPL
    bool smpl_enabled = true;
#else
    bool smpl_enabled = false;
#endif
    bool smpl_use_pear = true;
    std::string smpl_model_path = "smpl_data.pt";
    bool smpl_use_cuda = false;
    int smpl_iters = 60;
    std::string pear_model_path = "pear.onnx";
    bool pear_use_cuda = false;
    std::string rtmpose_model_path = "rtmpose26.onnx";
    bool rtmpose_use_cuda = false;
    std::string yolo_model_path = "yolov8n.onnx";
    bool yolo_use_cuda = false;
    float yolo_conf_threshold = 0.25f;
    float yolo_nms_threshold = 0.45f;
};

std::vector<std::string> SplitList(const std::string& value) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string token = comma == std::string::npos
            ? value.substr(start)
            : value.substr(start, comma - start);
        if (!token.empty()) {
            parts.push_back(token);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return parts;
}

void AppendPaths(const std::string& csv_value, std::vector<std::filesystem::path>* out_paths) {
    if (out_paths == nullptr) {
        return;
    }
    for (const auto& token : SplitList(csv_value)) {
        out_paths->push_back(token);
    }
}

void AppendStrings(const std::string& csv_value, std::vector<std::string>* out_values) {
    if (out_values == nullptr) {
        return;
    }
    for (const auto& token : SplitList(csv_value)) {
        out_values->push_back(token);
    }
}

std::filesystem::path ResolveExistingPath(const std::filesystem::path& input_path) {
    if (input_path.empty()) {
        return input_path;
    }

    std::error_code ec;
    if (input_path.is_absolute()) {
        return input_path;
    }
    if (std::filesystem::exists(input_path, ec)) {
        return std::filesystem::absolute(input_path, ec);
    }

    std::filesystem::path probe = std::filesystem::current_path(ec);
    for (int depth = 0; !probe.empty() && depth <= 4; ++depth) {
        const std::filesystem::path candidate = probe / input_path;
        if (std::filesystem::exists(candidate, ec)) {
            return std::filesystem::absolute(candidate, ec);
        }
        probe = probe.parent_path();
    }

    return input_path;
}

double ElapsedMilliseconds(const SteadyClock::time_point& start,
                           const SteadyClock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void LogProgress(const std::string& message) {
    std::cout << message << std::endl;
}

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  dataset_prep --videos cam1.mp4,cam2.mp4,...\n"
        << "               --pose-3d mocap_3d.jsonl --output out_dir [options]\n\n"
        << "Options:\n"
        << "  --camera-ids ids             Comma-separated camera ids matching the video order\n"
        << "  --calibration-dir dir        Directory with extrinsics.json and intrinsics_camN.json\n"
        << "  --sync-tolerance-ms value    Video sync tolerance in milliseconds\n"
        << "  --pose-timestamp-scale val   Multiply pose timestamps by this scale before matching\n"
        << "  --pose-timestamp-delta-ms v  Max pose/video timestamp mismatch\n"
        << "  --pose-frame-delta value     Max fallback frame index mismatch\n"
        << "  --frame-stride value         Read every Nth frame from each video\n"
        << "  --max-frames value           Stop after N synchronized frame groups\n"
        << "  --crop-res value             Output crop resolution for training export\n"
        << "  --crop-margin value          Expand the projected square crop by this factor\n"
        << "  --prefer-frame-seq           Match poses by frame index first\n"
        << "  --no-images                  Skip RGB export\n"
        << "  --no-masks                   Skip matte export\n"
        << "  --no-training-debug          Skip per-sample training debug overlays\n"
        << "  --no-pose-json               Skip per-frame 3D pose json export\n"
        << "  --modnet model.onnx          Required MODNet model for mask extraction\n"
        << "  --modnet-cuda                Request CUDA provider for MODNet\n"
        << "  --modnet-input value         MODNet input size for dynamic models\n"
        << "  --modnet-threshold value     Binarize matte at the provided threshold [0,1]\n"
#if DATASET_PREP_HAS_SMPL
        << "  --no-smpl                    Skip SMPL parameter fitting\n"
        << "  --smpl-from-mocap           Use the old mocap-only SMPL solver instead of PEAR\n"
        << "  --smpl-model path            Path to the SMPL model archive (.pt)\n"
        << "  --smpl-cuda                  Request CUDA for SMPL fitting when available\n"
        << "  --smpl-iters value           Number of optimization steps per person\n"
        << "  --pear-model path            Path to the PEAR ONNX model\n"
        << "  --pear-cuda                  Request CUDA provider for PEAR inference\n"
        << "  --cliff-model path           Deprecated alias for --pear-model\n"
        << "  --cliff-cuda                 Deprecated alias for --pear-cuda\n"
        << "  --rtmpose-model path         Path to the RTMPose ONNX model used to score PEAR views\n"
        << "  --rtmpose-cuda               Request CUDA provider for RTMPose inference\n"
        << "  --yolo-model path            Path to the YOLO ONNX model used for person crops\n"
        << "  --yolo-cuda                  Request CUDA provider for YOLO inference\n"
        << "  --yolo-conf-threshold value  Minimum YOLO confidence for person detections\n"
        << "  --yolo-nms-threshold value   NMS threshold for YOLO person detections\n"
#endif
        << "  --help                       Show this message\n";
}

bool ParseArgs(int argc, char* argv[], CliOptions* out_options) {
    if (out_options == nullptr) {
        return false;
    }

    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto require_value = [&](const char* name) -> const char* {
            if (index + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++index];
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage();
            return false;
        }
        if (arg == "--videos") {
            const char* value = require_value("--videos");
            if (!value) return false;
            AppendPaths(value, &options.video_paths);
            continue;
        }
        if (arg == "--pose-3d") {
            const char* value = require_value("--pose-3d");
            if (!value) return false;
            options.pose_3d_path = value;
            continue;
        }
        if (arg == "--camera-ids") {
            const char* value = require_value("--camera-ids");
            if (!value) return false;
            AppendStrings(value, &options.camera_ids);
            continue;
        }
        if (arg == "--calibration-dir") {
            const char* value = require_value("--calibration-dir");
            if (!value) return false;
            options.calibration_dir = value;
            continue;
        }
        if (arg == "--output") {
            const char* value = require_value("--output");
            if (!value) return false;
            options.output_dir = value;
            continue;
        }
        if (arg == "--sync-tolerance-ms") {
            const char* value = require_value("--sync-tolerance-ms");
            if (!value) return false;
            options.sync_tolerance_ms = std::stod(value);
            continue;
        }
        if (arg == "--pose-timestamp-scale") {
            const char* value = require_value("--pose-timestamp-scale");
            if (!value) return false;
            options.pose_timestamp_scale = std::stod(value);
            continue;
        }
        if (arg == "--pose-timestamp-delta-ms") {
            const char* value = require_value("--pose-timestamp-delta-ms");
            if (!value) return false;
            options.pose_timestamp_delta_ms = std::stod(value);
            continue;
        }
        if (arg == "--pose-frame-delta") {
            const char* value = require_value("--pose-frame-delta");
            if (!value) return false;
            options.pose_frame_delta = std::stoi(value);
            continue;
        }
        if (arg == "--frame-stride") {
            const char* value = require_value("--frame-stride");
            if (!value) return false;
            options.frame_stride = std::stoi(value);
            continue;
        }
        if (arg == "--max-frames") {
            const char* value = require_value("--max-frames");
            if (!value) return false;
            options.max_frames = std::stoi(value);
            continue;
        }
        if (arg == "--crop-res") {
            const char* value = require_value("--crop-res");
            if (!value) return false;
            options.crop_resolution = std::stoi(value);
            continue;
        }
        if (arg == "--crop-margin") {
            const char* value = require_value("--crop-margin");
            if (!value) return false;
            options.crop_margin = std::stof(value);
            continue;
        }
        if (arg == "--prefer-frame-seq") {
            options.prefer_pose_timestamps = false;
            continue;
        }
        if (arg == "--no-images") {
            options.save_images = false;
            continue;
        }
        if (arg == "--no-masks") {
            options.save_masks = false;
            continue;
        }
        if (arg == "--no-training-debug") {
            options.save_training_debug = false;
            continue;
        }
        if (arg == "--no-pose-json") {
            options.save_pose_json = false;
            continue;
        }
        if (arg == "--modnet") {
            const char* value = require_value("--modnet");
            if (!value) return false;
            options.modnet_model_path = value;
            continue;
        }
        if (arg == "--modnet-cuda") {
            options.modnet_use_cuda = true;
            continue;
        }
        if (arg == "--modnet-input") {
            const char* value = require_value("--modnet-input");
            if (!value) return false;
            options.modnet_input_size = std::stoi(value);
            continue;
        }
        if (arg == "--modnet-threshold") {
            const char* value = require_value("--modnet-threshold");
            if (!value) return false;
            options.modnet_binary_threshold = std::stof(value);
            continue;
        }
        if (arg == "--no-smpl") {
            options.smpl_enabled = false;
            continue;
        }
        if (arg == "--smpl-from-mocap") {
            options.smpl_use_pear = false;
            continue;
        }
        if (arg == "--smpl-model") {
            const char* value = require_value("--smpl-model");
            if (!value) return false;
            options.smpl_model_path = value;
            continue;
        }
        if (arg == "--smpl-cuda") {
            options.smpl_use_cuda = true;
            continue;
        }
        if (arg == "--smpl-iters") {
            const char* value = require_value("--smpl-iters");
            if (!value) return false;
            options.smpl_iters = std::stoi(value);
            continue;
        }
        if (arg == "--pear-model" || arg == "--cliff-model") {
            const char* value = require_value(arg.c_str());
            if (!value) return false;
            options.pear_model_path = value;
            if (arg == "--cliff-model") {
                std::cerr << "DatasetPrep: --cliff-model is deprecated; using it as --pear-model."
                          << std::endl;
            }
            continue;
        }
        if (arg == "--pear-cuda" || arg == "--cliff-cuda") {
            options.pear_use_cuda = true;
            if (arg == "--cliff-cuda") {
                std::cerr << "DatasetPrep: --cliff-cuda is deprecated; using it as --pear-cuda."
                          << std::endl;
            }
            continue;
        }
        if (arg == "--rtmpose-model") {
            const char* value = require_value("--rtmpose-model");
            if (!value) return false;
            options.rtmpose_model_path = value;
            continue;
        }
        if (arg == "--rtmpose-cuda") {
            options.rtmpose_use_cuda = true;
            continue;
        }
        if (arg == "--yolo-model") {
            const char* value = require_value("--yolo-model");
            if (!value) return false;
            options.yolo_model_path = value;
            continue;
        }
        if (arg == "--yolo-cuda") {
            options.yolo_use_cuda = true;
            continue;
        }
        if (arg == "--yolo-conf-threshold") {
            const char* value = require_value("--yolo-conf-threshold");
            if (!value) return false;
            options.yolo_conf_threshold = std::stof(value);
            continue;
        }
        if (arg == "--yolo-nms-threshold") {
            const char* value = require_value("--yolo-nms-threshold");
            if (!value) return false;
            options.yolo_nms_threshold = std::stof(value);
            continue;
        }

        std::cerr << "Unknown argument: " << arg << std::endl;
        return false;
    }

    if (options.video_paths.empty() || options.output_dir.empty()) {
        PrintUsage();
        return false;
    }
    if (!options.camera_ids.empty() && options.camera_ids.size() != options.video_paths.size()) {
        std::cerr << "--camera-ids must match the number of videos." << std::endl;
        return false;
    }
    if (options.camera_ids.empty()) {
        for (size_t index = 0; index < options.video_paths.size(); ++index) {
            options.camera_ids.push_back("cam" + std::to_string(index + 1u));
        }
    }
    options.modnet_model_path = ResolveExistingPath(options.modnet_model_path).string();
    options.smpl_model_path = ResolveExistingPath(options.smpl_model_path).string();
    options.pear_model_path = ResolveExistingPath(options.pear_model_path).string();
    options.rtmpose_model_path = ResolveExistingPath(options.rtmpose_model_path).string();
    options.yolo_model_path = ResolveExistingPath(options.yolo_model_path).string();
    if (options.modnet_model_path.empty()) {
        std::cerr << "Error: --modnet <model.onnx> is strictly required for background extraction." << std::endl;
        PrintUsage();
        return false;
    }
    if (!std::filesystem::exists(options.modnet_model_path)) {
        std::cerr << "Error: MODNet model file not found: " << options.modnet_model_path << std::endl;
        return false;
    }
    if (!options.calibration_dir.empty() && !std::filesystem::exists(options.calibration_dir)) {
        std::cerr << "Error: calibration directory not found: "
                  << options.calibration_dir << std::endl;
        return false;
    }
    if (options.crop_resolution <= 0) {
        std::cerr << "Error: --crop-res must be positive." << std::endl;
        return false;
    }
    if (!(options.crop_margin > 0.0f)) {
        std::cerr << "Error: --crop-margin must be positive." << std::endl;
        return false;
    }
    const bool yolo_required = options.smpl_enabled || !options.calibration_dir.empty();
    if (options.smpl_enabled && options.smpl_use_pear) {
        if (options.pear_model_path.empty() ||
            !std::filesystem::exists(options.pear_model_path)) {
            std::cerr << "Error: PEAR model file not found: "
                      << options.pear_model_path << std::endl;
            return false;
        }
        if (!options.rtmpose_model_path.empty() &&
            !std::filesystem::exists(options.rtmpose_model_path)) {
            std::cerr << "Warning: RTMPose model file not found, falling back to projected mocap "
                         "keypoints for PEAR view scoring: "
                      << options.rtmpose_model_path << std::endl;
            options.rtmpose_model_path.clear();
        }
    }
    if (yolo_required) {
        if (options.yolo_model_path.empty() ||
            !std::filesystem::exists(options.yolo_model_path)) {
            std::cerr << "Error: YOLO model file not found: "
                      << options.yolo_model_path << std::endl;
            return false;
        }
        if (!(options.yolo_conf_threshold >= 0.0f) || !(options.yolo_conf_threshold <= 1.0f)) {
            std::cerr << "Error: --yolo-conf-threshold must be in [0, 1]." << std::endl;
            return false;
        }
        if (!(options.yolo_nms_threshold >= 0.0f) || !(options.yolo_nms_threshold <= 1.0f)) {
            std::cerr << "Error: --yolo-nms-threshold must be in [0, 1]." << std::endl;
            return false;
        }
    }

    *out_options = std::move(options);
    return true;
}

const CameraCalibration* FindCalibrationBySourceIndex(
    const std::vector<CameraCalibration>& calibrations,
    int source_camera_index) {
    for (const auto& calibration : calibrations) {
        if (calibration.source_camera_index == source_camera_index) {
            return &calibration;
        }
    }
    return nullptr;
}

cv::Mat CropSquareWithPaddingAndResize(const cv::Mat& src,
                                       int x,
                                       int y,
                                       int size,
                                       int target_res,
                                       int interpolation) {
    if (src.empty() || size <= 0 || target_res <= 0) {
        return {};
    }

    const cv::Rect roi(x, y, size, size);
    const cv::Rect image_bounds(0, 0, src.cols, src.rows);
    const cv::Rect valid_roi = roi & image_bounds;

    cv::Mat square = cv::Mat::zeros(size, size, src.type());
    if (valid_roi.area() > 0) {
        const cv::Rect dst_roi(valid_roi.x - roi.x,
                               valid_roi.y - roi.y,
                               valid_roi.width,
                               valid_roi.height);
        src(valid_roi).copyTo(square(dst_roi));
    }

    if (size == target_res) {
        return square;
    }

    cv::Mat resized;
    cv::resize(square,
               resized,
               cv::Size(target_res, target_res),
               0.0,
               0.0,
               interpolation);
    return resized;
}

constexpr float kMinProjectedDepth = 0.1f;

struct ProjectedPersonCrop {
    cv::Matx33f K = cv::Matx33f::eye();
    float img_w = 0.0f;
    float img_h = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float bbox_x = 0.0f;
    float bbox_y = 0.0f;
    float bbox_w = 0.0f;
    float bbox_h = 0.0f;
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    int roi_x = 0;
    int roi_y = 0;
    int roi_size = 0;
    float detection_score = 0.0f;
    std::array<cv::Point2f, kMocapJointCount> projected_joints{};
    std::array<float, kMocapJointCount> joint_scores{};
};

bool BuildProjectedPersonCrop(const SyncedView& view,
                              const MocapPerson3D& person,
                              const CameraCalibration& calibration,
                              float crop_margin,
                              ProjectedPersonCrop* out_crop);

bool IsValidCrop(const ProjectedPersonCrop& crop) {
    return crop.roi_size > 0 && crop.crop_size > 0.0f;
}

size_t CountValidMatchedCrops(const std::vector<std::vector<ProjectedPersonCrop>>& matched_crops) {
    size_t count = 0u;
    for (const auto& view_crops : matched_crops) {
        for (const auto& crop : view_crops) {
            if (IsValidCrop(crop)) {
                ++count;
            }
        }
    }
    return count;
}

int CountValidSmplFits(const MocapFrame3D& pose3d) {
    int count = 0;
    for (const auto& person : pose3d.people) {
        if (person.smpl_valid) {
            ++count;
        }
    }
    return count;
}

cv::Rect2f CropBoundingBoxRect(const ProjectedPersonCrop& crop) {
    return cv::Rect2f(crop.bbox_x, crop.bbox_y, crop.bbox_w, crop.bbox_h);
}

float ComputeIntersectionOverUnion(const cv::Rect2f& left, const cv::Rect2f& right) {
    const float x0 = std::max(left.x, right.x);
    const float y0 = std::max(left.y, right.y);
    const float x1 = std::min(left.x + left.width, right.x + right.width);
    const float y1 = std::min(left.y + left.height, right.y + right.height);
    const float intersection_w = std::max(0.0f, x1 - x0);
    const float intersection_h = std::max(0.0f, y1 - y0);
    const float intersection_area = intersection_w * intersection_h;
    const float union_area = left.area() + right.area() - intersection_area;
    if (!(union_area > 0.0f)) {
        return 0.0f;
    }
    return intersection_area / union_area;
}

bool BuildYoloCropFromDetection(const ProjectedPersonCrop& projected_crop,
                                const cv::Rect2f& detection_bbox,
                                float detection_score,
                                float crop_margin,
                                ProjectedPersonCrop* out_crop) {
    if (out_crop == nullptr || !(detection_bbox.width > 1.0f) || !(detection_bbox.height > 1.0f)) {
        return false;
    }

    ProjectedPersonCrop crop = projected_crop;
    crop.bbox_x = detection_bbox.x;
    crop.bbox_y = detection_bbox.y;
    crop.bbox_w = detection_bbox.width;
    crop.bbox_h = detection_bbox.height;
    crop.crop_cx = detection_bbox.x + detection_bbox.width * 0.5f;
    crop.crop_cy = detection_bbox.y + detection_bbox.height * 0.5f;
    crop.crop_size = std::max(1.0f, std::max(detection_bbox.width, detection_bbox.height) * crop_margin);
    crop.roi_size = std::max(1, static_cast<int>(std::ceil(crop.crop_size)));
    crop.roi_x = static_cast<int>(std::floor(crop.crop_cx - crop.crop_size * 0.5f));
    crop.roi_y = static_cast<int>(std::floor(crop.crop_cy - crop.crop_size * 0.5f));
    crop.detection_score = detection_score;

    *out_crop = crop;
    return true;
}

float ScoreYoloMatch(const ProjectedPersonCrop& projected_crop, const YoloDetection& detection) {
    const cv::Rect2f projected_bbox = CropBoundingBoxRect(projected_crop);
    if (!(projected_bbox.width > 1.0f) || !(projected_bbox.height > 1.0f) ||
        !(detection.bbox.width > 1.0f) || !(detection.bbox.height > 1.0f)) {
        return -std::numeric_limits<float>::infinity();
    }

    const float iou = ComputeIntersectionOverUnion(projected_bbox, detection.bbox);
    const cv::Point2f projected_center(projected_crop.crop_cx, projected_crop.crop_cy);
    const cv::Point2f detection_center(
        detection.bbox.x + detection.bbox.width * 0.5f,
        detection.bbox.y + detection.bbox.height * 0.5f);
    const float center_distance = cv::norm(projected_center - detection_center);
    const float distance_scale = std::max(1.0f, std::max(projected_bbox.width, projected_bbox.height));
    const float normalized_distance = center_distance / distance_scale;
    const bool center_inside = detection.bbox.contains(projected_center);

    if (iou <= 0.0f && !center_inside && normalized_distance > 1.75f) {
        return -std::numeric_limits<float>::infinity();
    }

    return iou * 4.0f +
           (center_inside ? 1.5f : 0.0f) +
           detection.score * 0.5f -
           normalized_distance;
}

std::vector<YoloDetection> DetectPeopleInView(YoloPersonDetector& yolo_detector, const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    try {
        cv::Rect2f ignored_bbox;
        float ignored_score = 0.0f;
        yolo_detector.DetectPerson(image, &ignored_bbox, &ignored_score);
        return yolo_detector.last_detections();
    } catch (const Ort::Exception& e) {
        std::cerr << "DatasetPrep: YOLO inference failed: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "DatasetPrep: YOLO inference failed: " << e.what() << std::endl;
        return {};
    }
}

bool BuildCropFromYoloOnly(const SyncedView& view,
                           const YoloDetection& detection,
                           const CameraCalibration& calibration,
                           float crop_margin,
                           ProjectedPersonCrop* out_crop) {
    if (out_crop == nullptr || view.image.empty() ||
        !(detection.bbox.width > 1.0f) || !(detection.bbox.height > 1.0f)) {
        return false;
    }

    ProjectedPersonCrop crop;
    crop.img_w = static_cast<float>(view.image.cols);
    crop.img_h = static_cast<float>(view.image.rows);

    const float scale_x = calibration.image_width > 0
        ? crop.img_w / static_cast<float>(calibration.image_width)
        : 1.0f;
    const float scale_y = calibration.image_height > 0
        ? crop.img_h / static_cast<float>(calibration.image_height)
        : 1.0f;
    crop.fx = calibration.K(0, 0) * scale_x;
    crop.fy = calibration.K(1, 1) * scale_y;
    crop.cx = calibration.K(0, 2) * scale_x;
    crop.cy = calibration.K(1, 2) * scale_y;
    crop.K = cv::Matx33f(crop.fx, 0.0f, crop.cx,
                         0.0f, crop.fy, crop.cy,
                         0.0f, 0.0f, 1.0f);

    crop.bbox_x = detection.bbox.x;
    crop.bbox_y = detection.bbox.y;
    crop.bbox_w = detection.bbox.width;
    crop.bbox_h = detection.bbox.height;
    crop.crop_size = std::max(1.0f, std::max(crop.bbox_w, crop.bbox_h) * crop_margin);
    crop.crop_cx = crop.bbox_x + crop.bbox_w * 0.5f;
    crop.crop_cy = crop.bbox_y + crop.bbox_h * 0.5f;
    crop.roi_size = std::max(1, static_cast<int>(std::ceil(crop.crop_size)));
    crop.roi_x = static_cast<int>(std::floor(crop.crop_cx - crop.crop_size * 0.5f));
    crop.roi_y = static_cast<int>(std::floor(crop.crop_cy - crop.crop_size * 0.5f));
    crop.detection_score = detection.score;

    *out_crop = crop;
    return true;
}

std::vector<std::vector<ProjectedPersonCrop>> BuildMatchedYoloCrops(
    const SyncedFrameCollection& synced_frames,
    const MocapFrame3D& pose3d,
    const std::vector<CameraCalibration>& calibrations,
    YoloPersonDetector* yolo_detector,
    float crop_margin) {
    std::vector<std::vector<ProjectedPersonCrop>> matched_crops(
        synced_frames.views.size(),
        std::vector<ProjectedPersonCrop>(pose3d.people.size()));
    if (yolo_detector == nullptr) {
        return matched_crops;
    }

    struct CandidateAssignment {
        float score = -std::numeric_limits<float>::infinity();
        size_t person_index = 0u;
        size_t detection_index = 0u;
        ProjectedPersonCrop crop;
    };

    for (size_t view_index = 0; view_index < synced_frames.views.size(); ++view_index) {
        const auto& view = synced_frames.views[view_index];
        const auto* calibration =
            FindCalibrationBySourceIndex(calibrations, view.source_camera_index);
        if (calibration == nullptr) {
            continue;
        }

        const std::vector<YoloDetection> detections = DetectPeopleInView(*yolo_detector, view.image);
        if (detections.empty()) {
            continue;
        }

        std::vector<CandidateAssignment> candidates;
        candidates.reserve(pose3d.people.size() * detections.size());
        for (size_t person_index = 0; person_index < pose3d.people.size(); ++person_index) {
            ProjectedPersonCrop projected_crop;
            if (!BuildProjectedPersonCrop(
                    view, pose3d.people[person_index], *calibration, crop_margin, &projected_crop)) {
                continue;
            }

            for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
                const float score = ScoreYoloMatch(projected_crop, detections[detection_index]);
                if (!std::isfinite(score)) {
                    continue;
                }

                ProjectedPersonCrop matched_crop;
                if (!BuildYoloCropFromDetection(projected_crop,
                                                detections[detection_index].bbox,
                                                detections[detection_index].score,
                                                crop_margin,
                                                &matched_crop)) {
                    continue;
                }

                CandidateAssignment candidate;
                candidate.score = score;
                candidate.person_index = person_index;
                candidate.detection_index = detection_index;
                candidate.crop = std::move(matched_crop);
                candidates.push_back(std::move(candidate));
            }
        }

        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const CandidateAssignment& left, const CandidateAssignment& right) {
                      return left.score > right.score;
                  });

        std::vector<uint8_t> used_people(pose3d.people.size(), 0u);
        std::vector<uint8_t> used_detections(detections.size(), 0u);
        for (const auto& candidate : candidates) {
            if (used_people[candidate.person_index] || used_detections[candidate.detection_index]) {
                continue;
            }
            matched_crops[view_index][candidate.person_index] = candidate.crop;
            used_people[candidate.person_index] = 1u;
            used_detections[candidate.detection_index] = 1u;
        }
    }

    return matched_crops;
}

bool BuildProjectedPersonCrop(const SyncedView& view,
                              const MocapPerson3D& person,
                              const CameraCalibration& calibration,
                              float crop_margin,
                              ProjectedPersonCrop* out_crop) {
    if (out_crop == nullptr || view.image.empty()) {
        return false;
    }

    ProjectedPersonCrop crop;
    crop.img_w = static_cast<float>(view.image.cols);
    crop.img_h = static_cast<float>(view.image.rows);
    if (!(crop.img_w > 0.0f) || !(crop.img_h > 0.0f)) {
        return false;
    }

    const float scale_x =
        calibration.image_width > 0 ? crop.img_w / static_cast<float>(calibration.image_width) : 1.0f;
    const float scale_y =
        calibration.image_height > 0 ? crop.img_h / static_cast<float>(calibration.image_height) : 1.0f;
    crop.fx = calibration.K(0, 0) * scale_x;
    crop.fy = calibration.K(1, 1) * scale_y;
    crop.cx = calibration.K(0, 2) * scale_x;
    crop.cy = calibration.K(1, 2) * scale_y;
    crop.K = cv::Matx33f(
        crop.fx, 0.0f, crop.cx,
        0.0f, crop.fy, crop.cy,
        0.0f, 0.0f, 1.0f);
    if (!(crop.fx > 0.0f) || !(crop.fy > 0.0f)) {
        return false;
    }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    bool has_projected_joint = false;

    for (size_t joint_index = 0; joint_index < person.joints.size(); ++joint_index) {
        const auto& joint = person.joints[joint_index];
        if (!joint.IsValid()) {
            continue;
        }

        const cv::Vec3f joint_global(joint.xyz.x, joint.xyz.y, joint.xyz.z);
        const cv::Vec3f joint_local = calibration.R * joint_global + calibration.t;
        if (!(joint_local[2] > kMinProjectedDepth)) {
            continue;
        }

        const float u = (joint_local[0] * crop.fx / joint_local[2]) + crop.cx;
        const float v = (joint_local[1] * crop.fy / joint_local[2]) + crop.cy;
        crop.projected_joints[joint_index] = cv::Point2f(u, v);
        crop.joint_scores[joint_index] = joint.confidence;

        min_x = std::min(min_x, u);
        min_y = std::min(min_y, v);
        max_x = std::max(max_x, u);
        max_y = std::max(max_y, v);
        has_projected_joint = true;
    }

    if (!has_projected_joint) {
        return false;
    }

    const float bbox_w = max_x - min_x;
    const float bbox_h = max_y - min_y;
    if (!(bbox_w > 1.0f) || !(bbox_h > 1.0f)) {
        return false;
    }

    crop.bbox_x = min_x;
    crop.bbox_y = min_y;
    crop.bbox_w = bbox_w;
    crop.bbox_h = bbox_h;
    crop.crop_size = std::max(1.0f, std::max(bbox_w, bbox_h) * crop_margin);
    crop.crop_cx = (min_x + max_x) * 0.5f;
    crop.crop_cy = (min_y + max_y) * 0.5f;
    crop.roi_size = std::max(1, static_cast<int>(std::ceil(crop.crop_size)));
    crop.roi_x = static_cast<int>(std::floor(crop.crop_cx - crop.crop_size * 0.5f));
    crop.roi_y = static_cast<int>(std::floor(crop.crop_cy - crop.crop_size * 0.5f));

    *out_crop = crop;
    return true;
}

#if DATASET_PREP_HAS_SMPL
float MeanPositiveScore(const std::vector<float>& scores) {
    float sum = 0.0f;
    int count = 0;
    for (float score : scores) {
        if (score > 0.0f && std::isfinite(score)) {
            sum += score;
            ++count;
        }
    }
    return count > 0 ? (sum / static_cast<float>(count)) : 0.0f;
}

int CountScoresAbove(const std::vector<float>& scores, float threshold) {
    int count = 0;
    for (float score : scores) {
        if (std::isfinite(score) && score >= threshold) {
            ++count;
        }
    }
    return count;
}

cv::Vec3f CameraTranslationToWorld(const cv::Vec3f& translation_camera,
                                   const CameraCalibration& calibration) {
    return calibration.R.t() * (translation_camera - calibration.t);
}

bool EstimatePersonRootWorldTranslation(const MocapPerson3D& person,
                                        cv::Vec3f* out_translation_world) {
    if (out_translation_world == nullptr) {
        return false;
    }

    if (person.joints[19].IsValid()) {
        *out_translation_world = cv::Vec3f(person.joints[19].xyz.x,
                                           person.joints[19].xyz.y,
                                           person.joints[19].xyz.z);
        return true;
    }
    if (person.joints[11].IsValid() && person.joints[12].IsValid()) {
        *out_translation_world = cv::Vec3f(
            0.5f * (person.joints[11].xyz.x + person.joints[12].xyz.x),
            0.5f * (person.joints[11].xyz.y + person.joints[12].xyz.y),
            0.5f * (person.joints[11].xyz.z + person.joints[12].xyz.z));
        return true;
    }

    cv::Vec3f sum(0.0f, 0.0f, 0.0f);
    int count = 0;
    for (const auto& joint : person.joints) {
        if (!joint.IsValid(0.01f)) {
            continue;
        }
        sum += cv::Vec3f(joint.xyz.x, joint.xyz.y, joint.xyz.z);
        ++count;
    }
    if (count <= 0) {
        return false;
    }

    *out_translation_world = sum * (1.0f / static_cast<float>(count));
    return true;
}

bool ConvertPearTranslationToWorld(const SmplxResult& pear_result,
                                   const ProjectedPersonCrop& matched_crop,
                                   const CameraCalibration& calibration,
                                   cv::Vec3f* out_translation_world) {
    if (out_translation_world == nullptr || pear_result.camera_translation.size() < 3u) {
        return false;
    }

    // 1. Map PyTorch3D to OpenCV: Invert X and Y components
    cv::Vec3f translation_camera(
        -pear_result.camera_translation[0], 
        -pear_result.camera_translation[1], 
        pear_result.camera_translation[2]);

    // 2. Scale virtual Z depth to real camera metric depth
    const float f_virtual = 5000.0f;
    const float virtual_crop_size = 256.0f;
    const float focal_length = (matched_crop.fx + matched_crop.fy) * 0.5f;
    
    if (matched_crop.crop_size > 0.0f) {
        translation_camera[2] = translation_camera[2] * (focal_length / f_virtual) * (virtual_crop_size / matched_crop.crop_size);
    }

    if (!std::isfinite(translation_camera[0]) || !std::isfinite(translation_camera[1]) || 
        !std::isfinite(translation_camera[2]) || !(translation_camera[2] > 0.0f)) {
        return false;
    }

    // Note: We intentionally skip the Principal Point Shift here. 
    // This keeps the mesh centered in the crop for perfect "Warp Back" alignment.
    *out_translation_world = CameraTranslationToWorld(translation_camera, calibration);
    
    return std::isfinite((*out_translation_world)[0]) &&
           std::isfinite((*out_translation_world)[1]) &&
           std::isfinite((*out_translation_world)[2]);
}

std::vector<float> ConvertPearPoseToSmplAxisAngle(const SmplxResult& pear_result) {
    if (pear_result.global_orient.size() != 3u || pear_result.body_pose.size() < 63u) {
        return {};
    }

    // dataset_prep still optimizes against the legacy SMPL layer, so PEAR's extra
    // SMPL-X hand and face outputs are intentionally dropped here for now.
    std::vector<float> pose_values(kSmplPoseParamCount, 0.0f);
    std::copy_n(pear_result.global_orient.begin(), 3u, pose_values.begin());
    std::copy_n(pear_result.body_pose.begin(), 63u, pose_values.begin() + 3);
    return pose_values;
}

bool ConvertPearPoseToWorld(const SmplxResult& pear_result,
                            const CameraCalibration& calibration,
                            std::vector<float>* out_pose_world) {
    if (out_pose_world == nullptr) {
        return false;
    }

    std::vector<float> pose_world = ConvertPearPoseToSmplAxisAngle(pear_result);
    if (pose_world.size() != kSmplPoseParamCount) {
        return false;
    }

    cv::Vec3f root_local(pose_world[0], pose_world[1], pose_world[2]);
    cv::Matx33f root_local_rotation;
    cv::Rodrigues(root_local, root_local_rotation);

    // FLIP ORIENTATION: 180-degree rotation around Z-axis
    cv::Matx33f Rz_180(
        -1.0f,  0.0f,  0.0f,
         0.0f, -1.0f,  0.0f,
         0.0f,  0.0f,  1.0f
    );
    root_local_rotation = Rz_180 * root_local_rotation;

    const cv::Matx33f root_world_rotation = calibration.R.t() * root_local_rotation;
    cv::Vec3f root_world;
    cv::Rodrigues(root_world_rotation, root_world);
    pose_world[0] = root_world[0];
    pose_world[1] = root_world[1];
    pose_world[2] = root_world[2];

    *out_pose_world = std::move(pose_world);
    return true;
}

void ResetSmplFit(MocapPerson3D* person) {
    if (person == nullptr) {
        return;
    }

    person->smpl_valid = false;
    person->smpl_pose.assign(kSmplPoseParamCount, 0.0f);
    person->smpl_shape.assign(kSmplShapeParamCount, 0.0f);
    person->smpl_scale = std::numeric_limits<float>::quiet_NaN();
    person->smpl_translation = cv::Point3f(
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN());
}

struct PearViewEstimate {
    SmplifyMultiViewObservation observation;
    std::vector<float> pose_world;
    std::vector<float> betas;
    cv::Vec3f translation_world{0.0f, 0.0f, 0.0f};
    float score = 0.0f;
};

bool EstimatePersonSmplWithPear(const SyncedFrameCollection& synced_frames,
                                const std::vector<std::vector<ProjectedPersonCrop>>& matched_crops,
                                const std::vector<CameraCalibration>& calibrations,
                                PearEstimator& pear_estimator,
                                RtmPoseDetector* rtmpose_detector,
                                int crop_resolution,
                                int person_index,
                                MocapPerson3D* person) {
    if (person == nullptr) {
        return false;
    }

    std::vector<PearViewEstimate> views;
    views.reserve(synced_frames.views.size());
    int matched_view_count = 0;
    int pear_failure_count = 0;
    int pose_failure_count = 0;
    int translation_failure_count = 0;

    cv::Vec3f fallback_translation_world(0.0f, 0.0f, 0.0f);
    const bool have_fallback_translation =
        EstimatePersonRootWorldTranslation(*person, &fallback_translation_world);

    for (size_t view_index = 0; view_index < synced_frames.views.size(); ++view_index) {
        const auto& view = synced_frames.views[view_index];
        const auto* calibration =
            FindCalibrationBySourceIndex(calibrations, view.source_camera_index);
        if (calibration == nullptr) {
            continue;
        }
        if (view_index >= matched_crops.size() ||
            person_index < 0 ||
            static_cast<size_t>(person_index) >= matched_crops[view_index].size()) {
            continue;
        }
        const ProjectedPersonCrop& matched_crop = matched_crops[view_index][static_cast<size_t>(person_index)];
        if (!IsValidCrop(matched_crop)) {
            continue;
        }
        ++matched_view_count;

        const int interpolation =
            matched_crop.roi_size > crop_resolution ? cv::INTER_AREA : cv::INTER_CUBIC;
        cv::Mat crop_image = CropSquareWithPaddingAndResize(
            view.image,
            matched_crop.roi_x,
            matched_crop.roi_y,
            matched_crop.roi_size,
            crop_resolution,
            interpolation);
        if (crop_image.empty()) {
            continue;
        }

        SmplxResult pear_result;
        if (!pear_estimator.Estimate(crop_image, &pear_result) ||
            pear_result.betas.size() < kSmplShapeParamCount) {
            ++pear_failure_count;
            continue;
        }

        std::vector<float> pose_world;
        if (!ConvertPearPoseToWorld(pear_result, *calibration, &pose_world)) {
            ++pose_failure_count;
            continue;
        }

        std::vector<cv::Point2f> observation_keypoints;
        std::vector<float> observation_scores;
        bool have_rtmpose_observation = false;
        if (rtmpose_detector != nullptr) {
            std::vector<cv::Point2f> crop_keypoints;
            std::vector<float> crop_scores;
            if (rtmpose_detector->DetectPose(
                    crop_image,
                    &crop_keypoints,
                    &crop_scores,
                    view.video_frame_index) &&
                crop_keypoints.size() == crop_scores.size()) {
                observation_keypoints.reserve(crop_keypoints.size());
                observation_scores.reserve(crop_scores.size());
                const float crop_to_full_scale =
                    static_cast<float>(matched_crop.roi_size) /
                    static_cast<float>(crop_resolution);
                for (size_t index = 0; index < crop_keypoints.size(); ++index) {
                    observation_keypoints.emplace_back(
                        static_cast<float>(matched_crop.roi_x) +
                            crop_keypoints[index].x * crop_to_full_scale,
                        static_cast<float>(matched_crop.roi_y) +
                            crop_keypoints[index].y * crop_to_full_scale);
                    observation_scores.push_back(crop_scores[index]);
                }
                have_rtmpose_observation = CountScoresAbove(observation_scores, 0.01f) >= 6;
            }
        }

        bool use_detection_score = false;
        if (!have_rtmpose_observation) {
            bool has_mocap = false;
            for (float score : matched_crop.joint_scores) {
                if (score > 0.0f) {
                    has_mocap = true;
                    break;
                }
            }
            if (has_mocap) {
                observation_keypoints.assign(matched_crop.projected_joints.begin(),
                                             matched_crop.projected_joints.end());
                observation_scores.assign(matched_crop.joint_scores.begin(),
                                          matched_crop.joint_scores.end());
            } else {
                use_detection_score = true;
            }
        }

        cv::Vec3f translation_world(0.0f, 0.0f, 0.0f);
        if (!ConvertPearTranslationToWorld(pear_result, matched_crop, *calibration, &translation_world)) {
            if (!have_fallback_translation) {
                ++translation_failure_count;
                continue;
            }
            translation_world = fallback_translation_world;
        }

        PearViewEstimate estimate;
        estimate.observation.keypoints = std::move(observation_keypoints);
        estimate.observation.keypoint_scores = std::move(observation_scores);
        estimate.observation.K = matched_crop.K;
        estimate.observation.R = calibration->R;
        estimate.observation.t = calibration->t;
        estimate.observation.img_w = matched_crop.img_w;
        estimate.observation.img_h = matched_crop.img_h;
        estimate.pose_world = std::move(pose_world);
        estimate.betas.assign(pear_result.betas.begin(),
                              pear_result.betas.begin() + kSmplShapeParamCount);
        estimate.translation_world = translation_world;
        estimate.score = use_detection_score
            ? matched_crop.detection_score
            : static_cast<float>(CountScoresAbove(estimate.observation.keypoint_scores, 0.01f)) +
                MeanPositiveScore(estimate.observation.keypoint_scores);
        views.push_back(std::move(estimate));
    }

    if (views.empty()) {
        if (matched_view_count > 0) {
            std::cerr << "DatasetPrep: PEAR produced no usable view seed for person "
                      << person_index
                      << " (matched_views=" << matched_view_count
                      << ", pear_failures=" << pear_failure_count
                      << ", pose_failures=" << pose_failure_count
                      << ", translation_failures=" << translation_failure_count
                      << ")" << std::endl;
        }
        return false;
    }

    const auto best_view_it = std::max_element(
        views.begin(),
        views.end(),
        [](const PearViewEstimate& left, const PearViewEstimate& right) {
            return left.score < right.score;
        });
    if (best_view_it == views.end()) {
        return false;
    }

    if (best_view_it->pose_world.size() != kSmplPoseParamCount ||
        best_view_it->betas.size() < kSmplShapeParamCount) {
        return false;
    }

    person->smpl_pose = best_view_it->pose_world;
    person->smpl_shape.assign(best_view_it->betas.begin(),
                              best_view_it->betas.begin() + kSmplShapeParamCount);
    person->smpl_translation = cv::Point3f(
        best_view_it->translation_world[0],
        best_view_it->translation_world[1],
        best_view_it->translation_world[2]);
    person->smpl_scale = 1.0f;
    person->smpl_valid = true;
    return true;
}

bool BuildTrainingDebugOverlay(SMPLLayer& smpl_layer,
                               const MocapPerson3D& person,
                               const CameraCalibration& calibration,
                               ExportTrainingSample* sample) {
    if (sample == nullptr || sample->crop_image.empty() ||
        sample->pose.size() < kSmplPoseParamCount ||
        sample->betas.size() < kSmplShapeParamCount ||
        sample->cam.size() < 3u) {
        return false;
    }

    cv::Mat overlay = sample->crop_image.clone();

    try {
        torch::NoGradGuard no_grad;
        const auto device = smpl_layer.v_template.device();
        const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        auto pose_tensor = torch::from_blob(sample->pose.data(), {1, 24, 3}, torch::kFloat32)
                               .clone()
                               .to(device);
        auto betas_tensor = torch::from_blob(sample->betas.data(), {1, 10}, torch::kFloat32)
                                .clone()
                                .to(device);
        auto trans_zeros = torch::zeros({1, 3}, tensor_options);

        const auto smpl_out = smpl_layer.forward(betas_tensor, pose_tensor, trans_zeros);
        const auto verts_cpu = smpl_out.vertices.squeeze(0)
                                   .detach()
                                   .to(torch::kCPU)
                                   .to(torch::kFloat32)
                                   .contiguous()
                                   .clone();
        const auto trans = EstimateTranslation(
            sample->cam,
            sample->crop_cx,
            sample->crop_cy,
            sample->crop_size,
            sample->focal_length,
            sample->img_w,
            sample->img_h);

        float x0 = sample->crop_x0;
        float y0 = sample->crop_y0;
        if (!(sample->crop_w > 0.0f) || !(sample->crop_h > 0.0f)) {
            x0 = sample->crop_cx - static_cast<float>(overlay.cols) * 0.5f;
            y0 = sample->crop_cy - static_cast<float>(overlay.rows) * 0.5f;
        }
        const float cx_crop = sample->img_w * 0.5f - x0;
        const float cy_crop = sample->img_h * 0.5f - y0;
        int projected_target_joints = 0;

        const int current_visible = CountProjectedInFramePinhole(
            verts_cpu, trans, sample->y_sign, sample->focal_length,
            cx_crop, cy_crop, overlay.cols, overlay.rows);
        const int pos_visible = CountProjectedInFramePinhole(
            verts_cpu, trans, 1.0f, sample->focal_length,
            cx_crop, cy_crop, overlay.cols, overlay.rows);
        const int neg_visible = CountProjectedInFramePinhole(
            verts_cpu, trans, -1.0f, sample->focal_length,
            cx_crop, cy_crop, overlay.cols, overlay.rows);
        const float suggested_y_sign = (neg_visible > pos_visible) ? -1.0f : 1.0f;

        auto verts_acc = verts_cpu.accessor<float, 2>();
        std::vector<cv::Vec3f> camera_vertices;
        std::vector<cv::Point> projected_vertices;
        std::vector<uint8_t> vertex_valid;
        camera_vertices.resize(static_cast<size_t>(verts_acc.size(0)));
        projected_vertices.resize(static_cast<size_t>(verts_acc.size(0)));
        vertex_valid.assign(static_cast<size_t>(verts_acc.size(0)), 0u);
 
        for (int vertex_index = 0; vertex_index < verts_acc.size(0); ++vertex_index) {
            const float f_virt = 5000.0f;
            const float c_virt = 128.0f; // Center of 256x256 model space
            
            // Reconstruct the model-space translation using inverse scaling
            const float Z_virt = trans[2] * (f_virt / sample->focal_length) * (sample->crop_size / 256.0f);
            const float X_virt = trans[0] * (f_virt / sample->focal_length);
            const float Y_virt = trans[1] * (f_virt / sample->focal_length);

            // X, Y, Z are now in the model's native camera space
            const float X = verts_acc[vertex_index][0] + X_virt;
            const float Y = verts_acc[vertex_index][1] * sample->y_sign + Y_virt;
            const float Z = verts_acc[vertex_index][2] + Z_virt;
            
            camera_vertices[static_cast<size_t>(vertex_index)] = cv::Vec3f(X, Y, Z);
            if (!(Z > 1e-6f)) continue;

            // Project into 256x256 Model Space
            const float u_model = (f_virt * X / Z) + c_virt;
            const float v_model = (f_virt * Y / Z) + c_virt;
            
            // Map 256x256 back to output crop resolution (e.g., 512x512)
            const float scale_to_output = sample->crop_w / 256.0f;
            const float u_crop = u_model * scale_to_output;
            const float v_crop = v_model * scale_to_output;

            projected_vertices[static_cast<size_t>(vertex_index)] = cv::Point(
                static_cast<int>(std::lround(u_crop)),
                static_cast<int>(std::lround(v_crop)));
            
            vertex_valid[static_cast<size_t>(vertex_index)] = 
                (u_crop >= 0 && v_crop >= 0 && u_crop < overlay.cols && v_crop < overlay.rows);
        }

        const auto faces_cpu = smpl_layer.faces.detach()
                                   .to(torch::kCPU)
                                   .to(torch::kLong)
                                   .contiguous()
                                   .clone();
        if (faces_cpu.defined() &&
            faces_cpu.dim() == 2 &&
            faces_cpu.size(1) >= 3 &&
            faces_cpu.numel() > 0) {
            struct ProjectedTriangle {
                float depth = 0.0f;
                cv::Point points[3];
                cv::Scalar color;
            };

            std::vector<ProjectedTriangle> triangles;
            const int64_t* faces_ptr = faces_cpu.data_ptr<int64_t>();
            const int64_t face_count = faces_cpu.size(0);
            triangles.reserve(static_cast<size_t>(face_count));

            for (int64_t face_index = 0; face_index < face_count; ++face_index) {
                const int i0 = static_cast<int>(faces_ptr[face_index * 3 + 0]);
                const int i1 = static_cast<int>(faces_ptr[face_index * 3 + 1]);
                const int i2 = static_cast<int>(faces_ptr[face_index * 3 + 2]);
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= verts_acc.size(0) ||
                    i1 >= verts_acc.size(0) ||
                    i2 >= verts_acc.size(0)) {
                    continue;
                }

                const cv::Vec3f& p0 = camera_vertices[static_cast<size_t>(i0)];
                const cv::Vec3f& p1 = camera_vertices[static_cast<size_t>(i1)];
                const cv::Vec3f& p2 = camera_vertices[static_cast<size_t>(i2)];
                if (!(p0[2] > 1e-6f) || !(p1[2] > 1e-6f) || !(p2[2] > 1e-6f)) {
                    continue;
                }
                if (!vertex_valid[static_cast<size_t>(i0)] &&
                    !vertex_valid[static_cast<size_t>(i1)] &&
                    !vertex_valid[static_cast<size_t>(i2)]) {
                    continue;
                }

                const cv::Vec3f edge01 = p1 - p0;
                const cv::Vec3f edge02 = p2 - p0;
                const cv::Vec3f normal = edge01.cross(edge02);
                const cv::Vec3f center = (p0 + p1 + p2) * (1.0f / 3.0f);
                const bool front_facing = normal.dot(-center) > 0.0f;

                ProjectedTriangle triangle;
                triangle.depth = (p0[2] + p1[2] + p2[2]) * (1.0f / 3.0f);
                triangle.points[0] = projected_vertices[static_cast<size_t>(i0)];
                triangle.points[1] = projected_vertices[static_cast<size_t>(i1)];
                triangle.points[2] = projected_vertices[static_cast<size_t>(i2)];
                triangle.color = front_facing
                    ? cv::Scalar(60, 200, 60)
                    : cv::Scalar(60, 60, 200);
                triangles.push_back(triangle);
            }

            std::sort(triangles.begin(),
                      triangles.end(),
                      [](const ProjectedTriangle& a, const ProjectedTriangle& b) {
                          return a.depth > b.depth;
                      });

            cv::Mat mesh_layer(overlay.size(), overlay.type(), cv::Scalar::all(0));
            cv::Mat coverage(overlay.size(), CV_8U, cv::Scalar(0));
            for (const auto& triangle : triangles) {
                cv::fillConvexPoly(mesh_layer, triangle.points, 3, triangle.color, cv::LINE_AA);
                cv::fillConvexPoly(coverage, triangle.points, 3, cv::Scalar(255), cv::LINE_AA);
                const cv::Point* contour = triangle.points;
                const int contour_size = 3;
                cv::polylines(mesh_layer,
                              &contour,
                              &contour_size,
                              1,
                              true,
                              cv::Scalar(20, 20, 20),
                              1,
                              cv::LINE_AA);
            }

            cv::Mat blended;
            cv::addWeighted(sample->crop_image, 0.55, mesh_layer, 0.45, 0.0, blended);
            blended.copyTo(overlay, coverage);
        } else {
            for (int vertex_index = 0; vertex_index < verts_acc.size(0); vertex_index += 4) {
                if (!vertex_valid[static_cast<size_t>(vertex_index)]) {
                    continue;
                }
                cv::circle(overlay,
                           projected_vertices[static_cast<size_t>(vertex_index)],
                           1,
                           cv::Scalar(0, 255, 0),
                           -1,
                           cv::LINE_AA);
            }
        }

        for (const auto& joint : person.joints) {
            if (!joint.IsValid()) {
                continue;
            }
            const cv::Vec3f joint_global(joint.xyz.x, joint.xyz.y, joint.xyz.z);
            const cv::Vec3f joint_local = calibration.R * joint_global + calibration.t;
            if (!(joint_local[2] > 1e-6f)) {
                continue;
            }

            const float u = (sample->focal_length * joint_local[0] / joint_local[2]) + cx_crop;
            const float v = (sample->focal_length * joint_local[1] / joint_local[2]) + cy_crop;
            if (u < 0.0f || v < 0.0f ||
                u >= static_cast<float>(overlay.cols) ||
                v >= static_cast<float>(overlay.rows)) {
                continue;
            }

            cv::circle(overlay,
                       cv::Point(static_cast<int>(std::lround(u)),
                                 static_cast<int>(std::lround(v))),
                       3,
                       cv::Scalar(0, 255, 255),
                       -1,
                       cv::LINE_AA);
            projected_target_joints++;
        }

        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.4;
        const int thickness = 1;
        const cv::Scalar text_color = suggested_y_sign == sample->y_sign
            ? cv::Scalar(0, 255, 255)
            : cv::Scalar(0, 128, 255);
        const cv::Scalar bg_color(0, 0, 0);

        std::vector<std::string> lines;
        {
            std::ostringstream line;
            line << std::fixed << std::setprecision(1)
                 << "y_sign=" << sample->y_sign
                 << " visible=" << current_visible;
            lines.push_back(line.str());
        }
        {
            std::ostringstream line;
            line << std::fixed << std::setprecision(1)
                 << "pos=" << pos_visible
                 << " neg=" << neg_visible
                 << " suggested=" << suggested_y_sign;
            lines.push_back(line.str());
        }
        {
            std::ostringstream line;
            line << "target_joints=" << projected_target_joints;
            lines.push_back(line.str());
        }
        if (std::isfinite(sample->smpl_scale) && sample->smpl_scale > 0.0f) {
            std::ostringstream line;
            line << std::fixed << std::setprecision(3)
                 << "scale=" << sample->smpl_scale;
            lines.push_back(line.str());
        }
        {
            std::ostringstream line;
            line << std::fixed << std::setprecision(3)
                 << "root=[" << sample->pose[0] << ", "
                 << sample->pose[1] << ", "
                 << sample->pose[2] << "]";
            lines.push_back(line.str());
        }

        int y = 18;
        for (const auto& line : lines) {
            int baseline = 0;
            const cv::Size size = cv::getTextSize(line, font, scale, thickness, &baseline);
            const cv::Point text_origin(8, y);
            cv::rectangle(overlay,
                          text_origin + cv::Point(-4, -size.height - 4),
                          text_origin + cv::Point(size.width + 4, 4),
                          bg_color,
                          cv::FILLED);
            cv::putText(overlay, line, text_origin, font, scale, text_color, thickness, cv::LINE_AA);
            y += size.height + 10;
        }

        sample->crop_overlay = std::move(overlay);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "DatasetPrep: failed to build training debug overlay for "
                  << sample->camera_id << " frame " << sample->video_frame_index
                  << ": " << e.what() << std::endl;
        return false;
    }
}
#endif

bool BuildTrainingSample(const SyncedView& view,
                         BackgroundExtractor& background_extractor,
#if DATASET_PREP_HAS_SMPL
                         SMPLLayer* debug_smpl_layer,
#endif
                         const MocapPerson3D& person,
                         int person_index,
                         const CameraCalibration& calibration,
                         const ProjectedPersonCrop& matched_crop,
                         int crop_resolution,
                         ExportTrainingSample* out_sample) {
    if (out_sample == nullptr || view.image.empty() ||
        !person.smpl_valid || person.smpl_pose.size() < 3u ||
        !IsValidCrop(matched_crop)) {
        return false;
    }
    const float img_w = matched_crop.img_w;
    const float img_h = matched_crop.img_h;
    const float fx = matched_crop.fx;

    // 1. Calculate global orientation relative to camera
    cv::Vec3f global_orient(person.smpl_pose[0], person.smpl_pose[1], person.smpl_pose[2]);
    cv::Matx33f smpl_global_rotation;
    cv::Rodrigues(global_orient, smpl_global_rotation);

    const cv::Matx33f smpl_local_rotation = calibration.R * smpl_global_rotation;
    cv::Vec3f local_orient;
    cv::Rodrigues(smpl_local_rotation, local_orient);

    // 2. Calculate camera-space translation
    const float mocap_scale = (std::isfinite(person.smpl_scale) && person.smpl_scale > 0.0f) ? person.smpl_scale : 1.0f;
    const cv::Vec3f root_global(person.smpl_translation.x, person.smpl_translation.y, person.smpl_translation.z);
    
    // Scale extrinsics for metric parity
    const cv::Vec3f calibration_t_metric = calibration.t * mocap_scale;
    const cv::Vec3f root_local = calibration.R * root_global + calibration_t_metric;
    const float X = root_local[0];
    const float Y = root_local[1];
    const float Z = root_local[2];
    if (!(Z > kMinProjectedDepth)) {
        return false;
    }
    const float raw_crop_size = matched_crop.crop_size;
    const float raw_crop_cx = matched_crop.crop_cx;
    const float raw_crop_cy = matched_crop.crop_cy;
    const int roi_size = matched_crop.roi_size;
    const int roi_x = matched_crop.roi_x;
    const int roi_y = matched_crop.roi_y;
    const float virtual_size = 256.0f;
    const float scale_back = raw_crop_size / virtual_size;
    const float model_tx = raw_crop_cx - (virtual_size * 0.5f) * scale_back;
    const float model_ty = raw_crop_cy - (virtual_size * 0.5f) * scale_back;
    const cv::Matx23f model_to_full(
        scale_back, 0.0f, model_tx,
        0.0f, scale_back, model_ty);
    const int image_interpolation =
        roi_size > crop_resolution ? cv::INTER_AREA : cv::INTER_CUBIC;

    cv::Mat crop_image = CropSquareWithPaddingAndResize(
        view.image, roi_x, roi_y, roi_size, crop_resolution, image_interpolation);
    if (crop_image.empty()) {
        return false;
    }

    cv::Mat crop_matte;
    if (!background_extractor.ProcessImage(crop_image, &crop_matte)) {
        return false;
    }
    if (crop_matte.empty()) {
        return false;
    }

    const float resize_scale =
        static_cast<float>(crop_resolution) / static_cast<float>(roi_size);
    const float full_cx = img_w * 0.5f;
    const float full_cy = img_h * 0.5f;
    const float train_crop_cx = full_cx + (raw_crop_cx - full_cx) * resize_scale;
    const float train_crop_cy = full_cy + (raw_crop_cy - full_cy) * resize_scale;
    const float train_crop_size = raw_crop_size * resize_scale;
    const float train_crop_x0 =
        full_cx - (resize_scale * (full_cx - static_cast<float>(roi_x)));
    const float train_crop_y0 =
        full_cy - (resize_scale * (full_cy - static_cast<float>(roi_y)));
    const float train_focal = fx * resize_scale;
    if (!(train_crop_size > 0.0f) || !(train_focal > 0.0f)) {
        return false;
    }

    const float s = (2.0f * train_focal) / (Z * train_crop_size);
    const float tx = X - (Z * (train_crop_cx - full_cx)) / train_focal;
    const float ty = Y - (Z * (train_crop_cy - full_cy)) / train_focal;

    ExportTrainingSample sample;
    sample.camera_id = view.camera_id;
    sample.source_camera_index = view.source_camera_index;
    sample.video_frame_index = view.video_frame_index;
    sample.person_index = person_index;
    sample.person_id = person.person_id >= 0 ? person.person_id : person_index;
    sample.crop_image = std::move(crop_image);
    sample.crop_matte = std::move(crop_matte);
    sample.img_w = img_w;
    sample.img_h = img_h;
    sample.crop_cx = train_crop_cx;
    sample.crop_cy = train_crop_cy;
    sample.crop_size = train_crop_size;
    sample.crop_x0 = train_crop_x0;
    sample.crop_y0 = train_crop_y0;
    sample.crop_w = static_cast<float>(crop_resolution);
    sample.crop_h = static_cast<float>(crop_resolution);
    sample.focal_length = train_focal;
    sample.model_to_full = model_to_full;
    sample.y_sign = 1.0f;
    sample.smpl_scale = mocap_scale;
    sample.cam = {s, tx, ty};
    std::copy_n(person.smpl_pose.begin(),
                std::min(person.smpl_pose.size(), sample.pose.size()),
                sample.pose.begin());
    sample.pose[0] = local_orient[0];
    sample.pose[1] = local_orient[1];
    sample.pose[2] = local_orient[2];
    std::copy_n(person.smpl_shape.begin(),
                std::min(person.smpl_shape.size(), sample.betas.size()),
                sample.betas.begin());

#if DATASET_PREP_HAS_SMPL
    if (debug_smpl_layer != nullptr) {
        BuildTrainingDebugOverlay(*debug_smpl_layer, person, calibration, &sample);
    }
#endif

    *out_sample = std::move(sample);
    return true;
}

}  // namespace

}  // namespace dataset_prep

int main(int argc, char* argv[]) {
    using namespace dataset_prep;

    CliOptions options;
    if (!ParseArgs(argc, argv, &options)) {
        return 1;
    }

    MocapPoseParser parser;
    MocapPoseParser::ParseOptions parse_options;
    parse_options.timestamp_scale = options.pose_timestamp_scale;

    MocapSequence3D global_mocap;
    const bool use_mocap = !options.pose_3d_path.empty();
    if (use_mocap) {
        if (!parser.Parse(options.pose_3d_path, &global_mocap, parse_options)) {
            return 1;
        }
    }

    std::vector<VideoSourceConfig> video_sources;
    video_sources.reserve(options.video_paths.size());
    for (size_t index = 0; index < options.video_paths.size(); ++index) {
        VideoSourceConfig source;
        source.camera_id = options.camera_ids[index];
        source.source_camera_index = static_cast<int>(index);
        source.video_path = options.video_paths[index];
        source.frame_stride = options.frame_stride;
        video_sources.push_back(std::move(source));
    }

    std::vector<CameraCalibration> calibrations;
    if (!options.calibration_dir.empty()) {
        CameraCalibrationLoader calibration_loader;
        if (!calibration_loader.Load(options.calibration_dir, video_sources, &calibrations)) {
            return 1;
        }
    }

    VideoSynchronizer synchronizer(VideoSynchronizer::Options{options.sync_tolerance_ms});
    if (!synchronizer.Open(video_sources)) {
        return 1;
    }

    BackgroundExtractor background_extractor(BackgroundExtractor::Options{
        options.modnet_model_path,
        options.modnet_use_cuda,
        options.modnet_input_size,
        options.modnet_binary_threshold});
    if (!background_extractor.Initialize()) {
        return 1;
    }

    std::unique_ptr<YoloPersonDetector> yolo_detector;
#if DATASET_PREP_HAS_SMPL
    std::unique_ptr<SmplifyLiteMocapSolver> smpl_solver;
    std::unique_ptr<PearEstimator> pear_estimator;
    std::unique_ptr<RtmPoseDetector> rtmpose_detector;
    std::unique_ptr<SMPLLayer> debug_smpl_layer;
    if (options.smpl_enabled) {
        if (options.smpl_use_pear) {
            if (calibrations.empty()) {
                std::cerr << "DatasetPrep: PEAR-based fitting requires camera calibrations." << std::endl;
                return 1;
            }

            pear_estimator = std::make_unique<PearEstimator>(PearEstimator::Options{
                options.pear_model_path,
                options.pear_use_cuda});
            if (!pear_estimator->Initialize()) {
                std::cerr << "DatasetPrep: failed to initialize PEAR estimator using "
                          << options.pear_model_path << std::endl;
                return 1;
            }

            YoloPersonDetectorOptions yolo_options;
            yolo_options.conf_threshold = options.yolo_conf_threshold;
            yolo_options.nms_threshold = options.yolo_nms_threshold;
            yolo_options.use_cuda = options.yolo_use_cuda;
            yolo_detector = std::make_unique<YoloPersonDetector>(yolo_options);
            if (!yolo_detector->Load(options.yolo_model_path)) {
                std::cerr << "DatasetPrep: failed to initialize YOLO detector using "
                          << options.yolo_model_path << std::endl;
                return 1;
            }

            if (!options.rtmpose_model_path.empty()) {
                RtmPoseDetectorOptions pose_options;
                pose_options.use_cuda = options.rtmpose_use_cuda;
                rtmpose_detector = std::make_unique<RtmPoseDetector>(pose_options);
                if (!rtmpose_detector->Load(options.rtmpose_model_path)) {
                    std::cerr << "DatasetPrep: failed to initialize RTMPose from "
                              << options.rtmpose_model_path
                              << ", falling back to projected mocap keypoints."
                              << std::endl;
                    rtmpose_detector.reset();
                }
            }
        } else {
            SmplMocapFitOptions smpl_fit_options;
            smpl_fit_options.num_iters = options.smpl_iters;
            smpl_fit_options.use_cuda = options.smpl_use_cuda;
            smpl_solver = std::make_unique<SmplifyLiteMocapSolver>(
                options.smpl_model_path, smpl_fit_options);
            if (!smpl_solver->IsReady()) {
                std::cerr << "DatasetPrep: failed to initialize SMPL solver using "
                          << options.smpl_model_path << std::endl;
                return 1;
            }
        }

        if (options.save_training_debug) {
            try {
                debug_smpl_layer = std::make_unique<SMPLLayer>(options.smpl_model_path);
                torch::Device device(torch::kCPU);
                if (options.smpl_use_cuda && torch::cuda::is_available()) {
                    device = torch::Device(torch::kCUDA);
                }
                debug_smpl_layer->to(device);
                debug_smpl_layer->eval();
            } catch (const std::exception& e) {
                std::cerr << "DatasetPrep: failed to initialize training debug SMPL layer: "
                          << e.what() << std::endl;
                debug_smpl_layer.reset();
            }
        }
    }
#else
    if (options.smpl_enabled) {
        std::cerr << "DatasetPrep: this build was compiled without SMPL support." << std::endl;
        return 1;
    }
#endif

    if (!calibrations.empty()) {
        YoloPersonDetectorOptions yolo_options;
        yolo_options.conf_threshold = options.yolo_conf_threshold;
        yolo_options.nms_threshold = options.yolo_nms_threshold;
        yolo_options.use_cuda = options.yolo_use_cuda;
        if (!yolo_detector) {
            yolo_detector = std::make_unique<YoloPersonDetector>(yolo_options);
            if (!yolo_detector->Load(options.yolo_model_path)) {
                std::cerr << "DatasetPrep: failed to initialize YOLO detector using "
                          << options.yolo_model_path << std::endl;
                return 1;
            }
        }
    }

    DatasetExporter exporter(DatasetExporter::Options{
        options.output_dir,
        options.save_images,
        options.save_masks,
        options.save_training_debug,
        options.save_pose_json,
        true,
        3});
    if (!exporter.Initialize()) {
        return 1;
    }

    int processed_frames = 0;
    while (synchronizer.HasNextFrame()) {
        if (options.max_frames >= 0 && processed_frames >= options.max_frames) {
            break;
        }

        const SteadyClock::time_point frame_start_time = SteadyClock::now();

        SyncedFrameCollection synced_frames;
        if (!synchronizer.GetNextSyncedViews(&synced_frames)) {
            break;
        }

        const int reference_frame_seq = synced_frames.views.empty()
            ? -1
            : synced_frames.views.front().video_frame_index;
        const MocapFrame3D* matched_pose = nullptr;
        if (use_mocap) {
            matched_pose = global_mocap.FindClosestFrame(
                synced_frames.sync_timestamp_ms,
                reference_frame_seq,
                options.pose_timestamp_delta_ms,
                options.pose_frame_delta,
                options.prefer_pose_timestamps);
        }

        PoseLookupResult pose_lookup;
        pose_lookup.sync_index = synced_frames.sync_index;
        pose_lookup.sync_timestamp_ms = synced_frames.sync_timestamp_ms;

        MocapFrame3D pose3d;
        if (matched_pose != nullptr) {
            pose_lookup.frame = matched_pose;
            pose_lookup.pose_timestamp_ms = matched_pose->ReferenceTimestampMs();
            if (pose_lookup.pose_timestamp_ms >= 0.0 && pose_lookup.sync_timestamp_ms >= 0.0) {
                pose_lookup.timestamp_delta_ms =
                    std::abs(pose_lookup.pose_timestamp_ms - pose_lookup.sync_timestamp_ms);
            }
            pose3d = *matched_pose;
        }
        if (!use_mocap) {
            pose3d.frame_seq = synced_frames.sync_index;
            pose3d.timestamp_ms = synced_frames.sync_timestamp_ms;
            pose3d.people.resize(1);
            pose3d.people[0].person_id = 0;
        }

        {
            std::ostringstream message;
            message << "DatasetPrep: frame "
                    << (processed_frames + 1)
                    << " sync=" << synced_frames.sync_index
                    << " views=" << synced_frames.views.size()
                    << " people=" << pose3d.people.size()
                    << " starting";
            LogProgress(message.str());
        }

        const SteadyClock::time_point yolo_start_time = SteadyClock::now();
        LogProgress("DatasetPrep: matching YOLO crops...");
        std::vector<std::vector<ProjectedPersonCrop>> matched_yolo_crops;
        if (!use_mocap) {
            matched_yolo_crops.resize(synced_frames.views.size(),
                                      std::vector<ProjectedPersonCrop>(pose3d.people.size()));
            if (yolo_detector != nullptr) {
                for (size_t view_index = 0; view_index < synced_frames.views.size(); ++view_index) {
                    const auto& view = synced_frames.views[view_index];
                    const auto* calibration =
                        FindCalibrationBySourceIndex(calibrations, view.source_camera_index);
                    if (calibration == nullptr) {
                        continue;
                    }

                    const auto detections = DetectPeopleInView(*yolo_detector, view.image);
                    if (detections.empty()) {
                        continue;
                    }

                    const auto best_detection = *std::max_element(
                        detections.begin(),
                        detections.end(),
                        [](const YoloDetection& left, const YoloDetection& right) {
                            return left.score < right.score;
                        });

                    ProjectedPersonCrop crop;
                    if (BuildCropFromYoloOnly(view,
                                              best_detection,
                                              *calibration,
                                              options.crop_margin,
                                              &crop)) {
                        matched_yolo_crops[view_index][0] = crop;
                    }
                }
            }
        } else {
            matched_yolo_crops = BuildMatchedYoloCrops(
                synced_frames,
                pose3d,
                calibrations,
                yolo_detector.get(),
                options.crop_margin);
        }
        {
            std::ostringstream message;
            message << "DatasetPrep: YOLO crop matching finished with "
                    << CountValidMatchedCrops(matched_yolo_crops)
                    << " matched crops in "
                    << std::fixed << std::setprecision(1)
                    << ElapsedMilliseconds(yolo_start_time, SteadyClock::now()) << " ms";
            LogProgress(message.str());
        }

#if DATASET_PREP_HAS_SMPL
        if (options.smpl_enabled) {
            const SteadyClock::time_point smpl_start_time = SteadyClock::now();
            LogProgress("DatasetPrep: fitting SMPL parameters...");
            for (size_t person_index = 0; person_index < pose3d.people.size(); ++person_index) {
                auto& person = pose3d.people[person_index];
                ResetSmplFit(&person);

                if (options.smpl_use_pear && pear_estimator) {
                    if (EstimatePersonSmplWithPear(synced_frames,
                                                   matched_yolo_crops,
                                                   calibrations,
                                                   *pear_estimator,
                                                   rtmpose_detector.get(),
                                                   options.crop_resolution,
                                                   static_cast<int>(person_index),
                                                   &person)) {
                        continue;
                    }
                }

                if (smpl_solver) {
                    std::vector<cv::Point3f> joint_centers;
                    std::vector<cv::Vec4f> joint_rotations;
                    std::vector<float> joint_confidences;
                    joint_centers.reserve(person.joints.size());
                    joint_rotations.reserve(person.joints.size());
                    joint_confidences.reserve(person.joints.size());
                    for (const auto& joint : person.joints) {
                        joint_centers.push_back(joint.xyz);
                        joint_rotations.push_back(joint.quaternion);
                        joint_confidences.push_back(joint.confidence);
                    }

                    SmplParameters smpl_parameters;
                    person.smpl_valid = smpl_solver->FitToMocap(
                        joint_centers, joint_rotations, joint_confidences, &smpl_parameters);
                    if (person.smpl_valid) {
                        person.smpl_pose = std::move(smpl_parameters.thetas);
                        person.smpl_shape = std::move(smpl_parameters.betas);
                        person.smpl_scale = smpl_parameters.mocap_scale;
                        if (smpl_parameters.translation.size() >= 3u) {
                            person.smpl_translation = cv::Point3f(
                                smpl_parameters.translation[0],
                                smpl_parameters.translation[1],
                                smpl_parameters.translation[2]);
                        } else {
                            person.smpl_translation = cv::Point3f(
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN());
                        }
                    } else {
                        ResetSmplFit(&person);
                    }
                }
            }
            {
                std::ostringstream message;
                message << "DatasetPrep: SMPL fitting finished with "
                        << CountValidSmplFits(pose3d)
                        << "/" << pose3d.people.size()
                        << " valid people in "
                        << std::fixed << std::setprecision(1)
                        << ElapsedMilliseconds(smpl_start_time, SteadyClock::now()) << " ms";
                LogProgress(message.str());
            }
        }
#endif

        ExportFrameArtifacts artifacts;
        artifacts.synced_frames = synced_frames;
        artifacts.pose_lookup = pose_lookup;
        artifacts.pose3d = pose3d;
        artifacts.training_export_requested = !calibrations.empty();
        if (!artifacts.training_export_requested) {
            const SteadyClock::time_point mask_start_time = SteadyClock::now();
            LogProgress("DatasetPrep: extracting mattes...");
            if (!background_extractor.Process(synced_frames, &artifacts.masks)) {
                return 1;
            }
            {
                std::ostringstream message;
                message << "DatasetPrep: extracted "
                        << artifacts.masks.size()
                        << " mattes in "
                        << std::fixed << std::setprecision(1)
                        << ElapsedMilliseconds(mask_start_time, SteadyClock::now()) << " ms";
                LogProgress(message.str());
            }
        }

        if (artifacts.training_export_requested) {
            const SteadyClock::time_point training_start_time = SteadyClock::now();
            LogProgress("DatasetPrep: building training samples...");
            if (options.save_training_debug) {
                artifacts.full_overlays.resize(artifacts.synced_frames.views.size());
            }
            for (size_t view_index = 0; view_index < artifacts.synced_frames.views.size(); ++view_index) {
                const auto& view = artifacts.synced_frames.views[view_index];
                const auto* calibration =
                    FindCalibrationBySourceIndex(calibrations, view.source_camera_index);
                if (calibration == nullptr) {
                    continue;
                }

                for (size_t person_index = 0; person_index < artifacts.pose3d.people.size(); ++person_index) {
                    const auto& person = artifacts.pose3d.people[person_index];
                    if (view_index >= matched_yolo_crops.size() ||
                        person_index >= matched_yolo_crops[view_index].size()) {
                        continue;
                    }
                    ExportTrainingSample sample;
                    if (BuildTrainingSample(view,
                                            background_extractor,
#if DATASET_PREP_HAS_SMPL
                                            options.save_training_debug ? debug_smpl_layer.get() : nullptr,
#endif
                                            person,
                                            static_cast<int>(person_index),
                                            *calibration,
                                            matched_yolo_crops[view_index][person_index],
                                            options.crop_resolution,
                                            &sample)) {
                        if (options.save_training_debug && !sample.crop_overlay.empty()) {
                            cv::Mat full_overlay = view.image.clone();
                            const float scale_to_output = sample.crop_w / 256.0f;
                            const float scale_back = sample.model_to_full(0, 0);
                            if (scale_to_output > 0.0f && scale_back > 0.0f) {
                                const float crop_to_full_scale = scale_back / scale_to_output;
                                const cv::Matx23f crop_to_full(
                                    crop_to_full_scale, 0.0f, sample.model_to_full(0, 2),
                                    0.0f, crop_to_full_scale, sample.model_to_full(1, 2));
                                cv::Mat warped(full_overlay.size(), full_overlay.type(), cv::Scalar::all(0));
                                cv::warpAffine(sample.crop_overlay,
                                               warped,
                                               crop_to_full,
                                               full_overlay.size(),
                                               cv::INTER_LINEAR,
                                               cv::BORDER_CONSTANT,
                                               cv::Scalar::all(0));
                                cv::Mat mask;
                                cv::cvtColor(warped, mask, cv::COLOR_BGR2GRAY);
                                warped.copyTo(full_overlay, mask);
                            }
                            artifacts.full_overlays[view_index] = std::move(full_overlay);
                        }
                        artifacts.training_samples.push_back(std::move(sample));
                    }
                }
            }
            {
                std::ostringstream message;
                message << "DatasetPrep: built "
                        << artifacts.training_samples.size()
                        << " training samples in "
                        << std::fixed << std::setprecision(1)
                        << ElapsedMilliseconds(training_start_time, SteadyClock::now()) << " ms";
                LogProgress(message.str());
            }
        }

        const SteadyClock::time_point save_start_time = SteadyClock::now();
        LogProgress("DatasetPrep: saving outputs...");
        if (!exporter.SaveFrame(artifacts)) {
            return 1;
        }
        {
            std::ostringstream message;
            message << "DatasetPrep: frame "
                    << (processed_frames + 1)
                    << " completed in "
                    << std::fixed << std::setprecision(1)
                    << ElapsedMilliseconds(frame_start_time, SteadyClock::now())
                    << " ms"
                    << " (save "
                    << ElapsedMilliseconds(save_start_time, SteadyClock::now())
                    << " ms)";
            LogProgress(message.str());
        }

        ++processed_frames;
    }

    std::cout << "dataset_prep exported " << processed_frames
              << " synchronized frame groups to " << options.output_dir.string() << std::endl;
    return 0;
}
