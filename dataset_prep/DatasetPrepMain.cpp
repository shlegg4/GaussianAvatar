#include <algorithm>
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
#include "dataset_prep/processing/CliffEstimator.h"
#if DATASET_PREP_HAS_SMPL
#include "utils/HmrInferenceUtils.h"
#include "utils/HmrMathHelpers.h"
#include "utils/RtmPoseDetector.h"
#include "utils/SmplLBS.h"
#include "utils/SmplifyLite.h"
#endif

namespace dataset_prep {
namespace {

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
    bool smpl_use_cliff = true;
    std::string smpl_model_path = "smpl_data.pt";
    bool smpl_use_cuda = false;
    int smpl_iters = 60;
    std::string cliff_model_path = "cliff_hr48_static.onnx";
    bool cliff_use_cuda = false;
    std::string rtmpose_model_path = "rtmpose26.onnx";
    bool rtmpose_use_cuda = false;
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
        << "  --smpl-from-mocap           Use the old mocap-only SMPL solver instead of CLIFF\n"
        << "  --smpl-model path            Path to the SMPL model archive (.pt)\n"
        << "  --smpl-cuda                  Request CUDA for SMPL fitting when available\n"
        << "  --smpl-iters value           Number of optimization steps per person\n"
        << "  --cliff-model path           Path to the CLIFF ONNX model\n"
        << "  --cliff-cuda                 Request CUDA provider for CLIFF inference\n"
        << "  --rtmpose-model path         Path to the RTMPose ONNX model used for multi-view merge\n"
        << "  --rtmpose-cuda               Request CUDA provider for RTMPose inference\n"
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
            options.smpl_use_cliff = false;
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
        if (arg == "--cliff-model") {
            const char* value = require_value("--cliff-model");
            if (!value) return false;
            options.cliff_model_path = value;
            continue;
        }
        if (arg == "--cliff-cuda") {
            options.cliff_use_cuda = true;
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

        std::cerr << "Unknown argument: " << arg << std::endl;
        return false;
    }

    if (options.video_paths.empty() || options.pose_3d_path.empty() || options.output_dir.empty()) {
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
    options.cliff_model_path = ResolveExistingPath(options.cliff_model_path).string();
    options.rtmpose_model_path = ResolveExistingPath(options.rtmpose_model_path).string();
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
    if (options.smpl_enabled && options.smpl_use_cliff) {
        if (options.cliff_model_path.empty() ||
            !std::filesystem::exists(options.cliff_model_path)) {
            std::cerr << "Error: CLIFF model file not found: "
                      << options.cliff_model_path << std::endl;
            return false;
        }
        if (!options.rtmpose_model_path.empty() &&
            !std::filesystem::exists(options.rtmpose_model_path)) {
            std::cerr << "Warning: RTMPose model file not found, falling back to projected mocap "
                         "keypoints for the multi-view merge: "
                      << options.rtmpose_model_path << std::endl;
            options.rtmpose_model_path.clear();
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
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    int roi_x = 0;
    int roi_y = 0;
    int roi_size = 0;
    std::array<cv::Point2f, kMocapJointCount> projected_joints{};
    std::array<float, kMocapJointCount> joint_scores{};
};

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

std::vector<float> PoseToAxisAngleVector(const std::vector<float>& pose_values) {
    if (pose_values.size() == kSmplPoseParamCount) {
        return pose_values;
    }
    if (pose_values.size() == 144u) {
        return ConvertPose6dToAxisAngle(pose_values);
    }
    return {};
}

bool ConvertCliffPoseToWorld(const SmplResult& cliff_result,
                             const CameraCalibration& calibration,
                             std::vector<float>* out_pose_world) {
    if (out_pose_world == nullptr) {
        return false;
    }

    std::vector<float> pose_world = PoseToAxisAngleVector(cliff_result.pose);
    if (pose_world.size() != kSmplPoseParamCount) {
        return false;
    }

    cv::Vec3f root_local(pose_world[0], pose_world[1], pose_world[2]);
    cv::Matx33f root_local_rotation;
    cv::Rodrigues(root_local, root_local_rotation);

    const cv::Matx33f root_world_rotation = calibration.R.t() * root_local_rotation;
    cv::Vec3f root_world;
    cv::Rodrigues(root_world_rotation, root_world);
    pose_world[0] = root_world[0];
    pose_world[1] = root_world[1];
    pose_world[2] = root_world[2];

    *out_pose_world = std::move(pose_world);
    return true;
}

cv::Vec3f CameraTranslationToWorld(const cv::Vec3f& translation_camera,
                                   const CameraCalibration& calibration) {
    return calibration.R.t() * (translation_camera - calibration.t);
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

struct CliffViewEstimate {
    SmplifyMultiViewObservation observation;
    std::vector<float> pose_world;
    std::vector<float> betas;
    cv::Vec3f translation_world{0.0f, 0.0f, 0.0f};
    float score = 0.0f;
};

bool EstimatePersonSmplWithCliff(const SyncedFrameCollection& synced_frames,
                                 const std::vector<CameraCalibration>& calibrations,
                                 CliffEstimator& cliff_estimator,
                                 RtmPoseDetector* rtmpose_detector,
                                 SMPLLayer& smpl_layer,
                                 int crop_resolution,
                                 float crop_margin,
                                 int smpl_iters,
                                 MocapPerson3D* person) {
    if (person == nullptr) {
        return false;
    }

    std::vector<CliffViewEstimate> views;
    views.reserve(synced_frames.views.size());

    for (const auto& view : synced_frames.views) {
        const auto* calibration =
            FindCalibrationBySourceIndex(calibrations, view.source_camera_index);
        if (calibration == nullptr) {
            continue;
        }

        ProjectedPersonCrop projected_crop;
        if (!BuildProjectedPersonCrop(view, *person, *calibration, crop_margin, &projected_crop)) {
            continue;
        }

        const int interpolation =
            projected_crop.roi_size > crop_resolution ? cv::INTER_AREA : cv::INTER_CUBIC;
        cv::Mat crop_image = CropSquareWithPaddingAndResize(
            view.image,
            projected_crop.roi_x,
            projected_crop.roi_y,
            projected_crop.roi_size,
            crop_resolution,
            interpolation);
        if (crop_image.empty()) {
            continue;
        }

        SmplResult cliff_result;
        if (!cliff_estimator.Estimate(
                crop_image,
                projected_crop.crop_cx,
                projected_crop.crop_cy,
                projected_crop.crop_size,
                projected_crop.fx,
                static_cast<int>(projected_crop.img_w),
                static_cast<int>(projected_crop.img_h),
                &cliff_result) ||
            cliff_result.shape.size() < kSmplShapeParamCount ||
            cliff_result.camera.size() < 3u) {
            continue;
        }

        std::vector<float> pose_world;
        if (!ConvertCliffPoseToWorld(cliff_result, *calibration, &pose_world)) {
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
                    static_cast<float>(projected_crop.roi_size) /
                    static_cast<float>(crop_resolution);
                for (size_t index = 0; index < crop_keypoints.size(); ++index) {
                    observation_keypoints.emplace_back(
                        static_cast<float>(projected_crop.roi_x) +
                            crop_keypoints[index].x * crop_to_full_scale,
                        static_cast<float>(projected_crop.roi_y) +
                            crop_keypoints[index].y * crop_to_full_scale);
                    observation_scores.push_back(crop_scores[index]);
                }
                have_rtmpose_observation = CountScoresAbove(observation_scores, 0.01f) >= 6;
            }
        }

        if (!have_rtmpose_observation) {
            observation_keypoints.assign(projected_crop.projected_joints.begin(),
                                         projected_crop.projected_joints.end());
            observation_scores.assign(projected_crop.joint_scores.begin(),
                                      projected_crop.joint_scores.end());
        }

        const cv::Vec3f translation_camera = EstimateTranslation(
            cliff_result.camera,
            projected_crop.crop_cx,
            projected_crop.crop_cy,
            projected_crop.crop_size,
            projected_crop.fx,
            projected_crop.img_w,
            projected_crop.img_h);
        const cv::Vec3f translation_world =
            CameraTranslationToWorld(translation_camera, *calibration);

        CliffViewEstimate estimate;
        estimate.observation.keypoints = std::move(observation_keypoints);
        estimate.observation.keypoint_scores = std::move(observation_scores);
        estimate.observation.K = projected_crop.K;
        estimate.observation.R = calibration->R;
        estimate.observation.t = calibration->t;
        estimate.observation.img_w = projected_crop.img_w;
        estimate.observation.img_h = projected_crop.img_h;
        estimate.pose_world = std::move(pose_world);
        estimate.betas.assign(cliff_result.shape.begin(),
                              cliff_result.shape.begin() + kSmplShapeParamCount);
        estimate.translation_world = translation_world;
        estimate.score =
            static_cast<float>(CountScoresAbove(estimate.observation.keypoint_scores, 0.01f)) +
            MeanPositiveScore(estimate.observation.keypoint_scores);
        views.push_back(std::move(estimate));
    }

    if (views.empty()) {
        return false;
    }

    const auto best_view_it = std::max_element(
        views.begin(),
        views.end(),
        [](const CliffViewEstimate& left, const CliffViewEstimate& right) {
            return left.score < right.score;
        });
    if (best_view_it == views.end()) {
        return false;
    }

    SmplResult merged_seed;
    merged_seed.pose = best_view_it->pose_world;
    merged_seed.shape.assign(kSmplShapeParamCount, 0.0f);
    cv::Vec3f translation_world(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (const auto& view : views) {
        const float weight = std::max(view.score, 1e-3f);
        weight_sum += weight;
        for (size_t beta_index = 0; beta_index < merged_seed.shape.size(); ++beta_index) {
            merged_seed.shape[beta_index] += view.betas[beta_index] * weight;
        }
        translation_world += view.translation_world * weight;
    }
    if (weight_sum > 0.0f) {
        for (float& beta : merged_seed.shape) {
            beta /= weight_sum;
        }
        translation_world *= (1.0f / weight_sum);
    } else {
        merged_seed.shape = best_view_it->betas;
        translation_world = best_view_it->translation_world;
    }
    merged_seed.camera = {translation_world[0], translation_world[1], translation_world[2]};

    SmplifyLiteOptions multiview_options;
    multiview_options.num_iters = std::max(10, smpl_iters);
    multiview_options.min_iters = std::min(10, multiview_options.num_iters);
    multiview_options.pose_reg = 5.0f;
    multiview_options.betas_reg = 5e-4f;
    multiview_options.reproj_robust_sigma = 25.0f;

    std::vector<SmplifyMultiViewObservation> observations;
    observations.reserve(views.size());
    for (const auto& view : views) {
        observations.push_back(view.observation);
    }

    SmplResult refined_result = merged_seed;
    if (!SmplifyLiteMultiView(
            smpl_layer,
            observations,
            &refined_result,
            multiview_options,
            nullptr)) {
        refined_result = merged_seed;
    }

    std::vector<float> final_pose = PoseToAxisAngleVector(refined_result.pose);
    if (final_pose.size() != kSmplPoseParamCount ||
        refined_result.shape.size() < kSmplShapeParamCount ||
        refined_result.camera.size() < 3u) {
        return false;
    }

    person->smpl_pose = std::move(final_pose);
    person->smpl_shape.assign(refined_result.shape.begin(),
                              refined_result.shape.begin() + kSmplShapeParamCount);
    person->smpl_translation = cv::Point3f(
        refined_result.camera[0],
        refined_result.camera[1],
        refined_result.camera[2]);
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
        const auto verts_cpu = smpl_out.vertices.squeeze(0).to(torch::kCPU).contiguous();
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
            const float X = verts_acc[vertex_index][0] + trans[0];
            const float Y = verts_acc[vertex_index][1] * sample->y_sign + trans[1];
            const float Z = verts_acc[vertex_index][2] + trans[2];
            camera_vertices[static_cast<size_t>(vertex_index)] = cv::Vec3f(X, Y, Z);
            if (!(Z > 1e-6f)) {
                continue;
            }

            const float u = (sample->focal_length * X / Z) + cx_crop;
            const float v = (sample->focal_length * Y / Z) + cy_crop;
            projected_vertices[static_cast<size_t>(vertex_index)] = cv::Point(
                static_cast<int>(std::lround(u)),
                static_cast<int>(std::lround(v)));
            vertex_valid[static_cast<size_t>(vertex_index)] =
                (u >= 0.0f && v >= 0.0f &&
                 u < static_cast<float>(overlay.cols) &&
                 v < static_cast<float>(overlay.rows))
                    ? 1u
                    : 0u;
        }

        if (smpl_layer.faces_cpu.defined() && smpl_layer.faces_cpu.numel() > 0) {
            struct ProjectedTriangle {
                float depth = 0.0f;
                cv::Point points[3];
                cv::Scalar color;
            };

            std::vector<ProjectedTriangle> triangles;
            auto faces_acc = smpl_layer.faces_cpu.accessor<int64_t, 2>();
            triangles.reserve(static_cast<size_t>(faces_acc.size(0)));

            for (int face_index = 0; face_index < faces_acc.size(0); ++face_index) {
                const int i0 = static_cast<int>(faces_acc[face_index][0]);
                const int i1 = static_cast<int>(faces_acc[face_index][1]);
                const int i2 = static_cast<int>(faces_acc[face_index][2]);
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
                         int crop_resolution,
                         float crop_margin,
                         ExportTrainingSample* out_sample) {
    if (out_sample == nullptr || view.image.empty() ||
        !person.smpl_valid || person.smpl_pose.size() < 3u) {
        return false;
    }

    ProjectedPersonCrop projected_crop;
    if (!BuildProjectedPersonCrop(view, person, calibration, crop_margin, &projected_crop)) {
        return false;
    }
    const float img_w = projected_crop.img_w;
    const float img_h = projected_crop.img_h;
    const float fx = projected_crop.fx;

    cv::Vec3f global_orient(person.smpl_pose[0], person.smpl_pose[1], person.smpl_pose[2]);
    cv::Matx33f smpl_global_rotation;
    cv::Rodrigues(global_orient, smpl_global_rotation);

    const cv::Matx33f smpl_local_rotation = calibration.R * smpl_global_rotation;
    cv::Vec3f local_orient;
    cv::Rodrigues(smpl_local_rotation, local_orient);

    const float mocap_scale =
        (std::isfinite(person.smpl_scale) && person.smpl_scale > 0.0f)
            ? person.smpl_scale
            : 1.0f;
    const cv::Vec3f calibration_t_metric(
        calibration.t[0] * mocap_scale,
        calibration.t[1] * mocap_scale,
        calibration.t[2] * mocap_scale);
    cv::Vec3f root_global(0.0f, 0.0f, 0.0f);
    bool has_root_global = false;
    if (std::isfinite(person.smpl_translation.x) &&
        std::isfinite(person.smpl_translation.y) &&
        std::isfinite(person.smpl_translation.z)) {
        root_global = cv::Vec3f(person.smpl_translation.x,
                                person.smpl_translation.y,
                                person.smpl_translation.z);
        has_root_global = true;
    } else {
        if (person.joints[19].IsValid()) {
            root_global = cv::Vec3f(person.joints[19].xyz.x * mocap_scale,
                                    person.joints[19].xyz.y * mocap_scale,
                                    person.joints[19].xyz.z * mocap_scale);
            has_root_global = true;
        } else if (person.joints[11].IsValid() && person.joints[12].IsValid()) {
            root_global = cv::Vec3f(
                0.5f * (person.joints[11].xyz.x + person.joints[12].xyz.x) * mocap_scale,
                0.5f * (person.joints[11].xyz.y + person.joints[12].xyz.y) * mocap_scale,
                0.5f * (person.joints[11].xyz.z + person.joints[12].xyz.z) * mocap_scale);
            has_root_global = true;
        }
    }
    if (!has_root_global) {
        return false;
    }
    const cv::Vec3f root_local = calibration.R * root_global + calibration_t_metric;
    const float X = root_local[0];
    const float Y = root_local[1];
    const float Z = root_local[2];
    if (!(Z > kMinProjectedDepth)) {
        return false;
    }
    const float raw_crop_size = projected_crop.crop_size;
    const float raw_crop_cx = projected_crop.crop_cx;
    const float raw_crop_cy = projected_crop.crop_cy;
    const int roi_size = projected_crop.roi_size;
    const int roi_x = projected_crop.roi_x;
    const int roi_y = projected_crop.roi_y;
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
    if (!parser.Parse(options.pose_3d_path, &global_mocap, parse_options)) {
        return 1;
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

#if DATASET_PREP_HAS_SMPL
    std::unique_ptr<SmplifyLiteMocapSolver> smpl_solver;
    std::unique_ptr<CliffEstimator> cliff_estimator;
    std::unique_ptr<RtmPoseDetector> rtmpose_detector;
    std::unique_ptr<SMPLLayer> debug_smpl_layer;
    if (options.smpl_enabled) {
        if (options.smpl_use_cliff) {
            cliff_estimator = std::make_unique<CliffEstimator>(CliffEstimator::Options{
                options.cliff_model_path,
                options.cliff_use_cuda});
            if (!cliff_estimator->Initialize()) {
                std::cerr << "DatasetPrep: failed to initialize CLIFF estimator using "
                          << options.cliff_model_path << std::endl;
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

        if (options.save_training_debug || options.smpl_use_cliff) {
            try {
                debug_smpl_layer = std::make_unique<SMPLLayer>(options.smpl_model_path);
                torch::Device device(torch::kCPU);
                if (options.smpl_use_cuda && torch::cuda::is_available()) {
                    device = torch::Device(torch::kCUDA);
                }
                debug_smpl_layer->to(device);
                debug_smpl_layer->eval();
            } catch (const std::exception& e) {
                if (options.smpl_use_cliff) {
                    std::cerr << "DatasetPrep: failed to initialize SMPL layer for CLIFF "
                                 "multi-view fitting: "
                              << e.what() << std::endl;
                    return 1;
                }
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

        SyncedFrameCollection synced_frames;
        if (!synchronizer.GetNextSyncedViews(&synced_frames)) {
            break;
        }

        const int reference_frame_seq = synced_frames.views.empty()
            ? -1
            : synced_frames.views.front().video_frame_index;
        const MocapFrame3D* matched_pose = global_mocap.FindClosestFrame(
            synced_frames.sync_timestamp_ms,
            reference_frame_seq,
            options.pose_timestamp_delta_ms,
            options.pose_frame_delta,
            options.prefer_pose_timestamps);

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

#if DATASET_PREP_HAS_SMPL
        if (options.smpl_enabled) {
            for (auto& person : pose3d.people) {
                ResetSmplFit(&person);

                if (options.smpl_use_cliff && cliff_estimator && debug_smpl_layer) {
                    if (EstimatePersonSmplWithCliff(synced_frames,
                                                    calibrations,
                                                    *cliff_estimator,
                                                    rtmpose_detector.get(),
                                                    *debug_smpl_layer,
                                                    options.crop_resolution,
                                                    options.crop_margin,
                                                    options.smpl_iters,
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
        }
#endif

        ExportFrameArtifacts artifacts;
        artifacts.synced_frames = synced_frames;
        artifacts.pose_lookup = pose_lookup;
        artifacts.pose3d = pose3d;
        artifacts.training_export_requested = !calibrations.empty();
        if (!artifacts.training_export_requested) {
            if (!background_extractor.Process(synced_frames, &artifacts.masks)) {
                return 1;
            }
        }

        if (artifacts.training_export_requested) {
            for (const auto& view : artifacts.synced_frames.views) {
                const auto* calibration =
                    FindCalibrationBySourceIndex(calibrations, view.source_camera_index);
                if (calibration == nullptr) {
                    continue;
                }

                for (size_t person_index = 0; person_index < artifacts.pose3d.people.size(); ++person_index) {
                    const auto& person = artifacts.pose3d.people[person_index];
                    ExportTrainingSample sample;
                    if (BuildTrainingSample(view,
                                            background_extractor,
#if DATASET_PREP_HAS_SMPL
                                            options.save_training_debug ? debug_smpl_layer.get() : nullptr,
#endif
                                            person,
                                            static_cast<int>(person_index),
                                            *calibration,
                                            options.crop_resolution,
                                            options.crop_margin,
                                            &sample)) {
                        artifacts.training_samples.push_back(std::move(sample));
                    }
                }
            }
        }

        if (!exporter.SaveFrame(artifacts)) {
            return 1;
        }

        ++processed_frames;
    }

    std::cout << "dataset_prep exported " << processed_frames
              << " synchronized frame groups to " << options.output_dir.string() << std::endl;
    return 0;
}
