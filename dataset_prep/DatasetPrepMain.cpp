#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset_prep/export/DatasetExporter.h"
#include "dataset_prep/ingestion/CameraCalibrationLoader.h"
#include "dataset_prep/ingestion/VideoSynchronizer.h"
#include "dataset_prep/processing/BackgroundExtractor.h"
#include "dataset_prep/processing/SmplxOptimizer.h"
#include "dataset_prep/utils/CommandLine.h"
#include "dataset_prep/utils/GeometryUtils.h"
#include "dataset_prep/utils/ImageUtils.h"
#include "utils/RtmPoseDetector.h"
#include "utils/YoloPersonDetector.h"

#if DATASET_PREP_HAS_SMPL
#include "dataset_prep/utils/SmplxLBS.h"
#include <torch/script.h>
#endif

namespace fs = std::filesystem;

namespace dataset_prep {
namespace {

constexpr float kTriangulationMinScore = 0.000025f;
constexpr float kTemporalMaxRotationStepRad = 0.35f;
constexpr float kTemporalMaxTranslationStep = 0.15f;
constexpr int kTargetCropRes = 1024;
constexpr int kPoseInputRes = 256;
constexpr float kCropMargin = 1.25f;

const std::array<std::pair<int, int>, 17> kSmplxBones = {{
    {0, 1}, {0, 2}, {1, 4}, {2, 5}, {4, 7}, {5, 8},
    {0, 3}, {3, 6}, {6, 9}, {9, 12}, {12, 15},
    {9, 13}, {9, 14}, {13, 16}, {14, 17}, {16, 18}, {17, 19},
}};

struct TrainingCropMetadata {
    float crop_cx = 0.0f;
    float crop_cy = 0.0f;
    float crop_size = 0.0f;
    float crop_x0 = 0.0f;
    float crop_y0 = 0.0f;
    float crop_w = 0.0f;
    float crop_h = 0.0f;
    float focal_length = 0.0f;
};

struct TriangulationViewSample {
    const CameraCalibration* calibration = nullptr;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> scores;
};

struct TargetViewSample {
    const SyncedView* view = nullptr;
    const CameraCalibration* calibration = nullptr;
    cv::Mat crop_image;
    cv::Mat crop_image_to_save;
    cv::Mat crop_matte;
    cv::Mat crop_overlay;
    cv::Matx23f crop_trans = cv::Matx23f::eye();
    TrainingCropMetadata training_crop;
    size_t synced_view_index = 0u;
};

bool IsFinitePoint3(const cv::Point3f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

std::vector<float> RotateRootAxisAngleToCamera(const cv::Matx33f& camera_rotation,
                                               const std::vector<float>& global_orient_world) {
    std::vector<float> global_orient_camera =
        PadFloatVector(global_orient_world, static_cast<size_t>(kSmplxGlobalOrientParamCount));
    if (global_orient_camera.size() < 3u) {
        return global_orient_camera;
    }

    const cv::Vec3f root_world_axis_angle(
        global_orient_camera[0],
        global_orient_camera[1],
        global_orient_camera[2]);
    cv::Matx33f root_world_rotation = cv::Matx33f::eye();
    cv::Rodrigues(root_world_axis_angle, root_world_rotation);

    const cv::Matx33f root_camera_rotation = camera_rotation * root_world_rotation;
    cv::Vec3f root_camera_axis_angle;
    cv::Rodrigues(cv::Mat(root_camera_rotation), root_camera_axis_angle);

    global_orient_camera[0] = root_camera_axis_angle[0];
    global_orient_camera[1] = root_camera_axis_angle[1];
    global_orient_camera[2] = root_camera_axis_angle[2];
    return global_orient_camera;
}

std::map<std::string, CameraCalibration> BuildCalibrationByCameraId(
    const std::vector<CameraCalibration>& calibrations) {
    std::map<std::string, CameraCalibration> by_id;
    for (const auto& calibration : calibrations) {
        by_id[calibration.camera_id] = calibration;
    }
    return by_id;
}

TrainingCropMetadata BuildTrainingCropMetadata(const CameraCalibration* calibration,
                                               const cv::Rect2f& crop_bbox,
                                               int target_crop_res) {
    TrainingCropMetadata metadata;
    if (target_crop_res <= 0) {
        return metadata;
    }

    metadata.crop_cx = crop_bbox.x + crop_bbox.width * 0.5f;
    metadata.crop_cy = crop_bbox.y + crop_bbox.height * 0.5f;
    metadata.crop_size = std::max(crop_bbox.width, crop_bbox.height);
    metadata.crop_x0 = crop_bbox.x;
    metadata.crop_y0 = crop_bbox.y;
    metadata.crop_w = static_cast<float>(target_crop_res);
    metadata.crop_h = static_cast<float>(target_crop_res);
    metadata.focal_length = calibration != nullptr
        ? 0.5f * (calibration->K(0, 0) + calibration->K(1, 1))
        : static_cast<float>(target_crop_res);
    return metadata;
}

std::array<float, 3> BuildTrainingCameraParams(const cv::Vec3f& translation,
                                               const TrainingCropMetadata& crop,
                                               int img_width,
                                               int img_height) {
    const float tz = std::max(translation[2], 1e-6f);
    const float s = (2.0f * crop.focal_length) / (std::max(crop.crop_size, 1e-6f) * tz);
    const float full_cx = static_cast<float>(img_width) * 0.5f;
    const float full_cy = static_cast<float>(img_height) * 0.5f;
    const float u_screen = (crop.focal_length * translation[0] / tz) + full_cx;
    const float v_screen = (crop.focal_length * translation[1] / tz) + full_cy;
    const float denom = crop.crop_size * s * 0.5f + 1e-9f;

    return {
        s,
        (u_screen - crop.crop_cx) / denom,
        (v_screen - crop.crop_cy) / denom,
    };
}

void TriangulateJoints(const std::vector<TriangulationViewSample>& views,
                      std::vector<cv::Point3f>* out_joints,
                      std::vector<float>* out_scores) {
    if (out_joints == nullptr || out_scores == nullptr) {
        return;
    }

    out_joints->assign(
        kMocapJointCount,
        cv::Point3f(std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()));
    out_scores->assign(kMocapJointCount, 0.0f);

    for (size_t joint_index = 0; joint_index < kMocapJointCount; ++joint_index) {
        std::vector<cv::Matx34f> projections;
        std::vector<cv::Point2f> observations;
        std::vector<float> confidence_scores;

        for (const auto& sample : views) {
            if (sample.calibration == nullptr || sample.keypoints.size() <= joint_index ||
                sample.scores.size() <= joint_index) {
                continue;
            }
            const float score = sample.scores[joint_index];
            if (score < kTriangulationMinScore) {
                continue;
            }

            projections.push_back(BuildProjectionMatrix(*sample.calibration));
            observations.push_back(sample.keypoints[joint_index]);
            confidence_scores.push_back(score);
        }

        cv::Point3f point_world;
        if (TriangulatePointDLT(projections, observations, &point_world)) {
            (*out_joints)[joint_index] = point_world;
            float avg_score = 0.0f;
            for (float score : confidence_scores) {
                avg_score += score;
            }
            (*out_scores)[joint_index] = confidence_scores.empty()
                ? 0.0f
                : avg_score / static_cast<float>(confidence_scores.size());
        }
    }
}

float EstimateSubjectHeight(const std::vector<cv::Point3f>& joints,
                            const std::vector<float>& scores) {
    const auto is_valid = [&](int idx) {
        return idx >= 0 &&
               idx < static_cast<int>(joints.size()) &&
               idx < static_cast<int>(scores.size()) &&
               scores[static_cast<size_t>(idx)] >= kTriangulationMinScore &&
               IsFinitePoint3(joints[static_cast<size_t>(idx)]);
    };

    cv::Point3f top_pt;
    if (is_valid(0)) {
        top_pt = joints[0];
    } else if (is_valid(5) && is_valid(6)) {
        top_pt = (joints[5] + joints[6]) * 0.5f;
    } else {
        return -1.0f;
    }

    cv::Point3f bottom_pt;
    if (is_valid(15) && is_valid(16)) {
        bottom_pt = (joints[15] + joints[16]) * 0.5f;
    } else if (is_valid(11) && is_valid(12)) {
        bottom_pt = (joints[11] + joints[12]) * 0.5f;
    } else {
        return -1.0f;
    }

    return static_cast<float>(cv::norm(top_pt - bottom_pt));
}

}  // namespace

}  // namespace dataset_prep

using namespace dataset_prep;

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        const std::string first_arg = argv[1];
        if (first_arg == "--help" || first_arg == "-h") {
            PrintDatasetPrepUsage(std::cout);
            return 0;
        }
    }

    DatasetPrepOptions options;
    if (!ParseDatasetPrepCommandLine(argc, argv, &options)) {
        PrintDatasetPrepUsage(std::cerr);
        return 1;
    }

    fs::path calibration_dir;
    if (!ResolveCalibrationDirectory(&calibration_dir)) {
        std::cerr << "Failed to locate calibration directory containing data/extrinsics.json from "
                  << fs::current_path().string() << "\n";
        return 1;
    }

    CameraCalibrationLoader calibration_loader;
    std::vector<CameraCalibration> calibrations;
    if (!calibration_loader.Load(calibration_dir, options.sources, &calibrations)) {
        std::cerr << "Failed to load camera calibrations from " << calibration_dir.string() << "\n";
        return 1;
    }

    auto calibration_by_camera_id = BuildCalibrationByCameraId(calibrations);

    YoloPersonDetectorOptions yolo_opts;
    yolo_opts.conf_threshold = 0.25f;
    yolo_opts.use_cuda = true;
    YoloPersonDetector yolo(yolo_opts);
    if (!yolo.Load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\yolov8n.onnx")) {
        std::cerr << "Failed to load YOLO model.\n";
        return 1;
    }

    BackgroundExtractor::Options modnet_opts;
    modnet_opts.model_path = "C:\\Users\\Sam\\Documents\\GaussianAvatar\\modnet.onnx";
    modnet_opts.use_cuda = true;
    modnet_opts.input_size = kTargetCropRes;
    modnet_opts.binary_threshold = -1.0f;
    BackgroundExtractor modnet(modnet_opts);
    if (!modnet.Initialize()) {
        std::cerr << "Failed to load MODNet model.\n";
        return 1;
    }

    RtmPoseDetectorOptions rtmpose_opts;
    rtmpose_opts.use_cuda = true;
    rtmpose_opts.conf_threshold = 0.00005f;
    RtmPoseDetector rtmpose(rtmpose_opts);
    if (!rtmpose.Load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\rtmpose26.onnx")) {
        std::cerr << "Failed to load RTMPose model.\n";
        return 1;
    }

#if DATASET_PREP_HAS_SMPL
    torch::jit::script::Module smplx_layer;
    std::unique_ptr<SMPLXLayer> smplx_lbs;
    try {
        smplx_layer = torch::jit::load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\smplx_libtorch.pt");
        smplx_lbs = std::make_unique<SMPLXLayer>("C:\\Users\\Sam\\Documents\\GaussianAvatar\\smplx_data.pt");
    } catch (const c10::Error& error) {
        std::cerr << "Failed to load SMPL-X TorchScript module.\n" << error.what() << std::endl;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Failed to load SMPL-X LBS model.\n" << error.what() << std::endl;
        return 1;
    }
    smplx_layer.to(torch::kCPU);
    smplx_layer.eval();
    smplx_lbs->to(torch::kCPU);
    smplx_lbs->eval();
#endif

    DatasetExporter::Options export_opts;
    export_opts.output_dir = options.output_dir;
    export_opts.save_images = true;
    export_opts.save_masks = true;
    export_opts.save_training_debug = true;
    export_opts.save_pose_json = false;
    export_opts.append_manifest = true;
    DatasetExporter exporter(export_opts);
    if (!exporter.Initialize()) {
        std::cerr << "Failed to initialize DatasetExporter.\n";
        return 1;
    }

    VideoSynchronizer synchronizer({options.sync_tolerance_ms});
    if (!synchronizer.Open(options.sources)) {
        std::cerr << "Failed to open one or more input feeds.\n";
        return 1;
    }

    bool has_prev_pose = false;
    std::vector<float> prev_global_orient(static_cast<size_t>(kSmplxGlobalOrientParamCount), 0.0f);
    std::vector<float> prev_body_pose(static_cast<size_t>(kSmplxBodyPoseParamCount), 0.0f);
    cv::Vec3f prev_training_translation(0.0f, 0.0f, 2.5f);
    bool is_scale_calibrated = false;
    float current_scale_factor = 1.0f;
    const float kTargetMetricHeight = 1.7f;

    int selected_frame_count = 0;
    int exported_frame_count = 0;
    SyncedFrameCollection synced_frames;
    while (synchronizer.GetNextSyncedViews(&synced_frames)) {
        if (synced_frames.sync_index < options.start_frame_index) {
            continue;
        }
        if (options.max_frames >= 0 && selected_frame_count >= options.max_frames) {
            break;
        }
        ++selected_frame_count;

        TargetViewSample target_sample;
        std::vector<TriangulationViewSample> triangulation_views;
        for (size_t view_index = 0; view_index < synced_frames.views.size(); ++view_index) {
            const auto& view = synced_frames.views[view_index];
            cv::Rect2f person_bbox;
            float detection_score = 0.0f;
            if (!yolo.DetectPerson(view.image, &person_bbox, &detection_score) ||
                detection_score <= 0.0f || person_bbox.area() <= 0.0f) {
                continue;
            }

            const auto calibration_it = calibration_by_camera_id.find(view.camera_id);
            if (calibration_it == calibration_by_camera_id.end() || !calibration_it->second.valid) {
                continue;
            }

            cv::Rect2f crop_bbox;
            if (!ProcessBBox(person_bbox,
                             view.image.cols,
                             view.image.rows,
                             cv::Size(kPoseInputRes, kPoseInputRes),
                             kCropMargin,
                             &crop_bbox)) {
                continue;
            }

            cv::Mat crop_image;
            cv::Matx23f crop_trans = cv::Matx23f::eye();
            cv::Matx23f crop_inv_trans = cv::Matx23f::eye();
            if (!GeneratePatchImage(view.image,
                                    crop_bbox,
                                    1.0f,
                                    0.0f,
                                    false,
                                    cv::Size(kTargetCropRes, kTargetCropRes),
                                    &crop_image,
                                    &crop_trans,
                                    &crop_inv_trans)) {
                continue;
            }

            TriangulationViewSample tri_sample;
            tri_sample.calibration = &calibration_it->second;
            if (rtmpose.DetectPose(crop_image, &tri_sample.keypoints, &tri_sample.scores, synced_frames.sync_index)) {
                for (auto& point : tri_sample.keypoints) {
                    point = ApplyAffinePoint(crop_inv_trans, point);
                }
                triangulation_views.push_back(std::move(tri_sample));
            }

            if (view.camera_id == options.target_camera_id) {
                target_sample.view = &view;
                target_sample.calibration = &calibration_it->second;
                target_sample.crop_image = crop_image;
                target_sample.crop_overlay = crop_image.clone();
                target_sample.crop_trans = crop_trans;
                target_sample.synced_view_index = view_index;
                target_sample.training_crop = BuildTrainingCropMetadata(target_sample.calibration,
                                                                        crop_bbox,
                                                                        kTargetCropRes);
                if (modnet.ProcessImage(crop_image, &target_sample.crop_matte)) {
                    target_sample.crop_image_to_save = ApplyCropMatte(crop_image, target_sample.crop_matte);
                } else {
                    target_sample.crop_image_to_save = crop_image;
                }
            }
        }

        if (target_sample.view == nullptr) {
            continue;
        }

        std::vector<cv::Point3f> triangulated_joints;
        std::vector<float> triangulated_scores;
        TriangulateJoints(triangulation_views, &triangulated_joints, &triangulated_scores);

        if (!is_scale_calibrated) {
            const float unscaled_height = EstimateSubjectHeight(triangulated_joints, triangulated_scores);
            if (unscaled_height > 0.1f) {
                const float scale_factor = kTargetMetricHeight / unscaled_height;
                current_scale_factor = scale_factor;
                std::cout << "Dynamically calculated Scale Factor (SF): " << scale_factor
                          << " (Unscaled height: " << unscaled_height << ")" << std::endl;

                for (auto& pair : calibration_by_camera_id) {
                    pair.second.t *= scale_factor;
                }

                TriangulateJoints(triangulation_views, &triangulated_joints, &triangulated_scores);
                is_scale_calibrated = true;
            }
        }
         
        std::vector<float> smplx_shape(static_cast<size_t>(kSmplxShapeParamCount), 0.0f);
        std::vector<float> smplx_expression(static_cast<size_t>(kSmplxExpressionParamCount), 0.0f);
        std::vector<float> smplx_global_orient(static_cast<size_t>(kSmplxGlobalOrientParamCount), 0.0f);
        std::vector<float> smplx_body_pose(static_cast<size_t>(kSmplxBodyPoseParamCount), 0.0f);
        std::vector<float> smplx_jaw_pose(static_cast<size_t>(kSmplxJawPoseParamCount), 0.0f);
        std::vector<float> smplx_eye_pose(static_cast<size_t>(kSmplxEyePoseParamCount), 0.0f);
        std::vector<float> smplx_left_hand_pose(static_cast<size_t>(kSmplxHandPoseParamCount), 0.0f);
        std::vector<float> smplx_right_hand_pose(static_cast<size_t>(kSmplxHandPoseParamCount), 0.0f);
        cv::Vec3f training_camera_translation(0.0f, 0.0f, 2.5f);
        SmplxOptimizationResult optimization_result;
        bool has_optimization_result = false;
        std::vector<cv::Point3f> smplx_vertices_world;

#if DATASET_PREP_HAS_SMPL
        SmplxOptimizationInputs optimization_inputs;
        optimization_inputs.fixed_betas = smplx_shape;
        optimization_inputs.fixed_expression = smplx_expression;
        optimization_inputs.jaw_pose = smplx_jaw_pose;
        optimization_inputs.eye_pose = smplx_eye_pose;
        optimization_inputs.left_hand_pose = smplx_left_hand_pose;
        optimization_inputs.right_hand_pose = smplx_right_hand_pose;
        optimization_inputs.init_global_orient = smplx_global_orient;
        optimization_inputs.init_body_pose = smplx_body_pose;
        optimization_inputs.triangulated_joints = triangulated_joints;
        optimization_inputs.triangulated_scores = triangulated_scores;

        if (OptimizeSmplxPoseFromTriangulatedJoints(&smplx_layer, optimization_inputs, &optimization_result)) {
            has_optimization_result = true;
            smplx_global_orient = PadFloatVector(optimization_result.global_orient,
                                                 static_cast<size_t>(kSmplxGlobalOrientParamCount));
            smplx_body_pose = PadFloatVector(optimization_result.body_pose,
                                             static_cast<size_t>(kSmplxBodyPoseParamCount));
            if (!optimization_result.optimized_betas.empty()) {
                smplx_shape = PadFloatVector(optimization_result.optimized_betas,
                                             static_cast<size_t>(kSmplxShapeParamCount));
            }
            if (target_sample.calibration != nullptr) {
                training_camera_translation =
                    target_sample.calibration->R * optimization_result.translation_world +
                    target_sample.calibration->t;
            }

            std::vector<float> pose_axis_angle(55u * 3u, 0.0f);
            if (smplx_global_orient.size() >= 3u) {
                std::copy_n(smplx_global_orient.begin(), 3u, pose_axis_angle.begin());
            }
            if (smplx_body_pose.size() >= 63u) {
                std::copy_n(smplx_body_pose.begin(), 63u, pose_axis_angle.begin() + 3u);
            }

            torch::NoGradGuard no_grad;
            torch::Tensor betas = torch::from_blob(
                                      smplx_shape.data(),
                                      {1, static_cast<int64_t>(smplx_shape.size())},
                                      torch::kFloat32)
                                      .clone();
            torch::Tensor expression = torch::from_blob(
                                           smplx_expression.data(),
                                           {1, static_cast<int64_t>(smplx_expression.size())},
                                           torch::kFloat32)
                                           .clone();
            torch::Tensor pose = torch::from_blob(
                                     pose_axis_angle.data(),
                                     {1, 55, 3},
                                     torch::kFloat32)
                                     .clone();
            torch::Tensor jaw = torch::from_blob(
                                    smplx_jaw_pose.data(),
                                    {1, static_cast<int64_t>(smplx_jaw_pose.size())},
                                    torch::kFloat32)
                                    .clone();
            torch::Tensor eye = torch::from_blob(
                                    smplx_eye_pose.data(),
                                    {1, static_cast<int64_t>(smplx_eye_pose.size())},
                                    torch::kFloat32)
                                    .clone();
            torch::Tensor left_hand = torch::from_blob(
                                          smplx_left_hand_pose.data(),
                                          {1, static_cast<int64_t>(smplx_left_hand_pose.size())},
                                          torch::kFloat32)
                                          .clone();
            torch::Tensor right_hand = torch::from_blob(
                                           smplx_right_hand_pose.data(),
                                           {1, static_cast<int64_t>(smplx_right_hand_pose.size())},
                                           torch::kFloat32)
                                           .clone();
            torch::Tensor trans = torch::tensor(
                                    {optimization_result.translation_world[0],
                                     optimization_result.translation_world[1],
                                     optimization_result.translation_world[2]},
                                    torch::kFloat32)
                                    .reshape({1, 3});

            SmplxOutput smplx_out = smplx_lbs->forward(
                betas,
                expression,
                pose,
                jaw,
                eye,
                left_hand,
                right_hand,
                trans);
            torch::Tensor verts_cpu = smplx_out.vertices.squeeze(0).to(torch::kCPU).contiguous();
            smplx_vertices_world.reserve(static_cast<size_t>(verts_cpu.size(0)));
            for (int64_t i = 0; i < verts_cpu.size(0); ++i) {
                smplx_vertices_world.emplace_back(
                    verts_cpu[i][0].item<float>(),
                    verts_cpu[i][1].item<float>(),
                    verts_cpu[i][2].item<float>());
            }
        }
#endif

    if (has_optimization_result && target_sample.calibration != nullptr) {
        smplx_global_orient = RotateRootAxisAngleToCamera(target_sample.calibration->R,
                                  smplx_global_orient);
    }

        if (options.temporal_smooth_alpha < 1.0f && has_prev_pose) {
            SmoothAxisAngleBlocks(prev_global_orient,
                                  &smplx_global_orient,
                                  options.temporal_smooth_alpha,
                                  kTemporalMaxRotationStepRad);
            SmoothAxisAngleBlocks(prev_body_pose,
                                  &smplx_body_pose,
                                  options.temporal_smooth_alpha,
                                  kTemporalMaxRotationStepRad);
            training_camera_translation = SmoothTranslationStep(prev_training_translation,
                                                                training_camera_translation,
                                                                options.temporal_smooth_alpha,
                                                                kTemporalMaxTranslationStep);
        }
        prev_global_orient = smplx_global_orient;
        prev_body_pose = smplx_body_pose;
        prev_training_translation = training_camera_translation;
        has_prev_pose = true;

        std::vector<float> train_pose(72u, 0.0f);
        std::copy_n(smplx_global_orient.begin(), 3, train_pose.begin());
        std::copy_n(smplx_body_pose.begin(), 63, train_pose.begin() + 3);

        cv::Mat full_overlay = target_sample.view->image.clone();
        if (options.show_rtmpose_points_overlay) {
            for (const auto& tri_view : triangulation_views) {
                if (tri_view.calibration == target_sample.calibration) {
                    for (size_t i = 0; i < tri_view.keypoints.size(); ++i) {
                        if (i < tri_view.scores.size() && tri_view.scores[i] < kTriangulationMinScore) {
                            continue;
                        }

                        const cv::Point2f& full_pixel = tri_view.keypoints[i];
                        if (full_pixel.x >= 0.0f && full_pixel.y >= 0.0f &&
                            full_pixel.x < static_cast<float>(full_overlay.cols) &&
                            full_pixel.y < static_cast<float>(full_overlay.rows)) {
                            cv::circle(full_overlay,
                                       cv::Point(static_cast<int>(std::lround(full_pixel.x)),
                                                 static_cast<int>(std::lround(full_pixel.y))),
                                       4,
                                       cv::Scalar(255, 0, 255),
                                       -1,
                                       cv::LINE_AA);
                        }

                        cv::Point2f crop_pixel = ApplyAffinePoint(target_sample.crop_trans, full_pixel);
                        if (crop_pixel.x >= 0.0f && crop_pixel.y >= 0.0f &&
                            crop_pixel.x < static_cast<float>(target_sample.crop_overlay.cols) &&
                            crop_pixel.y < static_cast<float>(target_sample.crop_overlay.rows)) {
                            cv::circle(target_sample.crop_overlay,
                                       cv::Point(static_cast<int>(std::lround(crop_pixel.x)),
                                                 static_cast<int>(std::lround(crop_pixel.y))),
                                       5,
                                       cv::Scalar(255, 0, 255),
                                       -1,
                                       cv::LINE_AA);
                        }
                    }
                    break;
                }
            }
        }

        if (target_sample.calibration != nullptr) {
            for (const auto& point_world : triangulated_joints) {
                if (!IsFinitePoint3(point_world)) {
                    continue;
                }

                cv::Point2f full_pixel;
                if (!ProjectWorldPointToImage(*target_sample.calibration, point_world, &full_pixel)) {
                    continue;
                }

                if (full_pixel.x >= 0.0f && full_pixel.y >= 0.0f &&
                    full_pixel.x < static_cast<float>(full_overlay.cols) &&
                    full_pixel.y < static_cast<float>(full_overlay.rows)) {
                    cv::circle(full_overlay,
                               cv::Point(static_cast<int>(std::lround(full_pixel.x)),
                                         static_cast<int>(std::lround(full_pixel.y))),
                               2,
                               cv::Scalar(0, 255, 0),
                               -1,
                               cv::LINE_AA);
                }

                cv::Point2f crop_pixel = ApplyAffinePoint(target_sample.crop_trans, full_pixel);
                if (crop_pixel.x >= 0.0f && crop_pixel.y >= 0.0f &&
                    crop_pixel.x < static_cast<float>(target_sample.crop_overlay.cols) &&
                    crop_pixel.y < static_cast<float>(target_sample.crop_overlay.rows)) {
                    cv::circle(target_sample.crop_overlay,
                               cv::Point(static_cast<int>(std::lround(crop_pixel.x)),
                                         static_cast<int>(std::lround(crop_pixel.y))),
                               3,
                               cv::Scalar(0, 165, 255),
                               -1,
                               cv::LINE_AA);
                }
            }

            if (options.show_smpl_joints_overlay && !optimization_result.optimized_joints.empty()) {
                for (const auto& bone : kSmplxBones) {
                    const int idx0 = bone.first;
                    const int idx1 = bone.second;
                    if (idx0 < 0 || idx1 < 0 ||
                        idx0 >= static_cast<int>(optimization_result.optimized_joints.size()) ||
                        idx1 >= static_cast<int>(optimization_result.optimized_joints.size())) {
                        continue;
                    }

                    const cv::Point3f& p0_world = optimization_result.optimized_joints[static_cast<size_t>(idx0)];
                    const cv::Point3f& p1_world = optimization_result.optimized_joints[static_cast<size_t>(idx1)];
                    cv::Point2f p0_full;
                    cv::Point2f p1_full;
                    if (!ProjectWorldPointToImage(*target_sample.calibration, p0_world, &p0_full) ||
                        !ProjectWorldPointToImage(*target_sample.calibration, p1_world, &p1_full)) {
                        continue;
                    }

                    cv::Point2f p0_crop = ApplyAffinePoint(target_sample.crop_trans, p0_full);
                    cv::Point2f p1_crop = ApplyAffinePoint(target_sample.crop_trans, p1_full);
                    cv::line(target_sample.crop_overlay,
                             cv::Point(static_cast<int>(std::lround(p0_crop.x)), static_cast<int>(std::lround(p0_crop.y))),
                             cv::Point(static_cast<int>(std::lround(p1_crop.x)), static_cast<int>(std::lround(p1_crop.y))),
                             cv::Scalar(255, 0, 0),
                             2,
                             cv::LINE_AA);
                }
            }

            if (options.show_smpl_verts_overlay && has_optimization_result && !smplx_vertices_world.empty()) {
                for (const auto& vertex_world : smplx_vertices_world) {
                    cv::Point2f full_pixel;
                    if (!ProjectWorldPointToImage(*target_sample.calibration, vertex_world, &full_pixel)) {
                        continue;
                    }

                    if (full_pixel.x >= 0.0f && full_pixel.y >= 0.0f &&
                        full_pixel.x < static_cast<float>(full_overlay.cols) &&
                        full_pixel.y < static_cast<float>(full_overlay.rows)) {
                        cv::circle(full_overlay,
                                   cv::Point(static_cast<int>(std::lround(full_pixel.x)),
                                             static_cast<int>(std::lround(full_pixel.y))),
                                   1,
                                   cv::Scalar(255, 200, 0),
                                   -1,
                                   cv::LINE_AA);
                    }

                    cv::Point2f crop_pixel = ApplyAffinePoint(target_sample.crop_trans, full_pixel);
                    if (crop_pixel.x >= 0.0f && crop_pixel.y >= 0.0f &&
                        crop_pixel.x < static_cast<float>(target_sample.crop_overlay.cols) &&
                        crop_pixel.y < static_cast<float>(target_sample.crop_overlay.rows)) {
                        cv::circle(target_sample.crop_overlay,
                                   cv::Point(static_cast<int>(std::lround(crop_pixel.x)),
                                             static_cast<int>(std::lround(crop_pixel.y))),
                                   1,
                                   cv::Scalar(255, 255, 0),
                                   -1,
                                   cv::LINE_AA);
                    }
                }
            }
        }

        const std::array<float, 3> train_camera = BuildTrainingCameraParams(
            training_camera_translation,
            target_sample.training_crop,
            target_sample.view->image.cols,
            target_sample.view->image.rows);

        ExportTrainingSample export_sample;
        export_sample.camera_id = target_sample.view->camera_id;
        export_sample.source_camera_index = target_sample.view->source_camera_index;
        export_sample.video_frame_index = target_sample.view->video_frame_index;
        export_sample.person_index = 0;
        export_sample.person_id = 0;
        export_sample.crop_image = target_sample.crop_image_to_save;
        export_sample.crop_matte = target_sample.crop_matte;
        export_sample.crop_overlay = target_sample.crop_overlay;
        export_sample.img_w = static_cast<float>(target_sample.view->image.cols);
        export_sample.img_h = static_cast<float>(target_sample.view->image.rows);
        export_sample.crop_cx = target_sample.training_crop.crop_cx;
        export_sample.crop_cy = target_sample.training_crop.crop_cy;

        float S = target_sample.training_crop.crop_w / target_sample.training_crop.crop_size;
        export_sample.crop_size = target_sample.training_crop.crop_w;
        export_sample.crop_w = target_sample.training_crop.crop_w;
        export_sample.crop_h = target_sample.training_crop.crop_h;
        export_sample.focal_length = target_sample.training_crop.focal_length * S;
        export_sample.crop_x0 = target_sample.training_crop.crop_x0;
        export_sample.crop_y0 = target_sample.training_crop.crop_y0;

        if (target_sample.calibration != nullptr) {
            export_sample.full_fx = target_sample.calibration->K(0, 0);
            export_sample.full_fy = target_sample.calibration->K(1, 1);
            export_sample.full_cx = target_sample.calibration->K(0, 2);
            export_sample.full_cy = target_sample.calibration->K(1, 2);
        } else {
            export_sample.full_fx = target_sample.training_crop.focal_length;
            export_sample.full_fy = target_sample.training_crop.focal_length;
            export_sample.full_cx = export_sample.img_w * 0.5f;
            export_sample.full_cy = export_sample.img_h * 0.5f;
        }

        export_sample.crop_affine = {
            target_sample.crop_trans(0, 0), target_sample.crop_trans(0, 1), target_sample.crop_trans(0, 2),
            target_sample.crop_trans(1, 0), target_sample.crop_trans(1, 1), target_sample.crop_trans(1, 2)};

        export_sample.y_sign = 1.0f;
        export_sample.cam = {train_camera[0], train_camera[1], train_camera[2]};
        export_sample.pose = train_pose;
        export_sample.betas = PadFloatVector(smplx_shape, 10u);
        export_sample.body_model = "smplx";

        export_sample.translation = {
            training_camera_translation[0],
            training_camera_translation[1],
            training_camera_translation[2]};
        export_sample.body_model = "smplx";
        export_sample.smplx_shape = smplx_shape;
        export_sample.smplx_expression = smplx_expression;
        export_sample.smplx_global_orient = smplx_global_orient;
        export_sample.smplx_body_pose = smplx_body_pose;
        export_sample.smplx_jaw_pose = smplx_jaw_pose;
        export_sample.smplx_eye_pose = smplx_eye_pose;
        export_sample.smplx_left_hand_pose = smplx_left_hand_pose;
        export_sample.smplx_right_hand_pose = smplx_right_hand_pose;

        ExportFrameArtifacts artifacts;
        artifacts.synced_frames = synced_frames;
        artifacts.training_export_requested = true;
        artifacts.training_samples.push_back(std::move(export_sample));
        artifacts.full_overlays.resize(synced_frames.views.size());
        artifacts.full_overlays[target_sample.synced_view_index] = full_overlay;
        artifacts.triangulated_joints = triangulated_joints;
        artifacts.triangulated_scores = triangulated_scores;

        if (!exporter.SaveFrame(artifacts)) {
            std::cerr << "Failed to export frame " << synced_frames.sync_index << "\n";
            continue;
        }

        ++exported_frame_count;
    }

    std::cout << "Finished exporting " << exported_frame_count
              << " frames from " << selected_frame_count
              << " selected synchronized frames (stride " << options.frame_stride
              << ", start_frame_index " << options.start_frame_index
              << ", feeds " << options.sources.size() << ")." << std::endl;
    return 0;
}
