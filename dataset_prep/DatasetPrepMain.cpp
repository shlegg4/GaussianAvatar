#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "dataset_prep/processing/PearEstimator.h"
#include "dataset_prep/processing/BackgroundExtractor.h" 
#include "utils/YoloPersonDetector.h"
#if DATASET_PREP_HAS_SMPL
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace fs = std::filesystem;
using namespace dataset_prep;

// --- UTILITY FUNCTIONS ---

namespace {

constexpr int kPearInputRes = 256;
constexpr float kPearCameraFocal = 24.0f;
constexpr int64_t kSmplxShapeParamCount = 10;
constexpr int64_t kSmplxExpressionParamCount = 240;
constexpr int64_t kSmplxGlobalOrientParamCount = 3;
constexpr int64_t kSmplxBodyPoseParamCount = 63;
constexpr int64_t kSmplxJawPoseParamCount = 3;
constexpr int64_t kSmplxEyePoseParamCount = 6;
constexpr int64_t kSmplxHandPoseParamCount = 45;

std::vector<float> PadFloatVector(const std::vector<float>& values, size_t expected_count) {
    std::vector<float> padded(expected_count, 0.0f);
    const size_t copy_count = std::min(values.size(), expected_count);
    std::copy_n(values.begin(), copy_count, padded.begin());
    return padded;
}

#if DATASET_PREP_HAS_SMPL
torch::Tensor MakeTorchInputTensor(const std::vector<float>& values,
                                   int64_t expected_count,
                                   const torch::Device& device) {
    std::vector<float> padded = PadFloatVector(values, static_cast<size_t>(expected_count));
    auto tensor = torch::from_blob(
        padded.data(),
        {1, expected_count},
        torch::TensorOptions().dtype(torch::kFloat32));
    return tensor.clone().to(device);
}
#endif

bool SanitizeBBox(const cv::Rect2f& bbox, int img_width, int img_height, cv::Rect2f* out_bbox) {
    if (out_bbox == nullptr) {
        return false;
    }

    const float x1 = std::max(0.0f, bbox.x);
    const float y1 = std::max(0.0f, bbox.y);
    const float x2 = std::min(static_cast<float>(img_width - 1),
                              x1 + std::max(0.0f, bbox.width - 1.0f));
    const float y2 = std::min(static_cast<float>(img_height - 1),
                              y1 + std::max(0.0f, bbox.height - 1.0f));
    if (bbox.width * bbox.height <= 0.0f || x2 <= x1 || y2 <= y1) {
        return false;
    }

    *out_bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    return true;
}

bool ProcessBBox(const cv::Rect2f& bbox,
                 int img_width,
                 int img_height,
                 const cv::Size& input_shape,
                 float ratio,
                 cv::Rect2f* out_bbox) {
    if (out_bbox == nullptr) {
        return false;
    }

    cv::Rect2f clipped_bbox;
    if (!SanitizeBBox(bbox, img_width, img_height, &clipped_bbox)) {
        return false;
    }

    float w = clipped_bbox.width;
    float h = clipped_bbox.height;
    const float c_x = clipped_bbox.x + w * 0.5f;
    const float c_y = clipped_bbox.y + h * 0.5f;
    const float aspect_ratio = static_cast<float>(input_shape.width) /
                               static_cast<float>(input_shape.height);
    if (w > aspect_ratio * h) {
        h = w / aspect_ratio;
    } else if (w < aspect_ratio * h) {
        w = h * aspect_ratio;
    }

    const float expanded_w = w * ratio;
    const float expanded_h = h * ratio;
    *out_bbox = cv::Rect2f(c_x - expanded_w * 0.5f,
                           c_y - expanded_h * 0.5f,
                           expanded_w,
                           expanded_h);
    return true;
}

cv::Point2f Rotate2D(const cv::Point2f& point, float rot_rad) {
    const float sn = std::sin(rot_rad);
    const float cs = std::cos(rot_rad);
    return cv::Point2f(point.x * cs - point.y * sn,
                       point.x * sn + point.y * cs);
}

cv::Matx23f GenTransFromPatchCv(float c_x,
                                float c_y,
                                float src_width,
                                float src_height,
                                int dst_width,
                                int dst_height,
                                float scale,
                                float rot_deg,
                                bool inv = false) {
    const float src_w = src_width * scale;
    const float src_h = src_height * scale;
    const cv::Point2f src_center(c_x, c_y);
    const float rot_rad = static_cast<float>(CV_PI) * rot_deg / 180.0f;
    const cv::Point2f src_downdir = Rotate2D(cv::Point2f(0.0f, src_h * 0.5f), rot_rad);
    const cv::Point2f src_rightdir = Rotate2D(cv::Point2f(src_w * 0.5f, 0.0f), rot_rad);

    const cv::Point2f dst_center(dst_width * 0.5f, dst_height * 0.5f);
    const cv::Point2f dst_downdir(0.0f, dst_height * 0.5f);
    const cv::Point2f dst_rightdir(dst_width * 0.5f, 0.0f);

    std::array<cv::Point2f, 3> src = {
        src_center,
        src_center + src_downdir,
        src_center + src_rightdir,
    };
    std::array<cv::Point2f, 3> dst = {
        dst_center,
        dst_center + dst_downdir,
        dst_center + dst_rightdir,
    };

    const cv::Mat affine = inv
        ? cv::getAffineTransform(dst.data(), src.data())
        : cv::getAffineTransform(src.data(), dst.data());
    cv::Matx23d affine_d;
    affine.copyTo(affine_d);
    return cv::Matx23f(static_cast<float>(affine_d(0, 0)),
                       static_cast<float>(affine_d(0, 1)),
                       static_cast<float>(affine_d(0, 2)),
                       static_cast<float>(affine_d(1, 0)),
                       static_cast<float>(affine_d(1, 1)),
                       static_cast<float>(affine_d(1, 2)));
}

bool GeneratePatchImage(const cv::Mat& image,
                        const cv::Rect2f& bbox,
                        float scale,
                        float rot_deg,
                        bool do_flip,
                        const cv::Size& out_shape,
                        cv::Mat* out_patch,
                        cv::Matx23f* out_trans,
                        cv::Matx23f* out_inv_trans) {
    if (out_patch == nullptr || image.empty()) {
        return false;
    }

    cv::Mat src = image;
    float bb_c_x = bbox.x + bbox.width * 0.5f;
    const float bb_c_y = bbox.y + bbox.height * 0.5f;
    const float bb_width = bbox.width;
    const float bb_height = bbox.height;

    if (do_flip) {
        cv::flip(image, src, 1);
        bb_c_x = static_cast<float>(image.cols) - bb_c_x - 1.0f;
    }

    const cv::Matx23f trans = GenTransFromPatchCv(
        bb_c_x, bb_c_y, bb_width, bb_height,
        out_shape.width, out_shape.height, scale, rot_deg, false);
    cv::warpAffine(src,
                   *out_patch,
                   cv::Mat(trans),
                   out_shape,
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    if (out_trans != nullptr) {
        *out_trans = trans;
    }
    if (out_inv_trans != nullptr) {
        *out_inv_trans = GenTransFromPatchCv(
            bb_c_x, bb_c_y, bb_width, bb_height,
            out_shape.width, out_shape.height, scale, rot_deg, true);
    }
    return true;
}

cv::Matx33f DefaultPearCameraRotation() {
    return cv::Matx33f(-1.0f, 0.0f, 0.0f,
                       0.0f, -1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Matx33f ProjectionSignFlip() {
    return cv::Matx33f(-1.0f, 0.0f, 0.0f,
                       0.0f, -1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Matx33f ExtractPearCameraRotation(const SmplxResult& pear_result) {
    if (pear_result.camera_rt.size() >= 16u) {
        return cv::Matx33f(pear_result.camera_rt[0], pear_result.camera_rt[1], pear_result.camera_rt[2],
                           pear_result.camera_rt[4], pear_result.camera_rt[5], pear_result.camera_rt[6],
                           pear_result.camera_rt[8], pear_result.camera_rt[9], pear_result.camera_rt[10]);
    }
    return DefaultPearCameraRotation();
}

cv::Vec3f ExtractPearCameraTranslation(const SmplxResult& pear_result) {
    if (pear_result.camera_rt.size() >= 16u) {
        return cv::Vec3f(pear_result.camera_rt[3],
                         pear_result.camera_rt[7],
                         pear_result.camera_rt[11]);
    }
    if (pear_result.camera_translation.size() >= 3u) {
        return cv::Vec3f(pear_result.camera_translation[0],
                         pear_result.camera_translation[1],
                         pear_result.camera_translation[2]);
    }
    return cv::Vec3f(0.0f, 0.0f, 0.0f);
}

cv::Point2f ProjectPearCameraPoint(const cv::Vec3f& point_cam, int width, int height) {
    const float half_w = static_cast<float>(width) * 0.5f;
    const float half_h = static_cast<float>(height) * 0.5f;
    const float focal_x = half_w * kPearCameraFocal;
    const float focal_y = half_h * kPearCameraFocal;
    return cv::Point2f(half_w - focal_x * (point_cam[0] / point_cam[2]),
                       half_h - focal_y * (point_cam[1] / point_cam[2]));
}

void ApplyExtraRootRotation(const cv::Matx33f& extra_rotation, std::vector<float>* pose_axis_angle) {
    if (pose_axis_angle == nullptr || pose_axis_angle->size() < 3u) {
        return;
    }

    cv::Vec3f root_aa((*pose_axis_angle)[0], (*pose_axis_angle)[1], (*pose_axis_angle)[2]);
    cv::Matx33f root_rot;
    cv::Rodrigues(root_aa, root_rot);

    const cv::Matx33f rotated_root = extra_rotation * root_rot;
    cv::Vec3f rotated_root_aa;
    cv::Rodrigues(cv::Mat(rotated_root), rotated_root_aa);

    (*pose_axis_angle)[0] = rotated_root_aa[0];
    (*pose_axis_angle)[1] = rotated_root_aa[1];
    (*pose_axis_angle)[2] = rotated_root_aa[2];
}

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

TrainingCropMetadata BuildTrainingCropMetadata(const cv::Rect2f& crop_bbox,
                                               int img_width,
                                               int img_height,
                                               int target_crop_res) {
    TrainingCropMetadata metadata;

    if (crop_bbox.width <= 1e-6f || crop_bbox.height <= 1e-6f || target_crop_res <= 0) {
        return metadata;
    }

    const float full_cx = static_cast<float>(img_width) * 0.5f;
    const float full_cy = static_cast<float>(img_height) * 0.5f;
    const float crop_half = static_cast<float>(target_crop_res) * 0.5f;

    // The saved PEAR crop is an affine-normalized patch with a centered virtual
    // camera, not a literal window cut from the full-frame camera. Export the
    // crop metadata in that centered virtual-camera convention so the training
    // renderer uses principal point (target_crop_res / 2, target_crop_res / 2).
    metadata.crop_cx = full_cx;
    metadata.crop_cy = full_cy;
    metadata.crop_size = static_cast<float>(target_crop_res);
    metadata.crop_x0 = full_cx - crop_half;
    metadata.crop_y0 = full_cy - crop_half;
    metadata.crop_w = static_cast<float>(target_crop_res);
    metadata.crop_h = static_cast<float>(target_crop_res);
    metadata.focal_length = 0.5f * static_cast<float>(target_crop_res) * kPearCameraFocal;
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

cv::Mat ApplyCropMatte(const cv::Mat& crop_image, const cv::Mat& crop_matte) {
    if (crop_image.empty() || crop_matte.empty()) {
        return crop_image.clone();
    }

    cv::Mat matte_resized;
    if (crop_matte.size() != crop_image.size()) {
        cv::resize(crop_matte, matte_resized, crop_image.size(), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        matte_resized = crop_matte;
    }

    cv::Mat matte_f32;
    if (matte_resized.type() == CV_32F) {
        matte_f32 = matte_resized;
    } else {
        matte_resized.convertTo(matte_f32, CV_32F, 1.0 / 255.0);
    }

    cv::Mat matte_bgr;
    if (matte_f32.channels() == 1) {
        cv::cvtColor(matte_f32, matte_bgr, cv::COLOR_GRAY2BGR);
    } else {
        matte_bgr = matte_f32;
    }

    cv::Mat crop_f32;
    crop_image.convertTo(crop_f32, CV_32F, 1.0 / 255.0);

    cv::Mat matted_f32 = crop_f32.mul(matte_bgr);
    cv::Mat matted_crop;
    matted_f32.convertTo(matted_crop, crop_image.type(), 255.0);
    return matted_crop;
}

}  // namespace

cv::Mat CropSquareWithPaddingAndResize(const cv::Mat& src, int x, int y, int size, int target_res) {
    if (src.empty() || size <= 0 || target_res <= 0) return {};
    const cv::Rect roi(x, y, size, size);
    const cv::Rect image_bounds(0, 0, src.cols, src.rows);
    const cv::Rect valid_roi = roi & image_bounds;

    cv::Mat square = cv::Mat::zeros(size, size, src.type());
    if (valid_roi.area() > 0) {
        const cv::Rect dst_roi(valid_roi.x - roi.x, valid_roi.y - roi.y, valid_roi.width, valid_roi.height);
        src(valid_roi).copyTo(square(dst_roi));
    }
    
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(target_res, target_res), 0.0, 0.0, 
               size > target_res ? cv::INTER_AREA : cv::INTER_CUBIC);
    return resized;
}

std::vector<float> ConvertPearPoseToSmplAxisAngle(const SmplxResult& pear_result) {
    std::vector<float> pose_values(72, 0.0f);
    if (pear_result.global_orient.size() == 3u && pear_result.body_pose.size() >= 63u) {
        std::copy_n(pear_result.global_orient.begin(), 3u, pose_values.begin());
        std::copy_n(pear_result.body_pose.begin(), 63u, pose_values.begin() + 3);
    }
    return pose_values;
}

std::string JsonEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8u);
    for (char c : input) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

void WriteFloatArray(std::ostream& out, const std::vector<float>& values, size_t expected_count) {
    out << "[";
    for (size_t index = 0; index < expected_count; ++index) {
        if (index > 0u) out << ", ";
        out << (index < values.size() ? values[index] : 0.0f);
    }
    out << "]";
}

// --- MAIN PIPELINE ---

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: dataset_prep <video.mp4> <output_dir> <camera_id> [max_frames] [frame_stride]\n";
        return 1;
    }

    const fs::path video_path = argv[1];
    const fs::path output_dir = argv[2];
    const std::string camera_id = argv[3];
    const int max_frames = (argc > 4) ? std::stoi(argv[4]) : -1;
    const int frame_stride = (argc > 5) ? std::stoi(argv[5]) : 1;
    if (frame_stride <= 0) {
        std::cerr << "frame_stride must be >= 1.\n";
        return 1;
    }

    const int target_crop_res = 1024;
    const float crop_margin = 1.25f;
    // Initialize Outputs
    fs::create_directories(output_dir / "crops" / camera_id);
    fs::create_directories(output_dir / "overlays" / camera_id);
    fs::create_directories(output_dir / "overlays_full" / camera_id);
    fs::create_directories(output_dir / "mattes" / camera_id); // <-- NEW: Mattes Output Dir
    std::ofstream jsonl_out(output_dir / "frames.jsonl", std::ios::app);
    jsonl_out << std::fixed << std::setprecision(6);

    // Initialize Models
    YoloPersonDetectorOptions yolo_opts;
    yolo_opts.conf_threshold = 0.5f;
    yolo_opts.use_cuda = true;
    YoloPersonDetector yolo(yolo_opts);
    if (!yolo.Load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\yolov8n.onnx")) {
        std::cerr << "Failed to load YOLO model.\n";
        return 1;
    }

    PearEstimator::Options pear_opts;
    pear_opts.model_path = "C:\\Users\\Sam\\Documents\\GaussianAvatar\\pear_ehm.onnx";
    pear_opts.use_cuda = true;
    PearEstimator pear(pear_opts);
    if (!pear.Initialize()) {
        std::cerr << "Failed to load PEAR model.\n";
        return 1;
    }

    // <-- NEW: Initialize MODNet
    BackgroundExtractor::Options modnet_opts;
    modnet_opts.model_path = "C:\\Users\\Sam\\Documents\\GaussianAvatar\\modnet.onnx";
    modnet_opts.use_cuda = true;
    modnet_opts.input_size = target_crop_res;
    modnet_opts.binary_threshold = -1.0f;
    BackgroundExtractor modnet(modnet_opts);
    if (!modnet.Initialize()) {
        std::cerr << "Failed to load MODNet model.\n";
        return 1;
    }

#if DATASET_PREP_HAS_SMPL
    torch::jit::script::Module smplx_layer;
    try {
        smplx_layer = torch::jit::load("C:\\Users\\Sam\\Documents\\GaussianAvatar\\smplx_libtorch.pt");
    } catch (const c10::Error& error) {
        std::cerr << "Failed to load SMPL-X TorchScript module.\n"
                  << error.what() << std::endl;
        return 1;
    }
    smplx_layer.to(torch::kCPU);
    smplx_layer.eval();
#endif

    // Process Video
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video.\n";
        return 1;
    }

    cv::Mat frame;
    int frame_index = 0;
    int selected_frame_count = 0;
    int exported_frame_count = 0;

    while (cap.read(frame)) {
        const int current_frame_index = frame_index++;
        if ((current_frame_index % frame_stride) != 0) {
            continue;
        }
        if (max_frames >= 0 && selected_frame_count >= max_frames) {
            break;
        }
        ++selected_frame_count;

        std::cout << "Processing frame " << current_frame_index << "..." << std::endl;

        cv::Rect2f best_bbox;
        float best_score = 0.0f;
        yolo.DetectPerson(frame, &best_bbox, &best_score);

        if (best_score > 0.0f && best_bbox.area() > 0) {
            // 1. Calculate Crop
            cv::Rect2f crop_bbox;
            if (!ProcessBBox(best_bbox,
                             frame.cols,
                             frame.rows,
                             cv::Size(kPearInputRes, kPearInputRes),
                             crop_margin,
                             &crop_bbox)) {
                continue;
            }

            const TrainingCropMetadata training_crop = BuildTrainingCropMetadata(
                crop_bbox, frame.cols, frame.rows, target_crop_res);

            cv::Mat pear_crop_image;
            if (!GeneratePatchImage(frame,
                                    crop_bbox,
                                    1.0f,
                                    0.0f,
                                    false,
                                    cv::Size(kPearInputRes, kPearInputRes),
                                    &pear_crop_image,
                                    nullptr,
                                    nullptr)) {
                continue;
            }

            cv::Mat crop_image;
            cv::Matx23f crop_inv_trans(1.0f, 0.0f, 0.0f,
                                       0.0f, 1.0f, 0.0f);
            if (!GeneratePatchImage(frame,
                                    crop_bbox,
                                    1.0f,
                                    0.0f,
                                    false,
                                    cv::Size(target_crop_res, target_crop_res),
                                    &crop_image,
                                    nullptr,
                                    &crop_inv_trans)) {
                continue;
            }

            // 1.5. Extract Matte using MODNet <-- NEW
            cv::Mat crop_matte;
            if (!modnet.ProcessImage(crop_image, &crop_matte)) {
                std::cerr << "Warning: Failed to extract matte on frame " << current_frame_index << "\n";
            }
            cv::Mat crop_image_to_save = ApplyCropMatte(crop_image, crop_matte);
            cv::Mat pear_input_image = ApplyCropMatte(pear_crop_image, crop_matte);

            // 2. Run PEAR
            SmplxResult pear_result;
            if (pear.Estimate(pear_input_image, &pear_result)) {
                // 3. Coordinate Math
                const cv::Matx33f camera_rotation = ExtractPearCameraRotation(pear_result);
                const cv::Vec3f camera_translation = ExtractPearCameraTranslation(pear_result);
                std::vector<float> render_pose_local = ConvertPearPoseToSmplAxisAngle(pear_result);
                std::vector<float> train_pose_local = render_pose_local;
                const cv::Matx33f train_rotation = ProjectionSignFlip() * camera_rotation;
                const cv::Vec3f train_translation = ProjectionSignFlip() * camera_translation;
                ApplyExtraRootRotation(train_rotation, &train_pose_local);
                const std::array<float, 3> train_camera = BuildTrainingCameraParams(
                    train_translation, training_crop, frame.cols, frame.rows);
                const std::vector<float> smplx_global_orient =
                    PadFloatVector(pear_result.global_orient, static_cast<size_t>(kSmplxGlobalOrientParamCount));
                const std::vector<float> smplx_body_pose =
                    PadFloatVector(pear_result.body_pose, static_cast<size_t>(kSmplxBodyPoseParamCount));
                const std::vector<float> smplx_jaw_pose =
                    PadFloatVector(pear_result.jaw_pose, static_cast<size_t>(kSmplxJawPoseParamCount));
                const std::vector<float> smplx_eye_pose(
                    static_cast<size_t>(kSmplxEyePoseParamCount), 0.0f);
                const std::vector<float> smplx_left_hand_pose =
                    PadFloatVector(pear_result.left_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
                const std::vector<float> smplx_right_hand_pose =
                    PadFloatVector(pear_result.right_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
                const std::vector<float> smplx_expression =
                    PadFloatVector(pear_result.expression, static_cast<size_t>(kSmplxExpressionParamCount));

                // 4. Render Overlay
                cv::Mat crop_overlay = crop_image.clone();
                cv::Mat full_overlay = frame.clone();
                cv::Mat crop_mesh_overlay = cv::Mat::zeros(crop_overlay.size(), crop_overlay.type());
                 
#if DATASET_PREP_HAS_SMPL
                torch::NoGradGuard no_grad;

                // The exported SMPL-X TorchScript wrapper was traced on CPU tensors.
                // Keep dataset_prep inference on CPU here to avoid device-mismatch issues.
                const torch::Device smplx_device(torch::kCPU);

                if (!pear_result.betas.empty()) {
                    std::cout << "PEAR betas: [";
                    for (size_t beta_index = 0; beta_index < pear_result.betas.size(); ++beta_index) {
                        if (beta_index > 0u) {
                            std::cout << ", ";
                        }
                        std::cout << pear_result.betas[beta_index];
                    }
                    std::cout << "]\n";
                }

                std::vector<torch::jit::IValue> smplx_inputs;
                smplx_inputs.emplace_back(MakeTorchInputTensor(pear_result.betas,
                                                               kSmplxShapeParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_expression,
                                                               kSmplxExpressionParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_global_orient,
                                                               kSmplxGlobalOrientParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_body_pose,
                                                               kSmplxBodyPoseParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_jaw_pose,
                                                               kSmplxJawPoseParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_eye_pose,
                                                               kSmplxEyePoseParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_left_hand_pose,
                                                               kSmplxHandPoseParamCount,
                                                               smplx_device));
                smplx_inputs.emplace_back(MakeTorchInputTensor(smplx_right_hand_pose,
                                                               kSmplxHandPoseParamCount,
                                                               smplx_device));

                const auto smplx_out = smplx_layer.forward(smplx_inputs).toTuple();
                if (!smplx_out || smplx_out->elements().size() < 3u) {
                    std::cerr << "Warning: SMPL-X forward returned an unexpected output tuple on frame "
                              << current_frame_index << "\n";
                    continue;
                }

                auto verts_tensor = smplx_out->elements()[0].toTensor().squeeze(0).to(torch::kCPU).contiguous();
                auto joints_tensor = smplx_out->elements()[2].toTensor().squeeze(0).to(torch::kCPU).contiguous();
                auto joints_acc = joints_tensor.accessor<float, 2>();
                if (joints_acc.size(0) > 0) {
                    std::cout << "SMPL-X pelvis root joint 3D before verts_acc: ("
                              << joints_acc[0][0] << ", "
                              << joints_acc[0][1] << ", "
                              << joints_acc[0][2] << ")\n";
                }
                auto verts_acc = verts_tensor.accessor<float, 2>();

                std::vector<cv::Point> projected_vertices(
                    static_cast<size_t>(verts_acc.size(0)),
                    cv::Point(-1, -1));
                for (int i = 0; i < verts_acc.size(0); ++i) {
                    const cv::Vec3f point_model(verts_acc[i][0], verts_acc[i][1], verts_acc[i][2]);
                    const cv::Vec3f point_cam = camera_rotation * point_model + camera_translation;
                     
                    if (point_cam[2] > 1e-6f) {
                        const cv::Point2f projected = ProjectPearCameraPoint(
                            point_cam, crop_overlay.cols, crop_overlay.rows);
                        projected_vertices[i] = cv::Point(
                            static_cast<int>(std::lround(projected.x)),
                            static_cast<int>(std::lround(projected.y)));
                    }
                }

                for (const auto& pt : projected_vertices) {
                    if (pt.x >= 0 && pt.y >= 0 && pt.x < crop_overlay.cols && pt.y < crop_overlay.rows) {
                        cv::circle(crop_overlay, pt, 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                        cv::circle(crop_mesh_overlay, pt, 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                    }
                }

                // cv::Mat warped_overlay = cv::Mat::zeros(full_overlay.size(), full_overlay.type());
                // cv::warpAffine(crop_mesh_overlay,
                //                warped_overlay,
                //                cv::Mat(crop_inv_trans),
                //                full_overlay.size(),
                //                cv::INTER_LINEAR,
                //                cv::BORDER_CONSTANT,
                //                cv::Scalar(0, 0, 0));
                // cv::Mat mask;
                // cv::cvtColor(warped_overlay, mask, cv::COLOR_BGR2GRAY);
                // warped_overlay.copyTo(full_overlay, mask);
#endif

                // 5. Save Artifacts
                std::ostringstream stem;
                stem << "frame_" << std::setw(6) << std::setfill('0') << current_frame_index;
                
                std::string crop_path = (output_dir / "crops" / camera_id / ("crop_" + stem.str() + ".png")).string();
                std::string overlay_path = (output_dir / "overlays" / camera_id / ("overlay_" + stem.str() + ".png")).string();
                std::string full_path = (output_dir / "overlays_full" / camera_id / (stem.str() + ".jpg")).string();
                std::string matte_path = (output_dir / "mattes" / camera_id / ("matte_" + stem.str() + ".png")).string(); // <-- NEW
                
                cv::imwrite(crop_path, crop_image_to_save);
                cv::imwrite(overlay_path, crop_overlay);
                cv::imwrite(full_path, full_overlay);
                if (!crop_matte.empty()) {
                    cv::imwrite(matte_path, crop_matte); // <-- NEW
                }

                // 6. Write to JSONL
                jsonl_out << "{\"frame\":" << current_frame_index
                          << ",\"sync_index\":" << current_frame_index
                          << ",\"camera_id\":\"" << JsonEscape(camera_id) << "\""
                          << ",\"person_id\":0"
                          << ",\"video_frame_index\":" << current_frame_index
                          << ",\"crop\":\"" << JsonEscape(crop_path) << "\""
                          << ",\"mask\":\"" << JsonEscape(matte_path) << "\"" // <-- NEW: Added mask to JSON metadata
                          << ",\"overlay\":\"" << JsonEscape(overlay_path) << "\""
                          << ",\"img_w\":" << frame.cols
                          << ",\"img_h\":" << frame.rows
                          << ",\"crop_cx\":" << training_crop.crop_cx
                          << ",\"crop_cy\":" << training_crop.crop_cy
                          << ",\"crop_size\":" << training_crop.crop_size
                          << ",\"crop_x0\":" << training_crop.crop_x0
                          << ",\"crop_y0\":" << training_crop.crop_y0
                          << ",\"crop_w\":" << training_crop.crop_w
                          << ",\"crop_h\":" << training_crop.crop_h
                          << ",\"focal_length\":" << training_crop.focal_length
                          << ",\"y_sign\": 1.0" 
                          << ",\"pose\":";
                WriteFloatArray(jsonl_out, train_pose_local, 72u);
                jsonl_out << ",\"betas\":";
                WriteFloatArray(jsonl_out, pear_result.betas, 10u);
                jsonl_out << ",\"cam\":["
                          << train_camera[0] << ", "
                          << train_camera[1] << ", "
                          << train_camera[2] << "]";
                jsonl_out << ",\"body_model\":\"smplx\"";
                jsonl_out << ",\"smplx_shape\":";
                WriteFloatArray(jsonl_out, pear_result.betas, static_cast<size_t>(kSmplxShapeParamCount));
                jsonl_out << ",\"smplx_expression\":";
                WriteFloatArray(jsonl_out, smplx_expression, static_cast<size_t>(kSmplxExpressionParamCount));
                jsonl_out << ",\"smplx_global_orient\":";
                WriteFloatArray(jsonl_out, smplx_global_orient, static_cast<size_t>(kSmplxGlobalOrientParamCount));
                jsonl_out << ",\"smplx_body_pose\":";
                WriteFloatArray(jsonl_out, smplx_body_pose, static_cast<size_t>(kSmplxBodyPoseParamCount));
                jsonl_out << ",\"smplx_jaw_pose\":";
                WriteFloatArray(jsonl_out, smplx_jaw_pose, static_cast<size_t>(kSmplxJawPoseParamCount));
                jsonl_out << ",\"smplx_eye_pose\":";
                WriteFloatArray(jsonl_out, smplx_eye_pose, static_cast<size_t>(kSmplxEyePoseParamCount));
                jsonl_out << ",\"smplx_left_hand_pose\":";
                WriteFloatArray(jsonl_out, smplx_left_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
                jsonl_out << ",\"smplx_right_hand_pose\":";
                WriteFloatArray(jsonl_out, smplx_right_hand_pose, static_cast<size_t>(kSmplxHandPoseParamCount));
                if (!pear_result.camera_rt.empty()) {
                    jsonl_out << ",\"camera_rt\":";
                    WriteFloatArray(jsonl_out, pear_result.camera_rt, 16u);
                }
                jsonl_out
                           << "}\n";
                ++exported_frame_count;
            }
        }
    }

    std::cout << "Finished exporting " << exported_frame_count
              << " frames from " << selected_frame_count
              << " selected source frames (read " << frame_index
              << ", stride " << frame_stride << ")." << std::endl;
    return 0;
}
