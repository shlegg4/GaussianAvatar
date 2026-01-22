#include "HmrInferenceUtils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <torch/torch.h>
#include "HmrInferenceConstants.h"
#include "HmrMathHelpers.h"
#include "HmrOverlayHelpers.h"
#include "SmplifyLite.h"
#include "SmplLBS.h"
#include "TrtBuilder.h"
#include "YoloPersonDetector.h"

namespace {

const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
const std::vector<float> kStd  = {0.229f, 0.224f, 0.225f};

cv::Mat MakeModelInputBgr(const cv::Mat& img) {
    const int w = img.cols;
    const int h = img.rows;
    const float scale = std::min(static_cast<float>(kInputW) / w,
                                 static_cast<float>(kInputH) / h);
    const int new_w = static_cast<int>(std::round(w * scale));
    const int new_h = static_cast<int>(std::round(h * scale));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    const int pad_w = kInputW - new_w;
    const int pad_h = kInputH - new_h;
    const int left = pad_w / 2;
    const int top = pad_h / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, pad_h - top, left, pad_w - left,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

std::vector<float> GetCliffBBox(float cx, float cy, float box_size, float focal, int img_w, int img_h) {
    float x = cx - img_w / 2.0f;
    float y = cy - img_h / 2.0f;

    std::vector<float> bbox(3);
    bbox[0] = (x / focal) * 2.8f;
    bbox[1] = (y / focal) * 2.8f;
    bbox[2] = (box_size - 0.24f * focal) / (0.06f * focal);
    return bbox;
}

cv::Rect MakePaddedRect(const cv::Rect2f& bbox, int img_w, int img_h, float scale) {
    const float cx = bbox.x + bbox.width * 0.5f;
    const float cy = bbox.y + bbox.height * 0.5f;
    const float half_w = 0.5f * bbox.width * scale;
    const float half_h = 0.5f * bbox.height * scale;

    const int x0 = std::max(0, static_cast<int>(std::floor(cx - half_w)));
    const int y0 = std::max(0, static_cast<int>(std::floor(cy - half_h)));
    const int x1 = std::min(img_w, static_cast<int>(std::ceil(cx + half_w)));
    const int y1 = std::min(img_h, static_cast<int>(std::ceil(cy + half_h)));

    const int w = std::max(0, x1 - x0);
    const int h = std::max(0, y1 - y0);
    return cv::Rect(x0, y0, w, h);
} 

struct ProcessedCrop {
    cv::Mat img;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float scale_size = 0.0f;
};

ProcessedCrop MakeProcessedCrop(const cv::Mat& frame, const cv::Rect2f* bbox, float pad_scale) {
    ProcessedCrop result;
    const int img_w = frame.cols;
    const int img_h = frame.rows;

    float cx = img_w * 0.5f;
    float cy = img_h * 0.5f;
    float size = static_cast<float>(std::max(img_w, img_h));

    if (bbox) {
        cx = bbox->x + bbox->width * 0.5f;
        cy = bbox->y + bbox->height * 0.5f;
        size = std::max(bbox->width, bbox->height) * pad_scale;
    }

    const float half = size * 0.5f;
    const int x0 = static_cast<int>(std::floor(cx - half));
    const int y0 = static_cast<int>(std::floor(cy - half));
    const int x1 = static_cast<int>(std::ceil(cx + half));
    const int y1 = static_cast<int>(std::ceil(cy + half));

    const int crop_w = x1 - x0;
    const int crop_h = y1 - y0;
    cv::Mat cropped(crop_h, crop_w, frame.type(), cv::Scalar(0, 0, 0));

    const int src_x0 = std::max(0, x0);
    const int src_y0 = std::max(0, y0);
    const int src_x1 = std::min(img_w, x1);
    const int src_y1 = std::min(img_h, y1);
    const int dst_x0 = src_x0 - x0;
    const int dst_y0 = src_y0 - y0;
    const int copy_w = std::max(0, src_x1 - src_x0);
    const int copy_h = std::max(0, src_y1 - src_y0);

    if (copy_w > 0 && copy_h > 0) {
        frame(cv::Rect(src_x0, src_y0, copy_w, copy_h))
            .copyTo(cropped(cv::Rect(dst_x0, dst_y0, copy_w, copy_h)));
    }

    cv::resize(cropped, result.img, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_AREA);
    result.center_x = cx;
    result.center_y = cy;
    result.scale_size = size;
    return result;
}

ProcessedCrop MakeProcessedCropFromCenter(const cv::Mat& frame, float cx, float cy, float size) {
    ProcessedCrop result;
    const int img_w = frame.cols;
    const int img_h = frame.rows;

    const float half = size * 0.5f;
    const int x0 = static_cast<int>(std::floor(cx - half));
    const int y0 = static_cast<int>(std::floor(cy - half));
    const int x1 = static_cast<int>(std::ceil(cx + half));
    const int y1 = static_cast<int>(std::ceil(cy + half));

    const int crop_w = x1 - x0;
    const int crop_h = y1 - y0;
    cv::Mat cropped(crop_h, crop_w, frame.type(), cv::Scalar(0, 0, 0));

    const int src_x0 = std::max(0, x0);
    const int src_y0 = std::max(0, y0);
    const int src_x1 = std::min(img_w, x1);
    const int src_y1 = std::min(img_h, y1);
    const int dst_x0 = src_x0 - x0;
    const int dst_y0 = src_y0 - y0;
    const int copy_w = std::max(0, src_x1 - src_x0);
    const int copy_h = std::max(0, src_y1 - src_y0);

    if (copy_w > 0 && copy_h > 0) {
        frame(cv::Rect(src_x0, src_y0, copy_w, copy_h))
            .copyTo(cropped(cv::Rect(dst_x0, dst_y0, copy_w, copy_h)));
    }

    cv::resize(cropped, result.img, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_AREA);
    result.center_x = cx;
    result.center_y = cy;
    result.scale_size = size;
    return result;
}

std::vector<float> PreprocessImage(const cv::Mat& img) {
    cv::Mat padded;
    if (img.cols == kInputW && img.rows == kInputH) {
        padded = img.clone();
    } else {
        const int w = img.cols;
        const int h = img.rows;
        const float scale = std::min(static_cast<float>(kInputW) / w,
                                     static_cast<float>(kInputH) / h);
        const int new_w = static_cast<int>(std::round(w * scale));
        const int new_h = static_cast<int>(std::round(h * scale));

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

        const int pad_w = kInputW - new_w;
        const int pad_h = kInputH - new_h;
        const int left = pad_w / 2;
        const int top = pad_h / 2;

        cv::copyMakeBorder(resized, padded, top, pad_h - top, left, pad_w - left,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    // 2. Convert BGR -> RGB
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    // 3. Convert to Float [0.0, 1.0]
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);

    // 4. Split channels for CHW layout
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);

    std::vector<float> input_data;
    input_data.reserve(1 * 3 * kInputH * kInputW);

    // 5. Normalize (ImageNet stats) and Flatten to 1D vector
    for (int c = 0; c < 3; ++c) {
        // Iterate over pixels: H then W (Row-Major)
        for (int h = 0; h < kInputH; ++h) {
            const float* row_ptr = channels[c].ptr<float>(h); // Optimization: Get row pointer
            for (int w = 0; w < kInputW; ++w) {
                float val = row_ptr[w];
                input_data.push_back((val - kMean[c]) / kStd[c]);
            }
        }
    }
    return input_data;
}


bool EnsureOutputDir(const std::string& dir) {
    if (dir.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return !ec;
}

std::string MakeFrameName(int frame_idx) {
    std::ostringstream oss;
    oss << "overlays/overlay_" << std::setw(6) << std::setfill('0') << frame_idx << ".png";
    return oss.str();
}

std::string MakeInputName(int frame_idx) {
    std::ostringstream oss;
    oss << "inputs/input_" << std::setw(6) << std::setfill('0') << frame_idx << ".png";
    return oss.str();
}

std::string MakeObjName(int frame_idx) {
    std::ostringstream oss;
    oss << "smpl_" << std::setw(6) << std::setfill('0') << frame_idx << ".obj";
    return oss.str();
}

void WriteResult(std::ofstream& out, int frame_idx, const SmplResult& res) {
    out << "Frame: " << frame_idx << "\n";
    out << "Pose: ";
    for (float v : res.pose) out << v << " ";
    out << "\n";
    out << "Shape: ";
    for (float v : res.shape) out << v << " ";
    out << "\n";
    out << "Camera: ";
    for (float v : res.camera) out << v << " ";
    out << "\n";
    out << "----------------------------------------\n";
}


} // namespace

bool RunHmrInferenceOnVideo(const std::string& model_path,
                            const std::string& video_path,
                            const HmrOutputOptions& options,
                            ResultsDict* out_results) {
    if (out_results) out_results->clear();

    TrtLogger logger;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    try {
        engine = BuildEngineFromOnnx(model_path, logger);
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Failed to build TensorRT engine: " << e.what() << std::endl;
        std::cerr << "Ensure .onnx and .onnx.data are in the same folder." << std::endl;
        return false;
    }

    TrtUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "\n[ERROR] Failed to create TensorRT execution context." << std::endl;
        return false;
    }

    const int idx_image = engine->getBindingIndex("image");
    const int idx_bbox = engine->getBindingIndex("bbox");
    const int idx_pose = engine->getBindingIndex("pose");
    const int idx_betas = engine->getBindingIndex("betas");
    const int idx_cam = engine->getBindingIndex("cam");
    if (idx_image < 0 || idx_bbox < 0 || idx_pose < 0 || idx_betas < 0 || idx_cam < 0) {
        std::cerr << "\n[ERROR] Failed to find required TensorRT bindings (image/bbox/pose/betas/cam)." << std::endl;
        return false;
    }

    cv::VideoCapture cap(video_path);
    cv::Mat single_frame;
    bool use_single_frame = false;
    if (!cap.isOpened()) {
        single_frame = cv::imread(video_path);
        if (single_frame.empty()) {
            std::cerr << "Failed to open video or image." << std::endl;
            return false;
        }
        use_single_frame = true;
    }
 

    const bool save_outputs = options.save_outputs && !options.output_dir.empty();
    std::ofstream outfile;
    if (save_outputs) {
        if (!EnsureOutputDir(options.output_dir)) {
            std::cerr << "Failed to create output dir: " << options.output_dir << std::endl;
            return false;
        }
        EnsureOutputDir((std::filesystem::path(options.output_dir) / "overlays").string());
        EnsureOutputDir((std::filesystem::path(options.output_dir) / "inputs").string());
        outfile.open(std::filesystem::path(options.output_dir) / "output.txt");
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output.txt in " << options.output_dir << std::endl;
            return false;
        }
    }

    bool use_yolo = options.use_yolo && !options.yolo_model_path.empty();
    std::unique_ptr<SMPLLayer> smpl_layer;
    if (save_outputs || use_yolo) {
        smpl_layer = std::make_unique<SMPLLayer>(options.smpl_model_path);
    }

    YoloPersonDetector yolo_detector;
    if (use_yolo) {
        yolo_detector = YoloPersonDetector();
        if (!yolo_detector.Load(options.yolo_model_path)) {
            std::cerr << "Failed to load YOLO model: " << options.yolo_model_path << std::endl;
            use_yolo = false;
        }
    }

    const nvinfer1::Dims4 img_dims(1, 3, kInputH, kInputW);
    const nvinfer1::Dims2 bbox_dims(1, 3);

    if (!context->setBindingDimensions(idx_image, img_dims) ||
        !context->setBindingDimensions(idx_bbox, bbox_dims)) {
        std::cerr << "\n[ERROR] Failed to set TensorRT input dimensions." << std::endl;
        return false;
    }
    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "\n[ERROR] TensorRT input dimensions not fully specified." << std::endl;
        return false;
    }

    const auto img_out_dims = context->getBindingDimensions(idx_image);
    const auto bbox_out_dims = context->getBindingDimensions(idx_bbox);
    const auto pose_dims = context->getBindingDimensions(idx_pose);
    const auto betas_dims = context->getBindingDimensions(idx_betas);
    const auto cam_dims = context->getBindingDimensions(idx_cam);

    const size_t img_bytes = static_cast<size_t>(Volume(img_out_dims)) * sizeof(float);
    const size_t bbox_bytes = static_cast<size_t>(Volume(bbox_out_dims)) * sizeof(float);
    const size_t pose_bytes = static_cast<size_t>(Volume(pose_dims)) * sizeof(float);
    const size_t betas_bytes = static_cast<size_t>(Volume(betas_dims)) * sizeof(float);
    const size_t cam_bytes = static_cast<size_t>(Volume(cam_dims)) * sizeof(float);

    void* d_image = nullptr;
    void* d_bbox = nullptr;
    void* d_pose = nullptr;
    void* d_betas = nullptr;
    void* d_cam = nullptr;
    cudaStream_t stream = nullptr;

    try {
        CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");
        CheckCuda(cudaMalloc(&d_image, img_bytes), "cudaMalloc image failed");
        CheckCuda(cudaMalloc(&d_bbox, bbox_bytes), "cudaMalloc bbox failed");
        CheckCuda(cudaMalloc(&d_pose, pose_bytes), "cudaMalloc pose failed");
        CheckCuda(cudaMalloc(&d_betas, betas_bytes), "cudaMalloc betas failed");
        CheckCuda(cudaMalloc(&d_cam, cam_bytes), "cudaMalloc cam failed");
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] CUDA setup failed: " << e.what() << std::endl;
        if (d_image) cudaFree(d_image);
        if (d_bbox) cudaFree(d_bbox);
        if (d_pose) cudaFree(d_pose);
        if (d_betas) cudaFree(d_betas);
        if (d_cam) cudaFree(d_cam);
        if (stream) cudaStreamDestroy(stream);
        return false;
    }

    std::vector<void*> bindings(engine->getNbBindings(), nullptr);
    bindings[idx_image] = d_image;
    bindings[idx_bbox] = d_bbox;
    bindings[idx_pose] = d_pose;
    bindings[idx_betas] = d_betas;
    bindings[idx_cam] = d_cam;

    std::vector<float> pose_out(static_cast<size_t>(Volume(pose_dims)));
    std::vector<float> betas_out(static_cast<size_t>(Volume(betas_dims)));
    std::vector<float> cam_out(static_cast<size_t>(Volume(cam_dims)));

    std::cout << "Processing video..." << std::endl;

    auto process_frame = [&](const cv::Mat& frame, int frame_idx) {
        cv::Mat base_frame;
        if (frame.channels() == 4) {
            cv::cvtColor(frame, base_frame, cv::COLOR_BGRA2BGR);
        } else {
            base_frame = frame;
        }
        cv::Mat model_frame = base_frame;
        std::vector<float> bbox_data;
        float crop_cx = base_frame.cols * 0.5f;
        float crop_cy = base_frame.rows * 0.5f;
        float crop_size = static_cast<float>(std::max(base_frame.cols, base_frame.rows));
        std::optional<cv::Rect2f> yolo_bbox;
        std::optional<cv::Rect> yolo_crop;
        std::vector<cv::Point2f> yolo_keypoints;
        std::vector<float> yolo_keypoint_scores;
        std::optional<ProcessedCrop> crop_res;
        if (use_yolo) {
            cv::Rect2f person_bbox;
            std::vector<cv::Point2f> keypoints;
            std::vector<float> keypoint_scores;
            if (yolo_detector.DetectPerson(base_frame, &person_bbox, nullptr, &keypoints, &keypoint_scores)) {
                yolo_bbox = person_bbox;
                yolo_keypoints = keypoints;
                yolo_keypoint_scores = keypoint_scores;
                float center_x = person_bbox.x + person_bbox.width * 0.5f;
                float center_y = person_bbox.y + person_bbox.height * 0.5f;
                const float size = std::max(person_bbox.width, person_bbox.height) * 1.1f;

                const int left_hip_idx = 11;
                const int right_hip_idx = 12;
                const float kpt_threshold = 0.2f;
                if (static_cast<int>(keypoints.size()) > right_hip_idx &&
                    static_cast<int>(keypoint_scores.size()) > right_hip_idx) {
                    const float left_score = keypoint_scores[left_hip_idx];
                    const float right_score = keypoint_scores[right_hip_idx];
                    if (left_score >= kpt_threshold && right_score >= kpt_threshold) {
                        const cv::Point2f left_hip = keypoints[left_hip_idx];
                        const cv::Point2f right_hip = keypoints[right_hip_idx];
                        center_x = 0.5f * (left_hip.x + right_hip.x);
                        center_y = 0.5f * (left_hip.y + right_hip.y);
                    }
                }
 
                crop_res = MakeProcessedCropFromCenter(base_frame, center_x, center_y, size);
                model_frame = crop_res->img;
                crop_cx = crop_res->center_x;
                crop_cy = crop_res->center_y;
                crop_size = crop_res->scale_size;
                const float half = crop_res->scale_size * 0.5f;
                const int x0 = static_cast<int>(std::floor(crop_res->center_x - half));
                const int y0 = static_cast<int>(std::floor(crop_res->center_y - half));
                const int size_i = static_cast<int>(std::ceil(crop_res->scale_size));
                yolo_crop = cv::Rect(x0, y0, size_i, size_i) &
                            cv::Rect(0, 0, base_frame.cols, base_frame.rows);
            }
        }
        const float f_geo = std::max(base_frame.cols, base_frame.rows) * options.focal_length_scale;
        const float f_render = f_geo;
        if (crop_res) {
            bbox_data = GetCliffBBox(crop_res->center_x, crop_res->center_y,
                                     crop_res->scale_size, f_geo,
                                     base_frame.cols, base_frame.rows);
        } else {
            const float cx = base_frame.cols * 0.5f;
            const float cy = base_frame.rows * 0.5f;
            const float box_size = static_cast<float>(std::max(base_frame.cols, base_frame.rows));
            bbox_data = GetCliffBBox(cx, cy, box_size, f_geo, base_frame.cols, base_frame.rows);
        }
        std::vector<float> img_data = PreprocessImage(model_frame);

        try {
            CheckCuda(cudaMemcpyAsync(d_image, img_data.data(), img_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync image failed");
            CheckCuda(cudaMemcpyAsync(d_bbox, bbox_data.data(), bbox_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync bbox failed");

            if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
                throw std::runtime_error("TensorRT enqueue failed");
            }

            CheckCuda(cudaMemcpyAsync(pose_out.data(), d_pose, pose_bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync pose failed");
            CheckCuda(cudaMemcpyAsync(betas_out.data(), d_betas, betas_bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync betas failed");
            CheckCuda(cudaMemcpyAsync(cam_out.data(), d_cam, cam_bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync cam failed");

            CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

            SmplResult res;
            int pose_size = static_cast<int>(pose_out.size());

            res.pose.assign(pose_out.begin(), pose_out.end());
            res.shape.assign(betas_out.begin(), betas_out.end());
            res.camera.assign(cam_out.begin(), cam_out.end());

            float smplify_y_sign = 1.0f;
            if (use_yolo && smpl_layer && !yolo_keypoints.empty() &&
                yolo_keypoints.size() == yolo_keypoint_scores.size()) {
                SmplifyLiteOptions smplify_opts;
                smplify_opts.num_iters = 150;
                smplify_opts.keypoint_threshold = 0.2f;
                SmplifyLite(*smpl_layer, yolo_keypoints, yolo_keypoint_scores,
                            crop_cx, crop_cy, crop_size,
                            f_geo, f_render,
                            static_cast<float>(base_frame.cols), static_cast<float>(base_frame.rows),
                            &res, smplify_opts, &smplify_y_sign,
                            save_outputs ? &base_frame : nullptr,
                            save_outputs ? &options.output_dir : nullptr,
                            frame_idx);
            }

            if (out_results) {
                (*out_results)[frame_idx] = res;
            }

            if (save_outputs) {
                WriteResult(outfile, frame_idx, res);

                torch::NoGradGuard no_grad;
                auto betas = torch::from_blob(res.shape.data(), {1, 10}, torch::kFloat).clone();
                auto pose_axis = PoseToAxisAngle(res);
                auto trans = torch::zeros({1, 3}, torch::kFloat);

                auto smpl_out = smpl_layer->forward(betas, pose_axis, trans);
                cv::Mat overlay = base_frame.clone();
                float render_y_sign = smplify_y_sign;
                DrawVerticesOverlayPinhole(overlay, smpl_out.vertices, res.camera,
                                           crop_cx, crop_cy, crop_size, f_geo, f_render, render_y_sign);
                if (yolo_bbox) {
                    const cv::Scalar color_bbox(255, 0, 0);
                    cv::rectangle(overlay, yolo_bbox.value(), color_bbox, 2, cv::LINE_AA);
                }
                if (yolo_crop) {
                    const cv::Scalar color_crop(0, 165, 255);
                    cv::rectangle(overlay, yolo_crop.value(), color_crop, 2, cv::LINE_AA);
                }
                DrawOverlayDebug(overlay, res.camera, bbox_data);
                if (!yolo_keypoints.empty()) {
                    DrawPoseKeypoints(overlay, yolo_keypoints, yolo_keypoint_scores, 0.2f);
                }

                const auto out_path = std::filesystem::path(options.output_dir) / MakeFrameName(frame_idx);
                cv::imwrite(out_path.string(), overlay);
 
                const auto input_path = std::filesystem::path(options.output_dir) / MakeInputName(frame_idx);
                cv::imwrite(input_path.string(), model_frame);

                if (frame_idx == 150) {
                    const auto obj_path = std::filesystem::path(options.output_dir) / MakeObjName(frame_idx);
                    WriteObjVertices(smpl_out.vertices, obj_path.string());
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "Inference/postprocess error frame " << frame_idx << ": " << e.what() << std::endl;
        }
    };

    cv::Mat frame;
    int frame_idx = 0;
    if (use_single_frame) { 
        process_frame(single_frame, 0);
    } else {
        while (cap.read(frame)) {
            process_frame(frame, frame_idx);
            if (frame_idx % 30 == 0) std::cout << "Frame: " << frame_idx << std::endl;
            frame_idx++;
        }
    }

    if (outfile.is_open()) outfile.close();
    if (d_image) cudaFree(d_image);
    if (d_bbox) cudaFree(d_bbox);
    if (d_pose) cudaFree(d_pose);
    if (d_betas) cudaFree(d_betas);
    if (d_cam) cudaFree(d_cam);
    if (stream) cudaStreamDestroy(stream);
    std::cout << "Finished." << std::endl;
    return true;
}
