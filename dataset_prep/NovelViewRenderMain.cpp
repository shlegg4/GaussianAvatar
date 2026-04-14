#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <torch/torch.h>

#include "utils/SmplxLBS.h"
#include "utils/train/TrainJsonl.h"

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path jsonl_path;
    fs::path smplx_model_path = "smplx_data.pt";
    fs::path output_video_path = "novel_side_view.mp4";
    fs::path keypoints_json_path;
    fs::path rtmpose_keypoints_json_path;
    int width = 1280;
    int height = 720;
    double fps = 24.0;
    float yaw_deg = 90.0f;
    bool stabilize_root_yaw = false;
};

struct Triangle2D {
    std::array<cv::Point, 3> pts;
    float depth = 0.0f;
    float diffuse = 1.0f;
    float specular = 0.0f;
};

struct RenderCameraParams {
    cv::Vec3f center{0.0f, 0.0f, 0.0f};
    cv::Matx33f R = cv::Matx33f::eye();
    float focal = 1.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float z_offset = 1.0f;
};

struct RtmPoseFrameKeypoints {
    std::vector<cv::Point3f> joints_world;
    std::vector<float> scores;
};

cv::Vec3f MeanPoint(const torch::Tensor& points_cpu);
float MaxRadius(const torch::Tensor& points_cpu, const cv::Vec3f& center);

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  dataset_novel_view_render --jsonl <frames.jsonl> [options]\n"
        << "Options:\n"
        << "  --smplx-model <path>     Path to smplx_data.pt (default: smplx_data.pt)\n"
        << "  --output <path>          Output mp4 path (default: novel_side_view.mp4)\n"
        << "  --keypoints-json <path>  Output raw 3D keypoints JSON path\n"
        << "  --rtmpose-keypoints-json <path>  Input RTMPose 3D keypoints JSONL from dataset_prep\n"
        << "  --width <int>            Output width (default: 1280)\n"
        << "  --height <int>           Output height (default: 720)\n"
        << "  --fps <float>            Output fps (default: 24)\n"
        << "  --yaw-deg <float>        Novel view yaw in degrees (default: 90)\n"
        << "  --stabilize-root-yaw     Enable root yaw stabilization (default: off)\n"
        << "  --no-stabilize-root-yaw  Disable root yaw stabilization\n";
}

bool ParseArgs(int argc, char* argv[], Options* out_options) {
    if (out_options == nullptr) {
        return false;
    }

    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage();
            return false;
        }
        if (arg == "--jsonl") {
            const char* value = require_value("--jsonl");
            if (!value) return false;
            opts.jsonl_path = value;
        } else if (arg == "--smplx-model") {
            const char* value = require_value("--smplx-model");
            if (!value) return false;
            opts.smplx_model_path = value;
        } else if (arg == "--output") {
            const char* value = require_value("--output");
            if (!value) return false;
            opts.output_video_path = value;
        } else if (arg == "--keypoints-json") {
            const char* value = require_value("--keypoints-json");
            if (!value) return false;
            opts.keypoints_json_path = value;
        } else if (arg == "--rtmpose-keypoints-json") {
            const char* value = require_value("--rtmpose-keypoints-json");
            if (!value) return false;
            opts.rtmpose_keypoints_json_path = value;
        } else if (arg == "--width") {
            const char* value = require_value("--width");
            if (!value) return false;
            opts.width = std::max(1, std::stoi(value));
        } else if (arg == "--height") {
            const char* value = require_value("--height");
            if (!value) return false;
            opts.height = std::max(1, std::stoi(value));
        } else if (arg == "--fps") {
            const char* value = require_value("--fps");
            if (!value) return false;
            opts.fps = std::max(1.0, std::stod(value));
        } else if (arg == "--yaw-deg") {
            const char* value = require_value("--yaw-deg");
            if (!value) return false;
            opts.yaw_deg = std::stof(value);
        } else if (arg == "--stabilize-root-yaw") {
            opts.stabilize_root_yaw = true;
        } else if (arg == "--no-stabilize-root-yaw") {
            opts.stabilize_root_yaw = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (opts.jsonl_path.empty()) {
        std::cerr << "--jsonl is required\n";
        return false;
    }

    *out_options = std::move(opts);
    return true;
}

cv::Matx33f MakeYawRotation(float yaw_deg) {
    const float rad = yaw_deg * static_cast<float>(CV_PI) / 180.0f;
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    return cv::Matx33f(c, 0.0f, s,
                       0.0f, 1.0f, 0.0f,
                       -s, 0.0f, c);
}

RenderCameraParams BuildRenderCameraParams(const torch::Tensor& verts_cpu,
                                           int width,
                                           int height,
                                           float yaw_deg) {
    RenderCameraParams params;
    params.center = MeanPoint(verts_cpu);
    const float radius = MaxRadius(verts_cpu, params.center);
    params.R = MakeYawRotation(yaw_deg);
    params.focal = static_cast<float>(width) * 1.1f;
    params.cx = static_cast<float>(width) * 0.5f;
    params.cy = static_cast<float>(height) * 0.5f;
    params.z_offset = std::max(1.0f, radius * 3.0f);
    return params;
}

cv::Vec3f TransformWorldToNovelCamera(const cv::Vec3f& world_point,
                                      const RenderCameraParams& params) {
    cv::Vec3f vc = params.R * (world_point - params.center);
    vc[2] += params.z_offset;
    return vc;
}

cv::Vec3f MeanPoint(const torch::Tensor& points_cpu) {
    auto mean_t = points_cpu.mean(0);
    return cv::Vec3f(mean_t[0].item<float>(), mean_t[1].item<float>(), mean_t[2].item<float>());
}

float MaxRadius(const torch::Tensor& points_cpu, const cv::Vec3f& center) {
    auto acc = points_cpu.accessor<float, 2>();
    float max_r = 1e-3f;
    for (int i = 0; i < acc.size(0); ++i) {
        const float dx = acc[i][0] - center[0];
        const float dy = acc[i][1] - center[1];
        const float dz = acc[i][2] - center[2];
        const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
        max_r = std::max(max_r, r);
    }
    return max_r;
}

std::vector<Triangle2D> BuildTriangles(const torch::Tensor& verts_cpu,
                                       const torch::Tensor& faces_cpu,
                                       const RenderCameraParams& camera) {

    auto v_acc = verts_cpu.accessor<float, 2>();
    auto f_acc = faces_cpu.accessor<int64_t, 2>();

    std::vector<cv::Vec3f> v_cam(static_cast<size_t>(v_acc.size(0)), cv::Vec3f(0, 0, 0));
    for (int i = 0; i < v_acc.size(0); ++i) {
        const cv::Vec3f v_world(v_acc[i][0], v_acc[i][1], v_acc[i][2]);
        v_cam[static_cast<size_t>(i)] = TransformWorldToNovelCamera(v_world, camera);
    }

    std::vector<Triangle2D> triangles;
    triangles.reserve(static_cast<size_t>(f_acc.size(0)));

    for (int f = 0; f < f_acc.size(0); ++f) {
        const int i0 = static_cast<int>(f_acc[f][0]);
        const int i1 = static_cast<int>(f_acc[f][1]);
        const int i2 = static_cast<int>(f_acc[f][2]);
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            i0 >= static_cast<int>(v_cam.size()) ||
            i1 >= static_cast<int>(v_cam.size()) ||
            i2 >= static_cast<int>(v_cam.size())) {
            continue;
        }

        const cv::Vec3f a = v_cam[static_cast<size_t>(i0)];
        const cv::Vec3f b = v_cam[static_cast<size_t>(i1)];
        const cv::Vec3f c = v_cam[static_cast<size_t>(i2)];
        if (a[2] <= 1e-4f || b[2] <= 1e-4f || c[2] <= 1e-4f) {
            continue;
        }

        const cv::Vec3f n = (b - a).cross(c - a);
        const float n_norm = static_cast<float>(cv::norm(n));
        if (!(n_norm > 1e-8f)) {
            continue;
        }
        const cv::Vec3f nn = n * (1.0f / n_norm);
        const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(0.35f, -0.25f, 0.90f));
        const float diffuse = std::max(0.0f, nn.dot(light_dir));
        const cv::Vec3f view_dir = cv::normalize(cv::Vec3f(0.0f, 0.0f, 1.0f));
        const cv::Vec3f reflect_dir = cv::normalize(2.0f * nn * nn.dot(light_dir) - light_dir);
        const float spec = std::pow(std::max(0.0f, reflect_dir.dot(view_dir)), 24.0f);

        Triangle2D tri;
        tri.depth = (a[2] + b[2] + c[2]) / 3.0f;
        tri.diffuse = diffuse;
        tri.specular = spec;

        const auto project = [&](const cv::Vec3f& p) -> cv::Point {
            const float u = camera.focal * (p[0] / p[2]) + camera.cx;
            const float v = camera.focal * (p[1] / p[2]) + camera.cy;
            return cv::Point(static_cast<int>(std::lround(u)),
                             static_cast<int>(std::lround(v)));
        };

        tri.pts[0] = project(a);
        tri.pts[1] = project(b);
        tri.pts[2] = project(c);
        triangles.push_back(tri);
    }

    std::sort(triangles.begin(), triangles.end(), [](const Triangle2D& lhs, const Triangle2D& rhs) {
        return lhs.depth > rhs.depth;
    });

    return triangles;
}

void DrawTriangles(const std::vector<Triangle2D>& triangles, cv::Mat* image) {
    if (image == nullptr || image->empty()) {
        return;
    }

    float min_depth = std::numeric_limits<float>::infinity();
    float max_depth = -std::numeric_limits<float>::infinity();
    for (const auto& tri : triangles) {
        min_depth = std::min(min_depth, tri.depth);
        max_depth = std::max(max_depth, tri.depth);
    }
    const float depth_range = std::max(1e-6f, max_depth - min_depth);

    for (const auto& tri : triangles) {
        const cv::Point pts[3] = {tri.pts[0], tri.pts[1], tri.pts[2]};
        const float depth_t = (tri.depth - min_depth) / depth_range;
        const float ambient = 0.28f;
        const float diffuse_term = 0.62f * tri.diffuse;
        const float specular_term = 0.40f * tri.specular;
        const float depth_term = 0.85f + 0.15f * (1.0f - depth_t);
        const float shade = std::clamp((ambient + diffuse_term + specular_term) * depth_term, 0.0f, 1.25f);

        const cv::Scalar base(95.0, 178.0, 250.0);
        const cv::Scalar color(base[0] * shade, base[1] * shade, base[2] * shade);
        cv::fillConvexPoly(*image, pts, 3, color, cv::LINE_AA);
    }
}

std::vector<cv::Point> ProjectKeypoints(const torch::Tensor& joints_cpu,
                                        const RenderCameraParams& camera,
                                        int width,
                                        int height) {
    auto j_acc = joints_cpu.accessor<float, 2>();
    std::vector<cv::Point> projected(static_cast<size_t>(j_acc.size(0)), cv::Point(-1, -1));
    for (int i = 0; i < j_acc.size(0); ++i) {
        const cv::Vec3f jw(j_acc[i][0], j_acc[i][1], j_acc[i][2]);
        const cv::Vec3f jc = TransformWorldToNovelCamera(jw, camera);
        if (jc[2] <= 1e-4f) {
            continue;
        }
        const float u = camera.focal * (jc[0] / jc[2]) + camera.cx;
        const float v = camera.focal * (jc[1] / jc[2]) + camera.cy;
        if (u >= 0.0f && v >= 0.0f &&
            u < static_cast<float>(width) && v < static_cast<float>(height)) {
            projected[static_cast<size_t>(i)] =
                cv::Point(static_cast<int>(std::lround(u)), static_cast<int>(std::lround(v)));
        }
    }
    return projected;
}

void DrawKeypoints(const std::vector<cv::Point>& points, cv::Mat* image) {
    if (image == nullptr || image->empty()) {
        return;
    }

    for (const auto& p : points) {
        if (p.x < 0 || p.y < 0) {
            continue;
        }
        cv::circle(*image, p, 2, cv::Scalar(60, 230, 120), -1, cv::LINE_AA);
    }
}

void DrawKeypointsColored(const std::vector<cv::Point>& points,
                         cv::Mat* image,
                         const cv::Scalar& color,
                         int radius) {
    if (image == nullptr || image->empty()) {
        return;
    }

    for (const auto& p : points) {
        if (p.x < 0 || p.y < 0) {
            continue;
        }
        cv::circle(*image, p, radius, color, -1, cv::LINE_AA);
    }
}

bool ParseRtmPoseKeypointsLine(const std::string& line,
                               int* out_frame,
                               RtmPoseFrameKeypoints* out_keypoints) {
    if (out_frame == nullptr || out_keypoints == nullptr) {
        return false;
    }

    double frame_value = -1.0;
    if (!ExtractNumberField(line, "frame", &frame_value)) {
        return false;
    }

    std::vector<float> flat_points;
    std::vector<float> scores;
    if (!ExtractArrayField(line, "rtmpose_keypoints_3d", &flat_points) ||
        !ExtractArrayField(line, "rtmpose_scores", &scores)) {
        return false;
    }
    if (flat_points.size() % 3u != 0u) {
        return false;
    }

    RtmPoseFrameKeypoints parsed;
    const size_t joint_count = flat_points.size() / 3u;
    parsed.joints_world.reserve(joint_count);
    for (size_t i = 0; i < joint_count; ++i) {
        parsed.joints_world.emplace_back(flat_points[3u * i + 0u],
                                         flat_points[3u * i + 1u],
                                         flat_points[3u * i + 2u]);
    }
    parsed.scores = std::move(scores);

    *out_frame = static_cast<int>(frame_value);
    *out_keypoints = std::move(parsed);
    return true;
}

std::map<int, RtmPoseFrameKeypoints> LoadRtmPoseKeypoints(const fs::path& path) {
    std::map<int, RtmPoseFrameKeypoints> by_frame;
    std::ifstream input(path);
    if (!input.is_open()) {
        return by_frame;
    }

    std::string line;
    while (std::getline(input, line)) {
        int frame = -1;
        RtmPoseFrameKeypoints keypoints;
        if (ParseRtmPoseKeypointsLine(line, &frame, &keypoints)) {
            by_frame[frame] = std::move(keypoints);
        }
    }
    return by_frame;
}

std::vector<cv::Point> ProjectWorldKeypoints(const std::vector<cv::Point3f>& points_world,
                                             const std::vector<float>& scores,
                                             const RenderCameraParams& camera,
                                             int width,
                                             int height,
                                             float min_score) {
    std::vector<cv::Point> projected(points_world.size(), cv::Point(-1, -1));
    for (size_t i = 0; i < points_world.size(); ++i) {
        if (i < scores.size() && scores[i] < min_score) {
            continue;
        }

        const cv::Vec3f jw(points_world[i].x, points_world[i].y, points_world[i].z);
        const cv::Vec3f jc = TransformWorldToNovelCamera(jw, camera);
        if (jc[2] <= 1e-4f) {
            continue;
        }

        const float u = camera.focal * (jc[0] / jc[2]) + camera.cx;
        const float v = camera.focal * (jc[1] / jc[2]) + camera.cy;
        if (u >= 0.0f && v >= 0.0f &&
            u < static_cast<float>(width) && v < static_cast<float>(height)) {
            projected[i] = cv::Point(static_cast<int>(std::lround(u)),
                                     static_cast<int>(std::lround(v)));
        }
    }
    return projected;
}

void WriteFrameKeypointsJson(std::ostream& out,
                             const TrainSample& sample,
                             size_t sequence_index,
                             const torch::Tensor& joints_cpu,
                             bool first_entry) {
    auto j_acc = joints_cpu.accessor<float, 2>();
    if (!first_entry) {
        out << ",\n";
    }
    out << "  {\"sequence_index\":" << sequence_index
        << ",\"frame\":" << sample.frame
        << ",\"joint_count\":" << j_acc.size(0)
        << ",\"joints\":[";
    for (int i = 0; i < j_acc.size(0); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "["
            << std::fixed << std::setprecision(6)
            << j_acc[i][0] << ","
            << j_acc[i][1] << ","
            << j_acc[i][2] << "]";
    }
    out << "]}";
}

bool BuildPoseTensor(const TrainSample& sample, torch::Tensor* out_pose) {
    if (out_pose == nullptr || sample.pose.size() < 72u) {
        return false;
    }
    *out_pose = torch::from_blob(const_cast<float*>(sample.pose.data()), {1, 24, 3}, torch::kFloat32).clone();
    return true;
}

std::vector<float> CopyOrZeros(const std::vector<float>& src, size_t expected) {
    std::vector<float> out(expected, 0.0f);
    const size_t copy = std::min(src.size(), expected);
    std::copy_n(src.begin(), copy, out.begin());
    return out;
}

float ExtractYawFromRotation(const cv::Matx33f& rotation) {
    return std::atan2(rotation(0, 2), rotation(2, 2));
}

void StabilizeRootYaw(std::vector<TrainSample>* samples) {
    if (samples == nullptr || samples->empty()) {
        return;
    }

    bool initialized = false;
    float reference_yaw = 0.0f;

    for (auto& sample : *samples) {
        if (sample.pose.size() < 3u) {
            continue;
        }

        cv::Vec3f root_aa(sample.pose[0], sample.pose[1], sample.pose[2]);
        cv::Matx33f root_rot;
        cv::Rodrigues(root_aa, root_rot);

        const float current_yaw = ExtractYawFromRotation(root_rot);
        if (!initialized) {
            reference_yaw = current_yaw;
            initialized = true;
        }

        const float yaw_delta = current_yaw - reference_yaw;
        const float c = std::cos(-yaw_delta);
        const float s = std::sin(-yaw_delta);
        const cv::Matx33f yaw_correction(c, 0.0f, s,
                                         0.0f, 1.0f, 0.0f,
                                         -s, 0.0f, c);

        const cv::Matx33f corrected_rot = yaw_correction * root_rot;
        cv::Vec3f corrected_aa;
        cv::Rodrigues(cv::Mat(corrected_rot), corrected_aa);
        sample.pose[0] = corrected_aa[0];
        sample.pose[1] = corrected_aa[1];
        sample.pose[2] = corrected_aa[2];
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    Options options;
    if (!ParseArgs(argc, argv, &options)) {
        PrintUsage();
        return 1;
    }

    std::ifstream input(options.jsonl_path);
    if (!input.is_open()) {
        std::cerr << "Failed to open jsonl: " << options.jsonl_path.string() << "\n";
        return 1;
    }

    std::vector<TrainSample> samples;
    std::string line;
    while (std::getline(input, line)) {
        TrainSample sample;
        if (ParseTrainSample(line, &sample)) {
            if (sample.body_model == "smplx" || !sample.smplx_expression.empty()) {
                samples.push_back(std::move(sample));
            }
        }
    }

    if (samples.empty()) {
        std::cerr << "No SMPL-X samples parsed from " << options.jsonl_path.string() << "\n";
        return 1;
    }

    if (options.keypoints_json_path.empty()) {
        const fs::path stemmed = options.output_video_path.parent_path() /
                                 (options.output_video_path.stem().string() + "_keypoints.json");
        options.keypoints_json_path = stemmed;
    }
    if (options.rtmpose_keypoints_json_path.empty()) {
        options.rtmpose_keypoints_json_path = options.jsonl_path.parent_path() / "rtmpose_keypoints3d.jsonl";
    }

    const std::map<int, RtmPoseFrameKeypoints> rtmpose_by_frame =
        LoadRtmPoseKeypoints(options.rtmpose_keypoints_json_path);
    if (rtmpose_by_frame.empty()) {
        std::cout << "Warning: RTMPose keypoints JSONL not found or empty: "
                  << options.rtmpose_keypoints_json_path.string() << "\n";
    }

    if (options.stabilize_root_yaw) {
        StabilizeRootYaw(&samples);
    }

    std::ofstream keypoints_out(options.keypoints_json_path);
    if (!keypoints_out.is_open()) {
        std::cerr << "Failed to open keypoints json for writing: "
                  << options.keypoints_json_path.string() << "\n";
        return 1;
    }
    keypoints_out << "[\n";
    bool first_keypoint_entry = true;

    cv::VideoWriter writer;
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (!writer.open(options.output_video_path.string(),
                     fourcc,
                     options.fps,
                     cv::Size(options.width, options.height),
                     true)) {
        std::cerr << "Failed to open output video: " << options.output_video_path.string() << "\n";
        return 1;
    }

    torch::NoGradGuard no_grad;
    torch::Device device(torch::kCPU);

    SMPLXLayer smplx(options.smplx_model_path.string());
    smplx.to(device);
    smplx.eval();

    const torch::Tensor faces_cpu = smplx.faces.to(torch::kCPU).to(torch::kLong).contiguous();

    for (size_t frame_idx = 0; frame_idx < samples.size(); ++frame_idx) {
        const TrainSample& sample = samples[frame_idx];

        torch::Tensor pose;
        if (!BuildPoseTensor(sample, &pose)) {
            continue;
        }

        const std::vector<float> betas_v = CopyOrZeros(sample.betas, 10u);
        const std::vector<float> expr_v = CopyOrZeros(sample.smplx_expression, 100u);
        const std::vector<float> jaw_v = CopyOrZeros(sample.smplx_jaw_pose, 3u);
        const std::vector<float> eye_v = CopyOrZeros(sample.smplx_eye_pose, 6u);
        const std::vector<float> lhand_v = CopyOrZeros(sample.smplx_left_hand_pose, 45u);
        const std::vector<float> rhand_v = CopyOrZeros(sample.smplx_right_hand_pose, 45u);

        torch::Tensor betas = torch::from_blob(const_cast<float*>(betas_v.data()), {1, static_cast<int64_t>(betas_v.size())}, torch::kFloat32).clone().to(device);
        torch::Tensor expr = torch::from_blob(const_cast<float*>(expr_v.data()), {1, static_cast<int64_t>(expr_v.size())}, torch::kFloat32).clone().to(device);
        torch::Tensor jaw = torch::from_blob(const_cast<float*>(jaw_v.data()), {1, static_cast<int64_t>(jaw_v.size())}, torch::kFloat32).clone().to(device);
        torch::Tensor eye = torch::from_blob(const_cast<float*>(eye_v.data()), {1, static_cast<int64_t>(eye_v.size())}, torch::kFloat32).clone().to(device);
        torch::Tensor lhand = torch::from_blob(const_cast<float*>(lhand_v.data()), {1, static_cast<int64_t>(lhand_v.size())}, torch::kFloat32).clone().to(device);
        torch::Tensor rhand = torch::from_blob(const_cast<float*>(rhand_v.data()), {1, static_cast<int64_t>(rhand_v.size())}, torch::kFloat32).clone().to(device);

        std::array<float, 3> trans_arr = sample.translation;
        if (!sample.has_translation) {
            trans_arr = {0.0f, 0.0f, 0.0f};
        }
        torch::Tensor trans = torch::tensor({trans_arr[0], trans_arr[1], trans_arr[2]}, torch::kFloat32)
                                  .reshape({1, 3})
                                  .to(device);

        SmplxOutput smplx_out = smplx.forward(
            betas,
            expr,
            pose.to(device),
            jaw,
            eye,
            lhand,
            rhand,
            trans);

        torch::Tensor verts_cpu = smplx_out.vertices.squeeze(0).to(torch::kCPU).contiguous();
        torch::Tensor joints_cpu = smplx_out.joints.squeeze(0).to(torch::kCPU).contiguous();

        WriteFrameKeypointsJson(
            keypoints_out,
            sample,
            frame_idx,
            joints_cpu,
            first_keypoint_entry);
        first_keypoint_entry = false;

        const RenderCameraParams camera = BuildRenderCameraParams(
            verts_cpu,
            options.width,
            options.height,
            options.yaw_deg);

        cv::Mat frame(options.height, options.width, CV_8UC3, cv::Scalar(20, 20, 24));
        std::vector<Triangle2D> triangles = BuildTriangles(
            verts_cpu,
            faces_cpu,
            camera);
        DrawTriangles(triangles, &frame);

        const auto rtmpose_it = rtmpose_by_frame.find(sample.frame);
        if (rtmpose_it != rtmpose_by_frame.end()) {
            const std::vector<cv::Point> rtmpose_points_2d = ProjectWorldKeypoints(
                rtmpose_it->second.joints_world,
                rtmpose_it->second.scores,
                camera,
                options.width,
                options.height,
                0.05f);
            DrawKeypointsColored(rtmpose_points_2d, &frame, cv::Scalar(50, 180, 255), 3);
        }

        cv::putText(frame,
                    "Novel Side View",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(230, 230, 230),
                    2,
                    cv::LINE_AA);
        writer.write(frame);

        if ((frame_idx + 1u) % 25u == 0u) {
            std::cout << "Rendered " << (frame_idx + 1u) << " / " << samples.size() << " frames\n";
        }
    }

    keypoints_out << "\n]\n";
    keypoints_out.close();

    writer.release();
    std::cout << "Wrote novel-view video to " << options.output_video_path.string() << "\n";
    std::cout << "Wrote raw 3D keypoints to " << options.keypoints_json_path.string() << "\n";
    std::cout << "Loaded RTMPose keypoints from " << options.rtmpose_keypoints_json_path.string() << "\n";
    return 0;
}
