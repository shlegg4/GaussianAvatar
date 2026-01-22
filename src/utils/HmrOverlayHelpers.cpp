#include "HmrOverlayHelpers.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "HmrInferenceConstants.h"
#include "HmrMathHelpers.h"

void DrawVerticesOverlayPinhole(cv::Mat& frame, const torch::Tensor& verts, const std::vector<float>& cam,
                                float crop_cx, float crop_cy, float crop_size,
                                float f_geo, float f_render,
                                float y_sign_override) {
    if (cam.size() < 3) return;
    const float img_w = static_cast<float>(frame.cols);
    const float img_h = static_cast<float>(frame.rows);
    const cv::Vec3f t = EstimateTranslation(cam, crop_cx, crop_cy, crop_size, f_geo, img_w, img_h);

    auto verts_cpu = verts.squeeze(0).to(torch::kCPU).contiguous();
    const int width = frame.cols;
    const int height = frame.rows;
    const float cx = img_w * 0.5f;
    const float cy = img_h * 0.5f;

    float y_sign = y_sign_override;
    if (std::abs(y_sign_override) < 0.5f) {
        const int count_pos = CountProjectedInFramePinhole(verts_cpu, t, 1.0f, f_render, cx, cy, width, height);
        const int count_neg = CountProjectedInFramePinhole(verts_cpu, t, -1.0f, f_render, cx, cy, width, height);
        y_sign = (count_neg > count_pos) ? -1.0f : 1.0f;
    }

    auto verts_acc = verts_cpu.accessor<float, 2>();
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0] + t[0];
        const float Y = verts_acc[i][1] * y_sign + t[1];
        const float Z = verts_acc[i][2] + t[2];
        const float u = (f_render * X / (Z + 1e-9f)) + cx;
        const float v = (f_render * Y / (Z + 1e-9f)) + cy;
        const int ui = static_cast<int>(u);
        const int vi = static_cast<int>(v);
        if (ui < 0 || vi < 0 || ui >= width || vi >= height) continue;
        cv::circle(frame, cv::Point(ui, vi), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
}

void DrawVerticesOverlay(cv::Mat& frame, const torch::Tensor& verts, const std::vector<float>& cam) {
    if (cam.size() < 3) return;
    const float s = cam[0] * 20.0f;
    const float tx = cam[1];
    const float ty = cam[2];

    auto verts_cpu = verts.squeeze(0).to(torch::kCPU).contiguous();

    const int width = frame.cols;
    const int height = frame.rows;

    const int count_pos = CountProjectedInFrame(verts_cpu, s, tx, ty, 1.0f, width, height);
    const int count_neg = CountProjectedInFrame(verts_cpu, s, tx, ty, -1.0f, width, height);
    const float y_sign = (count_neg > count_pos) ? -1.0f : 1.0f;

    const float img_size = static_cast<float>(kInputW);
    const float scale_x = static_cast<float>(width) / img_size;
    const float scale_y = static_cast<float>(height) / img_size;

    auto verts_acc = verts_cpu.accessor<float, 2>();
    for (int i = 0; i < verts_acc.size(0); ++i) {
        const float X = verts_acc[i][0];
        const float Y = verts_acc[i][1] * y_sign;
        const float u = (s * (X + tx) + img_size * 0.5f) * scale_x;
        const float v = (s * (Y + ty) + img_size * 0.5f) * scale_y;
        const int ui = static_cast<int>(u);
        const int vi = static_cast<int>(v);
        if (ui < 0 || vi < 0 || ui >= width || vi >= height) continue;
        cv::circle(frame, cv::Point(ui, vi), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
}

void DrawOverlayDebug(cv::Mat& frame, const std::vector<float>& cam, const std::vector<float>& bbox) {
    if (cam.size() < 3 || bbox.size() < 3) return;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "cam[s,tx,ty]=[" << cam[0] << "," << cam[1] << "," << cam[2] << "] "
        << "bbox[cx,cy,s]=[" << bbox[0] << "," << bbox[1] << "," << bbox[2] << "]";
    const std::string text = oss.str();
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.5;
    const int thickness = 1;
    const cv::Scalar bg(0, 0, 0);
    const cv::Scalar fg(0, 255, 255);
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, font, scale, thickness, &baseline);
    const cv::Point org(10, 20);
    cv::rectangle(frame, org + cv::Point(0, -size.height - 4),
                  org + cv::Point(size.width + 4, 4), bg, cv::FILLED);
    cv::putText(frame, text, org, font, scale, fg, thickness, cv::LINE_AA);
}

void DrawPoseKeypoints(cv::Mat& frame, const std::vector<cv::Point2f>& keypoints,
                       const std::vector<float>& keypoint_scores, float min_score) {
    if (keypoints.empty() || keypoints.size() != keypoint_scores.size()) return;
    const cv::Scalar color(0, 255, 255);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (keypoint_scores[i] < min_score) continue;
        const cv::Point2f& p = keypoints[i];
        cv::circle(frame, p, 3, color, -1, cv::LINE_AA);
    }
}

void WriteObjVertices(const torch::Tensor& verts, const std::string& path) {
    auto verts_cpu = verts.squeeze(0).to(torch::kCPU).contiguous();
    auto verts_acc = verts_cpu.accessor<float, 2>();
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write OBJ: " + path);
    }
    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < verts_acc.size(0); ++i) {
        out << "v " << verts_acc[i][0] << " " << verts_acc[i][1] << " " << verts_acc[i][2] << "\n";
    }
}
