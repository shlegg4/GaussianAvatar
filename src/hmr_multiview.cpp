#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <optional>
#include <cmath>
#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "utils/HmrInferenceUtils.h"
#include "utils/HmrInferenceConstants.h"
#include "utils/HmrMathHelpers.h"
#include "utils/SmplifyLite.h"
#include "utils/RtmPoseDetector.h"
#include "utils/YoloPersonDetector.h"
#include "utils/ModNetMatte.h"
#include "utils/TrtBuilder.h"

namespace fs = std::filesystem;

// =====================================================================
// HELPER FUNCTIONS
// =====================================================================
namespace
{
    const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> kStd = {0.229f, 0.224f, 0.225f};

    std::vector<float> GetCliffBBox(float cx, float cy, float box_size, float focal, int img_w, int img_h)
    {
        float x = cx - img_w / 2.0f;
        float y = cy - img_h / 2.0f;
        std::vector<float> bbox(3);
        bbox[0] = (x / focal) * 2.8f;
        bbox[1] = (y / focal) * 2.8f;
        bbox[2] = (box_size - 0.24f * focal) / (0.06f * focal);
        return bbox;
    }

    cv::Rect MakePaddedRect(const cv::Rect2f &bbox, int img_w, int img_h, float scale)
    {
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

    std::optional<cv::Rect2f> PoseKeypointsToBbox(const std::vector<cv::Point2f> &keypoints,
                                                  const std::vector<float> &scores, float min_score, int img_w, int img_h)
    {
        if (keypoints.empty() || keypoints.size() != scores.size())
            return std::nullopt;
        float min_x = static_cast<float>(img_w), min_y = static_cast<float>(img_h);
        float max_x = 0.0f, max_y = 0.0f;
        int valid = 0;
        for (size_t i = 0; i < keypoints.size(); ++i)
        {
            if (scores[i] < min_score)
                continue;
            min_x = std::min(min_x, keypoints[i].x);
            min_y = std::min(min_y, keypoints[i].y);
            max_x = std::max(max_x, keypoints[i].x);
            max_y = std::max(max_y, keypoints[i].y);
            valid++;
        }
        if (valid < 2)
            return std::nullopt;
        min_x = std::max(0.0f, std::min(min_x, static_cast<float>(img_w - 1)));
        min_y = std::max(0.0f, std::min(min_y, static_cast<float>(img_h - 1)));
        max_x = std::max(0.0f, std::min(max_x, static_cast<float>(img_w - 1)));
        max_y = std::max(0.0f, std::min(max_y, static_cast<float>(img_h - 1)));
        return cv::Rect2f(min_x, min_y, std::max(1.0f, max_x - min_x), std::max(1.0f, max_y - min_y));
    }

    struct ProcessedCrop
    {
        cv::Mat img;
        float center_x = 0.0f, center_y = 0.0f, scale_size = 0.0f;
    };

    ProcessedCrop MakeProcessedCrop(const cv::Mat &frame, const cv::Rect2f *bbox, float pad_scale)
    {
        ProcessedCrop result;
        float cx = frame.cols * 0.5f, cy = frame.rows * 0.5f;
        float size = static_cast<float>(std::max(frame.cols, frame.rows));

        if (bbox)
        {
            cx = bbox->x + bbox->width * 0.5f;
            cy = bbox->y + bbox->height * 0.5f;
            size = bbox->height * pad_scale;
        }

        const float half = size * 0.5f;
        const int x0 = static_cast<int>(std::floor(cx - half)), y0 = static_cast<int>(std::floor(cy - half));
        const int x1 = static_cast<int>(std::ceil(cx + half)), y1 = static_cast<int>(std::ceil(cy + half));

        cv::Mat cropped(y1 - y0, x1 - x0, frame.type(), cv::Scalar(0, 0, 0));
        const int src_x0 = std::max(0, x0), src_y0 = std::max(0, y0);
        const int src_x1 = std::min(frame.cols, x1), src_y1 = std::min(frame.rows, y1);

        if (src_x1 - src_x0 > 0 && src_y1 - src_y0 > 0)
        {
            frame(cv::Rect(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0))
                .copyTo(cropped(cv::Rect(src_x0 - x0, src_y0 - y0, src_x1 - src_x0, src_y1 - src_y0)));
        }

        cv::resize(cropped, result.img, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_AREA);
        result.center_x = cx;
        result.center_y = cy;
        result.scale_size = size;
        return result;
    }

    std::vector<float> PreprocessImage(const cv::Mat &img)
    {
        cv::Mat padded;
        if (img.cols == kInputW && img.rows == kInputH)
            padded = img.clone();
        else
        {
            float scale = std::min(static_cast<float>(kInputW) / img.cols, static_cast<float>(kInputH) / img.rows);
            cv::resize(img, padded, cv::Size(), scale, scale, cv::INTER_AREA);
            int pad_w = kInputW - padded.cols, pad_h = kInputH - padded.rows;
            cv::copyMakeBorder(padded, padded, pad_h / 2, pad_h - pad_h / 2, pad_w / 2, pad_w - pad_w / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
        padded.convertTo(padded, CV_32F, 1.0 / 255.0);
        std::vector<cv::Mat> channels(3);
        cv::split(padded, channels);
        std::vector<float> input_data;
        input_data.reserve(3 * kInputH * kInputW);
        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < kInputH; ++h)
            {
                const float *row_ptr = channels[c].ptr<float>(h);
                for (int w = 0; w < kInputW; ++w)
                    input_data.push_back((row_ptr[w] - kMean[c]) / kStd[c]);
            }
        }
        return input_data;
    }
} // namespace

// =====================================================================
// DATA STRUCTURES
// =====================================================================
struct CameraDef
{
    int id;
    cv::Matx33f K;
    cv::Matx33f R;
    cv::Vec3f t;
    std::string video_path;
    std::shared_ptr<cv::VideoCapture> cap;
};

struct ViewHmrResult
{
    bool valid = false;
    ProcessedCrop crop;
    std::vector<float> hmr_bbox;
    SmplifyMultiViewObservation obs;
    SmplResult res;
    cv::Matx33f root_rot;
    cv::Vec3f t_local;
    cv::Rect pose_crop;
};

struct TriangulatedJoint
{
    int smpl_idx;
    cv::Point3f pt3d;
    float conf;
};

const std::vector<std::pair<int, int>> halpe26_to_smpl = {
    {11, 1}, {12, 2}, {13, 4}, {14, 5}, {15, 7}, {16, 8}, {20, 10}, {21, 11}, {18, 12}, {5, 16}, {6, 17}, {7, 18}, {8, 19}, {9, 20}, {10, 21}};

void SaveTriangulatedPly(const std::vector<TriangulatedJoint> &joints, const std::string &filename)
{
    std::ofstream out(filename);
    if (!out)
        return;

    // Write PLY Header
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << joints.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    // Write Vertex Data
    for (const auto &j : joints)
    {
        // We color arm joints Red (255,0,0) and others White (255,255,255) to make them stand out
        int r = 255, g = 255, b = 255;
        if (j.smpl_idx >= 16 && j.smpl_idx <= 21)
        {
            g = 0;
            b = 0; // Make arms red
        }
        out << j.pt3d.x << " " << j.pt3d.y << " " << j.pt3d.z << " "
            << r << " " << g << " " << b << "\n";
    }
    out.close();
    std::cout << "  [Debug] Saved triangulated skeleton to: " << filename << std::endl;
}

// =====================================================================
// ARCHITECTURE STEP 1: OPEN-CV EXTRINSICS & N-VIEW TRIANGULATION
// =====================================================================
cv::Point3f TriangulateSVD(const std::vector<cv::Point2f> &points2d, const std::vector<cv::Matx34f> &P_matrices)
{
    int num_views = points2d.size();
    cv::Mat A(num_views * 2, 4, CV_32F);
    for (int i = 0; i < num_views; ++i)
    {
        float x = points2d[i].x;
        float y = points2d[i].y;
        for (int j = 0; j < 4; ++j)
        {
            A.at<float>(i * 2, j) = x * P_matrices[i](2, j) - P_matrices[i](0, j);
            A.at<float>(i * 2 + 1, j) = y * P_matrices[i](2, j) - P_matrices[i](1, j);
        }
    }
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat X = vt.row(3);
    float w_coord = X.at<float>(0, 3);
    return cv::Point3f(X.at<float>(0, 0) / w_coord, X.at<float>(0, 1) / w_coord, X.at<float>(0, 2) / w_coord);
}

void TriangulatePerfectCameras(const std::vector<CameraDef> &cameras, const std::vector<ViewHmrResult> &views, std::vector<TriangulatedJoint> &out_joints)
{
    std::vector<cv::Matx34f> P(3);
    for (int i = 0; i < 3; ++i)
    {
        cv::Matx34f Rt;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
                Rt(r, c) = cameras[i].R(r, c);
            Rt(r, 3) = cameras[i].t[r];
        }
        P[i] = cameras[i].K * Rt;
    }

    out_joints.clear();
    for (const auto &map : halpe26_to_smpl)
    {
        int halpe_idx = map.first;
        int smpl_idx = map.second;

        std::vector<cv::Point2f> active_pts;
        std::vector<cv::Matx34f> active_P;
        float avg_conf = 0.0f;

        for (int i = 0; i < 3; ++i)
        {
            if (!views[i].valid)
                continue;
            float conf = views[i].obs.keypoint_scores[halpe_idx];
            if (conf > 0.001f)
            {
                active_pts.push_back(views[i].obs.keypoints[halpe_idx]);
                active_P.push_back(P[i]);
                avg_conf += conf;
            }
        }

        if (active_pts.size() >= 2)
        {
            TriangulatedJoint tj;
            tj.smpl_idx = smpl_idx;
            tj.pt3d = TriangulateSVD(active_pts, active_P);
            if (std::isnan(tj.pt3d.x) || std::isnan(tj.pt3d.y) || std::isnan(tj.pt3d.z) || std::abs(tj.pt3d.z) > 100.0f)
                continue;
            tj.conf = avg_conf / active_pts.size();
            if (smpl_idx >= 16 && smpl_idx <= 21)
                tj.conf *= 2.0f;
            out_joints.push_back(tj);
        }
    }
}

std::vector<TriangulatedJoint> RefineTriangulatedJointsAutograd(
    const std::vector<CameraDef> &cameras,
    const std::vector<ViewHmrResult> &views,
    const std::vector<TriangulatedJoint> &triangulated_joints,
    double conf_thresh = 1e-4,
    double huber_delta_px = 3.0,
    int iters = 60,
    double lr = 0.003)
{
    if (triangulated_joints.empty())
    {
        return triangulated_joints;
    }

    std::unordered_map<int, int> smpl_to_halpe;
    smpl_to_halpe.reserve(halpe26_to_smpl.size());
    for (const auto &m : halpe26_to_smpl)
    {
        smpl_to_halpe[m.second] = m.first;
    }

    const auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    const int64_t J = static_cast<int64_t>(triangulated_joints.size());
    auto X0 = torch::zeros({J, 3}, opts);
    auto X0_acc = X0.accessor<double, 2>();
    for (int64_t j = 0; j < J; ++j)
    {
        X0_acc[j][0] = triangulated_joints[j].pt3d.x;
        X0_acc[j][1] = triangulated_joints[j].pt3d.y;
        X0_acc[j][2] = triangulated_joints[j].pt3d.z;
    }

    struct ViewObs
    {
        bool active = false;
        torch::Tensor R;    // [3,3]
        torch::Tensor t;    // [3]
        torch::Tensor uv;   // [J,2]
        torch::Tensor conf; // [J]
        torch::Tensor inlier; // [J] bool
        double conf_scale = 1.0;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
    };

    std::vector<ViewObs> obs(3);
    for (int i = 0; i < 3; ++i)
    {
        obs[i].inlier = torch::zeros({J}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    }
    int active_view_count = 0;
    for (int i = 0; i < 3; ++i)
    {
        if (!views[i].valid)
            continue;

        obs[i].active = true;
        obs[i].fx = static_cast<double>(cameras[i].K(0, 0));
        obs[i].fy = static_cast<double>(cameras[i].K(1, 1));
        obs[i].cx = static_cast<double>(cameras[i].K(0, 2));
        obs[i].cy = static_cast<double>(cameras[i].K(1, 2));
        obs[i].R = torch::tensor({
                                     {static_cast<double>(cameras[i].R(0, 0)), static_cast<double>(cameras[i].R(0, 1)), static_cast<double>(cameras[i].R(0, 2))},
                                     {static_cast<double>(cameras[i].R(1, 0)), static_cast<double>(cameras[i].R(1, 1)), static_cast<double>(cameras[i].R(1, 2))},
                                     {static_cast<double>(cameras[i].R(2, 0)), static_cast<double>(cameras[i].R(2, 1)), static_cast<double>(cameras[i].R(2, 2))},
                                 },
                                 opts);
        obs[i].t = torch::tensor({
                                     static_cast<double>(cameras[i].t[0]),
                                     static_cast<double>(cameras[i].t[1]),
                                     static_cast<double>(cameras[i].t[2]),
                                 },
                                 opts);
        obs[i].uv = torch::zeros({J, 2}, opts);
        obs[i].conf = torch::zeros({J}, opts);

        auto uv_acc = obs[i].uv.accessor<double, 2>();
        auto conf_acc = obs[i].conf.accessor<double, 1>();
        for (int64_t j = 0; j < J; ++j)
        {
            const int smpl_idx = triangulated_joints[j].smpl_idx;
            auto it = smpl_to_halpe.find(smpl_idx);
            if (it == smpl_to_halpe.end())
                continue;
            const int halpe_idx = it->second;
            if (halpe_idx < 0 || halpe_idx >= static_cast<int>(views[i].obs.keypoint_scores.size()))
                continue;

            conf_acc[j] = static_cast<double>(views[i].obs.keypoint_scores[halpe_idx]);
            uv_acc[j][0] = static_cast<double>(views[i].obs.keypoints[halpe_idx].x);
            uv_acc[j][1] = static_cast<double>(views[i].obs.keypoints[halpe_idx].y);
        }

        obs[i].conf_scale = std::max(1e-8, obs[i].conf.max().item<double>());
        active_view_count++;
    }

    if (active_view_count < 2)
    {
        return triangulated_joints;
    }

    // One-shot residual-based outlier pruning from triangulated initialization.
    {
        std::vector<std::vector<double>> residuals(3, std::vector<double>(J, std::numeric_limits<double>::infinity()));
        std::vector<std::vector<bool>> valid(3, std::vector<bool>(J, false));

        for (int i = 0; i < 3; ++i)
        {
            if (!obs[i].active)
                continue;

            auto Xc0 = torch::matmul(X0, obs[i].R.transpose(0, 1)) + obs[i].t;
            auto Z0 = Xc0.index({torch::indexing::Slice(), 2});
            auto Z0c = Z0.clamp_min(1e-6);
            auto z_mask = Z0 > 1e-6;
            auto u0 = obs[i].fx * Xc0.index({torch::indexing::Slice(), 0}) / Z0c + obs[i].cx;
            auto v0 = obs[i].fy * Xc0.index({torch::indexing::Slice(), 1}) / Z0c + obs[i].cy;
            auto uv0 = torch::stack({u0, v0}, 1);
            auto r0 = uv0 - obs[i].uv;
            auto s0 = torch::sqrt((r0 * r0).sum(1) + 1e-12);

            auto z_acc = z_mask.accessor<bool, 1>();
            auto conf_acc = obs[i].conf.accessor<double, 1>();
            auto s_acc = s0.accessor<double, 1>();
            for (int64_t j = 0; j < J; ++j)
            {
                if (z_acc[j] && conf_acc[j] > conf_thresh)
                {
                    valid[i][j] = true;
                    residuals[i][j] = s_acc[j];
                }
            }
        }

        auto inlier_acc0 = obs[0].inlier.accessor<bool, 1>();
        auto inlier_acc1 = obs[1].inlier.accessor<bool, 1>();
        auto inlier_acc2 = obs[2].inlier.accessor<bool, 1>();

        for (int64_t j = 0; j < J; ++j)
        {
            bool keep[3] = {valid[0][j], valid[1][j], valid[2][j]};
            int count = static_cast<int>(keep[0]) + static_cast<int>(keep[1]) + static_cast<int>(keep[2]);

            if (count >= 3)
            {
                double r0 = residuals[0][j];
                double r1 = residuals[1][j];
                double r2 = residuals[2][j];
                int worst = 0;
                double worst_val = r0;
                if (r1 > worst_val)
                {
                    worst = 1;
                    worst_val = r1;
                }
                if (r2 > worst_val)
                {
                    worst = 2;
                    worst_val = r2;
                }

                std::array<double, 3> sorted = {r0, r1, r2};
                std::sort(sorted.begin(), sorted.end());
                const double median = sorted[1];
                if (worst_val > 25.0 || worst_val > 2.5 * median)
                {
                    keep[worst] = false;
                }
            }

            inlier_acc0[j] = keep[0];
            inlier_acc1[j] = keep[1];
            inlier_acc2[j] = keep[2];
        }
    }

    auto X = X0.clone().set_requires_grad(true);
    torch::optim::Adam optimizer({X}, torch::optim::AdamOptions(lr));

    for (int iter = 0; iter < iters; ++iter)
    {
        optimizer.zero_grad();
        auto total_loss = torch::zeros({}, opts);
        int contributing_views = 0;

        for (int i = 0; i < 3; ++i)
        {
            if (!obs[i].active)
                continue;

            auto Xc = torch::matmul(X, obs[i].R.transpose(0, 1)) + obs[i].t;
            auto Z = Xc.index({torch::indexing::Slice(), 2});
            auto Zc = Z.clamp_min(1e-6);
            auto z_mask = Z > 1e-6;
            auto u = obs[i].fx * Xc.index({torch::indexing::Slice(), 0}) / Zc + obs[i].cx;
            auto v = obs[i].fy * Xc.index({torch::indexing::Slice(), 1}) / Zc + obs[i].cy;
            auto uv_hat = torch::stack({u, v}, 1);

            auto conf_mask = obs[i].conf > conf_thresh;
            auto mask = conf_mask & z_mask & obs[i].inlier;
            if (!mask.any().item<bool>())
                continue;

            auto r = uv_hat.index({mask}) - obs[i].uv.index({mask});
            auto s = torch::sqrt((r * r).sum(1) + 1e-12);
            auto delta = huber_delta_px;
            auto wr = torch::where(s <= delta, torch::ones_like(s), delta / s);
            auto data = (wr.unsqueeze(1) * r).pow(2).sum(1);

            auto c = obs[i].conf.index({mask});
            auto c_norm = (c / obs[i].conf_scale).clamp(0.0, 1.0);
            auto w = c_norm * c_norm;
            total_loss = total_loss + (w * data).mean();
            contributing_views++;
        }

        if (contributing_views == 0)
            break;

        // Keep solution close to triangulation when observations are sparse/noisy.
        total_loss = total_loss + 1e-6 * (X - X0).pow(2).mean();
        total_loss.backward();
        optimizer.step();

        // Log the current reprojection error for debugging.
        if ((iter + 1) % 10 == 0 || iter == iters - 1)
        {
            std::cout << "    Iter " << iter + 1 << "/" << iters << ", Loss: " << total_loss.item<double>() << std::endl;
        }
    }

    auto X_ref = X.detach();
    auto ref_acc = X_ref.accessor<double, 2>();
    std::vector<TriangulatedJoint> refined = triangulated_joints;
    for (int64_t j = 0; j < J; ++j)
    {
        refined[j].pt3d.x = static_cast<float>(ref_acc[j][0]);
        refined[j].pt3d.y = static_cast<float>(ref_acc[j][1]);
        refined[j].pt3d.z = static_cast<float>(ref_acc[j][2]);
    }
    return refined;
}

// =====================================================================
// MAIN VIDEO LOOP
// =====================================================================
bool ProcessViewPose(cv::Mat &frame, YoloPersonDetector &yolo, RtmPoseDetector &rtmpose, float focal_scale,
                     std::vector<cv::Point2f> &out_keypoints, std::vector<float> &out_scores,
                     cv::Rect &out_pose_crop, ProcessedCrop &out_crop_res, std::vector<float> &out_hmr_bbox)
{
    cv::Rect2f yolo_bbox;
    float yolo_score = 0.0f;
    if (!yolo.DetectPerson(frame, &yolo_bbox, &yolo_score))
        return false;

    cv::Rect yolo_crop = MakePaddedRect(yolo_bbox, frame.cols, frame.rows, 1.2f);
    if (!rtmpose.DetectPose(frame(yolo_crop), &out_keypoints, &out_scores))
        return false;

    for (auto &kpt : out_keypoints)
    {
        kpt.x += yolo_crop.x;
        kpt.y += yolo_crop.y;
    }

    auto bbox_opt = PoseKeypointsToBbox(out_keypoints, out_scores, 0.0001f, frame.cols, frame.rows);
    if (!bbox_opt)
        return false;

    cv::Rect2f pose_bbox = bbox_opt.value();
    out_crop_res = MakeProcessedCrop(frame, &pose_bbox, 1.1f);

    const float half = out_crop_res.scale_size * 0.5f;
    out_pose_crop = cv::Rect(static_cast<int>(out_crop_res.center_x - half), static_cast<int>(out_crop_res.center_y - half),
                             static_cast<int>(out_crop_res.scale_size), static_cast<int>(out_crop_res.scale_size)) &
                    cv::Rect(0, 0, frame.cols, frame.rows);

    float f_geo = std::max(frame.cols, frame.rows) * focal_scale;
    out_hmr_bbox = GetCliffBBox(out_crop_res.center_x, out_crop_res.center_y, out_crop_res.scale_size, f_geo, frame.cols, frame.rows);
    return true;
}

int main(int argc, char *argv[])
{
    std::string hmr_model_path, rtmpose_model_path, yolo_model_path, modnet_model_path, output_dir_path;
    bool modnet_use_cuda = false;
    int frame_stride = 1;
    std::vector<std::string> video_paths;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc)
            output_dir_path = argv[++i];
        else if (arg == "--modnet" && i + 1 < argc)
            modnet_model_path = argv[++i];
        else if (arg == "--modnet-cuda")
            modnet_use_cuda = true;
        else if (arg == "--frame-stride" && i + 1 < argc)
            frame_stride = std::max(1, std::stoi(argv[++i]));
        else if (hmr_model_path.empty())
            hmr_model_path = arg;
        else if (rtmpose_model_path.empty())
            rtmpose_model_path = arg;
        else if (yolo_model_path.empty())
            yolo_model_path = arg;
        else
            video_paths.push_back(arg);
    }

    if (video_paths.size() != 3 || output_dir_path.empty())
    {
        std::cout << "Usage: hmr_multiview <hmr> <rtm> <yolo> --output <dir> <vid0> <vid1> <vid2>\n";
        return -1;
    }

    fs::path out_dir(output_dir_path);
    fs::create_directories(out_dir / "overlays");
    fs::create_directories(out_dir / "opt_vis");

    std::vector<CameraDef> cameras(3);
    for (int i = 0; i < 3; ++i)
    {
        cameras[i].id = i;
        cameras[i].video_path = video_paths[i];
        cameras[i].cap = std::make_shared<cv::VideoCapture>(cameras[i].video_path);
        cameras[i].R = cv::Matx33f::eye();
        cameras[i].t = cv::Vec3f(0, 0, 0);
    }

    // 1. YOLO Initialization
    YoloPersonDetectorOptions yolo_opts;
    // The default input_size is likely already correct in the struct,
    // but you can explicitly set it here if needed (e.g., yolo_opts.input_size = 640;)
    yolo_opts.conf_threshold = 0.5f;
    yolo_opts.nms_threshold = 0.4f;
    yolo_opts.use_cuda = true;
    YoloPersonDetector yolo(yolo_opts);
    yolo.Load(yolo_model_path);

    // 2. RTMPose Initialization
    RtmPoseDetectorOptions rtm_opts;
    rtm_opts.use_cuda = true;
    RtmPoseDetector rtmpose(rtm_opts);
    rtmpose.Load(rtmpose_model_path);

    // 3. ModNet

    bool use_modnet = !modnet_model_path.empty();
    ModNetMatteOptions modnet_opts;
    modnet_opts.input_size = 512;
    modnet_opts.use_cuda = modnet_use_cuda;
    ModNetMatte modnet(modnet_opts);
    if (use_modnet)
        use_modnet = modnet.Load(modnet_model_path);

    TrtLogger logger;
    auto engine = BuildEngineFromOnnx(hmr_model_path, logger);
    auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    void *d_image, *d_bbox, *d_pose, *d_betas, *d_cam;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMalloc(&d_image, 1 * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc(&d_bbox, 3 * sizeof(float));
    cudaMalloc(&d_pose, 144 * sizeof(float));
    cudaMalloc(&d_betas, 10 * sizeof(float));
    cudaMalloc(&d_cam, 3 * sizeof(float));
    std::vector<void *> bindings = {d_image, d_bbox, d_pose, d_betas, d_cam};
    std::vector<float> pose_out(144), betas_out(10), cam_out(3);

    int frame_idx = 0;
    while (true)
    {
        std::vector<cv::Mat> frames(3);
        bool all_read = true;
        for (int i = 0; i < 3; ++i)
            if (!cameras[i].cap->read(frames[i]))
                all_read = false;
        if (!all_read)
            break;

        if (frame_idx % frame_stride != 0 || frame_idx < 300 || frame_idx > 301)
        {
            frame_idx++;
            continue;
        }

        std::vector<ViewHmrResult> view_results(3);
        int valid_view_count = 0;

        for (int i = 0; i < 3; ++i)
        {
            std::vector<cv::Point2f> kpts;
            std::vector<float> scores;
            cv::Rect pose_crop;
            ProcessedCrop crop_res;
            std::vector<float> hmr_bbox;

            if (ProcessViewPose(frames[i], yolo, rtmpose, 1.2f, kpts, scores, pose_crop, crop_res, hmr_bbox))
            {
                float f_geo = std::max(frames[i].cols, frames[i].rows) * 1.2f;
                cameras[i].K = cv::Matx33f(f_geo, 0, frames[i].cols / 2.0f, 0, f_geo, frames[i].rows / 2.0f, 0, 0, 1);

                view_results[i].valid = true;
                view_results[i].crop = crop_res;
                view_results[i].hmr_bbox = hmr_bbox;
                view_results[i].pose_crop = pose_crop;
                view_results[i].obs.keypoints = kpts;
                view_results[i].obs.keypoint_scores = scores;
                view_results[i].obs.K = cameras[i].K;
                view_results[i].obs.R = cv::Matx33f::eye();
                view_results[i].obs.t = cv::Vec3f(0, 0, 0);
                view_results[i].obs.img_w = static_cast<float>(frames[i].cols);
                view_results[i].obs.img_h = static_cast<float>(frames[i].rows);

                valid_view_count++;
            }
        }

        if (valid_view_count < 2 || !view_results[0].valid)
        {
            frame_idx++;
            continue;
        }

        for (int i = 0; i < 3; ++i)
        {
            if (!view_results[i].valid)
                continue;

            std::vector<float> img_data = PreprocessImage(view_results[i].crop.img);
            cudaMemcpyAsync(d_image, img_data.data(), img_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_bbox, view_results[i].hmr_bbox.data(), view_results[i].hmr_bbox.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
            context->enqueueV2(bindings.data(), stream, nullptr);

            cudaMemcpyAsync(pose_out.data(), d_pose, 144 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(betas_out.data(), d_betas, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(cam_out.data(), d_cam, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            view_results[i].res.pose = pose_out;
            view_results[i].res.shape = betas_out;
            view_results[i].res.camera = cam_out;

            // Initial relative extrinsics from HMR to seed the optimizer
            view_results[i].t_local = EstimateTranslation(cam_out, view_results[i].crop.center_x, view_results[i].crop.center_y,
                                                          view_results[i].crop.scale_size, cameras[i].K(0, 0), frames[i].cols, frames[i].rows);
            view_results[i].root_rot = cv::Matx33f((float *)Rot6dToRotMatSingle(pose_out.data()).data);
        }

        // ==========================================
        // JOINT ARCHITECTURE IMPLEMENTATION
        // ==========================================

        // 1. Initial Guess: Lock View 0 at origin, compute relative start for others
        cameras[0].R = cv::Matx33f::eye();
        cameras[0].t = cv::Vec3f(0, 0, 0);
        cv::Matx33f R0_inv = view_results[0].root_rot.t();
        for (int i = 1; i < 3; ++i)
        {
            if (view_results[i].valid)
            {
                cameras[i].R = view_results[i].root_rot * R0_inv;
                cameras[i].t = view_results[i].t_local - (cameras[i].R * view_results[0].t_local);
            }
        }

        // 2. Triangulate 3D joints directly from multiview 2D keypoints
        std::vector<TriangulatedJoint> triangulated_joints;
        TriangulatePerfectCameras(cameras, view_results, triangulated_joints);
        if (!triangulated_joints.empty())
        {
            auto refined_joints = RefineTriangulatedJointsAutograd(cameras, view_results, triangulated_joints);

            char ply_name[64];
            snprintf(ply_name, sizeof(ply_name), "triangulated_frame_%05d.ply", frame_idx);
            SaveTriangulatedPly(refined_joints, (out_dir / "opt_vis" / ply_name).string());

            float total_frame_error = 0.0f;
            int total_frame_valid_points = 0;

            std::cout << "\n--- Triangulation Reprojection Error (Frame " << frame_idx << ") ---\n";

            for (int i = 0; i < 3; ++i)
            {
                if (!view_results[i].valid)
                    continue;

                float view_error_sum = 0.0f;
                int view_valid_points = 0;
                cv::Mat overlay = frames[i].clone();

                // Project triangulated joints vs original 2D RTMPose keypoints
                for (const auto &tj : refined_joints)
                {
                    int halpe_idx = -1;
                    for (const auto &map : halpe26_to_smpl)
                    {
                        if (map.second == tj.smpl_idx)
                        {
                            halpe_idx = map.first;
                            break;
                        }
                    }
                    if (halpe_idx < 0)
                        continue;

                    if (view_results[i].obs.keypoint_scores[halpe_idx] > 0.001f)
                    {
                        cv::Vec3f p_world(tj.pt3d.x, tj.pt3d.y, tj.pt3d.z);
                        cv::Vec3f p_cam = cameras[i].R * p_world + cameras[i].t;

                        if (p_cam[2] > 1e-3f)
                        {
                            float u_proj = (cameras[i].K(0, 0) * p_cam[0] / p_cam[2]) + cameras[i].K(0, 2);
                            float v_proj = (cameras[i].K(1, 1) * p_cam[1] / p_cam[2]) + cameras[i].K(1, 2);

                            cv::Point2f p2d_rtm = view_results[i].obs.keypoints[halpe_idx];
                            cv::Point2f p2d_reproj(u_proj, v_proj);

                            float dist = cv::norm(p2d_rtm - p2d_reproj);
                            view_error_sum += dist;
                            view_valid_points++;

                            if (u_proj >= 0 && u_proj < overlay.cols && v_proj >= 0 && v_proj < overlay.rows)
                            {
                                cv::line(overlay, p2d_rtm, p2d_reproj, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                                cv::circle(overlay, p2d_rtm, 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
                                cv::circle(overlay, p2d_reproj, 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
                            }
                        }
                    }
                }

                if (view_valid_points > 0)
                {
                    float mean_view_err = view_error_sum / view_valid_points;
                    std::cout << "  View " << i << " Mean Error: " << std::fixed << std::setprecision(2) << mean_view_err << " px\n";
                    total_frame_error += view_error_sum;
                    total_frame_valid_points += view_valid_points;
                }

                char filename[64];
                snprintf(filename, sizeof(filename), "overlay_%05d_view_%d.jpg", frame_idx, i);
                cv::imwrite((out_dir / "overlays" / filename).string(), overlay);
            }

            if (total_frame_valid_points > 0)
                std::cout << "  -> Frame Average Reprojection Error: " << (total_frame_error / total_frame_valid_points) << " px\n";
        }
        frame_idx++;
    }

    cudaFree(d_image);
    cudaFree(d_bbox);
    cudaFree(d_pose);
    cudaFree(d_betas);
    cudaFree(d_cam);
    cudaStreamDestroy(stream);
    return 0;
}
