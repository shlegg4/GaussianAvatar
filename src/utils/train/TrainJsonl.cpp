#include "utils/train/TrainJsonl.h"

#include <algorithm>
#include <cstdlib>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <utility>

namespace
{

cv::Matx33f ProjectionSignFlip()
{
    return cv::Matx33f(-1.0f, 0.0f, 0.0f,
                       0.0f, -1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Matx33f ExtractCameraRotation(const std::vector<float> &camera_rt)
{
    if (camera_rt.size() >= 16u)
    {
        return cv::Matx33f(camera_rt[0], camera_rt[1], camera_rt[2],
                           camera_rt[4], camera_rt[5], camera_rt[6],
                           camera_rt[8], camera_rt[9], camera_rt[10]);
    }
    return cv::Matx33f::eye();
}

void ApplyExtraRootRotation(const cv::Matx33f &extra_rotation, std::vector<float> *pose_axis_angle)
{
    if (pose_axis_angle == nullptr || pose_axis_angle->size() < 3u)
    {
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

std::vector<float> BuildLegacyPoseFromSmplx(const std::vector<float> &global_orient,
                                            const std::vector<float> &body_pose)
{
    std::vector<float> pose(72u, 0.0f);
    const size_t global_count = std::min<size_t>(3u, global_orient.size());
    std::copy_n(global_orient.begin(), global_count, pose.begin());

    const size_t body_count = std::min<size_t>(63u, body_pose.size());
    std::copy_n(body_pose.begin(), body_count, pose.begin() + 3);
    return pose;
}

bool TryBuildTranslationFromCameraRt(const std::vector<float> &camera_rt,
                                     std::array<float, 3> *translation)
{
    if (translation == nullptr || camera_rt.size() < 16u)
    {
        return false;
    }

    (*translation)[0] = -camera_rt[3];
    (*translation)[1] = -camera_rt[7];
    (*translation)[2] = camera_rt[11];
    return true;
}

} // namespace

bool ExtractStringField(const std::string &line, const std::string &key, std::string *out)
{
    const std::string tag = "\"" + key + "\":\"";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    size_t end = line.find('"', start);
    if (end == std::string::npos)
        return false;
    *out = line.substr(start, end - start);
    return true;
}

bool ExtractNumberField(const std::string &line, const std::string &key, double *out)
{
    const std::string tag = "\"" + key + "\":";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    const char *ptr = line.c_str() + start;
    char *end = nullptr;
    double val = std::strtod(ptr, &end);
    if (end == ptr)
        return false;
    *out = val;
    return true;
}

bool ExtractArrayField(const std::string &line, const std::string &key, std::vector<float> *out)
{
    const std::string tag = "\"" + key + "\":[";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    size_t end = line.find(']', start);
    if (end == std::string::npos)
        return false;
    std::string content = line.substr(start, end - start);
    out->clear();
    if (content.empty())
        return true;
    size_t pos = 0;
    while (pos < content.size())
    {
        size_t comma = content.find(',', pos);
        std::string token = (comma == std::string::npos) ? content.substr(pos) : content.substr(pos, comma - pos);
        if (!token.empty())
        {
            char *end_ptr = nullptr;
            float val = std::strtof(token.c_str(), &end_ptr);
            if (end_ptr != token.c_str())
            {
                out->push_back(val);
            }
        }
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return true;
}

bool ParseTrainSample(const std::string &line, TrainSample *out)
{
    TrainSample sample;
    std::vector<float> smplx_shape;
    std::vector<float> smplx_global_orient;
    std::vector<float> smplx_body_pose;
    std::vector<float> smplx_expression;
    std::vector<float> smplx_jaw_pose;
    std::vector<float> smplx_eye_pose;
    std::vector<float> smplx_left_hand_pose;
    std::vector<float> smplx_right_hand_pose;
    double value = 0.0;
    if (ExtractNumberField(line, "frame", &value))
    {
        sample.frame = static_cast<int>(value);
    }
    if (ExtractStringField(line, "body_model", &sample.body_model) && sample.body_model.empty())
    {
        sample.body_model = "smpl";
    }
    if (!ExtractStringField(line, "crop", &sample.crop_path))
        return false;
    if (sample.crop_path.empty())
        return false;

    if (!ExtractNumberField(line, "img_w", &value))
        return false;
    sample.img_w = static_cast<int>(value);
    if (!ExtractNumberField(line, "img_h", &value))
        return false;
    sample.img_h = static_cast<int>(value);
    if (!ExtractNumberField(line, "crop_cx", &value))
        return false;
    sample.crop_cx = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_cy", &value))
        return false;
    sample.crop_cy = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_size", &value))
        return false;
    sample.crop_size = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_x0", &value))
        sample.crop_x0 = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_y0", &value))
        sample.crop_y0 = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_w", &value))
        sample.crop_w = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_h", &value))
        sample.crop_h = static_cast<float>(value);
    if (!ExtractNumberField(line, "focal_length", &value))
        return false;
    sample.focal_length = static_cast<float>(value);
    if (!ExtractNumberField(line, "y_sign", &value))
        return false;
    sample.y_sign = static_cast<float>(value);

    const bool has_smplx_shape = ExtractArrayField(line, "smplx_shape", &smplx_shape) && !smplx_shape.empty();
    const bool has_smplx_global = ExtractArrayField(line, "smplx_global_orient", &smplx_global_orient) && !smplx_global_orient.empty();
    const bool has_smplx_body = ExtractArrayField(line, "smplx_body_pose", &smplx_body_pose) && !smplx_body_pose.empty();
    ExtractArrayField(line, "smplx_expression", &smplx_expression);
    ExtractArrayField(line, "smplx_jaw_pose", &smplx_jaw_pose);
    ExtractArrayField(line, "smplx_eye_pose", &smplx_eye_pose);
    ExtractArrayField(line, "smplx_left_hand_pose", &smplx_left_hand_pose);
    ExtractArrayField(line, "smplx_right_hand_pose", &smplx_right_hand_pose);

    if (!ExtractArrayField(line, "camera_rt", &sample.camera_rt))
    {
        sample.camera_rt.clear();
    }
    sample.has_translation = TryBuildTranslationFromCameraRt(sample.camera_rt, &sample.translation);
    sample.uses_smplx = has_smplx_shape || has_smplx_global || has_smplx_body ||
                        !smplx_expression.empty() || !smplx_jaw_pose.empty() ||
                        !smplx_eye_pose.empty() || !smplx_left_hand_pose.empty() ||
                        !smplx_right_hand_pose.empty() || sample.body_model == "smplx";
    if (sample.uses_smplx)
    {
        sample.body_model = "smplx";
        sample.smplx_expression = std::move(smplx_expression);
        sample.smplx_jaw_pose = std::move(smplx_jaw_pose);
        sample.smplx_eye_pose = std::move(smplx_eye_pose);
        sample.smplx_left_hand_pose = std::move(smplx_left_hand_pose);
        sample.smplx_right_hand_pose = std::move(smplx_right_hand_pose);
    }

    if (has_smplx_shape)
    {
        sample.betas = smplx_shape;
    }
    else if (!ExtractArrayField(line, "betas", &sample.betas))
    {
        return false;
    }

    if (has_smplx_global && has_smplx_body)
    {
        sample.pose = BuildLegacyPoseFromSmplx(smplx_global_orient, smplx_body_pose);
        if (sample.camera_rt.size() >= 16u)
        {
            const cv::Matx33f train_rotation = ProjectionSignFlip() * ExtractCameraRotation(sample.camera_rt);
            ApplyExtraRootRotation(train_rotation, &sample.pose);
        }
    }
    else if (!ExtractArrayField(line, "pose", &sample.pose))
    {
        return false;
    }

    if (!ExtractArrayField(line, "cam", &sample.cam))
    {
        sample.cam.clear();
    }

    if (sample.pose.empty() || sample.betas.empty() || (!sample.has_translation && sample.cam.empty()))
        return false;
    *out = std::move(sample);
    return true;
}
