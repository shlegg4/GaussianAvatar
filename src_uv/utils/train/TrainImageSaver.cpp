#include "utils/train/TrainImageSaver.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/image/TensorCvUtils.h"

namespace
{
constexpr size_t kSmplJointCount = 24u;

const std::array<const char *, kSmplJointCount> kSmplJointLabels = {
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand"};

struct PoseJsonRecord
{
    size_t sample_index = 0;
    int frame = -1;
    std::string image_file;
    PoseSampleExport pose;
};

struct SmplxJsonRecord
{
    size_t sample_index = 0;
    int frame = -1;
    std::string image_file;
    std::string crop_path;
    SmplxParamsExport params;
};

std::string EscapeJsonString(const std::string &value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value)
    {
        switch (ch)
        {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += ch;
            break;
        }
    }
    return escaped;
}

void WriteVec3(std::ostream &out, const std::array<float, 3> &value)
{
    out << "[" << value[0] << ", " << value[1] << ", " << value[2] << "]";
}

void WriteFloatVector(std::ostream &out, const std::vector<float> &values)
{
    out << "[";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void WriteFloatVectorSlice(std::ostream &out,
                           const std::vector<float> &values,
                           size_t offset,
                           size_t count)
{
    out << "[";
    if (offset < values.size())
    {
        const size_t end = std::min(values.size(), offset + count);
        for (size_t i = offset; i < end; ++i)
        {
            if (i > offset)
            {
                out << ", ";
            }
            out << values[i];
        }
    }
    out << "]";
}

void WritePoseBlock(std::ostream &out,
                    const char *block_name,
                    const std::array<float, 3> &transl,
                    const std::array<PoseJointExport, kSmplJointCount> &joints,
                    bool add_comma)
{
    out << "      \"" << block_name << "\": {\n";
    out << "        \"transl\": ";
    WriteVec3(out, transl);
    out << ",\n";
    out << "        \"joints\": [\n";
    for (size_t j = 0; j < kSmplJointCount; ++j)
    {
        out << "          {\"joint\": \"" << kSmplJointLabels[j] << "\", \"rot\": ";
        WriteVec3(out, joints[j].rot);
        out << ", \"transl\": ";
        WriteVec3(out, joints[j].transl);
        out << "}";
        if (j + 1 < kSmplJointCount)
        {
            out << ",";
        }
        out << "\n";
    }
    out << "        ]\n";
    out << "      }";
    if (add_comma)
    {
        out << ",";
    }
    out << "\n";
}

bool WritePoseBlocksJson(const std::filesystem::path &json_path,
                         int epoch,
                         const std::vector<PoseJsonRecord> &records)
{
    std::ofstream json_file(json_path);
    if (!json_file.is_open())
    {
        std::cerr << "Failed to open pose json path: " << json_path.string() << std::endl;
        return false;
    }

    json_file.setf(std::ios::fixed);
    json_file << std::setprecision(6);

    json_file << "{\n";
    json_file << "  \"epoch\": " << epoch << ",\n";
    json_file << "  \"transl_is_global\": true,\n";
    json_file << "  \"joint_labels\": [\n";
    for (size_t j = 0; j < kSmplJointCount; ++j)
    {
        json_file << "    \"" << kSmplJointLabels[j] << "\"";
        if (j + 1 < kSmplJointCount)
        {
            json_file << ",";
        }
        json_file << "\n";
    }
    json_file << "  ],\n";
    json_file << "  \"samples\": [\n";

    for (size_t i = 0; i < records.size(); ++i)
    {
        const auto &record = records[i];
        json_file << "    {\n";
        json_file << "      \"sample_index\": " << record.sample_index << ",\n";
        json_file << "      \"frame\": " << record.frame << ",\n";
        json_file << "      \"image_file\": \"" << record.image_file << "\",\n";
        WritePoseBlock(json_file, "original_pose", record.pose.original_transl, record.pose.original_pose, true);
        WritePoseBlock(json_file, "pose_delta", record.pose.delta_transl, record.pose.pose_delta, true);
        WritePoseBlock(json_file, "refined_pose", record.pose.refined_transl, record.pose.refined_pose, false);
        json_file << "    }";
        if (i + 1 < records.size())
        {
            json_file << ",";
        }
        json_file << "\n";
    }

    json_file << "  ]\n";
    json_file << "}\n";
    return true;
}

bool WriteEffectiveSmplxJson(const std::filesystem::path &json_path,
                             int epoch,
                             const std::vector<SmplxJsonRecord> &records)
{
    std::ofstream json_file(json_path);
    if (!json_file.is_open())
    {
        std::cerr << "Failed to open effective SMPL-X json path: " << json_path.string() << std::endl;
        return false;
    }

    json_file.setf(std::ios::fixed);
    json_file << std::setprecision(6);

    json_file << "{\n";
    json_file << "  \"epoch\": " << epoch << ",\n";
    json_file << "  \"model_type\": \"smplx\",\n";
    json_file << "  \"pose_layout\": {\n";
    json_file << "    \"global_orient\": 3,\n";
    json_file << "    \"body_pose\": 63,\n";
    json_file << "    \"jaw_pose\": 3,\n";
    json_file << "    \"eye_pose\": 6,\n";
    json_file << "    \"left_hand_pose\": 45,\n";
    json_file << "    \"right_hand_pose\": 45\n";
    json_file << "  },\n";
    json_file << "  \"samples\": [\n";

    for (size_t i = 0; i < records.size(); ++i)
    {
        const auto &record = records[i];
        json_file << "    {\n";
        json_file << "      \"sample_index\": " << record.sample_index << ",\n";
        json_file << "      \"frame\": " << record.frame << ",\n";
        json_file << "      \"image_file\": \"" << EscapeJsonString(record.image_file) << "\",\n";
        json_file << "      \"crop_path\": \"" << EscapeJsonString(record.crop_path) << "\",\n";
        json_file << "      \"body_model\": \"" << EscapeJsonString(record.params.body_model) << "\",\n";
        json_file << "      \"y_sign\": " << record.params.y_sign << ",\n";
        json_file << "      \"effective_smplx_params\": {\n";
        json_file << "        \"betas\": ";
        WriteFloatVector(json_file, record.params.betas);
        json_file << ",\n";
        json_file << "        \"transl\": ";
        WriteVec3(json_file, record.params.transl);
        json_file << ",\n";
        json_file << "        \"global_orient\": ";
        WriteFloatVectorSlice(json_file, record.params.pose_axis_angle, 0u, 3u);
        json_file << ",\n";
        json_file << "        \"body_pose\": ";
        WriteFloatVectorSlice(json_file, record.params.pose_axis_angle, 3u, 63u);
        json_file << ",\n";
        json_file << "        \"pose_axis_angle\": ";
        WriteFloatVector(json_file, record.params.pose_axis_angle);
        json_file << ",\n";
        json_file << "        \"expression\": ";
        WriteFloatVector(json_file, record.params.expression);
        json_file << ",\n";
        json_file << "        \"jaw_pose\": ";
        WriteFloatVector(json_file, record.params.jaw_pose);
        json_file << ",\n";
        json_file << "        \"left_eye_pose\": ";
        WriteFloatVectorSlice(json_file, record.params.eye_pose, 0u, 3u);
        json_file << ",\n";
        json_file << "        \"right_eye_pose\": ";
        WriteFloatVectorSlice(json_file, record.params.eye_pose, 3u, 3u);
        json_file << ",\n";
        json_file << "        \"eye_pose\": ";
        WriteFloatVector(json_file, record.params.eye_pose);
        json_file << ",\n";
        json_file << "        \"left_hand_pose\": ";
        WriteFloatVector(json_file, record.params.left_hand_pose);
        json_file << ",\n";
        json_file << "        \"right_hand_pose\": ";
        WriteFloatVector(json_file, record.params.right_hand_pose);
        json_file << "\n";
        json_file << "      }\n";
        json_file << "    }";
        if (i + 1 < records.size())
        {
            json_file << ",";
        }
        json_file << "\n";
    }

    json_file << "  ]\n";
    json_file << "}\n";
    return true;
}
} // namespace

int SaveEpochViewPairs(const std::vector<TrainSample> &samples,
                       const std::vector<CachedSampleData> &cached,
                       const std::filesystem::path &output_dir,
                       int epoch,
                       const RenderViewFn &render_fn)
{
    if (samples.size() != cached.size())
    {
        std::cerr << "SaveEpochViewPairs: samples/cached size mismatch." << std::endl;
        return 0;
    }

    std::filesystem::path epoch_dir = output_dir / "pairs" / ("epoch_" + std::to_string(epoch));
    std::error_code ec;
    std::filesystem::create_directories(epoch_dir, ec);
    if (ec)
    {
        std::cerr << "Failed to create pair output dir: " << epoch_dir.string() << std::endl;
        return 0;
    }

    int saved = 0;
    std::vector<PoseJsonRecord> pose_records;
    pose_records.reserve(samples.size());
    std::vector<SmplxJsonRecord> smplx_records;
    smplx_records.reserve(samples.size());
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (!cached[i].valid)
        {
            continue;
        }
        const auto &sample = samples[i];
        const auto &cached_entry = cached[i];

        RenderViewResult render_result = render_fn(i, sample, cached_entry);
        torch::Tensor render = render_result.image;
        if (!render.defined())
        {
            continue;
        }
        cv::Mat render_bgr = TensorToBgr(render);
        if (render_bgr.empty())
        {
            continue;
        }

        cv::Mat target_bgr = cached_entry.target_bgr.empty() ? cached_entry.crop_bgr : cached_entry.target_bgr;
        if (target_bgr.empty())
        {
            continue;
        }
        if (render_bgr.size() != target_bgr.size())
        {
            cv::resize(render_bgr, render_bgr, target_bgr.size(), 0, 0, cv::INTER_AREA);
        }

        cv::Mat diff_bgr;
        cv::absdiff(target_bgr, render_bgr, diff_bgr);

        cv::Mat side_by_side;
        std::vector<cv::Mat> panels = {target_bgr, render_bgr, diff_bgr};
        cv::hconcat(panels, side_by_side);

        std::ostringstream name;
        name << "view_" << std::setw(5) << std::setfill('0') << i;
        if (sample.frame >= 0)
        {
            name << "_frame_" << sample.frame;
        }
        name << ".png";

        std::filesystem::path out_path = epoch_dir / name.str();
        if (cv::imwrite(out_path.string(), side_by_side))
        {
            saved++;
            if (render_result.pose_export.valid)
            {
                PoseJsonRecord record;
                record.sample_index = i;
                record.frame = sample.frame;
                record.image_file = name.str();
                record.pose = render_result.pose_export;
                pose_records.push_back(record);
            }
            if (render_result.smplx_export.valid)
            {
                SmplxJsonRecord record;
                record.sample_index = i;
                record.frame = sample.frame;
                record.image_file = name.str();
                record.crop_path = sample.crop_path;
                record.params = render_result.smplx_export;
                smplx_records.push_back(record);
            }
        }
    }

    const auto json_path = epoch_dir / "pose_blocks.json";
    if (!WritePoseBlocksJson(json_path, epoch, pose_records))
    {
        std::cerr << "Failed to write pose block json for epoch " << epoch << std::endl;
    }

    const auto smplx_json_path = epoch_dir / "effective_smplx_params.json";
    if (!WriteEffectiveSmplxJson(smplx_json_path, epoch, smplx_records))
    {
        std::cerr << "Failed to write effective SMPL-X json for epoch " << epoch << std::endl;
    }

    return saved;
}
