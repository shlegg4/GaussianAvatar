#include "dataset_prep/export/DatasetExporter.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

namespace dataset_prep {
namespace {

void WriteVec3(std::ostream& out, const cv::Point3f& point) {
    out << "[" << point.x << ", " << point.y << ", " << point.z << "]";
}

void WriteVec4(std::ostream& out, const cv::Vec4f& value) {
    out << "[" << value[0] << ", " << value[1] << ", " << value[2] << ", " << value[3] << "]";
}

void WriteFloatArray(std::ostream& out,
                     const std::vector<float>& values,
                     size_t expected_count) {
    out << "[";
    for (size_t index = 0; index < expected_count; ++index) {
        if (index > 0u) {
            out << ", ";
        }
        const float value = index < values.size() ? values[index] : 0.0f;
        out << value;
    }
    out << "]";
}

std::string JsonEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8u);
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

}  // namespace

DatasetExporter::DatasetExporter(const Options& options)
    : options_(options) {}

bool DatasetExporter::Initialize() {
    if (options_.output_dir.empty()) {
        return false;
    }

    std::error_code error;
    std::filesystem::create_directories(options_.output_dir / "poses", error);
    if (error) {
        std::cerr << "DatasetExporter: failed to create output directory "
                  << options_.output_dir.string() << std::endl;
        return false;
    }
    return true;
}

bool DatasetExporter::SaveFrame(const ExportFrameArtifacts& artifacts) {
    if (!EnsureCameraDirectories(artifacts)) {
        return false;
    }

    const std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, options_.png_compression};
    if (HasTrainingSamples(artifacts)) {
        if (options_.save_images) {
            for (const auto& sample : artifacts.training_samples) {
                const auto crop_path = MakeTrainingCropPath(artifacts, sample);
                if (sample.crop_image.empty() ||
                    !cv::imwrite(crop_path.string(), sample.crop_image, png_params)) {
                    std::cerr << "DatasetExporter: failed to write "
                              << crop_path.string() << std::endl;
                    return false;
                }
            }
        }

        if (options_.save_masks) {
            for (const auto& sample : artifacts.training_samples) {
                const auto matte_path = MakeTrainingMattePath(artifacts, sample);
                if (sample.crop_matte.empty() ||
                    !cv::imwrite(matte_path.string(), sample.crop_matte, png_params)) {
                    std::cerr << "DatasetExporter: failed to write "
                              << matte_path.string() << std::endl;
                    return false;
                }
            }
        }
    } else {
        const std::string frame_stem = MakeFrameStem(artifacts.synced_frames.sync_index);
        if (options_.save_images) {
            for (const auto& view : artifacts.synced_frames.views) {
                const auto image_path = options_.output_dir / "images" / view.camera_id / (frame_stem + ".png");
                if (!cv::imwrite(image_path.string(), view.image, png_params)) {
                    std::cerr << "DatasetExporter: failed to write " << image_path.string() << std::endl;
                    return false;
                }
            }
        }

        if (options_.save_masks) {
            for (const auto& mask : artifacts.masks) {
                const auto matte_path = options_.output_dir / "masks" / mask.camera_id / (frame_stem + ".png");
                if (!mask.matte.empty() && !cv::imwrite(matte_path.string(), mask.matte, png_params)) {
                    std::cerr << "DatasetExporter: failed to write " << matte_path.string() << std::endl;
                    return false;
                }
            }
        }
    }

    std::filesystem::path pose_path;
    if (options_.save_pose_json) {
        const std::string frame_stem = MakeFrameStem(artifacts.synced_frames.sync_index);
        pose_path = options_.output_dir / "poses" / (frame_stem + ".json");
        if (!WritePoseJson(artifacts, pose_path)) {
            return false;
        }
    }

    if (options_.append_manifest && !AppendManifestLine(artifacts, pose_path)) {
        return false;
    }

    return true;
}

bool DatasetExporter::EnsureCameraDirectories(const ExportFrameArtifacts& artifacts) const {
    std::error_code error;
    if (HasTrainingSamples(artifacts)) {
        for (const auto& sample : artifacts.training_samples) {
            if (options_.save_images) {
                std::filesystem::create_directories(
                    (options_.output_dir / "crops" / sample.camera_id), error);
                if (error) {
                    return false;
                }
            }
            if (options_.save_masks) {
                std::filesystem::create_directories(
                    (options_.output_dir / "mattes" / sample.camera_id), error);
                if (error) {
                    return false;
                }
            }
        }
    } else {
        for (const auto& view : artifacts.synced_frames.views) {
            if (options_.save_images) {
                std::filesystem::create_directories(options_.output_dir / "images" / view.camera_id, error);
                if (error) {
                    return false;
                }
            }
        }
        for (const auto& mask : artifacts.masks) {
            if (options_.save_masks) {
                std::filesystem::create_directories(options_.output_dir / "masks" / mask.camera_id, error);
                if (error) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool DatasetExporter::WritePoseJson(const ExportFrameArtifacts& artifacts,
                                    const std::filesystem::path& pose_path) const {
    std::ofstream output(pose_path);
    if (!output.is_open()) {
        std::cerr << "DatasetExporter: failed to open " << pose_path.string() << std::endl;
        return false;
    }

    output.setf(std::ios::fixed);
    output << std::setprecision(6);

    output << "{\n";
    output << "  \"sync_index\": " << artifacts.synced_frames.sync_index << ",\n";
    output << "  \"sync_timestamp_ms\": " << artifacts.pose_lookup.sync_timestamp_ms << ",\n";
    output << "  \"pose_matched\": " << (artifacts.pose_lookup.frame ? "true" : "false") << ",\n";
    output << "  \"pose_timestamp_ms\": " << artifacts.pose_lookup.pose_timestamp_ms << ",\n";
    output << "  \"pose_timestamp_delta_ms\": " << artifacts.pose_lookup.timestamp_delta_ms << ",\n";
    output << "  \"reference_frame_seq\": " << artifacts.pose3d.frame_seq << ",\n";
    output << "  \"people\": [\n";
    for (size_t person_index = 0; person_index < artifacts.pose3d.people.size(); ++person_index) {
        const auto& person = artifacts.pose3d.people[person_index];
        output << "    {\n";
        output << "      \"person_id\": " << person.person_id << ",\n";
        output << "      \"score\": " << person.score << ",\n";
        output << "      \"mean_confidence\": " << person.MeanConfidence() << ",\n";
        output << "      \"joints\": [\n";
        for (size_t joint_index = 0; joint_index < person.joints.size(); ++joint_index) {
            const auto& joint = person.joints[joint_index];
            output << "        {\"index\": " << joint_index
                   << ", \"valid\": " << (joint.IsValid() ? "true" : "false")
                   << ", \"xyz\": ";
            WriteVec3(output, joint.xyz);
            output << ", \"quaternion\": ";
            WriteVec4(output, joint.quaternion);
            output << ", \"confidence\": " << joint.confidence << "}";
            if (joint_index + 1u < person.joints.size()) {
                output << ",";
            }
            output << "\n";
        }
        output << "      ],\n";
        output << "      \"smpl_valid\": " << (person.smpl_valid ? "true" : "false") << ",\n";
        output << "      \"smpl_pose\": ";
        WriteFloatArray(output, person.smpl_pose, kSmplPoseParamCount);
        output << ",\n";
        output << "      \"smpl_shape\": ";
        WriteFloatArray(output, person.smpl_shape, kSmplShapeParamCount);
        output << "\n";
        output << "    }";
        if (person_index + 1u < artifacts.pose3d.people.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ],\n";
    output << "  \"views\": [\n";
    for (size_t view_index = 0; view_index < artifacts.synced_frames.views.size(); ++view_index) {
        const auto& view = artifacts.synced_frames.views[view_index];
        output << "    {\"camera_id\": \"" << view.camera_id << "\""
               << ", \"video_frame_index\": " << view.video_frame_index
               << ", \"video_timestamp_ms\": " << view.video_timestamp_ms << "}";
        if (view_index + 1u < artifacts.synced_frames.views.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ]\n";
    output << "}\n";
    return true;
}

bool DatasetExporter::AppendManifestLine(const ExportFrameArtifacts& artifacts,
                                         const std::filesystem::path& pose_path) const {
    const auto manifest_path = options_.output_dir / "frames.jsonl";
    std::ofstream output(manifest_path, std::ios::app);
    if (!output.is_open()) {
        std::cerr << "DatasetExporter: failed to append " << manifest_path.string() << std::endl;
        return false;
    }

    output.setf(std::ios::fixed);
    output << std::setprecision(6);

    if (HasTrainingSamples(artifacts)) {
        for (const auto& sample : artifacts.training_samples) {
            const auto crop_path = std::filesystem::absolute(
                MakeTrainingCropPath(artifacts, sample)).lexically_normal();
            output << "{\"frame\":" << artifacts.synced_frames.sync_index
                   << ",\"sync_index\":" << artifacts.synced_frames.sync_index
                   << ",\"camera_id\":\"" << JsonEscape(sample.camera_id) << "\""
                   << ",\"person_id\":" << sample.person_id
                   << ",\"video_frame_index\":" << sample.video_frame_index
                   << ",\"image\":\"\""
                   << ",\"crop\":\"" << JsonEscape(crop_path.generic_string()) << "\""
                   << ",\"input\":\"\""
                   << ",\"overlay\":\"\""
                   << ",\"img_w\":" << sample.img_w
                   << ",\"img_h\":" << sample.img_h
                   << ",\"crop_cx\":" << sample.crop_cx
                   << ",\"crop_cy\":" << sample.crop_cy
                   << ",\"crop_size\":" << sample.crop_size
                   << ",\"crop_x0\":" << sample.crop_x0
                   << ",\"crop_y0\":" << sample.crop_y0
                   << ",\"crop_w\":" << sample.crop_w
                   << ",\"crop_h\":" << sample.crop_h
                   << ",\"focal_length\":" << sample.focal_length
                   << ",\"y_sign\":" << sample.y_sign
                   << ",\"pose\":";
            WriteFloatArray(output, sample.pose, kSmplPoseParamCount);
            output << ",\"betas\":";
            WriteFloatArray(output, sample.betas, kSmplShapeParamCount);
            output << ",\"cam\":";
            WriteFloatArray(output, sample.cam, 3u);
            if (!pose_path.empty()) {
                const auto normalized_pose = std::filesystem::absolute(pose_path).lexically_normal();
                output << ",\"pose_json\":\"" << JsonEscape(normalized_pose.generic_string()) << "\"";
            }
            output << "}\n";
        }
        return true;
    }

    output << "{\"sync_index\":" << artifacts.synced_frames.sync_index
           << ",\"timestamp_ms\":" << artifacts.synced_frames.sync_timestamp_ms
           << ",\"pose_matched\":" << (artifacts.pose_lookup.frame ? "true" : "false")
           << ",\"pose_frame_seq\":" << artifacts.pose3d.frame_seq
           << ",\"pose_timestamp_ms\":" << artifacts.pose_lookup.pose_timestamp_ms
           << ",\"pose_timestamp_delta_ms\":" << artifacts.pose_lookup.timestamp_delta_ms;
    if (!pose_path.empty()) {
        output << ",\"pose\":\"" << pose_path.generic_string() << "\"";
    }
    output << ",\"views\":[";
    for (size_t index = 0; index < artifacts.synced_frames.views.size(); ++index) {
        const auto& view = artifacts.synced_frames.views[index];
        output << "{\"camera_id\":\"" << view.camera_id << "\""
               << ",\"frame_index\":" << view.video_frame_index;
        if (options_.save_images) {
            output << ",\"image\":\"images/" << view.camera_id << "/"
                   << MakeFrameStem(artifacts.synced_frames.sync_index) << ".png\"";
        }
        if (options_.save_masks) {
            output << ",\"mask\":\"masks/" << view.camera_id << "/"
                   << MakeFrameStem(artifacts.synced_frames.sync_index)
                   << ".png\"";
        }
        output << "}";
        if (index + 1u < artifacts.synced_frames.views.size()) {
            output << ",";
        }
    }
    output << "]}\n";
    return true;
}

bool DatasetExporter::HasTrainingSamples(const ExportFrameArtifacts& artifacts) const {
    return !artifacts.training_samples.empty();
}

std::filesystem::path DatasetExporter::MakeTrainingCropPath(
    const ExportFrameArtifacts& artifacts,
    const ExportTrainingSample& sample) const {
    return options_.output_dir / "crops" / sample.camera_id /
           ("crop_" + MakeTrainingStem(artifacts, sample) + ".png");
}

std::filesystem::path DatasetExporter::MakeTrainingMattePath(
    const ExportFrameArtifacts& artifacts,
    const ExportTrainingSample& sample) const {
    return options_.output_dir / "mattes" / sample.camera_id /
           ("matte_" + MakeTrainingStem(artifacts, sample) + ".png");
}

std::string DatasetExporter::MakeTrainingStem(const ExportFrameArtifacts& artifacts,
                                              const ExportTrainingSample& sample) const {
    std::ostringstream stem;
    stem << MakeFrameStem(artifacts.synced_frames.sync_index)
         << "_p" << std::setw(3) << std::setfill('0') << std::max(sample.person_index, 0);
    return stem.str();
}

std::string DatasetExporter::MakeFrameStem(int sync_index) const {
    std::ostringstream stem;
    stem << "frame_" << std::setw(6) << std::setfill('0') << std::max(sync_index, 0);
    return stem.str();
}

}  // namespace dataset_prep
