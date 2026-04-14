#include "dataset_prep/utils/CommandLine.h"

#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace dataset_prep {
namespace {

bool ParseBoolFlagValue(const std::string& value, bool* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON") {
        *out_value = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off" || value == "OFF") {
        *out_value = false;
        return true;
    }
    return false;
}

bool ParseFeedSpec(const std::string& feed_spec, VideoSourceConfig* out_source) {
    if (out_source == nullptr) {
        return false;
    }

    const size_t delimiter = feed_spec.find('=');
    if (delimiter == std::string::npos || delimiter == 0u || delimiter + 1u >= feed_spec.size()) {
        return false;
    }

    out_source->camera_id = feed_spec.substr(0, delimiter);
    out_source->video_path = feed_spec.substr(delimiter + 1u);
    return !out_source->camera_id.empty() && !out_source->video_path.empty();
}

}  // namespace

void PrintDatasetPrepUsage(std::ostream& out) {
    out << "Usage:\n"
    << "  dataset_prep <video.mp4> <output_dir> <camera_id> [max_frames] [frame_stride] [start_frame_index]\n"
        << "  dataset_prep --output-dir <dir> --target-camera <camera_id>\n"
        << "               --feed <camera_id=video.mp4> [--feed <camera_id=video.mp4> ...]\n"
    << "               [--max-frames N] [--frame-stride N] [--start-frame-index N] [--sync-tolerance-ms MS]\n"
        << "               [--smooth-alpha A]\n"
        << "               [--show-rtmpose-overlay 0|1] [--show-smpl-joints-overlay 0|1] [--show-smpl-verts-overlay 0|1]\n";
}

bool ParseDatasetPrepCommandLine(int argc, char* argv[], DatasetPrepOptions* out_options) {
    if (out_options == nullptr) {
        return false;
    }

    DatasetPrepOptions options;
    if (argc >= 4 && argc <= 7 && std::string(argv[1]).rfind("--", 0) != 0) {
        options.output_dir = argv[2];
        options.target_camera_id = argv[3];
        options.max_frames = (argc > 4) ? std::stoi(argv[4]) : -1;
        options.frame_stride = (argc > 5) ? std::stoi(argv[5]) : 1;
        options.start_frame_index = (argc > 6) ? std::stoi(argv[6]) : 0;

        VideoSourceConfig source;
        source.camera_id = options.target_camera_id;
        source.video_path = argv[1];
        options.sources.push_back(std::move(source));
    } else {
        for (int arg_index = 1; arg_index < argc; ++arg_index) {
            const std::string arg = argv[arg_index];
            if (arg == "--output-dir") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --output-dir.\n";
                    return false;
                }
                options.output_dir = argv[++arg_index];
            } else if (arg == "--target-camera") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --target-camera.\n";
                    return false;
                }
                options.target_camera_id = argv[++arg_index];
            } else if (arg == "--feed") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --feed.\n";
                    return false;
                }

                VideoSourceConfig source;
                if (!ParseFeedSpec(argv[++arg_index], &source)) {
                    std::cerr << "Invalid --feed value. Expected <camera_id=video_path>.\n";
                    return false;
                }
                options.sources.push_back(std::move(source));
            } else if (arg == "--max-frames") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --max-frames.\n";
                    return false;
                }
                options.max_frames = std::stoi(argv[++arg_index]);
            } else if (arg == "--frame-stride") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --frame-stride.\n";
                    return false;
                }
                options.frame_stride = std::stoi(argv[++arg_index]);
            } else if (arg == "--start-frame-index") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --start-frame-index.\n";
                    return false;
                }
                options.start_frame_index = std::stoi(argv[++arg_index]);
            } else if (arg == "--sync-tolerance-ms") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --sync-tolerance-ms.\n";
                    return false;
                }
                options.sync_tolerance_ms = std::stod(argv[++arg_index]);
            } else if (arg == "--smooth-alpha") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --smooth-alpha.\n";
                    return false;
                }
                options.temporal_smooth_alpha = std::stof(argv[++arg_index]);
            } else if (arg == "--show-rtmpose-overlay") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --show-rtmpose-overlay.\n";
                    return false;
                }
                if (!ParseBoolFlagValue(argv[++arg_index], &options.show_rtmpose_points_overlay)) {
                    std::cerr << "Invalid value for --show-rtmpose-overlay (use 0/1, true/false, on/off).\n";
                    return false;
                }
            } else if (arg == "--show-smpl-joints-overlay") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --show-smpl-joints-overlay.\n";
                    return false;
                }
                if (!ParseBoolFlagValue(argv[++arg_index], &options.show_smpl_joints_overlay)) {
                    std::cerr << "Invalid value for --show-smpl-joints-overlay (use 0/1, true/false, on/off).\n";
                    return false;
                }
            } else if (arg == "--show-smpl-verts-overlay") {
                if (arg_index + 1 >= argc) {
                    std::cerr << "Missing value after --show-smpl-verts-overlay.\n";
                    return false;
                }
                if (!ParseBoolFlagValue(argv[++arg_index], &options.show_smpl_verts_overlay)) {
                    std::cerr << "Invalid value for --show-smpl-verts-overlay (use 0/1, true/false, on/off).\n";
                    return false;
                }
            } else if (!arg.empty() && arg[0] != '-') {
                VideoSourceConfig source;
                if (!ParseFeedSpec(arg, &source)) {
                    std::cerr << "Unknown positional argument: " << arg << "\n";
                    return false;
                }
                options.sources.push_back(std::move(source));
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return false;
            }
        }
    }

    if (options.output_dir.empty()) {
        std::cerr << "output_dir is required.\n";
        return false;
    }
    if (options.frame_stride <= 0) {
        std::cerr << "frame_stride must be >= 1.\n";
        return false;
    }
    if (options.start_frame_index < 0) {
        std::cerr << "start-frame-index must be >= 0.\n";
        return false;
    }
    if (options.temporal_smooth_alpha < 0.0f || options.temporal_smooth_alpha > 1.0f) {
        std::cerr << "smooth-alpha must be in [0, 1].\n";
        return false;
    }
    if (options.sources.empty()) {
        std::cerr << "At least one feed is required.\n";
        return false;
    }
    if (options.target_camera_id.empty()) {
        options.target_camera_id = options.sources.front().camera_id;
    }

    bool found_target_camera = false;
    for (size_t source_index = 0; source_index < options.sources.size(); ++source_index) {
        auto& source = options.sources[source_index];
        source.source_camera_index = static_cast<int>(source_index);
        source.frame_stride = options.frame_stride;
        if (source.camera_id == options.target_camera_id) {
            found_target_camera = true;
        }
    }

    if (!found_target_camera) {
        std::cerr << "target_camera_id must match one of the provided feeds.\n";
        return false;
    }

    *out_options = std::move(options);
    return true;
}

bool ResolveCalibrationDirectory(fs::path* out_dir) {
    if (out_dir == nullptr) {
        return false;
    }

    std::error_code ec;
    fs::path cursor = fs::current_path(ec);
    if (ec) {
        return false;
    }

    for (int depth = 0; depth < 6; ++depth) {
        const fs::path candidate = cursor / "data";
        if (fs::exists(candidate / "extrinsics.json", ec) && !ec) {
            *out_dir = fs::absolute(candidate).lexically_normal();
            return true;
        }

        if (!cursor.has_parent_path()) {
            break;
        }
        const fs::path parent = cursor.parent_path();
        if (parent == cursor) {
            break;
        }
        cursor = parent;
    }

    return false;
}

}  // namespace dataset_prep
