#include <iostream>
#include <string>
#include <filesystem>

#include "utils/irisOptimizationUtils.h"

namespace {

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  hmr_folder_optimizer <data_folder> [options]\n\n"
        << "Description:\n"
        << "  Processing pipeline that takes a folder containing:\n"
        << "    - poses.jsonl (3D keypoints)\n"
        << "    - intrinsics_camX.json\n"
        << "    - extrinsics.json\n"
        << "    - Videos (cam0.mp4, cam1.mp4, ...)\n"
        << "  It fits the SMPL mesh to the 3D keypoints and generates a dataset with matting.\n\n"
        << "Options:\n"
        << "  --output <dir>                 Save outputs to directory (Required)\n"
        << "  --smpl <path.pt>                SMPL model path (default: smpl_data.pt)\n"
        << "  --modnet <model.onnx>           MODNet matting model path\n"
        << "  --modnet-cuda                   Enable CUDA for MODNet\n"
        << "  --frame-stride <int>            Process every Nth frame (default 1)\n"
        << "  --vis                          Save visualization overlays (default: true)\n"
        << "  --no-vis                       Disable visualization overlays\n"
        << "  --help                         Show this help\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage();
        return -1;
    }

    std::string folder_path = argv[1];
    FolderOptimizerOptions options;
    options.input_folder = folder_path;

    // Default defaults
    options.save_visualization = true;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage();
            return 0;
        }
        if (arg == "--output" && i + 1 < argc) {
            options.output_dir = argv[++i];
            continue;
        }
        if (arg == "--smpl" && i + 1 < argc) {
            options.smpl_model_path = argv[++i];
            continue;
        }
        if (arg == "--modnet" && i + 1 < argc) {
            options.use_modnet = true;
            options.modnet_model_path = argv[++i];
            continue;
        }
        if (arg == "--modnet-cuda") {
            options.modnet_use_cuda = true;
            continue;
        }
        if (arg == "--frame-stride" && i + 1 < argc) {
            try {
                options.frame_stride = std::stoi(argv[++i]);
            } catch (...) {}
            continue;
        }
        if (arg == "--vis") {
            options.save_visualization = true;
            continue;
        }
        if (arg == "--no-vis") {
            options.save_visualization = false;
            continue;
        }
    }

    if (options.output_dir.empty()) {
        std::cerr << "Error: --output <dir> is required." << std::endl;
        return -1;
    }

    if (!std::filesystem::exists(folder_path)) {
        std::cerr << "Error: Input folder does not exist: " << folder_path << std::endl;
        return -1;
    }

    if (!RunFolderOptimization(options)) {
        return -1;
    }

    return 0;
}