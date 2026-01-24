#include <iostream>
#include <string>

#include "utils/HmrInferenceUtils.h"

namespace {

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  hmr_inference <model.onnx> <video.mp4|image.png> [options]\n\n"
        << "Options:\n"
        << "  --output <dir>                 Save outputs to directory\n"
        << "  --rtmpose <model.onnx>          RTMPose model path\n"
        << "  --yolo <model.onnx>             YOLO model path\n"
        << "  --modnet <model.onnx>           MODNet matting model path\n"
        << "  --modnet-cuda                   Enable CUDA for MODNet (if available)\n"
        << "  --modnet-input <int>            MODNet input size if model is dynamic\n"
        << "  --focal-scale <float>           Focal length scale (default 1.2)\n"
        << "  --frame-stride <int>            Process every Nth frame (default 1)\n"
        << "  --smplify-requires-yolo         Only run Smplify when YOLO detects a person\n"
        << "  --save-outputs                 Enable output saving (requires --output)\n"
        << "  --no-save                      Disable output saving\n"
        << "  --help                         Show this help\n\n"
        << "Legacy positional args still work: [output_dir] [rtmpose.onnx] [focal_scale] [yolo.onnx]\n";
}

bool LooksLikeNumber(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtof(s.c_str(), &end);
    return end && *end == '\0';
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        PrintUsage();
        return -1;
    }

    std::string model_path_str = argv[1];
    std::string video_path = argv[2];

    HmrOutputOptions options;
    bool focal_set = false;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage();
            return 0;
        }
        if (arg == "--output" && i + 1 < argc) {
            options.output_dir = argv[++i];
            options.save_outputs = true;
            continue;
        }
        if (arg == "--rtmpose" && i + 1 < argc) {
            options.use_rtmpose = true;
            options.rtmpose_model_path = argv[++i];
            continue;
        }
        if (arg == "--yolo" && i + 1 < argc) {
            options.use_yolo = true;
            options.yolo_model_path = argv[++i];
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
        if (arg == "--modnet-input" && i + 1 < argc) {
            try {
                options.modnet_input_size = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid modnet-input value." << std::endl;
                return -1;
            }
            continue;
        }
        if (arg == "--focal-scale" && i + 1 < argc) {
            try {
                options.focal_length_scale = std::stof(argv[++i]);
                focal_set = true;
            } catch (const std::exception&) {
                std::cerr << "Invalid focal-scale value." << std::endl;
                return -1;
            }
            continue;
        }
        if (arg == "--frame-stride" && i + 1 < argc) {
            try {
                options.frame_stride = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid frame-stride value." << std::endl;
                return -1;
            }
            continue;
        }
        if (arg == "--smplify-requires-yolo") {
            options.smplify_requires_yolo = true;
            continue;
        }
        if (arg == "--save-outputs") {
            options.save_outputs = true;
            continue;
        }
        if (arg == "--no-save") {
            options.save_outputs = false;
            continue;
        }

        // Legacy positional fallback.
        if (arg.size() >= 5 && arg.substr(arg.size() - 5) == ".onnx") {
            if (!options.use_rtmpose) {
                options.use_rtmpose = true;
                options.rtmpose_model_path = arg;
            } else if (!options.use_yolo) {
                options.use_yolo = true;
                options.yolo_model_path = arg;
            }
            continue;
        }
        if (!focal_set && LooksLikeNumber(arg)) {
            options.focal_length_scale = std::stof(arg);
            focal_set = true;
            continue;
        }
        if (options.output_dir.empty()) {
            options.output_dir = arg;
            options.save_outputs = true;
            continue;
        }
    }

    if (options.frame_stride < 1) {
        std::cerr << "frame-stride must be >= 1. Using 1." << std::endl;
        options.frame_stride = 1;
    }
    if (options.save_outputs && options.output_dir.empty()) {
        std::cerr << "Output saving requested but no output dir set." << std::endl;
        return -1;
    }

    ResultsDict results;
    ResultsDict* results_ptr = options.save_outputs ? &results : nullptr;
    if (!RunHmrInferenceOnVideo(model_path_str, video_path, options, results_ptr)) {
        return -1;
    }

    return 0;
}
