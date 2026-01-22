#include <iostream>
#include <string>

#include "utils/HmrInferenceUtils.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 6) {
        std::cout << "Usage: ./hmr_inference <model.onnx> <video.mp4|image.png> [output_dir] [yolo.onnx] [focal_scale]" << std::endl;
        return -1;
    }

    std::string model_path_str = argv[1];
    std::string video_path = argv[2];

    HmrOutputOptions options;
    if (argc == 4 && std::string(argv[3]).size() > 0) {
        const std::string arg3 = argv[3];
        if (arg3.size() >= 5 && arg3.substr(arg3.size() - 5) == ".onnx") {
            options.use_yolo = true;
            options.yolo_model_path = arg3;
        } else {
            options.output_dir = arg3;
            options.save_outputs = true;
        }
    }
    if (argc >= 5 && std::string(argv[4]).size() > 0) {
        options.use_yolo = true;
        options.yolo_model_path = argv[4];
        options.output_dir = argv[3];
        options.save_outputs = true;
    }
    if (argc == 6 && std::string(argv[5]).size() > 0) {
        try {
            options.focal_length_scale = std::stof(argv[5]);
        } catch (const std::exception&) {
            std::cerr << "Invalid focal_scale: " << argv[5] << std::endl;
            return -1;
        }
    }

    ResultsDict results;
    ResultsDict* results_ptr = options.save_outputs ? &results : nullptr;
    if (!RunHmrInferenceOnVideo(model_path_str, video_path, options, results_ptr)) {
        return -1;
    }

    return 0;
}
