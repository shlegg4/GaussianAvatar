#include <iostream>
#include <string>

#include "utils/HmrInferenceUtils.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cout << "Usage: ./hmr_inference <model.onnx> <video.mp4> [output_dir]" << std::endl;
        return -1;
    }

    std::string model_path_str = argv[1];
    std::string video_path = argv[2];

    HmrOutputOptions options;
    if (argc == 4 && std::string(argv[3]).size() > 0) {
        options.output_dir = argv[3];
        options.save_outputs = true;
    }

    ResultsDict results;
    ResultsDict* results_ptr = options.save_outputs ? &results : nullptr;
    if (!RunHmrInferenceOnVideo(model_path_str, video_path, options, results_ptr)) {
        return -1;
    }

    return 0;
}
