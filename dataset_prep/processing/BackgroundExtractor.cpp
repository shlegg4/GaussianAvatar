#include "dataset_prep/processing/BackgroundExtractor.h"

#include <iostream>

#include <opencv2/imgproc.hpp>

#include "dataset_prep/utils/ModNetMatte.h"

namespace dataset_prep {

class BackgroundExtractor::Impl {
public:
    explicit Impl(const Options& options)
        : matte(ModNetMatteOptions{options.input_size, options.use_cuda, true}) {}

    ModNetMatte matte;
};

BackgroundExtractor::BackgroundExtractor(const Options& options)
    : options_(options) {}

BackgroundExtractor::~BackgroundExtractor() = default;

bool BackgroundExtractor::Initialize() {
    if (options_.model_path.empty()) {
        std::cerr << "BackgroundExtractor: MODNet model path is required." << std::endl;
        return false;
    }

    impl_ = std::make_unique<Impl>(options_);
    if (!impl_->matte.Load(options_.model_path)) {
        std::cerr << "BackgroundExtractor: failed to load MODNet model " << options_.model_path << std::endl;
        return false;
    }
    ready_ = true;
    return true;
}

bool BackgroundExtractor::Process(const SyncedFrameCollection& synced_frames,
                                  std::vector<BackgroundExtractorResult>* out_results) {
    if (!ready_ || out_results == nullptr) {
        return false;
    }

    out_results->clear();
    out_results->reserve(synced_frames.views.size());

    for (const auto& view : synced_frames.views) {
        BackgroundExtractorResult result;
        result.camera_id = view.camera_id;
        result.source_camera_index = view.source_camera_index;

        cv::Mat matte_f32;
        if (!impl_->matte.ComputeMatte(view.image, &matte_f32)) {
            return false;
        }
        if (options_.binary_threshold >= 0.0f) {
            cv::threshold(matte_f32, matte_f32, options_.binary_threshold, 1.0, cv::THRESH_BINARY);
        }
        matte_f32.convertTo(result.matte, CV_8U, 255.0);
        if (result.matte.empty()) {
            return false;
        }
        result.foreground = impl_->matte.ApplyMatte(view.image, matte_f32);

        out_results->push_back(std::move(result));
    }

    return true;
}

bool BackgroundExtractor::ProcessImage(const cv::Mat& image, cv::Mat* out_matte) {
    if (!ready_ || out_matte == nullptr || image.empty()) {
        return false;
    }

    cv::Mat matte_f32;
    if (!impl_->matte.ComputeMatte(image, &matte_f32)) {
        return false;
    }
    if (options_.binary_threshold >= 0.0f) {
        cv::threshold(matte_f32, matte_f32, options_.binary_threshold, 1.0, cv::THRESH_BINARY);
    }
    matte_f32.convertTo(*out_matte, CV_8U, 255.0);
    return !out_matte->empty();
}

}  // namespace dataset_prep
