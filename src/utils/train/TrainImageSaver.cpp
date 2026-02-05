#include "utils/train/TrainImageSaver.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/image/TensorCvUtils.h"

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
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (!cached[i].valid)
        {
            continue;
        }
        const auto &sample = samples[i];
        const auto &cached_entry = cached[i];

        torch::Tensor render = render_fn(i, sample, cached_entry);
        if (!render.defined())
        {
            continue;
        }
        cv::Mat render_bgr = TensorToBgr(render);
        if (render_bgr.empty())
        {
            continue;
        }

        if (cached_entry.crop_bgr.empty())
        {
            continue;
        }
        cv::Mat target_bgr = cached_entry.crop_bgr;
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
        }
    }

    return saved;
}
