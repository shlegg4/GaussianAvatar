#include "utils/image/MaskUtils.h"

#include <opencv2/imgproc.hpp>

#include "utils/image/TensorCvUtils.h"

bool IsMaskCoverageValid(const torch::Tensor &render, const cv::Mat &matte, float max_outside_ratio)
{
    if (!render.defined() || matte.empty())
        return false;
    const int H = static_cast<int>(render.size(1));
    const int W = static_cast<int>(render.size(2));

    cv::Mat matte_gray;
    if (matte.channels() == 4)
    {
        std::vector<cv::Mat> channels;
        cv::split(matte, channels);
        matte_gray = channels[3];
    }
    else if (matte.channels() == 3)
    {
        cv::cvtColor(matte, matte_gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        matte_gray = matte;
    }

    if (matte_gray.size() != cv::Size(W, H))
    {
        cv::resize(matte_gray, matte_gray, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat matte_mask;
    cv::threshold(matte_gray, matte_mask, 127, 255, cv::THRESH_BINARY);

    cv::Mat render_bgr = TensorToBgr(render);
    if (render_bgr.empty())
        return false;
    cv::Mat render_gray;
    cv::cvtColor(render_bgr, render_gray, cv::COLOR_BGR2GRAY);
    cv::Mat render_mask;
    cv::threshold(render_gray, render_mask, 3, 255, cv::THRESH_BINARY);

    const int rendered_pixels = cv::countNonZero(render_mask);
    if (rendered_pixels == 0)
        return false;

    cv::Mat outside_mask;
    cv::bitwise_not(matte_mask, outside_mask);
    cv::Mat outside_render;
    cv::bitwise_and(render_mask, outside_mask, outside_render);
    const int outside_pixels = cv::countNonZero(outside_render);
    const float outside_ratio = static_cast<float>(outside_pixels) / static_cast<float>(rendered_pixels);
    return outside_ratio <= max_outside_ratio;
}

bool IsMaskCoverageValidTensor(const torch::Tensor &render, const torch::Tensor &matte_mask,
                               float max_outside_ratio, float render_threshold)
{
    if (!render.defined() || !matte_mask.defined())
        return false;
    if (render.dim() != 3 || render.size(0) != 3)
        return false;
    if (matte_mask.dim() != 3 || matte_mask.size(0) != 1)
        return false;
    if (render.size(1) != matte_mask.size(1) || render.size(2) != matte_mask.size(2))
        return false;

    auto render_gray = render.mean(0, true);
    auto render_mask = render_gray > render_threshold;
    auto matte_bin = matte_mask > 0.5f;
    auto rendered_pixels = render_mask.sum().item<float>();
    if (rendered_pixels <= 0.0f)
        return false;
    auto outside = render_mask.logical_and(matte_bin.logical_not());
    auto outside_pixels = outside.sum().item<float>();
    const float outside_ratio = outside_pixels / rendered_pixels;
    return outside_ratio <= max_outside_ratio;
}
