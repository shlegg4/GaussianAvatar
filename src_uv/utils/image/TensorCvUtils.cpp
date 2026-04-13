#include "utils/image/TensorCvUtils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat TensorToBgr(const torch::Tensor &image)
{
    if (!image.defined())
        return cv::Mat();
    auto img = image.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kU8).cpu();
    if (img.dim() != 3 || img.size(0) != 3)
        return cv::Mat();
    img = img.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

torch::Tensor LoadImageTensor(const cv::Mat &bgr, torch::Device device)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(
                      float_img.data,
                      {float_img.rows, float_img.cols, 3},
                      torch::kFloat)
                      .clone();
    tensor = tensor.permute({2, 0, 1}).contiguous().to(device);
    return tensor;
}

torch::Tensor LoadMatteMaskTensor(const cv::Mat &matte, int target_w, int target_h, torch::Device device)
{
    if (matte.empty())
        return torch::Tensor();

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

    if (matte_gray.size() != cv::Size(target_w, target_h))
    {
        cv::resize(matte_gray, matte_gray, cv::Size(target_w, target_h), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat float_mask;
    matte_gray.convertTo(float_mask, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(float_mask.data, {float_mask.rows, float_mask.cols, 1}, torch::kFloat).clone();
    tensor = tensor.permute({2, 0, 1}).contiguous().to(device);
    return tensor;
}

bool SaveImageTensorPng(const std::string &path, const torch::Tensor &image)
{
    if (!image.defined())
        return false;
    auto img = image.detach().clamp(0.0, 1.0).mul(255.0).to(torch::kU8).cpu();
    if (img.dim() != 3 || img.size(0) != 3)
        return false;
    img = img.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return cv::imwrite(path, bgr);
}
