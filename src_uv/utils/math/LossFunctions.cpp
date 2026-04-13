#include "utils/math/LossFunctions.h"

std::pair<torch::Tensor, torch::Tensor> create_separable_windows(int window_size, torch::Device device)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto t = torch::arange(window_size, options) - (window_size / 2.0f);
    auto gauss = torch::exp(-(t * t) / (2.0f * 1.5f * 1.5f));
    gauss = gauss / gauss.sum();

    auto window_v = gauss.view({1, 1, window_size, 1}).contiguous();
    auto window_h = gauss.view({1, 1, 1, window_size}).contiguous();
    return {window_v, window_h};
}

torch::Tensor ssim_fast(const torch::Tensor &img1,
                        const torch::Tensor &img2,
                        const torch::Tensor &window_v,
                        const torch::Tensor &window_h,
                        const torch::Tensor &matte_mask,
                        int window_size)
{
    auto inp1 = (img1.dim() == 3) ? img1.unsqueeze(0) : img1;
    auto inp2 = (img2.dim() == 3) ? img2.unsqueeze(0) : img2;
    auto mask = (matte_mask.dim() == 3) ? matte_mask.unsqueeze(0) : matte_mask;

    const int64_t B = inp1.size(0);
    const int64_t C = inp1.size(1);
    const int64_t H = inp1.size(2);
    const int64_t W = inp1.size(3);

    if (mask.dim() != 4 || mask.size(0) != B || mask.size(2) != H || mask.size(3) != W)
    {
        return torch::zeros({B}, inp1.options());
    }
    if (mask.size(1) != 1)
    {
        if (mask.size(1) == C)
        {
            mask = mask.mean(1, true);
        }
        else
        {
            mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice(), torch::indexing::Slice()});
        }
    }

    inp1 = inp1.reshape({B * C, 1, H, W});
    inp2 = inp2.reshape({B * C, 1, H, W});

    const int64_t pad = window_size / 2;
    const std::vector<int64_t> stride = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const std::vector<int64_t> pad_v = {pad, 0};
    const std::vector<int64_t> pad_h = {0, pad};
    auto blur = [&](const torch::Tensor &x)
    {
        auto out = torch::conv2d(x, window_v, c10::nullopt, stride, pad_v, dilation, 1);
        out = torch::conv2d(out, window_h, c10::nullopt, stride, pad_h, dilation, 1);
        return out;
    };

    auto mu1 = blur(inp1);
    auto mu2 = blur(inp2);

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = blur(inp1 * inp1) - mu1_sq;
    auto sigma2_sq = blur(inp2 * inp2) - mu2_sq;
    auto sigma12 = blur(inp1 * inp2) - mu1_mu2;

    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    auto ssim_map_folded = ((2.0f * mu1_mu2 + C1) * (2.0f * sigma12 + C2)) /
                           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));
    auto ssim_map = ssim_map_folded.reshape({B, C, H, W});
    auto masked_ssim = ssim_map * mask;
    auto valid_pixel_count = torch::clamp_min(mask.sum({1, 2, 3}) * static_cast<float>(C), 1e-6f);

    return masked_ssim.sum({1, 2, 3}) / valid_pixel_count;
}

torch::Tensor Downsample(const torch::Tensor &img)
{
    const bool squeeze_batch = (img.dim() == 3);
    auto inp = squeeze_batch ? img.unsqueeze(0) : img;
    if (inp.size(-2) < 2 || inp.size(-1) < 2)
    {
        return img;
    }
    auto out = torch::nn::functional::avg_pool2d(
        inp,
        torch::nn::functional::AvgPool2dFuncOptions(2).stride(2));
    return squeeze_batch ? out.squeeze(0) : out;
}
