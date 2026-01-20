#pragma once
#include <torch/torch.h>
#include <tuple>
 
// ==========================================
//    Forward Declarations
// ==========================================

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_alpha,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& alpha,
    const bool debug);

// ==========================================
//    GaussianRasterizer Class
// ==========================================

class GaussianRasterizer : public torch::autograd::Function<GaussianRasterizer> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor& means3D,
        const torch::Tensor& colors,
        const torch::Tensor& opacity,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const float scale_modifier,
        const torch::Tensor& viewmatrix,
        const torch::Tensor& projmatrix,
        const float tan_fovx,
        const float tan_fovy,
        const int image_height,
        const int image_width,
        const torch::Tensor& sh,
        const int degree,
        const torch::Tensor& campos,
        const bool debug) 
    {
        // Ideally, you should also add CHECK_INPUT(means3D), CHECK_INPUT(colors), etc. here
        // to catch errors before the forward pass runs.
        
        auto float_opts = means3D.options().dtype(torch::kFloat32);
        auto background = torch::zeros({3, image_height, image_width}, float_opts);
        auto cov3D_precomp = torch::zeros({0}, float_opts); 

        auto result = RasterizeGaussiansCUDA(
            background,
            means3D, colors, opacity, scales, rotations,
            scale_modifier, cov3D_precomp,
            viewmatrix, projmatrix,
            tan_fovx, tan_fovy,
            image_height, image_width,
            sh, degree, campos,
            false, debug
        );

        auto num_rendered = std::get<0>(result);
        auto out_color = std::get<1>(result);
        auto out_depth = std::get<2>(result);
        auto out_alpha = std::get<3>(result);
        auto radii = std::get<4>(result);
        auto geomBuffer = std::get<5>(result);
        auto binningBuffer = std::get<6>(result);
        auto imageBuffer = std::get<7>(result);

        ctx->save_for_backward({means3D, radii, colors, scales, rotations, cov3D_precomp, viewmatrix, projmatrix, sh, geomBuffer, binningBuffer, imageBuffer, out_depth, out_alpha});
        
        ctx->saved_data["scale_modifier"] = scale_modifier;
        ctx->saved_data["tan_fovx"] = tan_fovx;
        ctx->saved_data["tan_fovy"] = tan_fovy;
        ctx->saved_data["degree"] = degree;
        ctx->saved_data["campos"] = campos;
        ctx->saved_data["debug"] = debug;
        ctx->saved_data["R"] = num_rendered;

        return out_color;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto dL_dout_color = grad_outputs[0];

        // --- PROTECTION 1: Early exit if incoming gradient is undefined ---
        if (!dL_dout_color.defined()) {
            return torch::autograd::variable_list(16, torch::Tensor());
        }

        // --- PROTECTION 2: Ensure contiguous memory ---
        dL_dout_color = dL_dout_color.contiguous();

        auto saved = ctx->get_saved_variables();
        
        auto means3D = saved[0];
        auto radii = saved[1];
        auto colors = saved[2];
        auto scales = saved[3];
        auto rotations = saved[4];
        auto cov3D_precomp = saved[5];
        auto viewmatrix = saved[6];
        auto projmatrix = saved[7];
        auto sh = saved[8];
        auto geomBuffer = saved[9];
        auto binningBuffer = saved[10];
        auto imageBuffer = saved[11];
        auto out_depth = saved[12];
        auto alpha = saved[13];

        float scale_modifier = ctx->saved_data["scale_modifier"].toDouble();
        float tan_fovx = ctx->saved_data["tan_fovx"].toDouble();
        float tan_fovy = ctx->saved_data["tan_fovy"].toDouble();
        int degree = ctx->saved_data["degree"].toInt();
        auto campos = ctx->saved_data["campos"].toTensor();
        bool debug = ctx->saved_data["debug"].toBool();
        int R = ctx->saved_data["R"].toInt();

        auto dL_dout_depth = torch::zeros_like(out_depth);
        auto dL_dout_alpha = torch::zeros_like(alpha);
 
        int H = out_depth.size(1);
        int W = out_depth.size(2);
         
        auto background = torch::zeros({3, H, W}, means3D.options().dtype(torch::kFloat32));

        auto gradients = RasterizeGaussiansBackwardCUDA( 
            background,
            means3D, 
            radii, 
            colors, 
            scales, 
            rotations,
            scale_modifier, 
            cov3D_precomp,
            viewmatrix, 
            projmatrix,
            tan_fovx, 
            tan_fovy,
            dL_dout_color, 
            dL_dout_depth, 
            dL_dout_alpha,
            sh, 
            degree, 
            campos,
            geomBuffer, 
            R, 
            binningBuffer, 
            imageBuffer, 
            alpha, 
            debug
        );

        // Map gradients to inputs. 
        return {
            std::get<3>(gradients), // 0: means3D
            std::get<1>(gradients), // 1: colors
            std::get<2>(gradients), // 2: opacity
            std::get<6>(gradients), // 3: scales
            std::get<7>(gradients), // 4: rotations
            torch::Tensor(),        // 5: scale_modifier
            torch::Tensor(),        // 6: viewmatrix
            torch::Tensor(),        // 7: projmatrix
            torch::Tensor(),        // 8: tan_fovx
            torch::Tensor(),        // 9: tan_fovy
            torch::Tensor(),        // 10: image_height
            torch::Tensor(),        // 11: image_width
            std::get<5>(gradients), // 12: sh
            torch::Tensor(),        // 13: degree
            torch::Tensor(),        // 14: campos
            torch::Tensor()         // 15: debug
        };
    }
};