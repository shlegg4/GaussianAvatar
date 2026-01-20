#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <tuple>

#include "utils/SmplLBS.h"
#include "GaussianRasterizer.h" // Include our new wrapper

// ==========================================
// Helper: Save Tensor to PPM Image
// ==========================================
void save_image_ppm(const std::string& filename, torch::Tensor image) {
    // Input: [3, H, W] Float Tensor (0.0 - 1.0)
    image = image.permute({1, 2, 0}).mul(255.0).clamp(0, 255).to(torch::kByte).cpu();
    
    int H = image.size(0);
    int W = image.size(1);
    
    std::ofstream out(filename, std::ios::binary);
    out << "P6\n" << W << " " << H << "\n255\n";
    
    auto data_ptr = image.data_ptr<uint8_t>();
    out.write(reinterpret_cast<const char*>(data_ptr), H * W * 3);
    out.close();
    std::cout << "Saved image to " << filename << std::endl;
}

// ==========================================
// Helper: Get Camera Matrices (0,0,-3 looking at Origin)
// ==========================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_camera_setup(int W, int H, torch::Device device) {
    float fov = 60.0f;
    float n = 0.01f;
    float f = 100.0f;
    
    // 1. View Matrix (Simple translation back along Z)
    auto view = torch::eye(4, device);
    view.index_put_({2, 3}, 3.0f); // Translate world 3 units away from cam (effectively cam is at -3)
    // view = view.inverse(); // If representing Camera-to-World, invert. Here we built World-to-Camera directly.

    // 2. Projection Matrix
    auto proj = torch::zeros({4, 4}, device);
    float tan_half_fov = tan((fov / 2.0f) * 3.14159f / 180.0f);
    proj[0][0] = 1.0f / (tan_half_fov * ((float)W / H));
    proj[1][1] = 1.0f / tan_half_fov;
    proj[2][2] = -(f + n) / (f - n);
    proj[2][3] = -(2.0f * f * n) / (f - n);
    proj[3][2] = -1.0f;
    
    // 3. Cam Pos
    auto cam_pos = torch::tensor({0.0f, 0.0f, -3.0f}, device);
    
    // Rasterizer expects column-major / transposed matrices depending on implementation
    return {view.transpose(0, 1).contiguous(), proj.transpose(0, 1).contiguous(), cam_pos};
}

// ==========================================
// Gaussian Avatar (Same as before)
// ==========================================
struct GaussianAvatar : torch::nn::Module {
    std::shared_ptr<SMPLLayer> smpl;
    torch::Tensor g_scales, g_rots, g_opacities;
    torch::Tensor g_bary_coords, g_face_indices, faces_buffer;

    GaussianAvatar(const std::string& model_path) {
        smpl = std::make_shared<SMPLLayer>(model_path);
        register_module("smpl", smpl);
    }

    void init_gaussians(int num_gaussians, torch::Tensor faces_idx) {
        auto device = smpl->v_template.device();
        this->faces_buffer = faces_idx.to(device);
        int num_faces = faces_buffer.size(0);

        g_face_indices = torch::randint(0, num_faces, {num_gaussians}, torch::kLong).to(device);
        
        auto r1 = torch::rand({num_gaussians, 1}, device);
        auto r2 = torch::rand({num_gaussians, 1}, device);
        auto mask = (r1 + r2) > 1.0;
        r1.index_put_({mask}, 1.0 - r1.index({mask}));
        r2.index_put_({mask}, 1.0 - r2.index({mask}));
        
        auto w = 1.0 - r1 - r2;
        g_bary_coords = torch::cat({r1, r2, w}, 1);

        register_buffer("g_face_indices", g_face_indices);
        register_buffer("g_bary_coords", g_bary_coords);

        g_scales = torch::full({num_gaussians, 3}, -2.0, torch::requires_grad().device(device)); // Small log scale
        auto g_rots_init = torch::zeros({num_gaussians, 4}, torch::TensorOptions().device(device));
        g_rots_init.index_put_({torch::indexing::Slice(), 0}, 1.0);
        g_rots = g_rots_init.detach().clone().set_requires_grad(true);
        g_opacities = torch::full({num_gaussians, 1}, 2.0, torch::requires_grad().device(device)); // Visible

        register_parameter("g_scales", g_scales);
        register_parameter("g_rots", g_rots);
        register_parameter("g_opacities", g_opacities);
    }

    torch::Tensor forward(torch::Tensor betas, torch::Tensor pose, torch::Tensor trans) {
        auto smpl_out = smpl->forward(betas, pose, trans);
        auto verts = smpl_out.vertices[0];
        auto selected_faces = faces_buffer.index_select(0, g_face_indices);

        auto A = verts.index_select(0, selected_faces.index({torch::indexing::Slice(), 0}));
        auto B = verts.index_select(0, selected_faces.index({torch::indexing::Slice(), 1}));
        auto C = verts.index_select(0, selected_faces.index({torch::indexing::Slice(), 2}));

        auto u = g_bary_coords.index({torch::indexing::Slice(), 0}).unsqueeze(1);
        auto v = g_bary_coords.index({torch::indexing::Slice(), 1}).unsqueeze(1);
        auto w = g_bary_coords.index({torch::indexing::Slice(), 2}).unsqueeze(1);

        return u * A + v * B + w * C;
    }
};

// ==========================================
// Main
// ==========================================
int main() {
    try {
        if (!torch::cuda::is_available()) {
            std::cerr << "CUDA Required for Rasterizer!" << std::endl;
            return -1;
        }
        auto device = torch::kCUDA;

        // 1. Init Avatar
        std::string model_path = "smpl_data.pt";
        GaussianAvatar avatar(model_path);
        avatar.to(device);

        // Load Faces (Hack)
        std::ifstream input(model_path, std::ios::binary);
        std::vector<char> f_bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
        auto dict = torch::pickle_load(f_bytes).toGenericDict();
        torch::Tensor faces = dict.at("faces").toTensor().to(torch::kLong).to(device);
        
        avatar.init_gaussians(20000, faces); // More gaussians for a solid look

        // 2. Setup Rasterizer Params
        int W = 800, H = 800;
        auto [view_mat, proj_mat, cam_pos] = get_camera_setup(W, H, device);
        
        // FOV Tangents
        float tan_fov = tan(30.0f * 3.14159f / 180.0f); // 60 deg fov
        
        // Random Colors (SH Degree 0)
        int N = avatar.g_scales.size(0);
        auto sh = torch::rand({N, 1, 3}, device); // Random colors
        auto colors = torch::zeros({N, 3}, torch::TensorOptions().device(device));

        // 3. Optimization
        auto betas = torch::zeros({1, 10}, torch::requires_grad().device(device));
        auto pose_init = torch::zeros({1, 72}, torch::TensorOptions().device(device));
        pose_init.index_put_({0, 16 * 3}, 0.5); // Lift arm
        auto pose = pose_init.detach().clone().set_requires_grad(true);
        auto trans = torch::zeros({1, 3}, torch::requires_grad().device(device));
        
        torch::optim::Adam optimizer(avatar.parameters(), 0.01);

        std::cout << "Rendering..." << std::endl;

        for (int i = 0; i < 5; ++i) {
            optimizer.zero_grad();
            
            // Deform
            auto means3D = avatar.forward(betas, pose, trans);
            
            // Render
            auto image = GaussianRasterizer::apply(
                means3D,
                colors, // colors (valid tensor even if SH is used)
                avatar.g_opacities,
                avatar.g_scales,
                avatar.g_rots,
                1.0f, // scale_mod
                view_mat, proj_mat,
                tan_fov, tan_fov,
                H, W,
                sh,
                0, // Degree 0
                cam_pos,
                false // debug
            );

            // Loss: Try to make image white (Arbitrary test)
            auto loss = torch::mse_loss(image, torch::ones_like(image)); 
            
            loss.backward();
            // optimizer.step();
            
            std::cout << "Iter " << i << " Loss: " << loss.item<float>() << std::endl;

            if (i == 4) {
                save_image_ppm("render_output.ppm", image.detach());
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
