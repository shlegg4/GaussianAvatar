#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

// Include your corrected SMPL header
#include "SmplLBS.h"

// ==========================================
// Helper: Save Mesh to OBJ
// ==========================================
void save_obj(const std::string& filename, torch::Tensor vertices, torch::Tensor faces) {
    vertices = vertices.cpu();
    faces = faces.cpu();
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    out << "# SMPL Mesh\n";
    
    // Write Vertices
    auto v_acc = vertices.accessor<float, 2>();
    for (int i = 0; i < vertices.size(0); ++i) {
        out << "v " << v_acc[i][0] << " " << v_acc[i][1] << " " << v_acc[i][2] << "\n";
    }

    // Write Faces (OBJ is 1-based)
    auto f_acc = faces.accessor<int64_t, 2>();
    for (int i = 0; i < faces.size(0); ++i) {
        out << "f " << f_acc[i][0] + 1 << " " << f_acc[i][1] + 1 << " " << f_acc[i][2] + 1 << "\n";
    }
    
    out.close();
    std::cout << "Saved mesh to " << filename << std::endl;
}

// ==========================================
// Helper: Save Gaussian Centers to PLY
// ==========================================
void save_ply(const std::string& filename, torch::Tensor points) {
    points = points.cpu();
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << points.size(0) << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";

    auto p_acc = points.accessor<float, 2>();
    for (int i = 0; i < points.size(0); ++i) {
        out << p_acc[i][0] << " " << p_acc[i][1] << " " << p_acc[i][2] << "\n";
    }
    
    out.close();
    std::cout << "Saved point cloud to " << filename << std::endl;
}

// ==========================================
// Gaussian Avatar Class
// ==========================================
struct GaussianAvatar : torch::nn::Module {
    std::shared_ptr<SMPLLayer> smpl;
    
    // Gaussian State (Optimization Targets)
    torch::Tensor g_scales;
    torch::Tensor g_rots;
    torch::Tensor g_opacities;

    // Binding State (Fixed Topology)
    torch::Tensor g_bary_coords; // [N, 3]
    torch::Tensor g_face_indices; // [N]
    
    // Faces buffer (Need a copy here for indexing)
    torch::Tensor faces_buffer;

    GaussianAvatar(const std::string& model_path) {
        // 1. Initialize SMPL
        smpl = std::make_shared<SMPLLayer>(model_path);
        register_module("smpl", smpl);
        
        // Load faces immediately for initialization (Assuming smpl_data.pt has 'f')
        // If your .pt file doesn't have faces, you must load them separately.
        // For now, let's try to load them from the .pt file via the same loader logic
        // OR pass them in. 
        // TRICK: We will load faces in init_gaussians() or assume user provides them.
    }

    // Initialize Random Gaussians on Surface
    void init_gaussians(int num_gaussians, torch::Tensor faces_idx) {
        auto device = smpl->v_template.device();
        this->faces_buffer = faces_idx.to(device); // Store for skinning
        
        int num_faces = faces_buffer.size(0);

        std::cout << "Initializing " << num_gaussians << " random Gaussians on " << num_faces << " faces..." << std::endl;

        // 1. Pick N random faces
        g_face_indices = torch::randint(0, num_faces, {num_gaussians}, torch::dtype(torch::kLong).device(device));
        register_buffer("g_face_indices", g_face_indices);

        // 2. Generate Random Barycentric Coords
        auto r1 = torch::rand({num_gaussians, 1}, device);
        auto r2 = torch::rand({num_gaussians, 1}, device);
        
        // Flip to keep inside triangle
        auto mask = (r1 + r2) > 1.0;
        r1.index_put_({mask}, 1.0 - r1.index({mask}));
        r2.index_put_({mask}, 1.0 - r2.index({mask}));

        auto u = r1;
        auto v = r2;
        auto w = 1.0 - u - v;
        g_bary_coords = torch::cat({u, v, w}, 1); // [N, 3]
        register_buffer("g_bary_coords", g_bary_coords);

        // 3. Init Parameters
        // Scales: Log scale -5.0 (small)
        g_scales = torch::full({num_gaussians, 3}, -5.0, torch::requires_grad().device(device));
        register_parameter("g_scales", g_scales);

        // Rots: Identity (1, 0, 0, 0)
        g_rots = torch::zeros({num_gaussians, 4}, torch::requires_grad().device(device));
        g_rots.index_put_({torch::indexing::Slice(), 0}, 1.0);
        register_parameter("g_rots", g_rots);

        // Opacity: High (visible)
        g_opacities = torch::full({num_gaussians, 1}, 2.0, torch::requires_grad().device(device));
        register_parameter("g_opacities", g_opacities);
    }

    // Forward: Deform Gaussians
    torch::Tensor forward(torch::Tensor betas, torch::Tensor pose, torch::Tensor trans) {
        // 1. Run SMPL to get deformed vertices
        auto smpl_out = smpl->forward(betas, pose, trans);
        auto verts_deformed = smpl_out.vertices[0]; // [V, 3] (Single batch for now)

        // 2. Gather Triangle Corners
        // We need [N, 3] corners based on g_face_indices
        auto selected_face_indices = faces_buffer.index_select(0, g_face_indices); // [N, 3]

        auto A = verts_deformed.index_select(0, selected_face_indices.index({torch::indexing::Slice(), 0}));
        auto B = verts_deformed.index_select(0, selected_face_indices.index({torch::indexing::Slice(), 1}));
        auto C = verts_deformed.index_select(0, selected_face_indices.index({torch::indexing::Slice(), 2}));

        // 3. Barycentric Interpolation
        // P = uA + vB + wC
        auto u = g_bary_coords.index({torch::indexing::Slice(), 0}).unsqueeze(1);
        auto v = g_bary_coords.index({torch::indexing::Slice(), 1}).unsqueeze(1);
        auto w = g_bary_coords.index({torch::indexing::Slice(), 2}).unsqueeze(1);

        auto g_means_deformed = u * A + v * B + w * C;

        return g_means_deformed; // [N, 3]
    }
    
    // Helper to expose SMPL output for saving mesh
    SmplOutput get_smpl_output(torch::Tensor betas, torch::Tensor pose, torch::Tensor trans) {
        return smpl->forward(betas, pose, trans);
    }
};

// ==========================================
// Main Function
// ==========================================
int main() {
    try {
        // 1. Setup Device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Running on GPU." << std::endl;
            device = torch::kCUDA;
        } else {
            std::cout << "CUDA not found. Running on CPU." << std::endl;
        }

        // 2. Initialize Avatar
        std::string model_path = "smpl_data.pt";
        GaussianAvatar avatar(model_path);
        avatar.to(device);

        // 3. Load Faces Manually (Since we didn't add them to SMPLLayer logic yet)
        // You MUST ensure your smpl_data.pt has 'f' or load them here.
        // For this example, I will load them from the .pt file using the same pickle logic
        // OR you can modify your python script to include 'faces' in the saved dict.
        
        std::cout << "Loading Faces from " << model_path << "..." << std::endl;
        auto bytes = torch::jit::load(model_path).attr("f").toTensor(); // This won't work on raw pickle.
        
        // HACK: Re-open pickle just to get faces (Inefficient but robust for now)
        std::ifstream input(model_path, std::ios::binary);
        std::vector<char> f_bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
        input.close();
        auto dict = torch::pickle_load(f_bytes).toGenericDict();
        
        // Ensure you added "faces" to your Python script output_dict!
        // If you forgot, add: "faces": torch.tensor(model.faces.astype(np.int64))
        if (!dict.contains("faces")) {
             throw std::runtime_error("Error: 'faces' not found in smpl_data.pt. Please update your Python conversion script.");
        }
        torch::Tensor faces = dict.at("faces").toTensor().to(torch::kLong).to(device);
        
        // 4. Initialize 10,000 Random Gaussians
        avatar.init_gaussians(10000, faces);

        // 5. Setup Inputs
        int batch_size = 1;
        auto betas = torch::zeros({batch_size, 10}, torch::requires_grad().device(device));
        auto pose = torch::zeros({batch_size, 72}, torch::requires_grad().device(device));
        auto trans = torch::zeros({batch_size, 3}, torch::requires_grad().device(device));
        
        // Move arm slightly to verify skinning
        // Joint 16/17 are arms usually. 
        pose.index_put_({0, 16 * 3}, 0.5); // Rotate arm

        // 6. Optimizer
        torch::optim::Adam optimizer(avatar.parameters(), torch::optim::AdamOptions(0.01));

        std::cout << "\n--- Starting Simulation ---\n";

        for (int i = 0; i < 5; ++i) {
            optimizer.zero_grad();

            // Forward
            auto deformed_means = avatar.forward(betas, pose, trans);

            // Dummy Loss: Pull all Gaussians to Origin (Just to test gradient flow)
            auto loss = torch::mse_loss(deformed_means, torch::zeros_like(deformed_means));

            loss.backward();
            optimizer.step();

            std::cout << "Iter " << i + 1 
                      << " | Loss: " << loss.item<float>() 
                      << " | Mean Pos: " << deformed_means.mean().item<float>() << std::endl;
        }

        std::cout << "\n--- Saving Results ---\n";
        
        // Get final state
        auto smpl_res = avatar.get_smpl_output(betas, pose, trans);
        auto gaussian_means = avatar.forward(betas, pose, trans);

        save_obj("output_mesh.obj", smpl_res.vertices[0], faces);
        save_ply("output_gaussians.ply", gaussian_means);

    } catch (const c10::Error& e) {
        std::cerr << "!!! LIBTORCH ERROR !!!\n" << e.msg() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "!!! STANDARD ERROR !!!\n" << e.what() << std::endl;
        return -1;
    }
    return 0;
}