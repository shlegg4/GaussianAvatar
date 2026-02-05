#include "utils/train/ViewerExport.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

namespace
{
std::filesystem::path MakeEpochDir(const std::filesystem::path &root, int epoch)
{
    return root / ("epoch_" + std::to_string(epoch));
}
}

bool SaveViewerData(const std::filesystem::path &output_dir, int epoch,
                    const torch::Tensor &positions,
                    const torch::Tensor &colors,
                    const torch::Tensor &opacities,
                    const torch::Tensor &scales,
                    const torch::Tensor &rotations,
                    const torch::Tensor &sh,
                    int sh_degree)
{
    // --- Validation Checks ---
    if (!positions.defined() || !colors.defined() || !opacities.defined() || !scales.defined() ||
        !rotations.defined())
    {
        std::cerr << "SaveViewerData: missing tensor inputs." << std::endl;
        return false;
    }
    if (positions.dim() != 2 || positions.size(1) != 3)
    {
        std::cerr << "SaveViewerData: positions must be Nx3." << std::endl;
        return false;
    }
    if (colors.dim() != 2 || colors.size(1) != 3)
    {
        std::cerr << "SaveViewerData: colors must be Nx3." << std::endl;
        return false;
    }
    if (opacities.dim() != 2 || opacities.size(1) != 1)
    {
        std::cerr << "SaveViewerData: opacities must be Nx1." << std::endl;
        return false;
    }
    if (scales.dim() != 2 || scales.size(1) != 3)
    {
        std::cerr << "SaveViewerData: scales must be Nx3." << std::endl;
        return false;
    }
    if (rotations.dim() != 2 || rotations.size(1) != 4)
    {
        std::cerr << "SaveViewerData: rotations must be Nx4." << std::endl;
        return false;
    }

    const int64_t count = positions.size(0);
    if (colors.size(0) != count || opacities.size(0) != count || scales.size(0) != count ||
        rotations.size(0) != count)
    {
        std::cerr << "SaveViewerData: tensor counts do not match." << std::endl;
        return false;
    }

    // --- Prepare CPU Access ---
    auto pos_cpu = positions.to(torch::kCPU).contiguous();
    auto col_cpu = colors.to(torch::kCPU).contiguous();
    auto opa_cpu = opacities.to(torch::kCPU).contiguous();
    auto sca_cpu = scales.to(torch::kCPU).contiguous();
    auto rot_cpu = rotations.to(torch::kCPU).contiguous();
    torch::Tensor sh_cpu;
    const int sh_coeffs = (sh_degree > 0) ? (sh_degree + 1) * (sh_degree + 1) : 0;
    if (sh_coeffs > 0)
    {
        if (!sh.defined() || sh.dim() != 3 || sh.size(1) != sh_coeffs || sh.size(2) != 3 ||
            sh.size(0) != count)
        {
            std::cerr << "SaveViewerData: sh must be Nx" << sh_coeffs << "x3." << std::endl;
            return false;
        }
        sh_cpu = sh.to(torch::kCPU).contiguous();
    }

    const float *pos_ptr = pos_cpu.data_ptr<float>();
    const float *col_ptr = col_cpu.data_ptr<float>();
    const float *opa_ptr = opa_cpu.data_ptr<float>();
    const float *sca_ptr = sca_cpu.data_ptr<float>();
    const float *rot_ptr = rot_cpu.data_ptr<float>();
    const float *sh_ptr = (sh_coeffs > 0) ? sh_cpu.data_ptr<float>() : nullptr;

    // --- Data Layout Configuration ---
    // Stride 15 aligns with the Desktop Viewer and robust Web Viewers.
    // Layout: [0-2 Pos] [3-5 Col] [6 Op] [7 Pad] [8-10 Scale] [11-14 Rot]
    const size_t stride = 15u + static_cast<size_t>(sh_coeffs) * 3u;
    std::vector<float> buffer;
    buffer.resize(static_cast<size_t>(count) * stride);

    for (int64_t i = 0; i < count; ++i)
    {
        const size_t base = static_cast<size_t>(i) * stride;
        const size_t pos_base = static_cast<size_t>(i) * 3u;
        const size_t rot_base = static_cast<size_t>(i) * 4u;

        // Position
        buffer[base + 0] = pos_ptr[pos_base + 0];
        buffer[base + 1] = pos_ptr[pos_base + 1];
        buffer[base + 2] = pos_ptr[pos_base + 2];

        // Color
        buffer[base + 3] = std::clamp(col_ptr[pos_base + 0], 0.0f, 1.0f);
        buffer[base + 4] = std::clamp(col_ptr[pos_base + 1], 0.0f, 1.0f);
        buffer[base + 5] = std::clamp(col_ptr[pos_base + 2], 0.0f, 1.0f);

        // Opacity
        buffer[base + 6] = std::clamp(opa_ptr[i], 0.0f, 1.0f);

        // --- FIXED: PADDING at Index 7 ---
        // Some viewers use this for average scale, others skip it. 
        // Filling it with avg scale is safer than 0.0 if a viewer tries to use it.
        buffer[base + 7] = (sca_ptr[pos_base + 0] + sca_ptr[pos_base + 1] + sca_ptr[pos_base + 2]) / 3.0f;

        // Scale (Shifted to 8, 9, 10)
        buffer[base + 8] = sca_ptr[pos_base + 0];
        buffer[base + 9] = sca_ptr[pos_base + 1];
        buffer[base + 10] = sca_ptr[pos_base + 2];

        // Rotation (Shifted to 11, 12, 13, 14)
        buffer[base + 11] = rot_ptr[rot_base + 0];
        buffer[base + 12] = rot_ptr[rot_base + 1];
        buffer[base + 13] = rot_ptr[rot_base + 2];
        buffer[base + 14] = rot_ptr[rot_base + 3];

        // Spherical Harmonics (Shifted to 15+)
        if (sh_coeffs > 0)
        {
            const size_t sh_base = base + 15u;
            const size_t sh_src_base = static_cast<size_t>(i) * static_cast<size_t>(sh_coeffs) * 3u;
            std::memcpy(buffer.data() + sh_base, sh_ptr + sh_src_base,
                        static_cast<size_t>(sh_coeffs) * 3u * sizeof(float));
        }
    }

    // --- File Writing ---
    std::error_code ec;
    auto epoch_dir = MakeEpochDir(output_dir, epoch);
    std::filesystem::create_directories(epoch_dir, ec);
    if (ec)
    {
        std::cerr << "SaveViewerData: failed to create output dir: " << epoch_dir.string() << std::endl;
        return false;
    }

    const std::string bin_filename = "avatar_data.bin";
    const std::string json_filename = "avatar_state.json";

    const auto bin_path = epoch_dir / bin_filename;
    std::ofstream bin_file(bin_path, std::ios::binary);
    if (!bin_file.is_open())
    {
        std::cerr << "SaveViewerData: failed to open " << bin_path.string() << std::endl;
        return false;
    }
    bin_file.write(reinterpret_cast<const char *>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size() * sizeof(float)));
    bin_file.close();

    const auto json_path = epoch_dir / json_filename;
    std::ofstream json_file(json_path);
    if (!json_file.is_open())
    {
        std::cerr << "SaveViewerData: failed to open " << json_path.string() << std::endl;
        return false;
    }
    
    // Writing standard JSON Manifest
    json_file << "{\n";
    json_file << "  \"num_gaussians\": " << count << ",\n";
    json_file << "  \"sh_degree\": " << sh_degree << ",\n";
    json_file << "  \"blob_filename\": \"" << bin_filename << "\",\n";
    json_file << "  \"data_file\": \"" << bin_filename << "\",\n"; // Backward compatibility
    json_file << "  \"layout\": [\n";
    json_file << "    \"pos_x\", \"pos_y\", \"pos_z\",\n";
    json_file << "    \"r\", \"g\", \"b\",\n";
    json_file << "    \"opacity\",\n";
    json_file << "    \"padding_avg_scale\",\n"; // Documenting the gap
    json_file << "    \"scale_x\", \"scale_y\", \"scale_z\",\n";
    json_file << "    \"rot_w\", \"rot_x\", \"rot_y\", \"rot_z\"";
    if (sh_coeffs > 0)
    {
        json_file << ",\n";
        for (int coeff = 0; coeff < sh_coeffs; ++coeff)
        {
            json_file << "    \"sh" << coeff << "_r\", \"sh" << coeff << "_g\", \"sh" << coeff << "_b\"";
            if (coeff + 1 < sh_coeffs)
            {
                json_file << ",\n";
            }
        }
    }
    json_file << "\n  ]\n";
    json_file << "}\n";
    json_file.close();

    std::cout << "Saved viewer data to " << epoch_dir.string() << std::endl;
    return true;
}

bool SaveViewerDataOverwrite(const std::filesystem::path &output_dir,
                             const torch::Tensor &positions,
                             const torch::Tensor &colors,
                             const torch::Tensor &opacities,
                             const torch::Tensor &scales,
                             const torch::Tensor &rotations,
                             const torch::Tensor &sh,
                             int sh_degree)
{
    return SaveViewerData(output_dir, 0, positions, colors, opacities, scales, rotations, sh, sh_degree);
}