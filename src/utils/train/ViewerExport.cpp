#include "utils/train/ViewerExport.h"

#include "GaussianRasterizer.h"
#include "utils/render/RenderMathUtils.h"

#if defined(GAUSS_HAS_OPEN3D) && GAUSS_HAS_OPEN3D
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

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

bool ExportOrientedPointCloudPly(const std::filesystem::path &path,
                                 const torch::Tensor &positions,
                                 const torch::Tensor &rotations,
                                 const torch::Tensor &scales,
                                 const torch::Tensor &opacities,
                                 const torch::Tensor &colors,
                                 float opacity_threshold,
                                 int samples_per_gaussian)
{
    if (!positions.defined() || !rotations.defined() || !scales.defined() ||
        !opacities.defined() || !colors.defined())
    {
        std::cerr << "ExportOrientedPointCloudPly: missing tensor inputs." << std::endl;
        return false;
    }
    if (positions.dim() != 2 || positions.size(1) != 3)
    {
        std::cerr << "ExportOrientedPointCloudPly: positions must be Nx3." << std::endl;
        return false;
    }
    if (rotations.dim() != 2 || rotations.size(1) != 4)
    {
        std::cerr << "ExportOrientedPointCloudPly: rotations must be Nx4." << std::endl;
        return false;
    }
    if (scales.dim() != 2 || scales.size(1) != 3)
    {
        std::cerr << "ExportOrientedPointCloudPly: scales must be Nx3." << std::endl;
        return false;
    }
    if (opacities.dim() != 2 || opacities.size(1) != 1)
    {
        std::cerr << "ExportOrientedPointCloudPly: opacities must be Nx1." << std::endl;
        return false;
    }
    if (colors.dim() != 2 || colors.size(1) != 3)
    {
        std::cerr << "ExportOrientedPointCloudPly: colors must be Nx3." << std::endl;
        return false;
    }

    const int64_t count = positions.size(0);
    if (rotations.size(0) != count || scales.size(0) != count ||
        opacities.size(0) != count || colors.size(0) != count)
    {
        std::cerr << "ExportOrientedPointCloudPly: tensor counts do not match." << std::endl;
        return false;
    }

    auto pos_cpu = positions.to(torch::kCPU).contiguous();
    auto rot_cpu = rotations.to(torch::kCPU).contiguous();
    auto sca_cpu = scales.to(torch::kCPU).contiguous();
    auto opa_cpu = opacities.to(torch::kCPU).contiguous();
    auto col_cpu = colors.to(torch::kCPU).contiguous();

    const float *pos_ptr = pos_cpu.data_ptr<float>();
    const float *rot_ptr = rot_cpu.data_ptr<float>();
    const float *sca_ptr = sca_cpu.data_ptr<float>();
    const float *opa_ptr = opa_cpu.data_ptr<float>();
    const float *col_ptr = col_cpu.data_ptr<float>();

    struct OrientedPoint
    {
        float x, y, z;
        float nx, ny, nz;
        uint8_t r, g, b;
    };

    std::vector<OrientedPoint> points;
    const int sample_count = std::max(0, samples_per_gaussian);
    points.reserve(static_cast<size_t>(count) * static_cast<size_t>(sample_count + 1));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float threshold = std::clamp(opacity_threshold, 0.0f, 1.0f);
    for (int64_t i = 0; i < count; ++i)
    {
        const float opacity = std::clamp(opa_ptr[i], 0.0f, 1.0f);
        if (opacity < threshold)
        {
            continue;
        }

        float qw = rot_ptr[i * 4 + 0];
        float qx = rot_ptr[i * 4 + 1];
        float qy = rot_ptr[i * 4 + 2];
        float qz = rot_ptr[i * 4 + 3];

        const float q_norm_sq = qw * qw + qx * qx + qy * qy + qz * qz;
        if (q_norm_sq > 1e-12f)
        {
            const float inv_norm = 1.0f / std::sqrt(q_norm_sq);
            qw *= inv_norm;
            qx *= inv_norm;
            qy *= inv_norm;
            qz *= inv_norm;
        }

        float vx_x = 1.0f - 2.0f * (qy * qy + qz * qz);
        float vx_y = 2.0f * (qx * qy + qw * qz);
        float vx_z = 2.0f * (qx * qz - qw * qy);

        float vy_x = 2.0f * (qx * qy - qw * qz);
        float vy_y = 1.0f - 2.0f * (qx * qx + qz * qz);
        float vy_z = 2.0f * (qy * qz + qw * qx);

        float nz_x = 2.0f * (qx * qz + qw * qy);
        float nz_y = 2.0f * (qy * qz - qw * qx);
        float nz_z = 1.0f - 2.0f * (qx * qx + qy * qy);

        const float n_norm_sq = nz_x * nz_x + nz_y * nz_y + nz_z * nz_z;
        if (n_norm_sq > 1e-12f)
        {
            const float inv_n_norm = 1.0f / std::sqrt(n_norm_sq);
            nz_x *= inv_n_norm;
            nz_y *= inv_n_norm;
            nz_z *= inv_n_norm;
        }

        const size_t base3 = static_cast<size_t>(i) * 3u;
        const float cx = pos_ptr[base3 + 0];
        const float cy = pos_ptr[base3 + 1];
        const float cz = pos_ptr[base3 + 2];
        const float sx = std::max(0.0f, sca_ptr[base3 + 0]);
        const float sy = std::max(0.0f, sca_ptr[base3 + 1]);

        OrientedPoint p{};
        p.x = cx;
        p.y = cy;
        p.z = cz;
        p.nx = nz_x;
        p.ny = nz_y;
        p.nz = nz_z;
        p.r = static_cast<uint8_t>(std::lround(std::clamp(col_ptr[base3 + 0], 0.0f, 1.0f) * 255.0f));
        p.g = static_cast<uint8_t>(std::lround(std::clamp(col_ptr[base3 + 1], 0.0f, 1.0f) * 255.0f));
        p.b = static_cast<uint8_t>(std::lround(std::clamp(col_ptr[base3 + 2], 0.0f, 1.0f) * 255.0f));
        points.push_back(p);

        const float radius_cutoff = 0.8f;

        for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx)
        {
            float r = radius_cutoff * std::sqrt(dist(gen));
            float theta = 2.0f * 3.1415926535f * dist(gen);

            float u = r * std::cos(theta) * sx;
            float v = r * std::sin(theta) * sy;

            OrientedPoint p_samp = p;
            p_samp.x = cx + (u * vx_x) + (v * vy_x);
            p_samp.y = cy + (u * vx_y) + (v * vy_y);
            p_samp.z = cz + (u * vx_z) + (v * vy_z);
            points.push_back(p_samp);
        }
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "ExportOrientedPointCloudPly: failed to open " << path.string() << std::endl;
        return false;
    }

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "element vertex " << points.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float nx\n";
    out << "property float ny\n";
    out << "property float nz\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto &p : points)
    {
        out.write(reinterpret_cast<const char *>(&p.x), sizeof(float));
        out.write(reinterpret_cast<const char *>(&p.y), sizeof(float));
        out.write(reinterpret_cast<const char *>(&p.z), sizeof(float));
        out.write(reinterpret_cast<const char *>(&p.nx), sizeof(float));
        out.write(reinterpret_cast<const char *>(&p.ny), sizeof(float));
        out.write(reinterpret_cast<const char *>(&p.nz), sizeof(float));
        out.write(reinterpret_cast<const char *>(&p.r), sizeof(uint8_t));
        out.write(reinterpret_cast<const char *>(&p.g), sizeof(uint8_t));
        out.write(reinterpret_cast<const char *>(&p.b), sizeof(uint8_t));
    }

    if (!out.good())
    {
        std::cerr << "ExportOrientedPointCloudPly: failed while writing " << path.string() << std::endl;
        return false;
    }

    std::cout << "Exported oriented point cloud to " << path.string()
              << " with " << points.size() << " points." << std::endl;
    return true;
}

bool ExtractMeshTSDF_Open3D(const std::filesystem::path &output_path,
                            const torch::Tensor &means3D,
                            const torch::Tensor &colors_or_sh,
                            const torch::Tensor &opacities,
                            const torch::Tensor &scales,
                            const torch::Tensor &rotations,
                            int sh_degree,
                            int H,
                            int W,
                            bool save_debug_frames)
{
    // ==========================================
    // 1. Input Validation
    // ==========================================
    if (!means3D.defined() || means3D.dim() != 2 || means3D.size(1) != 3)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: means3D must be Nx3." << std::endl;
        return false;
    }
    if (!opacities.defined() || opacities.dim() != 2 || opacities.size(1) != 1)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: opacities must be Nx1." << std::endl;
        return false;
    }
    if (!scales.defined() || scales.dim() != 2 || scales.size(1) != 3)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: scales must be Nx3." << std::endl;
        return false;
    }
    if (!rotations.defined() || rotations.dim() != 2 || rotations.size(1) != 4)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: rotations must be Nx4." << std::endl;
        return false;
    }
    if (H <= 0 || W <= 0)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: invalid image size." << std::endl;
        return false;
    }
    if (!means3D.is_cuda())
    {
        std::cerr << "ExtractMeshTSDF_Open3D: means3D must be on CUDA." << std::endl;
        return false;
    }

    const int64_t count = means3D.size(0);
    if (count <= 0 || opacities.size(0) != count || scales.size(0) != count || rotations.size(0) != count)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: tensor counts do not match." << std::endl;
        return false;
    }

    const torch::Device device = means3D.device();
    if (opacities.device() != device || scales.device() != device || rotations.device() != device)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: all render tensors must share the same device." << std::endl;
        return false;
    }

    // ==========================================
    // 2. Colors and Spherical Harmonics Setup
    // ==========================================
    torch::Tensor colors;
    torch::Tensor sh;

    if (colors_or_sh.defined() && colors_or_sh.dim() == 3)
    {
        const int64_t expected_coeffs = static_cast<int64_t>((sh_degree + 1) * (sh_degree + 1));
        if (sh_degree <= 0 || colors_or_sh.size(0) != count || colors_or_sh.size(2) != 3 || colors_or_sh.size(1) < expected_coeffs)
        {
            std::cerr << "ExtractMeshTSDF_Open3D: sh tensor must be Nx((degree+1)^2)x3." << std::endl;
            return false;
        }
        if (colors_or_sh.device() != device)
        {
            std::cerr << "ExtractMeshTSDF_Open3D: sh tensor must be on the same device as means3D." << std::endl;
            return false;
        }
        colors = torch::zeros({0}, colors_or_sh.options().dtype(torch::kFloat32));
        sh = colors_or_sh.contiguous();
    }
    else
    {
        if (!colors_or_sh.defined() || colors_or_sh.dim() != 2 || colors_or_sh.size(0) != count || colors_or_sh.size(1) != 3)
        {
            std::cerr << "ExtractMeshTSDF_Open3D: colors tensor must be Nx3." << std::endl;
            return false;
        }
        if (colors_or_sh.device() != device)
        {
            std::cerr << "ExtractMeshTSDF_Open3D: colors tensor must be on the same device as means3D." << std::endl;
            return false;
        }
        colors = colors_or_sh.contiguous();
        sh = torch::zeros({0}, colors_or_sh.options().dtype(torch::kFloat32));
    }

    // ==========================================
    // 3. Environment & Camera Initialization
    // ==========================================
    torch::NoGradGuard no_grad;
    std::cout << "Starting in-memory TSDF integration..." << std::endl;

    constexpr int kNumCameras = 1000;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kDepthScale = 1000.0f;
    constexpr float kDepthTruncMeters = 4.0f;

    auto float_opts = means3D.options().dtype(torch::kFloat32);

    open3d::pipelines::integration::ScalableTSDFVolume volume(
        0.002,
        0.004,
        open3d::pipelines::integration::TSDFVolumeColorType::RGB8);

    std::filesystem::path debug_dir;
    if (save_debug_frames)
    {
        debug_dir = output_path.parent_path() / "tsdf_views";
        std::error_code debug_ec;
        std::filesystem::create_directories(debug_dir, debug_ec);
        if (debug_ec)
        {
            std::cerr << "ExtractMeshTSDF_Open3D: failed to create debug frame directory: " << debug_dir.string() << std::endl;
            return false;
        }
    }

    // Perspective Constants (Match Viewer Exactly)
    const float fovy_deg = 45.0f;
    const float aspect = static_cast<float>(W) / static_cast<float>(H);
    const float fovy_rad = fovy_deg * kPi / 180.0f;
    const float tan_fovy = std::tan(fovy_rad * 0.5f);
    const float tan_fovx = tan_fovy * aspect;
    const float zNear = 0.01f;
    const float zFar = 100.0f;

    // OpenGL Perspective Matrix
    auto proj_cpu = torch::zeros({4, 4}, float_opts);
    proj_cpu.index_put_({0, 0}, 1.0f / (aspect * tan_fovy));
    proj_cpu.index_put_({1, 1}, 1.0f / tan_fovy);
    proj_cpu.index_put_({2, 2}, -(zFar + zNear) / (zFar - zNear));
    proj_cpu.index_put_({2, 3}, -(2.0f * zFar * zNear) / (zFar - zNear));
    proj_cpu.index_put_({3, 2}, -1.0f);
    auto proj_ten_t = proj_cpu.transpose(0, 1).contiguous().to(device);

    // Center and Radius (Match Viewer's Bounds)
    auto pos_cpu = means3D.detach().to(torch::kCPU).contiguous();
    const float *p_ptr = pos_cpu.data_ptr<float>();
    const int64_t num_pts = means3D.size(0);

    Eigen::Vector3f min_b(p_ptr[0], p_ptr[1], p_ptr[2]);
    Eigen::Vector3f max_b(p_ptr[0], p_ptr[1], p_ptr[2]);
    for (int64_t idx = 0; idx < num_pts; idx += 10)
    {
        min_b.x() = std::min(min_b.x(), p_ptr[idx * 3 + 0]);
        min_b.y() = std::min(min_b.y(), p_ptr[idx * 3 + 1]);
        min_b.z() = std::min(min_b.z(), p_ptr[idx * 3 + 2]);
        max_b.x() = std::max(max_b.x(), p_ptr[idx * 3 + 0]);
        max_b.y() = std::max(max_b.y(), p_ptr[idx * 3 + 1]);
        max_b.z() = std::max(max_b.z(), p_ptr[idx * 3 + 2]);
    }

    const Eigen::Vector3f center = (min_b + max_b) * 0.5f;
    const Eigen::Vector3f extents = (max_b - min_b) * 0.5f;
    const float radius = std::max({extents.x(), extents.y(), extents.z(), 0.1f});
    const float cam_dist = radius * 3.0f;

    // Open3D Camera Intrinsic
    const float focal_y = (static_cast<float>(H) * 0.5f) / tan_fovy;
    const float focal_x = (static_cast<float>(W) * 0.5f) / tan_fovx;
    open3d::camera::PinholeCameraIntrinsic intrinsic(
        W, H, focal_x, focal_y, static_cast<double>(W) * 0.5, static_cast<double>(H) * 0.5);

    // Safeguard colors/SH tensors for the PyTorch Wrapper
    auto render_colors = (colors_or_sh.dim() == 2) ? colors_or_sh : torch::zeros({0}, float_opts.device(device));
    auto render_sh = (colors_or_sh.dim() == 3) ? colors_or_sh : torch::zeros({0}, float_opts.device(device));

    // Reusable image buffers
    open3d::geometry::Image o3d_color;
    open3d::geometry::Image o3d_depth;
    using namespace torch::indexing; // Lifted outside the loop

    // ==========================================
    // 4. Rendering & Integration Loop
    // ==========================================
    const float golden_ratio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const float angle_increment = 2.0f * kPi * golden_ratio;

    for (int i = 0; i < kNumCameras; ++i)
    {
        // 1. Calculate Fibonacci Sphere position (y goes smoothly from 1 to -1)
        float t = static_cast<float>(i) / static_cast<float>(kNumCameras - 1);
        float y = 1.0f - (t * 2.0f);
        float radius_at_y = std::sqrt(1.0f - y * y);
        float theta = angle_increment * static_cast<float>(i);

        float x = std::cos(theta) * radius_at_y;
        float z = std::sin(theta) * radius_at_y;

        // V is the unit vector pointing from the center outward to the camera
        Eigen::Vector3f V(x, y, z);
        Eigen::Vector3f eye = center + V * cam_dist;

        // 2. Build the Rotation Matrix (R) exactly like your current math expects
        // In your math, the object is at +Z in camera space, so the Forward vector (Row 2)
        // must point from the camera (eye) to the object (center).
        Eigen::Vector3f F = (center - eye).normalized();

        // Handle the singularity if the camera is exactly at the North/South pole
        Eigen::Vector3f world_up(0.0f, 1.0f, 0.0f);
        if (std::abs(F.y()) > 0.999f)
        {
            world_up = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        }

        // Calculate Right (Row 0) and Up (Row 1) vectors orthogonally
        Eigen::Vector3f right = world_up.cross(F).normalized();
        Eigen::Vector3f up = F.cross(right).normalized();

        Eigen::Matrix3f R;
        R.row(0) = right;
        R.row(1) = up;
        R.row(2) = F;

        // 3. EXACT same View Translation math as your current code!
        // This perfectly centers the object at `cam_dist` on the Z-axis.
        Eigen::Vector3f view_trans = Eigen::Vector3f(0.0f, 0.0f, cam_dist) - R * center;

        auto view_cpu = torch::eye(4, float_opts);
        view_cpu.index_put_({0, 0}, R(0, 0));
        view_cpu.index_put_({0, 1}, R(0, 1));
        view_cpu.index_put_({0, 2}, R(0, 2));
        view_cpu.index_put_({0, 3}, view_trans.x());
        view_cpu.index_put_({1, 0}, R(1, 0));
        view_cpu.index_put_({1, 1}, R(1, 1));
        view_cpu.index_put_({1, 2}, R(1, 2));
        view_cpu.index_put_({1, 3}, view_trans.y());
        view_cpu.index_put_({2, 0}, R(2, 0));
        view_cpu.index_put_({2, 1}, R(2, 1));
        view_cpu.index_put_({2, 2}, R(2, 2));
        view_cpu.index_put_({2, 3}, view_trans.z());

        auto view_ten = view_cpu.transpose(0, 1).contiguous().to(device);

        auto proj_t = torch::matmul(view_ten, proj_ten_t).contiguous();

        Eigen::Vector3f cam_pos_eigen = -R.transpose() * view_trans;
        auto campos = torch::tensor({cam_pos_eigen.x(), cam_pos_eigen.y(), cam_pos_eigen.z()}, float_opts.device(device));

        // Render Frame via GaussianRasterizer
        auto outputs = GaussianRasterizer::apply(
            means3D, render_colors, opacities, scales, rotations,
            1.0f, // scale_modifier
            view_ten, proj_t, tan_fovx, tan_fovy, H, W,
            render_sh,
            sh_degree, // sh_degree forced to 0
            campos, false);

        auto out_color = outputs[0];
        auto out_alpha = outputs[1];
        auto out_depth = outputs[2];

        auto rgb_u8 = out_color.detach().clamp(0.0f, 1.0f).mul(255.0f).to(torch::kUInt8).permute({1, 2, 0}).contiguous().to(torch::kCPU);

        auto depth_meters = out_depth.detach() / torch::clamp_min(out_alpha.detach(), 1e-6f);
        depth_meters = torch::abs(depth_meters);
        depth_meters = torch::where(out_alpha.detach() > 1e-4f, depth_meters, torch::zeros_like(depth_meters));
        depth_meters = torch::clamp(depth_meters, 0.0f, kDepthTruncMeters);

        auto depth_u16 = (depth_meters.squeeze(0) * kDepthScale).to(torch::kUInt16).contiguous().to(torch::kCPU);

        // Flip horizontally to undo the rasterizer's X-Flip
        cv::Mat rgb_mat(H, W, CV_8UC3, rgb_u8.data_ptr<uint8_t>());
        cv::Mat depth_mat(H, W, CV_16UC1, depth_u16.data_ptr<uint16_t>());
        cv::flip(rgb_mat, rgb_mat, 1);
        cv::flip(depth_mat, depth_mat, 1);

        cv::medianBlur(depth_mat, depth_mat, 5);

        if (save_debug_frames)
        {
            // 1. Convert RGB for saving
            cv::Mat bgr;
            cv::cvtColor(rgb_mat, bgr, cv::COLOR_RGB2BGR);

            // 2. Create a visualizable version of the depth map
            cv::Mat depth_vis;
            // Scale the 16-bit values (0 to Max Depth) down to 8-bit (0 to 255)
            // We use kDepthTruncMeters * kDepthScale as the logical maximum distance
            double scale_factor = 255.0 / (kDepthTruncMeters * kDepthScale);
            depth_mat.convertTo(depth_vis, CV_8UC1, scale_factor);

            // 3. Apply a heatmap color scheme (Jet or Turbo) so depth changes pop out
            cv::Mat depth_color;
            cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_JET);

            // 4. Save the frames
            const auto rgb_path = (debug_dir / ("rgb_" + std::to_string(i) + ".png")).string();
            const auto depth_vis_path = (debug_dir / ("depth_vis_" + std::to_string(i) + ".png")).string();

            cv::imwrite(rgb_path, bgr);
            cv::imwrite(depth_vis_path, depth_color);

            // Optional: You can also save the raw 16-bit depth if you need to inspect actual values later
            // const auto depth_raw_path = (debug_dir / ("depth_raw_" + std::to_string(i) + ".png")).string();
            // cv::imwrite(depth_raw_path, depth_mat);
        }

        o3d_color.Prepare(W, H, 3, 1);
        std::memcpy(o3d_color.data_.data(), rgb_mat.data, o3d_color.data_.size());

        o3d_depth.Prepare(W, H, 1, 2);
        std::memcpy(o3d_depth.data_.data(), depth_mat.data, o3d_depth.data_.size());

        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            o3d_color, o3d_depth, kDepthScale, kDepthTruncMeters, false);

        // Extrinsic for Open3D (World-to-Camera, OpenCV coordinate system)
        Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                extrinsic(r, c) = view_cpu.index({r, c}).item<float>();
            }
        }
        // extrinsic.row(0) *= -1.0;
        extrinsic.row(1) *= -1.0;

        volume.Integrate(*rgbd, intrinsic, extrinsic);

        if (i % 10 == 0)
        {
            std::cout << "Integrated frame " << i << "/" << kNumCameras << std::endl;
        }
    }

    // POINT CLOUD EXTRACTION FOR DEBUGGING
    std::cout << "Extracting TSDF point cloud for debugging..." << std::endl;
    auto pcd = volume.ExtractPointCloud();
    if (pcd && !pcd->points_.empty())
    {
        std::string pcd_out = (output_path.parent_path() / "debug_tsdf_cloud.ply").string();
        open3d::io::WritePointCloud(pcd_out, *pcd);
        std::cout << "Saved debug point cloud to " << pcd_out << std::endl;
    }
    else
    {
        std::cerr << "Warning: Extracted point cloud is empty!" << std::endl;
    }

    // ==========================================
    // 5. Mesh Extraction & Output
    // ==========================================
    std::cout << "Extracting triangle mesh via Marching Cubes..." << std::endl;
    auto mesh = volume.ExtractTriangleMesh();

    if (!mesh || mesh->vertices_.empty())
    {
        std::cerr << "ExtractMeshTSDF_Open3D: failed to extract mesh (empty volume)." << std::endl;
        return false;
    }

    mesh->ComputeVertexNormals();

    std::error_code ec;
    const auto out_parent = output_path.parent_path();
    if (!out_parent.empty())
    {
        std::filesystem::create_directories(out_parent, ec);
    }

    if (ec)
    {
        std::cerr << "ExtractMeshTSDF_Open3D: failed to create output directory: " << out_parent.string() << std::endl;
        return false;
    }

    if (!open3d::io::WriteTriangleMesh(output_path.string(), *mesh))
    {
        std::cerr << "ExtractMeshTSDF_Open3D: failed to write mesh: " << output_path.string() << std::endl;
        return false;
    }

    std::cout << "Saved TSDF mesh to " << output_path.string() << std::endl;
    return true;
}

bool ExtractMeshPoisson_Open3D(const std::filesystem::path &output_path,
                               const torch::Tensor &positions,
                               const torch::Tensor &colors_or_sh,
                               const torch::Tensor &opacities,
                               const torch::Tensor &scales,
                               const torch::Tensor &rotations,
                               int sh_degree,
                               float opacity_threshold,
                               int samples_per_gaussian,
                               int depth)
{
#if defined(GAUSS_HAS_OPEN3D) && GAUSS_HAS_OPEN3D
    std::cout << "Starting High-Detail Poisson Surface Reconstruction..." << std::endl;

    auto pos_cpu = positions.detach().to(torch::kCPU).contiguous();
    auto rot_cpu = rotations.detach().to(torch::kCPU).contiguous();
    auto opa_cpu = opacities.detach().to(torch::kCPU).contiguous();
    auto col_cpu = colors_or_sh.detach().to(torch::kCPU).contiguous();
    auto sca_cpu = scales.detach().to(torch::kCPU).contiguous(); // NEW

    const int64_t count = pos_cpu.size(0);
    const float *pos_ptr = pos_cpu.data_ptr<float>();
    const float *rot_ptr = rot_cpu.data_ptr<float>();
    const float *opa_ptr = opa_cpu.data_ptr<float>();
    const float *col_ptr = col_cpu.data_ptr<float>();
    const float *sca_ptr = sca_cpu.data_ptr<float>(); // NEW

    open3d::geometry::PointCloud pcd;
    // Pre-allocate memory for the mean + samples
    pcd.points_.reserve(count * (samples_per_gaussian + 1));
    pcd.normals_.reserve(count * (samples_per_gaussian + 1));
    pcd.colors_.reserve(count * (samples_per_gaussian + 1));

    const float SH_C0 = 0.28209479177387814f;
    const bool is_sh = (sh_degree > 0 && col_cpu.dim() == 3);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float radius_cutoff = 0.2f; // Stay slightly inside the Gaussian edge

    int added_points = 0;
    for (int64_t i = 0; i < count; ++i)
    {
        if (opa_ptr[i] < opacity_threshold)
            continue;

        float qw = rot_ptr[i * 4 + 0];
        float qx = rot_ptr[i * 4 + 1];
        float qy = rot_ptr[i * 4 + 2];
        float qz = rot_ptr[i * 4 + 3];

        // Normal (Z-axis)
        float nz_x = 2.0f * (qx * qz + qw * qy);
        float nz_y = 2.0f * (qy * qz - qw * qx);
        float nz_z = 1.0f - 2.0f * (qx * qx + qy * qy);
        float n_len = std::sqrt(nz_x * nz_x + nz_y * nz_y + nz_z * nz_z);
        if (n_len > 1e-6f)
        {
            nz_x /= n_len;
            nz_y /= n_len;
            nz_z /= n_len;
        }

        // Outward Normal Check
        float dir_x = pos_ptr[i * 3 + 0];
        float dir_z = pos_ptr[i * 3 + 2];
        if ((nz_x * dir_x + nz_z * dir_z) < 0)
        {
            nz_x = -nz_x;
            nz_y = -nz_y;
            nz_z = -nz_z;
        }

        // Color
        float r, g, b;
        if (is_sh)
        {
            const int sh_dim = col_cpu.size(1);
            const size_t base_idx = i * (sh_dim * 3);

            // Base Color (DC Component)
            r = col_ptr[base_idx + 0] * SH_C0;
            g = col_ptr[base_idx + 1] * SH_C0;
            b = col_ptr[base_idx + 2] * SH_C0;

            // Add Degree 1 Lighting (Ambient Occlusion & Directional Shadows)
            if (sh_degree > 0 && sh_dim >= 4)
            {
                const float SH_C1 = 0.4886025119029199f;
                // We use the outward normal (nz_x, nz_y, nz_z) as the viewing direction
                // to bake the color as if looking directly at the surface
                float x = nz_x;
                float y = nz_y;
                float z = nz_z;

                r -= SH_C1 * y * col_ptr[base_idx + 3] + SH_C1 * z * col_ptr[base_idx + 6] - SH_C1 * x * col_ptr[base_idx + 9];
                g -= SH_C1 * y * col_ptr[base_idx + 4] + SH_C1 * z * col_ptr[base_idx + 7] - SH_C1 * x * col_ptr[base_idx + 10];
                b -= SH_C1 * y * col_ptr[base_idx + 5] + SH_C1 * z * col_ptr[base_idx + 8] - SH_C1 * x * col_ptr[base_idx + 11];
            }
            // Move from SH space (-0.5 to 0.5) to RGB space (0.0 to 1.0)
            r += 0.5f;
            g += 0.5f;
            b += 0.5f;
        }
        else
        {
            r = col_ptr[i * 3 + 0];
            g = col_ptr[i * 3 + 1];
            b = col_ptr[i * 3 + 2];
        }

        r = std::clamp(r, 0.0f, 1.0f);
        g = std::clamp(g, 0.0f, 1.0f);
        b = std::clamp(b, 0.0f, 1.0f);

        // Center position
        float cx = pos_ptr[i * 3 + 0];
        float cy = pos_ptr[i * 3 + 1];
        float cz = pos_ptr[i * 3 + 2];

        // 1. Add the mean point
        pcd.points_.emplace_back(cx, cy, cz);
        pcd.normals_.emplace_back(nz_x, nz_y, nz_z);
        pcd.colors_.emplace_back(r, g, b);
        added_points++;

        // 2. Add the surface samples
        if (samples_per_gaussian > 0)
        {
            // X and Y axes for the disk
            float vx_x = 1.0f - 2.0f * (qy * qy + qz * qz);
            float vx_y = 2.0f * (qx * qy + qw * qz);
            float vx_z = 2.0f * (qx * qz - qw * qy);
            float vy_x = 2.0f * (qx * qy - qw * qz);
            float vy_y = 1.0f - 2.0f * (qx * qx + qz * qz);
            float vy_z = 2.0f * (qy * qz + qw * qx);

            float sx = std::max(0.0f, sca_ptr[i * 3 + 0]);
            float sy = std::max(0.0f, sca_ptr[i * 3 + 1]);

            for (int sample_idx = 0; sample_idx < samples_per_gaussian; ++sample_idx)
            {
                float r_samp = radius_cutoff * std::sqrt(dist(gen));
                float theta = 2.0f * 3.1415926535f * dist(gen);

                float u = r_samp * std::cos(theta) * sx;
                float v = r_samp * std::sin(theta) * sy;

                pcd.points_.emplace_back(
                    cx + (u * vx_x) + (v * vy_x),
                    cy + (u * vx_y) + (v * vy_y),
                    cz + (u * vx_z) + (v * vy_z));
                pcd.normals_.emplace_back(nz_x, nz_y, nz_z);
                pcd.colors_.emplace_back(r, g, b);
                added_points++;
            }
        }
    }

    std::cout << "Running Poisson on " << added_points << " oriented surface points (Depth: " << depth << ")..." << std::endl;

    auto poisson_res = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(pcd, depth);
    auto mesh = std::get<0>(poisson_res);
    auto densities = std::get<1>(poisson_res);

    if (!densities.empty() && !mesh->vertices_.empty())
    {
        std::vector<double> sorted_densities = densities;
        std::sort(sorted_densities.begin(), sorted_densities.end());
        double density_thresh = sorted_densities[sorted_densities.size() / 100]; // 1% trim

        std::vector<bool> remove_mask(mesh->vertices_.size(), false);
        for (size_t i = 0; i < densities.size(); ++i)
        {
            if (densities[i] < density_thresh)
            {
                remove_mask[i] = true;
            }
        }
        mesh->RemoveVerticesByMask(remove_mask);
    }

    std::cout << "Sanitizing degenerate geometry..." << std::endl;
    mesh->RemoveDuplicatedVertices();
    mesh->RemoveDuplicatedTriangles();
    mesh->RemoveDegenerateTriangles();
    mesh->RemoveUnreferencedVertices();

    std::cout << "Applying Taubin smoothing to remove noise while preserving features..." << std::endl;
    // 15-20 iterations is safe here because the 'mu' parameter prevents shrinking/melting.
    mesh = mesh->FilterSmoothTaubin(15, 0.5, -0.53);

    std::cout << "Decimating mesh to reduce triangle count..." << std::endl;
    mesh = mesh->SimplifyQuadricDecimation(250000, std::numeric_limits<double>::infinity(), 1.0);

    std::error_code ec;
    std::filesystem::create_directories(output_path.parent_path(), ec);
    mesh->ComputeVertexNormals();

    if (!open3d::io::WriteTriangleMesh(output_path.string(), *mesh))
    {
        std::cerr << "ExtractMeshPoisson: Failed to write mesh." << std::endl;
        return false;
    }

    return true;
#else
    return false;
#endif
}