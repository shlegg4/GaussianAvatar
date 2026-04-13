#include "utils/render/GaussianUtils.h"

#include "SharedGaussian.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

torch::Tensor BakeSHToRGB(const torch::Tensor &sh,
                          const torch::Tensor &rotations,
                          int sh_degree)
{
    if (!sh.defined() || sh.dim() != 3 || sh.size(2) != 3)
    {
        return torch::Tensor();
    }

    // Ensure rotations tensor is valid so we can compute the normals
    if (!rotations.defined() || rotations.dim() != 2 || rotations.size(1) != 4)
    {
        std::cerr << "BakeSHToRGB: missing or invalid rotations tensor." << std::endl;
        return torch::Tensor();
    }

    const int64_t count = sh.size(0);
    if (count <= 0)
    {
        return torch::Tensor();
    }

    auto sh_cpu = sh.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto rot_cpu = rotations.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto rgb = torch::zeros({count, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    const float *sh_ptr = sh_cpu.data_ptr<float>();
    const float *rot_ptr = rot_cpu.data_ptr<float>();
    float *rgb_ptr = rgb.data_ptr<float>();
    const int64_t coeffs = sh_cpu.size(1);

    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float SH_C1 = 0.4886025119029199f;
    constexpr float SH_C2_0 = 1.0925484305920792f;
    constexpr float SH_C2_1 = -1.0925484305920792f;
    constexpr float SH_C2_2 = 0.31539156525252005f;
    constexpr float SH_C2_3 = -1.0925484305920792f;
    constexpr float SH_C2_4 = 0.5462742152960396f;
    constexpr float SH_C3_0 = -0.5900435899266435f;
    constexpr float SH_C3_1 = 2.890611442640554f;
    constexpr float SH_C3_2 = -0.4570457994644658f;
    constexpr float SH_C3_3 = 0.3731763325901154f;
    constexpr float SH_C3_4 = -0.4570457994644658f;
    constexpr float SH_C3_5 = 1.445305721320277f;
    constexpr float SH_C3_6 = -0.5900435899266435f;

    const int max_degree = std::max(0, std::min(3, sh_degree));

    // Define our +- 10 degree sampling cone in local space
    // 10 degrees = 0.1745 radians. sin(10) ~ 0.1736, cos(10) ~ 0.9848
    const float sin10 = 0.173648f;
    const float cos10 = 0.984808f;
    const float local_dirs[5][3] = {
        {0.0f, 0.0f, 1.0f},    // Center (Exact Normal)
        {sin10, 0.0f, cos10},  // Right
        {-sin10, 0.0f, cos10}, // Left
        {0.0f, sin10, cos10},  // Up
        {0.0f, -sin10, cos10}  // Down
    };

    for (int64_t i = 0; i < count; ++i)
    {
        // Extract quaternion for this Gaussian
        float qw = rot_ptr[i * 4 + 0];
        float qx = rot_ptr[i * 4 + 1];
        float qy = rot_ptr[i * 4 + 2];
        float qz = rot_ptr[i * 4 + 3];

        // Precompute the 3x3 Rotation Matrix from the Quaternion
        float x2 = qx + qx;
        float y2 = qy + qy;
        float z2 = qz + qz;
        float xx = qx * x2;
        float yy = qy * y2;
        float zz = qz * z2;
        float xy = qx * y2;
        float xz = qx * z2;
        float yz = qy * z2;
        float wx = qw * x2;
        float wy = qw * y2;
        float wz = qw * z2;

        float m00 = 1.0f - (yy + zz);
        float m01 = xy - wz;
        float m02 = xz + wy;
        float m10 = xy + wz;
        float m11 = 1.0f - (xx + zz);
        float m12 = yz - wx;
        float m20 = xz - wy;
        float m21 = yz + wx;
        float m22 = 1.0f - (xx + yy);

        for (int c = 0; c < 3; ++c)
        {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(coeffs) * 3u;
            auto coeff = [&](int idx) -> float
            {
                if (idx < 0 || idx >= coeffs)
                    return 0.0f;
                return sh_ptr[base + static_cast<size_t>(idx) * 3u + static_cast<size_t>(c)];
            };

            auto eval_dir = [&](float x, float y, float z) -> float
            {
                const float xx = x * x;
                const float yy = y * y;
                const float zz = z * z;
                const float xy = x * y;
                const float yz = y * z;
                const float xz = x * z;

                float result = SH_C0 * coeff(0);
                if (max_degree > 0)
                {
                    result += -SH_C1 * y * coeff(1) + SH_C1 * z * coeff(2) - SH_C1 * x * coeff(3);
                }
                if (max_degree > 1)
                {
                    result += SH_C2_0 * xy * coeff(4) + SH_C2_1 * yz * coeff(5) +
                              SH_C2_2 * (2.0f * zz - xx - yy) * coeff(6) +
                              SH_C2_3 * xz * coeff(7) + SH_C2_4 * (xx - yy) * coeff(8);
                }
                if (max_degree > 2)
                {
                    result += SH_C3_0 * y * (3.0f * xx - yy) * coeff(9) +
                              SH_C3_1 * xy * z * coeff(10) +
                              SH_C3_2 * y * (4.0f * zz - xx - yy) * coeff(11) +
                              SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * coeff(12) +
                              SH_C3_4 * x * (4.0f * zz - xx - yy) * coeff(13) +
                              SH_C3_5 * z * (xx - yy) * coeff(14) +
                              SH_C3_6 * x * (xx - 3.0f * yy) * coeff(15);
                }
                return result;
            };

            // Sample the 10-degree cone and average the results
            float sum_val = 0.0f;
            for (int d = 0; d < 5; ++d)
            {
                float vx = local_dirs[d][0];
                float vy = local_dirs[d][1];
                float vz = local_dirs[d][2];

                // Rotate the local cone vector into the Gaussian's global orientation
                float world_x = m00 * vx + m01 * vy + m02 * vz;
                float world_y = m10 * vx + m11 * vy + m12 * vz;
                float world_z = m20 * vx + m21 * vy + m22 * vz;

                // Normalize just to be mathematically safe after matrix multiplication
                float len = std::sqrt(world_x * world_x + world_y * world_y + world_z * world_z) + 1e-6f;
                sum_val += eval_dir(world_x / len, world_y / len, world_z / len);
            }

            // Average the 5 cone samples
            rgb_ptr[i * 3 + c] = (sum_val / 5.0f) + 0.5f;
        }
    }

    return rgb.to(sh.device());
}

bool BuildSharedGaussianBuffer(const torch::Tensor &positions,
                               const torch::Tensor &colors,
                               const torch::Tensor &opacities,
                               const torch::Tensor &scales,
                               const torch::Tensor &rotations,
                               const torch::Tensor &sh,
                               int sh_degree,
                               std::vector<float> *out)
{
    if (!positions.defined() || !colors.defined() || !opacities.defined() || !scales.defined() ||
        !rotations.defined())
        return false;
    if (positions.dim() != 2 || positions.size(1) != 3)
        return false;
    if (colors.dim() != 2 || colors.size(1) != 3)
        return false;
    if (opacities.dim() != 2 || opacities.size(1) != 1)
        return false;
    if (scales.dim() != 2 || scales.size(1) != 3)
        return false;
    if (rotations.dim() != 2 || rotations.size(1) != 4)
        return false;

    const int64_t count = positions.size(0);
    if (colors.size(0) != count || opacities.size(0) != count || scales.size(0) != count ||
        rotations.size(0) != count)
        return false;

    auto pos_cpu = positions.to(torch::kCPU).contiguous();
    auto col_cpu = colors.to(torch::kCPU).contiguous();
    auto opa_cpu = opacities.to(torch::kCPU).contiguous();
    auto sca_cpu = scales.to(torch::kCPU).contiguous();
    auto rot_cpu = rotations.to(torch::kCPU).contiguous();
    torch::Tensor sh_cpu;
    const int sh_coeffs = (sh_degree > 0) ? (sh_degree + 1) * (sh_degree + 1) : 0;
    if (sh_coeffs > 0)
    {
        if (!sh.defined() || sh.dim() != 3 || sh.size(1) != sh_coeffs || sh.size(2) != 3)
            return false;
        if (sh.size(0) != count)
            return false;
        sh_cpu = sh.to(torch::kCPU).contiguous();
    }

    const float *pos_ptr = pos_cpu.data_ptr<float>();
    const float *col_ptr = col_cpu.data_ptr<float>();
    const float *opa_ptr = opa_cpu.data_ptr<float>();
    const float *sca_ptr = sca_cpu.data_ptr<float>();
    const float *rot_ptr = rot_cpu.data_ptr<float>();

    const size_t stride = static_cast<size_t>(shared_gaussian::kSharedStrideFloats) + 7u +
                          static_cast<size_t>(sh_coeffs) * 3u;
    out->resize(static_cast<size_t>(count) * stride);
    float *dst = out->data();
    for (int64_t i = 0; i < count; ++i)
    {
        const size_t base = static_cast<size_t>(i) * stride;
        const size_t pos_base = static_cast<size_t>(i) * 3;
        dst[base + 0] = pos_ptr[pos_base + 0];
        dst[base + 1] = pos_ptr[pos_base + 1];
        dst[base + 2] = pos_ptr[pos_base + 2];
        dst[base + 3] = std::clamp(col_ptr[pos_base + 0], 0.0f, 1.0f);
        dst[base + 4] = std::clamp(col_ptr[pos_base + 1], 0.0f, 1.0f);
        dst[base + 5] = std::clamp(col_ptr[pos_base + 2], 0.0f, 1.0f);
        dst[base + 6] = std::clamp(opa_ptr[i], 0.0f, 1.0f);
        dst[base + 7] = (sca_ptr[pos_base + 0] + sca_ptr[pos_base + 1] + sca_ptr[pos_base + 2]) / 3.0f;
        dst[base + 8] = sca_ptr[pos_base + 0];
        dst[base + 9] = sca_ptr[pos_base + 1];
        dst[base + 10] = sca_ptr[pos_base + 2];
        const size_t rot_base = static_cast<size_t>(i) * 4;
        dst[base + 11] = rot_ptr[rot_base + 0];
        dst[base + 12] = rot_ptr[rot_base + 1];
        dst[base + 13] = rot_ptr[rot_base + 2];
        dst[base + 14] = rot_ptr[rot_base + 3];

        if (sh_coeffs > 0)
        {
            const float *sh_ptr = sh_cpu.data_ptr<float>() + (static_cast<size_t>(i) * sh_coeffs * 3u);
            float *dst_sh = dst + base + shared_gaussian::kSharedStrideFloats + 7u;
            std::memcpy(dst_sh, sh_ptr, static_cast<size_t>(sh_coeffs) * 3u * sizeof(float));
        }
    }
    return true;
}
