#include "utils/render/RenderMathUtils.h"

#include <algorithm>

CameraProjectionOutput BuildCameraProjection(const CameraProjectionInput &input, torch::Device device)
{
    const int render_w = std::max(1, input.render_width);
    const int render_h = std::max(1, input.render_height);
    const int full_w = std::max(1, input.full_width);
    const int full_h = std::max(1, input.full_height);

    const float full_cx = static_cast<float>(full_w) * 0.5f;
    const float full_cy = static_cast<float>(full_h) * 0.5f;

    float x0 = input.crop_x0;
    float y0 = input.crop_y0;
    if (input.crop_w <= 0.0f || input.crop_h <= 0.0f)
    {
        x0 = input.crop_cx - (static_cast<float>(render_w) * 0.5f);
        y0 = input.crop_cy - (static_cast<float>(render_h) * 0.5f);
    }

    const float cx_crop = full_cx - x0;
    const float cy_crop = full_cy - y0;
    const float fx = (input.focal_x > 0.0f) ? input.focal_x : input.focal;
    const float fy = (input.focal_y > 0.0f) ? input.focal_y : input.focal;

    CameraProjectionOutput output;
    output.principal_x = cx_crop;
    output.principal_y = cy_crop;
    std::tie(output.view_mat, output.proj_mat, output.tan_fovx, output.tan_fovy) =
        BuildProjection(fx, fy, render_w, render_h, cx_crop, cy_crop, device);
    return output;
}

torch::Tensor BuildViewMatrixFromCameraRt(const std::vector<float> &camera_rt, torch::Device device)
{
    if (camera_rt.size() < 16u)
    {
        return torch::eye(4, torch::TensorOptions().device(device)).transpose(0, 1).contiguous();
    }

    auto view_rm = torch::tensor(
        {
            {camera_rt[0], camera_rt[1], camera_rt[2], camera_rt[3]},
            {camera_rt[4], camera_rt[5], camera_rt[6], camera_rt[7]},
            {camera_rt[8], camera_rt[9], camera_rt[10], camera_rt[11]},
            {camera_rt[12], camera_rt[13], camera_rt[14], camera_rt[15]},
        },
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    return view_rm.transpose(0, 1).contiguous();
}

torch::Tensor CameraPositionFromViewMatrix(const torch::Tensor &view_mat)
{
    using torch::indexing::Slice;

    auto view_rm = view_mat.transpose(0, 1).contiguous();
    auto view_inv = torch::inverse(view_rm);
    return view_inv.index({Slice(0, 3), 3}).contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height,
                                                                       torch::Device device)
{
    const float n = 0.01f;
    const float f = 100.0f;
    const float tan_fovx = (static_cast<float>(width) * 0.5f) / std::max(focal, 1e-6f);
    const float tan_fovy = (static_cast<float>(height) * 0.5f) / std::max(focal, 1e-6f);

    auto view = torch::eye(4, device);
    auto proj = torch::zeros({4, 4}, device);
    proj[0][0] = 1.0f / tan_fovx;
    proj[1][1] = 1.0f / tan_fovy;
    proj[2][2] = f / (f - n);
    proj[2][3] = -(f * n) / (f - n);
    proj[3][2] = 1.0f;

    return {view.transpose(0, 1).contiguous(), proj.transpose(0, 1).contiguous(), tan_fovx, tan_fovy};
}

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float focal, int width, int height,
                                                                       float cx, float cy, torch::Device device)
{
    return BuildProjection(focal, focal, width, height, cx, cy, device);
}

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float fx, float fy,
                                                                       int width, int height,
                                                                       float cx, float cy, torch::Device device)
{
    const float n = 0.01f;
    const float f = 100.0f;
    const float tan_fovx = (static_cast<float>(width) * 0.5f) / std::max(fx, 1e-6f);
    const float tan_fovy = (static_cast<float>(height) * 0.5f) / std::max(fy, 1e-6f);

    auto view = torch::eye(4, device);
    auto proj = torch::zeros({4, 4}, device);
    proj[0][0] = 1.0f / tan_fovx;
    proj[1][1] = 1.0f / tan_fovy;
    proj[0][2] = (2.0f * cx - static_cast<float>(width)) / std::max(static_cast<float>(width), 1e-6f);
    proj[1][2] = (2.0f * cy - static_cast<float>(height)) / std::max(static_cast<float>(height), 1e-6f);
    proj[2][2] = f / (f - n);
    proj[2][3] = -(f * n) / (f - n);
    proj[3][2] = 1.0f;

    return {view.transpose(0, 1).contiguous(), proj.transpose(0, 1).contiguous(), tan_fovx, tan_fovy};
}

torch::Tensor ComputeTriFrames(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C)
{
    auto X = torch::nn::functional::normalize(B - A, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto N = torch::cross(B - A, C - A, 1);
    N = torch::nn::functional::normalize(N, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto Y = torch::cross(N, X, 1);
    return torch::stack({X, Y, N}, 2);
}
torch::Tensor MatrixToQuat(const torch::Tensor &rot_mat)
{
    using torch::indexing::Slice;

    // 1. Extract all 9 elements of the rotation matrices
    auto m00 = rot_mat.index({Slice(), 0, 0});
    auto m01 = rot_mat.index({Slice(), 0, 1});
    auto m02 = rot_mat.index({Slice(), 0, 2});

    auto m10 = rot_mat.index({Slice(), 1, 0});
    auto m11 = rot_mat.index({Slice(), 1, 1});
    auto m12 = rot_mat.index({Slice(), 1, 2});

    auto m20 = rot_mat.index({Slice(), 2, 0});
    auto m21 = rot_mat.index({Slice(), 2, 1});
    auto m22 = rot_mat.index({Slice(), 2, 2});

    auto tr = m00 + m11 + m22;

    // 2. Define the exact same masks, but keep them as booleans
    auto mask1 = tr > 0.0f;
    auto mask2 = (~mask1) & (m00 > m11) & (m00 > m22);
    auto mask3 = (~mask1) & (~mask2) & (m11 > m22);
    // mask4 is implicitly whatever is left

    // Expand masks for torch::where broadcasting [N] -> [N, 1]
    auto m1_exp = mask1.unsqueeze(1);
    auto m2_exp = mask2.unsqueeze(1);
    auto m3_exp = mask3.unsqueeze(1);

    // Safe epsilon to prevent NaN in sqrt or division by zero in unselected branches
    const float eps = 1e-6f;

    // --- Case 1 ---
    auto S1 = torch::sqrt(torch::clamp_min(tr + 1.0f, eps)) * 2.0f;
    auto q1 = torch::stack({
        0.25f * S1,
        (m21 - m12) / S1,
        (m02 - m20) / S1,
        (m10 - m01) / S1
    }, 1);

    // --- Case 2 ---
    auto S2 = torch::sqrt(torch::clamp_min(1.0f + m00 - m11 - m22, eps)) * 2.0f;
    auto q2 = torch::stack({
        (m21 - m12) / S2,
        0.25f * S2,
        (m01 + m10) / S2,
        (m02 + m20) / S2
    }, 1);

    // --- Case 3 ---
    auto S3 = torch::sqrt(torch::clamp_min(1.0f + m11 - m00 - m22, eps)) * 2.0f;
    auto q3 = torch::stack({
        (m02 - m20) / S3,
        (m01 + m10) / S3,
        0.25f * S3,
        (m12 + m21) / S3
    }, 1);

    // --- Case 4 ---
    auto S4 = torch::sqrt(torch::clamp_min(1.0f + m22 - m00 - m11, eps)) * 2.0f;
    auto q4 = torch::stack({
        (m10 - m01) / S4,
        (m02 + m20) / S4,
        (m12 + m21) / S4,
        0.25f * S4
    }, 1);

    // 3. Blend everything together using nested torch::where
    auto q_final = torch::where(m1_exp, q1,
                        torch::where(m2_exp, q2,
                            torch::where(m3_exp, q3, q4)));

    // 4. Normalize and return
    return torch::nn::functional::normalize(q_final, torch::nn::functional::NormalizeFuncOptions().dim(1));
}


torch::Tensor QuatMultiply(const torch::Tensor &p, const torch::Tensor &q)
{
    auto pw = p.select(1, 0);
    auto px = p.select(1, 1);
    auto py = p.select(1, 2);
    auto pz = p.select(1, 3);
    auto qw = q.select(1, 0);
    auto qx = q.select(1, 1);
    auto qy = q.select(1, 2);
    auto qz = q.select(1, 3);

    auto w = pw * qw - px * qx - py * qy - pz * qz;
    auto x = pw * qx + px * qw + py * qz - pz * qy;
    auto y = pw * qy - px * qz + py * qw + pz * qx;
    auto z = pw * qz + px * qy - py * qx + pz * qw;

    return torch::stack({w, x, y, z}, 1);
}

torch::Tensor QuatToMat3(const torch::Tensor &q, torch::Device device)
{
    (void)device;
    auto qn = torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().dim(1));

    auto w = qn.slice(1, 0, 1);
    auto x = qn.slice(1, 1, 2);
    auto y = qn.slice(1, 2, 3);
    auto z = qn.slice(1, 3, 4);

    auto x2 = x * x;
    auto y2 = y * y;
    auto z2 = z * z;
    auto xy = x * y;
    auto xz = x * z;
    auto yz = y * z;
    auto wx = w * x;
    auto wy = w * y;
    auto wz = w * z;

    auto r00 = 1.0f - 2.0f * (y2 + z2);
    auto r01 = 2.0f * (xy - wz);
    auto r02 = 2.0f * (xz + wy);
    auto r10 = 2.0f * (xy + wz);
    auto r11 = 1.0f - 2.0f * (x2 + z2);
    auto r12 = 2.0f * (yz - wx);
    auto r20 = 2.0f * (xz - wy);
    auto r21 = 2.0f * (yz + wx);
    auto r22 = 1.0f - 2.0f * (x2 + y2);

    return torch::cat({r00, r01, r02, r10, r11, r12, r20, r21, r22}, 1).view({-1, 3, 3});
}

torch::Tensor RotateSH(const torch::Tensor &sh, const torch::Tensor &rotations)
{
    if (!sh.defined() || sh.dim() < 3 || sh.size(0) == 0)
        return sh;
    if (sh.size(1) < 4)
        return sh;

    auto rot_mats = QuatToMat3(rotations, sh.device());
    auto sh_dc = sh.slice(1, 0, 1);    // [N,1,3]
    auto sh_y = sh.slice(1, 1, 2);     // [N,1,3]
    auto sh_z = sh.slice(1, 2, 3);     // [N,1,3]
    auto sh_x = sh.slice(1, 3, 4);     // [N,1,3]

    // Preserve legacy convention used by training/viewers:
    // coeff1->Y, coeff2->Z, coeff3->X with a sign flip on X output.
    auto sh_vec = torch::cat({sh_x, sh_y, sh_z}, 1); // [N,3,3]
    auto rotated = torch::bmm(rot_mats, sh_vec);     // [N,3,3]

    auto out_y = rotated.slice(1, 1, 2);
    auto out_z = rotated.slice(1, 2, 3);
    auto out_x = -rotated.slice(1, 0, 1);
    auto sh_rot_band1 = torch::cat({out_y, out_z, out_x}, 1); // [N,3,3]

    if (sh.size(1) > 4)
    {
        auto sh_high_zero = torch::zeros({sh.size(0), sh.size(1) - 4, 3}, sh.options());
        return torch::cat({sh_dc, sh_rot_band1, sh_high_zero}, 1);
    }

    return torch::cat({sh_dc, sh_rot_band1}, 1);
}
