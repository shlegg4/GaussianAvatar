#include "utils/render/RenderMathUtils.h"

#include <algorithm>

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
    const float n = 0.01f;
    const float f = 100.0f;
    const float tan_fovx = (static_cast<float>(width) * 0.5f) / std::max(focal, 1e-6f);
    const float tan_fovy = (static_cast<float>(height) * 0.5f) / std::max(focal, 1e-6f);

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
    auto m00 = rot_mat.index({Slice(), 0, 0});
    auto m11 = rot_mat.index({Slice(), 1, 1});
    auto m22 = rot_mat.index({Slice(), 2, 2});
    auto tr = m00 + m11 + m22;

    const auto num = rot_mat.size(0);
    auto q = torch::zeros({num, 4}, rot_mat.options());

    // --- OPTIMIZATION: Removed if(mask.any().item<bool>()) checks ---
    // We execute the tensor operations blindly. 
    // If a mask is empty, the index operation is a fast GPU no-op 
    // and does not trigger a CPU sync.

    // Case 1: Trace > 0
    auto mask1 = tr > 0;
    {
        // Safe S calculation (masked values ignored by index_put_)
        auto S = torch::sqrt(torch::clamp_min(tr.index({mask1}) + 1.0f, 0.0f)) * 2.0f;
        q.index_put_({mask1, 0}, 0.25f * S);
        q.index_put_({mask1, 1}, (rot_mat.index({mask1, 2, 1}) - rot_mat.index({mask1, 1, 2})) / S);
        q.index_put_({mask1, 2}, (rot_mat.index({mask1, 0, 2}) - rot_mat.index({mask1, 2, 0})) / S);
        q.index_put_({mask1, 3}, (rot_mat.index({mask1, 1, 0}) - rot_mat.index({mask1, 0, 1})) / S);
    }

    // Case 2: m00 is max
    auto mask2 = (~mask1) & (m00 > m11) & (m00 > m22);
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m00.index({mask2}) - m11.index({mask2}) - m22.index({mask2}), 0.0f)) * 2.0f;
        q.index_put_({mask2, 0}, (rot_mat.index({mask2, 2, 1}) - rot_mat.index({mask2, 1, 2})) / S);
        q.index_put_({mask2, 1}, 0.25f * S);
        q.index_put_({mask2, 2}, (rot_mat.index({mask2, 0, 1}) + rot_mat.index({mask2, 1, 0})) / S);
        q.index_put_({mask2, 3}, (rot_mat.index({mask2, 0, 2}) + rot_mat.index({mask2, 2, 0})) / S);
    }

    // Case 3: m11 is max
    auto mask3 = (~mask1) & (~mask2) & (m11 > m22);
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m11.index({mask3}) - m00.index({mask3}) - m22.index({mask3}), 0.0f)) * 2.0f;
        q.index_put_({mask3, 0}, (rot_mat.index({mask3, 0, 2}) - rot_mat.index({mask3, 2, 0})) / S);
        q.index_put_({mask3, 1}, (rot_mat.index({mask3, 0, 1}) + rot_mat.index({mask3, 1, 0})) / S);
        q.index_put_({mask3, 2}, 0.25f * S);
        q.index_put_({mask3, 3}, (rot_mat.index({mask3, 1, 2}) + rot_mat.index({mask3, 2, 1})) / S);
    }

    // Case 4: m22 is max
    auto mask4 = (~mask1) & (~mask2) & (~mask3);
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m22.index({mask4}) - m00.index({mask4}) - m11.index({mask4}), 0.0f)) * 2.0f;
        q.index_put_({mask4, 0}, (rot_mat.index({mask4, 1, 0}) - rot_mat.index({mask4, 0, 1})) / S);
        q.index_put_({mask4, 1}, (rot_mat.index({mask4, 0, 2}) + rot_mat.index({mask4, 2, 0})) / S);
        q.index_put_({mask4, 2}, (rot_mat.index({mask4, 1, 2}) + rot_mat.index({mask4, 2, 1})) / S);
        q.index_put_({mask4, 3}, 0.25f * S);
    }

    return torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().dim(1));
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
