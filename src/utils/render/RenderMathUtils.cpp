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

    auto mask1 = tr > 0;
    auto S1 = torch::sqrt(torch::clamp_min(tr.index({mask1}) + 1.0f, 0.0f)) * 2.0f;
    q.index_put_({mask1, 0}, 0.25f * S1);
    q.index_put_({mask1, 1}, (rot_mat.index({mask1, 2, 1}) - rot_mat.index({mask1, 1, 2})) / S1);
    q.index_put_({mask1, 2}, (rot_mat.index({mask1, 0, 2}) - rot_mat.index({mask1, 2, 0})) / S1);
    q.index_put_({mask1, 3}, (rot_mat.index({mask1, 1, 0}) - rot_mat.index({mask1, 0, 1})) / S1);

    auto mask2 = (~mask1) & (m00 > m11) & (m00 > m22);
    auto S2 = torch::sqrt(torch::clamp_min(1.0f + m00.index({mask2}) - m11.index({mask2}) - m22.index({mask2}),
                                           0.0f)) *
              2.0f;
    q.index_put_({mask2, 0}, (rot_mat.index({mask2, 2, 1}) - rot_mat.index({mask2, 1, 2})) / S2);
    q.index_put_({mask2, 1}, 0.25f * S2);
    q.index_put_({mask2, 2}, (rot_mat.index({mask2, 0, 1}) + rot_mat.index({mask2, 1, 0})) / S2);
    q.index_put_({mask2, 3}, (rot_mat.index({mask2, 0, 2}) + rot_mat.index({mask2, 2, 0})) / S2);

    auto mask3 = (~mask1) & (~mask2) & (m11 > m22);
    auto S3 = torch::sqrt(torch::clamp_min(1.0f + m11.index({mask3}) - m00.index({mask3}) - m22.index({mask3}),
                                           0.0f)) *
              2.0f;
    q.index_put_({mask3, 0}, (rot_mat.index({mask3, 0, 2}) - rot_mat.index({mask3, 2, 0})) / S3);
    q.index_put_({mask3, 1}, (rot_mat.index({mask3, 0, 1}) + rot_mat.index({mask3, 1, 0})) / S3);
    q.index_put_({mask3, 2}, 0.25f * S3);
    q.index_put_({mask3, 3}, (rot_mat.index({mask3, 1, 2}) + rot_mat.index({mask3, 2, 1})) / S3);

    auto mask4 = (~mask1) & (~mask2) & (~mask3);
    auto S4 = torch::sqrt(torch::clamp_min(1.0f + m22.index({mask4}) - m00.index({mask4}) - m11.index({mask4}),
                                           0.0f)) *
              2.0f;
    q.index_put_({mask4, 0}, (rot_mat.index({mask4, 1, 0}) - rot_mat.index({mask4, 0, 1})) / S4);
    q.index_put_({mask4, 1}, (rot_mat.index({mask4, 0, 2}) + rot_mat.index({mask4, 2, 0})) / S4);
    q.index_put_({mask4, 2}, (rot_mat.index({mask4, 1, 2}) + rot_mat.index({mask4, 2, 1})) / S4);
    q.index_put_({mask4, 3}, 0.25f * S4);
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
    auto qn = torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto w = qn.select(1, 0);
    auto x = qn.select(1, 1);
    auto y = qn.select(1, 2);
    auto z = qn.select(1, 3);

    auto ww = w * w;
    auto xx = x * x;
    auto yy = y * y;
    auto zz = z * z;
    auto wx = w * x;
    auto wy = w * y;
    auto wz = w * z;
    auto xy = x * y;
    auto xz = x * z;
    auto yz = y * z;

    auto m00 = ww + xx - yy - zz;
    auto m01 = 2.0f * (xy - wz);
    auto m02 = 2.0f * (xz + wy);
    auto m10 = 2.0f * (xy + wz);
    auto m11 = ww - xx + yy - zz;
    auto m12 = 2.0f * (yz - wx);
    auto m20 = 2.0f * (xz - wy);
    auto m21 = 2.0f * (yz + wx);
    auto m22 = ww - xx - yy + zz;

    auto row0 = torch::stack({m00, m01, m02}, 1);
    auto row1 = torch::stack({m10, m11, m12}, 1);
    auto row2 = torch::stack({m20, m21, m22}, 1);
    return torch::stack({row0, row1, row2}, 1).to(device);
}

torch::Tensor RotateSH(const torch::Tensor &sh, const torch::Tensor &rotations)
{
    if (!sh.defined() || sh.dim() < 3 || sh.size(1) < 4)
        return sh;

    auto rots_norm = torch::nn::functional::normalize(rotations,
        torch::nn::functional::NormalizeFuncOptions().dim(1));

    auto rot_mats = QuatToMat3(rots_norm, sh.device());

    using torch::indexing::Slice;

    auto sh_d1 = sh.index({Slice(), Slice(1, 4), Slice()});
    auto sh_y = sh_d1.index({Slice(), 0, Slice()});
    auto sh_z = sh_d1.index({Slice(), 1, Slice()});
    auto sh_x = sh_d1.index({Slice(), 2, Slice()});

    auto sh_vec = torch::stack({sh_x, sh_y, sh_z}, 1);

    auto rotated_vec = torch::bmm(rot_mats, sh_vec);

    auto out_sh = sh.clone();

    out_sh.index_put_({Slice(), 1, Slice()}, rotated_vec.index({Slice(), 1, Slice()}));
    out_sh.index_put_({Slice(), 2, Slice()}, rotated_vec.index({Slice(), 2, Slice()}));
    out_sh.index_put_({Slice(), 3, Slice()}, -rotated_vec.index({Slice(), 0, Slice()}));
    if (sh.size(1) > 4)
    {
        out_sh.index_put_({Slice(), Slice(4, torch::indexing::None), Slice()}, 0.0f);
    }

    return out_sh;
}
