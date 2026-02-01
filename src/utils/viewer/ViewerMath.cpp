#include "utils/viewer/ViewerMath.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kShC1 = 0.4886025119029199f;
constexpr float kShC2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constexpr float kShC3[7] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
} // namespace

Vec3 Normalize(const Vec3 &v)
{
    const float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 <= 0.0f)
        return v;
    const float inv_len = 1.0f / std::sqrt(len2);
    return {v.x * inv_len, v.y * inv_len, v.z * inv_len};
}

Vec3 LoadCoeff(const float *sh, int idx)
{
    const float *p = sh + idx * 3;
    return {p[0], p[1], p[2]};
}

Vec3 AddScaled(const Vec3 &base, const Vec3 &coeff, float scale)
{
    return {base.x + coeff.x * scale, base.y + coeff.y * scale, base.z + coeff.z * scale};
}

Vec3 ComputeColorFromSH(int degree, const float *sh, const Vec3 &pos, const Vec3 &cam)
{
    Vec3 dir{pos.x - cam.x, pos.y - cam.y, pos.z - cam.z};
    dir = Normalize(dir);

    Vec3 result = LoadCoeff(sh, 0);
    result = {result.x * kShC0, result.y * kShC0, result.z * kShC0};

    if (degree > 0)
    {
        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        result = AddScaled(result, LoadCoeff(sh, 1), kShC1 * y);
        result = AddScaled(result, LoadCoeff(sh, 2), kShC1 * z);
        result = AddScaled(result, LoadCoeff(sh, 3), kShC1 * x);

        if (degree > 1)
        {
            const float xx = x * x;
            const float yy = y * y;
            const float zz = z * z;
            const float xy = x * y;
            const float yz = y * z;
            const float xz = x * z;
            result = AddScaled(result, LoadCoeff(sh, 4), kShC2[0] * xy);
            result = AddScaled(result, LoadCoeff(sh, 5), kShC2[1] * yz);
            result = AddScaled(result, LoadCoeff(sh, 6), kShC2[2] * (2.0f * zz - xx - yy));
            result = AddScaled(result, LoadCoeff(sh, 7), kShC2[3] * xz);
            result = AddScaled(result, LoadCoeff(sh, 8), kShC2[4] * (xx - yy));

            if (degree > 2)
            {
                result = AddScaled(result, LoadCoeff(sh, 9), kShC3[0] * y * (3.0f * xx - yy));
                result = AddScaled(result, LoadCoeff(sh, 10), kShC3[1] * xy * z);
                result = AddScaled(result, LoadCoeff(sh, 11), kShC3[2] * y * (4.0f * zz - xx - yy));
                result = AddScaled(result, LoadCoeff(sh, 12), kShC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy));
                result = AddScaled(result, LoadCoeff(sh, 13), kShC3[4] * x * (4.0f * zz - xx - yy));
                result = AddScaled(result, LoadCoeff(sh, 14), kShC3[5] * z * (xx - yy));
                result = AddScaled(result, LoadCoeff(sh, 15), kShC3[6] * x * (xx - 3.0f * yy));
            }
        }
    }
    result.x += 0.5f;
    result.y += 0.5f;
    result.z += 0.5f;
    result.x = std::max(result.x, 0.0f);
    result.y = std::max(result.y, 0.0f);
    result.z = std::max(result.z, 0.0f);
    return result;
}

Mat4 RotationYawPitch(float yaw, float pitch)
{
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);

    Mat4 ry = Identity();
    ry.m[0] = cy;
    ry.m[2] = -sy;
    ry.m[8] = sy;
    ry.m[10] = cy;

    Mat4 rx = Identity();
    rx.m[5] = cp;
    rx.m[6] = sp;
    rx.m[9] = -sp;
    rx.m[10] = cp;

    return Multiply(rx, ry);
}

Mat4 Identity()
{
    Mat4 out{};
    out.m[0] = out.m[5] = out.m[10] = out.m[15] = 1.0f;
    return out;
}

Mat4 Multiply(const Mat4 &a, const Mat4 &b)
{
    Mat4 out{};
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            out.m[col * 4 + row] = sum;
        }
    }
    return out;
}

Mat4 Perspective(float fovy_deg, float aspect, float z_near, float z_far)
{
    const float fovy = fovy_deg * 3.14159265f / 180.0f;
    const float tan_fovy = std::tan(fovy * 0.5f);
    const float tan_fovx = tan_fovy * aspect;
    Mat4 out{};
    // Match the rasterizer's +Z forward projection convention.
    out.m[0] = 1.0f / tan_fovx;
    out.m[5] = 1.0f / tan_fovy;
    out.m[10] = z_far / (z_far - z_near);
    out.m[11] = 1.0f;
    out.m[14] = -(z_far * z_near) / (z_far - z_near);
    return out;
}

Mat4 LookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up)
{
    Vec3 f{center.x - eye.x, center.y - eye.y, center.z - eye.z};
    const float f_len = std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
    if (f_len > 0.0f)
    {
        f.x /= f_len;
        f.y /= f_len;
        f.z /= f_len;
    }
    Vec3 s{f.y * up.z - f.z * up.y, f.z * up.x - f.x * up.z, f.x * up.y - f.y * up.x};
    const float s_len = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    if (s_len > 0.0f)
    {
        s.x /= s_len;
        s.y /= s_len;
        s.z /= s_len;
    }
    Vec3 u{s.y * f.z - s.z * f.y, s.z * f.x - s.x * f.z, s.x * f.y - s.y * f.x};

    Mat4 out = Identity();
    out.m[0] = s.x;
    out.m[4] = s.y;
    out.m[8] = s.z;
    out.m[1] = u.x;
    out.m[5] = u.y;
    out.m[9] = u.z;
    // Use +Z forward to match the rasterizer's projection convention.
    out.m[2] = f.x;
    out.m[6] = f.y;
    out.m[10] = f.z;
    out.m[12] = -(s.x * eye.x + s.y * eye.y + s.z * eye.z);
    out.m[13] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
    out.m[14] = -(f.x * eye.x + f.y * eye.y + f.z * eye.z);
    return out;
}

torch::Tensor Mat4ToTensorRowMajor(const Mat4 &m, torch::Device device)
{
    auto t = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            // Mat4 is column-major; map to logical row/col here.
            t[r][c] = m.m[c * 4 + r];
        }
    }
    return t;
}

Vec3 RotatePoint(const Vec3 &p, const Vec3 &center, float yaw, float pitch)
{
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);

    const float x = p.x - center.x;
    const float y = p.y - center.y;
    const float z = p.z - center.z;

    const float x1 = cy * x + sy * z;
    const float z1 = -sy * x + cy * z;

    const float y2 = cp * y - sp * z1;
    const float z2 = sp * y + cp * z1;

    return {x1 + center.x, y2 + center.y, z2 + center.z};
}

Vec3 AddVec3(const Vec3 &a, const Vec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 SubVec3(const Vec3 &a, const Vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 ScaleVec3(const Vec3 &v, float s)
{
    return {v.x * s, v.y * s, v.z * s};
}

Vec3 CrossVec3(const Vec3 &a, const Vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3 CameraPosFromView(const Mat4 &view)
{
    // view is world->camera, column-major. Camera position is -R^T * t.
    const Vec3 t{view.m[12], view.m[13], view.m[14]};
    return {
        -(view.m[0] * t.x + view.m[1] * t.y + view.m[2] * t.z),
        -(view.m[4] * t.x + view.m[5] * t.y + view.m[6] * t.z),
        -(view.m[8] * t.x + view.m[9] * t.y + view.m[10] * t.z)};
}

Mat4 Mat4FromQuatTranslation(const Quat &q, const Vec3 &t)
{
    const float w = q.w;
    const float x = q.x;
    const float y = q.y;
    const float z = q.z;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;

    Mat4 out = Identity();
    // Column-major rotation.
    out.m[0] = 1.0f - 2.0f * (yy + zz);
    out.m[1] = 2.0f * (xy + wz);
    out.m[2] = 2.0f * (xz - wy);

    out.m[4] = 2.0f * (xy - wz);
    out.m[5] = 1.0f - 2.0f * (xx + zz);
    out.m[6] = 2.0f * (yz + wx);

    out.m[8] = 2.0f * (xz + wy);
    out.m[9] = 2.0f * (yz - wx);
    out.m[10] = 1.0f - 2.0f * (xx + yy);

    out.m[12] = t.x;
    out.m[13] = t.y;
    out.m[14] = t.z;
    return out;
}

torch::Tensor QuatToMat3(const Quat &q, torch::Device device)
{
    const float w = q.w;
    const float x = q.x;
    const float y = q.y;
    const float z = q.z;

    auto t = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    t[0][0] = 1.0f - 2.0f * (y * y + z * z);
    t[0][1] = 2.0f * (x * y - w * z);
    t[0][2] = 2.0f * (x * z + w * y);
    t[1][0] = 2.0f * (x * y + w * z);
    t[1][1] = 1.0f - 2.0f * (x * x + z * z);
    t[1][2] = 2.0f * (y * z - w * x);
    t[2][0] = 2.0f * (x * z - w * y);
    t[2][1] = 2.0f * (y * z + w * x);
    t[2][2] = 1.0f - 2.0f * (x * x + y * y);
    return t;
}

torch::Tensor MulQuatTensor(const torch::Tensor &a, const torch::Tensor &b)
{
    auto aw = a.index({torch::indexing::Slice(), 0});
    auto ax = a.index({torch::indexing::Slice(), 1});
    auto ay = a.index({torch::indexing::Slice(), 2});
    auto az = a.index({torch::indexing::Slice(), 3});
    auto bw = b.index({torch::indexing::Slice(), 0});
    auto bx = b.index({torch::indexing::Slice(), 1});
    auto by = b.index({torch::indexing::Slice(), 2});
    auto bz = b.index({torch::indexing::Slice(), 3});

    auto w = aw * bw - ax * bx - ay * by - az * bz;
    auto x = aw * bx + ax * bw + ay * bz - az * by;
    auto y = aw * by - ax * bz + ay * bw + az * bx;
    auto z = aw * bz + ax * by - ay * bx + az * bw;
    return torch::stack({w, x, y, z}, 1);
}

Quat NormalizeQuat(const Quat &q)
{
    const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n <= 0.0f)
        return q;
    const float inv = 1.0f / n;
    return {q.w * inv, q.x * inv, q.y * inv, q.z * inv};
}

Quat MulQuat(const Quat &a, const Quat &b)
{
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

Quat ConjugateQuat(const Quat &q)
{
    return {q.w, -q.x, -q.y, -q.z};
}

Quat AxisAngleQuat(const Vec3 &axis, float angle)
{
    const float half = 0.5f * angle;
    const float s = std::sin(half);
    return NormalizeQuat({std::cos(half), axis.x * s, axis.y * s, axis.z * s});
}

Vec3 RotatePointQuat(const Vec3 &p, const Vec3 &center, const Quat &q)
{
    Vec3 v{p.x - center.x, p.y - center.y, p.z - center.z};
    Quat qv{0.0f, v.x, v.y, v.z};
    Quat qn = NormalizeQuat(q);
    Quat res = MulQuat(MulQuat(qn, qv), ConjugateQuat(qn));
    return {center.x + res.x, center.y + res.y, center.z + res.z};
}

Vec3 RotateVec3Quat(const Vec3 &v, const Quat &q)
{
    return RotatePointQuat(v, {0.0f, 0.0f, 0.0f}, q);
}

Quat YawPitchQuat(float yaw, float pitch)
{
    const float hy = 0.5f * yaw;
    const float hp = 0.5f * pitch;
    const float cy = std::cos(hy);
    const float sy = std::sin(hy);
    const float cp = std::cos(hp);
    const float sp = std::sin(hp);

    Quat qy{cy, 0.0f, sy, 0.0f};
    Quat qx{cp, sp, 0.0f, 0.0f};
    return NormalizeQuat(MulQuat(qy, qx));
}

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float fovy_deg, int width, int height,
                                                                       torch::Device device)
{
    const float n = 0.01f;
    const float f = 100.0f;
    const float fovy = fovy_deg * 3.14159265f / 180.0f;
    const float tan_fovy = std::tan(fovy * 0.5f);
    const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    const float tan_fovx = tan_fovy * aspect;

    auto view = torch::eye(4, device);
    auto proj = torch::zeros({4, 4}, device);
    proj[0][0] = 1.0f / tan_fovx;
    proj[1][1] = 1.0f / tan_fovy;
    proj[2][2] = f / (f - n);
    proj[2][3] = -(f * n) / (f - n);
    proj[3][2] = 1.0f;

    return {view.transpose(0, 1).contiguous(), proj.transpose(0, 1).contiguous(), tan_fovx, tan_fovy};
}

Vec3 TransformPoint(const Mat4 &m, const Vec3 &p)
{
    return {
        m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
        m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14]};
}
