#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "GaussianRasterizer.h"
#include "SharedGaussian.h"

namespace
{
struct ViewerOptions
{
    std::string shm_name = "GaussianAvatarShared";
    int width = 1280;
    int height = 720;
    float point_size = 1.0f;
    int retry_ms = 500;
};

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Mat4
{
    float m[16] = {};
};

Mat4 Identity();
Mat4 Multiply(const Mat4 &a, const Mat4 &b);

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
        result = AddScaled(result, LoadCoeff(sh, 1), -kShC1 * y);
        result = AddScaled(result, LoadCoeff(sh, 2), kShC1 * z);
        result = AddScaled(result, LoadCoeff(sh, 3), -kShC1 * x);

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

struct Quat
{
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

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

struct CameraState
{
    float yaw = 0.0f;
    float pitch = 0.0f;
    Vec3 position{0.0f, 0.0f, 2.5f};
    Vec3 target{0.0f, 0.0f, 0.0f};
    Quat model_rot{1.0f, 0.0f, 0.0f, 0.0f};
    Vec3 model_offset{0.0f, 0.0f, 0.0f};
    bool rotating = false;
    bool panning = false;
    double last_x = 0.0;
    double last_y = 0.0;
};

CameraState g_camera{};

struct BoundsState
{
    bool valid = false;
    Vec3 min;
    Vec3 max;
};

BoundsState g_bounds{};

void MouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        g_camera.rotating = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        g_camera.panning = (action == GLFW_PRESS);
    glfwGetCursorPos(window, &g_camera.last_x, &g_camera.last_y);
}

void CursorPosCallback(GLFWwindow * /*window*/, double xpos, double ypos)
{
    const double dx = xpos - g_camera.last_x;
    const double dy = ypos - g_camera.last_y;
    g_camera.last_x = xpos;
    g_camera.last_y = ypos;
    if (g_camera.rotating)
    {
        const float yaw_delta = static_cast<float>(dx) * 0.005f;
        const Quat qy = AxisAngleQuat({0.0f, 1.0f, 0.0f}, yaw_delta);
        g_camera.model_rot = NormalizeQuat(MulQuat(qy, g_camera.model_rot));
    }
    if (g_camera.panning)
    {
        const float pan_scale = 0.005f;
        g_camera.model_offset.x += -static_cast<float>(dx) * pan_scale;
        g_camera.model_offset.y += static_cast<float>(dy) * pan_scale;
    }
}

void ScrollCallback(GLFWwindow * /*window*/, double /*xoffset*/, double yoffset)
{
    const float zoom = static_cast<float>(yoffset) * 0.2f;
    g_camera.model_offset.z += zoom;
}

void FrameCameraToBounds()
{
    if (!g_bounds.valid)
        return;
    Vec3 center{
        0.5f * (g_bounds.min.x + g_bounds.max.x),
        0.5f * (g_bounds.min.y + g_bounds.max.y),
        0.5f * (g_bounds.min.z + g_bounds.max.z)};
    Vec3 extents{
        0.5f * (g_bounds.max.x - g_bounds.min.x),
        0.5f * (g_bounds.max.y - g_bounds.min.y),
        0.5f * (g_bounds.max.z - g_bounds.min.z)};
    const float radius = std::max({extents.x, extents.y, extents.z, 0.1f});
    const float dist = radius * 3.0f;
    g_camera.target = center;
    g_camera.position = {center.x, center.y, center.z + dist};
}

bool ParseArgs(int argc, char **argv, ViewerOptions *options)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--shm" && i + 1 < argc)
        {
            options->shm_name = argv[++i];
        }
        else if (arg == "--width" && i + 1 < argc)
        {
            options->width = std::stoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            options->height = std::stoi(argv[++i]);
        }
        else if (arg == "--point-size" && i + 1 < argc)
        {
            options->point_size = std::stof(argv[++i]);
        }
        else if (arg == "--retry-ms" && i + 1 < argc)
        {
            options->retry_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h")
        {
            return false;
        }
        else
        {
            std::cerr << "Unknown arg: " << arg << std::endl;
            return false;
        }
    }
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    ViewerOptions options;
    if (!ParseArgs(argc, argv, &options))
    {
        std::cout << "Usage: gaussian_viewer [--shm <name>] [--width <int>] [--height <int>]"
                     " [--point-size <float>] [--retry-ms <int>]\n";
        return 0;
    }

    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW." << std::endl;
        return 1;
    }

    GLFWwindow *window = glfwCreateWindow(options.width, options.height, "Gaussian Avatar Viewer", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create window." << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetMouseButtonCallback(window, MouseButtonCallback);
    glfwSetCursorPosCallback(window, CursorPosCallback);
    glfwSetScrollCallback(window, ScrollCallback);

    if (!torch::cuda::is_available())
    {
        std::cerr << "CUDA Required for Rasterizer!" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    torch::Device device(torch::kCUDA);

    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    shared_gaussian::SharedGaussianReader reader;
    std::vector<float> points;
    uint32_t point_count = 0;
    uint32_t stride = 0;

    auto last_retry = std::chrono::steady_clock::now() - std::chrono::milliseconds(options.retry_ms);

    bool prev_x = false;
    bool prev_y = false;
    bool prev_z = false;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        const bool key_x = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
        const bool key_y = glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS;
        const bool key_z = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
        if (key_x && !prev_x)
        {
            g_camera.model_rot = NormalizeQuat(MulQuat(AxisAngleQuat({1.0f, 0.0f, 0.0f}, 1.57079633f),
                                                      g_camera.model_rot));
        }
        if (key_y && !prev_y)
        {
            g_camera.model_rot = NormalizeQuat(MulQuat(AxisAngleQuat({0.0f, 1.0f, 0.0f}, 1.57079633f),
                                                      g_camera.model_rot));
        }
        if (key_z && !prev_z)
        {
            g_camera.model_rot = NormalizeQuat(MulQuat(AxisAngleQuat({0.0f, 0.0f, 1.0f}, 1.57079633f),
                                                      g_camera.model_rot));
        }
        prev_x = key_x;
        prev_y = key_y;
        prev_z = key_z;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
        {
            FrameCameraToBounds();
        }
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            g_camera = CameraState{};
        }
        if (!reader.IsOpen())
        {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_retry).count() >= options.retry_ms)
            {
                last_retry = now;
                reader.Open(options.shm_name);
            }
        }
        uint32_t sh_degree = 0;
        if (reader.IsOpen())
        {
            if (const auto *header = reader.Header())
            {
                sh_degree = header->sh_degree;
            }
            uint64_t version = 0;
            uint64_t frame = 0;
            if (reader.ReadLatest(&points, &point_count, &stride, &frame, &version))
            {
                (void)frame;
            }
        }

        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.05f, 0.05f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const float fovy_deg = 45.0f;

        const uint32_t sh_coeffs = (sh_degree > 0) ? (sh_degree + 1) * (sh_degree + 1) : 0;
        const uint32_t base_stride = shared_gaussian::kSharedStrideFloats + 7u;
        const uint32_t sh_stride = base_stride + sh_coeffs * 3u;
        const bool use_sh = (sh_degree > 0) && (stride >= sh_stride);

        if (point_count > 0 && stride >= base_stride)
        {
            const float *ptr = points.data();
            std::vector<float> pos_data;
            std::vector<float> col_data;
            std::vector<float> opa_data;
            std::vector<float> scale_data;
            std::vector<float> rot_data;
            std::vector<float> sh_data;

            pos_data.resize(static_cast<size_t>(point_count) * 3u);
            col_data.resize(static_cast<size_t>(point_count) * 3u);
            opa_data.resize(static_cast<size_t>(point_count));
            scale_data.resize(static_cast<size_t>(point_count) * 3u);
            rot_data.resize(static_cast<size_t>(point_count) * 4u);
            if (use_sh)
            {
                sh_data.resize(static_cast<size_t>(point_count) * sh_coeffs * 3u);
            }

            Vec3 min_v{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity()};
            Vec3 max_v{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity()};
            for (uint32_t i = 0; i < point_count; ++i)
            {
                const float *p = ptr + static_cast<size_t>(i) * stride;
                const size_t base = static_cast<size_t>(i) * 3u;
                pos_data[base + 0] = p[0];
                pos_data[base + 1] = p[1];
                pos_data[base + 2] = p[2];
                min_v.x = std::min(min_v.x, p[0]);
                min_v.y = std::min(min_v.y, p[1]);
                min_v.z = std::min(min_v.z, p[2]);
                max_v.x = std::max(max_v.x, p[0]);
                max_v.y = std::max(max_v.y, p[1]);
                max_v.z = std::max(max_v.z, p[2]);
                col_data[base + 0] = p[3];
                col_data[base + 1] = p[4];
                col_data[base + 2] = p[5];
                opa_data[i] = p[6];
                scale_data[base + 0] = p[8];
                scale_data[base + 1] = p[9];
                scale_data[base + 2] = p[10];
                const size_t rot_base = static_cast<size_t>(i) * 4u;
                rot_data[rot_base + 0] = p[11];
                rot_data[rot_base + 1] = p[12];
                rot_data[rot_base + 2] = p[13];
                rot_data[rot_base + 3] = p[14];

                if (use_sh)
                {
                    const float *sh_ptr = p + base_stride;
                    const size_t sh_base = static_cast<size_t>(i) * sh_coeffs * 3u;
                    std::memcpy(sh_data.data() + sh_base, sh_ptr,
                                static_cast<size_t>(sh_coeffs) * 3u * sizeof(float));
                }
            }
            g_bounds.valid = true;
            g_bounds.min = min_v;
            g_bounds.max = max_v;

            torch::NoGradGuard no_grad;
            auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            auto means3D = torch::from_blob(pos_data.data(), {static_cast<int64_t>(point_count), 3}, opts)
                               .clone()
                               .to(device);
            auto colors = torch::from_blob(col_data.data(), {static_cast<int64_t>(point_count), 3}, opts)
                              .clone()
                              .to(device);
            auto opacities = torch::from_blob(opa_data.data(), {static_cast<int64_t>(point_count), 1}, opts)
                                 .clone()
                                 .to(device);
            auto scales = torch::from_blob(scale_data.data(), {static_cast<int64_t>(point_count), 3}, opts)
                              .clone()
                              .to(device);
            auto rotations = torch::from_blob(rot_data.data(), {static_cast<int64_t>(point_count), 4}, opts)
                                 .clone()
                                 .to(device);

            torch::Tensor sh_tensor = torch::zeros({0}, opts).to(device);
            if (use_sh)
            {
                sh_tensor = torch::from_blob(sh_data.data(),
                                             {static_cast<int64_t>(point_count), static_cast<int64_t>(sh_coeffs), 3},
                                             opts)
                                .clone()
                                .to(device);
                colors = torch::zeros({0}, opts).to(device);
            }

            const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
            const float fovy_rad = fovy_deg * 3.14159265f / 180.0f;
            const float tan_fovy = std::tan(fovy_rad * 0.5f);
            const float tan_fovx = tan_fovy * aspect;

            Quat q_model = NormalizeQuat(g_camera.model_rot);
            Mat4 view_mat = LookAt({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f});
            Mat4 proj_mat = Perspective(fovy_deg, aspect, 0.01f, 100.0f);
            Mat4 view_proj = Multiply(proj_mat, view_mat);

            torch::Tensor view_t = Mat4ToTensorRowMajor(view_mat, device).transpose(0, 1).contiguous();
            torch::Tensor proj_t = Mat4ToTensorRowMajor(view_proj, device).transpose(0, 1).contiguous();
            auto cam_pos = torch::tensor({0.0f, 0.0f, 0.0f}, opts).to(device);

            auto pivot = torch::tensor({g_camera.target.x, g_camera.target.y, g_camera.target.z}, opts).to(device);
            auto rot = QuatToMat3(q_model, device);
            means3D = (means3D - pivot).matmul(rot.transpose(0, 1)) + pivot;
            {
                auto q_model_t = torch::tensor({q_model.w, q_model.x, q_model.y, q_model.z}, opts)
                                     .to(device)
                                     .unsqueeze(0)
                                     .expand({rotations.size(0), 4});
                rotations = MulQuatTensor(q_model_t, rotations);
                auto rot_norm = torch::norm(rotations, 2, 1, true).clamp_min(1e-8f);
                rotations = rotations / rot_norm;
            }
            if (g_bounds.valid)
            {
                Vec3 center{
                    0.5f * (g_bounds.min.x + g_bounds.max.x),
                    0.5f * (g_bounds.min.y + g_bounds.max.y),
                    0.5f * (g_bounds.min.z + g_bounds.max.z)};
                Vec3 extents{
                    0.5f * (g_bounds.max.x - g_bounds.min.x),
                    0.5f * (g_bounds.max.y - g_bounds.min.y),
                    0.5f * (g_bounds.max.z - g_bounds.min.z)};
                const float radius = std::max({extents.x, extents.y, extents.z, 0.1f});
                const float dist = radius * 3.0f;
                const float offset_z = dist - center.z;
                auto base_offset = torch::tensor({0.0f, 0.0f, offset_z}, opts).to(device);
                auto user_offset = torch::tensor({g_camera.model_offset.x, g_camera.model_offset.y,
                                                  g_camera.model_offset.z},
                                                 opts)
                                      .to(device);
                means3D = means3D + base_offset + user_offset;
            }

            float scale_modifier = 1.0f;
            if (const auto *header = reader.Header())
            {
                scale_modifier = header->render_scale_modifier;
            }
            scale_modifier *= options.point_size;

            auto image = GaussianRasterizer::apply(
                means3D,
                colors,
                opacities,
                scales,
                rotations,
                scale_modifier,
                view_t,
                proj_t,
                tan_fovx,
                tan_fovy,
                height,
                width,
                sh_tensor,
                use_sh ? static_cast<int>(sh_degree) : 0,
                cam_pos,
                false);

            auto img = image.detach().clamp(0.0f, 1.0f).mul(255.0f).to(torch::kU8).cpu();
            img = img.permute({1, 2, 0}).contiguous();

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glRasterPos2f(-1.0f, -1.0f);
            glPixelZoom(1.0f, 1.0f);
            glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data_ptr<uint8_t>());
        }

        glfwSwapBuffers(window);
    }

    reader.Close();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
