#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "GaussianRasterizer.h"
#include "SharedGaussian.h"
#include "utils/viewer/ViewerMath.h"

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
    bool prev_r = false;
    bool randomize_colors = false;

    float render_scale_modifier = 1.0f;
    uint64_t last_version = 0;

    // --- Persistent GPU State ---
    torch::Tensor raw_gpu;
    torch::Tensor means3D, colors, opacities, scales, rotations, sh_tensor;
    torch::Tensor random_colors;
    int64_t random_colors_count = 0;
    bool gpu_data_dirty = false;
    const uint32_t base_stride = shared_gaussian::kSharedStrideFloats + 7u;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        const bool key_x = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
        const bool key_y = glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS;
        const bool key_z = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
        const bool key_r = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
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
        if (key_r && !prev_r)
        {
            randomize_colors = !randomize_colors;
            random_colors = torch::Tensor();
            random_colors_count = 0;
            gpu_data_dirty = true; // Force color update
        }
        prev_r = key_r;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
        {
            FrameCameraToBounds();
        }

        // --- Reader Update Logic ---
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
            const auto *header = reader.Header();
            if (header)
            {
                sh_degree = header->sh_degree;
                render_scale_modifier = header->render_scale_modifier;
                const uint64_t current_version = header->data_version.load(std::memory_order_acquire);
                if (current_version != last_version)
                {
                    uint64_t version = 0;
                    uint64_t frame = 0;
                    if (reader.ReadLatest(&points, &point_count, &stride, &frame, &version))
                    {
                        last_version = version;
                        (void)frame;
                        gpu_data_dirty = true; // New data arrived, mark for upload
                    }
                }
            }
        }

        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.05f, 0.05f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (point_count > 0 && stride >= base_stride)
        {
            const uint32_t sh_coeffs = (sh_degree > 0) ? (sh_degree + 1) * (sh_degree + 1) : 0;
            const uint32_t sh_stride = base_stride + sh_coeffs * 3u;
            const bool use_sh = (sh_degree > 0) && (stride >= sh_stride) && !randomize_colors;

            // --- OPTIMIZATION 1: Conditional Upload & GPU Slicing ---
            // We only run this block if new data arrived from Shared Memory.
            // Otherwise, we reuse 'means3D', 'colors', etc., which stay on VRAM.
            if (gpu_data_dirty)
            {
                // Upload ONE large contiguous blob. No CPU loop!
                auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

                // Copy points -> GPU raw tensor
                raw_gpu = torch::from_blob(points.data(),
                                           {static_cast<int64_t>(point_count), static_cast<int64_t>(stride)},
                                           opts_cpu)
                              .to(device, /*non_blocking=*/false, /*copy=*/true);

                using namespace torch::indexing;

                // Create views/slices on GPU (Instant)
                means3D = raw_gpu.index({Slice(), Slice(0, 3)}).contiguous();
                // Colors are temporarily extracted even if SH is used, just to be safe
                colors = raw_gpu.index({Slice(), Slice(3, 6)}).contiguous();
                opacities = raw_gpu.index({Slice(), Slice(6, 7)}).contiguous();
                scales = raw_gpu.index({Slice(), Slice(8, 11)}).contiguous();
                rotations = raw_gpu.index({Slice(), Slice(11, 15)}).contiguous();

                // Handle SH or Random Colors
                if (use_sh)
                {
                    auto sh_flat = raw_gpu.index({Slice(), Slice(static_cast<int64_t>(base_stride),
                                                                 static_cast<int64_t>(base_stride + sh_coeffs * 3))});
                    sh_tensor = sh_flat.view({static_cast<int64_t>(point_count),
                                              static_cast<int64_t>(sh_coeffs), 3})
                                    .contiguous();
                    // Rasterizer usually ignores 'colors' if sh_degree > 0, but we can clear it to save memory/confusion
                    colors = torch::zeros({0}, device);
                }
                else
                {
                    sh_tensor = torch::zeros({0}, device);

                    if (randomize_colors)
                    {
                        if (!random_colors.defined() || random_colors_count != static_cast<int64_t>(point_count))
                        {
                            random_colors = torch::rand({static_cast<int64_t>(point_count), 3}, device);
                            random_colors_count = static_cast<int64_t>(point_count);
                        }
                        colors = random_colors;
                        // Reset opacities to full if needed, or keep original
                        opacities = torch::ones({static_cast<int64_t>(point_count), 1}, device);
                    }
                }

                // Update Bounds (Lazy CPU check just for "Frame Camera")
                // We do a strided check to avoid iterating 100k points on CPU
                if (!g_bounds.valid)
                {
                    Vec3 min_v{points[0], points[1], points[2]};
                    Vec3 max_v{points[0], points[1], points[2]};
                    // Sample every 100th point for speed
                    for (size_t i = 0; i < points.size(); i += stride * 100)
                    {
                        min_v.x = std::min(min_v.x, points[i + 0]);
                        min_v.y = std::min(min_v.y, points[i + 1]);
                        min_v.z = std::min(min_v.z, points[i + 2]);
                        max_v.x = std::max(max_v.x, points[i + 0]);
                        max_v.y = std::max(max_v.y, points[i + 1]);
                        max_v.z = std::max(max_v.z, points[i + 2]);
                    }
                    g_bounds.min = min_v;
                    g_bounds.max = max_v;
                    g_bounds.valid = true;
                }

                gpu_data_dirty = false;
            }

            // --- RENDER ---
            const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
            const float fovy_deg = 45.0f;
            const float fovy_rad = fovy_deg * 3.14159265f / 180.0f;
            const float tan_fovy = std::tan(fovy_rad * 0.5f);
            const float tan_fovx = tan_fovy * aspect;

            Quat q_model = NormalizeQuat(g_camera.model_rot);
            Mat4 proj_mat = Perspective(fovy_deg, aspect, 0.01f, 100.0f);

            Vec3 base_offset{0.0f, 0.0f, 0.0f};
            if (g_bounds.valid)
            {
                // Simple framing logic
                Vec3 center{0.5f * (g_bounds.min.x + g_bounds.max.x), 0.5f * (g_bounds.min.y + g_bounds.max.y), 0.5f * (g_bounds.min.z + g_bounds.max.z)};
                Vec3 extents{0.5f * (g_bounds.max.x - g_bounds.min.x), 0.5f * (g_bounds.max.y - g_bounds.min.y), 0.5f * (g_bounds.max.z - g_bounds.min.z)};
                const float radius = std::max({extents.x, extents.y, extents.z, 0.1f});
                base_offset.z = (radius * 3.0f) - center.z;
            }

            Vec3 pivot{g_camera.target.x, g_camera.target.y, g_camera.target.z};
            Vec3 user_offset{g_camera.model_offset.x, g_camera.model_offset.y, g_camera.model_offset.z};
            Vec3 offset = AddVec3(base_offset, user_offset);
            Vec3 r_pivot = RotateVec3Quat(pivot, q_model);
            Vec3 view_trans = AddVec3(SubVec3(pivot, r_pivot), offset);

            Mat4 view_mat = Mat4FromQuatTranslation(q_model, view_trans);
            Mat4 view_proj = Multiply(proj_mat, view_mat);
            torch::Tensor view_ten = Mat4ToTensorRowMajor(view_mat, device).transpose(0, 1).contiguous();
            torch::Tensor proj_ten_t = Mat4ToTensorRowMajor(proj_mat, device).transpose(0, 1).contiguous();

            using namespace torch::indexing;
            view_ten.index_put_({Slice(), 0}, view_ten.index({Slice(), 0}) * -1.0f);

            torch::Tensor proj_t = torch::matmul(view_ten, proj_ten_t).contiguous();

            Vec3 cam_pos_v = CameraPosFromView(view_mat);
            auto cam_pos = torch::tensor({cam_pos_v.x, cam_pos_v.y, cam_pos_v.z}, device);

            float scale_final = render_scale_modifier * options.point_size;

            auto outputs = GaussianRasterizer::apply(
                means3D, colors, opacities, scales, rotations,
                scale_final,
                view_ten, proj_t, tan_fovx, tan_fovy,
                height, width,
                sh_tensor,
                use_sh ? static_cast<int>(sh_degree) : 0,
                cam_pos,
                false);

            // --- OPTIMIZATION 2: GPU Permute ---
            // Perform CHW -> HWC on GPU, then download. Fixes OpenMP spin.
            auto img_tensor = outputs[0].detach().clamp(0.0f, 1.0f).mul(255.0f).to(torch::kU8).permute({1, 2, 0}).contiguous().cpu();

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glRasterPos2f(-1.0f, -1.0f);
            glPixelZoom(1.0f, 1.0f);
            glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, img_tensor.data_ptr<uint8_t>());
        }

        glfwSwapBuffers(window);
    }

    reader.Close();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}