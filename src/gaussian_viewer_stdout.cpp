#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <torch/torch.h>

#include "GaussianRasterizer.h"
#include "SharedGaussian.h"
#include "utils/SmplLBS.h"
#include "utils/viewer/ViewerMath.h"

namespace
{
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
    if (mask1.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(tr.index({mask1}) + 1.0f, 0.0f)) * 2.0f;
        q.index_put_({mask1, 0}, 0.25f * S);
        q.index_put_({mask1, 1}, (rot_mat.index({mask1, 2, 1}) - rot_mat.index({mask1, 1, 2})) / S);
        q.index_put_({mask1, 2}, (rot_mat.index({mask1, 0, 2}) - rot_mat.index({mask1, 2, 0})) / S);
        q.index_put_({mask1, 3}, (rot_mat.index({mask1, 1, 0}) - rot_mat.index({mask1, 0, 1})) / S);
    }

    auto mask2 = (~mask1) & (m00 > m11) & (m00 > m22);
    if (mask2.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m00.index({mask2}) - m11.index({mask2}) - m22.index({mask2}),
                                              0.0f)) *
                 2.0f;
        q.index_put_({mask2, 0}, (rot_mat.index({mask2, 2, 1}) - rot_mat.index({mask2, 1, 2})) / S);
        q.index_put_({mask2, 1}, 0.25f * S);
        q.index_put_({mask2, 2}, (rot_mat.index({mask2, 0, 1}) + rot_mat.index({mask2, 1, 0})) / S);
        q.index_put_({mask2, 3}, (rot_mat.index({mask2, 0, 2}) + rot_mat.index({mask2, 2, 0})) / S);
    }

    auto mask3 = (~mask1) & (~mask2) & (m11 > m22);
    if (mask3.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m11.index({mask3}) - m00.index({mask3}) - m22.index({mask3}),
                                              0.0f)) *
                 2.0f;
        q.index_put_({mask3, 0}, (rot_mat.index({mask3, 0, 2}) - rot_mat.index({mask3, 2, 0})) / S);
        q.index_put_({mask3, 1}, (rot_mat.index({mask3, 0, 1}) + rot_mat.index({mask3, 1, 0})) / S);
        q.index_put_({mask3, 2}, 0.25f * S);
        q.index_put_({mask3, 3}, (rot_mat.index({mask3, 1, 2}) + rot_mat.index({mask3, 2, 1})) / S);
    }

    auto mask4 = (~mask1) & (~mask2) & (~mask3);
    if (mask4.any().item<bool>())
    {
        auto S = torch::sqrt(torch::clamp_min(1.0f + m22.index({mask4}) - m00.index({mask4}) - m11.index({mask4}),
                                              0.0f)) *
                 2.0f;
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

struct ViewerOptions
{
    std::string shm_name = "GaussianAvatarShared";
    std::string pose_shm_name;
    std::string bind_shm_name;
    std::string smpl_model_path = "smpl_data.pt";
    int width = 1280;
    int height = 720;
    float point_size = 1.0f;
    int retry_ms = 500;
    int target_fps = 60; 
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
    bool dirty = true; 
};

CameraState g_camera{};
std::mutex g_camera_mutex;

std::mutex g_pose_mutex;
shared_gaussian::SharedPoseWriter g_pose_writer;
bool g_pose_writer_ready = false;
std::vector<float> g_pose_payload(shared_gaussian::kSharedPoseFloats, 0.0f);
std::mutex g_smpl_mutex;
std::vector<float> g_smpl_pose(72, 0.0f);
std::array<float, 3> g_smpl_trans{{0.0f, 0.0f, 0.0f}};
bool g_smpl_dirty = true;

struct BoundsState
{
    bool valid = false;
    Vec3 min;
    Vec3 max;
};

BoundsState g_bounds{};

// --- Input Handling ---

void InputListenLoop()
{
    std::string line;
    auto get_val = [&](const std::string& src, const std::string& key) -> float {
        size_t pos = src.find(key);
        if (pos == std::string::npos) return 0.0f;
        size_t start = src.find(':', pos) + 1;
        size_t end = src.find_first_of(",}", start);
        try { return std::stof(src.substr(start, end - start)); } catch(...) { return 0.0f; }
    };
    auto has_key = [&](const std::string& src, const std::string& key) -> bool {
        return src.find(key) != std::string::npos;
    };
    auto extract_floats = [&](const std::string& src) -> std::vector<float> {
        std::string cleaned = src;
        for (char &c : cleaned)
        {
            if (c == ',' || c == '[' || c == ']' || c == '{' || c == '}')
                c = ' ';
        }
        std::stringstream ss(cleaned);
        std::string token;
        std::vector<float> vals;
        while (ss >> token)
        {
            try
            {
                size_t idx = 0;
                float v = std::stof(token, &idx);
                if (idx > 0)
                    vals.push_back(v);
            }
            catch (...)
            {
            }
        }
        return vals;
    };
    auto quat_from_euler = [&](float yaw, float pitch, float roll) -> Quat {
        const Quat qy = AxisAngleQuat({0.0f, 1.0f, 0.0f}, yaw);
        const Quat qx = AxisAngleQuat({1.0f, 0.0f, 0.0f}, pitch);
        const Quat qz = AxisAngleQuat({0.0f, 0.0f, 1.0f}, roll);
        return NormalizeQuat(MulQuat(qy, MulQuat(qx, qz)));
    };

    while (std::getline(std::cin, line))
    {
        std::lock_guard<std::mutex> lock(g_camera_mutex);
        bool changed = false;

        if (line.find("smpl_pose") != std::string::npos)
        {
            const bool is_delta = (line.find("smpl_pose_delta") != std::string::npos);
            const auto vals = extract_floats(line);
            if (vals.size() >= 72)
            {
                std::lock_guard<std::mutex> pose_lock(g_pose_mutex);
                for (size_t i = 0; i < 72; ++i)
                {
                    g_pose_payload[i] = is_delta ? (g_pose_payload[i] + vals[i]) : vals[i];
                }
                if (g_pose_writer_ready)
                {
                    g_pose_writer.Write(g_pose_payload.data(), static_cast<uint32_t>(g_pose_payload.size()));
                }
                {
                    std::lock_guard<std::mutex> smpl_lock(g_smpl_mutex);
                    for (size_t i = 0; i < 72; ++i)
                    {
                        g_smpl_pose[i] = is_delta ? (g_smpl_pose[i] + vals[i]) : vals[i];
                    }
                    g_smpl_dirty = true;
                }
            }
        }
        else if (line.find("smpl_trans") != std::string::npos)
        {
            const auto vals = extract_floats(line);
            if (vals.size() >= 3)
            {
                std::lock_guard<std::mutex> pose_lock(g_pose_mutex);
                g_pose_payload[72] = vals[0];
                g_pose_payload[73] = vals[1];
                g_pose_payload[74] = vals[2];
                if (g_pose_writer_ready)
                {
                    g_pose_writer.Write(g_pose_payload.data(), static_cast<uint32_t>(g_pose_payload.size()));
                }
                {
                    std::lock_guard<std::mutex> smpl_lock(g_smpl_mutex);
                    g_smpl_trans[0] = vals[0];
                    g_smpl_trans[1] = vals[1];
                    g_smpl_trans[2] = vals[2];
                    g_smpl_dirty = true;
                }
            }
        }
        else if (line.find("\"pose\"") != std::string::npos || line.find("pose_delta") != std::string::npos) {
            const bool is_delta = line.find("pose_delta") != std::string::npos;
            const bool has_quat = has_key(line, "\"qw\"") || has_key(line, "\"qx\"") ||
                                  has_key(line, "\"qy\"") || has_key(line, "\"qz\"");
            Quat new_q = g_camera.model_rot;
            if (has_quat)
            {
                const float qw = get_val(line, "\"qw\"");
                const float qx = get_val(line, "\"qx\"");
                const float qy = get_val(line, "\"qy\"");
                const float qz = get_val(line, "\"qz\"");
                new_q = NormalizeQuat({qw, qx, qy, qz});
            }
            else
            {
                const bool use_deg = has_key(line, "\"yaw_deg\"") || has_key(line, "\"pitch_deg\"") ||
                                     has_key(line, "\"roll_deg\"");
                float yaw = use_deg ? get_val(line, "\"yaw_deg\"") : get_val(line, "\"yaw\"");
                float pitch = use_deg ? get_val(line, "\"pitch_deg\"") : get_val(line, "\"pitch\"");
                float roll = use_deg ? get_val(line, "\"roll_deg\"") : get_val(line, "\"roll\"");
                if (use_deg)
                {
                    constexpr float kDegToRad = 3.14159265f / 180.0f;
                    yaw *= kDegToRad;
                    pitch *= kDegToRad;
                    roll *= kDegToRad;
                }
                new_q = quat_from_euler(yaw, pitch, roll);
            }
            g_camera.model_rot = is_delta ? NormalizeQuat(MulQuat(new_q, g_camera.model_rot)) : new_q;
            changed = true;
        }
        else if (line.find("rotate") != std::string::npos) {
            float dx = get_val(line, "\"dx\"");
            const float yaw_delta = dx; 
            const Quat qy = AxisAngleQuat({0.0f, 1.0f, 0.0f}, yaw_delta);
            g_camera.model_rot = NormalizeQuat(MulQuat(qy, g_camera.model_rot));
            changed = true;
        }
        else if (line.find("pan") != std::string::npos) {
            float dx = get_val(line, "\"dx\"");
            float dy = get_val(line, "\"dy\"");
            g_camera.model_offset.x += dx;
            g_camera.model_offset.y += dy;
            changed = true;
        }
        else if (line.find("zoom") != std::string::npos) {
            float delta = get_val(line, "\"delta\"");
            g_camera.model_offset.z += delta;
            changed = true;
        }

        if (changed) g_camera.dirty = true;
    }
}

void MouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/)
{
    std::lock_guard<std::mutex> lock(g_camera_mutex);
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        g_camera.rotating = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        g_camera.panning = (action == GLFW_PRESS);
    glfwGetCursorPos(window, &g_camera.last_x, &g_camera.last_y);
}

void CursorPosCallback(GLFWwindow * /*window*/, double xpos, double ypos)
{
    std::lock_guard<std::mutex> lock(g_camera_mutex);
    const double dx = xpos - g_camera.last_x;
    const double dy = ypos - g_camera.last_y;
    g_camera.last_x = xpos;
    g_camera.last_y = ypos;
    
    bool changed = false;
    if (g_camera.rotating)
    {
        const float yaw_delta = static_cast<float>(dx) * 0.005f;
        const Quat qy = AxisAngleQuat({0.0f, 1.0f, 0.0f}, yaw_delta);
        g_camera.model_rot = NormalizeQuat(MulQuat(qy, g_camera.model_rot));
        changed = true;
    }
    if (g_camera.panning)
    {
        const float pan_scale = 0.005f;
        g_camera.model_offset.x += -static_cast<float>(dx) * pan_scale;
        g_camera.model_offset.y += static_cast<float>(dy) * pan_scale;
        changed = true;
    }
    if (changed) g_camera.dirty = true;
}

void ScrollCallback(GLFWwindow * /*window*/, double /*xoffset*/, double yoffset)
{
    std::lock_guard<std::mutex> lock(g_camera_mutex);
    const float zoom = static_cast<float>(yoffset) * 0.2f;
    g_camera.model_offset.z += zoom;
    g_camera.dirty = true;
}

bool ParseArgs(int argc, char **argv, ViewerOptions *options)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--shm" && i + 1 < argc) options->shm_name = argv[++i];
        else if (arg == "--pose-shm" && i + 1 < argc) options->pose_shm_name = argv[++i];
        else if (arg == "--bind-shm" && i + 1 < argc) options->bind_shm_name = argv[++i];
        else if (arg == "--smpl" && i + 1 < argc) options->smpl_model_path = argv[++i];
        else if (arg == "--width" && i + 1 < argc) options->width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) options->height = std::stoi(argv[++i]);
        else if (arg == "--point-size" && i + 1 < argc) options->point_size = std::stof(argv[++i]);
        else if (arg == "--retry-ms" && i + 1 < argc) options->retry_ms = std::stoi(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc) options->target_fps = std::stoi(argv[++i]);
        else if (arg == "--file" && i + 1 < argc) i++; 
        else if (arg == "--stream-stdout") {}
        else if (arg == "--help" || arg == "-h") return false;
    }
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif

    ViewerOptions options;
    if (!ParseArgs(argc, argv, &options)) return 0;
    if (options.pose_shm_name.empty())
    {
        options.pose_shm_name = options.shm_name + "_pose";
    }
    if (options.bind_shm_name.empty())
    {
        options.bind_shm_name = options.shm_name + "_bind";
    }
    g_pose_writer_ready = g_pose_writer.Init(options.pose_shm_name, shared_gaussian::kSharedPoseFloats);

    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(options.width, options.height, "Gaussian Avatar Viewer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); 

    if (!torch::cuda::is_available()) { glfwDestroyWindow(window); glfwTerminate(); return 1; }
    torch::Device device(torch::kCUDA);

    std::shared_ptr<SMPLLayer> smpl;
    torch::Tensor smpl_faces;
    bool smpl_ready = false;
    try
    {
        smpl = std::make_shared<SMPLLayer>(options.smpl_model_path);
        smpl->to(device);
        std::ifstream smpl_in(options.smpl_model_path, std::ios::binary);
        std::vector<char> f_bytes((std::istreambuf_iterator<char>(smpl_in)), (std::istreambuf_iterator<char>()));
        auto dict = torch::pickle_load(f_bytes).toGenericDict();
        smpl_faces = dict.at("faces").toTensor().to(torch::kLong).to(device);
        smpl_ready = smpl_faces.defined();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load SMPL model for viewer LBS: " << e.what() << std::endl;
    }

    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::thread input_thread(InputListenLoop);
    input_thread.detach();

    shared_gaussian::SharedGaussianReader reader;
    shared_gaussian::SharedBindReader bind_reader;
    std::vector<float> points;
    uint32_t point_count = 0;
    uint32_t stride = 0;

    std::vector<float> bind_betas;
    std::vector<float> bind_payload;
    uint32_t bind_count = 0;
    uint32_t bind_stride = 0;
    uint64_t last_bind_version = 0;

    auto last_retry = std::chrono::steady_clock::now();
    auto last_bind_retry = std::chrono::steady_clock::now();
    
    // FPS Limiter
    auto frame_duration = std::chrono::milliseconds(1000 / options.target_fps);
    auto next_frame_time = std::chrono::steady_clock::now();

    // --- PERSISTENT GPU STATE ---
    // Moved outside the loop to prevent re-allocation every frame
    torch::Tensor raw_gpu;
    torch::Tensor means3D, colors, opacities, scales, rotations, sh_tensor;
    bool gpu_data_dirty = false;
    bool use_sh = false;
    int active_sh_degree = 0;

    torch::Tensor bind_bary, bind_offsets, bind_rots, bind_face_indices, bind_betas_t;
    bool bind_data_dirty = false;
    torch::Tensor smpl_pose_t, smpl_trans_t;
    bool smpl_pose_dirty = true;
    
    bool randomize_colors = false;
    torch::Tensor random_colors;
    int64_t random_colors_count = 0;
    uint64_t last_version = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        bool should_render = false;

        // 1. Check Data Updates (SHM)
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
        float render_scale_modifier = 1.0f;
        
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
                        should_render = true;
                        gpu_data_dirty = true; // Mark that we need to re-upload to GPU
                    }
                }
            }
        }

        if (!bind_reader.IsOpen())
        {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_bind_retry).count() >= options.retry_ms)
            {
                last_bind_retry = now;
                bind_reader.Open(options.bind_shm_name);
            }
        }

        if (bind_reader.IsOpen())
        {
            const auto *bind_header = bind_reader.Header();
            if (bind_header)
            {
                const uint64_t current_version = bind_header->data_version.load(std::memory_order_acquire);
                if (current_version != last_bind_version)
                {
                    uint64_t version = 0;
                    if (bind_reader.ReadLatest(&bind_betas, &bind_payload, &bind_count, &bind_stride, &version))
                    {
                        last_bind_version = version;
                        bind_data_dirty = true;
                        should_render = true;
                    }
                }
            }
        }

        // 2. Check Input Updates
        {
            std::lock_guard<std::mutex> lock(g_camera_mutex);
            if (g_camera.dirty) {
                should_render = true;
                g_camera.dirty = false;
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_smpl_mutex);
            if (g_smpl_dirty)
            {
                should_render = true;
                smpl_pose_dirty = true;
                g_smpl_dirty = false;
            }
        }

        if (should_render && point_count > 0)
        {
            int width = options.width;
            int height = options.height;
            const float fovy_deg = 45.0f;
            
            // --- OPTIMIZED DATA UPLOAD ---
            // We only touch the PCIe bus if the Shared Memory data actually changed.
            if (gpu_data_dirty)
            {
                auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

                // 1. Upload Interleaved Data
                // Copy the CPU vector to GPU memory
                raw_gpu = torch::from_blob(points.data(), 
                    {static_cast<int64_t>(point_count), static_cast<int64_t>(stride)}, 
                    opts_cpu).to(device, /*non_blocking=*/false, /*copy=*/true);

                using namespace torch::indexing;

                // 2. Slice Data (Zero-copy views on GPU)
                means3D   = raw_gpu.index({Slice(), Slice(0, 3)}).contiguous();
                colors    = raw_gpu.index({Slice(), Slice(3, 6)}).contiguous(); 
                opacities = raw_gpu.index({Slice(), Slice(6, 7)}).contiguous();
                scales    = raw_gpu.index({Slice(), Slice(8, 11)}).contiguous();
                rotations = raw_gpu.index({Slice(), Slice(11, 15)}).contiguous();

                // 3. Handle SH / Color Modes
                const uint32_t base_stride = shared_gaussian::kSharedStrideFloats + 7u;
                const uint32_t sh_coeffs = (sh_degree > 0) ? (sh_degree + 1) * (sh_degree + 1) : 0;
                const uint32_t sh_stride_needed = base_stride + sh_coeffs * 3u;
                use_sh = (sh_degree > 0) && (stride >= sh_stride_needed) && !randomize_colors;
                active_sh_degree = use_sh ? static_cast<int>(sh_degree) : 0;

                if (use_sh)
                {
                    auto sh_flat = raw_gpu.index({Slice(), Slice(static_cast<int64_t>(base_stride), 
                                                                 static_cast<int64_t>(base_stride + sh_coeffs * 3))});
                    sh_tensor = sh_flat.view({static_cast<int64_t>(point_count), 
                                              static_cast<int64_t>(sh_coeffs), 3}).contiguous();
                    colors = torch::zeros({0}, device);
                }
                else
                {
                    sh_tensor = torch::zeros({0}, device);
                    if (randomize_colors) {
                        if (!random_colors.defined() || random_colors_count != static_cast<int64_t>(point_count)) {
                            random_colors = torch::rand({static_cast<int64_t>(point_count), 3}, device);
                            random_colors_count = static_cast<int64_t>(point_count);
                        }
                        colors = random_colors;
                        opacities = torch::ones({static_cast<int64_t>(point_count), 1}, device);
                    }
                }

                // 4. Update Bounds (Lazy CPU stride)
                {
                     std::lock_guard<std::mutex> lock(g_camera_mutex);
                     if (!g_bounds.valid && point_count > 0) {
                          const float* p = points.data();
                          Vec3 min_v{p[0], p[1], p[2]};
                          Vec3 max_v{p[0], p[1], p[2]};
                          // Sampling for speed
                          for(uint32_t k=0; k<std::min(point_count, 1000u); k+=10) {
                              size_t idx = k*stride;
                              min_v.x = std::min(min_v.x, p[idx]);
                              max_v.x = std::max(max_v.x, p[idx]);
                              min_v.y = std::min(min_v.y, p[idx+1]);
                              // ... 
                          }
                          g_bounds.min = min_v; g_bounds.max = max_v; 
                          g_bounds.valid = true;
                     }
                }
                
                gpu_data_dirty = false;
            }

            if (bind_data_dirty && bind_count > 0 && bind_stride >= shared_gaussian::kSharedBindStrideFloats)
            {
                auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                auto bind_cpu = torch::from_blob(bind_payload.data(),
                                                 {static_cast<int64_t>(bind_count),
                                                  static_cast<int64_t>(bind_stride)},
                                                 opts_cpu).clone();
                using namespace torch::indexing;
                bind_bary = bind_cpu.index({Slice(), Slice(0, 3)}).to(device).contiguous();
                bind_offsets = bind_cpu.index({Slice(), Slice(3, 6)}).to(device).contiguous();
                bind_rots = bind_cpu.index({Slice(), Slice(6, 10)}).to(device).contiguous();
                auto face_idx_f = bind_cpu.index({Slice(), Slice(10, 11)}).squeeze(1);
                bind_face_indices = face_idx_f.to(torch::kLong).to(device).contiguous();

                if (!bind_betas.empty())
                {
                    auto betas_cpu = torch::from_blob(bind_betas.data(),
                                                      {1, static_cast<int64_t>(bind_betas.size())},
                                                      opts_cpu).clone();
                    bind_betas_t = betas_cpu.to(device);
                }

                bind_data_dirty = false;
                smpl_pose_dirty = true;
            }

            bool use_smpl_deform = smpl_ready && bind_bary.defined() && bind_offsets.defined() &&
                                   bind_rots.defined() && bind_face_indices.defined() &&
                                   bind_betas_t.defined() && (bind_bary.size(0) == point_count);

            if (use_smpl_deform)
            {
                if (smpl_pose_dirty)
                {
                    std::array<float, 3> trans_vals;
                    std::vector<float> pose_vals;
                    {
                        std::lock_guard<std::mutex> lock(g_smpl_mutex);
                        trans_vals = g_smpl_trans;
                        pose_vals = g_smpl_pose;
                    }
                    auto pose_cpu = torch::from_blob(pose_vals.data(), {1, 24, 3}, torch::kFloat).clone();
                    smpl_pose_t = pose_cpu.to(device);
                    auto trans_cpu = torch::from_blob(trans_vals.data(), {1, 3}, torch::kFloat).clone();
                    smpl_trans_t = trans_cpu.to(device);
                    smpl_pose_dirty = false;
                }

                auto smpl_out = smpl->forward(bind_betas_t, smpl_pose_t, smpl_trans_t);
                auto verts_posed = smpl_out.vertices[0];
                auto verts_canon = smpl->v_template.to(device);

                using namespace torch::indexing;
                auto selected_faces = smpl_faces.index_select(0, bind_face_indices);
                auto A = verts_posed.index_select(0, selected_faces.index({Slice(), 0}));
                auto B = verts_posed.index_select(0, selected_faces.index({Slice(), 1}));
                auto C = verts_posed.index_select(0, selected_faces.index({Slice(), 2}));

                auto A_can = verts_canon.index_select(0, selected_faces.index({Slice(), 0}));
                auto B_can = verts_canon.index_select(0, selected_faces.index({Slice(), 1}));
                auto C_can = verts_canon.index_select(0, selected_faces.index({Slice(), 2}));

                auto R_posed = ComputeTriFrames(A, B, C);
                auto R_canon = ComputeTriFrames(A_can, B_can, C_can);
                auto R_skin = torch::bmm(R_posed, R_canon.transpose(1, 2));
                auto posed_offsets = torch::bmm(R_skin, bind_offsets.unsqueeze(2)).squeeze(2);

                auto u = bind_bary.index({Slice(), 0}).unsqueeze(1);
                auto v = bind_bary.index({Slice(), 1}).unsqueeze(1);
                auto w = bind_bary.index({Slice(), 2}).unsqueeze(1);
                auto skinned_pos = u * A + v * B + w * C;
                means3D = skinned_pos + posed_offsets;

                auto q_skin = MatrixToQuat(R_skin);
                rotations = QuatMultiply(q_skin, bind_rots);
            }

            if (use_smpl_deform && colors.numel() == 0 && raw_gpu.defined() && !use_sh)
            {
                using namespace torch::indexing;
                colors = raw_gpu.index({Slice(), Slice(3, 6)}).contiguous();
            }

            // --- RENDER MATRICES ---
            const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
            const float fovy_rad = fovy_deg * 3.14159265f / 180.0f;
            const float tan_fovy = std::tan(fovy_rad * 0.5f);
            const float tan_fovx = tan_fovy * aspect;

            Quat q_model;
            Vec3 target, model_offset;
            {
                std::lock_guard<std::mutex> lock(g_camera_mutex);
                q_model = NormalizeQuat(g_camera.model_rot);
                target = g_camera.target;
                model_offset = g_camera.model_offset;
            }

            Mat4 proj_mat = Perspective(fovy_deg, aspect, 0.01f, 100.0f);
            
            Vec3 base_offset{0.0f, 0.0f, 0.0f};
            if (g_bounds.valid) {
                 base_offset.z = 5.0f; 
            }
            Vec3 pivot{target.x, target.y, target.z};
            Vec3 u_off{model_offset.x, model_offset.y, model_offset.z};
            Vec3 view_trans = AddVec3(SubVec3(pivot, RotateVec3Quat(pivot, q_model)), AddVec3(base_offset, u_off));
            Mat4 view_mat = Mat4FromQuatTranslation(q_model, view_trans);
            Mat4 view_proj = Multiply(proj_mat, view_mat);

            torch::Tensor view_ten = Mat4ToTensorRowMajor(view_mat, device).transpose(0, 1).contiguous();
            torch::Tensor proj_t = Mat4ToTensorRowMajor(view_proj, device).transpose(0, 1).contiguous();
            Vec3 cp = CameraPosFromView(view_mat);
            auto cam_pos = torch::tensor({cp.x, cp.y, cp.z}, device);

            // --- RASTERIZE ---
            // Uses cached tensors (means3D etc)
            const int sh_degree_to_use = active_sh_degree;
            torch::Tensor sh_to_use = (active_sh_degree > 0 && sh_tensor.numel() > 0) ? sh_tensor : torch::zeros({0}, device);
            auto outputs = GaussianRasterizer::apply(
                means3D, colors, opacities, scales, rotations,
                render_scale_modifier * options.point_size,
                view_ten, proj_t, tan_fovx, tan_fovy, height, width,
                sh_to_use, sh_degree_to_use, cam_pos, false);

            std::cerr << "[Viewer] SH degree: " << sh_degree_to_use << ", Points: " << point_count << ", Use SH: " << (use_sh ? "Yes" : "No") 
                      << ", Use SMPL Deform: " << (use_smpl_deform ? "Yes" : "No") << std::endl;
            
            // --- DOWNLOAD ---
            // Perform CHW -> HWC on GPU to prevent CPU OpenMP spin
            auto img_tensor = outputs[0].detach().clamp(0.0f, 1.0f).mul(255.0f).to(torch::kU8)
                  .permute({1, 2, 0}).contiguous().cpu();

            uint8_t *raw_data = img_tensor.data_ptr<uint8_t>();
            const size_t data_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
            const char magic[] = "IMG";
            const uint32_t size_header = static_cast<uint32_t>(data_size);
            
            std::cout.write(magic, 3);
            std::cout.write(reinterpret_cast<const char *>(&size_header), sizeof(size_header));
            std::cout.write(reinterpret_cast<const char *>(raw_data), data_size);
            std::cout.flush();

            // --- FRAME PACING ---
            auto now = std::chrono::steady_clock::now();
            next_frame_time += frame_duration;
            if (now > next_frame_time) next_frame_time = now + frame_duration;
            std::this_thread::sleep_until(next_frame_time);
        }
        else
        {
            // Sleep when idle
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); 
            next_frame_time = std::chrono::steady_clock::now();
        }
    }
    
    reader.Close();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}