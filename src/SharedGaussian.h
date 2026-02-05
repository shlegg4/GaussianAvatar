#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace shared_gaussian
{
constexpr uint32_t kSharedMagic = 0x53554147; // "GAUS"
constexpr uint32_t kSharedVersion = 2;
constexpr uint32_t kSharedStrideFloats = 8; // xyz + rgba + scale
constexpr uint32_t kSharedPoseMagic = 0x45534F50; // "POSE"
constexpr uint32_t kSharedPoseVersion = 1;
constexpr uint32_t kSharedPoseFloats = 75; // 72 pose + 3 translation
constexpr uint32_t kSharedBindMagic = 0x444E4942; // "BIND"
constexpr uint32_t kSharedBindVersion = 1;
constexpr uint32_t kSharedBindStrideFloats = 11; // bary(3) + offset(3) + rot(4) + face_idx(1)

struct alignas(64) SharedGaussianHeader
{
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t capacity = 0;
    uint32_t count = 0;
    uint32_t stride_floats = 0;
    uint32_t sh_degree = 0;
    float render_scale_modifier = 0.0f;
    uint32_t reserved0 = 0;
    uint64_t frame_id = 0;
    std::atomic<uint64_t> data_version;
    uint64_t reserved1[4] = {};
};

static_assert(sizeof(SharedGaussianHeader) % 64 == 0, "SharedGaussianHeader size must be 64-byte aligned.");

struct SharedGaussianMapping
{
#ifdef _WIN32
    HANDLE handle = nullptr;
#endif
    void *base = nullptr;
    size_t size_bytes = 0;
    SharedGaussianHeader *header = nullptr;
    float *data = nullptr;
};

class SharedGaussianWriter
{
  public:
    bool Init(const std::string &name, uint32_t capacity, uint32_t stride_floats, uint32_t sh_degree,
              float render_scale_modifier)
    {
#ifdef _WIN32
        Close();
        const size_t total_size = sizeof(SharedGaussianHeader) + static_cast<size_t>(capacity) *
                                                               static_cast<size_t>(stride_floats) * sizeof(float);
        DWORD size_low = static_cast<DWORD>(total_size & 0xFFFFFFFF);
        DWORD size_high = static_cast<DWORD>((total_size >> 32) & 0xFFFFFFFF);
        HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, size_high, size_low,
                                            name.c_str());
        if (!mapping)
            return false;

        void *base = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
        if (!base)
        {
            CloseHandle(mapping);
            return false;
        }

        map_.handle = mapping;
        map_.base = base;
        map_.size_bytes = total_size;
        map_.header = reinterpret_cast<SharedGaussianHeader *>(base);
        map_.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + sizeof(SharedGaussianHeader));

        std::memset(map_.base, 0, sizeof(SharedGaussianHeader));
        map_.header->magic = kSharedMagic;
        map_.header->version = kSharedVersion;
        map_.header->capacity = capacity;
        map_.header->stride_floats = stride_floats;
        map_.header->sh_degree = sh_degree;
        map_.header->render_scale_modifier = render_scale_modifier;
        map_.header->data_version.store(0, std::memory_order_relaxed);
        return true;
#else
        (void)name;
        (void)capacity;
        (void)stride_floats;
        (void)sh_degree;
        (void)render_scale_modifier;
        return false;
#endif
    }

    bool Write(const float *data, uint32_t count, uint64_t frame_id)
    {
#ifdef _WIN32
        if (!map_.header || !map_.data)
            return false;
        if (count > map_.header->capacity)
            count = map_.header->capacity;
        const size_t copy_bytes = static_cast<size_t>(count) *
                                  static_cast<size_t>(map_.header->stride_floats) * sizeof(float);
        std::memcpy(map_.data, data, copy_bytes);
        map_.header->count = count;
        map_.header->frame_id = frame_id;
        uint64_t next = map_.header->data_version.load(std::memory_order_relaxed) + 1;
        map_.header->data_version.store(next, std::memory_order_release);
        return true;
#else
        (void)data;
        (void)count;
        (void)frame_id;
        return false;
#endif
    }

    void Close()
    {
#ifdef _WIN32
        if (map_.base)
        {
            UnmapViewOfFile(map_.base);
        }
        if (map_.handle)
        {
            CloseHandle(map_.handle);
        }
#endif
        map_ = SharedGaussianMapping{};
    }

    ~SharedGaussianWriter() { Close(); }

  private:
    SharedGaussianMapping map_{};
};

class SharedGaussianReader
{
  public:
    bool Open(const std::string &name)
    {
#ifdef _WIN32
        Close();
        HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, name.c_str());
        if (!mapping)
            return false;
        void *base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!base)
        {
            CloseHandle(mapping);
            return false;
        }
        map_.handle = mapping;
        map_.base = base;
        map_.header = reinterpret_cast<SharedGaussianHeader *>(base);
        if (map_.header->magic != kSharedMagic || map_.header->version != kSharedVersion)
        {
            Close();
            return false;
        }
        map_.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + sizeof(SharedGaussianHeader));
        return true;
#else
        (void)name;
        return false;
#endif
    }

    bool IsOpen() const { return map_.header != nullptr; }

    void Close()
    {
#ifdef _WIN32
        if (map_.base)
        {
            UnmapViewOfFile(map_.base);
        }
        if (map_.handle)
        {
            CloseHandle(map_.handle);
        }
#endif
        map_ = SharedGaussianMapping{};
    }

    const SharedGaussianHeader *Header() const { return map_.header; }

    bool ReadLatest(std::vector<float> *out, uint32_t *out_count, uint32_t *out_stride, uint64_t *out_frame,
                    uint64_t *out_version) const
    {
        if (!map_.header || !map_.data)
            return false;
        for (int attempt = 0; attempt < 3; ++attempt)
        {
            uint64_t before = map_.header->data_version.load(std::memory_order_acquire);
            uint32_t count = map_.header->count;
            uint32_t stride = map_.header->stride_floats;
            uint64_t frame = map_.header->frame_id;
            size_t total_floats = static_cast<size_t>(count) * static_cast<size_t>(stride);
            out->resize(total_floats);
            if (total_floats > 0)
            {
                std::memcpy(out->data(), map_.data, total_floats * sizeof(float));
            }
            uint64_t after = map_.header->data_version.load(std::memory_order_acquire);
            if (before == after)
            {
                if (out_count)
                    *out_count = count;
                if (out_stride)
                    *out_stride = stride;
                if (out_frame)
                    *out_frame = frame;
                if (out_version)
                    *out_version = after;
                return true;
            }
        }
        return false;
    }

    ~SharedGaussianReader() { Close(); }

  private:
    SharedGaussianMapping map_{};
};

struct alignas(64) SharedPoseHeader
{
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t pose_floats = 0;
    uint32_t reserved0 = 0;
    std::atomic<uint64_t> data_version;
    uint64_t reserved1[6] = {};
};

static_assert(sizeof(SharedPoseHeader) % 64 == 0, "SharedPoseHeader size must be 64-byte aligned.");

struct SharedPoseMapping
{
#ifdef _WIN32
    HANDLE handle = nullptr;
#endif
    void *base = nullptr;
    size_t size_bytes = 0;
    SharedPoseHeader *header = nullptr;
    float *data = nullptr;
};

class SharedPoseWriter
{
  public:
    bool Init(const std::string &name, uint32_t pose_floats = kSharedPoseFloats)
    {
#ifdef _WIN32
        Close();
        const size_t total_size = sizeof(SharedPoseHeader) +
                                  static_cast<size_t>(pose_floats) * sizeof(float);
        DWORD size_low = static_cast<DWORD>(total_size & 0xFFFFFFFF);
        DWORD size_high = static_cast<DWORD>((total_size >> 32) & 0xFFFFFFFF);
        HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, size_high, size_low,
                                            name.c_str());
        if (!mapping)
            return false;

        void *base = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
        if (!base)
        {
            CloseHandle(mapping);
            return false;
        }

        map_.handle = mapping;
        map_.base = base;
        map_.size_bytes = total_size;
        map_.header = reinterpret_cast<SharedPoseHeader *>(base);
        map_.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + sizeof(SharedPoseHeader));

        std::memset(map_.base, 0, sizeof(SharedPoseHeader));
        map_.header->magic = kSharedPoseMagic;
        map_.header->version = kSharedPoseVersion;
        map_.header->pose_floats = pose_floats;
        map_.header->data_version.store(0, std::memory_order_relaxed);
        return true;
#else
        (void)name;
        (void)pose_floats;
        return false;
#endif
    }

    bool Write(const float *data, uint32_t count)
    {
#ifdef _WIN32
        if (!map_.header || !map_.data)
            return false;
        const uint32_t max_count = map_.header->pose_floats;
        if (count > max_count)
            count = max_count;
        const size_t copy_bytes = static_cast<size_t>(count) * sizeof(float);
        std::memcpy(map_.data, data, copy_bytes);
        uint64_t next = map_.header->data_version.load(std::memory_order_relaxed) + 1;
        map_.header->data_version.store(next, std::memory_order_release);
        return true;
#else
        (void)data;
        (void)count;
        return false;
#endif
    }

    void Close()
    {
#ifdef _WIN32
        if (map_.base)
        {
            UnmapViewOfFile(map_.base);
        }
        if (map_.handle)
        {
            CloseHandle(map_.handle);
        }
#endif
        map_ = SharedPoseMapping{};
    }

    ~SharedPoseWriter() { Close(); }

  private:
    SharedPoseMapping map_{};
};

class SharedPoseReader
{
  public:
    bool Open(const std::string &name)
    {
#ifdef _WIN32
        Close();
        HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, name.c_str());
        if (!mapping)
            return false;
        void *base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!base)
        {
            CloseHandle(mapping);
            return false;
        }
        map_.handle = mapping;
        map_.base = base;
        map_.header = reinterpret_cast<SharedPoseHeader *>(base);
        if (map_.header->magic != kSharedPoseMagic || map_.header->version != kSharedPoseVersion)
        {
            Close();
            return false;
        }
        map_.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + sizeof(SharedPoseHeader));
        return true;
#else
        (void)name;
        return false;
#endif
    }

    bool IsOpen() const { return map_.header != nullptr; }

    void Close()
    {
#ifdef _WIN32
        if (map_.base)
        {
            UnmapViewOfFile(map_.base);
        }
        if (map_.handle)
        {
            CloseHandle(map_.handle);
        }
#endif
        map_ = SharedPoseMapping{};
    }

    const SharedPoseHeader *Header() const { return map_.header; }

    bool ReadLatest(std::vector<float> *out, uint64_t *out_version) const
    {
        if (!map_.header || !map_.data)
            return false;
        for (int attempt = 0; attempt < 3; ++attempt)
        {
            uint64_t before = map_.header->data_version.load(std::memory_order_acquire);
            const uint32_t count = map_.header->pose_floats;
            out->resize(count);
            if (count > 0)
            {
                std::memcpy(out->data(), map_.data, static_cast<size_t>(count) * sizeof(float));
            }
            uint64_t after = map_.header->data_version.load(std::memory_order_acquire);
            if (before == after)
            {
                if (out_version)
                    *out_version = after;
                return true;
            }
        }
        return false;
    }

    ~SharedPoseReader() { Close(); }

  private:
    SharedPoseMapping map_{};
};

struct alignas(64) SharedBindHeader
{
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    uint32_t stride_floats = 0;
    uint32_t betas_count = 0;
    uint32_t reserved0 = 0;
    std::atomic<uint64_t> data_version;
    uint64_t reserved1[5] = {};
};

static_assert(sizeof(SharedBindHeader) % 64 == 0, "SharedBindHeader size must be 64-byte aligned.");

struct SharedBindMapping
{
#ifdef _WIN32
    HANDLE handle = nullptr;
#endif
    void *base = nullptr;
    size_t size_bytes = 0;
    SharedBindHeader *header = nullptr;
    float *data = nullptr;
};

class SharedBindWriter
{
  public:
    bool Init(const std::string &name, uint32_t count, uint32_t stride_floats, uint32_t betas_count)
    {
#ifdef _WIN32
        Close();
        const size_t total_floats = static_cast<size_t>(betas_count) +
                                    static_cast<size_t>(count) * static_cast<size_t>(stride_floats);
        const size_t total_size = sizeof(SharedBindHeader) + total_floats * sizeof(float);
        DWORD size_low = static_cast<DWORD>(total_size & 0xFFFFFFFF);
        DWORD size_high = static_cast<DWORD>((total_size >> 32) & 0xFFFFFFFF);
        HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, size_high, size_low,
                                            name.c_str());
        if (!mapping)
            return false;

        void *base = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
        if (!base)
        {
            CloseHandle(mapping);
            return false;
        }

        map_.handle = mapping;
        map_.base = base;
        map_.size_bytes = total_size;
        map_.header = reinterpret_cast<SharedBindHeader *>(base);
        map_.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + sizeof(SharedBindHeader));

        std::memset(map_.base, 0, sizeof(SharedBindHeader));
        map_.header->magic = kSharedBindMagic;
        map_.header->version = kSharedBindVersion;
        map_.header->count = count;
        map_.header->stride_floats = stride_floats;
        map_.header->betas_count = betas_count;
        map_.header->data_version.store(0, std::memory_order_relaxed);
        return true;
#else
        (void)name;
        (void)count;
        (void)stride_floats;
        (void)betas_count;
        return false;
#endif
    }

    bool Write(const float *betas, uint32_t betas_count, const float *data, uint32_t count)
    {
#ifdef _WIN32
        if (!map_.header || !map_.data)
            return false;
        if (betas_count > map_.header->betas_count)
            betas_count = map_.header->betas_count;
        if (count > map_.header->count)
            count = map_.header->count;
        const size_t betas_bytes = static_cast<size_t>(betas_count) * sizeof(float);
        std::memcpy(map_.data, betas, betas_bytes);
        const size_t stride = static_cast<size_t>(map_.header->stride_floats);
        const size_t data_bytes = static_cast<size_t>(count) * stride * sizeof(float);
        std::memcpy(map_.data + map_.header->betas_count, data, data_bytes);
        uint64_t next = map_.header->data_version.load(std::memory_order_relaxed) + 1;
        map_.header->data_version.store(next, std::memory_order_release);
        return true;
#else
        (void)betas;
        (void)betas_count;
        (void)data;
        (void)count;
        return false;
#endif
    }

    void Close()
    {
#ifdef _WIN32
        if (map_.base)
        {
            UnmapViewOfFile(map_.base);
        }
        if (map_.handle)
        {
            CloseHandle(map_.handle);
        }
#endif
        map_ = SharedBindMapping{};
    }

    ~SharedBindWriter() { Close(); }

  private:
    SharedBindMapping map_{};
};

class SharedBindReader
{
  public:
    bool Open(const std::string &name)
    {
#ifdef _WIN32
        Close();
        HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, name.c_str());
        if (!mapping)
            return false;
        void *base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!base)
        {
            CloseHandle(mapping);
            return false;
        }
        map_.handle = mapping;
        map_.base = base;
        map_.header = reinterpret_cast<SharedBindHeader *>(base);
        if (map_.header->magic != kSharedBindMagic || map_.header->version != kSharedBindVersion)
        {
            Close();
            return false;
        }
        map_.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + sizeof(SharedBindHeader));
        return true;
#else
        (void)name;
        return false;
#endif
    }

    bool IsOpen() const { return map_.header != nullptr; }

    void Close()
    {
#ifdef _WIN32
        if (map_.base)
        {
            UnmapViewOfFile(map_.base);
        }
        if (map_.handle)
        {
            CloseHandle(map_.handle);
        }
#endif
        map_ = SharedBindMapping{};
    }

    const SharedBindHeader *Header() const { return map_.header; }

    bool ReadLatest(std::vector<float> *out_betas, std::vector<float> *out_data,
                    uint32_t *out_count, uint32_t *out_stride, uint64_t *out_version) const
    {
        if (!map_.header || !map_.data)
            return false;
        for (int attempt = 0; attempt < 3; ++attempt)
        {
            uint64_t before = map_.header->data_version.load(std::memory_order_acquire);
            const uint32_t count = map_.header->count;
            const uint32_t stride = map_.header->stride_floats;
            const uint32_t betas_count = map_.header->betas_count;
            out_betas->resize(betas_count);
            if (betas_count > 0)
            {
                std::memcpy(out_betas->data(), map_.data, static_cast<size_t>(betas_count) * sizeof(float));
            }
            const size_t total_floats = static_cast<size_t>(count) * static_cast<size_t>(stride);
            out_data->resize(total_floats);
            if (total_floats > 0)
            {
                std::memcpy(out_data->data(), map_.data + betas_count, total_floats * sizeof(float));
            }
            uint64_t after = map_.header->data_version.load(std::memory_order_acquire);
            if (before == after)
            {
                if (out_count)
                    *out_count = count;
                if (out_stride)
                    *out_stride = stride;
                if (out_version)
                    *out_version = after;
                return true;
            }
        }
        return false;
    }

    ~SharedBindReader() { Close(); }

  private:
    SharedBindMapping map_{};
};
} // namespace shared_gaussian
