#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "utils/render/RenderMathUtils.h"
#include "utils/train/TrainCache.h"
#include "utils/train/TrainTypes.h"

struct PackedSlice3D
{
    int64_t offset = 0;
    int64_t length = 0;
    std::array<int64_t, 3> shape{0, 0, 0};
};

struct TrainingBatch
{
    torch::Tensor packed_images_u8;
    torch::Tensor packed_masks;
    torch::Tensor pose_base_batch; // [B, 24, 3]
    torch::Tensor trans_base_batch; // [B, 3]
    torch::Tensor time_tensor; // [B, 1]
    torch::Tensor crop_params; // [B, 3]
    torch::Tensor y_signs; // [B]
    torch::Tensor view_mats; // [B, 4, 4]
    torch::Tensor proj_mats; // [B, 4, 4]
    std::vector<float> tan_fovx;
    std::vector<float> tan_fovy;
    std::vector<PackedSlice3D> image_slices;
    std::vector<PackedSlice3D> mask_slices;
    std::vector<int64_t> sample_indices;
};

template <typename T>
class SafeQueue
{
public:
    bool Push(T value)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_)
            {
                return false;
            }
            queue_.push(std::move(value));
        }
        cv_.notify_one();
        return true;
    }

    bool TryPop(T *out)
    {
        if (out == nullptr)
        {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
        {
            return false;
        }
        *out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool WaitPop(T *out, const std::atomic<bool> &active)
    {
        if (out == nullptr)
        {
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, &active]() {
            return !queue_.empty() || closed_ || !active.load();
        });

        if (queue_.empty())
        {
            return false;
        }

        *out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool WaitPopFor(T *out, const std::atomic<bool> &active, std::chrono::milliseconds timeout)
    {
        if (out == nullptr)
        {
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [this, &active]() {
            return !queue_.empty() || closed_ || !active.load();
        });

        if (queue_.empty())
        {
            return false;
        }

        *out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void Close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool Empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t Size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool closed_ = false;
};

class GaussianDataLoader
{
public:
    GaussianDataLoader(const std::vector<TrainSample> &samples,
                       const std::vector<CachedSampleData> &cached,
                       const torch::Tensor &all_poses,
                       const torch::Tensor &all_trans,
                       const torch::Tensor &all_time,
                       const torch::Tensor &all_crops,
                       std::vector<int64_t> ordered_indices,
                       int batch_size,
                       torch::Device device,
                       int num_workers = 4,
                       size_t max_prefetch_batches = 8)
        : samples_(samples),
          cached_(cached),
          all_poses_(all_poses),
          all_trans_(all_trans),
          all_time_(all_time),
          all_crops_(all_crops),
          batch_size_(std::max(1, batch_size)),
          device_(device),
          max_prefetch_batches_(std::max<size_t>(1, max_prefetch_batches)),
          active_(true)
    {
        const int worker_count = std::max(1, num_workers);
        workers_alive_.store(worker_count);

        workers_.reserve(static_cast<size_t>(worker_count));
        for (int i = 0; i < worker_count; ++i)
        {
            workers_.emplace_back(&GaussianDataLoader::WorkerLoop, this);
        }

        Reset(std::move(ordered_indices));
    }

    ~GaussianDataLoader()
    {
        active_.store(false);
        task_queue_.Close();
        result_queue_.Close();
        result_space_cv_.notify_all();

        for (auto &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    GaussianDataLoader(const GaussianDataLoader &) = delete;
    GaussianDataLoader &operator=(const GaussianDataLoader &) = delete;

    bool Next(TrainingBatch *out_batch)
    {
        if (out_batch == nullptr)
        {
            return false;
        }

        for (;;)
        {
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                if (worker_exception_)
                {
                    std::rethrow_exception(worker_exception_);
                }
            }

            TrainingBatch batch;
            if (result_queue_.WaitPopFor(&batch, active_, std::chrono::milliseconds(10)))
            {
                *out_batch = std::move(batch);
                result_space_cv_.notify_one();
                return true;
            }

            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                if (worker_exception_)
                {
                    std::rethrow_exception(worker_exception_);
                }
            }

            const int64_t done = epoch_task_done_.load();
            const int64_t total = epoch_task_total_.load();
            if (done >= total && result_queue_.Empty())
            {
                return false;
            }

            if (!active_.load() && result_queue_.Empty())
            {
                return false;
            }
        }
    }

    void Reset(std::vector<int64_t> ordered_indices)
    {
        if (!active_.load())
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (worker_exception_)
            {
                std::rethrow_exception(worker_exception_);
            }
        }

        int64_t pushed_tasks = 0;
        epoch_task_done_.store(0);
        epoch_task_total_.store(0);

        for (size_t i = 0; i < ordered_indices.size(); i += static_cast<size_t>(batch_size_))
        {
            std::vector<int64_t> batch_indices;
            batch_indices.reserve(static_cast<size_t>(batch_size_));
            const size_t end = std::min(ordered_indices.size(), i + static_cast<size_t>(batch_size_));
            for (size_t j = i; j < end; ++j)
            {
                batch_indices.push_back(ordered_indices[j]);
            }
            if (batch_indices.empty())
            {
                continue;
            }
            if (!task_queue_.Push(std::move(batch_indices)))
            {
                break;
            }
            ++pushed_tasks;
        }

        epoch_task_total_.store(pushed_tasks);
    }

private:
    void WorkerLoop()
    {
        try
        {
            std::vector<int64_t> task_indices;
            while (active_.load())
            {
                if (!task_queue_.WaitPop(&task_indices, active_))
                {
                    break;
                }

                if (task_indices.empty())
                {
                    epoch_task_done_.fetch_add(1);
                    continue;
                }

                std::vector<int64_t> valid_batch_indices;
                valid_batch_indices.reserve(task_indices.size());
                for (const int64_t idx : task_indices)
                {
                    if (idx < 0 || static_cast<size_t>(idx) >= cached_.size())
                    {
                        continue;
                    }
                    if (cached_[static_cast<size_t>(idx)].valid)
                    {
                        valid_batch_indices.push_back(idx);
                    }
                }

                if (valid_batch_indices.empty())
                {
                    epoch_task_done_.fetch_add(1);
                    continue;
                }

                std::sort(valid_batch_indices.begin(), valid_batch_indices.end(),
                          [this](int64_t a, int64_t b) {
                              const auto &ea = cached_[static_cast<size_t>(a)];
                              const auto &eb = cached_[static_cast<size_t>(b)];
                              return (static_cast<int64_t>(ea.crop_bgr.total()) * ea.crop_bgr.channels()) >
                                     (static_cast<int64_t>(eb.crop_bgr.total()) * eb.crop_bgr.channels());
                          });

                TrainingBatch batch = BuildBatch(valid_batch_indices);

                {
                    std::unique_lock<std::mutex> lock(result_space_mutex_);
                    result_space_cv_.wait(lock, [this]() {
                        return !active_.load() || result_queue_.Size() < max_prefetch_batches_;
                    });
                }

                if (!active_.load())
                {
                    epoch_task_done_.fetch_add(1);
                    break;
                }

                if (!result_queue_.Push(std::move(batch)))
                {
                    epoch_task_done_.fetch_add(1);
                    break;
                }

                epoch_task_done_.fetch_add(1);
            }
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                if (!worker_exception_)
                {
                    worker_exception_ = std::current_exception();
                }
            }
            active_.store(false);
            task_queue_.Close();
            result_queue_.Close();
            result_space_cv_.notify_all();
        }

        if (workers_alive_.fetch_sub(1) == 1)
        {
            result_queue_.Close();
        }
        result_space_cv_.notify_all();
    }

    TrainingBatch BuildBatch(const std::vector<int64_t> &valid_batch_indices) const
    {
        TrainingBatch batch;
        batch.sample_indices = valid_batch_indices;
        const bool use_pinned = device_.is_cuda();

        int64_t total_image_values = 0;
        int64_t total_mask_floats = 0;

        std::vector<float> y_sign_cpu;
        y_sign_cpu.reserve(valid_batch_indices.size());

        std::vector<torch::Tensor> view_mats_cpu;
        std::vector<torch::Tensor> proj_mats_cpu;
        view_mats_cpu.reserve(valid_batch_indices.size());
        proj_mats_cpu.reserve(valid_batch_indices.size());

        batch.image_slices.reserve(valid_batch_indices.size());
        batch.mask_slices.reserve(valid_batch_indices.size());
        batch.tan_fovx.reserve(valid_batch_indices.size());
        batch.tan_fovy.reserve(valid_batch_indices.size());

        for (const int64_t idx : valid_batch_indices)
        {
            const auto sample_index = static_cast<size_t>(idx);
            const auto &entry = cached_[sample_index];
            const auto &sample = samples_[sample_index];

            const int H = entry.crop_bgr.rows;
            const int W = entry.crop_bgr.cols;
            const int C = entry.crop_bgr.channels();

            PackedSlice3D image_slice;
            image_slice.offset = total_image_values;
            image_slice.length = static_cast<int64_t>(H) * static_cast<int64_t>(W) * static_cast<int64_t>(C);
            image_slice.shape = {
                static_cast<int64_t>(H),
                static_cast<int64_t>(W),
                static_cast<int64_t>(C)};
            batch.image_slices.push_back(image_slice);
            total_image_values += image_slice.length;

            PackedSlice3D mask_slice;
            mask_slice.offset = total_mask_floats;
            mask_slice.length = entry.matte_mask.numel();
            mask_slice.shape = {
                entry.matte_mask.size(0),
                entry.matte_mask.size(1),
                entry.matte_mask.size(2)};
            batch.mask_slices.push_back(mask_slice);
            total_mask_floats += mask_slice.length;

            y_sign_cpu.push_back(sample.y_sign);

            const float full_cx = static_cast<float>(sample.img_w) * 0.5f;
            const float full_cy = static_cast<float>(sample.img_h) * 0.5f;
            const float x0 = (sample.crop_w > 0.0f) ? sample.crop_x0 : (sample.crop_cx - static_cast<float>(W) * 0.5f);
            const float y0 = (sample.crop_h > 0.0f) ? sample.crop_y0 : (sample.crop_cy - static_cast<float>(H) * 0.5f);

            torch::Tensor view_cpu;
            torch::Tensor proj_cpu;
            float tan_fovx = 0.0f;
            float tan_fovy = 0.0f;
            std::tie(view_cpu, proj_cpu, tan_fovx, tan_fovy) =
                BuildProjection(sample.focal_length,
                                W,
                                H,
                                full_cx - x0,
                                full_cy - y0,
                                torch::kCPU);

            view_mats_cpu.push_back(view_cpu);
            proj_mats_cpu.push_back(proj_cpu);
            batch.tan_fovx.push_back(tan_fovx);
            batch.tan_fovy.push_back(tan_fovy);
        }

        auto image_cpu_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto mask_cpu_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        if (use_pinned)
        {
            image_cpu_opts = image_cpu_opts.pinned_memory(true);
            mask_cpu_opts = mask_cpu_opts.pinned_memory(true);
        }

        auto packed_images_cpu = torch::empty({total_image_values}, image_cpu_opts);
        auto packed_masks_cpu = torch::empty({total_mask_floats}, mask_cpu_opts);

        uint8_t *image_ptr = packed_images_cpu.data_ptr<uint8_t>();
        float *mask_ptr = packed_masks_cpu.data_ptr<float>();

        for (size_t i = 0; i < valid_batch_indices.size(); ++i)
        {
            const auto sample_index = static_cast<size_t>(valid_batch_indices[i]);
            const auto &entry = cached_[sample_index];

            const cv::Mat source = entry.crop_bgr.isContinuous() ? entry.crop_bgr : entry.crop_bgr.clone();
            std::memcpy(image_ptr + batch.image_slices[i].offset,
                        source.data,
                        static_cast<size_t>(batch.image_slices[i].length) * sizeof(uint8_t));
            std::memcpy(mask_ptr + batch.mask_slices[i].offset,
                        entry.matte_mask.data_ptr<float>(),
                        static_cast<size_t>(batch.mask_slices[i].length) * sizeof(float));
        }

        batch.packed_images_u8 = packed_images_cpu.to(device_, use_pinned);
        batch.packed_masks = packed_masks_cpu.to(device_, use_pinned);

        auto idx_cpu = torch::from_blob(
                           const_cast<int64_t *>(batch.sample_indices.data()),
                           {static_cast<int64_t>(batch.sample_indices.size())},
                           torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                           .clone();
        if (use_pinned)
        {
            idx_cpu = idx_cpu.pin_memory();
        }
        auto idx_tensor = idx_cpu.to(device_, use_pinned);

        batch.pose_base_batch = all_poses_.index_select(0, idx_tensor).view({-1, 24, 3});
        batch.trans_base_batch = all_trans_.index_select(0, idx_tensor).view({-1, 3});
        batch.time_tensor = all_time_.index_select(0, idx_tensor);
        batch.crop_params = all_crops_.index_select(0, idx_tensor).view({-1, 3});

        auto y_sign_tensor = torch::from_blob(
                                 y_sign_cpu.data(),
                                 {static_cast<int64_t>(y_sign_cpu.size())},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                                 .clone();
        if (use_pinned)
        {
            y_sign_tensor = y_sign_tensor.pin_memory();
        }
        batch.y_signs = y_sign_tensor.to(device_, use_pinned);

        auto view_stack = torch::stack(view_mats_cpu);
        auto proj_stack = torch::stack(proj_mats_cpu);
        if (use_pinned)
        {
            view_stack = view_stack.pin_memory();
            proj_stack = proj_stack.pin_memory();
        }
        batch.view_mats = view_stack.to(device_, use_pinned);
        batch.proj_mats = proj_stack.to(device_, use_pinned);

        return batch;
    }

    const std::vector<TrainSample> &samples_;
    const std::vector<CachedSampleData> &cached_;
    const torch::Tensor &all_poses_;
    const torch::Tensor &all_trans_;
    const torch::Tensor &all_time_;
    const torch::Tensor &all_crops_;
    int batch_size_ = 1;
    torch::Device device_;
    size_t max_prefetch_batches_ = 8;
    std::atomic<int64_t> epoch_task_total_{0};
    std::atomic<int64_t> epoch_task_done_{0};

    SafeQueue<std::vector<int64_t>> task_queue_;
    SafeQueue<TrainingBatch> result_queue_;
    std::vector<std::thread> workers_;

    std::atomic<bool> active_{false};
    std::atomic<int> workers_alive_{0};
    std::mutex result_space_mutex_;
    std::condition_variable result_space_cv_;

    std::mutex error_mutex_;
    std::exception_ptr worker_exception_;
};
