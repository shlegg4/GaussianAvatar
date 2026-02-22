#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <vector>

#include "utils/train/TrainCache.h"
#include "utils/train/TrainTypes.h"

struct PoseJointExport
{
    std::array<float, 3> rot{{0.0f, 0.0f, 0.0f}};
    std::array<float, 3> transl{{0.0f, 0.0f, 0.0f}};
};

struct PoseSampleExport
{
    bool valid = false;
    std::array<float, 3> original_transl{{0.0f, 0.0f, 0.0f}};
    std::array<float, 3> delta_transl{{0.0f, 0.0f, 0.0f}};
    std::array<float, 3> refined_transl{{0.0f, 0.0f, 0.0f}};
    std::array<PoseJointExport, 24> original_pose{};
    std::array<PoseJointExport, 24> pose_delta{};
    std::array<PoseJointExport, 24> refined_pose{};
};

struct RenderViewResult
{
    torch::Tensor image;
    PoseSampleExport pose_export;
};

using RenderViewFn = std::function<RenderViewResult(size_t sample_index,
                                                    const TrainSample &sample,
                                                    const CachedSampleData &cached_entry)>;

int SaveEpochViewPairs(const std::vector<TrainSample> &samples,
                       const std::vector<CachedSampleData> &cached,
                       const std::filesystem::path &output_dir,
                       int epoch,
                       const RenderViewFn &render_fn);
