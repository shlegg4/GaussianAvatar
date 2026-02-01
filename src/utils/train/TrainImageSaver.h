#pragma once

#include <filesystem>
#include <functional>
#include <vector>

#include "utils/train/TrainCache.h"
#include "utils/train/TrainTypes.h"

using RenderViewFn = std::function<torch::Tensor(size_t sample_index,
                                                 const TrainSample &sample,
                                                 const CachedSampleData &cached_entry)>;

int SaveEpochViewPairs(const std::vector<TrainSample> &samples,
                       const std::vector<CachedSampleData> &cached,
                       const std::filesystem::path &output_dir,
                       int epoch,
                       const RenderViewFn &render_fn);
