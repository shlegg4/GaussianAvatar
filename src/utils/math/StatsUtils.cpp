#include "utils/math/StatsUtils.h"

#include <algorithm>
#include <cmath>

float ComputeMedian(std::vector<float> values)
{
    if (values.empty())
        return 0.0f;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 1)
        return values[mid];
    return 0.5f * (values[mid - 1] + values[mid]);
}

float ComputeMad(const std::vector<float> &values, float median)
{
    std::vector<float> deviations;
    deviations.reserve(values.size());
    for (float v : values)
    {
        deviations.push_back(std::abs(v - median));
    }
    return ComputeMedian(std::move(deviations));
}

std::vector<float> ComputeAverageBetas(const std::vector<TrainSample> &samples)
{
    std::vector<float> sum;
    size_t count = 0;
    for (const auto &sample : samples)
    {
        if (sample.betas.empty())
            continue;
        if (sum.empty())
        {
            sum.assign(sample.betas.size(), 0.0f);
        }
        if (sample.betas.size() != sum.size())
            continue;
        for (size_t i = 0; i < sum.size(); ++i)
        {
            sum[i] += sample.betas[i];
        }
        count++;
    }
    if (count == 0)
        return {};
    for (float &v : sum)
    {
        v /= static_cast<float>(count);
    }
    return sum;
}
