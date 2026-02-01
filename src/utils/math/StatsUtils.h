#pragma once

#include <vector>

#include "utils/train/TrainTypes.h"

float ComputeMedian(std::vector<float> values);
float ComputeMad(const std::vector<float> &values, float median);
std::vector<float> ComputeAverageBetas(const std::vector<TrainSample> &samples);
