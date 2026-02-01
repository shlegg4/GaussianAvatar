#pragma once

#include <string>
#include <vector>

#include "utils/train/TrainTypes.h"

bool ExtractStringField(const std::string &line, const std::string &key, std::string *out);
bool ExtractNumberField(const std::string &line, const std::string &key, double *out);
bool ExtractArrayField(const std::string &line, const std::string &key, std::vector<float> *out);
bool ParseTrainSample(const std::string &line, TrainSample *out);
