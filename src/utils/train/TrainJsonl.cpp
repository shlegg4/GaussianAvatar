#include "utils/train/TrainJsonl.h"

#include <cstdlib>
#include <utility>

bool ExtractStringField(const std::string &line, const std::string &key, std::string *out)
{
    const std::string tag = "\"" + key + "\":\"";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    size_t end = line.find('"', start);
    if (end == std::string::npos)
        return false;
    *out = line.substr(start, end - start);
    return true;
}

bool ExtractNumberField(const std::string &line, const std::string &key, double *out)
{
    const std::string tag = "\"" + key + "\":";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    const char *ptr = line.c_str() + start;
    char *end = nullptr;
    double val = std::strtod(ptr, &end);
    if (end == ptr)
        return false;
    *out = val;
    return true;
}

bool ExtractArrayField(const std::string &line, const std::string &key, std::vector<float> *out)
{
    const std::string tag = "\"" + key + "\":[";
    size_t start = line.find(tag);
    if (start == std::string::npos)
        return false;
    start += tag.size();
    size_t end = line.find(']', start);
    if (end == std::string::npos)
        return false;
    std::string content = line.substr(start, end - start);
    out->clear();
    if (content.empty())
        return true;
    size_t pos = 0;
    while (pos < content.size())
    {
        size_t comma = content.find(',', pos);
        std::string token = (comma == std::string::npos) ? content.substr(pos) : content.substr(pos, comma - pos);
        if (!token.empty())
        {
            char *end_ptr = nullptr;
            float val = std::strtof(token.c_str(), &end_ptr);
            if (end_ptr != token.c_str())
            {
                out->push_back(val);
            }
        }
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return true;
}

bool ParseTrainSample(const std::string &line, TrainSample *out)
{
    TrainSample sample;
    double value = 0.0;
    if (ExtractNumberField(line, "frame", &value))
    {
        sample.frame = static_cast<int>(value);
    }
    if (!ExtractStringField(line, "crop", &sample.crop_path))
        return false;
    if (sample.crop_path.empty())
        return false;

    if (!ExtractNumberField(line, "img_w", &value))
        return false;
    sample.img_w = static_cast<int>(value);
    if (!ExtractNumberField(line, "img_h", &value))
        return false;
    sample.img_h = static_cast<int>(value);
    if (!ExtractNumberField(line, "crop_cx", &value))
        return false;
    sample.crop_cx = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_cy", &value))
        return false;
    sample.crop_cy = static_cast<float>(value);
    if (!ExtractNumberField(line, "crop_size", &value))
        return false;
    sample.crop_size = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_x0", &value))
        sample.crop_x0 = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_y0", &value))
        sample.crop_y0 = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_w", &value))
        sample.crop_w = static_cast<float>(value);
    if (ExtractNumberField(line, "crop_h", &value))
        sample.crop_h = static_cast<float>(value);
    if (!ExtractNumberField(line, "focal_length", &value))
        return false;
    sample.focal_length = static_cast<float>(value);
    if (!ExtractNumberField(line, "y_sign", &value))
        return false;
    sample.y_sign = static_cast<float>(value);

    if (!ExtractArrayField(line, "pose", &sample.pose))
        return false;
    if (!ExtractArrayField(line, "betas", &sample.betas))
        return false;
    if (!ExtractArrayField(line, "cam", &sample.cam))
        return false;

    if (sample.pose.empty() || sample.betas.empty() || sample.cam.empty())
        return false;
    *out = std::move(sample);
    return true;
}
