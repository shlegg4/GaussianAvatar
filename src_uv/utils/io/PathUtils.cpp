#include "utils/io/PathUtils.h"

#include <array>
#include <filesystem>

bool ReplaceFirst(std::string *value, const std::string &from, const std::string &to)
{
    size_t pos = value->find(from);
    if (pos == std::string::npos)
        return false;
    value->replace(pos, from.size(), to);
    return true;
}

std::string DeriveMattePath(const std::string &crop_path)
{
    std::string matte_path = crop_path;
    if (!ReplaceFirst(&matte_path, "\\crops\\", "\\mattes\\"))
    {
        ReplaceFirst(&matte_path, "/crops/", "/mattes/");
    }
    if (!ReplaceFirst(&matte_path, "crop_", "matte_"))
    {
        ReplaceFirst(&matte_path, "crop-", "matte-");
    }
    return matte_path;
}

std::string DeriveFullFramePath(const std::string &crop_path)
{
    std::filesystem::path full_path(crop_path);
    std::string normalized = full_path.generic_string();

    if (!ReplaceFirst(&normalized, "/crops/", "/overlays_full/"))
    {
        if (!ReplaceFirst(&normalized, "\\crops\\", "\\overlays_full\\"))
        {
            return std::string();
        }
    }

    std::filesystem::path candidate_path(normalized);
    std::string stem = candidate_path.stem().string();
    if (!ReplaceFirst(&stem, "crop_", "frame_"))
    {
        return std::string();
    }

    const size_t person_suffix = stem.rfind("_p");
    if (person_suffix != std::string::npos)
    {
        stem = stem.substr(0, person_suffix);
    }

    const std::array<const char *, 3> exts = {".jpg", ".png", ".jpeg"};
    std::error_code ec;
    for (const char *ext : exts)
    {
        std::filesystem::path trial = candidate_path.parent_path() / (stem + ext);
        if (std::filesystem::exists(trial, ec))
        {
            return trial.string();
        }
        ec.clear();
    }

    return std::string();
}
