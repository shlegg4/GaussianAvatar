#include "utils/io/PathUtils.h"

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
