#pragma once
#include <optional>
#include <string>

namespace prosper {
std::optional<std::string> sandbox_normalize_subpath(const std::string& sub);
std::string clamp_normalize_relative(const std::string& sub);
bool mount_path_matches(const std::string& path, const char* mount);
bool has_dot_component(const std::string& p);
} // namespace prosper
