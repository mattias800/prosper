#pragma once
// Install a completed sibling file over its destination without first deleting that destination.
#include <filesystem>
#include <system_error>

namespace prosper::host {
void replace_file(const std::filesystem::path& source, const std::filesystem::path& destination,
                  std::error_code& error);
}   // namespace prosper::host
