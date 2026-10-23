#include "host/platform/file_replace.hpp"
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

namespace prosper::host {
void replace_file(const std::filesystem::path& source, const std::filesystem::path& destination,
                  std::error_code& error) {
#ifdef _WIN32
    if (MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING) != 0)
        error.clear();
    else
        error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
#else
    std::filesystem::rename(source, destination, error);
#endif
}
}   // namespace prosper::host
