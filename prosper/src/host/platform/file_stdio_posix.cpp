#include "host/platform/file_stdio.hpp"

#include <cerrno>

namespace prosper::host {

std::filesystem::path absolute_file_path(const std::filesystem::path& path,
                                         std::error_code& error) {
    error.clear();
    if (path.empty() || path.native().find('\0') != std::string::npos) {
        error = std::make_error_code(std::errc::invalid_argument);
        return {};
    }
    // Do not lexically normalize: symlink/../file must keep its real POSIX meaning.
    return std::filesystem::absolute(path, error);
}

std::filesystem::path native_file_path(const std::filesystem::path& path, std::error_code& error) {
    return absolute_file_path(path, error);
}

std::FILE* open_native_file(const std::filesystem::path& path, FileOpenMode mode,
                            std::error_code& error) {
    const auto native = native_file_path(path, error);
    if (error) return nullptr;
    const char* spelling = mode == FileOpenMode::ReadBinary    ? "rb"
                           : mode == FileOpenMode::WriteBinary ? "wb"
                                                               : "a";
    errno = 0;
    std::FILE* result = std::fopen(native.c_str(), spelling);
    if (!result)
        error = errno ? std::error_code(errno, std::generic_category())
                      : std::make_error_code(std::errc::io_error);
    return result;
}

} // namespace prosper::host
