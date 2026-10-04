#include "host/platform/file_stdio.hpp"

#include <cerrno>
#include <string>

namespace prosper::host {

std::filesystem::path absolute_file_path(const std::filesystem::path& path,
                                         std::error_code& error) {
    namespace fs = std::filesystem;
    error.clear();
    const auto& spelling = path.native();
    if (spelling.empty() || spelling.find(L'\0') != std::wstring::npos) {
        error = std::make_error_code(std::errc::invalid_argument);
        return {};
    }
    if (spelling.starts_with(L"\\\\?\\")) return path;
    if (spelling.starts_with(L"\\\\.\\")) {
        error = std::make_error_code(std::errc::operation_not_supported);
        return {};
    }
    fs::path absolute = path;
    if (!absolute.is_absolute()) {
        // C:relative uses per-drive state, not the one captured cwd. Do not silently reinterpret it.
        if (absolute.has_root_name()) {
            error = std::make_error_code(std::errc::operation_not_supported);
            return {};
        }
        const fs::path cwd = fs::current_path(error);
        if (error) return {};
        absolute = absolute.has_root_directory() ? cwd.root_name() / absolute : cwd / absolute;
    }
    return absolute.lexically_normal().make_preferred();
}

std::filesystem::path native_file_path(const std::filesystem::path& path, std::error_code& error) {
    namespace fs = std::filesystem;
    const auto absolute = absolute_file_path(path, error);
    if (error) return {};
    const std::wstring full = absolute.native();
    if (full.starts_with(L"\\\\?\\")) return absolute;
    if (full.starts_with(L"\\\\")) return fs::path(L"\\\\?\\UNC\\" + full.substr(2));
    if (full.size() >= 3 && full[1] == L':' && full[2] == L'\\') return fs::path(L"\\\\?\\" + full);
    error = std::make_error_code(std::errc::operation_not_supported);
    return {};
}

std::FILE* open_native_file(const std::filesystem::path& path, FileOpenMode mode,
                            std::error_code& error) {
    const auto native = native_file_path(path, error);
    if (error) return nullptr;
    const wchar_t* spelling = mode == FileOpenMode::ReadBinary    ? L"rb"
                              : mode == FileOpenMode::WriteBinary ? L"wb"
                                                                  : L"a";
    errno = 0;
    std::FILE* result = ::_wfopen(native.c_str(), spelling);
    if (!result)
        error = errno ? std::error_code(errno, std::generic_category())
                      : std::make_error_code(std::errc::io_error);
    return result;
}

}   // namespace prosper::host
