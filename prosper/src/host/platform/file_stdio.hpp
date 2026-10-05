// Native host paths for diagnostic stdio, not guest-path translation (#4378).
// Windows uses absolute extended drive/UNC paths and wide stdio without changing registry or
// manifest settings. POSIX preserves native bytes and symlink-sensitive components. Callers own
// returned FILEs and must check writes and close separately from a successful open.
// Inputs use std::filesystem::path's native spelling (wide on Windows, bytes on POSIX). This
// interface does not reinterpret a caller's narrow configuration string as UTF-8 or change its
// encoding policy; callers wanting Unicode use an explicitly constructed native/Unicode path.
#pragma once

#include <cstdio>
#include <filesystem>
#include <system_error>

namespace prosper::host {

enum class FileOpenMode { ReadBinary, WriteBinary, AppendText };

// Freeze an ordinary relative path against the current directory once. On Windows, ambiguous
// drive-relative and ordinary Win32 \\.\ device paths are unsupported. Existing \\?\ spellings
// (including caller-supplied extended namespaces) pass through unchanged: this does not validate
// their availability or interpret their namespace semantics. Empty/embedded-NUL paths fail.
std::filesystem::path absolute_file_path(const std::filesystem::path& path, std::error_code& error);
std::filesystem::path native_file_path(const std::filesystem::path& path, std::error_code& error);
std::FILE* open_native_file(const std::filesystem::path& path, FileOpenMode mode,
                            std::error_code& error);

} // namespace prosper::host
