#include "host/image/guest_executable.hpp"

#include "host/image/boot_program.hpp"
#include "host/image/module_path_policy.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <optional>

namespace prosper {
namespace {
std::mutex argv_mutex;
std::optional<std::vector<std::string>> entry_argv;

bool main_path_allowed(const std::filesystem::path& relative) {
    for (const auto& component : relative) {
        std::string word = component.string();
        for (char& c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (word == ".." || word == "." || word == "fakelib" || word.find(':') != std::string::npos)
            return false;
    }
    std::string leaf = relative.filename().string();
    for (char& c : leaf) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (leaf.rfind(kSonyLibraryPrefix, 0) == 0) return false;
    for (const char* prefix : kRootRefusedPlatformPrefixes)
        if (leaf.rfind(prefix, 0) == 0) return false;
    return leaf == "eboot.bin" ||
           (leaf.size() > 4 && leaf.compare(leaf.size() - 4, 4, ".elf") == 0);
}
}   // namespace

std::string resolve_guest_executable(const std::string& dump_root, const std::string& guest_path,
                                     std::string* error) {
    namespace fs = std::filesystem;
    const auto fail = [&](const char* why) {
        if (error) *error = why;
        return std::string{};
    };
    if (dump_root.empty() || guest_path.empty()) return fail("empty app0 or executable path");
    std::string path = guest_path;
    std::replace(path.begin(), path.end(), '\\', '/');
    if (path.front() != '/') path = "/app0/" + path;
    if (path.compare(0, 6, "/app0/") != 0) return fail("executable is outside app0");
    const fs::path relative(path.substr(6));
    if (relative.is_absolute() || !main_path_allowed(relative))
        return fail("executable path contains a refused component or filename");
    std::error_code ec;
    const fs::path root = fs::canonical(fs::path(dump_root), ec);
    if (ec) return fail("app0 is unavailable");
    const fs::path host = fs::canonical(resolve_host_path_case((root / relative).string()), ec);
    if (ec || !fs::is_regular_file(host, ec)) return fail("executable is unavailable");
    const fs::path contained = host.lexically_relative(root);
    if (contained.empty() || contained.is_absolute() || !main_path_allowed(contained))
        return fail("executable symlink escapes app0 or selects a refused filename");
    return host.string();
}

void set_guest_entry_arguments(const std::vector<std::string>& arguments) {
    std::lock_guard lock(argv_mutex);
    entry_argv = arguments;
}

std::vector<std::string> guest_entry_arguments(const char* configured_arguments) {
    std::lock_guard lock(argv_mutex);
    if (entry_argv) return *entry_argv;
    std::vector<std::string> args{"/app0/eboot.bin"};
    if (configured_arguments) {
        const char* p = configured_arguments;
        while (*p) {
            while (*p == ' ') ++p;
            if (!*p) break;
            const char* begin = p;
            while (*p && *p != ' ') ++p;
            args.emplace_back(begin, static_cast<size_t>(p - begin));
        }
    }
    return args;
}
} // namespace prosper
