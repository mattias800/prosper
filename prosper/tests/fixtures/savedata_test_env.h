// Shared set-up for the savedata HLE tests: an environment setter that works on every host, and a
// minimal application root that declares a title id the way a real dump does.
//
// Going through set_app0_root() with a real sce_sys/param.json, rather than a test-only setter, is
// deliberate: it exercises the one param.json parse the shipping code uses, so a test built on this
// cannot pass against a title-id derivation the guest never sees.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace prosper_test {

// setenv/unsetenv on POSIX; _putenv_s on Windows, where an empty value removes the variable.
inline void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// Creates (replacing any previous one) `<base>/<name>/sce_sys/param.json` declaring `title_id`, and
// returns the application root to hand to set_app0_root().
inline std::string make_app0(const std::filesystem::path& base, const char* name,
                             const std::string& title_id) {
    const std::filesystem::path root = base / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "sce_sys", ec);
    std::ofstream p(root / "sce_sys" / "param.json", std::ios::binary);
    p << "{\"titleId\":\"" << title_id << "\"}";
    return root.string();
}

}   // namespace prosper_test
