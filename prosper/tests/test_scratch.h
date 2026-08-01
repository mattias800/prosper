// Per-process scratch paths for tests that must touch the real filesystem (#1613).
//
// Two rules this exists to enforce, both learned the hard way:
//
//  1. **Never `/tmp`.** On the Linux dev box `/tmp` is a RAM-backed tmpfs with a per-user quota
//     shared by every concurrent agent. A `.prgcap` is 200 MB-2.7 GB, so a couple of capture tests
//     can exhaust it, and an exhausted tmpfs does not merely fail the write — it takes down the
//     tooling for everyone on the machine. `std::filesystem::temp_directory_path()` resolves to
//     exactly that tmpfs unless `TMPDIR` happens to be set, so it is not a safe default either.
//
//  2. **Never a path two processes can both pick.** ctest runs cases in parallel under `-j`, the
//     same test binary can be registered as several ctest cases with different environments
//     (`game_compute_exec` / `game_compute_exec_no_adaptive_storage_result` are one binary), and
//     several worktrees on this box run ctest at the same time. A fixed path is then a data race
//     whose symptom is a *content-comparison* assertion failing on correct code — the failure mode
//     most likely to send a reader hunting for a capture or GDS defect that does not exist.
//
// The root comes from `PROSPER_TEST_SCRATCH_DIR`, which `prosper/CMakeLists.txt` sets per ctest
// case to `<build-dir>/test-scratch/<ctest case name>`; running a test binary by hand instead uses
// `<cwd>/prosper-test-scratch/<binary or fallback name>`. A pid component underneath that keeps two
// concurrent ctest invocations (two worktrees, or a manual run alongside one) apart, and the
// directory is removed when the process exits normally — including after a failed CHECK, because
// the test harness counts failures and returns from main rather than aborting.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace prosper_test {

namespace detail {

inline int scratch_process_id() {
#if defined(_WIN32)
    return static_cast<int>(::_getpid());
#else
    return static_cast<int>(::getpid());
#endif
}

// Creates the directory on construction and removes it (and everything under it) on destruction.
// Held by a function-local static so the removal runs during normal process exit.
struct ScratchDirectory {
    std::filesystem::path path;

    ScratchDirectory() {
        std::filesystem::path root;
        if (const char* configured = std::getenv("PROSPER_TEST_SCRATCH_DIR");
            configured != nullptr && *configured != '\0') {
            root = configured;
        } else {
            // Hand-run binary: keep artifacts next to the invocation instead of on the tmpfs.
            std::error_code cwd_error;
            std::filesystem::path cwd = std::filesystem::current_path(cwd_error);
            if (cwd_error) cwd = std::filesystem::path(".");
            root = cwd / "prosper-test-scratch";
        }
        path = root / ("pid-" + std::to_string(scratch_process_id()));
        std::error_code error;
        std::filesystem::remove_all(path, error);   // debris from a recycled pid
        error.clear();
        std::filesystem::create_directories(path, error);
        // Fail LOUDLY and immediately. If the directory does not exist, every test below writes to a
        // path that cannot be opened and then fails on a *content* assertion — which is precisely the
        // mystifying failure class this header exists to remove (#1613). A read-only or full build
        // directory must not masquerade as a capture defect.
        if (error || !std::filesystem::is_directory(path)) {
            std::fprintf(stderr,
                         "FATAL: cannot create the test scratch directory '%s': %s\n"
                         "       (root came from %s)\n",
                         path.string().c_str(), error.message().c_str(),
                         std::getenv("PROSPER_TEST_SCRATCH_DIR") != nullptr
                             ? "PROSPER_TEST_SCRATCH_DIR"
                             : "the current working directory");
            std::fflush(stderr);
            std::abort();
        }
    }

    // NOTE for a future forking test: `path` is captured at construction, not re-derived from the
    // live pid, so a forked child that returns from main (or calls exit()) would delete its
    // *parent's* directory. Call test_scratch_dir() only after the fork, or have the child leave via
    // _exit(). Nothing in the suite forks today; both remove_all calls below are bounded to a
    // `pid-<n>` leaf, so even a mis-set PROSPER_TEST_SCRATCH_DIR cannot widen the deletion.
    ~ScratchDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
        // Courtesy: drop the per-case root too, but only via the non-recursive remove, which fails
        // harmlessly while a concurrent process still holds a pid directory underneath it.
        std::filesystem::remove(path.parent_path(), error);
    }

    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;
};

}   // namespace detail

// The calling process's private scratch directory. Created on first use, removed at normal exit.
inline const std::filesystem::path& test_scratch_dir() {
    static const detail::ScratchDirectory directory;
    return directory.path;
}

// A path for `name` inside this process's scratch directory. `name` may contain '/' to nest, in
// which case the caller creates the intermediate directories.
inline std::filesystem::path test_scratch_path(std::string_view name) {
    return test_scratch_dir() / std::filesystem::path(std::string(name));
}

// Convenience for the common `write it, read it back, compare` shape.
inline std::string test_scratch_file(std::string_view name) {
    return test_scratch_path(name).string();
}

}   // namespace prosper_test
