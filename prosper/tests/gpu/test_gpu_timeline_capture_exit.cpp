#include "gpu/timeline/gpu_timeline.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "fixtures/test_scratch.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace prosper::gpu;

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// #1621: the parent re-invokes itself as `argv[0] --child` and then reads a capture the CHILD wrote,
// so the two processes must agree on a directory. `test_scratch_dir()` is per-PID, so letting each
// side derive its own would put the child's files where the parent does not look — and the symptom
// would be "the ordered-DMA predecessor capture is incomplete", i.e. a content assertion failing on
// correct code, which is the exact failure class tests/fixtures/test_scratch.h exists to remove. The parent
// therefore computes the directory and hands it over; `std::system` inherits the environment.
// (PROSPER_TEST_SCRATCH_DIR is inherited too, but the child would append its OWN pid under it.)
static const char* kArtifactDirEnv = "PROSPER_TEST_TIMELINE_CAPTURE_DIR";

// The three artifact paths, rooted at `dir`. Both processes build them the same way from the same
// directory, so the names stay in one place.
struct ArtifactPaths {
    std::string timeline, endpoint, predecessor;
};

static ArtifactPaths artifact_paths(const std::filesystem::path& dir) {
    return {(dir / "timeline-capture-exit.prgtl").string(),
            (dir / "timeline-capture-endpoint.prgcap").string(),
            (dir / "timeline-capture-predecessor.prgcap").string()};
}

static int run_child() {
    // Handed down by the parent. Falling back to this process's own scratch directory keeps a
    // hand-run `test_gpu_timeline_capture_exit --child` working off the tmpfs.
    const char* dir = std::getenv(kArtifactDirEnv);
    const ArtifactPaths paths = artifact_paths(
        dir != nullptr && *dir != '\0' ? std::filesystem::path(dir) : prosper_test::test_scratch_dir());

    set_test_env("PROSPER_GPU_TIMELINE", paths.timeline.c_str());
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE", paths.endpoint.c_str());
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "2");
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR", paths.predecessor.c_str());
    set_test_env("PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE", "1");

    GpuState predecessor;
    predecessor.command_order = 10;
    predecessor.dma_copies.push_back({0x200000, 0x100000000ull, 16, 0, 10, 0});
    begin_gpu_timeline_submit(1);
    record_gpu_timeline_submit(predecessor, 1);

    GpuState endpoint;
    endpoint.command_order = 20;
    begin_gpu_timeline_submit(2);
    record_gpu_timeline_submit(endpoint, 2);
    return 3; // EXIT_AFTER_CAPTURE must have terminated successfully before this point.
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--child") return run_child();

    const std::filesystem::path dir = prosper_test::test_scratch_dir();
    const ArtifactPaths paths = artifact_paths(dir);
    set_test_env(kArtifactDirEnv, dir.string().c_str());

    const std::string command = std::string("\"") + argv[0] + "\" --child";
    const int status = std::system(command.c_str());
#ifdef _WIN32
    const int exit_code = status;
#else
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    if (exit_code != 0) {
        std::fprintf(stderr, "expected child exit 0 after ordered-DMA capture, got %d\n",
                     exit_code);
        return 1;
    }
    GpuCaptureFile predecessor;
    std::string error;
    if (!read_gpu_capture(paths.predecessor, predecessor, error) ||
        predecessor.dma_copies.size() != 1 || predecessor.operations.size() != 1 ||
        predecessor.operations[0].kind != SubmitOperationKind::DmaCopy) {
        std::fprintf(stderr, "ordered-DMA predecessor capture is incomplete: %s\n", error.c_str());
        return 1;
    }
    std::error_code ec;
    std::filesystem::remove(paths.timeline, ec);
    std::filesystem::remove(paths.endpoint, ec);
    std::filesystem::remove(paths.predecessor, ec);
    return 0;
}
