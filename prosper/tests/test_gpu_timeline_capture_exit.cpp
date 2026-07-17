#include "../src/gpu/gpu_timeline.hpp"

#include <cstdio>
#include <cstdlib>
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

static int run_child() {
    set_test_env("PROSPER_GPU_TIMELINE", "timeline-capture-exit.prgtl");
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE", "timeline-capture-endpoint.prgcap");
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "2");
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR",
                 "timeline-capture-predecessor.prgcap");
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
    return 0; // EXIT_AFTER_CAPTURE must have terminated with status 2 before this point.
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--child") return run_child();
    const std::string command = std::string("\"") + argv[0] + "\" --child";
    const int status = std::system(command.c_str());
#ifdef _WIN32
    const int exit_code = status;
#else
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    if (exit_code != 2) {
        std::fprintf(stderr, "expected child exit 2 after predecessor capture failure, got %d\n",
                     exit_code);
        return 1;
    }
    return 0;
}
