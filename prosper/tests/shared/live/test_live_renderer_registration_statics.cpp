// #3892: pin WHEN the live renderer callback's prelude statics initialise, before anyone moves them
// into registration-owned state.
//
// The submit callback that register_live_renderer installs declares ~60 function-local statics.
// Moving them into a `CallbackState` owned by the registration would silently change when they
// initialise, and gpu_replay registers the renderer more than once per process. This test registers
// the live renderer twice in ONE process and pins the current contract:
//   1. The PROSPER_RENDER_DELAY_MS clock starts at the FIRST CALLBACK, not at registration, and the
//      variable is read there (set after registration, it still applies).
//   2. Statics declared after the callback's early returns initialise on the first submit INSIDE
//      the render window: the decoded-texture budget reads PROSPER_TEXTURE_DECODE_CACHE_MB then,
//      and prints its line then, not on a skipped submit.
//   3. A second registration reuses every process-lifetime static: the delay clock is not
//      restarted, neither variable is re-read, and neither the budget line nor the warmup
//      announcement prints again.
//   4. A callback on a new thread (whose `static thread_local` prelude state is fresh) still shares
//      the process-lifetime statics above: the gates are open and nothing initialises twice.
// A change that keeps these arms green keeps the behaviour the prelude has today. Needs its own
// process: every arm depends on the statics being untouched when it starts.
#include "fixtures/render_runner.h"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
#else
#include <unistd.h>
#endif

using namespace prosper::gpu;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    std::fflush(stdout);
    failures += !ok;
}

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// Captures this process's stderr (fd 2) into a file for the duration of one arm.
class StderrCapture {
public:
    explicit StderrCapture(std::string path) : path_(std::move(path)) {
        std::fflush(stderr);
        saved_ = dup(fileno(stderr));
        FILE* f = std::fopen(path_.c_str(), "w");
        if (f) { dup2(fileno(f), fileno(stderr)); std::fclose(f); }
    }
    std::string finish() {
        std::fflush(stderr);
        if (saved_ >= 0) { dup2(saved_, fileno(stderr)); saved_ = -1; }
        std::ifstream in(path_);
        std::stringstream text;
        text << in.rdbuf();
        std::fputs(text.str().c_str(), stderr);   // keep the log visible in the ctest output
        return text.str();
    }
    ~StderrCapture() { if (saved_ >= 0) finish(); }
private:
    std::string path_;
    int saved_ = -1;
};

static size_t count(const std::string& text, const char* needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++n;
    return n;
}

int main() {
    // Arms 1 and 2 need the variables absent at registration time; set them only where noted.
    // (The ctest registration runs this binary with a clean environment for both.)
    prosper::register_builtin_hle();
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0;
    constexpr size_t GuestBytes = 0x100000;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), GuestBytes, 2, 0,
                             reinterpret_cast<uint64_t>("registration-statics"), 0) == 0 && guest,
          "fixture maps tracked guest addresses");
    if (!guest) return 1;
    check(!std::getenv("PROSPER_RENDER_DELAY_MS") && !std::getenv("PROSPER_TEXTURE_DECODE_CACHE_MB"),
          "fixture starts with neither variable set");

    // A fullscreen red triangle into a 64x64 colour target: a real rendered submit whose result is
    // non-empty, so "rendered" and "skipped by the warmup gate" are distinguishable by the return.
    const uint32_t vs_words[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    const uint32_t fs_words[]{0x7E0002F2u, 0x7E020280u, 0x7E040280u,
        0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    constexpr uint32_t W = 64, H = 64;
    DrawItem draw;
    draw.vs = recompile_vertex(vs_words, std::size(vs_words));
    draw.fs = recompile_fragment(fs_words, std::size(fs_words));
    draw.vertex_count = 3;
    draw.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    draw.ps.color_write_mask = 0xf;
    draw.color0_base = guest;
    draw.color0_width = W; draw.color0_height = H;
    check(!draw.vs.empty() && !draw.fs.empty(), "fullscreen shaders compile");
    if (draw.vs.empty() || draw.fs.empty()) return 1;
    auto rendered = [](const std::vector<uint8_t>& pixels) {
        return pixels.size() == size_t(W) * H * 4 && pixels[0] == 255 && pixels[2] == 0;
    };
    const std::string log_dir = std::getenv("PROSPER_TEST_LOG_DIR")
        ? std::getenv("PROSPER_TEST_LOG_DIR") : ".";

    // ---- First registration. The delay variable is set AFTER registration: arm 1 pins that the
    // callback, not register_live_renderer, reads it.
    prosper::frontend::register_live_renderer(".", false);
    set_env("PROSPER_RENDER_DELAY_MS", "400");
    set_env("PROSPER_TEXTURE_DECODE_CACHE_MB", "77");
    // Longer than the delay: if registration started the clock, the first callback would render.
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    {
        StderrCapture capture(log_dir + "/registration_statics_skip.log");
        const auto skipped = render_submit_items({draw}, W, H);
        const std::string log = capture.finish();
        check(skipped.empty(),
              "1: the delay clock starts at the first callback, not at registration "
              "(600 ms after registering, the first submit is still inside a 400 ms warmup)");
        check(count(log, "decoded texture cache budget") == 0,
              "2: a submit skipped by the warmup gate does not initialise the post-gate statics");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    {
        StderrCapture capture(log_dir + "/registration_statics_first.log");
        const auto first = render_submit_items({draw}, W, H);
        const std::string log = capture.finish();
        check(rendered(first), "the first submit after the warmup renders");
        check(count(log, "wall-clock warmup complete after") == 1 &&
                  log.find("at submit 1\n") != std::string::npos,
              "1: the warmup ends at the process's second submit (ordinal 1)");
        check(count(log, "decoded texture cache budget = 77.0 MiB") == 1,
              "2: the first in-window submit reads PROSPER_TEXTURE_DECODE_CACHE_MB and prints the "
              "budget line once");
    }

    // ---- Second registration, as gpu_replay does. Both variables change first; neither may be
    // re-read, and the delay clock must not restart.
    set_env("PROSPER_RENDER_DELAY_MS", "60000");
    set_env("PROSPER_TEXTURE_DECODE_CACHE_MB", "33");
    prosper::frontend::register_live_renderer(".", false);
    {
        StderrCapture capture(log_dir + "/registration_statics_second.log");
        const auto again = render_submit_items({draw}, W, H);
        const std::string log = capture.finish();
        check(rendered(again),
              "3: a second registration reuses the delay clock and value (a 60 s delay set before "
              "it is not re-read)");
        check(count(log, "decoded texture cache budget") == 0,
              "3: the budget static is process-lifetime: no second budget line, 33 MiB never read");
        check(count(log, "wall-clock warmup complete after") == 0,
              "3: the warmup announcement is process-lifetime and does not repeat");
    }

    // ---- A callback on a new thread: its thread_local prelude state starts fresh, while the
    // process-lifetime statics above must still apply.
    {
        StderrCapture capture(log_dir + "/registration_statics_thread.log");
        std::vector<uint8_t> threaded;
        std::thread worker([&] { threaded = render_submit_items({draw}, W, H); });
        worker.join();
        const std::string log = capture.finish();
        check(rendered(threaded),
              "4: a callback on a new thread renders with the process-lifetime gates already open");
        check(count(log, "decoded texture cache budget") == 0,
              "4: process-lifetime statics are shared across threads (no budget line on the new "
              "thread)");
    }

    unmap(guest, GuestBytes, 0, 0, 0, 0);
    std::printf("live renderer registration statics: %d failures\n", failures);
    return failures ? 1 : 0;
}
