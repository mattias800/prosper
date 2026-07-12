// screenshot — run a game and capture a PNG every N frames until M screenshots, then exit.
//
// Reuses the shared boot path (boot_program) and the shared live renderer (frontends/shared), so the
// game boots and composites exactly as boot_trace / prosper-app do; this tool just samples the
// present layer (present_readback) periodically and writes PNGs. Frames are counted as rendered frames
// (present_frame_seq). Files are named <titleCode>_<runTimestamp>_<index>.png so every screenshot from
// one run shares a prefix and sorts together in a folder of many runs.
//
//   screenshot <app0-dir> [--every N] [--count M] [--out DIR] [--timeout SECS]
//              [--warmup-seconds S] [--warmup-submits N]
//     <app0-dir>   REQUIRED. The game dump root (e.g. .../PPSA24651-app0). Title code = its basename
//                  with a trailing "-app0" stripped.
//     --every N    rendered frames between screenshots       (default 60)
//     --count M    number of screenshots to take, then exit  (default 30)
//     --out DIR    output directory                          (default ".")
//     --timeout S  give up after S seconds if the game isn't rendering enough (default 900; 0 = none)
//     --warmup-seconds S  skip Vulkan rendering for S seconds before capture
//     --warmup-submits N  skip Vulkan rendering before submit N
//
// Reaching a rendering frame loop needs the guest switches the render frontier documents; this tool
// defaults PROSPER_GUEST_FS=1 and PROSPER_GUEST_ARGS=-force-gfx-direct (Unity/Messenger recipe) if
// they aren't already set. For other titles, set the appropriate env before running (e.g. a UE4
// title: PROSPER_GUEST_ARGS= PROSPER_NULL_PAGE=1). Linux-only (the guest substrate is).
#include "loader/linker.hpp"          // Program
#include "host/boot_program.hpp"       // boot_program
#include "host/exec_image.hpp"         // run_entry
#include "gpu/videoout_present.hpp"    // present_count / present_readback / present_width/height
#include "live_renderer.hpp"           // register_live_renderer (frontends/shared)

#include <zlib.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>
#include <thread>
#include <chrono>
#include <ctime>
#include <unistd.h>

using namespace prosper;

namespace {

bool parse_nonnegative_double(const char* text, double& value) {
    char* end = nullptr;
    errno = 0;
    const double parsed = strtod(text, &end);
    if (errno || end == text || *end != '\0' || parsed < 0.0 || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

bool parse_nonnegative_int(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

void put_chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t hdr[4] = { (uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8), (uint8_t)len };
    fwrite(hdr, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);
    uint32_t crc = crc32(0, (const Bytef*)type, 4);
    if (len) crc = crc32(crc, (const Bytef*)data, (uInt)len);
    uint8_t crcb[4] = { (uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc };
    fwrite(crcb, 1, 4, f);
}

// Write w*h RGBA8 as an 8-bit RGBA PNG (filter None per row, zlib-compressed IDAT).
bool write_png(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h,
               std::string& error) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (1 + (size_t)w * 4));
    for (uint32_t y = 0; y < h; y++) {
        raw.push_back(0);   // filter: None
        raw.insert(raw.end(), rgba + (size_t)y * w * 4, rgba + (size_t)(y + 1) * w * 4);
    }
    uLongf clen = compressBound((uLong)raw.size());
    std::vector<uint8_t> comp(clen);
    const int zrc = compress2(comp.data(), &clen, raw.data(), (uLong)raw.size(), Z_BEST_SPEED);
    if (zrc != Z_OK) {
        error = "zlib compress2 failed (" + std::to_string(zrc) + ")";
        return false;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        error = std::string("open failed: ") + strerror(errno);
        return false;
    }
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13] = {
        (uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
        (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h,
        8,   // bit depth
        6,   // colour type: RGBA
        0, 0, 0 };
    put_chunk(f, "IHDR", ihdr, 13);
    put_chunk(f, "IDAT", comp.data(), clen);
    put_chunk(f, "IEND", nullptr, 0);
    if (ferror(f)) {
        error = std::string("write failed: ") + strerror(errno);
        fclose(f);
        return false;
    }
    if (fclose(f) != 0) {
        error = std::string("close failed: ") + strerror(errno);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string dump, out = ".";
    int every = 60, count = 30, timeout = 900;
    double seconds = 0.0;   // >0 => capture every N wall-clock seconds instead of every N rendered frames
    double warmup_seconds = -1.0;  // <0 => preserve the environment; explicit values override it
    int warmup_submits = -1;
    bool warmup_seconds_set = false, warmup_submits_set = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--every"   && i + 1 < argc) every = atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
        else if (a == "--count"   && i + 1 < argc) count = atoi(argv[++i]);
        else if (a == "--out"     && i + 1 < argc) out   = argv[++i];
        else if (a == "--timeout" && i + 1 < argc) timeout = atoi(argv[++i]);
        else if (a == "--warmup-seconds" && i + 1 < argc) {
            warmup_seconds_set = true;
            if (!parse_nonnegative_double(argv[++i], warmup_seconds)) {
                fprintf(stderr, "screenshot: --warmup-seconds requires a non-negative number\n");
                return 2;
            }
        }
        else if (a == "--warmup-submits" && i + 1 < argc) {
            warmup_submits_set = true;
            if (!parse_nonnegative_int(argv[++i], warmup_submits)) {
                fprintf(stderr, "screenshot: --warmup-submits requires a non-negative integer\n");
                return 2;
            }
        }
        else if (!a.empty() && a[0] != '-' && dump.empty()) dump = a;
        else { fprintf(stderr, "screenshot: unknown/duplicate arg '%s'\n", a.c_str()); return 2; }
    }
    if (dump.empty()) {
        fprintf(stderr, "usage: screenshot <app0-dir> [--every N=60 | --seconds S] [--count M=30] [--out DIR] [--timeout SECS] [--warmup-seconds S] [--warmup-submits N]\n");
        return 2;
    }
    if (every < 1) every = 1;
    if (count < 1) count = 1;

    // Title code = dump basename with a trailing "-app0" removed (e.g. PPSA24651-app0 -> PPSA24651).
    std::string code = dump;
    if (auto sl = code.find_last_of("/\\"); sl != std::string::npos) code = code.substr(sl + 1);
    if (!code.empty() && code.back() == '/') code.pop_back();
    if (auto p = code.rfind("-app0"); p != std::string::npos && p == code.size() - 5) code = code.substr(0, p);
    if (code.empty()) code = "GAME";

    // Run-start timestamp shared by every screenshot in this run, so a run groups when sorted.
    char ts[32];
    { time_t t = time(nullptr); struct tm tmv; localtime_r(&t, &tmv); strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", &tmv); }

    // Zero-pad the index to the width of the largest index (min 2), for correct lexical sort.
    int pad = 2; for (int m = count - 1, d = 1; ; m /= 10, d++) { if (m < 10) { if (d > pad) pad = d; break; } }

    // Sane render-frontier defaults (don't override if the caller set them for another title).
    setenv("PROSPER_GUEST_FS",   "1", 0);
    setenv("PROSPER_GUEST_ARGS", "-force-gfx-direct", 0);
    if (warmup_seconds_set) {
        char delay_ms[32];
        snprintf(delay_ms, sizeof delay_ms, "%lld",
                 (long long)(warmup_seconds * 1000.0 + 0.5));
        setenv("PROSPER_RENDER_DELAY_MS", delay_ms, 1);
    }
    if (warmup_submits_set) {
        char first_submit[32];
        snprintf(first_submit, sizeof first_submit, "%d", warmup_submits);
        setenv("PROSPER_RENDER_FIRST", first_submit, 1);
    }

    const int64_t render_delay_ms = getenv("PROSPER_RENDER_DELAY_MS")
        ? std::max<int64_t>(0, atoll(getenv("PROSPER_RENDER_DELAY_MS"))) : 0;
    const int render_first = getenv("PROSPER_RENDER_FIRST")
        ? std::max(0, atoi(getenv("PROSPER_RENDER_FIRST"))) : 0;
    const bool warming_up = render_delay_ms > 0 || render_first > 0;

    std::error_code out_ec;
    std::filesystem::create_directories(out, out_ec);
    if (out_ec) {
        fprintf(stderr, "screenshot: cannot create output directory '%s': %s\n",
                out.c_str(), out_ec.message().c_str());
        return 1;
    }

    // Register the shared live renderer (feeds the present layer; no BMP spam), then boot + run the
    // guest on its own thread while this thread samples the present layer.
    prosper::frontend::register_live_renderer(".", /*dump_bmps=*/false);
    Program prog; std::string err;
    if (!boot_program(dump, prog, &err)) { fprintf(stderr, "screenshot: boot failed: %s\n", err.c_str()); return 1; }
    std::thread guest([&prog] { run_entry(prog.imgs[0]); });
    guest.detach();

    const bool time_mode = seconds > 0;   // capture on wall-clock interval vs. every N rendered frames
    if (time_mode)
        fprintf(stderr, "[shot] %s: %d screenshots, one every %gs -> %s/%s_%s_*.png\n",
                code.c_str(), count, seconds, out.c_str(), code.c_str(), ts);
    else
        fprintf(stderr, "[shot] %s: %d screenshots, every %d frames -> %s/%s_%s_*.png\n",
                code.c_str(), count, every, out.c_str(), code.c_str(), ts);
    if (warming_up)
        fprintf(stderr, "[shot] warmup: %lld ms, %d submits; capture suppressed until warmup ends\n",
                (long long)render_delay_ms, render_first);

    std::vector<uint8_t> buf;
    int saved = 0;
    uint64_t next = (uint64_t)every;   // frame-mode: rendered-frame # for the next shot
    auto t0 = std::chrono::steady_clock::now();
    auto last_cap = t0;                // time-mode: wall-clock of the previous shot
    while (saved < count) {
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        if (timeout > 0 && el > timeout) {
            fprintf(stderr, "[shot] timeout after %.0fs with %d/%d saved — game not rendering enough "
                            "(wrong guest env for this title? see the README)\n", el, saved, count);
            break;
        }
        bool due = time_mode ? (std::chrono::duration<double>(now - last_cap).count() >= seconds)
                             : (gpu::present_frame_seq() >= next);
        const bool rendered = gpu::present_has_frame();
        const bool wall_warmup_done = el * 1000.0 >= (double)render_delay_ms;
        const bool rendered_capture = rendered && wall_warmup_done;
        const bool raw_scanout = wall_warmup_done && render_first == 0 &&
                                 gpu::present_front_index() >= 0 &&
                                 gpu::present_width() > 0 && gpu::present_height() > 0;
        if ((rendered_capture || raw_scanout) && due) {
            uint64_t at = gpu::present_frame_seq();
            // Size the buffer from the RENDERED frame's dims, not the guest display dims: under
            // PROSPER_RENDER_SCALE the frame is smaller, and present_readback returns g_frame.size()
            // (scaled). Using the display dims made buf.size() != readback bytes, so the exact-size
            // guard below dropped EVERY screenshot silently (#399). Fall back to display dims only when
            // no rendered frame is present (the raw-scanout path).
            uint32_t fw = rendered ? gpu::present_frame_width() : 0;
            uint32_t fh = rendered ? gpu::present_frame_height() : 0;
            uint32_t w = fw ? fw : gpu::present_width(), h = fh ? fh : gpu::present_height();
            if (w && h) {
                buf.resize((size_t)w * h * 4);
                if (gpu::present_readback(buf.data(), buf.size()) == buf.size()) {
                    char fn[1024];
                    snprintf(fn, sizeof fn, "%s/%s_%s_%0*d.png", out.c_str(), code.c_str(), ts, pad, saved);
                    std::string png_error;
                    if (write_png(fn, buf.data(), w, h, png_error)) {
                        fprintf(stderr, "[shot] %d/%d  %s  (frame %llu, %.1fs)\n",
                                saved + 1, count, fn, (unsigned long long)at, el);
                        saved++;
                    } else {
                        fprintf(stderr, "[shot] PNG write failed: %s: %s\n", fn, png_error.c_str());
                    }
                    if (time_mode) last_cap = now; else next = at + (uint64_t)every;
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    fprintf(stderr, "[shot] done: %d screenshot(s) in %s\n", saved, out.c_str());
    _exit(0);   // the guest thread is detached and running guest code; don't block on teardown
}
