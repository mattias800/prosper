// bench_buffer_arena -- CPU cost of keeping nine overlapping 43 MiB constant-ring windows resident,
// with a private copy per window (before ADR 0010) and with one shared arena.
//
// Not a ctest: run it by hand. It uses the compute backend's own compare/copy functions
// (compute_buffer_bytes.hpp) over a synthetic guest ring that the "guest" rewrites sparsely each frame,
// and counts what each strategy has to compare and copy per frame. The windows are the nine bases
// measured on Black Flag Resynced, so the numbers describe that title's validation work, not a
// generic case.
//
//   bench_buffer_arena [frames]        default 60
#include "shared/compute/buffer_arena_registry.hpp"
#include "shared/compute/compute_buffer_bytes.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace prosper::frontend;
using Clock = std::chrono::steady_clock;

namespace {
constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kWindow = 45088768;
constexpr uint64_t kRing = 72 * kMiB;   // readable guest memory around the ring
constexpr uint64_t kGuestBase = 0x4062000000ull;
// Window offsets inside the ring, from the measured bases (0x4062600000 is ring offset 6 MiB).
constexpr uint64_t kWindowOffsets[] = {0x60b900, 0x60c300, 0x60c700, 0x60d100, 0x60d500,
                                       0x609e00, 0xb5f600, 0xb00000, 0xb73800};

struct Resident {
    std::vector<uint8_t> bytes;
};

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// The guest rewrites a few KiB scattered across the ring each frame.
void guest_frame(std::vector<uint8_t>& ring, unsigned frame) {
    for (unsigned k = 0; k < 24; ++k) {
        const size_t at = (static_cast<size_t>(frame) * 7919u + k * 1597331u) % (ring.size() - 4096u);
        for (unsigned b = 0; b < 2048; ++b) ring[at + b] = static_cast<uint8_t>(frame * 31u + k + b);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const unsigned frames = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 60u;
    std::vector<uint8_t> ring(kRing, 0x5a);

    // ---- private copy per window (the old behaviour) ------------------------------------------
    std::vector<Resident> windows(std::size(kWindowOffsets));
    for (auto& w : windows) w.bytes.assign(kWindow, 0);
    uint64_t private_compared = 0, private_copied = 0;
    double private_ms = 0;
    for (unsigned frame = 0; frame < frames; ++frame) {
        guest_frame(ring, frame);
        const auto t = Clock::now();
        for (size_t i = 0; i < windows.size(); ++i) {
            const uint8_t* source = ring.data() + kWindowOffsets[i];
            size_t first = 0, last = 0;
            const bool equal =
                compute_buffers_diff_span(windows[i].bytes.data(), source, kWindow, &first, &last);
            private_compared += kWindow;
            if (!equal) {
                copy_compute_buffer(windows[i].bytes.data() + first, source + first, last - first + 1);
                private_copied += last - first + 1;
            }
        }
        private_ms += ms_since(t);
    }

    // ---- one shared arena (ADR 0010) ----------------------------------------------------------
    BufferArenaRegistry registry;
    Resident arena;
    uint64_t arena_base = 0, arena_bytes = 0;
    uint64_t arena_compared = 0, arena_copied = 0;
    double arena_ms = 0, plan_ns = 0;
    uint64_t plans = 0;
    std::vector<uint8_t> ring2(kRing, 0x5a);
    for (unsigned frame = 0; frame < frames; ++frame) {
        guest_frame(ring2, frame);
        bool validated_this_frame = false;   // the journal says "unchanged" for later windows
        for (size_t i = 0; i < std::size(kWindowOffsets); ++i) {
            const uint64_t addr = kGuestBase + kWindowOffsets[i];
            const auto p = Clock::now();
            BufferArenaDecision d = registry.find(addr, kWindow, 256);
            if (d.action == BufferArenaAction::Private) {
                d = registry.plan(addr, kWindow, kGuestBase, kGuestBase + kRing, 256);
                registry.commit(d);
            }
            plan_ns += std::chrono::duration<double, std::nano>(Clock::now() - p).count();
            ++plans;
            if (d.action == BufferArenaAction::Private) continue;
            const auto t = Clock::now();
            if (d.arena.base != arena_base || d.arena.bytes != arena_bytes) {   // (re)built
                arena.bytes.assign(d.arena.bytes, 0);
                arena_base = d.arena.base;
                arena_bytes = d.arena.bytes;
                validated_this_frame = false;
            }
            if (!validated_this_frame) {
                const uint8_t* source = ring2.data() + (arena_base - kGuestBase);
                size_t first = 0, last = 0;
                const bool equal =
                    compute_buffers_diff_span(arena.bytes.data(), source, arena_bytes, &first, &last);
                arena_compared += arena_bytes;
                if (!equal) {
                    copy_compute_buffer(arena.bytes.data() + first, source + first, last - first + 1);
                    arena_copied += last - first + 1;
                }
                validated_this_frame = true;
            }
            arena_ms += ms_since(t);
        }
    }

    const auto mib = [](uint64_t b) { return static_cast<double>(b) / kMiB; };
    std::printf("frames=%u windows/frame=%zu window=%.1f MiB\n", frames, std::size(kWindowOffsets),
                mib(kWindow));
    std::printf("private: %8.1f ms/frame  compared %8.1f MiB/frame  copied %8.1f MiB/frame\n",
                private_ms / frames, mib(private_compared) / frames, mib(private_copied) / frames);
    std::printf("arena:   %8.1f ms/frame  compared %8.1f MiB/frame  copied %8.1f MiB/frame\n",
                arena_ms / frames, mib(arena_compared) / frames, mib(arena_copied) / frames);
    std::printf("arena planning: %.0f ns per window (%llu windows); resident arena %.1f MiB vs %.1f MiB "
                "of private copies\n",
                plan_ns / static_cast<double>(plans), static_cast<unsigned long long>(plans),
                mib(arena_bytes), mib(kWindow * std::size(kWindowOffsets)));
    return 0;
}
