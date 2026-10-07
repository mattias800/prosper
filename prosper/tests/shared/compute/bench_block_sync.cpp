// bench_block_sync -- CPU cost of refreshing one resident compute buffer after the guest rewrites a few
// scattered pieces of it, comparing the two strategies the backend has:
//   span:   compare, then copy the single [first,last] extent (the path before block sync);
//   blocks: one pass that copies only the 64 KiB blocks that differ (sync_compute_buffer_blocks).
// Not a ctest: run it by hand.  bench_block_sync [frames] [MiB]   (defaults 60 and 59).
#include "shared/compute/compute_buffer_bytes.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace prosper::frontend;
using Clock = std::chrono::steady_clock;

namespace {
constexpr uint64_t kMiB = 1ull << 20;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// The guest rewrites 24 scattered 2 KiB pieces per frame (the shape measured on a constant ring).
void guest_frame(std::vector<uint8_t>& ring, unsigned frame) {
    for (unsigned k = 0; k < 24; ++k) {
        const size_t at = (static_cast<size_t>(frame) * 7919u + k * 1597331u) % (ring.size() - 4096u);
        for (unsigned b = 0; b < 2048; ++b) ring[at + b] = static_cast<uint8_t>(frame * 31u + k + b);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const unsigned frames = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 60u;
    const uint64_t bytes = (argc > 2 ? static_cast<uint64_t>(std::atoi(argv[2])) : 59u) * kMiB;
    std::vector<uint8_t> source(bytes, 0x5a), span_copy(bytes, 0x5a), block_copy(bytes, 0x5a);
    double span_ms = 0, block_ms = 0;
    uint64_t span_bytes = 0, block_bytes = 0;
    for (unsigned frame = 0; frame < frames; ++frame) {
        guest_frame(source, frame);
        auto t0 = Clock::now();
        size_t first = 0, last = 0;
        if (!compute_buffers_diff_span(span_copy.data(), source.data(), bytes, &first, &last)) {
            copy_compute_buffer(span_copy.data() + first, source.data() + first, last - first + 1);
            span_bytes += last - first + 1;
        }
        span_ms += ms_since(t0);
        t0 = Clock::now();
        block_bytes += sync_compute_buffer_blocks(block_copy.data(), source.data(), bytes);
        block_ms += ms_since(t0);
    }
    std::printf("buffer %.0f MiB, %u frames, 24 scattered 2 KiB rewrites per frame\n",
                static_cast<double>(bytes) / kMiB, frames);
    std::printf("span:   %7.2f ms/frame  copied %7.2f MiB/frame\n", span_ms / frames,
                static_cast<double>(span_bytes) / kMiB / frames);
    std::printf("blocks: %7.2f ms/frame  copied %7.2f MiB/frame\n", block_ms / frames,
                static_cast<double>(block_bytes) / kMiB / frames);
    return span_copy == block_copy ? 0 : 1;
}
