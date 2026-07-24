#include "rtt_injection.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using prosper::frontend::inject_rtt_pixels;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    // #1293: a 1D consumer was provisionally allocated at RGBA8 (4 B/px), then an exact-size FP16
    // RTT injection memcpy'd 8 B/px into it. The helper must resize from the producer's real stride.
    std::vector<uint8_t> fp16_source(4 * 2 * 8);
    for (size_t i = 0; i < fp16_source.size(); ++i) fp16_source[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> consumer(4 * 2 * 4, 0xcc);
    CHECK(inject_rtt_pixels(consumer, 4, 2, fp16_source, 4, 2, 8));
    CHECK(consumer.size() == fp16_source.size());
    CHECK(consumer == fp16_source);

    // The mismatched-extent path must use the same 8-byte destination stride while nearest-scaling.
    std::vector<uint8_t> scaled;
    CHECK(inject_rtt_pixels(scaled, 2, 1, fp16_source, 4, 2, 8));
    CHECK(scaled.size() == 2 * 1 * 8);
    CHECK(std::equal(scaled.begin(), scaled.begin() + 8, fp16_source.begin()));
    CHECK(std::equal(scaled.begin() + 8, scaled.end(), fp16_source.begin() + 16));

    std::vector<uint8_t> malformed(7), fallback_bytes = {0xaa, 0xbb, 0xcc, 0xdd};
    consumer = fallback_bytes;
    CHECK(!inject_rtt_pixels(consumer, 1, 1, malformed, 1, 1, 8));
    CHECK(consumer == fallback_bytes); // rejection preserves the caller's guest-decode fallback buffer

    if (!failures) std::printf("rtt_injection: OK\n");
    return failures ? 1 : 0;
}
