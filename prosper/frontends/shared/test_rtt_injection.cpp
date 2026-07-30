#include "rtt_injection.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using prosper::frontend::inject_rtt_pixels;
using prosper::frontend::exact_rtt_snapshot_borrowable;
using prosper::frontend::fill_repeating_pixel;
using prosper::frontend::RttInjectionCache;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    const uint8_t rgba_pixel[] = {0x12, 0x34, 0x56, 0x78};
    std::vector<uint8_t> repeated_rgba(5 * sizeof(rgba_pixel));
    CHECK(fill_repeating_pixel(
        repeated_rgba, rgba_pixel, sizeof(rgba_pixel)));
    for (size_t i = 0; i < repeated_rgba.size(); ++i)
        CHECK(repeated_rgba[i] == rgba_pixel[i % sizeof(rgba_pixel)]);
    const uint8_t fp16_pixel[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint8_t> repeated_fp16(4096 * sizeof(fp16_pixel));
    CHECK(fill_repeating_pixel(
        repeated_fp16, fp16_pixel, sizeof(fp16_pixel)));
    CHECK(std::equal(repeated_fp16.end() - sizeof(fp16_pixel),
                     repeated_fp16.end(), fp16_pixel));
    std::vector<uint8_t> empty;
    CHECK(fill_repeating_pixel(empty, rgba_pixel, sizeof(rgba_pixel)));
    std::vector<uint8_t> malformed_repeat(7, 0xcc);
    CHECK(!fill_repeating_pixel(
        malformed_repeat, rgba_pixel, sizeof(rgba_pixel)));
    CHECK(std::all_of(malformed_repeat.begin(), malformed_repeat.end(),
                      [](uint8_t byte) { return byte == 0xcc; }));

    CHECK(exact_rtt_snapshot_borrowable(4, 2, 4, 2, 8, 4 * 2 * 8));
    CHECK(!exact_rtt_snapshot_borrowable(2, 1, 4, 2, 8, 4 * 2 * 8));
    CHECK(!exact_rtt_snapshot_borrowable(4, 2, 4, 2, 8, 4 * 2 * 8 - 1));
    CHECK(!exact_rtt_snapshot_borrowable(4, 2, 4, 2, 0, 0));
    CHECK(!exact_rtt_snapshot_borrowable(
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 16,
        std::numeric_limits<size_t>::max()));

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

    // The integer render-scale fast path expands pixels and rows without changing nearest-neighbor
    // selection. Distinct values in both axes catch accidental row/pixel replication mixups.
    const std::vector<uint8_t> rgba_source = {
        1, 2, 3, 4,  5, 6, 7, 8,
        9,10,11,12, 13,14,15,16,
    };
    std::vector<uint8_t> rgba_scaled;
    CHECK(inject_rtt_pixels(rgba_scaled, 4, 4, rgba_source, 2, 2, 4));
    const std::vector<uint8_t> rgba_expected = {
        1,2,3,4, 1,2,3,4, 5,6,7,8, 5,6,7,8,
        1,2,3,4, 1,2,3,4, 5,6,7,8, 5,6,7,8,
        9,10,11,12, 9,10,11,12, 13,14,15,16, 13,14,15,16,
        9,10,11,12, 9,10,11,12, 13,14,15,16, 13,14,15,16,
    };
    CHECK(rgba_scaled == rgba_expected);

    // Keep a non-integer scale covered as well; this takes the generic specialized-width loop.
    std::vector<uint8_t> odd_scaled;
    CHECK(inject_rtt_pixels(odd_scaled, 3, 3, rgba_source, 2, 2, 4));
    const std::vector<uint8_t> odd_expected = {
        1,2,3,4, 1,2,3,4, 5,6,7,8,
        1,2,3,4, 1,2,3,4, 5,6,7,8,
        9,10,11,12, 9,10,11,12, 13,14,15,16,
    };
    CHECK(odd_scaled == odd_expected);

    // A submit-scoped cache returns the immutable producer itself for exact views and one shared
    // nearest-scaled materialization for every repeated consumer of the same version and shape.
    auto shared_source = std::make_shared<const std::vector<uint8_t>>(rgba_source);
    std::weak_ptr<const std::vector<uint8_t>> retained_source = shared_source;
    RttInjectionCache cache;
    CHECK(cache.materialize(shared_source, 2, 2, 2, 2, 4) == shared_source);
    auto cached_scaled = cache.materialize(shared_source, 4, 4, 2, 2, 4);
    auto repeated_scaled = cache.materialize(shared_source, 4, 4, 2, 2, 4);
    CHECK(cached_scaled && cached_scaled == repeated_scaled &&
          cached_scaled->data() == repeated_scaled->data() &&
          *cached_scaled == rgba_expected && cache.size() == 1);
    auto different_shape = cache.materialize(shared_source, 3, 3, 2, 2, 4);
    CHECK(different_shape && *different_shape == odd_expected &&
          different_shape != cached_scaled && cache.size() == 2);
    auto equal_new_version = std::make_shared<const std::vector<uint8_t>>(rgba_source);
    auto new_version_scaled = cache.materialize(equal_new_version, 4, 4, 2, 2, 4);
    CHECK(new_version_scaled && *new_version_scaled == rgba_expected &&
          new_version_scaled != cached_scaled && cache.size() == 3);
    shared_source.reset();
    CHECK(!retained_source.expired());

    std::vector<uint8_t> malformed(7), fallback_bytes = {0xaa, 0xbb, 0xcc, 0xdd};
    consumer = fallback_bytes;
    CHECK(!inject_rtt_pixels(consumer, 1, 1, malformed, 1, 1, 8));
    CHECK(consumer == fallback_bytes); // rejection preserves the caller's guest-decode fallback buffer
    CHECK(!cache.materialize(
        std::make_shared<const std::vector<uint8_t>>(malformed), 1, 1, 1, 1, 8));
    CHECK(cache.size() == 3);

    if (!failures) std::printf("rtt_injection: OK\n");
    return failures ? 1 : 0;
}
