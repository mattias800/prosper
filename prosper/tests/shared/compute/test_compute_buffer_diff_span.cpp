// The proof obligation behind the differential upload (#3696).
//
// `setup > buffers` was 63.3% of all compute time on Grand Theft Auto V, almost all of it one
// program re-uploading a 256 MiB read-only binding 108 times in 75 s. The pooled host-visible
// allocation it uploads into retains its previous tenant's bytes, and measurement showed the
// difference between that and current guest memory starts at byte 0 on every one of those 108
// bindings but ENDS at a median 41.3 MiB -- so 215 MiB of each copy was already correct.
//
// The upload therefore copies only the inclusive [first,last] extent the comparison reports. That is
// safe if and only if the extent is EXACT. A span that is too wide merely wastes work; a span that is
// too NARROW leaves a stale byte in a GPU buffer, which no test of the shader's output would
// necessarily catch and which would surface as a wrong pixel much later.
//
// So the property asserted here is not "the span looks right" but the one that actually matters:
// **copying only the reported span reproduces a full copy, byte for byte.** Every case below checks
// that directly against a reference full copy, in addition to the span's own bounds.

#include "shared/compute/compute_buffer_bytes.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const std::string& message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message.c_str());
    failures += !ok;
}

// The whole contract in one place: compare, copy only what it reported, and require the result to
// equal a full copy. `expect_first`/`expect_last` pin the extent itself where the case knows it.
void check_case(const std::string& name, std::vector<uint8_t> dst, const std::vector<uint8_t>& src,
                bool expect_equal, size_t expect_first = 0, size_t expect_last = 0) {
    const std::vector<uint8_t> before = dst;
    size_t first = SIZE_MAX, last = SIZE_MAX;
    const bool equal =
        prosper::frontend::compute_buffers_diff_span(dst.data(), src.data(), dst.size(),
                                                             &first, &last);
    check(equal == expect_equal, name + ": equality reported correctly");
    if (equal) {
        // Nothing is copied on the equal path, so the destination must already match.
        check(dst == src, name + ": reported equal and really is equal");
        return;
    }
    check(first == expect_first, name + ": first differing byte is " + std::to_string(expect_first) +
                                 " (got " + std::to_string(first) + ")");
    check(last == expect_last, name + ": last differing byte is " + std::to_string(expect_last) +
                               " (got " + std::to_string(last) + ")");
    check(first <= last && last < dst.size(), name + ": span is well formed and in range");
    // Bytes outside the span must already be equal -- this is the claim the copy relies on, checked
    // independently of the span's own arithmetic.
    bool outside_equal = true;
    for (size_t i = 0; i < dst.size(); ++i) {
        if (i >= first && i <= last) continue;
        if (before[i] != src[i]) { outside_equal = false; break; }
    }
    check(outside_equal, name + ": every byte OUTSIDE the span was already equal");
    // And the property that matters: the narrowed copy equals a full copy.
    std::memcpy(dst.data() + first, src.data() + first, last - first + 1);
    check(dst == src, name + ": copying ONLY the span reproduces a full copy");
}

std::vector<uint8_t> pattern(size_t bytes, uint32_t seed) {
    std::vector<uint8_t> out(bytes);
    std::mt19937 rng(seed);
    for (auto& byte : out) byte = static_cast<uint8_t>(rng());
    return out;
}

}  // namespace

int main() {
    // Small buffers take the single-threaded path; anything at or above 8 MiB takes the eight-worker
    // parallel path, where the per-worker first/last must combine correctly across slice boundaries.
    // Both are exercised, because a combine bug is invisible below the threshold.
    constexpr size_t kSmall = 4096;
    constexpr size_t kParallel = 12u << 20;   // > the 8 MiB threshold, not a multiple of it

    for (const size_t bytes : {kSmall, kParallel}) {
        const std::string tag = bytes == kSmall ? "small" : "parallel";
        const std::vector<uint8_t> src = pattern(bytes, 1u);

        check_case(tag + "/equal", src, src, true);

        auto one_at = [&](size_t index) {
            std::vector<uint8_t> dst = src;
            dst[index] ^= 0xffu;
            return dst;
        };
        check_case(tag + "/first-byte", one_at(0), src, false, 0, 0);
        check_case(tag + "/last-byte", one_at(bytes - 1), src, false, bytes - 1, bytes - 1);
        check_case(tag + "/middle-byte", one_at(bytes / 2), src, false, bytes / 2, bytes / 2);

        // Both ends differ: the span must cover everything between them even though the interior is
        // identical. A per-worker minimum/maximum that failed to combine would report one end only.
        {
            std::vector<uint8_t> dst = src;
            dst[0] ^= 0xffu;
            dst[bytes - 1] ^= 0xffu;
            check_case(tag + "/both-ends", dst, src, false, 0, bytes - 1);
        }

        // The measured GTA V shape: differs from byte 0 to a point well before the end, with a long
        // identical tail. This is the case the optimization exists for.
        {
            const size_t head = bytes / 6;
            std::vector<uint8_t> dst = src;
            for (size_t i = 0; i < head; ++i) dst[i] ^= 0x5au;
            check_case(tag + "/head-differs-long-identical-tail", dst, src, false, 0, head - 1);
        }

        // Entirely different -- the span degenerates to the whole buffer and the copy is a full copy.
        {
            std::vector<uint8_t> dst = pattern(bytes, 2u);
            // A random buffer can coincide with the source at its first or last byte; force both.
            dst[0] = static_cast<uint8_t>(src[0] ^ 0xffu);
            dst[bytes - 1] = static_cast<uint8_t>(src[bytes - 1] ^ 0xffu);
            check_case(tag + "/all-different", dst, src, false, 0, bytes - 1);
        }
    }

    // A difference landing exactly on the parallel path's slice boundaries. The workers split the
    // range into eight, so a byte at a slice edge is where an off-by-one in the combine shows up.
    {
        const std::vector<uint8_t> src = pattern(kParallel, 7u);
        for (int slice = 1; slice < 8; ++slice) {
            const size_t edge = (kParallel / 8) * static_cast<size_t>(slice);
            if (edge >= kParallel) continue;
            std::vector<uint8_t> dst = src;
            dst[edge] ^= 0xffu;
            check_case("slice-edge/" + std::to_string(slice), dst, src, false, edge, edge);
        }
    }

    // Zero bytes: equal, and nothing is read.
    {
        std::vector<uint8_t> empty;
        size_t first = SIZE_MAX, last = SIZE_MAX;
        check(prosper::frontend::compute_buffers_diff_span(nullptr, nullptr, 0,
                                                                  &first, &last),
              "zero bytes compares equal");
    }

    if (failures) {
        std::printf("compute_buffer_diff_span: %d FAILED\n", failures);
        return 1;
    }
    std::printf("compute_buffer_diff_span: OK\n");
    return 0;
}
