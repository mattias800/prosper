// Equivalence and boundary checks for the 16-bit index expansion, plus an opt-in `--bench`
// arm that measures the two kernels against each other.
//
// The AVX2 kernel is NOT enabled by default (see `index_expand.hpp`). The correctness checks
// below therefore call it directly rather than through the dispatcher, so they exercise it
// whatever `PROSPER_INDEX_EXPAND_SIMD` says in the environment ctest happens to run under.

#include "gpu/execute/index_expand.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TEST_HAS_AVX2_KERNEL 1
#else
#define TEST_HAS_AVX2_KERNEL 0
#endif

namespace {

// Set by the checks that actually ran, and printed at the end. A green run must be able to say
// which kernels it compared; "passed" on a host where the AVX2 arm silently never ran says
// nothing about the AVX2 arm.
bool exercised_scalar = false;
bool exercised_avx2 = false;
bool exercised_edge_avx2 = false;

bool check_one(size_t count, size_t source_offset, size_t output_offset, size_t peak) {
    constexpr uint32_t guard = 0xfeedfaceu;
    std::vector<uint16_t> source(count + source_offset + 8, 0xabcdu);
    std::vector<uint32_t> output(count + output_offset + 8, guard);
    uint16_t* src = source.data() + source_offset;
    uint32_t* dst = output.data() + output_offset;
    for (size_t i = 0; i < count; ++i)
        src[i] = static_cast<uint16_t>((i * 31337u + 7919u) & 0xffffu);
    if (count) src[peak % count] = 0xffffu;

    const uint32_t expected_max = count ? *std::max_element(src, src + count) : 0u;
    const auto check_output = [&](uint32_t maximum) {
        if (maximum != expected_max) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            const uint32_t expected = i >= output_offset && i < output_offset + count
                ? src[i - output_offset] : guard;
            if (output[i] != expected) return false;
        }
        return true;
    };
    if (!check_output(prosper::gpu::copy_indices_u16_max_scalar(dst, src, count))) return false;
    exercised_scalar = true;
    std::fill(output.begin(), output.end(), guard);
#if TEST_HAS_AVX2_KERNEL
    if (prosper::gpu::index_expand_avx2_available()) {
        if (!check_output(prosper::gpu::copy_indices_u16_max_avx2(dst, src, count))) return false;
        exercised_avx2 = true;
        std::fill(output.begin(), output.end(), guard);
    }
#endif
    return check_output(prosper::gpu::copy_indices_u16_max(dst, src, count));
}

#if defined(__linux__)
// The guest index range is validated by `guest_readable(index_addr, n * esz)` and nothing more,
// so a legitimate range can end exactly at a mapping edge. Read one byte past it and a valid
// draw faults. Both kernels are driven here, and the AVX2 one directly: routing this through
// the dispatcher would quietly skip it whenever the SIMD path is off, which is the default.
bool check_mapping_edge() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return false;
    void* pages = mmap(nullptr, static_cast<size_t>(page_size) * 2,
                       PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) return false;
    const bool protected_page = mprotect(static_cast<uint8_t*>(pages) + page_size,
                                         static_cast<size_t>(page_size), PROT_NONE) == 0;
    bool valid = protected_page;
    if (valid) {
        for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 33u}) {
            auto* src = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(pages) + page_size)
                        - count;
            for (size_t i = 0; i < count; ++i) src[i] = static_cast<uint16_t>(i * 67u);
            const auto check = [&](uint32_t maximum, const std::vector<uint32_t>& dst) {
                bool good = maximum == (count - 1) * 67u && dst[count] == 0xfeedfaceu;
                for (size_t i = 0; i < count; ++i) good &= dst[i] == i * 67u;
                return good;
            };
            std::vector<uint32_t> dst(count + 1, 0xfeedfaceu);
            valid &= check(prosper::gpu::copy_indices_u16_max_scalar(dst.data(), src, count), dst);
            std::fill(dst.begin(), dst.end(), 0xfeedfaceu);
            valid &= check(prosper::gpu::copy_indices_u16_max(dst.data(), src, count), dst);
#if TEST_HAS_AVX2_KERNEL
            if (prosper::gpu::index_expand_avx2_available()) {
                std::fill(dst.begin(), dst.end(), 0xfeedfaceu);
                valid &= check(prosper::gpu::copy_indices_u16_max_avx2(dst.data(), src, count), dst);
                exercised_edge_avx2 = true;
            }
#endif
        }
    }
    munmap(pages, static_cast<size_t>(page_size) * 2);
    return valid;
}
#endif

// The default is OFF, and that is the whole point of this table: only the exact string "1"
// opts in. A change that made an unset variable, an empty string, or "0" enable the kernel
// would silently re-ship an unjustified default, which is what this case exists to catch.
bool check_enable_policy() {
    struct Case { const char* setting; bool requested; };
    static constexpr std::array<Case, 10> cases = {{
        {nullptr, false}, {"", false}, {"0", false}, {"1", true}, {"10", false},
        {"01", false}, {"1 ", false}, {" 1", false}, {"true", false}, {"yes", false},
    }};
    bool valid = true;
    for (const Case& one : cases) {
        const bool actual = prosper::gpu::index_expand_simd_requested(one.setting);
        if (actual != one.requested) {
            std::fprintf(stderr, "index expansion opt-in wrong for \"%s\": %d expected %d\n",
                         one.setting ? one.setting : "(unset)", actual, one.requested);
            valid = false;
        }
    }
    // The live accessor must agree with the policy applied to this process's own environment.
    const bool expected = prosper::gpu::index_expand_simd_requested(
                              std::getenv("PROSPER_INDEX_EXPAND_SIMD")) &&
                          prosper::gpu::index_expand_avx2_available();
    if (prosper::gpu::index_expand_simd_enabled() != expected) {
        std::fputs("index expansion enablement disagrees with its own opt-in policy\n", stderr);
        valid = false;
    }
    return valid;
}

// Opt-in measurement arm.
//
// Three properties make the number usable on a machine several agents share:
//  * Arms are INTERLEAVED per sample (A,B,A,B) rather than all-of-A then all-of-B, so machine
//    drift cannot masquerade as an effect.
//  * Each sample times a BATCH of calls, because one call at the smallest size is comparable to
//    the cost of reading the clock.
//  * The MINIMUM over samples is reported beside the mean. On a contended box the mean measures
//    the box; the minimum is the closest available estimate of the kernel.
// Both arms run from the same resident buffers, so the data is cache-warm. That bias favours
// the SIMD arm -- a hot loop is compute-bound, where wider lanes help most, while the real call
// reads guest memory the renderer has not just touched. Read the result as an upper bound on
// what the kernel could save, never as a frame-level effect.
void run_benchmark() {
    std::printf("index expansion benchmark (interleaved arms, batched samples; "
                "min approximates the kernel, mean includes the box)\n");
#if TEST_HAS_AVX2_KERNEL
    if (!prosper::gpu::index_expand_avx2_available())
        std::printf("  AVX2 unavailable on this host -- scalar arm only\n");
    const bool have_simd = prosper::gpu::index_expand_avx2_available();
#else
    std::printf("  no AVX2 kernel compiled for this target -- scalar arm only\n");
    const bool have_simd = false;
#endif
    std::mt19937 generator(20260926u);
    uint64_t sink = 0;
    for (size_t count : {64u, 384u, 3072u, 24576u, 196608u}) {
        std::vector<uint16_t> source(count);
        for (uint16_t& index : source) index = static_cast<uint16_t>(generator() & 0xffffu);
        std::vector<uint32_t> destination(count);
        constexpr size_t samples = 40;
        const size_t batch = std::max<size_t>(8, 2000000u / (count + 1));

        double scalar_min = 1e30, scalar_total = 0, simd_min = 1e30, simd_total = 0;
        for (size_t sample = 0; sample < samples; ++sample) {
            auto start = std::chrono::steady_clock::now();
            for (size_t call = 0; call < batch; ++call) {
                sink += prosper::gpu::copy_indices_u16_max_scalar(destination.data(), source.data(), count);
                sink += destination[count / 2];
            }
            const double scalar_ns = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - start).count() / double(batch);
            scalar_min = std::min(scalar_min, scalar_ns);
            scalar_total += scalar_ns;
#if TEST_HAS_AVX2_KERNEL
            if (have_simd) {
                start = std::chrono::steady_clock::now();
                for (size_t call = 0; call < batch; ++call) {
                    sink += prosper::gpu::copy_indices_u16_max_avx2(destination.data(), source.data(), count);
                    sink += destination[count / 2];
                }
                const double simd_ns = std::chrono::duration<double, std::nano>(
                    std::chrono::steady_clock::now() - start).count() / double(batch);
                simd_min = std::min(simd_min, simd_ns);
                simd_total += simd_ns;
            }
#endif
        }
        std::printf("  n=%-7zu batch %-6zu scalar min %9.1f ns (%.4f ns/index) mean %9.1f",
                    count, batch, scalar_min, scalar_min / double(count), scalar_total / samples);
        if (have_simd)
            std::printf(" | avx2 min %9.1f ns (%.4f ns/index) mean %9.1f | min-vs-min saved %+.1f ns/call (%+.1f%%)",
                        simd_min, simd_min / double(count), simd_total / samples,
                        scalar_min - simd_min, 100.0 * (scalar_min - simd_min) / scalar_min);
        std::printf("\n");
    }
    std::printf("  [sink %llu]\n", static_cast<unsigned long long>(sink & 0xffu));
}

} // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc > 1 && std::string(argv[1]) == "--bench";

    if (!check_enable_policy()) return 1;

    constexpr std::array<size_t, 19> sizes = {
        0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 8193};
    for (size_t count : sizes)
        for (size_t source_offset : {0u, 1u, 3u})
            for (size_t output_offset : {0u, 1u, 3u})
                for (size_t peak : {0u, 7u, 8u, 15u, 257u})
                    if (!check_one(count, source_offset, output_offset, peak)) {
                        std::fprintf(stderr, "index expansion failed count=%zu source=%zu output=%zu peak=%zu\n",
                                     count, source_offset, output_offset, peak);
                        return 1;
                    }
#if defined(__linux__)
    if (!check_mapping_edge()) {
        std::fputs("index expansion crossed a validated guest mapping edge\n", stderr);
        return 1;
    }
#endif
    if (!exercised_scalar) {
        std::fputs("index expansion checked no kernel at all\n", stderr);
        return 1;
    }
#if TEST_HAS_AVX2_KERNEL
    // Not a failure: an AVX2-less host is a legitimate target. Say so, so a green run is never
    // read as having compared the two kernels when it could not.
    if (prosper::gpu::index_expand_avx2_available() && !(exercised_avx2 && exercised_edge_avx2)) {
        std::fputs("AVX2 is available but its kernel was not compared\n", stderr);
        return 1;
    }
#endif
    std::printf("index expansion and maximum controls passed (scalar%s%s, SIMD opt-in %s)\n",
                exercised_avx2 ? " + avx2" : " only",
                exercised_edge_avx2 ? " + avx2 mapping edge" : "",
                prosper::gpu::index_expand_simd_enabled() ? "ON" : "off");

    if (benchmark) run_benchmark();
    return 0;
}
