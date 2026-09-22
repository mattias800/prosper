#pragma once

// Lifted out of live_compute.cpp's anonymous namespaces so the code that operates on them
// can live in its own translation units. Declared in prosper/frontends/shared/live; who may include it is
// a decision this banner does not make -- say so with --note if it is restricted.

#include "shared/live/live_compute.hpp"
#include "shared/compute/storage_write_mask_spirv.hpp"
#include "shared/diagnostics/trip_bound_witness.hpp"
#include "shared/compute/compute_authority_live_census.hpp"
#include "shared/compute/compute_image_borrow_census.hpp"
#include "shared/compute/compute_timing_selector.hpp"
#include "shared/compute/compute_buffer_timing.hpp"
#include "shared/compute/compute_transfer_gate_census.hpp"
#include "shared/compute/storage_image_alias_plan.hpp"
#include "shared/live/decode_scratch.hpp"  // pooled full-surface intermediates (#3309's mechanism)
#include "shared/live/live_target_format.hpp"
#include "shared/live/packed_rtt_conversion.hpp"
#include "shared/live/gpu_retile.hpp"
#include "shared/rtt/rtt_scale.hpp"
#include "shared/rtt/rtt_authority.hpp"
#include "shared/device/pipeline_cache_file.hpp"  // #3425: one checked envelope for both stages
#include "shared/device/vulkan_device_select.hpp"
#include "shared/device/image_robustness.hpp"  // #3531: the recompiler's OOB image-read contract
#include "shared/texture/write_watch_census.hpp"
#include "shared/texture/write_watch_policy.hpp"
#include "diagnostics/env_numeric.hpp"   // #3253: a typo must not select a different setting
#include "shared/perf/performance_capture.hpp"      // bounded F8 post-trigger compute timing
#include "shared/perf/performance_timing_policy.hpp" // F8 measures without enabling verbose timing logs

#include "gpu/texture/bc_decode.hpp"
#include "gpu/diagnostics/vk_object_names.hpp"   // #3578
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/diagnostics/gpu_memory_budget_vk.hpp"  // #3533: how much of the heap does prosper hold?
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/host_read_barrier.hpp"  // #3249: a host read of a dispatch result needs an availability op
#include "gpu/execute/float_controls_probe.hpp"  // #3479: the device gate on SignedZeroInfNanPreserve
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/resources/atomic_image_staging.hpp"  // #3195: the LOGICAL/PHYSICAL atomic-image extent split
#include "gpu/resources/image_identity.hpp"
#include "gpu/resources/compressed_source_authority.hpp"
#include "gpu/resources/spirv_storage_match.hpp"  // #3204: SPIR-V/guest storage agreement  // #3204: named image-identity predicates
#include "gpu/texture/tile.hpp"
#include "gpu/capture/writer_provenance.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "host/platform/gpu_submit_gate.hpp"  // #3225: refuse submits once the frontend shuts down

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_HAVE_TARGET_F16C 1
#endif

namespace prosper::frontend {


// Large storage-image format conversions are independent per texel. Astro Bot's 4K FP16 target is
// 8.3 million texels, so leaving its lookup/pack walk on one core made a ~5 ms GPU dispatch wait on
// hundreds of milliseconds of host work. Keep small surfaces scalar and cap large conversions at
// sixteen workers because both the detiler and these loops eventually become memory-bandwidth-bound.
// Astro Bot's 4K path is still measurably faster at 16 on a 16-core Strix Halo; 32 regresses.
template <class Body>
void parallel_compute_texels(size_t count, size_t work_bytes, Body&& body,
                             unsigned max_threads = 16u) {
    if (!count) return;
    static const unsigned configured = [] {
        const char* value = std::getenv("PROSPER_COMPUTE_CONVERSION_THREADS");
        if (!value || !*value) return 0u;
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        return static_cast<unsigned>(std::min(parsed, 32ul));
    }();
    const unsigned hardware = std::thread::hardware_concurrency();
    const unsigned wanted = std::min(
        configured ? configured : std::min(hardware ? hardware : 4u, 16u), max_threads);
    const unsigned by_work = static_cast<unsigned>(std::min<size_t>(
        count, std::max<size_t>(1, work_bytes / (512u * 1024u))));
    const unsigned threads = std::max(1u, std::min(wanted ? wanted : 1u, by_work));
    if (threads <= 1) {
        body(size_t{0}, count);
        return;
    }
    const size_t chunk = (count + threads - 1) / threads;
    // jthread makes a partially-created set exception-safe: if the next OS thread cannot be
    // created, already-started workers are still joined. Finish the unspawned ranges on this
    // thread so transient resource pressure degrades to less parallelism instead of aborting.
    std::vector<std::jthread> workers;
    workers.reserve(threads - 1);
    unsigned next_worker = 1;
    try {
        for (; next_worker < threads; ++next_worker) {
            const size_t begin = static_cast<size_t>(next_worker) * chunk;
            const size_t end = std::min(count, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back([&body, begin, end] { body(begin, end); });
        }
    } catch (const std::system_error&) {
        // Fall through: ranges that did not get a worker run synchronously below.
    }
    body(size_t{0}, std::min(count, chunk));
    for (; next_worker < threads; ++next_worker) {
        const size_t begin = static_cast<size_t>(next_worker) * chunk;
        const size_t end = std::min(count, begin + chunk);
        if (begin >= end) break;
        body(begin, end);
    }
}

inline bool compute_buffers_equal(const void* lhs, const void* rhs, size_t bytes) {
    // glibc's vectorized memcmp is excellent for ordinary bindings. A 32 MiB persistent SSBO still
    // costs several milliseconds on one core, though, and the common maintenance-kernel result is
    // unchanged. Eight independent ranges measured best on the target APU (more workers became
    // memory-bandwidth/thread-start limited). This remains an exact comparison of every byte.
    constexpr size_t kParallelThreshold = 8u << 20;
    if (bytes < kParallelThreshold) return std::memcmp(lhs, rhs, bytes) == 0;
    const auto* a = static_cast<const uint8_t*>(lhs);
    const auto* b = static_cast<const uint8_t*>(rhs);
    std::atomic<bool> equal{true};
    parallel_compute_texels(bytes, bytes,
        [&](size_t begin, size_t end) {
            if (std::memcmp(a + begin, b + begin, end - begin) != 0)
                equal.store(false, std::memory_order_relaxed);
        }, 8u);
    return equal.load(std::memory_order_relaxed);
}

// Exact comparison that also reports WHERE the buffers differ. Returns true when equal; writes the
// inclusive first/last differing byte offsets otherwise.
//
// **The extent is load-bearing, not diagnostic.** The upload copies ONLY [first,last], so every byte
// outside it is written on the strength of this function having proved it already equal. A span one
// byte too NARROW leaves a stale byte in a GPU buffer, which no shader-output assertion necessarily
// catches and which surfaces as a wrong pixel much later. An earlier revision of this comment said
// "for diagnostics only" -- it was written when the span merely reported, and it survived into the
// revision that made the copy depend on it, where it would have licensed a future reader to make the
// narrowing approximate. Keep it exact; `tests/shared/compute/test_compute_buffer_diff_span.cpp` pins
// the property that the narrowed copy reproduces a full one.
//
// It is also the only thing that can answer where a difference lies: `compute_buffers_equal` above
// scans every byte by construction (eight workers, each memcmp-ing its whole slice), so a
// full-length compare time says nothing about whether one byte changed or all of them.
inline bool compute_buffers_diff_span(const void* lhs, const void* rhs, size_t bytes,
                               size_t* first, size_t* last) {
    const auto* a = static_cast<const uint8_t*>(lhs);
    const auto* b = static_cast<const uint8_t*>(rhs);
    std::atomic<size_t> lowest{SIZE_MAX};
    std::atomic<size_t> highest{0};
    const auto scan = [&](size_t begin, size_t end) {
        if (begin >= end) return;
        // Block-wise from both ends, never byte-at-a-time (a 41 MiB span scanned per byte would cost
        // more than the copy this avoids) and never a whole-slice memcmp first. An earlier version
        // did exactly that -- memcmp the slice to decide equality, then narrow -- which scanned every
        // differing slice TWICE and made the comparison 20% slower than the plain equality check it
        // replaced, eating a third of the win. The forward scan below already proves equality by
        // reaching `end`, so the extra pass bought nothing. 64 KiB blocks so the per-call overhead
        // stays negligible against memcmp's throughput.
        constexpr size_t kBlock = 64u << 10;
        size_t f = begin;
        while (f < end) {
            const size_t step = std::min(kBlock, end - f);
            if (std::memcmp(a + f, b + f, step) != 0) break;
            f += step;
        }
        if (f >= end) return;                 // this slice is equal
        while (f < end && a[f] == b[f]) ++f;
        size_t l = end;                       // exclusive while narrowing
        while (l > f + 1) {
            const size_t step = std::min(kBlock, l - f - 1);
            if (std::memcmp(a + l - step, b + l - step, step) != 0) break;
            l -= step;
        }
        --l;                                  // inclusive
        while (l > f && a[l] == b[l]) --l;
        size_t seen = lowest.load(std::memory_order_relaxed);
        while (f < seen && !lowest.compare_exchange_weak(seen, f, std::memory_order_relaxed)) {}
        seen = highest.load(std::memory_order_relaxed);
        while (l > seen && !highest.compare_exchange_weak(seen, l, std::memory_order_relaxed)) {}
    };
    constexpr size_t kParallelThreshold = 8u << 20;
    if (bytes < kParallelThreshold) scan(0, bytes);
    else parallel_compute_texels(bytes, bytes, scan, 8u);
    const size_t f = lowest.load(std::memory_order_relaxed);
    if (f == SIZE_MAX) return true;
    if (first) *first = f;
    if (last) *last = highest.load(std::memory_order_relaxed);
    return false;
}

inline void copy_compute_buffer(void* destination, const void* source, size_t bytes) {
    constexpr size_t kParallelThreshold = 8u << 20;
    if (bytes < kParallelThreshold) {
        std::memcpy(destination, source, bytes);
        return;
    }
    auto* dst = static_cast<uint8_t*>(destination);
    const auto* src = static_cast<const uint8_t*>(source);
    parallel_compute_texels(bytes, bytes,
        [&](size_t begin, size_t end) {
            std::memcpy(dst + begin, src + begin, end - begin);
        }, 8u);
}

}  // namespace prosper::frontend
