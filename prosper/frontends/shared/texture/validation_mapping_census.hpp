#pragma once
// validation_mapping_census.hpp -- the [validation-census] exit summary (always on): which KIND of host
// memory backs the guest ranges the renderer compares byte-for-byte.
//
// WHY. On Windows the page-protection write watch is retired (the exception frame lands in the guest's
// red zone), so a cross-submit validation is a full `memcmp`. The only fault-free, kernel-tracked
// alternative is `GetWriteWatch`, and it accepts PRIVATE allocations only: it rejects every section
// view (pagefile-backed, file-backed, placeholder-replaced) with ERROR_INVALID_PARAMETER. Whether that
// is worth building depends on how many compared bytes live in private memory versus section views.
//
// TWO SOURCES, reported separately because they are different code paths:
//   renderer -- `safe_equal`: sampled-source and texture validations. The outcome is known, so
//               `changed` counts only a compare that found differing bytes (an expected-missing or
//               short-prefix result is a validation, not a change).
//   compute  -- the compute buffer cache's full compares (`WriteWatchCensus::record_exact_compare`).
//               Recorded where the compare runs, before its result is known, so no `changed` figure.
//
// SIZES. Each source also keeps a size histogram of its compares (below 64 KiB, 1 MiB, 16 MiB, 64 MiB,
// and larger), so a large byte total can be attributed to a few big ranges or to many small ones.
//
// SOURCE POINTERS. The compute hit paths classify the pointer they compare against. For a zero-padded or
// atomic-image binding that can be prosper's own seed vector (host heap, MEM_PRIVATE) rather than guest
// memory, which would read as GetWriteWatch-coverable when it is not. It did not occur on Black Flag
// (0% private); a title that shows private bytes here should be checked for that before it is believed.
//
// A range is classified by its FIRST byte. A range straddling a private/section boundary is attributed
// whole, which is fine for this question: guest allocations are page-granular and the answer is a share.
//
// Observes only: it never changes what a compare returns or which compares happen.
#include "diagnostics/exit_census.hpp"   // register_census: atexit never runs here (every frontend _exit()s)
#include "host/platform/mapping_class.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace prosper::frontend {

class ValidationMappingCensus {
public:
    static constexpr int kClasses = 5;      // prosper::host::MappingClass values, then kSmall
    static constexpr int kSmall = 4;        // below kMinClassifiedBytes: counted, not classified
    static constexpr int kSizeBuckets = 5;  // <64 KiB, <1 MiB, <16 MiB, <64 MiB, 64 MiB and larger

    struct Row {
        uint64_t validations = 0;
        uint64_t bytes = 0;
        uint64_t changed = 0;        // validations whose bytes differed (the compare found a write)
        uint64_t changed_bytes = 0;
    };

    // `changed` is whether the compare found differing bytes; pass false when the outcome is not
    // known or the compare did not run to a verdict.
    void record(int cls, uint64_t bytes, bool changed) {
        if (cls < 0 || cls >= kClasses) cls = 0;
        const int bucket = size_bucket(bytes);
        size_validations_[bucket].fetch_add(1, std::memory_order_relaxed);
        size_bytes_[bucket].fetch_add(bytes, std::memory_order_relaxed);
        validations_[cls].fetch_add(1, std::memory_order_relaxed);
        bytes_[cls].fetch_add(bytes, std::memory_order_relaxed);
        if (changed) {
            changed_[cls].fetch_add(1, std::memory_order_relaxed);
            changed_bytes_[cls].fetch_add(bytes, std::memory_order_relaxed);
        }
    }

    Row row(int cls) const {
        Row r;
        r.validations = validations_[cls].load(std::memory_order_relaxed);
        r.bytes = bytes_[cls].load(std::memory_order_relaxed);
        r.changed = changed_[cls].load(std::memory_order_relaxed);
        r.changed_bytes = changed_bytes_[cls].load(std::memory_order_relaxed);
        return r;
    }

    static int size_bucket(uint64_t bytes) {
        if (bytes < (64ull << 10)) return 0;
        if (bytes < (1ull << 20)) return 1;
        if (bytes < (16ull << 20)) return 2;
        if (bytes < (64ull << 20)) return 3;
        return 4;
    }
    uint64_t size_validations(int bucket) const { return size_validations_[bucket].load(std::memory_order_relaxed); }
    uint64_t size_bytes(int bucket) const { return size_bytes_[bucket].load(std::memory_order_relaxed); }

    uint64_t total_validations() const {
        uint64_t n = 0;
        for (int i = 0; i < kClasses; ++i) n += row(i).validations;
        return n;
    }

    // One line per class plus the figure the decision turns on: the share of compared bytes a
    // fault-free `GetWriteWatch` could cover (private memory only). `outcome_known` false prints
    // `changed=n/a`.
    std::string format(const char* source, bool outcome_known) const {
        static const char* const names[kClasses] = {"untracked", "private", "mapped-view", "other", "small-unclassified"};
        uint64_t total_bytes = 0;
        for (int i = 0; i < kClasses; ++i) total_bytes += row(i).bytes;
        std::string out;
        char line[360];
        for (int i = 0; i < kClasses; ++i) {
            const Row r = row(i);
            const double share = total_bytes ? 100.0 * static_cast<double>(r.bytes) /
                                                   static_cast<double>(total_bytes) : 0.0;
            if (outcome_known) {
                std::snprintf(line, sizeof(line),
                              "[validation-census] source=%s class=%s validations=%llu bytes=%llu "
                              "(%.1f%% of compared bytes) changed=%llu changed_bytes=%llu\n",
                              source, names[i], static_cast<unsigned long long>(r.validations),
                              static_cast<unsigned long long>(r.bytes), share,
                              static_cast<unsigned long long>(r.changed),
                              static_cast<unsigned long long>(r.changed_bytes));
            } else {
                std::snprintf(line, sizeof(line),
                              "[validation-census] source=%s class=%s validations=%llu bytes=%llu "
                              "(%.1f%% of compared bytes) changed=n/a\n",
                              source, names[i], static_cast<unsigned long long>(r.validations),
                              static_cast<unsigned long long>(r.bytes), share);
            }
            out += line;
        }
        const Row priv = row(static_cast<int>(prosper::host::MappingClass::Private));
        const double private_share = total_bytes ? 100.0 * static_cast<double>(priv.bytes) /
                                                       static_cast<double>(total_bytes) : 0.0;
        std::snprintf(line, sizeof(line),
                      "[validation-census] source=%s GetWriteWatch-coverable (private memory only) = "
                      "%.1f%% of %llu compared bytes\n",
                      source, private_share, static_cast<unsigned long long>(total_bytes));
        out += line;
        static const char* const size_names[kSizeBuckets] = {"<64KiB", "<1MiB", "<16MiB", "<64MiB", ">=64MiB"};
        std::string sizes = std::string("[validation-census] source=") + source + " sizes:";
        for (int b = 0; b < kSizeBuckets; ++b) {
            std::snprintf(line, sizeof(line), " %s n=%llu %.2fGB", size_names[b],
                          static_cast<unsigned long long>(size_validations(b)),
                          static_cast<double>(size_bytes(b)) / 1e9);
            sizes += line;
        }
        out += sizes + "\n";
        return out;
    }

private:
    std::atomic<uint64_t> size_validations_[kSizeBuckets] = {};
    std::atomic<uint64_t> size_bytes_[kSizeBuckets] = {};
    std::atomic<uint64_t> validations_[kClasses] = {};
    std::atomic<uint64_t> bytes_[kClasses] = {};
    std::atomic<uint64_t> changed_[kClasses] = {};
    std::atomic<uint64_t> changed_bytes_[kClasses] = {};
};

// Never destroyed: the end-of-run report reads them after static destruction has begun elsewhere.
inline ValidationMappingCensus& renderer_mapping_census() {
    static ValidationMappingCensus* const census = new ValidationMappingCensus();
    return *census;
}
inline ValidationMappingCensus& compute_mapping_census() {
    static ValidationMappingCensus* const census = new ValidationMappingCensus();
    return *census;
}

// Prints nothing when it counted nothing, so an unrelated run stays quiet.
inline bool report_validation_mapping_census() {
    bool printed = false;
    if (renderer_mapping_census().total_validations()) {
        std::fputs(renderer_mapping_census().format("renderer", true).c_str(), stderr);
        printed = true;
    }
    if (compute_mapping_census().total_validations()) {
        std::fputs(compute_mapping_census().format("compute", false).c_str(), stderr);
        printed = true;
    }
    return printed;
}

// Always on, like the perf observers whose summary lines it joins: a counter pair and, for ranges of at
// least kMinClassifiedBytes, one VirtualQuery -- compares are O(resources) and each reads at least that many
// bytes, so the cost sits beside a memcmp it is small against. No switch, so nothing to register or retire.
inline void ensure_validation_mapping_census_registered() {
    // Registered on first use, never during static initialisation (see exit_census.hpp).
    static const bool registered = (prosper::diagnostics::register_census(
        nullptr, [] { return report_validation_mapping_census(); }), true);
    (void)registered;
}

// Ranges below this are counted in the size histogram but not classified: a 16-byte constant buffer is
// not what the question is about, and a syscall per such compare would be the dominant cost.
inline constexpr uint64_t kMinClassifiedBytes = 4096;

// The renderer's `safe_equal`. `extent` is the bytes the compare actually ran over, `changed` whether
// it found differing bytes.
inline void note_validation_mapping(uint64_t address, uint64_t extent, bool changed) {
    ensure_validation_mapping_census_registered();
    const int cls = extent < kMinClassifiedBytes
                        ? ValidationMappingCensus::kSmall
                        : static_cast<int>(prosper::host::classify_host_mapping(address));
    renderer_mapping_census().record(cls, extent, changed);
}

// The compute buffer cache's compares, recorded before their verdict.
inline void note_compute_compare_mapping(const void* source, uint64_t bytes) {
    ensure_validation_mapping_census_registered();
    const int cls = bytes < kMinClassifiedBytes
                        ? ValidationMappingCensus::kSmall
                        : static_cast<int>(prosper::host::classify_host_mapping(reinterpret_cast<uintptr_t>(source)));
    compute_mapping_census().record(cls, bytes, false);
}

}  // namespace prosper::frontend
