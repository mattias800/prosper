#pragma once
// validation_mapping_census.hpp -- PROSPER_VALIDATION_MAPPING_CENSUS (diagnostic): which KIND of host
// memory backs the guest ranges the renderer validates with a full byte compare.
//
// WHY. On Windows the page-protection write watch is retired (the exception frame lands in the
// guest's red zone), so every cross-submit validation is a full `memcmp`. The only fault-free,
// kernel-tracked alternative is `GetWriteWatch`, and it accepts PRIVATE allocations only: it rejects
// every section view (pagefile-backed, file-backed, placeholder-replaced) with ERROR_INVALID_PARAMETER.
// Whether that is worth building depends on how many validated bytes live in private memory versus
// direct-memory section views, which nothing measured until this existed.
//
// Observes only: it never changes what a compare returns or which compares happen.
#include "diagnostics/env_cache.hpp"
#include "diagnostics/exit_census.hpp"   // register_census: atexit never runs here (every frontend _exit()s)
#include "host/platform/mapping_class.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace prosper::frontend::submit_renderer {

class ValidationMappingCensus {
public:
    static constexpr int kClasses = 4;   // prosper::host::MappingClass values

    struct Row {
        uint64_t validations = 0;
        uint64_t bytes = 0;
        uint64_t changed = 0;        // validations whose bytes differed (the memcmp found a write)
        uint64_t changed_bytes = 0;
    };

    void record(int cls, uint64_t bytes, bool equal) {
        if (cls < 0 || cls >= kClasses) cls = 0;
        validations_[cls].fetch_add(1, std::memory_order_relaxed);
        bytes_[cls].fetch_add(bytes, std::memory_order_relaxed);
        if (!equal) {
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

    // One line per class plus the figure the decision turns on: the share of compared bytes a
    // fault-free `GetWriteWatch` could cover (private memory only).
    std::string format() const {
        static const char* const names[kClasses] = {
            "untracked", "private", "mapped-view", "other"};
        uint64_t total_bytes = 0;
        for (int i = 0; i < kClasses; ++i) total_bytes += row(i).bytes;
        std::string out;
        char line[320];
        for (int i = 0; i < kClasses; ++i) {
            const Row r = row(i);
            const double share = total_bytes ? 100.0 * static_cast<double>(r.bytes) /
                                                   static_cast<double>(total_bytes) : 0.0;
            std::snprintf(line, sizeof(line),
                          "[validation-census] class=%s validations=%llu bytes=%llu (%.1f%% of compared "
                          "bytes) changed=%llu changed_bytes=%llu\n",
                          names[i], static_cast<unsigned long long>(r.validations),
                          static_cast<unsigned long long>(r.bytes), share,
                          static_cast<unsigned long long>(r.changed),
                          static_cast<unsigned long long>(r.changed_bytes));
            out += line;
        }
        const Row priv = row(static_cast<int>(prosper::host::MappingClass::Private));
        const double private_share = total_bytes ? 100.0 * static_cast<double>(priv.bytes) /
                                                       static_cast<double>(total_bytes) : 0.0;
        std::snprintf(line, sizeof(line),
                      "[validation-census] GetWriteWatch-coverable (private memory only) = %.1f%% of "
                      "%llu compared bytes\n",
                      private_share, static_cast<unsigned long long>(total_bytes));
        out += line;
        return out;
    }

private:
    std::atomic<uint64_t> validations_[kClasses] = {};
    std::atomic<uint64_t> bytes_[kClasses] = {};
    std::atomic<uint64_t> changed_[kClasses] = {};
    std::atomic<uint64_t> changed_bytes_[kClasses] = {};
};

// Never destroyed: the end-of-run report reads it after static destruction has begun elsewhere.
inline ValidationMappingCensus& validation_mapping_census() {
    static ValidationMappingCensus* const census = new ValidationMappingCensus();
    return *census;
}

// Prints nothing when it counted nothing, so an unrelated run stays quiet.
inline bool report_validation_mapping_census() {
    const ValidationMappingCensus& c = validation_mapping_census();
    uint64_t total = 0;
    for (int i = 0; i < ValidationMappingCensus::kClasses; ++i) total += c.row(i).validations;
    if (!total) return false;
    const std::string text = c.format();
    std::fputs(text.c_str(), stderr);
    return true;
}

// Called by the shared guest compare. A latched presence check when the switch is off.
inline void note_validation_mapping(uint64_t address, uint64_t bytes, bool equal) {
    if (!PROSPER_ENV_ON("PROSPER_VALIDATION_MAPPING_CENSUS")) return;
    // Registered on first use, never during static initialisation (see exit_census.hpp).
    static const bool registered = (prosper::diagnostics::register_census(
        nullptr, [] { return report_validation_mapping_census(); }), true);
    (void)registered;
    validation_mapping_census().record(static_cast<int>(prosper::host::classify_host_mapping(address)), bytes, equal);
}

}  // namespace prosper::frontend::submit_renderer
