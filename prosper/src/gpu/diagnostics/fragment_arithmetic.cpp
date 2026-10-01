#include "gpu/diagnostics/fragment_arithmetic.hpp"
#include "diagnostics/perf/perf_ledger.hpp"
#include <cstdio>
#include <mutex>
#include <set>
#include <tuple>

namespace prosper::gpu {

void observe_fragment_arithmetic(const FragmentArithmeticObservation& observation,
                                 uint64_t lookup_program, bool module_produced) {
    namespace perf = prosper::diagnostics::perf;
    // F9 re-realization is not an additional live compiler-use population. Ordinary offline
    // callers have no suppression scope and still announce, even before any flip window exists.
    if (perf::thread_draw_drop_suppression()) return;
    const auto add = [](perf::Counter counter) { if (perf::enabled()) perf::add(counter); };
    add(perf::Counter::FragmentArithmeticRequests);
    if (!observation.families) return;
    add(observation.float_mode.available ? perf::Counter::FragmentArithmeticKnownMode
                                         : perf::Counter::FragmentArithmeticUnknownMode);
    if (!module_produced) add(perf::Counter::FragmentArithmeticRefusedRequests);
    if (observation.families & 1u) add(perf::Counter::FragmentArithmeticAddRequests);
    if (observation.families & 2u) add(perf::Counter::FragmentArithmeticMulRequests);
    if (observation.truncated_emissions)
        add(perf::Counter::FragmentArithmeticTruncatedRequests);

    // A bounded provenance/site-prefix inventory, NOT a unique shader census. Include both the
    // current lookup address and immutable producer address; address aliases sharing a module are
    // announced rather than silently inheriting the first address's warning.
    using SitePrefix = std::array<uint64_t, FragmentArithmeticObservation::kSiteLimit>;
    using Key = std::tuple<uint64_t, uint64_t, uint64_t, bool, uint8_t, uint8_t, SitePrefix, bool, bool>;
    struct State {
        std::mutex mutex;
        std::set<Key> seen;
        bool full_announced = false;
    };
    static State* const state = new State; // exit-safe, first-use only
    SitePrefix prefix{};
    for (size_t i = 0; i < observation.site_count; ++i)
        prefix[i] = (uint64_t(observation.sites[i].pc) << 8) |
                    (1u + static_cast<uint8_t>(observation.sites[i].family));
    const Key key{lookup_program, observation.producing_program, observation.source_fingerprint,
                  observation.float_mode.available, observation.float_mode.value,
                  observation.families, prefix, observation.truncated_emissions != 0, module_produced};
    std::lock_guard lock(state->mutex);
    if (state->seen.contains(key)) return;
    constexpr size_t kAnnouncementLimit = 1024;
    if (state->seen.size() == kAnnouncementLimit) {
        add(perf::Counter::FragmentArithmeticInventoryOverflowRequests);
        if (!state->full_announced) {
            state->full_announced = true;
            std::fprintf(stderr, "[fragment-arithmetic-unverified] announcement inventory full "
                "(1024); further provenance/site lines suppressed; request counters continue; "
                "see unverified-fragment-f32-arithmetic alarm; inventory=ADD/MUL-only\n");
        }
        return;
    }
    state->seen.insert(key);
    char mode[24] = "unavailable";
    if (observation.float_mode.available)
        std::snprintf(mode, sizeof mode, "0x%02x", observation.float_mode.value);
    for (size_t i = 0; i < observation.site_count; ++i) {
        const auto& site = observation.sites[i];
        std::fprintf(stderr, "[fragment-arithmetic-unverified] stage=fragment program=0x%llx "
            "producing-program=0x%llx source-fingerprint=%016llx pc=%u pc-unit=dwords family=F32-%s "
            "FLOAT_MODE=%s module-produced=%s observation=compiler-request "
            "semantics=host-dependent-unverified(input/output-denorm,rounding) "
            "inventory=ADD/MUL-only sites-truncated=%s next=retain-guest-mode-and-raw-program; "
            "compare-denorm/rounding-oracle; cf.#4059 (not-a-measured-GPU-failure)\n",
            static_cast<unsigned long long>(lookup_program),
            static_cast<unsigned long long>(observation.producing_program),
            static_cast<unsigned long long>(observation.source_fingerprint), site.pc,
            site.family == FragmentArithmeticFamily::Add ? "ADD" : "MUL", mode,
            module_produced ? "yes" : "no", observation.truncated_emissions ? "yes" : "no");
    }
}

} // namespace prosper::gpu
