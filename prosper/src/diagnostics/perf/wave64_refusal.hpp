#pragma once
// Default-on announcements of actual, proven Wave64 refusals (#3992). No host-vendor or OS gate.
// This is separate from the lock-free ledger: only a REFUSAL pays the bounded identity lookup
// and first-observation log. Accepted uses add only a coarse counter; no guest reads or clocks.
#include "diagnostics/perf/perf_ledger.hpp"
#include <array>
#include <cstdio>
#include <mutex>

namespace prosper::diagnostics::perf {

enum class Wave64Refusal : uint8_t {
    FragmentRecompile, ComputeRecompile, FragmentSubgroup, ComputeSubgroup, Count
};
inline constexpr size_t kWave64RefusalCount = static_cast<size_t>(Wave64Refusal::Count);
inline constexpr const char* kWave64RefusalNames[] = {
    "fragment/recompile", "compute/recompile", "fragment/subgroup-contract",
    "compute/subgroup-contract"
};
inline constexpr Counter kWave64RefusalCounters[] = {
    Counter::Wave64FragmentRecompile, Counter::Wave64ComputeRecompile,
    Counter::Wave64FragmentSubgroup, Counter::Wave64ComputeSubgroup
};

// Identities are compile variants when available, otherwise run-local program addresses.
// The site is part of the key: these are distinct REFUSAL identities, not unique shader contents.
// Saturation stays explicit; it never stops counting refused uses or claims an exact shader total.
template<size_t Capacity> struct Wave64RefusalInventory {
    enum class Observation { New, Known, Unidentified, Full };
    struct Key { uint64_t id = 0; uint8_t site = 0; bool compile = false; };
    std::array<Key, Capacity> keys{};
    size_t size = 0;
    Observation observe(Wave64Refusal site, uint64_t program, uint64_t identity) {
        const Key key{identity ? identity : program, static_cast<uint8_t>(site), identity != 0};
        if (!key.id) return Observation::Unidentified;
        for (size_t i = 0; i < size; ++i)
            if (keys[i].id == key.id && keys[i].site == key.site && keys[i].compile == key.compile)
                return Observation::Known;
        if (size == Capacity) return Observation::Full;
        keys[size++] = key;
        return Observation::New;
    }
};

inline void observe_wave64_shader(uint32_t guest_wave, bool compute) {
    if (!enabled() || guest_wave != 64) return;
    if (compute ? thread_dispatch_skip_suppression() : thread_draw_drop_suppression()) return;
    add(Counter::Wave64ShaderChecks);
}

inline void note_unsupported_wave64(Wave64Refusal site, uint32_t guest_wave,
                                    uint64_t program, uint64_t identity = 0,
                                    uint32_t wave_reasons = UINT32_MAX,
                                    uint32_t host_min = 0, uint32_t host_max = 0) {
    const size_t i = static_cast<size_t>(site);
    if (!enabled() || guest_wave != 64 || i >= kWave64RefusalCount) return;
    const bool compute = site == Wave64Refusal::ComputeRecompile ||
                         site == Wave64Refusal::ComputeSubgroup;
    if (compute ? thread_dispatch_skip_suppression() : thread_draw_drop_suppression()) return;
    add(kWave64RefusalCounters[i]);
    struct State {
        std::mutex mutex;
        Wave64RefusalInventory<2048> inventory;
        bool unknown_announced[kWave64RefusalCount]{};
        bool overflow_announced = false;
    };
    static State* const state = new State(); // exit-safe; allocated only on the first refusal
    std::lock_guard<std::mutex> lock(state->mutex);
    using Observation = Wave64RefusalInventory<2048>::Observation;
    const auto observed = state->inventory.observe(site, program, identity);
    if (observed == Observation::Known) return;
    if (observed == Observation::Full) {
        add(Counter::Wave64InventoryOverflow);
        if (!state->overflow_announced) {
            state->overflow_announced = true;
            std::fprintf(stderr, "[wave64-unsupported] refusal identity inventory full (2048); "
                                 "refused-use counts remain complete, distinct identity count is "
                                 "a lower bound; see unsupported-wave64-shaders alarm\n");
        }
        return;
    }
    if (observed == Observation::Unidentified) {
        add(Counter::Wave64UnidentifiedRefusals);
        if (state->unknown_announced[i]) return;
        state->unknown_announced[i] = true;
    } else {
        add(Counter::Wave64NewRefusalIdentities);
    }
    char host[48] = "unavailable", reasons[32] = "unavailable";
    if (host_min && host_max)
        std::snprintf(host, sizeof host, "%u..%u", host_min, host_max);
    if (wave_reasons != UINT32_MAX)
        std::snprintf(reasons, sizeof reasons, "0x%x", wave_reasons);
    std::fprintf(stderr,
        "[wave64-unsupported] stage=%s program=0x%llx identity=0x%llx "
        "refusal=%s guest-wave=64 host-subgroups=%s wave-reasons=%s consequence=%s "
        "next=%s\n",
        compute ? "compute" : "fragment", (unsigned long long)program,
        (unsigned long long)identity, kWave64RefusalNames[i], host, reasons,
        compute ? "dispatch-skipped/output-unwritten" : "draw-dropped/content-missing",
        site == Wave64Refusal::FragmentRecompile || site == Wave64Refusal::ComputeRecompile
            ? "recompiler/resource-binding; PROSPER_DBG_PROGRAM=<program> for rejection pc"
            : "subgroup-contract/lowering; see the adjacent backend skip and wave reason bits");
}

} // namespace prosper::diagnostics::perf
