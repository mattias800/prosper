#pragma once
// Default-on announcements of actual, proven Wave64 refusals (#3992). No host-vendor or OS gate.
// This is separate from the lock-free ledger: only a REFUSAL pays the bounded identity lookup
// and first-observation log. Accepted uses add only a coarse counter; no guest reads or clocks.
#include "diagnostics/perf/perf_ledger.hpp"
#include "diagnostics/perf/fragment_vote_diagnostic.hpp"
#include <array>
#include <cstdio>
#include <mutex>
#include <string>

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

// A candidate route for a refused program: names only, empty = none.
struct Wave64Candidate {
    char route[40] = "";
    char reason[64] = "";
};

// One refusal as text. Pure, so the wording is testable without capturing stderr.
//
// A RECOMPILE refusal never consulted the device. It used to print the same
// `host-subgroups=unavailable` a subgroup refusal prints when its host range is missing, and that
// read as "this GPU cannot run Wave64": Space Adventure Cobra's missing 3D (#4508) was taken that
// way on a device that offers 64-lane subgroups. Those sites now say `not-consulted`, and name
// the recompiler as the cause.
//
// `candidate_*` (ADR 0028): what the program's analysis WOULD route it to, as two trailing fields
// `candidate-route=<name> candidate-reason=<reason>` using the route vocabulary of #4754. They are
// never a second `route=`: a refused dispatch's route is `refused` and only #4754 prints it. Null
// prints no field, which is every other caller.
inline std::string wave64_refusal_line(Wave64Refusal site, uint64_t program, uint64_t identity,
                                       uint32_t wave_reasons, uint32_t host_min, uint32_t host_max,
                                       const gpu::FragmentVoteLoweringDiagnostic& lowering,
                                       const char* candidate_route = nullptr,
                                       const char* candidate_reason = nullptr) {
    const size_t i = static_cast<size_t>(site);
    if (i >= kWave64RefusalCount) return {};
    const bool compute =
        site == Wave64Refusal::ComputeRecompile || site == Wave64Refusal::ComputeSubgroup;
    const bool recompile =
        site == Wave64Refusal::FragmentRecompile || site == Wave64Refusal::ComputeRecompile;
    const char* const absent = recompile ? "not-consulted" : "unavailable";
    char host[48], reasons[32];
    std::snprintf(host, sizeof host, "%s", absent);
    std::snprintf(reasons, sizeof reasons, "%s", absent);
    if (host_min && host_max) std::snprintf(host, sizeof host, "%u..%u", host_min, host_max);
    if (wave_reasons != UINT32_MAX) std::snprintf(reasons, sizeof reasons, "0x%x", wave_reasons);
    char detail[256] = "";
    if (site == Wave64Refusal::FragmentSubgroup) {
        if (!lowering.attempted) {
            std::snprintf(detail, sizeof detail, " lowering=not-attempted");
        } else if (lowering.refusal == gpu::FragmentVoteRefusal::UnprovedVote &&
                   lowering.failed_vote.available) {
            const auto& vote = lowering.failed_vote;
            char opcode[32] = "unavailable";
            if (vote.predicate_opcode != UINT32_MAX)
                std::snprintf(opcode, sizeof opcode, "%u", vote.predicate_opcode);
            std::snprintf(detail, sizeof detail,
                          " lowering=unproved-vote vote-source-word=%zu vote-result-id=%u "
                          "vote-predicate-id=%u predicate-def-op=%s",
                          vote.source_word, vote.vote_result_id, vote.predicate_id, opcode);
        } else {
            std::snprintf(detail, sizeof detail, " lowering=%s failed-vote=unavailable",
                          gpu::fragment_vote_refusal_name(lowering.refusal));
        }
    }
    char route_field[160] = "";
    if (candidate_route && *candidate_route)
        std::snprintf(route_field, sizeof route_field, " candidate-route=%s candidate-reason=%s",
                      candidate_route, candidate_reason ? candidate_reason : "unspecified");
    char line[1280];
    std::snprintf(
        line, sizeof line,
        "[wave64-unsupported] stage=%s program=0x%llx identity=0x%llx "
        "refusal=%s guest-wave=64 host-subgroups=%s wave-reasons=%s consequence=%s "
        "next=%s%s%s\n",
        compute ? "compute" : "fragment", (unsigned long long)program, (unsigned long long)identity,
        kWave64RefusalNames[i], host, reasons,
        compute ? "dispatch-skipped/output-unwritten" : "draw-dropped/content-missing",
        recompile
            ? "prosper recompiler/resource-binding, NOT a host limit; "
              "PROSPER_DBG_PROGRAM=<program> for rejection pc"
            : "subgroup-contract/lowering; see the adjacent backend skip and wave reason bits",
        detail, route_field);
    return line;
}

// What this device offers a guest Wave64 FRAGMENT program, for one start-up line. The four facts
// are the ones the renderer's own admission test reads (render_runner.h, fragment_subgroup_skip).
struct FragmentWave64Host {
    bool size_control = false;            // an exact subgroup size can be required at all
    bool fragment_required_size = false;  // ...and for the fragment stage
    bool fragment_subgroups = false;      // subgroup operations are legal in fragment shaders
    uint32_t min_size = 0, max_size = 0;  // the device's subgroup size range
    bool native() const {
        return size_control && fragment_required_size && fragment_subgroups && min_size <= 64 &&
               max_size >= 64;
    }
};

// Both directions on purpose: a line that appears only when something is missing makes its
// absence unreadable. The narrow-host text says what a player will SEE, because that is the
// question the per-shader lines never answered: a PS5 pixel shader is compiled for 64 pixels at a
// time, and a GPU whose fragment subgroups are narrower can only run the ones whose result
// provably does not depend on that width. The rest have their draws dropped.
inline std::string fragment_wave64_host_line(const FragmentWave64Host& host) {
    char line[640];
    if (host.native()) {
        std::snprintf(
            line, sizeof line,
            "[render] guest Wave64 fragment programs: NATIVE (this GPU offers 64-lane fragment "
            "subgroups, range %u..%u)\n",
            host.min_size, host.max_size);
    } else {
        std::snprintf(
            line, sizeof line,
            "[render] WARNING: this GPU cannot run PS5 Wave64 fragment programs at their own "
            "width (fragment subgroups %u..%u, exact-size-control=%d fragment-exact-size=%d "
            "fragment-subgroup-ops=%d; 64 lanes are needed). Programs proven independent of the "
            "wave width still run; every other one has its draws DROPPED, so affected titles "
            "lose geometry or lighting. Each is named once by a [wave64-unsupported] "
            "refusal=fragment/subgroup-contract line. GPUs with 64-lane fragment subgroups run "
            "them all (AMD RDNA under RADV, for example).\n",
            host.min_size, host.max_size, static_cast<int>(host.size_control),
            static_cast<int>(host.fragment_required_size),
            static_cast<int>(host.fragment_subgroups));
    }
    return line;
}

inline void announce_fragment_wave64_host(const FragmentWave64Host& host) {
    const std::string line = fragment_wave64_host_line(host);
    std::fputs(line.c_str(), stderr);
}

inline void observe_wave64_shader(uint32_t guest_wave, bool compute) {
    if (!enabled() || guest_wave != 64) return;
    if (compute ? thread_dispatch_skip_suppression() : thread_draw_drop_suppression()) return;
    add(Counter::Wave64ShaderChecks);
}

inline void note_unsupported_wave64(Wave64Refusal site, uint32_t guest_wave, uint64_t program,
                                    uint64_t identity = 0, uint32_t wave_reasons = UINT32_MAX,
                                    uint32_t host_min = 0, uint32_t host_max = 0,
                                    const gpu::FragmentVoteLoweringDiagnostic& lowering = {},
                                    Wave64Candidate (*candidate)(const void*) = nullptr,
                                    const void* candidate_arg = nullptr) {
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
    // Only a line that will actually print pays for the analysis (after the dedupe above).
    const Wave64Candidate cand = candidate ? candidate(candidate_arg) : Wave64Candidate{};
    const std::string line = wave64_refusal_line(site, program, identity, wave_reasons, host_min,
                                                 host_max, lowering, cand.route, cand.reason);
    std::fputs(line.c_str(), stderr);
}

} // namespace prosper::diagnostics::perf
