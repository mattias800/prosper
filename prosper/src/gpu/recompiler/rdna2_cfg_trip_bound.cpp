// Guest-GDS exclusion and phase-specific bounds for the optional CFG witness.
#include "gpu/recompiler/rdna2_cfg_trip_bound.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
// True when the guest program itself reads or writes GDS. The witness lives in the internal GDS
// buffer, which is guest-addressable, so instrumenting such a program would change its INPUT -- and
// a diagnostic that perturbs the state it measures can manufacture or suppress the behaviour under
// test. Decoded from the whole program rather than one phase: a GDS access in any phase disqualifies
// the program, and `ins` here is only the current phase's slice.
bool program_touches_guest_gds(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return false;
    std::vector<Rdna2Inst> decoded;
    if (!rdna2_walk(code, dwords, decoded)) return true;   // undecodable: refuse, fail closed
    return std::any_of(decoded.begin(), decoded.end(),
                       [](const Rdna2Inst& in) { return in.fmt == Rdna2Format::DS && in.ds_gds; });
}

uint32_t emitted_loop_trip_bound(uint64_t program_address, uint32_t phase, uint32_t start_pc,
                                 uint32_t end_pc, const uint32_t* code, size_t dwords) {
    const ComputeTripBoundSettings settings = compute_trip_bound_settings();
    const uint32_t bound = settings.bound;
    if (!bound) return 0u;
    if (settings.only_program && program_address != settings.only_program) return 0u;

    // A PHASE SELECTOR IS REQUIRED, and this is a coherence requirement rather than ergonomics.
    //
    // A barrier-phased program emits one dispatcher per phase, and each phase has its OWN dispatch
    // table -- ordinal 9 means different guest pcs in phase 0 and phase 2. The witness is a single
    // record: if two phases can hit during one dispatch, its phase field is whichever invocation
    // stored last and its ordinal extrema are a mixture of two incompatible maps, which the host then
    // prints as one phase's range. There is no reading of that record that is true.
    //
    // Discovery still works with the bound armed and no phase chosen: every phase prints its dispatch
    // map (see the caller), so one run tells you how many phases exist and what each covers. Only the
    // emission is withheld.
    if (settings.only_phase == ComputeTripBoundSettings::kAllPhases) {
        static std::once_flag once;
        std::call_once(once, [] {
            fprintf(
                stderr,
                "[cfg-trip-bound] PROSPER_CFG_TRIP_BOUND_PHASE is REQUIRED and is unset: no "
                "bound emitted. One witness record cannot describe two phases -- their dispatch "
                "ordinals index different tables. The dispatch maps below list every phase; "
                "re-run with PROSPER_CFG_TRIP_BOUND_PHASE=<k>.\n");
        });
        return 0u;
    }
    if (phase != settings.only_phase) return 0u;

    // Refuse to instrument a program that uses GDS itself; see program_touches_guest_gds.
    if (program_touches_guest_gds(code, dwords)) {
        static std::mutex refused_mutex;
        static std::set<uint64_t> refused;
        bool first = false;
        {
            std::lock_guard lock(refused_mutex);
            first = refused.insert(program_address).second;
        }
        if (first)
            fprintf(stderr,
                    "[cfg-trip-bound] program 0x%llx REFUSED: it accesses GDS itself, and the "
                    "witness would overwrite its data. Not instrumented.\n",
                    static_cast<unsigned long long>(program_address));
        return 0u;
    }

    static std::mutex announce_mutex;
    static std::set<std::pair<uint64_t, uint32_t>> announced;
    bool first = false;
    {
        std::lock_guard lock(announce_mutex);
        first = announced.insert({program_address, phase}).second;
    }
    if (first) {
        char scope[128] = "every back-edge traversal";
        if (settings.only_ordinal != ComputeTripBoundSettings::kAllOrdinals)
            snprintf(scope, sizeof scope,
                     "ONLY traversals about to dispatch ordinal %u (see this phase's dispatch map)",
                     settings.only_ordinal);
        fprintf(stderr,
                "[cfg-trip-bound] program 0x%llx phase %u (guest pc %u..<%u, end-exclusive) "
                "bounded at %u iterations counting %s "
                "(DIAGNOSTIC: truncates guest control flow)\n",
                static_cast<unsigned long long>(program_address), phase, start_pc, end_pc, bound,
                scope);
    }
    return bound;
}

}   // namespace prosper::gpu
