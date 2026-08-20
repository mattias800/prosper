#pragma once
// PROSPER_CFG_TRIP_BOUND witness ownership: save the guest's dwords, seed ours, report, restore.
//
// Header-only and separate from live_compute.cpp so the contract that matters here -- that guest GDS
// is byte-identical after an instrumented dispatch -- can be asserted by a test instead of resting on
// a reading of the code. See tests/shared/diagnostics/test_cfg_trip_bound.cpp.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>

namespace prosper::frontend {

// SAVE / PREPARE / REPORT / RESTORE.
//
// The witness lives in the internal GDS buffer, which is ONE persistent, guest-visible allocation
// shared by every dispatch. Proving that the instrumented program does not access GDS is necessary
// but not sufficient: it says nothing about the guest program that runs NEXT and does. Overwriting
// those dwords and leaving them overwritten perturbs unrelated guest state, which is exactly the
// property this diagnostic must not have -- an instrument that changes the machine it measures can
// manufacture or suppress the behaviour under test, and here it could do so for a different program
// entirely.
//
// So the original dwords are saved before the dispatch and put back after it. During the dispatch the
// values are ours (the instrumented program provably cannot read them); afterwards guest memory is
// byte-identical to what it was. The alternative -- a dedicated non-guest binding -- is the better
// long-term shape and is not what this does; save/restore is chosen because its correctness argument
// is short enough to check by reading it.
struct TripBoundWitnessScope {
    uint32_t* slots = nullptr;
    uint32_t saved[prosper::gpu::kComputeTripWitnessDwordCount]{};

    explicit TripBoundWitnessScope(const prosper::gpu::ComputeItem& item) {
        // `trip_witness_instrumented` is an EMISSION result, not an intention: a program the
        // selectors accept can still emit nothing (a structured loop, or a phase ordinal that does
        // not exist), and reading the witness on intent alone reports whatever guest data was there.
        if (!item.trip_witness_instrumented) return;
        uint8_t* gds = prosper::gpu::compute_gds_backing();
        if (!gds) return;
        slots = reinterpret_cast<uint32_t*>(gds) + prosper::gpu::kComputeTripWitnessDword;
        for (uint32_t i = 0; i < prosper::gpu::kComputeTripWitnessDwordCount; ++i)
            saved[i] = slots[i];
        slots[0] = 0;            // hit flag
        slots[1] = 0;            // phase
        slots[2] = 0;            // max trips        (atomic max)
        slots[3] = UINT32_MAX;   // min ordinal      (atomic min -- must start at the identity)
        slots[4] = 0;            // max ordinal      (atomic max)
    }

    ~TripBoundWitnessScope() {
        if (!slots) return;
        for (uint32_t i = 0; i < prosper::gpu::kComputeTripWitnessDwordCount; ++i)
            slots[i] = saved[i];
    }

    void report(const prosper::gpu::ComputeItem& item) const {
        if (!slots || !slots[0]) return;
        // Every reported field is a DISPATCHER quantity. `dispatch-range` spans switch-case ordinals,
        // which index the phase's dispatch table -- resolve one against the `dispatch map:` line the
        // same phase prints when the bound arms, and never by hand: this record's per-invocation
        // field was labelled wrongly twice, in opposite directions, from exactly that inference
        // (trap 172), and its racy form published a tighter range than the truth (trap 173).
        std::fprintf(stderr,
                     "[cfg-trip-bound] HIT program=0x%llx phase=%u trips=%u dispatch-range=%u..%u "
                     "submit=%llu dispatch=%llu order=%llu groups=%ux%ux%u\n",
                     static_cast<unsigned long long>(item.code_addr), slots[1], slots[2],
                     slots[3], slots[4],
                     static_cast<unsigned long long>(item.submit_no),
                     static_cast<unsigned long long>(item.dispatch_index),
                     static_cast<unsigned long long>(item.command_order),
                     item.launch.groups_x, item.launch.groups_y, item.launch.groups_z);
    }
};

}  // namespace prosper::frontend
