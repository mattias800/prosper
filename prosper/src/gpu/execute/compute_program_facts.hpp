// Per-program facts every compute dispatch needs before its resources are finalized, memoized by
// exact program bytes.
//
// realize_compute_dispatches used to recompute all of these on every dispatch: decode the whole
// program (rdna2_walk), then run the native-multiwave probe, whose structured-CFG proofs
// (detect_forward_ifs -> branch_is_workgroup_uniform / sgpr_dead_at_merge) are the most expensive
// analyses the recompiler has. Every one of them is a pure function of the program's bytes, so a
// program dispatched thousands of times per second paid for the same answer thousands of times.
// Measured on Sonic Frontiers' submitting thread (2026-09-27, perf line profile): ~5-7% of all
// samples were this probe plus its allocator traffic.
//
// Exactness:
//  * An entry is keyed by (program address, dword count) and VALIDATED against a stored copy of the
//    program bytes on every lookup, so a rewritten or recycled program is re-analyzed, never served
//    stale.
//  * The probe's only side effect is recording terminal reject reasons for the program (the skip
//    diagnostic). Those records are captured at analysis time and replayed, in order, on every hit,
//    so the reject-reason map evolves exactly as it did when the probe ran per dispatch.
//  * When the verbose recompiler stream is enabled for the program (PROSPER_DBG /
//    PROSPER_DBG_PROGRAM) the cache is bypassed, so every diagnostic line is still printed per
//    dispatch.
//  * PROSPER_NO_COMPUTE_PROGRAM_FACTS_CACHE=1 recomputes every dispatch on the same binary.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "gpu/recompiler/compute_wave_route.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

namespace prosper::gpu {

// The raw nested wide-data inventory admit_compute_nested_wide_data needs: the proven nested child
// loads (rdna2_proven_raw_nested_wide_data_loads) and, only when there is a child, the proven
// immediate parent loads (rdna2_proven_raw_immediate_wide_data_loads). Both are pure functions of
// the decoded program and the most expensive per-dispatch analysis left on the submit thread once
// decode was memoized: The Blood of Dawnwalker's AgcSubmission thread spent 18 of 40 gdb stack
// samples in them (2026-10-09), recomputing the same answer for every dispatch.
struct NestedWideDataFacts {
    std::vector<uint32_t> nested;    // child load PCs; empty for almost every program
    std::vector<uint32_t> parents;   // immediate parent PCs; computed only when `nested` is not empty
};

struct ComputeProgramFacts {
    std::vector<uint32_t> code;   // exact bytes the facts were derived from
    RefusedShaderMemo refused_shader_memo;
    uint64_t address = 0;
    std::vector<Rdna2Inst> decoded;   // rdna2_walk(code, dwords)
    bool prefers_native_multiwave = false;  // compute_shader_prefers_native_multiwave(decoded, ...)
    bool uses_gds = false;                  // a DS GDS access the dispatch must bind a buffer for
    // ADR 0028: every cross-lane operation and the control-flow context it sits in. Host
    // independent, so it is memoized with the program; the route is chosen per host at the decline
    // site (select_compute_wave_route).
    // Computed on first use (a decline that is about to print its line), so a host that never refuses
    // a Wave64 compute program never pays for it.
    const ComputeWaveOpFacts& wave_ops() const;
    mutable std::once_flag wave_ops_once;
    mutable ComputeWaveOpFacts wave_ops_value;
    // Computed on first use and then served for every later dispatch of these exact bytes.
    const NestedWideDataFacts& nested_wide_data() const;
    mutable std::once_flag nested_wide_once;
    mutable NestedWideDataFacts nested_wide_value;
    // The exchange route was announced for this program (first sighting, lock-free).
    mutable std::atomic<bool> exchange_announced{false};
    // Terminal reject reasons the probe recorded, replayed for the current address on every use.
    std::vector<std::pair<std::string, std::string>> probe_reject_reasons;
};

struct ComputeProgramFactsStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;       // opt-out or verbose diagnostics: computed, not cached
    uint64_t probe_evaluations = 0;
    uint64_t nested_wide_evaluations = 0;   // nested_wide_data() analyses actually run
    uint64_t entries = 0;
    uint64_t bytes = 0;
};

// Facts for the program at `code` (a guest pointer valid for `dwords` dwords). The probe's reject
// reasons have already been (re)recorded for `diagnostic.program_address` when this returns.
std::shared_ptr<const ComputeProgramFacts> compute_program_facts(
    const uint32_t* code, size_t dwords, const RecompileDiagnosticContext& diagnostic);

// False under PROSPER_NO_COMPUTE_PROGRAM_FACTS_CACHE=1: every dispatch re-derives its facts and
// the executor also restores the unconditional path-specialization copy (a same-binary A/B arm).
// Side-effect-free lookup for a diagnostic: the cached facts when the exact bytes are cached, else
// a fresh analysis that is NOT stored. It touches no statistics and replays no terminal reject
// reasons (the full compute_program_facts overwrites `terminal_reject_reasons()[program]` with the
// probe's record on a hit, which could clobber a later, more specific reason).
std::shared_ptr<const ComputeProgramFacts>
compute_program_facts_peek(const uint32_t* code, size_t dwords, uint64_t program_address);

bool compute_program_facts_cache_enabled();

// False under PROSPER_NO_NESTED_WIDE_FACTS_MEMO=1: the executor re-derives the nested wide-data
// inventory on every dispatch, as it did before the memo (the same-binary A/B control arm).
// Diagnostic algorithm control; the admission decision is identical in both arms.
bool compute_nested_wide_facts_memo_enabled();

ComputeProgramFactsStats compute_program_facts_stats();
void reset_compute_program_facts_for_test();

// True when specialize_compute_resource_paths could change anything for this table: only a
// proven-null BVH resource or (wave64) a zero-record raw buffer can specialize a path. When false
// both specializers return 0 without touching the instruction stream or the table, so the caller
// may skip copying the decoded program into a scratch stream at all.
bool compute_resource_paths_may_specialize(const ShaderResourceTable& table, uint32_t wave_size);

} // namespace prosper::gpu
