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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

namespace prosper::gpu {

struct ComputeProgramFacts {
    std::vector<uint32_t> code;             // exact bytes the facts were derived from
    uint64_t address = 0;
    std::vector<Rdna2Inst> decoded;         // rdna2_walk(code, dwords)
    bool prefers_native_multiwave = false;  // compute_shader_prefers_native_multiwave(decoded, ...)
    bool uses_gds = false;                  // a DS GDS access the dispatch must bind a buffer for
    // Terminal reject reasons the probe recorded, replayed for the current address on every use.
    std::vector<std::pair<std::string, std::string>> probe_reject_reasons;
};

struct ComputeProgramFactsStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;       // opt-out or verbose diagnostics: computed, not cached
    uint64_t probe_evaluations = 0;
    uint64_t entries = 0;
    uint64_t bytes = 0;
};

// Facts for the program at `code` (a guest pointer valid for `dwords` dwords). The probe's reject
// reasons have already been (re)recorded for `diagnostic.program_address` when this returns.
std::shared_ptr<const ComputeProgramFacts> compute_program_facts(
    const uint32_t* code, size_t dwords, const RecompileDiagnosticContext& diagnostic);

// False under PROSPER_NO_COMPUTE_PROGRAM_FACTS_CACHE=1: every dispatch re-derives its facts and
// the executor also restores the unconditional path-specialization copy (a same-binary A/B arm).
bool compute_program_facts_cache_enabled();

ComputeProgramFactsStats compute_program_facts_stats();
void reset_compute_program_facts_for_test();

// True when specialize_compute_resource_paths could change anything for this table: only a
// proven-null BVH resource or (wave64) a zero-record raw buffer can specialize a path. When false
// both specializers return 0 without touching the instruction stream or the table, so the caller
// may skip copying the decoded program into a scratch stream at all.
bool compute_resource_paths_may_specialize(const ShaderResourceTable& table, uint32_t wave_size);

} // namespace prosper::gpu
