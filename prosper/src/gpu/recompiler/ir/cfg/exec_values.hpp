// exec_values.hpp -- static identities for EXEC values and the subset/complement relations
// between them (IR_CONTROL_FLOW_PLAN.md step 7, ADR 0038).
//
// The emitter relates EXEC values by SPIR-V id (RegState::exec_subset_of, exec_complement_in), so
// every relation is lost where the dispatcher reloads EXEC at a case entry. This pass gives each
// EXEC definition an identity that does not depend on emission: the entry value, one value per
// writing instruction, a phi where reaching definitions differ, and a reload per dispatcher case.
// The relations are the ones the emitter's recognisers record (record_exec_mask_subset,
// exec_masked_predicate, write_vopc_mask, merge_exec_arm, the narrowing loop header), stated over
// those identities. The rules, and where they reach further than the emitter, are in AGENTS.md.
//
// Plain data: building it allocates no SPIR-V id. Nothing in emission reads it yet (step 7a); the
// shadow census (exec_value_census.hpp) compares it with the SPIR-V-id relations.
#pragma once

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace prosper::gpu::ir::cfg {

// Mask locations, by the register number the emitter keys them with: EXEC (126; 127 is the same
// location), VCC (106; 107 likewise) and an SGPR mask by its base register (0..105).
inline constexpr int kExecLocation = 126;
inline constexpr int kVccLocation = 106;
// The mask location an operand names, or -1.
int exec_mask_location(const Operand& operand);

struct ExecValue {
    enum class Kind : uint8_t { Entry, Write, Phi, DispatchReload };
    Kind kind = Kind::Entry;
    uint32_t block = kNone;   // where it is defined (Phi, DispatchReload: the block it opens)
    uint32_t inst = kNone;   // Write: the writer's index in the instruction vector
    // Where a Write is defined. A lane mask a compare or a 64-bit mask operation writes to VCC or
    // an SGPR pair is a value too: it becomes EXEC through a copy.
    int location = kExecLocation;
    // The value this one equals: the source of a copy into EXEC (s_mov_b64, s_wqm_b64), and for a
    // DispatchReload the value that reaches its block. kNone otherwise.
    uint32_t same_as = kNone;
};

const char* exec_value_kind_name(ExecValue::Kind kind);

// complement_in[a] = {whole, part}: a = whole & ~part, with part proved within whole.
struct ExecComplement {
    uint32_t whole = kNone, part = kNone;
    bool present() const { return whole != kNone; }
};

struct ExecValues {
    std::vector<ExecValue> values;
    // Per instruction index: the value in EXEC before and after it. kNone outside the graph.
    std::vector<uint32_t> exec_before, exec_after;
    // Per instruction index: the (location, value) pairs it defines.
    std::vector<std::vector<std::pair<int, uint32_t>>> defined;
    // Per value: subset_of[a] = b when a = b & x for some x; kNone when no relation is proved.
    std::vector<uint32_t> subset_of;
    std::vector<ExecComplement> complement_in;   // per value

    // Follows same_as to the value a copy chain names.
    uint32_t representative(uint32_t value) const;
    // True when `reader` is `owner` or a subset of it through subset_of and same_as. `path`, when
    // given, receives the values the proof visits, reader first.
    bool within(uint32_t reader, uint32_t owner, std::vector<uint32_t>* path = nullptr) const;
    // The complement fact of `value`'s representative.
    ExecComplement complement(uint32_t value) const;
};

struct ExecValueOptions {
    // Every block entry is a dispatcher case that reloads EXEC: the graph was built with
    // SplitPolicy::split_at_services, and each block's entry value is a DispatchReload.
    bool dispatch_reloads = false;
};

// `ins` is the instruction vector `cfg` was built from.
ExecValues build_exec_values(const GuestCfg& cfg, std::span<const Rdna2Inst> ins,
                             const ExecValueOptions& options = {});

}   // namespace prosper::gpu::ir::cfg
