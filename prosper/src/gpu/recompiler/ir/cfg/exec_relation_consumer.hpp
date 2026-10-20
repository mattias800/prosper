// Guarded consumption of static EXEC relations (IR_CONTROL_FLOW_PLAN step 7b).
// An emission scope maps immutable SPIR-V ids to unambiguous static values. An absent mapping,
// conflicting observation or unequal answer keeps the emitter's id-based answer. In particular,
// this guard cannot introduce the static pass's broader loop induction or WQM-copy assumptions.
#pragma once

#include "gpu/recompiler/ir/cfg/exec_values.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
}   // namespace prosper::gpu

namespace prosper::gpu::ir::cfg {

using ExecPartition = std::optional<std::pair<uint32_t, uint32_t>>;

struct ExecRelationCounts {
    size_t equal = 0, differs = 0, unmapped = 0;
};

// Optional observation and fault injection for tests: corrupt the static answers before the
// equality guard, or withhold them altogether to reproduce the previous consumer.
class ExecRelationCapture {
public:
    enum class Mode { Observe, Withhold, Corrupt };
    explicit ExecRelationCapture(Mode mode = Mode::Observe);
    ~ExecRelationCapture();
    ExecRelationCapture(const ExecRelationCapture&) = delete;
    ExecRelationCapture& operator=(const ExecRelationCapture&) = delete;
    Mode mode;
    ExecRelationCounts within, complement;
    static ExecRelationCapture* active();

private:
    ExecRelationCapture* previous_;
};

// One scope per emission route, nested routes restoring their caller's map on return. Neither
// this scope nor its observations allocate a SPIR-V id or alter RegState.
class ExecRelationScope {
public:
    explicit ExecRelationScope(const SpirvCompute& builder);
    ~ExecRelationScope();
    ExecRelationScope(const ExecRelationScope&) = delete;
    ExecRelationScope& operator=(const ExecRelationScope&) = delete;
    static ExecRelationScope* active();
    void begin_program(const GuestCfg& cfg, std::span<const Rdna2Inst> ins, bool dispatcher);
    void observe(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in, bool after);
    std::optional<bool> within(uint32_t reader, uint32_t owner) const;
    std::optional<ExecPartition> complement(uint32_t value) const;

private:
    void note(uint32_t id, uint32_t value);
    uint32_t value_of(uint32_t id) const;
    uint32_t id_of(uint32_t value) const;
    const SpirvCompute& builder_;
    ExecRelationScope* previous_;
    ExecValues values_;
    std::vector<Rdna2Inst> ins_;
    std::unordered_map<uint32_t, uint32_t> index_of_pc_;
    std::map<uint32_t, uint32_t> value_of_id_;   // kNone permanently invalidates a conflicting id
};

bool guarded_exec_within(uint32_t reader, uint32_t owner, bool by_id);
ExecPartition guarded_exec_complement(uint32_t value, ExecPartition by_id);

}   // namespace prosper::gpu::ir::cfg
