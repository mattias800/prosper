// command_processor.hpp — apply a decoded PM4 stream to a GPU state snapshot.
//
// This is the second CommandProcessor stage (after pm4_decode): walk the decoded ops and fold them
// into a `GpuState` — the register files (context/shader/user-config) plus index type and the list of
// draws. A later stage translates a GpuState + its draws into Vulkan pipelines/draw calls (M4).
//
// Register application matches Kyty's CommandProcessor (GraphicsRun.cpp `cp_op_indirect_cx_regs`,
// MIT): a Set{Cx,Sh,Uc}RegistersIndirect packet points at an array of `num_regs` {offset, value}
// pairs (a ShaderRegister[]); each writes `value` into the class's register file at `offset`. Because
// prosper maps the guest image 1:1, the packet's `regs_vaddr` is directly readable as a host pointer.
#pragma once
#include "pm4_decode.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {

// A {register offset, value} pair — the element type of a Set*RegistersIndirect array.
struct ShaderReg { uint32_t offset; uint32_t value; };

// Folded GPU state after replaying a command stream.
struct GpuState {
    std::unordered_map<uint32_t, uint32_t> cx, sh, uc;   // register files by offset
    uint32_t index_type = 0;                             // last SetIndexType
    // A draw + the register state AT THE DRAW. A submit changes shaders/mask/blend between its draws,
    // so the state a draw actually uses is the register values when its packet executes — NOT the
    // end-of-submit fold. `state` is a snapshot captured at the draw (shared between consecutive draws
    // with no register write in between); it is null for draws pushed without apply() (hand-built test
    // states), and callers fall back to the folded end state. Foundational for a future multi-draw
    // executor (a reference consumer exists in the sibling PR #31); the default renderer still uses the
    // folded state, so this field is inert until that executor lands. (Snapshot infra ported from #31.)
    // Indexed draws (sceAgcDcbDrawIndex): `indexed` is set only when the packet carried the full
    // operand set; `index_addr` is the guest address of the index buffer (1:1-mapped, so directly
    // readable) and `modifier` the raw 64-bit draw-modifier bits. The index ELEMENT SIZE lives in the
    // draw's register snapshot (`state->index_type`, from the preceding SetIndexType; 0 = 16-bit,
    // 1 = 32-bit). The executor fetches the index data and renders with vkCmdDrawIndexed (#64).
    struct Draw {
        uint32_t index_count;
        std::shared_ptr<const GpuState> state;
        bool     indexed = false;
        uint64_t index_addr = 0;
        uint64_t modifier = 0;
    };
    std::vector<Draw> draws;                             // one per DrawIndexAuto / DrawIndex

    // Safety cap on an indirect register count (a malformed/huge count won't run away).
    static constexpr uint32_t kMaxRegsPerPacket = 4096;

    // Fold one decoded command into the state. Reads register arrays via their (1:1-mapped) vaddr.
    void apply(const Pm4Command& c);

    // State-at-draw snapshot for the i-th draw, or the folded end state if that draw has no snapshot.
    const GpuState& state_at_draw(size_t i) const {
        return (i < draws.size() && draws[i].state) ? *draws[i].state : *this;
    }

private:
    // Snapshot sharing: one snapshot is reused by consecutive draws until a register write dirties it.
    std::shared_ptr<const GpuState> last_snapshot_;
    bool state_dirty_ = true;
};

// Decode `dwords` dwords at `buf` and apply every op to `st`. Returns the number of packets applied.
size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st);

} // namespace prosper::gpu
