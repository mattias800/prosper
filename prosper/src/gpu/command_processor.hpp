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
    // A draw + the register state AT THE DRAW. The render state, shaders, and descriptor SGPRs a
    // draw uses are the register values when the draw packet executes — NOT the end-of-submit fold
    // (the game changes mask/blend/shaders between the draws of one submit; last-writer-wins state
    // rendered every draw with the final draw's pipeline, which black-masked whole frames).
    // `state` shares one snapshot between consecutive draws with no register writes in between; it
    // is null when a draw was pushed without apply() (hand-built states in tests) — callers fall
    // back to the folded end state then.
    struct Draw { uint32_t index_count; std::shared_ptr<const GpuState> state; };
    std::vector<Draw> draws;                             // one per DrawIndexAuto

    // In-stream flips (sceAgcDcbSetFlip), collected during the fold and executed AFTER the submit's
    // draws render (execute_pending_flips) — the Dcb queues its flip after the frame's draws, and the
    // flip's scanout must observe the frame content, not precede it.
    struct PendingFlip { uint32_t handle; int32_t bufidx; uint32_t mode; int64_t arg; };
    std::vector<PendingFlip> pending_flips;

    // Safety cap on an indirect register count (a malformed/huge count won't run away).
    static constexpr uint32_t kMaxRegsPerPacket = 4096;

    // Fold one decoded command into the state. Reads register arrays via their (1:1-mapped) vaddr.
    void apply(const Pm4Command& c);

private:
    // Snapshot sharing: reused by consecutive draws until a register write dirties the state.
    std::shared_ptr<const GpuState> last_snapshot_;
    bool state_dirty_ = true;
};

// Decode `dwords` dwords at `buf` and apply every op to `st`. Returns the number of packets applied.
size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st);

// Execute (and clear) the submit's collected in-stream flips: advance the videoout flip status, fire
// the flip-completion events, and scan out the flipped guest buffer. Call AFTER the submit's draws
// have been executed — the flip packet follows the frame's draws in the Dcb.
void execute_pending_flips(GpuState& st);

} // namespace prosper::gpu
