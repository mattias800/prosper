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
    // Gen5 indexed-draw binding state (issue #232, DOLL/UE4 geometry). SetIndexBuffer/SetIndexCount
    // set these; DrawIndexOffset consumes them to emit an indexed Draw. Persist across draws within a
    // submit (as on hardware) until re-set.
    uint64_t index_base = 0;                             // last SetIndexBuffer address
    uint32_t index_num  = 0;                             // last SetIndexCount
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
        // Gen5 DrawIndexOffset provenance (#304): the index ELEMENT SIZE is never set by DOLL (no
        // SetIndexSize / no VGT_INDEX_TYPE register), so index_type defaults to 16-bit — but DOLL's
        // UE4 Slate/UMG quad index buffers are actually 32-bit. The executor auto-detects the real
        // element size from buffer content and, for an offset draw, must recompute the address at the
        // detected element size (index_base + index_offset*elem). Store the raw base + element offset
        // so it can. (A DrawIndex carries an absolute index_addr instead; from_offset stays false.)
        uint64_t index_base   = 0;
        uint32_t index_offset = 0;
        bool     from_offset  = false;
    };
    std::vector<Draw> draws;                             // one per DrawIndexAuto / DrawIndex
    uint64_t dispatch_count = 0;                         // DispatchDirect packets seen (no execution yet)

    // GPU predication window (#319): the 64-bit condition address the last SetPredication opened
    // (0 = no window). A packet-predicated Jump inside the window is executed/skipped on the
    // condition value read at fold time. Cleared by the end form (addr == 0).
    uint64_t pred_cond_addr = 0;
    // Jump recursion depth (a jump target could itself contain a jump; bounded to stop a cycle).
    uint32_t jump_depth = 0;

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

// WAIT_REG_MEM queue-pause support (issue #312 — see the block comment in command_processor.cpp).
// last_fold_deferred(): the most recent run_command_buffer hit an unsatisfied wait, so its
// remaining memory effects (and its EOP equeue pulse) are pending in the deferred FIFO.
// flush_deferred_streams(): re-check the FIFO (call at every submit, under the same submit mutex
// as run_command_buffer); returns how many deferred streams COMPLETED — the caller owes one EOP
// equeue pulse per completed stream.
bool last_fold_deferred();
int  flush_deferred_streams();

} // namespace prosper::gpu

// Apply every pending pipe-drain completion write NOW, in submission order (#312 — see the
// deferred-completion-write block in command_processor.cpp). Callable from C (the EOP-event
// worker in hle_kernel_time.cpp binds it weakly).
extern "C" void prosper_gpu_drain_completion_writes();

namespace prosper::gpu {

} // namespace prosper::gpu
