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
        uint64_t command_order = 0;
    };
    std::vector<Draw> draws;                             // one per DrawIndexAuto / DrawIndex
    // Compute dispatch + register state AT the packet. Compute is not executed yet, but retaining
    // the state makes skipped-producer provenance inspectable instead of reducing every dispatch
    // to one process-lifetime counter (#524).
    struct Dispatch {
        // Raw API dimensions. Modifier.USE_THREAD_DIMENSIONS selects threads (set) or groups (clear).
        uint32_t threads_x = 0, threads_y = 0, threads_z = 0;
        uint64_t modifier = 0;
        std::shared_ptr<const GpuState> state;
        uint64_t command_order = 0;
    };
    std::vector<Dispatch> dispatches;                     // current submit's DispatchDirect packets
    uint64_t dispatch_count = 0;                          // process-lifetime DispatchDirect count
    uint64_t command_order = 0;                           // process-lifetime applied PM4 ordinal

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

// WAIT_REG_MEM barrier model, OPT-IN via PROSPER_WAIT_DEFER=1 (issue #312 — see the block
// comment in command_processor.cpp, including the measured verdict on why it is not default:
// it eliminates the canary-152 wait-ordering corruption class but a second, order-independent
// injection remains and deferral latency makes DOLL menu-drive runs die earlier overall).
// last_fold_deferred(): the most recent top-level run_command_buffer hit an unsatisfied wait, so
// its remaining memory effects are gated behind that barrier in a deferred stream.
// deferred_pending(): one or more deferred streams still hold gated effects — the caller must
// ensure a re-check cadence exists (hle_agc's 2 ms watchdog) so a satisfied/timed-out barrier
// releases even if the guest never submits again.
// flush_deferred_streams(): release the queue's gated tail in strict submission order (call at
// every submit, under the same submit mutex as run_command_buffer); returns how many streams
// fully completed, and re-fires the EOP equeue pulse when any did.
bool last_fold_deferred();
bool deferred_pending();
int  flush_deferred_streams();
// submit_completion_pulse(): fire the submit's GPU-EOP equeue pulse — immediately when no gated
// writes are pending, else OWED and delivered when the gated tail drains (the hardware contract:
// the EOP interrupt fires only after everything before it in the ring executed; pulsing earlier
// lets the guest's completion scan free label blocks our gated writes then stomp — see the
// visibility-contract block in command_processor.cpp). Call instead of prosper_eq_trigger_eop
// from the submit paths, under the submit mutex.
void submit_completion_pulse();

} // namespace prosper::gpu

// Apply every pending pipe-drain completion write NOW, in submission order (#312 — see the
// deferred-completion-write block in command_processor.cpp). Callable from C (the EOP-event
// worker in hle_kernel_time.cpp binds it weakly).
extern "C" void prosper_gpu_drain_completion_writes();

namespace prosper::gpu {

} // namespace prosper::gpu
