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
    uint32_t num_instances = 1;                          // default before SetNumInstances; zero discards
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
    // readable). Both DrawIndex and DrawIndexAuto retain their raw 64-bit `modifier`. The index
    // ELEMENT SIZE lives in the draw's register snapshot (`state->index_type`, from the preceding
    // SetIndexType; 0 = 16-bit, 1 = 32-bit). The executor fetches indexed data and renders with
    // vkCmdDrawIndexed (#64).
    struct Draw {
        uint32_t index_count;
        uint32_t instance_count = 1;
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
    // Address-backed DMA_DATA memory effects execute in the same ordered backend timeline as
    // draws/dispatches. Immediate fills retain their established completion-queue behavior until a
    // general copy appears; effects after that boundary join the ordered timeline and cannot overtake
    // it (#312/#189).
    struct DmaCopy {
        uint64_t dst = 0, src = 0;
        uint32_t bytes = 0, sels = 0;
        uint64_t command_order = 0;
        uint64_t packet_addr = 0;
    };
    std::vector<DmaCopy> dma_copies;
    // Some PM4 consumers (indirect register arrays, waits, and jump/predication memory) are still
    // folded eagerly. If one follows a retained DMA, or WAIT_DEFER owns either copy dependency,
    // executing the submit would consume stale bytes. Preserve the DMA record for diagnostics and
    // capture truthfulness, but reject backend execution of the whole submit.
    bool dma_execution_rejected = false;
    // Once a submit retains an address-backed DMA, later modeled memory effects must remain beside
    // it instead of entering the asynchronous completion FIFO and overtaking it. Own WRITE_DATA's
    // inline payload because the command buffer may be recycled before ordered execution.
    struct MemoryEffect {
        Pm4Command cmd;
        std::vector<uint32_t> write_data;

        MemoryEffect() = default;
        MemoryEffect(const Pm4Command& source, uint64_t order) : cmd(source) {
            cmd.stream_order = order;
            if (cmd.kind == Pm4Command::Kind::WriteData && cmd.wd_data && cmd.wd_num)
                write_data.assign(cmd.wd_data, cmd.wd_data + cmd.wd_num);
            rebind();
        }
        MemoryEffect(const MemoryEffect& other) : cmd(other.cmd), write_data(other.write_data) {
            rebind();
        }
        MemoryEffect& operator=(const MemoryEffect& other) {
            if (this != &other) { cmd = other.cmd; write_data = other.write_data; rebind(); }
            return *this;
        }
        MemoryEffect(MemoryEffect&& other) noexcept
            : cmd(other.cmd), write_data(std::move(other.write_data)) { rebind(); }
        MemoryEffect& operator=(MemoryEffect&& other) noexcept {
            if (this != &other) { cmd = other.cmd; write_data = std::move(other.write_data); rebind(); }
            return *this;
        }

    private:
        void rebind() {
            if (cmd.kind == Pm4Command::Kind::WriteData)
                cmd.wd_data = write_data.empty() ? nullptr : write_data.data();
        }
    };
    std::vector<MemoryEffect> ordered_memory_effects;
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
    // Register-file domain bound (#1264). Real CP register writes address a bounded physical register
    // file; an out-of-range offset lands nowhere on hardware. Titles exploit that: Blue Prince builds
    // its Set*RegistersIndirect arrays incrementally (record num=0, then PatchSetAddress +
    // PatchAddRegisters), and the counted range spans raw stale arena data (arbitrary 32-bit
    // "offsets") interleaved with the real registers. Folding those into the unordered_map register
    // files grew cx to ~94,000 dead entries whose per-draw snapshot copies (Draw case in apply())
    // took seconds per submit — the #1195-family "loading crawl" on that title.
    //
    // Scope of the "cannot change rendering" claim: no CURRENT consumer (render_state, executor,
    // timeline, capture) reads any offset at or above this bound — verified in the #1264 review.
    // Note that the bound also drops prosper's own bit31-tagged AGC VIRTUAL registers
    // (pm4_registers.hpp: CX/SH/UC_NOP, PA_SC_FSR_*, TEXTURE_GRADIENT_CONTROL, ...) when a title
    // writes back the defaults it got from GetRegisterDefaults2 — e.g. (0x80003ffd, 0) is a real
    // TEXTURE_GRADIENT_CONTROL default write, not stale data. That is safe today (nothing reads
    // them; CONFIDENCE: MED for the virtual-register encoding itself) — but if FSR / texture-
    // gradient consumption is ever implemented, ingest for those named offsets must be re-enabled
    // deliberately. Kept generous (64K dwords) versus the few-hundred range real packets use.
    static constexpr uint32_t kRegOffsetLimit = 0x10000;

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

// Execute one retained address-backed DMA_DATA operation at its ordered submit position.
// Re-validates the destination and either the raw guest source or a bounded authoritative renderer
// snapshot immediately before the byte copy, then notifies renderer caches.
void execute_ordered_dma_copy(const GpuState::DmaCopy& copy,
                              const uint8_t* authoritative_source = nullptr);
// Execute a retained post-DMA memory effect through the same mappedness, generation, freelist, and
// label-wakeup guards as the legacy completion path, then invalidate renderer-owned guest caches.
void execute_ordered_memory_effect(const GpuState::MemoryEffect& effect);

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
void submit_completion_pulse(bool submit_rejected = false);

} // namespace prosper::gpu

// Apply every pending pipe-drain completion write NOW, in submission order (#312 — see the
// deferred-completion-write block in command_processor.cpp). Callable from C (the EOP-event
// worker in hle_kernel_time.cpp binds it weakly).
extern "C" void prosper_gpu_drain_completion_writes();
// Apply only resource uploads/clears needed by the synchronous renderer. Fence/label completions
// stay queued until after the guest submit import returns.
extern "C" void prosper_gpu_drain_renderer_writes();
// Enable the stricter SDK-13 completion contract for the process. Monotonic: once a modern AGC
// caller is observed, completion labels remain post-submit for the lifetime of that game process.
extern "C" void prosper_gpu_enable_post_submit_visibility();
// Hold asynchronous fence/label publication while a guest submit import is still executing.
extern "C" void prosper_gpu_submit_scope_begin();
extern "C" void prosper_gpu_submit_scope_end();
extern "C" bool prosper_gpu_submit_scope_active();

namespace prosper::gpu {

} // namespace prosper::gpu
