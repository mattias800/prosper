// command_processor.cpp — see command_processor.hpp.
#include "command_processor.hpp"
#include "hle/sync_futex.hpp"   // wake_label_waiters (shared with sceKernelWaitOnAddress's futex)

// hle_graphics.cpp: perform the videoout flip for an in-stream SetFlip packet — advances the flip
// status (count/flipArg/currentBuffer) that sceVideoOutGetFlipStatus reports, exactly like the API
// flip does. The game's frame pacer polls that status for its submitted flipArg; a dropped in-stream
// flip stalls the frame loop at one rendered frame.
extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx, uint32_t flip_mode, int64_t flip_arg);
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace prosper::gpu {

// Wake any thread blocked in sync_on_address (a futex) on `addr`. A GPU completion label write only
// changes memory; a futex waiter does NOT wake on a value change — it needs an explicit FUTEX_WAKE. The
// game's render/producer threads sync_on_address on the very labels the GPU writes via RELEASE_MEM /
// WRITE_DATA, so without this wake they block forever on already-satisfied semaphores (the documented
// 3-thread render deadlock — see hle_kernel_mem.cpp). This provides that missing GPU-completion wake.
// wake_label_waiters shares the sync HLE's futex implementation and skips the syscalls (this runs per
// RELEASE_MEM/WRITE_DATA packet) when no thread is blocked.
// CONFIDENCE: HIGH (matches the futex model of sceKernelWaitOnAddress; guest+host share the address space).
static void wake_on_label(uint64_t addr) { wake_label_waiters(addr); }

// Disabled only for bring-up bisection. Honoring the Dcb's memory writes is correct default behavior:
// because our CommandProcessor folds each submit synchronously, the pipe has "drained" by the time we
// apply a packet, so this IS the end-of-pipe moment. Set PROSPER_NO_EOP_WRITE=1 to suppress the writes.
static bool eop_writes_disabled() {
    const char* off = getenv("PROSPER_NO_EOP_WRITE");
    return off && off[0] == '1';
}

// A monotonic 64-bit "GPU clock" for RELEASE_MEM data_sel==3 (GpuClock64). Real hardware writes the GPU
// timestamp; a strictly increasing counter has the property the game's poll needs (a later fence reads a
// larger value). CONFIDENCE: MED — units differ from HW ticks, but monotonicity is what a >= poll checks.
static uint64_t gpu_clock64() {
    return (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
}

// Honor a RELEASE_MEM / EVENT_WRITE_EOP completion write. data_sel (Kyty GraphicsCbReleaseMem allows {2,3};
// shadPS4 DataSelect enum): 1=write 32-bit value, 2=write 64-bit value, 3=write 64-bit GPU clock. The write
// uses memcpy so an only-4-byte-aligned 64-bit label is handled portably. CONFIDENCE: HIGH — address,
// data_sel and value are decoded directly from the packet the game's ReleaseMem call built.
static void honor_eop_write(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.rel_addr || (c.rel_addr & 3)) return;
    void* dst = (void*)(uintptr_t)c.rel_addr;
    switch (c.rel_data_sel) {
        // data_sel==0 is "interrupt only, NO data write" (PM4 spec) — writing anyway clobbers 8 bytes
        // at a live label address (and a mis-extraction that yields 0 has a garbage value dword too,
        // so skipping is right in both readings). CONFIDENCE: MED.
        case 0: return;
        case 1: { uint32_t v = (uint32_t)c.rel_value; memcpy(dst, &v, sizeof v); break; }
        case 2: { uint64_t v = c.rel_value;           memcpy(dst, &v, sizeof v); break; }
        case 3: { uint64_t v = gpu_clock64();         memcpy(dst, &v, sizeof v); break; }
        // A RELEASE_MEM ALWAYS writes a completion fence — that's its purpose. Our stack-arg ABI
        // extraction can mis-read data_sel (it arrives as a pointer, not the 1/2/3 enum), so don't SKIP the
        // write on an unrecognized selector: default to the 64-bit value. Skipping it starved the render
        // thread's completion wait and stalled the frame loop after the first real draw. CONFIDENCE: MED.
        default:
            // ... but only when the packet actually carried a value: a short-decoded packet's
            // rel_value is a fabricated 0 that could move a fence label BACKWARDS.
            if (!c.rel_value_valid) return;
            { uint64_t v = c.rel_value;               memcpy(dst, &v, sizeof v); }
            break;
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   EOP write [0x%llx] data_sel=%u value=0x%llx\n",
                (unsigned long long)c.rel_addr, c.rel_data_sel, (unsigned long long)c.rel_value);
    wake_on_label(c.rel_addr);   // wake any sync_on_address futex waiter on this completion label
}

// Honor a WRITE_DATA packet: copy the inline dwords to the destination address (same synchronous timing).
static void honor_write_data(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.wd_addr || (c.wd_addr & 3) || !c.wd_data || !c.wd_num) return;
    memcpy((void*)(uintptr_t)c.wd_addr, c.wd_data, (size_t)c.wd_num * 4);
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   WriteData [0x%llx] %u dwords (first=0x%08x)\n",
                (unsigned long long)c.wd_addr, c.wd_num, c.wd_data[0]);
    wake_on_label(c.wd_addr);   // wake any sync_on_address futex waiter on this written label
}

void GpuState::apply(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::SetRegsIndirect: {
            if (c.regs_vaddr == 0 || c.num_regs == 0 || c.num_regs > kMaxRegsPerPacket) return;
            auto* regs = reinterpret_cast<const ShaderReg*>(static_cast<uintptr_t>(c.regs_vaddr));
            auto& file = (c.reg_class == RegClass::Cx) ? cx
                       : (c.reg_class == RegClass::Sh) ? sh : uc;
            if (getenv("PROSPER_RESDUMP")) {
                const char* cn = c.reg_class == RegClass::Cx ? "Cx" : c.reg_class == RegClass::Sh ? "Sh" : "Uc";
                fprintf(stderr, "[regindir] class=%s num=%u vaddr=0x%llx first pairs:", cn, c.num_regs,
                        (unsigned long long)c.regs_vaddr);
                for (uint32_t i = 0; i < c.num_regs && i < 6; i++)
                    fprintf(stderr, " (off=0x%x val=0x%x)", regs[i].offset, regs[i].value);
                fprintf(stderr, "\n");
            }
            for (uint32_t i = 0; i < c.num_regs; i++) file[regs[i].offset] = regs[i].value;
            state_dirty_ = true;   // register state changed -> the next draw needs a fresh snapshot
            break;
        }
        case K::SetShRegDirect:
            // SET_SH_REG writes a consecutive RANGE (the driver uploads the whole user-data SGPR block
            // this way). Write every value, not just the first — else all descriptors past the range's
            // first register are silently dropped (which left the shaders' V#/T# SGPRs empty).
            if (c.sh_reg_data && c.sh_reg_count) {
                for (uint32_t k = 0; k < c.sh_reg_count && c.sh_reg_count <= kMaxRegsPerPacket; k++)
                    sh[c.sh_reg_offset + k] = c.sh_reg_data[k];
            } else {
                sh[c.sh_reg_offset] = c.sh_reg_value;   // fallback (single-register / legacy path)
            }
            state_dirty_ = true;
            break;
        case K::SetIndexType:
            index_type = c.index_size;
            state_dirty_ = true;
            break;
        case K::DrawIndexAuto: {
            // Snapshot the register state AT THE DRAW (shared with consecutive draws until a register
            // write dirties it), so a future per-draw executor can render each draw under its own
            // shaders/mask/blend instead of the end-of-submit fold. Inert for the current renderer.
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            draws.push_back({ c.index_count, last_snapshot_ });
            break;
        }
        case K::ReleaseMem:
            honor_eop_write(c);     // EOP completion label write (correct synchronous end-of-pipe timing)
            break;
        case K::WriteData:
            honor_write_data(c);    // inline data write requested by the Dcb
            break;
        case K::Flip:
            // The GPU reaching the SetFlip packet IS the flip moment (synchronous fold): perform the
            // videoout flip so GetFlipStatus advances and the game's frame pacer sees its flipArg
            // complete. Only for a fully-decoded payload — a short packet must not fabricate a flip.
            if (c.flip_valid) prosper_vo_flip_from_gpu(c.flip_handle, c.flip_bufidx, c.flip_mode, c.flip_arg);
            break;
        default:
            break;   // events / waits / unknown: no register-state effect (handled later)
    }
}

size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st) {
    std::vector<Pm4Command> ops;
    decode_pm4(buf, dwords, ops);
    for (const auto& c : ops) st.apply(c);
    return ops.size();
}

} // namespace prosper::gpu
