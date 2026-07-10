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

// A monotonic 64-bit "GPU clock" for RELEASE_MEM data_sel==3 (GpuClock64). On real hardware the GPU
// EOP timestamp is the SAME counter the guest reads via sceKernelReadTsc (Kyty: GraphicsRender writes
// KernelReadTsc() for the EOP timestamp; GetGpuCoreClockFrequency == GetTscFrequency), so we share
// the guest TSC clock rather than a separate steady_clock (#156). It reports monotonic nanoseconds at
// the 1 GHz that sceKernelGetTscFrequency advertises, so a guest that reads two fence timestamps and
// divides the delta by the queried frequency gets real seconds — AND a GPU fence timestamp lies on the
// same timeline as a CPU sceKernelReadTsc value (the old steady_clock had a disjoint epoch/period).
extern "C" uint64_t prosper_guest_tsc_ns();   // hle_kernel_time.cpp — same source as sceKernelReadTsc
static uint64_t gpu_clock64() { return prosper_guest_tsc_ns(); }

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
        // Cases 1/2 need the same rel_value_valid guard the default case got: a short-decoded
        // packet's rel_value is a fabricated 0 that could move a satisfied fence label BACKWARDS
        // (re-blocking a `*label >= expected` poll).
        case 1: { if (!c.rel_value_valid) return;
                  uint32_t v = (uint32_t)c.rel_value; memcpy(dst, &v, sizeof v); break; }
        case 2: { if (!c.rel_value_valid) return;
                  uint64_t v = c.rel_value;           memcpy(dst, &v, sizeof v); break; }
        case 3: { uint64_t v = gpu_clock64();         memcpy(dst, &v, sizeof v); break; }
        // Unknown selector: LOG AND SKIP. The old default wrote the 64-bit value for ANY
        // unrecognized data_sel — a band-aid for the swap-stub stack-arg mis-extraction that made
        // data_sel arrive as a pointer (fixed in exec_image_linux.cpp emit_swap_stub: handlers now
        // see real stack args, verified live with data_sel=0x2/0x3). With the root cause gone, an
        // unknown selector means a genuinely unexpected packet: writing 8 bytes on a guess could
        // clobber the dword after a 32-bit label. Log so the gap is visible, never write.
        default:
            fprintf(stderr, "[agc] RELEASE_MEM: unknown data_sel=%u addr=0x%llx value=0x%llx — write SKIPPED\n",
                    c.rel_data_sel, (unsigned long long)c.rel_addr, (unsigned long long)c.rel_value);
            return;
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   EOP write [0x%llx] data_sel=%u value=0x%llx\n",
                (unsigned long long)c.rel_addr, c.rel_data_sel, (unsigned long long)c.rel_value);
    wake_on_label(c.rel_addr);   // wake any sync_on_address futex waiter on this completion label
}

// Honor an address-carrying EVENT_WRITE (#132): the timestamp/label variant writes a completion
// value to its address. Our GPU folds synchronously (submit == pipe drain), so by the time we
// process this packet the event has "happened" — write a monotonic GPU clock (the value a timestamp
// event carries) and wake any waiter, resolving the "a guest waiting on the label blocks forever"
// case. Address-less events (event_addr == 0, the pipeline-sync variants: partial-flush, cache
// inval) stay no-ops. CONFIDENCE: LOW on the value for the counter-sample event types
// (ZPASS_DONE / streamout stats read a counter, not a timestamp) — but a defined monotonic write is
// strictly better than the old discard (no write at all, which is what blocked the waiter). No title
// currently exercises this (the Messenger fences via ReleaseMem/WriteData), so it's latent.
static void honor_event_write(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.event_addr || (c.event_addr & 3)) return;
    uint64_t v = gpu_clock64();
    memcpy((void*)(uintptr_t)c.event_addr, &v, sizeof v);
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   EventWrite [0x%llx] event_type=%u -> clock 0x%llx\n",
                (unsigned long long)c.event_addr, c.event_type, (unsigned long long)v);
    wake_on_label(c.event_addr);   // wake any sync_on_address futex waiter on this completion label
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
                fprintf(stderr, "[regindir] class=%s num=%u vaddr=0x%llx pairs:", cn, c.num_regs,
                        (unsigned long long)c.regs_vaddr);
                for (uint32_t i = 0; i < c.num_regs && i < 40; i++)
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
            if (getenv("PROSPER_RESDUMP")) {
                fprintf(stderr, "[shdirect] off=0x%x count=%u vals:", c.sh_reg_offset,
                        c.sh_reg_data ? c.sh_reg_count : 1u);
                if (c.sh_reg_data && c.sh_reg_count)
                    for (uint32_t k = 0; k < c.sh_reg_count && k < 40; k++)
                        fprintf(stderr, " 0x%x", c.sh_reg_data[k]);
                else fprintf(stderr, " 0x%x", c.sh_reg_value);
                fprintf(stderr, "\n");
            }
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
        case K::SetIndexBase:
            index_base = c.ib_addr;   // bind index-buffer base (issue #232)
            break;
        case K::SetIndexCount:
            index_num = c.index_count;   // bind index count (issue #232)
            break;
        case K::DrawIndexOffset: {
            // Gen5 indexed draw (issue #232). Uses the bound index base + count; DrawIndexOffset's own
            // count (c.index_count) overrides the SetIndexCount state when non-zero. The element size is
            // the current SetIndexType (0=16-bit, 1=32-bit), captured in the per-draw snapshot.
            if (getenv("PROSPER_RESDUMP")) {   // draw-vs-bind association diagnostic (#273)
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                fprintf(stderr, "[drawpkt] idx#%zu es=0x%08x count=%u dirty=%d ud=[%08x %08x %08x %08x | %08x %08x %08x %08x]\n",
                        draws.size(), rd(0xc8), c.index_count ? c.index_count : index_num, (int)state_dirty_,
                        rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f), rd(0x90), rd(0x91), rd(0x92), rd(0x93));
            }
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            uint32_t elem = index_type ? 4u : 2u;
            Draw d;
            d.index_count = c.index_count ? c.index_count : index_num;
            d.state = last_snapshot_;
            if (index_base && d.index_count) {
                d.indexed = true;
                d.index_addr = index_base + (uint64_t)c.index_offset * elem;
                // Preserve the raw base + element offset so the executor can recompute the address if
                // it auto-detects a different element size (#304 — DOLL's 32-bit Slate index buffers).
                d.index_base = index_base;
                d.index_offset = c.index_offset;
                d.from_offset = true;
            }
            draws.push_back(std::move(d));
            break;
        }
        case K::DrawIndexAuto:
        case K::DrawIndex: {
            // Snapshot the register state AT THE DRAW (shared with consecutive draws until a register
            // write dirties it), so a future per-draw executor can render each draw under its own
            // shaders/mask/blend instead of the end-of-submit fold. Inert for the current renderer.
            // The snapshot also carries index_type — the index element size a DrawIndex needs (#64).
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            Draw d;
            d.index_count = c.index_count;
            d.state = last_snapshot_;
            if (c.kind == K::DrawIndex) {
                // Mark as indexed only when the packet was fully decoded — a short packet's addr/
                // modifier would be fabricated zeros, and `indexed` promises index_addr is real.
                d.indexed = c.di_valid;
                d.index_addr = c.di_index_addr;
                d.modifier = c.di_modifier;
            }
            draws.push_back(std::move(d));
            break;
        }
        case K::ReleaseMem:
            honor_eop_write(c);     // EOP completion label write (correct synchronous end-of-pipe timing)
            break;
        case K::WriteData:
            honor_write_data(c);    // inline data write requested by the Dcb
            break;
        case K::EventWrite:
            honor_event_write(c);   // address-carrying (timestamp/label) EVENT_WRITE completion (#132)
            break;
        case K::WaitRegMem:
            // Synchronous fold: by the time we process this packet the CPU-side producer has
            // usually already written the label, so the wait is normally satisfied. We can't
            // BLOCK here (single-threaded fold) — but a wait that is NOT satisfied means the
            // dependency would have been violated (packets after this one read data the guest
            // hadn't produced at submit), so surface it loudly instead of silently proceeding.
            if (c.wm_valid && c.wm_addr && !(c.wm_addr & 3)) {
                uint64_t mem = 0; memcpy(&mem, (const void*)(uintptr_t)c.wm_addr, sizeof mem);
                uint64_t v = mem & c.wm_mask, r = c.wm_ref;
                bool sat;
                switch (c.wm_func) {           // PM4 WAIT_REG_MEM compare functions
                    case 0: sat = true;    break;
                    case 1: sat = v <  r;  break;
                    case 2: sat = v <= r;  break;
                    case 3: sat = v == r;  break;
                    case 4: sat = v != r;  break;
                    case 5: sat = v >= r;  break;
                    case 6: sat = v >  r;  break;
                    default: sat = false;  break;
                }
                if (!sat)
                    fprintf(stderr, "[agc] WaitRegMem NOT satisfied at fold time: [0x%llx]&0x%llx = 0x%llx, func=%u ref=0x%llx — dependency violated\n",
                            (unsigned long long)c.wm_addr, (unsigned long long)c.wm_mask,
                            (unsigned long long)v, c.wm_func, (unsigned long long)r);
            }
            break;
        case K::DispatchDirect:
            // Compute dispatch (issue #213 — DOLL's UE4 compute prologue). Recorded for stats/
            // diagnostics only: prosper has no compute execution path yet. The fence cluster the
            // guest builds around each dispatch (label init + EOP write + wait) completes at fold
            // time independently of the dispatch itself, so skipping the shader work cannot hang
            // the stream — it only leaves compute-written buffers stale (surfaced by the counter).
            dispatch_count++;
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
