// command_processor.cpp — see command_processor.hpp.
#include "command_processor.hpp"
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace prosper::gpu {

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
        case 1: { uint32_t v = (uint32_t)c.rel_value; memcpy(dst, &v, sizeof v); break; }
        case 2: { uint64_t v = c.rel_value;           memcpy(dst, &v, sizeof v); break; }
        case 3: { uint64_t v = gpu_clock64();         memcpy(dst, &v, sizeof v); break; }
        default: return;   // None / unsupported: nothing to write
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   EOP write [0x%llx] data_sel=%u value=0x%llx\n",
                (unsigned long long)c.rel_addr, c.rel_data_sel, (unsigned long long)c.rel_value);
}

// Honor a WRITE_DATA packet: copy the inline dwords to the destination address (same synchronous timing).
static void honor_write_data(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.wd_addr || (c.wd_addr & 3) || !c.wd_data || !c.wd_num) return;
    memcpy((void*)(uintptr_t)c.wd_addr, c.wd_data, (size_t)c.wd_num * 4);
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   WriteData [0x%llx] %u dwords (first=0x%08x)\n",
                (unsigned long long)c.wd_addr, c.wd_num, c.wd_data[0]);
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
            break;
        case K::SetIndexType:
            index_type = c.index_size;
            break;
        case K::DrawIndexAuto:
            draws.push_back({ c.index_count });
            break;
        case K::ReleaseMem:
            honor_eop_write(c);     // EOP completion label write (correct synchronous end-of-pipe timing)
            break;
        case K::WriteData:
            honor_write_data(c);    // inline data write requested by the Dcb
            break;
        default:
            break;   // events / flips / unknown: no register-state effect (handled later)
    }
}

size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st) {
    std::vector<Pm4Command> ops;
    decode_pm4(buf, dwords, ops);
    for (const auto& c : ops) st.apply(c);
    return ops.size();
}

} // namespace prosper::gpu
