// command_processor.cpp — see command_processor.hpp.
#include "command_processor.hpp"
#include <cstdlib>
#include <cstdint>
#include <cstdio>

namespace prosper::gpu {

// EOP fence writeback (bring-up experiment). RELEASE_MEM asks the GPU to write a completion value to a
// label address when the pipe drains; a CPU thread then polls that label to know the submit finished.
// Because our CommandProcessor folds each Dcb synchronously, the "GPU" is done at submit time, so writing
// the label here is correct end-of-pipe semantics. The DESTINATION address is HIGH-confidence (WaitRegMem
// polls the same address); the VALUE is LOW-confidence (unsure which captured arg / width the game's poll
// wants), so PROSPER_AGC_FENCE selects the variant without a rebuild while we find what unblocks the game:
//   1 = write a7 (32-bit)   2 = write a6 (32-bit)   3 = write 0xFFFFFFFF (satisfy >=/!= compares)
//   4 = write 64-bit a7:a6  5 = write 64-bit 0xFFFF..FF   (unset/0 = do nothing)
// CONFIDENCE: LOW — value mapping is a hypothesis; see docs/RENDER_LOOP.md fence-handshake section.
static void maybe_fence_write(const Pm4Command& c) {
    const char* mode = getenv("PROSPER_AGC_FENCE");
    if (!mode || mode[0] == '0' || !c.rel_addr || (c.rel_addr & 3)) return;
    auto* p32 = (volatile uint32_t*)(uintptr_t)c.rel_addr;
    auto* p64 = (volatile uint64_t*)(uintptr_t)c.rel_addr;
    switch (mode[0]) {
        case '1': *p32 = c.rel_v7; break;
        case '2': *p32 = c.rel_v6; break;
        case '3': *p32 = 0xFFFFFFFFu; break;
        case '4': *p64 = (uint64_t)c.rel_v6 | ((uint64_t)c.rel_v7 << 32); break;
        case '5': *p64 = ~0ull; break;
        default: break;
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   fence-write [0x%llx] mode=%c (a6=0x%x a7=0x%x)\n",
                (unsigned long long)c.rel_addr, mode[0], c.rel_v6, c.rel_v7);
}

void GpuState::apply(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::SetRegsIndirect: {
            if (c.regs_vaddr == 0 || c.num_regs == 0 || c.num_regs > kMaxRegsPerPacket) return;
            auto* regs = reinterpret_cast<const ShaderReg*>(static_cast<uintptr_t>(c.regs_vaddr));
            auto& file = (c.reg_class == RegClass::Cx) ? cx
                       : (c.reg_class == RegClass::Sh) ? sh : uc;
            for (uint32_t i = 0; i < c.num_regs; i++) file[regs[i].offset] = regs[i].value;
            break;
        }
        case K::SetShRegDirect:
            sh[c.sh_reg_offset] = c.sh_reg_value;
            break;
        case K::SetIndexType:
            index_type = c.index_size;
            break;
        case K::DrawIndexAuto:
            draws.push_back({ c.index_count });
            break;
        case K::ReleaseMem:
            maybe_fence_write(c);   // EOP completion label write (bring-up, PROSPER_AGC_FENCE-gated)
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
