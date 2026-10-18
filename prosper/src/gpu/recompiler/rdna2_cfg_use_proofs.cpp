// Whole-CFG use queries mechanically split from rdna2_emit_cfg.cpp.
#include "gpu/recompiler/rdna2_cfg_use_proofs.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <set>
#include <vector>

namespace prosper::gpu {

// Complex CFG dispatch and the narrow loop structurizers persist B64 mask values, but not the
// separate one-word-validity state required by Wave32 aliases. Conservatively find any B32 mask
// copy that the region could create. The source set is deliberately path-insensitive: a pair made a
// mask on any path may reach a later copy on another dispatcher edge, and rejecting an impossible
// ordering is safer than silently restoring a stale Boolean.
bool has_unpersisted_b32_mask_lifetime(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                                       const RegState& entry) {
    std::set<int> possible_mask_sources{106, 126};   // VCC_LO and EXEC_LO
    for (const auto& kv : entry.sreg_bool) possible_mask_sources.insert(kv.first);
    for (int reg : entry.sreg_bool_b32) possible_mask_sources.insert(reg);
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            if (scalar_write_is_b64_mask(in, base)) possible_mask_sources.insert(base);
        });
    }
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi || in.fmt != Rdna2Format::SOP1) continue;
        if (in.opcode == 0x09 &&
            !(wqm_has_numeric_destination(in) && wqm_has_intrinsic_numeric_source(in)))
            return true;   // only mask-domain s_wqm_b32 requires the unpersisted width state
        if (in.opcode != 0x03 && in.opcode != 0x07) continue;
        const bool register_source =
            in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::Special;
        if ((register_source && possible_mask_sources.contains(in.src[0].value)) ||
            in.dst.value == 126)
            return true;
    }
    return false;
}

// Does [lo, hi) contain a `s_mov_b32 sX, m0` -- the instruction that starts an entry-M0 token
// lifetime (#3133)? A STATIC property of the decoded stream, so it can be asked before any block is
// emitted, which is what the loop emitters need: their header phis are built before the body runs,
// and a loop-carried token has no value to seed one with. Returns the destination register (>= 0)
// of the first such save, or -1.
int entry_m0_save_in_range(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x03 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 124 && in.dst.value <= 105)
            return in.dst.value;
    }
    return -1;
}

}   // namespace prosper::gpu
