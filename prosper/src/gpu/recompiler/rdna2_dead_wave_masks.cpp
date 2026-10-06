// See rdna2_dead_wave_masks.hpp. Moved unchanged out of rdna2_emit_cfg.cpp.
#include "gpu/recompiler/rdna2_dead_wave_masks.hpp"

#include <unordered_map>

namespace prosper::gpu {

std::unordered_set<uint32_t> dead_wave_mask_writes(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins) {
        const bool reads_scc =
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05)) ||
            (in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a || in.opcode == 0x0b)) ||
            (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x05 || in.opcode == 0x06)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (reads_scc) return {};
    }
    auto is_mask = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 &&
               (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 || in.opcode == 0x15 ||
                in.opcode == 0x17 || in.opcode == 0x19 || in.opcode == 0x1b || in.opcode == 0x1d) &&
               (in.dst.value == 106 || in.dst.value == 107);
    };
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) by_pc[ins[i].pc] = i;
    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
                            in.opcode != 0x03;
        if (!branch || in.opcode != 0x02) {
            if (i + 1 < ins.size()) succ[i].push_back(i + 1);
        }
        if (branch) {
            const uint32_t target =
                in.pc + in.len_dwords + static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
            auto it = by_pc.find(target);
            if (it == by_pc.end()) return {};   // malformed/unbounded CFG: make no dead-write claim
            succ[i].push_back(it->second);
        }
    }
    auto uses = [&](const Rdna2Inst& in) -> uint8_t {
        uint8_t bits = 0;
        for (uint8_t k = 0; k < in.n_src; ++k) {
            if (in.src[k].kind != OperandKind::SGPR && in.src[k].kind != OperandKind::Special)
                continue;
            if (in.src[k].value == 106) bits |= 1;
            if (in.src[k].value == 107) bits |= 2;
        }
        if (is_mask(in) && bits) bits = 3;   // every modeled mask logical reads a full B64 pair
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)))
            bits |= 3;
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07)) bits |= 3;
        return bits;
    };
    auto defs = [&](const Rdna2Inst& in) -> uint8_t {
        if (is_mask(in)) return 3;
        // A cmpx writes EXEC and has NO VCC destination, so it must NOT count as defining VCC —
        // the decoder gives every VOPC e32 dst = 106 (VCC_LO), so without this exclusion a cmpx
        // would satisfy both conjuncts and record a phantom definition. A private copy of the
        // windows here listed three of the six, so every v_cmpx_*_f64/_i64/_u64/_u16 was recorded
        // as defining VCC; a preceding live `s_and_b64 vcc` then looked overwritten before use,
        // was classified dead and ELIDED, leaving stale VCC at the real consumer with no
        // diagnostic. Kernel 32r13v pins it. Use the one shared predicate (#2120).
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            (in.dst.value == 106 || in.dst.value == 107))
            return 3;
        if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a) return 3;
        if (in.fmt == Rdna2Format::VOP3 && (in.sdst.value == 106 || in.sdst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::SOP1) {
            if (in.dst.value == 106) return in.opcode == 0x04 ? 3 : 1;
            if (in.dst.value == 107) return 2;
        }
        if (in.fmt == Rdna2Format::SMEM) {
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0:
                case 0x8: n = 1; break;
                case 0x1:
                case 0x9: n = 2; break;
                case 0x2:
                case 0xa: n = 4; break;
                case 0x3:
                case 0xb: n = 8; break;
                case 0x4:
                case 0xc: n = 16; break;
                default: break;
            }
            uint8_t bits = 0;
            if (n && in.dst.value <= 106 && 106 < in.dst.value + static_cast<int>(n)) bits |= 1;
            if (n && in.dst.value <= 107 && 107 < in.dst.value + static_cast<int>(n)) bits |= 2;
            return bits;
        }
        return 0;
    };
    // Least-fixed-point dataflow rooted only in observable (non-candidate) VCC reads. A mask
    // candidate propagates liveness to its B64 input only when its output is itself live; this also
    // removes dead self-dependent mask chains inside loops without mistaking the cycle for a use.
    std::vector<uint8_t> live_in(ins.size(), 0), live_out(ins.size(), 0);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t ri = ins.size(); ri-- > 0;) {
            uint8_t out = 0;
            for (size_t s : succ[ri]) out |= live_in[s];
            const uint8_t def = defs(ins[ri]), use = uses(ins[ri]);
            const uint8_t propagated_use = is_mask(ins[ri]) && !(out & def) ? 0 : use;
            const uint8_t in = propagated_use | (out & static_cast<uint8_t>(~def));
            if (out != live_out[ri] || in != live_in[ri]) {
                live_out[ri] = out;
                live_in[ri] = in;
                changed = true;
            }
        }
    }
    std::unordered_set<uint32_t> dead;
    for (size_t i = 0; i < ins.size(); ++i)
        if (is_mask(ins[i]) && !(live_out[i] & 3)) dead.insert(ins[i].pc);
    return dead;
}

}   // namespace prosper::gpu
