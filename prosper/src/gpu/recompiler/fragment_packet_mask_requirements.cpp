#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_waitcnt.hpp"
#include "gpu/recompiler/fragment_packet_quad_swizzle.hpp"
#include <algorithm>
#include <map>

namespace prosper::gpu {
namespace {
// Physical halves remain separate until a COMPLETE pair has been defined on every predecessor.
constexpr uint8_t exec_lo = 1, exec_hi = 2, vcc_lo = 4, vcc_hi = 8, scc = 16;
constexpr uint8_t exec = exec_lo | exec_hi, vcc = vcc_lo | vcc_hi;
uint8_t word(int reg) {
    switch (reg) {
        case 126: return exec_lo;
        case 127: return exec_hi;
        case 106: return vcc_lo;
        case 107: return vcc_hi;
        case 253: return scc;
        default: return 0;
    }
}
uint8_t complete(uint8_t words) {
    return ((words & exec) == exec ? kPacketInitialExec : 0) |
           ((words & vcc) == vcc ? kPacketInitialVcc : 0) | ((words & scc) ? kPacketInitialScc : 0);
}
bool saveexec(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeAndSaveexecB64;
}
bool inventoried(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            return in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64 ||
                   in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64 ||
                   in.opcode == 0x08 || in.opcode == 0x0a || in.opcode == kSop1OpcodeBcnt1I32B64 ||
                   in.opcode == kSop1OpcodeFf1I32B64 || in.opcode == kSop1OpcodeAndSaveexecB64;
        case Rdna2Format::SOP2:
            return in.opcode == kSop2OpcodeAddU32 || in.opcode == kSop2OpcodeAddcU32 ||
                   in.opcode == kSop2OpcodeCselectB32 || in.opcode == kSop2OpcodeCselectB64 ||
                   in.opcode == kSop2OpcodeAndB32 || in.opcode == kSop2OpcodeAndB64 ||
                   in.opcode == kSop2OpcodeOrB32 || in.opcode == kSop2OpcodeOrB64;
        case Rdna2Format::SOPK:
            return in.opcode == kSopkOpcodeMovkI32 || in.opcode == kSopkOpcodeCmovkI32;
        case Rdna2Format::SOPC: return in.opcode <= 0x13;
        case Rdna2Format::SOPP: return true; // control flow checked separately
        case Rdna2Format::VOP1:
            return in.opcode == 1 || in.opcode == 0x2a || in.opcode == 0x2e || in.opcode == 0x33;
        case Rdna2Format::VOP2: return in.opcode == 3 || in.opcode == 8;
        case Rdna2Format::VOP3:
            return in.opcode == 0x360 || in.opcode == 0x365 || in.opcode == 0x366;
        case Rdna2Format::VOPC:
            return (in.opcode >= 0xc0 && in.opcode <= 0xc7) ||
                   (in.opcode >= 0xd0 && in.opcode <= 0xd7);
        case Rdna2Format::VINTRP: return in.opcode <= 1;
        case Rdna2Format::SMEM:
            return in.opcode >= kSmemOpcodeBufferLoadDword &&
                   in.opcode <= kSmemOpcodeBufferLoadDwordX4;
        case Rdna2Format::MIMG: return in.opcode == 0x24 || in.opcode == 0x27;
        case Rdna2Format::EXP: return true;
        case Rdna2Format::DS: return !packet_quad_swizzle_gap(in);
        default: return false;
    }
}
} // namespace
bool FragmentPacketMaskRequirements::defined_before(uint32_t pc, uint8_t state) const {
    const auto it =
        std::lower_bound(entries.begin(), entries.end(), pc,
                         [](const Entry& entry, uint32_t value) { return entry.pc < value; });
    return it != entries.end() && it->pc == pc && (it->defined & state) == state;
}
uint64_t FragmentPacketMaskRequirements::retained_bytes() const {
    return entries.capacity() * sizeof(Entry) + rejection.capacity() + 1;
}
uint8_t fragment_packet_initial_mask_availability(const FragmentInvocationPacket& input) {
    if (input.mask_state_available)
        return kPacketInitialExec | kPacketInitialVcc | kPacketInitialScc;
    return (input.exec_available ? kPacketInitialExec : 0) |
           (input.vcc_available ? kPacketInitialVcc : 0) |
           (input.scc_available ? kPacketInitialScc : 0);
}
const char* fragment_packet_missing_initial_mask(const FragmentPacketMaskRequirements& facts,
                                                 uint8_t available, uint32_t& pc) {
    constexpr uint8_t bits[]{kPacketInitialExec, kPacketInitialVcc, kPacketInitialScc};
    constexpr const char* names[]{"packet-entry-exec-unavailable", "packet-entry-vcc-unavailable",
                                  "packet-entry-scc-unavailable"};
    pc = UINT32_MAX;
    const char* reason = nullptr;
    for (uint32_t state = 0; state < 3; ++state)
        if ((facts.demanded & bits[state]) && !(available & bits[state]) &&
            (!reason || facts.first_read_pc[state] < pc)) {
            pc = facts.first_read_pc[state];
            reason = names[state];
        }
    return reason;
}
FragmentPacketMaskRequirements
fragment_packet_mask_requirements(const std::vector<uint32_t>& code,
                                  const std::vector<Rdna2Inst>& ins) {
    FragmentPacketMaskRequirements result;
    result.source_words = &code;
    if (code.empty() || code.size() > 4096 || ins.empty() || !ins.back().is_end ||
        ins.back().pc + ins.back().len_dwords != code.size()) {
        result.rejection = "packet-mask-program-inventory-incomplete";
        return result;
    }
    std::map<uint32_t, size_t> indices;
    for (size_t i = 0; i < ins.size(); ++i) indices.emplace(ins[i].pc, i);
    std::vector<uint8_t> defined(ins.size());
    std::vector<bool> reached(ins.size());
    reached.front() = true;
    const auto meet = [&](size_t target, uint8_t value) {
        if (!reached[target]) {
            reached[target] = true;
            defined[target] = value;
        } else
            defined[target] &= value;
    };
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        // The bounded packet dispatcher admits only direct forward control flow. Unknown jumps
        // cannot manufacture a MUST fact or certify that a dead-looking entry word is unused.
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if ((branch && (branch_target(in) <= in.pc || !indices.contains(branch_target(in)))) ||
            (in.fmt == Rdna2Format::SOPP && !branch && !in.is_end && in.opcode != 0 &&
             !(in.opcode == 0x0c && rdna2_waitcnt_effects_known(uint16_t(in.simm16)))) ||
            (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20 && in.opcode <= 0x22) ||
            (in.fmt == Rdna2Format::SOPK &&
             (in.opcode == kSopkOpcodeCallB64 || in.opcode == kSopkOpcodeSubvectorLoopBegin ||
              in.opcode == kSopkOpcodeSubvectorLoopEnd))) {
            result.rejection = "packet-mask-control-flow-unimplemented";
            return result;
        }
        if (!reached[i]) continue;
        if (!inventoried(in)) {
            result.rejection = "packet-mask-instruction-effects-unimplemented";
            return result;
        }
        result.entries.push_back({in.pc, complete(defined[i])});
        uint8_t reads = 0, writes = 0;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
            const auto& operand = in.src[source];
            if (operand.kind != OperandKind::Special && operand.kind != OperandKind::SGPR) continue;
            if (operand.kind == OperandKind::Special &&
                (operand.value == 251 || operand.value == 252)) {
                reads |= operand.value == 251 ? vcc : exec;
                continue;
            }
            const uint32_t width = in.fmt == Rdna2Format::SMEM ? (source == 0 ? 4u : 1u)
                                   : in.fmt == Rdna2Format::MIMG
                                       ? (source == 1   ? 8u
                                          : source == 2 ? 4u
                                                        : 1u)
                                       : scalar_alu_source_words(in, source);
            if (!width || width == UINT32_MAX || width > 8) {
                result.rejection = "packet-mask-source-form-unimplemented";
                return result;
            }
            for (uint32_t offset = 0; offset < width; ++offset)
                reads |= word(operand.value + static_cast<int>(offset));
        }
        const uint32_t old_destination = scalar_implicit_destination_read_width(in);
        for (uint32_t offset = 0; offset < old_destination; ++offset)
            reads |= word(in.dst.value + static_cast<int>(offset));
        const bool readlane = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360;
        const bool vector = in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 ||
                            in.fmt == Rdna2Format::VOP3 || in.fmt == Rdna2Format::VOPC ||
                            in.fmt == Rdna2Format::VINTRP || in.fmt == Rdna2Format::MIMG ||
                            in.fmt == Rdna2Format::EXP || in.fmt == Rdna2Format::DS;
        if (vector && !readlane) reads |= exec;
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 4 && in.opcode <= 9)
            reads |= in.opcode <= 5 ? scc : in.opcode <= 7 ? vcc : exec;
        if ((in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == kSop2OpcodeCselectB32 || in.opcode == kSop2OpcodeCselectB64 ||
              in.opcode == kSop2OpcodeAddcU32)) ||
            (in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32))
            reads |= scc;
        if (saveexec(in)) reads |= exec;
        // RDNA2 ISA 70648 sections3.3/3.9: scalar complete writes ignore EXEC; regular V_CMP
        // writes VCC[n]=EXEC[n]&test for EVERY lane. CMPX writes EXEC, never defines VCC, and
        // necessarily reads OLD EXEC. SCC-transparent MOV/FF1 never become SCC definitions.
        const bool conditional_move =
            (in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32);
        if (!conditional_move)
            for_each_scalar_write(in, [&](int reg, uint32_t width) {
                for (uint32_t offset = 0; offset < width; ++offset)
                    writes |= word(reg + static_cast<int>(offset));
            });
        if (in.fmt == Rdna2Format::VOPC) {
            if (vopc_is_cmpx(in.opcode))
                writes |= exec;
            else if (!(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105))
                writes |= vcc;
        }
        if (saveexec(in)) writes |= exec | scc;
        if ((in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == 0x08 || in.opcode == 0x0a || in.opcode == kSop1OpcodeBcnt1I32B64)) ||
            (in.fmt == Rdna2Format::SOP2 && in.opcode != kSop2OpcodeCselectB32 &&
             in.opcode != kSop2OpcodeCselectB64) ||
            in.fmt == Rdna2Format::SOPC)
            writes |= scc;
        const uint8_t missing = reads & ~defined[i];
        constexpr uint8_t words[]{exec, vcc, scc};
        constexpr uint8_t bits[]{kPacketInitialExec, kPacketInitialVcc, kPacketInitialScc};
        for (uint32_t state = 0; state < 3; ++state)
            if (missing & words[state]) {
                result.demanded |= bits[state];
                result.first_read_pc[state] = std::min(result.first_read_pc[state], in.pc);
            }
        const uint8_t after = defined[i] | writes;
        if (in.is_end) continue;
        if (branch) meet(indices.at(branch_target(in)), after);
        if ((!branch || in.opcode != 2) && i + 1 < ins.size()) meet(i + 1, after);
    }
    return result;
}
}   // namespace prosper::gpu
