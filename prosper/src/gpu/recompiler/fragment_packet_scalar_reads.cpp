#include "gpu/recompiler/fragment_packet_scalar_reads.hpp"
#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_waitcnt.hpp"
#include <algorithm>

namespace prosper::gpu {
namespace {
// Known scalar effects only. An unknown writer cannot leave an apparent entry-origin alive.
// This is NOT a shader admission list: the packet compiler and resource/launch gates remain
// responsible for execution. Known non-MOV writes merely kill the affected word's origin.
bool known_effects(const Rdna2Inst& in) {
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
        case Rdna2Format::SOPP:
            return in.opcode == 0 || (in.opcode == 1 && in.is_end) ||
                   (in.opcode == 0x0c && rdna2_waitcnt_effects_known(uint16_t(in.simm16)));
        case Rdna2Format::VOP1:
            return in.opcode == 1 || in.opcode == 2 || in.opcode == 0x2a || in.opcode == 0x2e ||
                   in.opcode == 0x33;
        case Rdna2Format::VOP2: return in.opcode == 3 || in.opcode == 8;
        case Rdna2Format::VOP3:
            return in.opcode == 0x360 || in.opcode == 0x365 || in.opcode == 0x366;
        case Rdna2Format::VOPC:
            return (in.opcode >= 0xc0 && in.opcode <= 0xc7) ||
                   (in.opcode >= 0xd0 && in.opcode <= 0xd7);
        case Rdna2Format::VINTRP: return in.opcode <= 1;
        case Rdna2Format::MIMG: return in.opcode == 0x24 || in.opcode == 0x27;
        case Rdna2Format::EXP: return true;
        default: return false;
    }
}
bool modifiers(const Rdna2Inst& in) {
    return in.has_modifier || in.has_sdwa || in.has_dpp || in.clamp || in.omod ||
           std::any_of(std::begin(in.src_abs), std::end(in.src_abs), [](bool v) { return v; }) ||
           std::any_of(std::begin(in.src_neg), std::end(in.src_neg), [](bool v) { return v; });
}
} // namespace
uint64_t FragmentPacketScalarReadRequirements::retained_bytes() const {
    return sites.capacity() * sizeof(FragmentPacketScalarReadSite) + rejection.capacity() + 1;
}
FragmentPacketScalarReadRequirements
fragment_packet_scalar_read_requirements(const std::vector<uint32_t>& code,
                                         const std::vector<Rdna2Inst>& instructions) {
    FragmentPacketScalarReadRequirements result;
    result.source_words = &code;
    result.has_smem = std::any_of(instructions.begin(), instructions.end(),
                                  [](const auto& in) { return in.fmt == Rdna2Format::SMEM; });
    // No memory demand needs descriptor-origin analysis. Other packet inventories still diagnose
    // unknown/nonterminating programs; this facility cannot grant admission by returning empty.
    if (!result.has_smem) return result;
    const auto refuse = [&](const char* reason, uint32_t pc) {
        result.rejection = std::string(reason) + ":pc=" + std::to_string(pc);
        result.sites.clear();   // never publish a partially certified original-program manifest
        return result;
    };
    if (code.empty() || code.size() > 4096 || instructions.empty())
        return refuse("packet-scalar-program-inventory-incomplete", UINT32_MAX);
    std::array<uint32_t, 106> origins;
    for (uint32_t reg = 0; reg < origins.size(); ++reg) origins[reg] = reg;
    uint32_t next_pc = 0;
    bool ended = false;
    for (const auto& in : instructions) {
        if (ended || in.synthetic_terminator || in.pc != next_pc || !in.len_dwords ||
            in.len_dwords > code.size() - next_pc)
            return refuse("packet-scalar-program-inventory-incomplete", in.pc);
        next_pc += in.len_dwords;
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode))
            return refuse("packet-scalar-control-flow-unimplemented", in.pc);
        std::array<uint32_t, 2> moved{UINT32_MAX, UINT32_MAX};
        uint32_t move_words = 0;
        if (in.fmt == Rdna2Format::SMEM) {
            if (const auto* gap = packet_resource_instruction_gap(in)) return refuse(gap, in.pc);
            const uint32_t base = static_cast<uint32_t>(in.src[0].value);
            FragmentPacketScalarReadSite site{in.pc, in.opcode, 1u << (in.opcode - 8), in.literal};
            for (uint32_t word = 0; word < 4; ++word) {
                site.entry_words[word] = origins[base + word];
                if (site.entry_words[word] == UINT32_MAX)
                    return refuse("packet-scalar-descriptor-origin-unproved", in.pc);
            }
            result.sites.push_back(site);
        } else {
            if (modifiers(in) || !known_effects(in))
                return refuse("packet-scalar-instruction-effects-unimplemented", in.pc);
            if (in.fmt == Rdna2Format::SOP1 &&
                (in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64) &&
                in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0) {
                move_words = in.opcode == kSop1OpcodeMovB32 ? 1u : 2u;
                const uint32_t base = static_cast<uint32_t>(in.src[0].value);
                if (base <= origins.size() - move_words)
                    for (uint32_t word = 0; word < move_words; ++word)
                        moved[word] = origins[base + word];
            }
        }
        // Capture all MOV sources BEFORE killing destinations, including overlapping B64 moves.
        for_each_scalar_write(in, [&](int base, uint32_t words) {
            for (uint32_t word = 0; word < words; ++word) {
                const int reg = base + static_cast<int>(word);
                if (reg >= 0 && static_cast<size_t>(reg) < origins.size())
                    origins[reg] = UINT32_MAX;
            }
        });
        if (move_words && in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 &&
            static_cast<uint32_t>(in.dst.value) <= origins.size() - move_words)
            for (uint32_t word = 0; word < move_words; ++word)
                origins[in.dst.value + word] = moved[word];
        if (in.is_end) {
            if (next_pc != code.size())
                return refuse("packet-scalar-program-inventory-incomplete", in.pc);
            ended = true;
        }
    }
    if (!ended || next_pc != code.size())
        return refuse("packet-scalar-program-inventory-incomplete", next_pc);
    return result;
}
}   // namespace prosper::gpu
