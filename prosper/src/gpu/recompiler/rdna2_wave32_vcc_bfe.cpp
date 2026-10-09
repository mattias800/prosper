// rdna2_wave32_vcc_bfe.cpp -- see rdna2_wave32_vcc_bfe.hpp.
#include "gpu/recompiler/rdna2_wave32_vcc_bfe.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <set>

namespace prosper::gpu {

std::unordered_set<uint32_t> proven_wave32_local_vcc_bfe_pcs(const std::vector<Rdna2Inst>& ins) {
    const auto branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };
    std::unordered_set<uint32_t> targets;
    for (const Rdna2Inst& in : ins)
        if (branch(in)) targets.insert(scalar_branch_target(in));
    std::unordered_set<uint32_t> proven;
    std::set<int> written;
    for (const Rdna2Inst& in : ins) {
        if (targets.contains(in.pc)) written.clear();
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == kSop2OpcodeBfeU64 &&
            (in.dst.value == 106 || in.dst.value == 107)) {
            const auto local = [&](const Operand& op, uint32_t words) {
                if (op.kind == OperandKind::InlineInt || op.kind == OperandKind::Literal)
                    return true;
                if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
                for (uint32_t w = 0; w < words; ++w)
                    if (!written.contains(op.value + static_cast<int>(w))) return false;
                return true;
            };
            if (local(in.src[0], 2u) && local(in.src[1], 1u)) proven.insert(in.pc);
        }
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t w = 0; w < width; ++w) written.insert(base + static_cast<int>(w));
        });
        const bool ends_block =
            branch(in) || rdna2_escapes_decoded_effects(in) ||
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x01 || in.opcode == kSoppOpcodeBarrier));
        if (ends_block) written.clear();
    }
    return proven;
}

}   // namespace prosper::gpu
