// rdna2_local_vcc_data.cpp -- see rdna2_local_vcc_data.hpp.
#include "gpu/recompiler/rdna2_local_vcc_data.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <set>

namespace prosper::gpu {

std::unordered_set<uint32_t> proven_local_vcc_scalar_write_pcs(const std::vector<Rdna2Inst>& ins) {
    // Every SOPP that can transfer control, the debugger-conditional branches included: their
    // taken edge joins the block they land in just like an ordinary branch's (#4819 review).
    const auto branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && (sopp_opcode_is_direct_branch(in.opcode) ||
                                               (in.opcode >= kSoppOpcodeCbranchCdbgsys &&
                                                in.opcode <= kSoppOpcodeCbranchCdbgsysAndUser));
    };
    std::unordered_set<uint32_t> targets;
    for (const Rdna2Inst& in : ins)
        if (branch(in)) targets.insert(scalar_branch_target(in));
    std::unordered_set<uint32_t> proven;
    std::set<int> written;
    for (const Rdna2Inst& in : ins) {
        if (targets.contains(in.pc)) written.clear();
        if (in.fmt == Rdna2Format::SOP2 && (in.dst.value == 106 || in.dst.value == 107)) {
            bool local = true;
            for (uint32_t k = 0; k < in.n_src && local; ++k) {
                const Operand& op = in.src[k];
                if (op.kind == OperandKind::InlineInt || op.kind == OperandKind::Literal) continue;
                if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) {
                    local = false;
                    break;
                }
                const uint32_t words = scalar_alu_source_words(in, k);
                if (words == UINT32_MAX || words > 2u) {
                    local = false;
                    break;
                }
                for (uint32_t w = 0; w < words; ++w)
                    local = local && written.contains(op.value + static_cast<int>(w));
            }
            if (local) proven.insert(in.pc);
        }
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t w = 0; w < width; ++w) written.insert(base + static_cast<int>(w));
        });
        const bool ends_block =
            branch(in) || rdna2_escapes_decoded_effects(in) ||
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x01 || in.opcode == kSoppOpcodeBarrier ||
                                             in.opcode == kSoppOpcodeTrap));
        if (ends_block) written.clear();
    }
    return proven;
}

}   // namespace prosper::gpu
