#include "gpu/recompiler/rdna2_saved_mask_phi.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {
namespace {
uint32_t identity_phi(const SpirvCompute& b, uint32_t value) {
    if (!value || b.code.size() > 65536) return 0;
    uint32_t block = 0, header = 0, anchor = 0, left = 0, right = 0;
    bool found = false;
    for (size_t at = 0; at < b.code.size();) {
        const uint32_t width = b.code[at] >> 16, op = b.code[at] & 65535u;
        if (!width || width > b.code.size() - at) return 0;
        if (op == Op_Label) {
            if (width != 2) return 0;
            block = b.code[at + 1];
        }
        if (op == Op_Phi && width >= 3 && b.code[at + 2] == value) {
            if (found || width != 7 || b.code[at + 1] != b.t_bool || !block) return 0;
            found = true;
            header = block;
            left = b.code[at + 4];
            right = b.code[at + 6];
            const uint32_t a = b.code[at + 3], c = b.code[at + 5];
            if (a == c && a != value)
                anchor = a;
            else if (a == value && c != value)
                anchor = c;
            else if (c == value && a != value)
                anchor = a;
            else
                return 0;
        }
        at += width;
    }
    if (!found || !anchor || !left || !right || left == right || header == left || header == right)
        return 0;
    bool a = false, c = false;
    block = 0;
    for (size_t at = 0; at < b.code.size();) {
        const uint32_t width = b.code[at] >> 16, op = b.code[at] & 65535u;
        if (op == Op_Label) block = b.code[at + 1];
        if (op == Op_Branch && width == 2 && b.code[at + 1] == header) {
            a = a || block == left;
            c = c || block == right;
        }
        at += width;
    }
    return a && c ? anchor : 0;
}
} // namespace

uint32_t saved_mask_phi_identity(const SpirvCompute& b, uint32_t value) {
    if (value == b.bconst_true || value == b.bconst_false) return 0;
    return identity_phi(b, value);
}

void record_saved_mask_phi_subset(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    if (!b.is_compute || b.is_fragment || b.is_vertex || b.ngg_one_lane || b.ngg_workgroup_shell ||
        b.wave_size != 64 || b.native_subgroup_size != 64 || rs.vector_wave_size != 64 ||
        !rs.exec || rs.exec_subset_of.contains(rs.exec) || in.fmt != Rdna2Format::SOP1 ||
        in.opcode != kSop1OpcodeMovB64 || in.dst.value != 126 ||
        in.src[0].kind != OperandKind::SGPR || in.src[0].value < 0 || in.src[0].value > 104 ||
        in.synthetic_terminator || in.pc >= b.saved_mask_original_words.size())
        return;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    if (original != in) return;
    const auto mask = rs.sreg_bool.find(in.src[0].value);
    if (mask == rs.sreg_bool.end() || mask->second != rs.exec ||
        rs.sreg_bool_b32.contains(in.src[0].value))
        return;
    // Both closed incoming values name the same immutable Bool, including a loop's self edge.
    // Retain only the new ancestry; the current move and every DATA/lifetime mark stay unchanged.
    if (uint32_t anchor = saved_mask_phi_identity(b, rs.exec))
        rs.exec_subset_of.try_emplace(rs.exec, anchor);
}
}   // namespace prosper::gpu
