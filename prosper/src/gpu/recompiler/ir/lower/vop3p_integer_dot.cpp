// Mechanically relocated integer-dot route; no arithmetic or admission change.
#include "gpu/recompiler/ir/lower/vop3p_integer_dot_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3p_integer_dot_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_dot.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
std::optional<bool> Vop3pIntegerDotLowering::route_dot() {
    if (in.opcode >= 0x14u && in.opcode <= 0x19u) {
        const uint32_t old_d = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = emit_v_dot(b, rs, in, ok);
        if (ok) predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

bool Vop3pIntegerDotLowering::lower() {
    if (auto routed = route_dot()) return *routed;
    return false;
}
bool lower_vop3p_integer_dot(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3P) return false;
    return Vop3pIntegerDotLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
