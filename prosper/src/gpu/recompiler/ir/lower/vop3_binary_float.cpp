// Six already admitted binary float cases mechanically moved; observation calls stay in order.
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_binary_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/fragment_arithmetic_observation.hpp"
#include <cstdio>
#include <cstdlib>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3BinaryFloatLowering::lower() {
    const auto* def = find_vector_binary_float_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] binary float callbacks missing\n");
        std::abort();
    }
 // Existing observation calls retain their position before any float source is read.
    if (in.opcode == 0x103)
        b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Add);
    else if (in.opcode == 0x108)
        b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Mul);
    vreg[in.dst.value] = fresult(emit_ternary_integer_op(
        b, *def, [&](uint32_t source) { return fv(static_cast<int>(source)); }));
    return true;
}
bool lower_vop3_binary_float_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                    bool allow_wave, const BinaryFloatSource& fv,
                                    const BinaryFloatResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3BinaryFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
