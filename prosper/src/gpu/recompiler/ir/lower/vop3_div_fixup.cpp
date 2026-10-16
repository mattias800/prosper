// Existing division-fixup call mechanically moved; the original value helper is unchanged.
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_div_fixup.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3DivFixupLowering::lower() {
    if (in.opcode != 0x15f) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] fixup callbacks missing\n");
        std::abort();
    }
    vreg[in.dst.value] = fresult(emit_v_div_fixup_f32(b, fv(0), fv(1), fv(2)));
    return true;
}
bool lower_vop3_div_fixup_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                 bool allow_wave, const DivFixupSource& fv,
                                 const DivFixupResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3DivFixupLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
