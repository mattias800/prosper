// Existing RTZ pack call moved verbatim; its value helper already delegates to shared IR.
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_rtz_pack_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3RtzPackLowering::lower() {
    if (in.opcode != 0x12f) return false;
    if (!fv) {
        std::fprintf(stderr, "[ir] RTZ source callback missing\n");
        std::abort();
    }
    vreg[in.dst.value] = b.pack_half2x16_rtz(fv(0), fv(1));   // v_cvt_pkrtz VOP3: RTZ clamp (#452)
    return true;
}
bool lower_vop3_rtz_pack_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                bool allow_wave, const RtzPackSource& fv) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3RtzPackLowering({b, rs, in, ok, allow_wave}, fv).lower();
}
}   // namespace prosper::gpu::ir
