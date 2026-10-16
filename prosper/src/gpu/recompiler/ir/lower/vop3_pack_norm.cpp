// Existing signed/unsigned normalized packing, mechanically moved with extract_case_routes.py.
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3PackNormLowering::lower() {
    if (in.opcode != 0x368 && in.opcode != 0x369) return false;
    if (!fv) {
        std::fprintf(stderr, "[ir] normalized-pack source callback missing\n");
        std::abort();
    }
    // Clamp and normalize two f32 values, then pack src0 in bits[15:0] and src1 in
    // bits[31:16]. Astro's title ship VS uses the unsigned form (exact first word
    // d7690002). AMD specifies round-to-nearest-even for the normalized conversion.
    const bool is_signed = in.opcode == 0x368;
    const float scale = is_signed ? 32767.0f : 65535.0f;
    const uint32_t lo = b.pack_norm(fv(0), 16, is_signed, scale);
    const uint32_t hi = b.pack_norm(fv(1), 16, is_signed, scale);
    vreg[in.dst.value] = b.ibin(Op_BitwiseOr, lo, b.ibin(Op_ShiftLeftLogical, hi, b.uconst(16)));
    return true;
}
bool lower_vop3_pack_norm(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3PackNormSource& fv) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3PackNormLowering({b, rs, in, ok, allow_wave}, fv).lower();
}
}   // namespace prosper::gpu::ir
