// Existing LDEXP guard and body mechanically moved with extract_case_routes.py.
#include "gpu/recompiler/ir/lower/vop3_ldexp_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_ldexp_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3LdexpLowering::lower() {
    if (in.opcode != 0x362) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] float callbacks missing\n");
        std::abort();
    }
    // AMD opcode 866: D.f = S0.f * 2**S1.i. The exact GTA V production packet is
    // d7620000,0002030d (`v_ldexp_f32 v0,v13,v1`) with no modifiers. Admit that proven
    // shape only for now: ABS/NEG on the integer exponent and output-modifier denormal
    // behavior need separate contracts, and silently ignoring either would corrupt
    // mip/scale reconstruction. ldexp_f32_bits covers the full unmodified i32 exponent
    // domain without relying on GLSL.std.450 Ldexp's undefined overflow cases.
    // src0 is the FLOAT operand, and ABS/NEG on a float source is the ordinary VOP3
    // modifier every other float op here already applies through fv() -- hardware order
    // ABS then NEG. Admitting it is not a new contract (#3138). The original gate
    // rejected all three sources together, which also refused this well-defined case:
    // Stray's `0x3011560000` is `v_ldexp_f32 v39, |v35|, -2` (`d7620127,00018523`,
    // ABS[2:0]=001), and that ONE instruction failed a 3684-dword fragment stage and
    // discarded 1536 full-screen 3840x2160 draws per routed boot.
    //
    // src1 keeps rejecting, and for a reason the gate's own comment gives: it is the
    // integer EXPONENT, where "absolute value" and "negate" are not float modifiers at
    // all and silently ignoring either would corrupt mip/scale reconstruction. CLAMP is
    // the ordinary float saturate of the exact result via fresult() (Kena #4706,
    // `d762800c,0001850c`; CONFIDENCE: HIGH). OMOD keeps rejecting: hardware ignores it
    // on f32 with denormals enabled, so it needs the MODE contract.
    if (in.src_abs[1] || in.src_abs[2] || in.src_neg[1] || in.src_neg[2] || in.omod) {
        ok = false;
    } else {
        vreg[in.dst.value] = fresult(b.ldexp_f32_bits(fv(0), val(in.src[1])));
    }
    return true;
}
bool lower_vop3_ldexp(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave,
                      const Vop3LdexpSource& fv, const Vop3LdexpResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3LdexpLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
