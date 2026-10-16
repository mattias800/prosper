// Mechanically extracted existing cubemap face and signed major-axis cases.
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3CubeAxisLowering::lower() {
    if (in.opcode != 0x144 && in.opcode != 0x147) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] cube callbacks missing\n");
        std::abort();
    }
    // Cubemap coordinate ops (#273 — DOLL's title post PSes' reflection-probe math):
    // v_cubeid_f32 (0x144) face id, v_cubesc_f32 (0x145) S numerator, v_cubetc_f32
    // (0x146) T numerator, v_cubema_f32 (0x147) 2*major-axis. Src order (x, y, z);
    // face-major selection per the AMD ISA / GL cubemap convention:
    //   |z|>=|x| && |z|>=|y| : id = z<0?5:4  sc = z<0?-x:x  tc = -y      ma = 2z
    //   else |y|>=|x|        : id = y<0?3:2  sc = x         tc = z<0.. = y<0?-z...
    //   (see per-op emission below)
    // VERIFIED(round-trip llvm-mc gfx1030: 0xd544-0xd547). Execution-tested (kernel in
    // test_rdna2_to_spirv) against the GL major-axis table. CONFIDENCE: MED.
    uint32_t x = fv(0), y = fv(1), z = fv(2);
    uint32_t ax = b.fext1(Glsl_FAbs, x), ay = b.fext1(Glsl_FAbs, y), az = b.fext1(Glsl_FAbs, z);
    uint32_t zmaj =
        b.land(b.fcmp(Op_FOrdGreaterThanEqual, az, ax), b.fcmp(Op_FOrdGreaterThanEqual, az, ay));
    uint32_t ymaj = b.fcmp(Op_FOrdGreaterThanEqual, ay, ax);      // (only used when !zmaj)
    uint32_t zneg = b.fcmp(Op_FOrdLessThan, z, b.uconst(0));
    uint32_t yneg = b.fcmp(Op_FOrdLessThan, y, b.uconst(0));
    uint32_t xneg = b.fcmp(Op_FOrdLessThan, x, b.uconst(0));
    auto fneg = [&](uint32_t v) { return b.fbin(Op_FSub, b.uconst(0), v); };
    uint32_t r2;
    switch (in.opcode) {
        case 0x144: {   // face id: z-major 4/5, y-major 2/3, x-major 0/1 (negative = +1)
            uint32_t idz = b.sel(zneg, b.uconst(fbits(5.0f)), b.uconst(fbits(4.0f)));
            uint32_t idy = b.sel(yneg, b.uconst(fbits(3.0f)), b.uconst(fbits(2.0f)));
            uint32_t idx2 = b.sel(xneg, b.uconst(fbits(1.0f)), b.uconst(fbits(0.0f)));
            r2 = b.sel(zmaj, idz, b.sel(ymaj, idy, idx2));
            break;
        }
        default: {      // 0x147 ma: 2 * major axis (signed)
            uint32_t maj = b.sel(zmaj, z, b.sel(ymaj, y, x));
            r2 = b.fbin(Op_FMul, b.uconst(fbits(2.0f)), maj);
            break;
        }
    }
    vreg[in.dst.value] = fresult(r2);
    return true;
}
bool lower_vop3_cube_axis(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const CubeAxisSource& fv,
                          const CubeAxisResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3CubeAxisLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
