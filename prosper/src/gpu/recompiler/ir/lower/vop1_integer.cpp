// Plain per-lane VOP1 integer cases, extracted verbatim with extract_case_routes.py.
#include "gpu/recompiler/ir/lower/vop1_integer_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool lower_vop1_integer(SpirvCompute& b, const Rdna2Inst& in, uint32_t a, uint32_t& d, bool& ok) {
    if (in.fmt != Rdna2Format::VOP1) return false;
    switch (in.opcode) {
        case 0x01: d = a; break;                              // v_mov_b32
        case 0x37: d = b.iun(Op_Not, a); break;               // v_not_b32
        case 0x38: d = b.iun(Op_BitReverse, a); break;        // v_bfrev_b32
        case kVop1OpcodeFfbhU32: {                            // v_ffbh_u32 (plain e32)
            // AMD RDNA2 ISA: count the zeroes preceding the first set bit from the MSB;
            // return -1 when no bit is set. GLSL.std.450 FindUMsb instead returns the SET
            // BIT INDEX (and all bits set at zero). OR bit zero in before calling it (which
            // leaves every nonzero input's highest bit unchanged), convert index -> count,
            // then explicitly select the architectural zero sentinel. GTA V uses the exact in-place
            // e32 packets 7e087304 / 7e047302 in its compute culling kernels.
            //
            // Keep SDWA out of this admission. The decoder's generic DWORD SDWA path can
            // carry NEG/ABS, which the VOP1 prologue above applies in the f32 domain; that
            // is not an integer source modifier and would silently change FFBH's operand.
            // DPP reaches and rejects in the stage-specific source-transform block above.
            if (in.has_sdwa || in.has_dpp) {
                ok = false;
                break;
            }
            d = b.ffbh_u32(a);
            break;
        }
        case kVop1OpcodeFfblB32: {   // v_ffbl_b32 (plain e32)
            // GTA V's Wave32 terrain dispatcher reaches the exact 0x7e047515 packet at
            // pc106. Like FFBH above this integer scan has no valid SDWA/DPP admission:
            // accepting either would let the common VOP1 modifier path reinterpret the
            // source as float before scanning its bits.
            if (in.has_sdwa || in.has_dpp) {
                ok = false;
                break;
            }
            d = b.ffbl_b32(a);
            break;
        }
        default: return false;
    }
    return true;
}
}   // namespace prosper::gpu::ir
