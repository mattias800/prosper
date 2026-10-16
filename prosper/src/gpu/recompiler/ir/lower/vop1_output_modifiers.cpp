// Existing numeric VOP1 output stage; operand, DPP and predicate bookkeeping stays with the caller.
#include "gpu/recompiler/ir/lower/vop1_output_modifiers.hpp"
#include "gpu/recompiler/ir/lower/float_output_modifiers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
namespace prosper::gpu::ir {
void lower_vop1_output_modifiers(SpirvCompute& b, const Rdna2Inst& in, uint32_t& d, bool& ok) {
    // SDWA output modifiers: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result
    // opcodes only (mirrors the VOP2/VOP3 fresult path; DOLL VS: `v_exp_f32_sdwa … clamp`).
    // A modifier on a non-float-result op would silently drop — reject loudly instead.
    if (ok && (in.omod || in.clamp)) switch (in.opcode) {
            case 0x05:
            case 0x06:
            case 0x0B:
            case 0x0E:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x27:
            case 0x2A:
            case 0x2B:
            case 0x2E:
            case 0x33:
            case 0x35:
            case 0x36: d = ir::modify_float_result(b, in, d, ok); break;
            default: ok = false; break;
        }
}
}   // namespace prosper::gpu::ir
