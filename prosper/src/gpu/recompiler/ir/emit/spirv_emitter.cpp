// spirv_emitter.cpp — the SPIR-V half of each IR opcode's definition (see ir.hpp).
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <span>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu::ir {

namespace {

uint32_t emit_one(SpirvCompute& b, const Inst& inst, const uint32_t* x) {
    switch (inst.op) {
        case Opcode::Input: break;
        case Opcode::ConstU32: return b.uconst(inst.imm);
        case Opcode::ConstTrue: return b.btrue();
        case Opcode::ConstFalse: return b.bfalse();
        case Opcode::BoolAnd: return b.land(x[0], x[1]);
        case Opcode::F32Clamp01: return b.clamp01(x[0]);
        case Opcode::PackHalfLo: return b.pack_half_lo(x[0]);
        case Opcode::F32Add: return b.fbin(Op_FAdd, x[0], x[1]);
        case Opcode::F32Sub: return b.fbin(Op_FSub, x[0], x[1]);
        case Opcode::F32Mul: return b.fbin(Op_FMul, x[0], x[1]);
        case Opcode::F32NMin: return b.fext2(Glsl_NMin, x[0], x[1]);
        case Opcode::F32NMax: return b.fext2(Glsl_NMax, x[0], x[1]);
        case Opcode::FfbhU32: return b.ffbh_u32(x[0]);
        case Opcode::F32Fract: return b.fext1(Glsl_Fract, x[0]);
        case Opcode::F32Trunc: return b.fext1(Glsl_Trunc, x[0]);
        case Opcode::F32Ceil: return b.fext1(Glsl_Ceil, x[0]);
        case Opcode::F32RoundEven: return b.fext1(Glsl_RoundEven, x[0]);
        case Opcode::F32Floor: return b.fext1(Glsl_Floor, x[0]);
        case Opcode::F32Exp2: return b.fext1(Glsl_Exp2, x[0]);
        case Opcode::F32Log2: return b.fext1(Glsl_Log2, x[0]);
        case Opcode::F32Rcp: return b.frcp(x[0]);
        case Opcode::F32InverseSqrt: return b.fext1(Glsl_InverseSqrt, x[0]);
        case Opcode::F32Sqrt: return b.fext1(Glsl_Sqrt, x[0]);
        case Opcode::F32Sin: return b.fext1(Glsl_Sin, x[0]);
        case Opcode::F32Cos: return b.fext1(Glsl_Cos, x[0]);
        case Opcode::I32ToF32: return b.cvt_i2f(x[0]);
        case Opcode::U32ToF32: return b.cvt_u2f(x[0]);
        case Opcode::F32ToU32Sat: return b.cvt_f2u(x[0]);
        case Opcode::F32ToI32Sat: return b.cvt_f2i(x[0]);
        case Opcode::BfeS: return b.bfe_s(x[0], x[1], x[2]);
        case Opcode::UnpackHalf: return b.unpack_half(x[0], inst.imm);
        case Opcode::F32Ge: return b.fcmp(Op_FOrdGreaterThanEqual, x[0], x[1]);
        case Opcode::IAdd: return b.ibin(Op_IAdd, x[0], x[1]);
        case Opcode::ISub: return b.ibin(Op_ISub, x[0], x[1]);
        case Opcode::IMul: return b.ibin(Op_IMul, x[0], x[1]);
        case Opcode::UMulHi: return b.umul_hi(x[0], x[1]);
        case Opcode::SMulHi: return b.smul_hi(x[0], x[1]);
        case Opcode::And: return b.ibin(Op_BitwiseAnd, x[0], x[1]);
        case Opcode::Or: return b.ibin(Op_BitwiseOr, x[0], x[1]);
        case Opcode::Xor: return b.ibin(Op_BitwiseXor, x[0], x[1]);
        case Opcode::Not: return b.iun(Op_Not, x[0]);
        case Opcode::BitReverse: return b.iun(Op_BitReverse, x[0]);
        case Opcode::BitCount: return b.iun(Op_BitCount, x[0]);
        case Opcode::Shl: return b.ibin(Op_ShiftLeftLogical, x[0], x[1]);
        case Opcode::LShr: return b.ibin(Op_ShiftRightLogical, x[0], x[1]);
        case Opcode::AShr: return b.sbin(Op_ShiftRightArithmetic, x[0], x[1]);
        case Opcode::SMin: return b.sext2(Glsl_SMin, x[0], x[1]);
        case Opcode::SMax: return b.sext2(Glsl_SMax, x[0], x[1]);
        case Opcode::UMin: return b.uext2(Glsl_UMin, x[0], x[1]);
        case Opcode::UMax: return b.uext2(Glsl_UMax, x[0], x[1]);
        case Opcode::BfeU: return b.bfe_u(x[0], x[1], x[2]);
        case Opcode::Select: return b.sel(x[0], x[1], x[2]);
        case Opcode::BoolSelect: return b.bsel(x[0], x[1], x[2]);
        case Opcode::IEq: return b.ucmp(Op_IEqual, x[0], x[1]);
        case Opcode::INe: return b.ucmp(Op_INotEqual, x[0], x[1]);
        case Opcode::ULt: return b.ucmp(Op_ULessThan, x[0], x[1]);
        case Opcode::ULe: return b.ucmp(Op_ULessThanEqual, x[0], x[1]);
        case Opcode::UGt: return b.ucmp(Op_UGreaterThan, x[0], x[1]);
        case Opcode::UGe: return b.ucmp(Op_UGreaterThanEqual, x[0], x[1]);
        case Opcode::SLt: return b.scmp(Op_SLessThan, x[0], x[1]);
        case Opcode::SLe: return b.scmp(Op_SLessThanEqual, x[0], x[1]);
        case Opcode::SGt: return b.scmp(Op_SGreaterThan, x[0], x[1]);
        case Opcode::SGe: return b.scmp(Op_SGreaterThanEqual, x[0], x[1]);
    }
    std::fprintf(stderr, "[ir] emit_one reached an input\n");   // emit_spirv binds inputs itself
    std::abort();
}

} // namespace

namespace {
template <class InputReader>
std::vector<uint32_t> emit_block(SpirvCompute& b, const Block& block,
                                 const InputReader& read_input) {
    std::vector<uint32_t> ids(block.size(), 0);
    for (size_t i = 0; i < block.size(); ++i) {
        const Inst& inst = block.insts()[i];
        if (inst.op == Opcode::Input) {
            ids[i] = read_input(inst.imm);
            continue;
        }
        uint32_t args[3] = {0, 0, 0};
        for (uint8_t slot = 0; slot < opcode_info(inst.op).arity; ++slot)
            args[slot] = ids[inst.args[slot].index];
        ids[i] = emit_one(b, inst, args);
    }
    return ids;
}
}   // namespace
std::vector<uint32_t> emit_spirv(SpirvCompute& b, const Block& block,
                                 std::span<const uint32_t> inputs) {
    return emit_block(b, block,
                      [inputs](uint32_t slot) { return slot < inputs.size() ? inputs[slot] : 0; });
}
std::vector<uint32_t> emit_spirv_resolving_inputs(SpirvCompute& b, const Block& block,
                                                  const SpirvInputResolver& read_input) {
    if (block.input_count() && !read_input) {
        std::fprintf(stderr, "[ir] input resolver missing\n");
        std::abort();
    }
    return emit_block(b, block, read_input);
}

} // namespace prosper::gpu::ir
