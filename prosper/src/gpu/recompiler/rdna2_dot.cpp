#include "gpu/recompiler/rdna2_dot.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"

namespace prosper::gpu {

uint32_t emit_v_dot(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const uint32_t s0 = operand_bits(b, rs, in, in.src[0], &ok);
    const uint32_t s1 = operand_bits(b, rs, in, in.src[1], &ok);
    const uint32_t s2 = operand_bits(b, rs, in, in.src[2], &ok);
    if (!ok) return b.uconst(0);

    switch (in.opcode) {
        case 0x14: {  // v_dot2_i32_i16: signed 2x16-bit dot product + accumulator
            const uint32_t half0_0 = (in.vop3p_opsel >> 0) & 1u;
            const uint32_t half0_1 = (in.vop3p_opsel >> 1) & 1u;
            const uint32_t half1_0 = (in.vop3p_opsel_hi >> 0) & 1u;
            const uint32_t half1_1 = (in.vop3p_opsel_hi >> 1) & 1u;

            const uint32_t e0_0 = b.bfe_s(s0, b.uconst(half0_0 * 16u), b.uconst(16));
            const uint32_t e0_1 = b.bfe_s(s1, b.uconst(half0_1 * 16u), b.uconst(16));
            const uint32_t p0 = b.ibin(Op_IMul, e0_0, e0_1);

            const uint32_t e1_0 = b.bfe_s(s0, b.uconst(half1_0 * 16u), b.uconst(16));
            const uint32_t e1_1 = b.bfe_s(s1, b.uconst(half1_1 * 16u), b.uconst(16));
            const uint32_t p1 = b.ibin(Op_IMul, e1_0, e1_1);

            return b.ibin(Op_IAdd, b.ibin(Op_IAdd, p0, p1), s2);
        }
        case 0x15: {  // v_dot2_u32_u16: unsigned 2x16-bit dot product + accumulator
            const uint32_t half0_0 = (in.vop3p_opsel >> 0) & 1u;
            const uint32_t half0_1 = (in.vop3p_opsel >> 1) & 1u;
            const uint32_t half1_0 = (in.vop3p_opsel_hi >> 0) & 1u;
            const uint32_t half1_1 = (in.vop3p_opsel_hi >> 1) & 1u;

            const uint32_t e0_0 = b.bfe_u(s0, b.uconst(half0_0 * 16u), b.uconst(16));
            const uint32_t e0_1 = b.bfe_u(s1, b.uconst(half0_1 * 16u), b.uconst(16));
            const uint32_t p0 = b.ibin(Op_IMul, e0_0, e0_1);

            const uint32_t e1_0 = b.bfe_u(s0, b.uconst(half1_0 * 16u), b.uconst(16));
            const uint32_t e1_1 = b.bfe_u(s1, b.uconst(half1_1 * 16u), b.uconst(16));
            const uint32_t p1 = b.ibin(Op_IMul, e1_0, e1_1);

            return b.ibin(Op_IAdd, b.ibin(Op_IAdd, p0, p1), s2);
        }
        case 0x16: {  // v_dot4_i32_i8: signed 4x8-bit dot product + accumulator
            uint32_t sum = s2;
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t offset = k * 8u;
                const uint32_t e0 = b.bfe_s(s0, b.uconst(offset), b.uconst(8));
                const uint32_t e1 = b.bfe_s(s1, b.uconst(offset), b.uconst(8));
                sum = b.ibin(Op_IAdd, sum, b.ibin(Op_IMul, e0, e1));
            }
            return sum;
        }
        case 0x17: {  // v_dot4_u32_u8: unsigned 4x8-bit dot product + accumulator
            uint32_t sum = s2;
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t offset = k * 8u;
                const uint32_t e0 = b.bfe_u(s0, b.uconst(offset), b.uconst(8));
                const uint32_t e1 = b.bfe_u(s1, b.uconst(offset), b.uconst(8));
                sum = b.ibin(Op_IAdd, sum, b.ibin(Op_IMul, e0, e1));
            }
            return sum;
        }
        case 0x18: {  // v_dot8_i32_i4: signed 8x4-bit dot product + accumulator
            uint32_t sum = s2;
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t offset = k * 4u;
                const uint32_t e0 = b.bfe_s(s0, b.uconst(offset), b.uconst(4));
                const uint32_t e1 = b.bfe_s(s1, b.uconst(offset), b.uconst(4));
                sum = b.ibin(Op_IAdd, sum, b.ibin(Op_IMul, e0, e1));
            }
            return sum;
        }
        case 0x19: {  // v_dot8_u32_u4: unsigned 8x4-bit dot product + accumulator
            uint32_t sum = s2;
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t offset = k * 4u;
                const uint32_t e0 = b.bfe_u(s0, b.uconst(offset), b.uconst(4));
                const uint32_t e1 = b.bfe_u(s1, b.uconst(offset), b.uconst(4));
                sum = b.ibin(Op_IAdd, sum, b.ibin(Op_IMul, e0, e1));
            }
            return sum;
        }
        default: ok = false; return b.uconst(0);
    }
}

}  // namespace prosper::gpu
