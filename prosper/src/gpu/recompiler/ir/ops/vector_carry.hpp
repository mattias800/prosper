// vector_carry.hpp — the ONE definition of each carry op's per-lane value (ADR 0012).
//
// A carry op is two halves, like a VOPC compare. This header is the first: a VectorCarryDef's
// `build` appends the lane's 32-bit result D and its carry (or borrow) as a Bool, from the two
// source words and, for the _co_ci forms, the lane's carry-in bit. The second half is the same for
// every carry op and lives in ir/lower/carry_mask_write.cpp: EXEC-mask the carry (ISA 3.9) and write
// it to VCC (VOP2) or to the VOP3B SDST. Reading the carry-in mask (VCC or an SGPR pair) is the
// lowering's job too; a definition only sees the lane's bit, so it knows nothing about masks.
//
// The VOP2 (e32) and VOP3B encodings of one operation share one build function.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <span>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

struct VectorCarryIn {
    Value s0, s1;
    Value cin = {};   // Bool: the lane's carry-in bit (reads_carry_in only)
};
struct VectorCarryOut {
    Value d;   // U32
    Value carry;   // Bool: the carry-out (add) or borrow-out (sub, subrev)
};

struct VectorCarryDef {
    Rdna2Format fmt;
    uint32_t opcode;
    bool reads_carry_in;   // the _co_ci forms
    VectorCarryOut (*build)(Builder& ir, const VectorCarryIn& in);
};

std::span<const VectorCarryDef> vector_carry_defs();
const VectorCarryDef* find_vector_carry(Rdna2Format fmt, uint32_t opcode);

// One lane, on known sources; `cin` is 0 or 1 and ignored without reads_carry_in.
struct KnownCarry {
    Known d, carry;
};
KnownCarry eval_vector_carry(const VectorCarryDef& def, Known s0, Known s1, Known cin = 0u);

// The SPIR-V ids of D and of the (unmasked) carry Bool. `cin_mask` is the id of the lane's
// carry-in Bool; it must be nonzero exactly when the definition reads a carry-in.
struct CarryIds {
    uint32_t d, carry;
};
CarryIds emit_vector_carry(SpirvCompute& b, const VectorCarryDef& def, uint32_t s0, uint32_t s1,
                           uint32_t cin_mask = 0);

}   // namespace prosper::gpu::ir
