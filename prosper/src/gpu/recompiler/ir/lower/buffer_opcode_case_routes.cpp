// Mechanical opcode-classification routes; access and descriptor proofs remain with emit_alu.
#include "gpu/recompiler/ir/lower/buffer_opcode_case_routes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <optional>

namespace prosper::gpu::ir {
std::optional<bool> BufferOpcodeCaseRoutes::classify() {
    switch (in.opcode) {
        case 0x8:
        case 0x9:
        case 0xA:
        case 0xB:
        case kMubufOpcodeLoadDword:
        case kMubufOpcodeLoadDwordX2:
        case kMubufOpcodeLoadDwordX4:
        case kMubufOpcodeLoadDwordX3: raw_loads(); break;
        case 0x0:
        case 0x1:
        case 0x2:
        case 0x3:
        case kMubufOpcodeStoreDword:
        case kMubufOpcodeStoreDwordX2:
        case kMubufOpcodeStoreDwordX4:
        case kMubufOpcodeStoreDwordX3:
        case 0x4:
        case 0x5:
        case 0x6:
        case 0x7: format_and_stores(); break;
        case kMubufOpcodeAtomicSwap:
        case kMubufOpcodeAtomicAdd:
        case kMubufOpcodeAtomicSub:
        case kMubufOpcodeAtomicSmin:
        case kMubufOpcodeAtomicUmin:
        case kMubufOpcodeAtomicSmax:
        case kMubufOpcodeAtomicUmax:
        case kMubufOpcodeAtomicAnd:
        case kMubufOpcodeAtomicOr:
        case kMubufOpcodeAtomicXor: integer_atomics(); break;
        case kMubufOpcodeAtomicFmin:
        case kMubufOpcodeAtomicFmax:
        case kMubufOpcodeAtomicSwapX2:
        case kMubufOpcodeAtomicOrX2: float_and_wide_atomics(); break;
        default: ok = false; return true;           // remaining typed/atomic opcodes deferred
    }
    return std::nullopt;
}

void BufferOpcodeCaseRoutes::raw_loads() {
    switch (in.opcode) {
        case 0x8:
            n = 1;
            raw_subword = true;
            raw_bits = 8;
            break;   // buffer_load_ubyte
        case 0x9:
            n = 1;
            raw_subword = true;
            raw_signed = true;
            raw_bits = 8;
            break; // sbyte
        case 0xA:
            n = 1;
            raw_subword = true;
            raw_bits = 16;
            break;  // buffer_load_ushort
        case 0xB:
            n = 1;
            raw_subword = true;
            raw_signed = true;
            raw_bits = 16;
            break; // sshort
        case kMubufOpcodeLoadDword: n = 1; break;
        case kMubufOpcodeLoadDwordX2: n = 2; break;
        case kMubufOpcodeLoadDwordX4: n = 4; break;
        case kMubufOpcodeLoadDwordX3: n = 3; break; // x3 sorts after x4 (llvm-mc gfx1010)
        default: break;
    }
}

void BufferOpcodeCaseRoutes::format_and_stores() {
    switch (in.opcode) {
        case 0x0:
            n = 1;
            is_format = true;
            break;   // buffer_load_format_x  (vertex fetch)
        case 0x1:
            n = 2;
            is_format = true;
            break;   // buffer_load_format_xy
        case 0x2:
            n = 3;
            is_format = true;
            break;   // buffer_load_format_xyz
        case 0x3:
            n = 4;
            is_format = true;
            break;   // buffer_load_format_xyzw
        case kMubufOpcodeStoreDword:
            n = 1;
            is_store = true;
            break;
        case kMubufOpcodeStoreDwordX2:
            n = 2;
            is_store = true;
            break;
        case kMubufOpcodeStoreDwordX4:
            n = 4;
            is_store = true;
            break;
        case kMubufOpcodeStoreDwordX3:
            n = 3;
            is_store = true;
            break;
        case 0x4:
            n = 1;
            is_format = true;
            is_store = true;
            break;   // buffer_store_format_x
        case 0x5:
            n = 2;
            is_format = true;
            is_store = true;
            break;   // buffer_store_format_xy
        case 0x6:
            n = 3;
            is_format = true;
            is_store = true;
            break;   // buffer_store_format_xyz
        case 0x7:
            n = 4;
            is_format = true;
            is_store = true;
            break;   // buffer_store_format_xyzw
        default: break;
    }
}

void BufferOpcodeCaseRoutes::integer_atomics() {
    switch (in.opcode) {
    // 32-bit atomic RMW family (RDNA2 MUBUF opcodes 0x30..0x3b). Each does
            // mem = mem OP VDATA and returns the PRE-op value in VDATA. They all share the generic
        // OpAtomic<Op>(ptr, Device, AcqRel, value) shape emitted by cbuf_atomic_rtn. VDATA is one
        // dword. CMPSWAP (0x31, two-operand), CSUB (0x34, conditional-subtract), INC/DEC
        // (0x3c/0x3d, wrap semantics), and most x2 64-bit variants stay deferred
        // (fail-visible) via the default case below. Opcodes 0x50/0x5a have separately
        // guarded GTA V lowerings: one true qword RMW, never two ordinary 32-bit atomics.
        case kMubufOpcodeAtomicSwap:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicExchange;
            break;
        case kMubufOpcodeAtomicAdd:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicIAdd;
            break;
        case kMubufOpcodeAtomicSub:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicISub;
            break;
        case kMubufOpcodeAtomicSmin:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicSMin;
            break;
        case kMubufOpcodeAtomicUmin:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicUMin;
            break;
        case kMubufOpcodeAtomicSmax:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicSMax;
            break;
        case kMubufOpcodeAtomicUmax:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicUMax;
            break;
        case kMubufOpcodeAtomicAnd:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicAnd;
            break;
        case kMubufOpcodeAtomicOr:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicOr;
            break;
        case kMubufOpcodeAtomicXor:
            n = 1;
            is_atomic = true;
            atomic_op = Op_AtomicXor;
            break;
        default: break;
    }
}

void BufferOpcodeCaseRoutes::float_and_wide_atomics() {
    switch (in.opcode) {
        case kMubufOpcodeAtomicFmin:
            n = 1;
            is_atomic = true;
            is_atomic_fminmax = true;
            atomic_fmin = true;
            break;
        case kMubufOpcodeAtomicFmax:
            n = 1;
            is_atomic = true;
            is_atomic_fminmax = true;
            atomic_fmin = false;
            break;
        case kMubufOpcodeAtomicSwapX2:
            n = 2;
            is_atomic = true;
            is_atomic_x2 = true;
            atomic_op = Op_AtomicExchange;
            break;
        case kMubufOpcodeAtomicOrX2:
            n = 2;
            is_atomic = true;
            is_atomic_x2 = true;
            atomic_op = Op_AtomicOr;
            break;
        default: break;
    }
}

}   // namespace prosper::gpu::ir
