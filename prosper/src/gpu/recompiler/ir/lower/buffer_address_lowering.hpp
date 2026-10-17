// buffer_address_lowering.hpp — the byte address and dword index of a MUBUF/MTBUF access
// (HOW_TO_ADD_A_FAMILY.md, "Memory instructions").
//
// emit_alu's MUBUF/MTBUF case decides everything around the address: which descriptor the access
// resolves to (and so `stride`), whether the V# is a folded per-vertex fetch, the format, the
// coherence and every refusal. It hands those facts here and gets back the two ids every load,
// store and atomic below it consumes. Nothing here reads or writes provenance, marks or proof state.
#pragma once

#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>

namespace prosper::gpu::ir {

// What emit_alu has decided about the instruction by the time it computes the address.
struct BufferAddressFacts {
    uint32_t offset = 0;            // the instruction's 12-bit OFFSET
    uint32_t stride = 0;            // the resolved V#'s STRIDE (0 when none resolved)
    bool offen = false;             // OFFEN
    bool idxen = false;             // IDXEN
    bool folded_vfetch = false;     // the resolved V# base already includes OFFSET/SOFFSET (#206)
    bool dyn_vfetch = false;        // the element is gl_VertexIndex
    bool instance_vfetch = false;   // the element is gl_InstanceIndex
};

struct BufferAddressIds {
    uint32_t addr = 0;   // byte address of the element
    uint32_t idx = 0;    // addr >> 2: the dword the access starts at
};

// Emits the address in emit_alu's order; reads VADDR and SOFFSET through operand_bits, which
// clears `ok` exactly as emit_alu's `val` did.
BufferAddressIds lower_buffer_address(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                      const BufferAddressFacts& facts);

}   // namespace prosper::gpu::ir
