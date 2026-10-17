// buffer_address.cpp — the MUBUF/MTBUF address, moved verbatim out of emit_alu's MUBUF/MTBUF case
// by tools/refactor/extract_case_routes.py (buffer_address_lowering.hpp).
#include "gpu/recompiler/ir/lower/buffer_address_lowering.hpp"

namespace prosper::gpu::ir {

BufferAddressIds lower_buffer_address(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                      const BufferAddressFacts& facts) {
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o, &ok); };
    const uint32_t offset = facts.offset, stride = facts.stride;
    const bool offen = facts.offen, idxen = facts.idxen, folded_vfetch = facts.folded_vfetch,
               dyn_vfetch = facts.dyn_vfetch, instance_vfetch = facts.instance_vfetch;
    // Byte address of the element (#148): idxen -> a VADDR VGPR is an element index (×stride);
    // offen -> a per-lane byte offset; both terms ADD; when idxen AND offen are set VADDR is TWO
    // consecutive VGPRs ([0]=index, [1]=byte offset). Plus the inst offset and SOFFSET.
    uint32_t addr;
    // Const-folded per-vertex attribute fetch (#206): the element address is exactly
    // gl_VertexIndex*stride. (1) The NGG fetch-shader prologue that computes the element index
    // isn't fully modeled and folds to a constant, so every vertex would read record 0 -> a
    // degenerate single point that rasterizes nothing (the whole scene stayed the blue clear).
    // (2) The resolved V# base ALREADY includes this attribute's byte offset within the
    // interleaved record, so the shader's instruction OFFSET + SOFFSET must NOT be added again
    // (double-counting pushes the read OOB -> robustBufferAccess 0, the same collapse). So use
    // gl_VertexIndex*stride and drop the shader's element-index/constant offset terms. The
    // zero-stride specialization below is limited to !OFFEN: packed-format alignment does
    // not yet account for OFFEN's potentially unaligned dynamic byte offset. Zero-stride OFFEN
    // stays on the prior path; non-folded fetches keep the faithful address (incl. #148's
    // idxen+offen both-terms fix). CONFIDENCE: HIGH — makes
    // PPSA01885 (Unity/IL2CPP) render real geometry instead of a degenerate collapse.
    if (folded_vfetch && idxen && (stride || !offen)) {
        // A zero-stride V# still has its OFFSET/SOFFSET folded into the bound base.
        // With !OFFEN and stride zero, no dynamic address term remains. Adding
        // SOFFSET here would apply it twice.
        if (stride) {
            const uint32_t element = dyn_vfetch ? b.load_vertex_index()
                                   : instance_vfetch ? b.load_instance_index()
                                                     : val(in.src[0]);
            addr = b.ibin(Op_IMul, element, b.uconst(stride));
        } else {
            addr = b.uconst(0);
        }
    } else {
        addr = b.uconst(offset);
        if (idxen && stride) addr = b.ibin(Op_IAdd, addr, b.ibin(Op_IMul, val(in.src[0]), b.uconst(stride)));
        if (offen) {
            Operand off_vgpr{ OperandKind::VGPR, idxen ? in.src[0].value + 1 : in.src[0].value };
            addr = b.ibin(Op_IAdd, addr, val(off_vgpr));
        }
        addr = b.ibin(Op_IAdd, addr, val(in.src[2]));          // SOFFSET
    }
    uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
    return {addr, idx};
}

}   // namespace prosper::gpu::ir
