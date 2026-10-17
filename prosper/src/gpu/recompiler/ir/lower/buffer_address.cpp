// buffer_address.cpp — the MUBUF/MTBUF address (buffer_address_lowering.hpp). The value is the
// definition in ops/buffer_address.cpp; this file chooses its form from emit_alu's facts and reads
// the guest words where the pre-IR emitter read them.
#include "gpu/recompiler/ir/lower/buffer_address_lowering.hpp"

#include "gpu/recompiler/ir/ops/buffer_address.hpp"

namespace prosper::gpu::ir {

namespace {

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
BufferAddressShape shape_of(const BufferAddressFacts& f) {
    if (f.folded_vfetch && f.idxen && (f.stride || !f.offen)) {
        // A zero-stride V# still has its OFFSET/SOFFSET folded into the bound base.
        // With !OFFEN and stride zero, no dynamic address term remains. Adding
        // SOFFSET here would apply it twice.
        if (f.stride) return {BufferAddressForm::FoldedElement, 0, f.stride, false, false};
        return {BufferAddressForm::FoldedZero, 0, 0, false, false};
    }
    // Byte address of the element (#148): idxen -> a VADDR VGPR is an element index (×stride);
    // offen -> a per-lane byte offset; both terms ADD; when idxen AND offen are set VADDR is TWO
    // consecutive VGPRs ([0]=index, [1]=byte offset). Plus the inst offset and SOFFSET.
    return {BufferAddressForm::Faithful, f.offset, f.stride, f.idxen && f.stride != 0, f.offen};
}

}   // namespace

BufferAddressIds lower_buffer_address(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                      const BufferAddressFacts& facts) {
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o, &ok); };
    const auto ids = emit_buffer_address(b, shape_of(facts), [&](BufferAddressInput what) {
        switch (what) {
            case BufferAddressInput::Index: return val(in.src[0]);
            case BufferAddressInput::Voffset: {
                Operand off_vgpr{OperandKind::VGPR,
                                 facts.idxen ? in.src[0].value + 1 : in.src[0].value};
                return val(off_vgpr);
            }
            case BufferAddressInput::Soffset: return val(in.src[2]);
            case BufferAddressInput::Element:
                return facts.dyn_vfetch        ? b.load_vertex_index()
                       : facts.instance_vfetch ? b.load_instance_index()
                                               : val(in.src[0]);
        }
        return 0u;
    });
    return {ids[0], ids[1]};
}

}   // namespace prosper::gpu::ir
