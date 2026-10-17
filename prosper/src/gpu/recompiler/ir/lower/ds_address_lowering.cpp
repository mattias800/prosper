// Extractor-relocated pure DS arithmetic, with no accesses or proof state.
#include "gpu/recompiler/ir/lower/ds_address_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
uint32_t lower_ds_read_word_index(SpirvCompute& b, uint32_t idx, int k) {
    const uint32_t at = k ? b.ibin(Op_IAdd, idx, b.uconst((uint32_t)k)) : idx;
    return at;
}
uint32_t lower_ds_store_word_index(SpirvCompute& b, uint32_t idx, int k) {
    const uint32_t at = k ? b.ibin(Op_IAdd, idx, b.uconst((uint32_t)k)) : idx;
    return at;
}
uint32_t lower_ds_byte_address(SpirvCompute& b, const Rdna2Inst& in, const DsAddressRead& vread) {
    uint32_t addr = b.ibin(Op_IAdd, vread(in.src[0].value), b.uconst(in.literal));
    if (in.ds_gds) addr = b.ibin(Op_BitwiseAnd, addr, b.uconst(0xFFFFu));
    uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
    return idx;
}
std::array<uint32_t, 2> lower_ds_read2st64_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                                       const DsAddressRead& vread, uint32_t roff0,
                                                       uint32_t roff1) {
    const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
    const uint32_t indices[2] = {
        b.ibin(Op_IAdd, base, b.uconst(roff0)),
        b.ibin(Op_IAdd, base, b.uconst(roff1)),
    };
    return {indices[0], indices[1]};
}
std::array<uint32_t, 2> lower_ds_read2_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                                   const DsAddressRead& vread) {
    const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
    const uint32_t indices[2] = {
        b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu)),
        b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu)),
    };
    return {indices[0], indices[1]};
}
std::array<uint32_t, 4> lower_ds_read2_b64_address(SpirvCompute& b, const Rdna2Inst& in,
                                                   const DsAddressRead& vread) {
    const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
    const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst((in.literal & 0xFFu) * 2u));
    const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(((in.literal >> 8) & 0xFFu) * 2u));
    const uint32_t indices[4] = {
        idx0,
        b.ibin(Op_IAdd, idx0, b.uconst(1)),
        idx1,
        b.ibin(Op_IAdd, idx1, b.uconst(1)),
    };
    return {indices[0], indices[1], indices[2], indices[3]};
}
DsAddressPair lower_ds_write2_b64_address(SpirvCompute& b, const Rdna2Inst& in,
                                          const DsAddressRead& vread) {
    const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
    const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst((in.literal & 0xFFu) * 2u));
    const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(((in.literal >> 8) & 0xFFu) * 2u));
    return {idx0, idx1};
}
DsAddressPair lower_ds_write2st64_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                              const DsAddressRead& vread, uint32_t off0,
                                              uint32_t off1) {
    const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
    const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst(off0));
    const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(off1));
    return {idx0, idx1};
}
DsAddressPair lower_ds_write2_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                          const DsAddressRead& vread) {
    const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
    const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu));
    const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu));
    return {idx0, idx1};
}
}   // namespace prosper::gpu::ir
