// Bind the existing DS facts/reads. Accesses, bounds, EXEC and proof stay with the caller.
#include "gpu/recompiler/ir/lower/ds_address_lowering.hpp"
#include "gpu/recompiler/ir/ops/ds_address.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
DsAddressPair pair(SpirvCompute& b, const Rdna2Inst& in, const DsAddressRead& read,
                   uint32_t offset0, uint32_t offset1) {
    const auto ids = emit_ds_address_values(b, build_ds_pair_address(offset0, offset1),
                                            [&](uint32_t) { return read(in.src[0].value); });
    return {ids[0], ids[1]};
}
uint32_t word_index(SpirvCompute& b, uint32_t idx, int k) {
    return emit_ds_address_values(b, build_ds_word_index(static_cast<uint32_t>(k)),
                                  [&](uint32_t) { return idx; })[0];
}
} // namespace
DsAddressPair lower_ds_write2_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                          const DsAddressRead& vread) {
    return pair(b, in, vread, in.literal & 255u, (in.literal >> 8) & 255u);
}
DsAddressPair lower_ds_write2st64_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                              const DsAddressRead& vread, uint32_t off0,
                                              uint32_t off1) {
    return pair(b, in, vread, off0, off1);
}
DsAddressPair lower_ds_write2_b64_address(SpirvCompute& b, const Rdna2Inst& in,
                                          const DsAddressRead& vread) {
    return pair(b, in, vread, (in.literal & 255u) * 2u, ((in.literal >> 8) & 255u) * 2u);
}
std::array<uint32_t, 4> lower_ds_read2_b64_address(SpirvCompute& b, const Rdna2Inst& in,
                                                   const DsAddressRead& vread) {
    const auto idx = pair(b, in, vread, (in.literal & 255u) * 2u, ((in.literal >> 8) & 255u) * 2u);
    const auto ids = emit_ds_address_values(
        b, build_ds_pair_words(), [&](uint32_t slot) { return slot ? idx.second : idx.first; });
    return {ids[0], ids[1], ids[2], ids[3]};
}
std::array<uint32_t, 2> lower_ds_read2_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                                   const DsAddressRead& vread) {
    const auto idx = pair(b, in, vread, in.literal & 255u, (in.literal >> 8) & 255u);
    return {idx.first, idx.second};
}
std::array<uint32_t, 2> lower_ds_read2st64_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                                       const DsAddressRead& vread, uint32_t roff0,
                                                       uint32_t roff1) {
    const auto idx = pair(b, in, vread, roff0, roff1);
    return {idx.first, idx.second};
}
uint32_t lower_ds_byte_address(SpirvCompute& b, const Rdna2Inst& in, const DsAddressRead& vread) {
    return emit_ds_address_values(b, build_ds_byte_address(in.literal, in.ds_gds),
                                  [&](uint32_t) { return vread(in.src[0].value); })[0];
}
uint32_t lower_ds_store_word_index(SpirvCompute& b, uint32_t idx, int k) {
    return word_index(b, idx, k);
}
uint32_t lower_ds_read_word_index(SpirvCompute& b, uint32_t idx, int k) {
    return word_index(b, idx, k);
}
} // namespace prosper::gpu::ir
