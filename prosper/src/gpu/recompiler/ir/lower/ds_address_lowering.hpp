// Pure DS address values; access, bounds, EXEC and proof stay with the caller.
#pragma once
#include <array>
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using DsAddressRead = std::function<uint32_t(int)>;
struct DsAddressPair {
    uint32_t first, second;
};
DsAddressPair lower_ds_write2_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                          const DsAddressRead& vread);
DsAddressPair lower_ds_write2st64_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                              const DsAddressRead& vread, uint32_t off0,
                                              uint32_t off1);
DsAddressPair lower_ds_write2_b64_address(SpirvCompute& b, const Rdna2Inst& in,
                                          const DsAddressRead& vread);
std::array<uint32_t, 4> lower_ds_read2_b64_address(SpirvCompute& b, const Rdna2Inst& in,
                                                   const DsAddressRead& vread);
std::array<uint32_t, 2> lower_ds_read2_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                                   const DsAddressRead& vread);
std::array<uint32_t, 2> lower_ds_read2st64_b32_address(SpirvCompute& b, const Rdna2Inst& in,
                                                       const DsAddressRead& vread, uint32_t roff0,
                                                       uint32_t roff1);
uint32_t lower_ds_byte_address(SpirvCompute& b, const Rdna2Inst& in, const DsAddressRead& vread);
uint32_t lower_ds_store_word_index(SpirvCompute& b, uint32_t idx, int k);
uint32_t lower_ds_read_word_index(SpirvCompute& b, uint32_t idx, int k);
}   // namespace prosper::gpu::ir
