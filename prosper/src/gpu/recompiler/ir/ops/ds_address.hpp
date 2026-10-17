// Shared pure DS address/index definitions; no access, wave state or proof ownership.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
#include <functional>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct DsAddressBlock {
    Block block;
    std::vector<Value> outputs;
};
DsAddressBlock build_ds_byte_address(uint32_t literal, bool gds);
DsAddressBlock build_ds_pair_address(uint32_t offset0, uint32_t offset1);
DsAddressBlock build_ds_pair_words();
DsAddressBlock build_ds_word_index(uint32_t component);
using DsAddressValueReader = std::function<uint32_t(uint32_t)>;
std::vector<uint32_t> emit_ds_address_values(SpirvCompute& b, const DsAddressBlock& built,
                                             const DsAddressValueReader& read);
} // namespace prosper::gpu::ir
