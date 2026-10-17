// Pure buffer store values. Reads are appended at their old expression positions; all memory,
// descriptor/routing, EXEC and proof decisions remain caller-owned.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
#include <functional>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
enum class BufferStoreKind : uint8_t { Integer, Half, Unorm, Snorm, Ufloat };
struct BufferStoreShape {
    BufferStoreKind kind = BufferStoreKind::Integer;
    uint32_t count = 4, physical_count = 4, bytes = 1;
};
// Input slots name VDATA component0..3 or the already-resolved byte address/dword index(4).
struct BufferStoreBlock {
    Block block;
    std::vector<uint32_t> inputs;
    std::vector<Value> outputs;
};
BufferStoreBlock build_buffer_store_fields(uint32_t component, uint32_t bytes, uint32_t mask);
BufferStoreBlock build_buffer_store_index(uint32_t component);
BufferStoreBlock build_buffer_store_word(const BufferStoreShape& shape);
BufferStoreBlock build_buffer_store_dword(const BufferStoreShape& shape, uint32_t dword);
using BufferStoreValueReader = std::function<uint32_t(uint32_t)>;
std::vector<uint32_t> emit_buffer_store_values(SpirvCompute& b, const BufferStoreBlock& block,
                                               const BufferStoreValueReader& read);
}   // namespace prosper::gpu::ir
