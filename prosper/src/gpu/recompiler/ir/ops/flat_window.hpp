// Pure low-word window offsets. Resource authority and all accesses belong to the caller.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
struct FlatWindowOffsetBlock {
    Block block;
    Value byte_offset;
};
struct FlatWindowComponentBlock {
    Block block;
    Value byte_address, dword_index;
};
FlatWindowOffsetBlock build_flat_window_byte_offset();
FlatWindowComponentBlock build_flat_window_component(uint32_t bits, uint32_t component);
} // namespace prosper::gpu::ir
