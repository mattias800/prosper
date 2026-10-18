// Existing MIMG normalization. Readers and accesses remain in emit_alu; this adapter receives
// an already-read word, so nested access arguments keep their original evaluation positions.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct LegacyImageSpatialNormalizer {
    SpirvCompute& b;
    bool& ok;
    const bool normalize_sampler_coordinates;
    uint32_t operator()(uint32_t coordinate, uint32_t extent) const;
};
// Only the operand-first 2D and array callers use this IR adapter. Other MIMG paths retain the
// relocated raw adapter until their own migration. Neither adapter owns source reads or accesses.
struct ImageOperandCoordinateNormalizer {
    SpirvCompute& b;
    bool& ok;
    const bool normalize_sampler_coordinates;
    uint32_t operator()(uint32_t coordinate, uint32_t extent) const;
};
} // namespace prosper::gpu::ir
