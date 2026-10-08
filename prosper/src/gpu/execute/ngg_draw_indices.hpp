// ngg_draw_indices.hpp -- the index buffer of an indexed merged ES+GS NGG draw (#3135 phase P6).
//
// An ordinary indexed draw hands its indices to the backend, which binds them for vkCmdDrawIndexed.
// A merged-NGG draw cannot: its subgroup planner (ngg_subgroup_plan.hpp) needs the index VALUES on
// the CPU, because it deduplicates ES vertices by value within a subgroup and each ES lane's VertexID
// (v5) is the index that lane runs. So the indices are read once at realization and end up in the
// launch records; the description the backend receives carries no index buffer at all.
//
// WHERE the indices live and at what element size is decided by the ordinary path's own rule
// (resolve_draw_index_source in gpu_execute.hpp, #304/#3009), so the two paths cannot read one buffer
// two ways. This file decodes the bytes and applies primitive restart.
//
// PRIMITIVE RESTART. GE_MULTI_PRIM_IB_RESET_EN.RESET_EN [0] enables it and VGT_MULTI_PRIM_IB_RESET_INDX
// holds the restart value. What the hardware does with a restart index inside a LIST is not
// established from public sources, and no title has shown one, so it is refused rather than
// modelled: an enabled restart is refused when any index equals the restart value (compared at the
// element width: a 16-bit index against the value's low 16 bits, a deliberate superset), or when
// the value is not in the draw state. An enabled restart that no index matches cannot change what
// is drawn and is admitted. CONFIDENCE: HIGH on the bit and the register (Mesa's gfx10.3
// S_03092C_RESET_EN and R_02840C); the refusal is the fail-closed choice.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {

struct GpuState;

// The most indices one merged-NGG draw may carry, and the bound on an index VALUE; more is refused
// (ngg-index-count, ngg-index-range). Every live shape recorded so far has 6 or 18 (#3135); the
// ordinary path's sanity cap is the same 2^20 (#461).
inline constexpr uint32_t kNggMaxIndices = 1u << 20;

struct NggIndexRestart {
    bool enabled = false;
    bool value_known = false;
    uint32_t value = 0;
};

// GE_MULTI_PRIM_IB_RESET_EN (reset value 0: disabled) and VGT_MULTI_PRIM_IB_RESET_INDX at the draw.
NggIndexRestart read_ngg_index_restart(const GpuState& state);

struct NggDrawIndices {
    std::shared_ptr<const std::vector<uint32_t>> indices;   // null when refused
    uint32_t max_index = 0;
    const char* refusal = nullptr;   // a static rule name, null when the indices were read
};

// Decodes `count` little-endian indices of `element_bytes` (2 or 4) from `bytes`. Refusals:
// ngg-index-element-size, ngg-index-count (zero, or above kNggMaxIndices), ngg-index-restart-unknown,
// ngg-index-restart, ngg-index-range (an index value at or above kNggMaxIndices: the vertex range
// max_index + 1 sizes the fold and every vertex buffer, #461).
NggDrawIndices decode_ngg_draw_indices(const void* bytes, uint32_t element_bytes, uint32_t count,
                                       const NggIndexRestart& restart);

}   // namespace prosper::gpu
