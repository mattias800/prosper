// ngg_draw_indices.cpp -- see ngg_draw_indices.hpp.
#include "gpu/execute/ngg_draw_indices.hpp"

#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace prosper::gpu {

namespace P = prosper::agc::Pm4;

NggIndexRestart read_ngg_index_restart(const GpuState& state) {
    NggIndexRestart restart;
    const auto enable = state.uc.find(P::GE_MULTI_PRIM_IB_RESET_EN);
    restart.enabled = enable != state.uc.end() && (enable->second & 1u) != 0;
    const auto value = state.cx.find(P::VGT_MULTI_PRIM_IB_RESET_INDX);
    restart.value_known = value != state.cx.end();
    restart.value = restart.value_known ? value->second : 0u;
    return restart;
}

NggDrawIndices decode_ngg_draw_indices(const void* bytes, uint32_t element_bytes, uint32_t count,
                                       const NggIndexRestart& restart) {
    NggDrawIndices out;
    if (element_bytes != 2u && element_bytes != 4u) {
        out.refusal = "ngg-index-element-size";
        return out;
    }
    if (count == 0 || count > kNggMaxIndices || !bytes) {
        out.refusal = "ngg-index-count";
        return out;
    }
    if (restart.enabled && !restart.value_known) {
        out.refusal = "ngg-index-restart-unknown";
        return out;
    }
    const uint32_t restart_value = element_bytes == 2u ? restart.value & 0xffffu : restart.value;
    auto indices = std::make_shared<std::vector<uint32_t>>(count);
    const auto* src = static_cast<const unsigned char*>(bytes);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t value = 0;
        if (element_bytes == 2u) {
            uint16_t narrow = 0;
            std::memcpy(&narrow, src + size_t{2} * i, 2u);
            value = narrow;
        } else {
            std::memcpy(&value, src + size_t{4} * i, 4u);
        }
        if (restart.enabled && value == restart_value) {
            out.refusal = "ngg-index-restart";
            return out;
        }
        (*indices)[i] = value;
        out.max_index = std::max(out.max_index, value);
    }
    // #461's guard, as a refusal: the vertex RANGE max_index + 1 sizes the fold and every vertex
    // buffer, so one garbage or torn index would grow them to the 256 MiB ceiling on every draw (and
    // 0xffffffff would wrap the range to 0). The ordinary path clamps; a clamp here would leave ES
    // lanes fetching past the grown buffers, so the draw is refused by name instead.
    if (out.max_index >= kNggMaxIndices) {
        out.refusal = "ngg-index-range";
        out.max_index = 0;
        return out;
    }
    out.indices = std::move(indices);
    return out;
}

}   // namespace prosper::gpu
