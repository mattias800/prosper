#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;

// Only the newly admitted unsigned D16 form is restricted here. Existing float/normalized
// stores keep their established channel routing, including sparse masks.
bool unsigned_d16_store_supported(const Rdna2Inst& instruction);
uint32_t mimg_d16_store_component(SpirvCompute& builder, uint32_t packed, uint32_t half,
                                  bool unsigned_data);

template <class Read>
bool mimg_store_components(SpirvCompute& builder, const Rdna2Inst& instruction, bool float_data,
                           bool unsigned_data, Read&& read, uint32_t values[4]) {
    if (instruction.mimg_d16 && !float_data &&
        (!unsigned_data || !unsigned_d16_store_supported(instruction)))
        return false;
    uint32_t rank = 0;
    for (uint32_t channel = 0; channel < 4; ++channel) {
        if (!(instruction.mimg_dmask & (1u << channel))) continue;
        const int word =
            instruction.dst.value + static_cast<int>(instruction.mimg_d16 ? rank / 2 : rank);
        values[channel] = instruction.mimg_d16 ? mimg_d16_store_component(builder, read(word),
                                                                          rank & 1u, unsigned_data)
                                               : read(word);
        ++rank;
    }
    return true;
}
} // namespace prosper::gpu
