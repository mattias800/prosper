// Complete words from the established native legacy-sample lowering, never opcode admission.
#pragma once
#include <array>
#include <cstdint>

namespace prosper::gpu {
struct RegState;
struct Rdna2Inst;
struct NativeImageResultPublication {
    uint32_t pc = UINT32_MAX;
    uint32_t exec = 0;
    int destination = -1;
    uint32_t count = 0;
    std::array<uint32_t, 4> values{};
};
// Called only after the existing emitter has written every selected sample result.
void record_native_image_result(RegState& rs, const Rdna2Inst& in);
// Consume exactly one instruction's result generations; wider readers keep their old marks.
bool consume_native_image_result(RegState& rs, const Rdna2Inst& in);
} // namespace prosper::gpu
