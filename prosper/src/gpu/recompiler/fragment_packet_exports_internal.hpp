#pragma once
#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
bool emit_packet_architectural_export(SpirvCompute&, RegState&, const Rdna2Inst&, uint32_t base,
                                      uint32_t stride, uint32_t enabled);
} // namespace prosper::gpu
