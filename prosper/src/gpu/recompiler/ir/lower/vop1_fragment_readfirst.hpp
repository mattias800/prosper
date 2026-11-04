#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
uint32_t native_fragment_readfirst(SpirvCompute&, const RegState&, const Rdna2Inst&, uint32_t);
}
} // namespace prosper::gpu
