// MIMG D16 value packing; callers own every image access, refusal and masked destination write.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using ImageD16Writer = std::function<void(uint32_t)>;
void lower_image_d16_results(SpirvCompute& b, const Rdna2Inst& in, const uint32_t out[4],
                             const ImageD16Writer& write_packed);
uint32_t lower_image_d16_gather_word(SpirvCompute& b, const uint32_t out[4], uint32_t word);
} // namespace prosper::gpu::ir
