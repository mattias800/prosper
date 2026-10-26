// DATA ancestry for a uniform-counter EXEC loop; never route admission.
#pragma once
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
struct DivLoop;
bool uniform_counter_loop_data_boundary(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                        const DivLoop& loop, const RegState& entry);
} // namespace prosper::gpu
