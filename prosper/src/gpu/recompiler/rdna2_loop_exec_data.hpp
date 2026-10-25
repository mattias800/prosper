// Code-only native Wave64 DATA bounds; never a loop-route certificate.
#pragma once
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
struct DivLoop;
bool native_loop_data_exec_bound(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                 const DivLoop& loop, const RegState& entry);
} // namespace prosper::gpu
