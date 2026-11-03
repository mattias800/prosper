// SOPP instruction routes borrowed from the ALU emitter.
#pragma once
#include <cstdint>
#include <unordered_set>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
bool sopp_drops_scc_branch(const Rdna2Inst& in, const std::unordered_set<uint32_t>* safe_execz);
struct SoppCaseRoutes {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    bool& ok;
    const std::unordered_set<uint32_t>*& safe_execz;
    bool lower();
};
}   // namespace ir
}   // namespace prosper::gpu
