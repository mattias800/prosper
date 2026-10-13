// Current SCC data provenance for Wave32 and the scoped Wave64 fragment scalar-VCC admission.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
namespace prosper::gpu {
struct RegState;
struct SpirvCompute;
struct ScalarSourceMarks;
bool scalar_instruction_writes_scc(const Rdna2Inst& in);
bool scalar_instruction_preserves_scc(const Rdna2Inst& in);
void record_wave32_scalar_condition(RegState& rs, const Rdna2Inst& in,
                                    const ScalarSourceMarks& source);
bool scalar_condition_source_defined(const SpirvCompute& b, const RegState& rs);
bool scalar_condition_defined(SpirvCompute& b, const RegState& rs, uint32_t pc);
} // namespace prosper::gpu
