// Current scalar generations may retain a vector producer's existing EXEC-scoped proof.
#pragma once

namespace prosper::gpu {
struct Rdna2Inst;
struct RegState;
struct ScalarSourceMarks;
struct SpirvCompute;

const ScalarSourceMarks* current_scalar_exec_scope(const RegState& rs, int reg);
bool scalar_scope_first_lane_agrees(const SpirvCompute& b, const RegState& rs,
                                    const Rdna2Inst& read);
void record_scalar_exec_scope(RegState& rs, const Rdna2Inst& in, int reg, bool ballot,
                              const ScalarSourceMarks& source, bool first_agrees = false);
} // namespace prosper::gpu
