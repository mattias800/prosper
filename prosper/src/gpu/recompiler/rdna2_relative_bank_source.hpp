// Current-cohort result proof for an authenticated, closed local-index packet.
#pragma once
#include "gpu/recompiler/rdna2_reg_state.hpp"

#include <cstdint>
#include <optional>
#include <set>

namespace prosper::gpu {
struct Rdna2Inst;
struct SpirvCompute;

struct RelativeBankPublication {
    uint32_t pc = 0;
    int destination = -1;
    uint32_t input_exec = 0;
    uint32_t cohort = 0;
    uint32_t index_value = 0;
    int index_reg = -1;
    bool m0_memory = false;
};
struct RelativeBankSourceProof {
    ScalarSourceMarks source;
    RelativeBankPublication publication;
};
using RelativeBankRootReader = const std::set<uint64_t>& (*)(const RegState&,
                                                             const ScalarSourceMarks&, uint32_t);

std::optional<RelativeBankSourceProof> prove_relative_bank_source(const SpirvCompute& b,
                                                                  const RegState& rs,
                                                                  const Rdna2Inst& in,
                                                                  RelativeBankRootReader roots);
// The vector-word owner supplies its canonical root reader.
std::optional<RelativeBankSourceProof>
relative_bank_source_proof(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in);
void publish_relative_bank_word(RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source,
                                const RelativeBankPublication& publication);
} // namespace prosper::gpu
