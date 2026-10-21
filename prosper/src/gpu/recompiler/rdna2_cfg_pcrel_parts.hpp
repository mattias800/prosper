// Private format passes for the original PC-relative table proof.
#pragma once
#include "gpu/recompiler/rdna2_cfg_pcrel_tables.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
void cfg_pcrel_scalar_unary(
    const std::function<void(int)>& kill, const prosper::gpu::Rdna2Inst& in,
    std::unordered_map<int, uint64_t>& pcoff, std::unordered_set<int>& pchi,
    std::unordered_map<int, uint32_t>& fact_pc,
    const std::function<bool(const Operand&, const Rdna2Inst&, uint32_t*)>& imm_of,
    std::unordered_map<int, uint32_t>& kconst);
void cfg_pcrel_scalar_binary(
    const prosper::gpu::Rdna2Inst& in, std::unordered_map<int, uint64_t>& pcoff,
    const std::function<bool(const Operand&, const Rdna2Inst&, uint32_t*)>& imm_of,
    const std::function<void(int)>& kill, std::unordered_map<int, uint32_t>& fact_pc,
    std::unordered_set<int>& pchi);
void cfg_pcrel_scalar_immediate(const prosper::gpu::Rdna2Inst& in,
                                const std::function<void(int)>& kill,
                                std::unordered_map<int, uint32_t>& kconst,
                                std::unordered_map<int, uint32_t>& fact_pc);
void cfg_pcrel_scalar_load(const prosper::gpu::Rdna2Inst& in,
                           std::unordered_map<int, uint64_t>& pcoff, std::unordered_set<int>& pchi,
                           std::unordered_map<int, uint32_t>& kconst,
                           std::unordered_map<int, uint32_t>& fact_pc,
                           std::vector<uint32_t>& br_targets, size_t& dwords,
                           size_t*& required_dwords, PcrelTables& out, const uint32_t*& code,
                           const std::function<void(int)>& kill);
void cfg_pcrel_vector_writes(const prosper::gpu::Rdna2Inst& in,
                             const std::function<void(int)>& kill);
void cfg_pcrel_typed_load(const prosper::gpu::Rdna2Inst& in,
                          std::unordered_map<int, uint64_t>& pcoff, std::unordered_set<int>& pchi,
                          std::unordered_map<int, uint32_t>& kconst,
                          std::unordered_map<int, uint32_t>& fact_pc,
                          std::vector<uint32_t>& br_targets, size_t& dwords,
                          size_t*& required_dwords, PcrelTables& out, const uint32_t*& code);
void cfg_pcrel_raw_load(const prosper::gpu::Rdna2Inst& in, std::unordered_map<int, uint64_t>& pcoff,
                        std::unordered_set<int>& pchi, std::unordered_map<int, uint32_t>& kconst,
                        std::unordered_map<int, uint32_t>& fact_pc,
                        std::vector<uint32_t>& br_targets, size_t& dwords, size_t*& required_dwords,
                        PcrelTables& out, const uint32_t*& code);
}  // namespace prosper::gpu
