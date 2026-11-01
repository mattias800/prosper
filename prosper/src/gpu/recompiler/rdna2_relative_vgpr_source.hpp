// A declared private VGPR allocation bounds runtime MOVRELS reads, never their values.
#pragma once
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
int allocated_vgpr_max(const std::vector<Rdna2Inst>&, uint32_t allocation);
void add_allocated_vgprs(std::set<int>& registers, uint32_t allocation);
void add_allocated_relative_write(const Rdna2Inst&, uint32_t allocation, std::set<int>& writes,
                                  std::set<int>& reads);
bool allocated_vgpr_program_fits(const uint32_t* code, uint32_t words, uint32_t allocation);
bool allocated_relative_write_needs_dispatcher(const std::vector<Rdna2Inst>& instructions);
bool compute_vgpr_allocation_is_canonical(uint32_t words, uint32_t rsrc1, uint32_t wave_size);
std::optional<std::set<int>> relative_vgpr_sources(const SpirvCompute&, const RegState&,
                                                   const Rdna2Inst&);
// Returns false when the caller has no allocation; the legacy lowering owns that case.
bool lower_allocated_movrels(SpirvCompute&, RegState&, const Rdna2Inst&, uint32_t& result);
} // namespace prosper::gpu
