// ADR 0038: retain old successful modules; retry refusals with explicitly unloaded roots.
#pragma once
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace prosper::gpu {
struct RecompileDiagnosticContext;
struct Rdna2Inst;
struct RegState;
struct Operand;
enum class ArbitraryPlaceholderRead { FreshMask, LoopCarry };
struct ArbitraryPlaceholderReads {
    uint64_t fresh_mask = 0, loop_carry = 0;
    uint64_t total() const { return fresh_mask + loop_carry; }
};
bool placeholder_roots_are_unloaded(const PlaceholderRoots&);
void append_unproved_loop_reads(PlaceholderRoots&, const SpirvCompute&, uint32_t header,
                                const std::vector<int>& words);
bool arbitrary_placeholder_attempt_active();
bool arbitrary_placeholder_retry_active();
bool refuse_arbitrary_scalar_publication(SpirvCompute&, const RegState&, const Rdna2Inst&);
bool refuse_arbitrary_placeholder_operand(SpirvCompute&, const RegState&, const Rdna2Inst&,
                                          const Operand&, bool* ok);
bool refuse_arbitrary_placeholder_export(SpirvCompute&, const RegState&, const Rdna2Inst&,
                                         bool& ok);
bool refuse_arbitrary_placeholder_old_destination(SpirvCompute&, const RegState&, const Rdna2Inst&,
                                                  bool& ok);
std::vector<uint32_t>
recompile_with_arbitrary_placeholders(const RecompileDiagnosticContext&,
                                      const std::function<std::vector<uint32_t>()>&);
bool admit_arbitrary_placeholder_read(SpirvCompute&, uint32_t pc, ArbitraryPlaceholderRead,
                                      const PlaceholderRoots&);
bool refuse_arbitrary_placeholder_lane_read(SpirvCompute&, const RegState&, const Rdna2Inst&,
                                            bool& ok);
uint32_t read_vector_for_arbitrary_retry(SpirvCompute&, RegState&, const Rdna2Inst&, int, bool&);
uint32_t read_mimg_source_for_arbitrary_retry(SpirvCompute&, RegState&, const Rdna2Inst&, int,
                                              bool&);
// Semantically inert metadata identifies only reads in the selected module, not failed attempts.
ArbitraryPlaceholderReads arbitrary_placeholder_reads(std::span<const uint32_t> module);
} // namespace prosper::gpu
