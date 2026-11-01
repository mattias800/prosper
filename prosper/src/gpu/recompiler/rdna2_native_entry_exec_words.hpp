#pragma once
// Numerical words of an authenticated original entry EXEC, independently of later ballots.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
struct ComputeDispatchLaunchProfile;
struct NativeEntryExecWordPlan;
bool native_entry_exec_word_move(const SpirvCompute&, const Rdna2Inst&);
void initialize_native_entry_exec_words(SpirvCompute&, const ComputeDispatchLaunchProfile&,
                                        const std::vector<Rdna2Inst>&, const uint32_t*, size_t);
void constrain_native_entry_exec_words_region(SpirvCompute&, const std::vector<Rdna2Inst>&);
bool native_entry_exec_word_proven(const SpirvCompute&, const Rdna2Inst&);
uint32_t native_compute_vcc_data_mask(SpirvCompute&, const RegState&);
std::optional<std::array<uint32_t, 2>> native_entry_exec_words(SpirvCompute&, const Rdna2Inst&);
} // namespace prosper::gpu
