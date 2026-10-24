// Bounded integer values only; this query never authenticates a register source.
#pragma once
#include <cstdint>
#include <optional>
#include <set>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
std::optional<std::set<uint32_t>> relative_index_values(const SpirvCompute& b, const RegState& rs,
                                                        uint32_t value, uint32_t exec);
} // namespace prosper::gpu
