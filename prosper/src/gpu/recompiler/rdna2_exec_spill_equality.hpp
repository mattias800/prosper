// Value equalities to the CURRENT architectural EXEC words. These facts grant only that a
// complete restore leaves EXEC unchanged; mask typing and physical-word validity stay separate.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace prosper::gpu {
enum class ExecSpillWord { Low, High };
struct ExecSpillEquality {
    std::map<std::pair<int, int>, ExecSpillWord> slots;
    std::map<int, ExecSpillWord> scalars;
    bool operator==(const ExecSpillEquality&) const = default;
};
void meet_exec_spill_equality(ExecSpillEquality& into, const ExecSpillEquality& incoming);
// Snapshot sources before writes. True only for an actual full B64-to-EXEC move whose two
// source words each still equal CURRENT EXEC. Every other EXEC transition ends all old facts.
bool advance_exec_spill_equality(ExecSpillEquality& equality, const Rdna2Inst& instruction,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write);
} // namespace prosper::gpu
