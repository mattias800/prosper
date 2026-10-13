// DATA written throughout current EXEC also belongs to unchanged saved copies of that exact
// value. Equality never authenticates scalar words, a mask type, a resource or a wider parent.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct ActiveVectorWords;

struct SavedExecDataEquality {
    std::set<int> current_equal;
    std::map<int, std::set<int>> aliases;
    bool operator==(const SavedExecDataEquality&) const = default;
};

void meet_saved_exec_data(SavedExecDataEquality& into, const SavedExecDataEquality& incoming);
void advance_saved_exec_data(SavedExecDataEquality& equality, const Rdna2Inst& instruction,
                             const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                             int mask_write);
void refresh_equal_saved_exec_word(ActiveVectorWords& state, int word);
} // namespace prosper::gpu
