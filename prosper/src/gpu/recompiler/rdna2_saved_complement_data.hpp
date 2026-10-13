// Exact complementary DATA survives an unchanged B64 copy, independently of nested CURRENT
// EXEC. Parent aliases and DATA obligations remain MUST facts of their current generations.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct ActiveVectorWords;
struct ComplementaryExecData;

struct SavedComplementData {
    std::set<int> parents, first_words;
    bool operator==(const SavedComplementData&) const = default;
};
using SavedComplementValues = std::map<int, SavedComplementData>;

struct SavedComplementTransfer {
    int destination = -1;
    std::optional<SavedComplementData> captured, restored;
};

void meet_saved_complement_data(SavedComplementValues& into, const SavedComplementValues& incoming);
SavedComplementTransfer
prepare_saved_complement_data(ActiveVectorWords& state, const Rdna2Inst& instruction,
                              const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                              int mask_write);
void complete_saved_complement_data(ActiveVectorWords& state, SavedComplementTransfer transfer);
void retire_saved_complement_word(ComplementaryExecData& proof, int word);
} // namespace prosper::gpu
