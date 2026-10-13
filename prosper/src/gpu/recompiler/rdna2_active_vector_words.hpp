// CFG MUST facts for vector words owned by the current guest EXEC. These are lane-wise DATA
// guarantees, not whole-vector initialization, a mask-type certificate or memory authority.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_complementary_exec_data.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {

struct RegState;
struct SpirvCompute;

struct ActiveVectorWords {
    bool enabled = false;
    std::set<int> current;
    // Immutable producer ENTRY EXEC, not a full-wave or raw-launch certificate. Only actual
    // copies/logic prove equality or a subset; partial scalar overwrites retire pair lifetimes.
    bool current_is_entry = false, current_within_entry = false;
    std::set<int> entry_words, entry_masks, within_entry_masks;
    // A vector word is genuine wherever the CURRENT value of this saved B64 mask enables a lane.
    // Overwriting a vector with a possibly missing source retires it from every guarantee.
    std::map<int, std::set<int>> masks;
    ComplementaryExecData complement;
    bool operator==(const ActiveVectorWords&) const = default;
};

bool tracks_active_vector_words(const SpirvCompute& builder, bool barrier_half_phase);
void seed_active_vector_words(ActiveVectorWords& facts, const RegState& initial,
                              const std::set<int>& relayed_vectors);
void meet_active_vector_words(ActiveVectorWords& into, const ActiveVectorWords& incoming);
// Re-anchor only an existing hard vector mark to the newly loaded actual EXEC; other provenance
// (scalar relays, spill slots, memory patterns and loop roots) retains its own obligations.
void restore_active_vector_words(const std::set<int>& current, RegState& state);
void restore_active_vector_instruction(const std::map<uint32_t, std::set<int>>& after,
                                       const Rdna2Inst& in, RegState& state);

// Distinguish a current lane's ordinary inputs from peer-lane/slot reads. Scalar DATA must retain
// its independent MUST ownership; membership here never turns a Bool mask into physical DATA.
bool active_vector_sources_defined(const ActiveVectorWords& state, const Rdna2Inst& in,
                                   const std::set<int>& relayed_vectors, const std::set<int>& masks,
                                   const std::set<int>& scalar_words,
                                   const std::set<int>& relayed_words, bool definite_scc);

// Source validity is captured before any scalar/mask destination changes. A completed active
// write has already passed the actual lowering; this fact never authorizes an absent resource.
void advance_active_vector_words(ActiveVectorWords& state, const Rdna2Inst& in,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write,
                                 bool genuine_active_sources, bool complete_active_word);

}   // namespace prosper::gpu
