// Provenance of ordinary vector words relayed into scalar lane-mask consumers (#4787).
// Internal compiler facts only; these marks grant no entry/register or memory authority.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {

struct RegState;
struct SpirvCompute;

struct ScalarSourceMarks {
    bool placeholder = false;
    bool memory = false;
    bool hard = false;
    std::set<uint64_t> blanket_roots;
    // Nonzero only for a hard placeholder whose lanes are provably confined to the lanes OUTSIDE
    // this SPIR-V EXEC value: a partial VALU write under that EXEC replaced every other lane with
    // clean data. A later reader under the very same EXEC value never touches those lanes.
    uint32_t outside_exec = 0;
};
using VectorWordMarks = std::map<int, ScalarSourceMarks>;

void union_source_marks(ScalarSourceMarks& into, const ScalarSourceMarks& other);
// Call only at a producing stage entry, after its actual input VGPRs have been published.
void seed_vector_word_definedness(RegState& rs, const std::vector<Rdna2Inst>& instructions,
                                  bool native_interpolation = false, uint32_t wave_size = 64);
ScalarSourceMarks vector_source_marks(const RegState& rs, const Rdna2Inst& in);
ScalarSourceMarks vector_relative_source_marks(const SpirvCompute& b, const RegState& rs,
                                               const Rdna2Inst& in);
ScalarSourceMarks vector_lane_read_marks(const RegState& rs, const Rdna2Inst& in);
// A fresh comparison/carry predicate cannot authenticate fabricated input storage. Genuine memory
// data and checked loop-header assumptions remain separate from unavailable-source placeholders.
bool fresh_mask_sources_defined(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in);
void record_vector_word_marks(RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source);
void join_vector_word_marks(VectorWordMarks& into, const VectorWordMarks& other);
void mark_vector_loop_carried(RegState& rs, const std::vector<Rdna2Inst>& ins, uint32_t lo,
                              uint32_t hi, std::map<int, std::set<uint64_t>>& carried);
// Loop roots use physical scalar numbers 0..253 and disjoint vector numbers 256..511.
inline constexpr int kVectorBlanketBase = 256;
ScalarSourceMarks carried_word_marks(const RegState& rs, int word);

} // namespace prosper::gpu
