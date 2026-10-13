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
    // Loop-header assumptions, including the per-iteration validity of an EXEC-confined word.
    // A hard word remains unavailable outside its owner even when these assumptions hold.
    std::set<uint64_t> blanket_roots;
    // A complete writer proved its active lanes under this immutable EXEC value. Hard marks have
    // missing lanes only outside it; soft marks also retain their global conditional proof above.
    uint32_t outside_exec = 0;
    // Only an audited complete VALU writer publishes the independent soft-mark domain. An old
    // outside_exec value on a memory result is not that arithmetic publication.
    bool scoped_complete_word = false;
    // For a soft mark, the owner's active domain depends only on these source roots, while its
    // global blanket_roots may additionally depend on the inactive old generation. Hard marks
    // continue to carry their active-domain obligations in blanket_roots.
    std::set<uint64_t> scoped_blanket_roots;
};
using VectorWordMarks = std::map<int, ScalarSourceMarks>;

// The successful MIMG lowering's actual active-lane reads, before any destination replacement.
// This authenticates only its new confined-word proof; it does not change image admission policy.
struct MimgSourceProof {
    uint32_t pc = UINT32_MAX;
    bool enabled = false;
    bool defined = false;
    std::set<int> words;
    std::set<uint64_t> roots;
};

// Pure write effect, shared by structured provenance and current-EXEC CFG DATA facts.
bool vector_write_completes_active_word(const Rdna2Inst& in);
void union_source_marks(ScalarSourceMarks& into, const ScalarSourceMarks& other);
// Called only after a genuine Bool-domain logical producer has emitted its result.
void record_exec_mask_subset(RegState& rs, uint32_t result, uint32_t left, uint32_t right,
                             uint32_t opcode, uint32_t previous_exec);
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
void retire_vector_word_roots(ScalarSourceMarks& marks, const std::set<uint64_t>& violated);
bool vector_word_defined_in_exec(const RegState& rs, const ScalarSourceMarks& marks, uint32_t exec);
void begin_mimg_source_proof(RegState& rs, uint32_t pc);
void record_mimg_source_word(RegState& rs, int reg);
void record_mimg_address_operand(RegState& rs, const Rdna2Inst& in, uint32_t index);
uint32_t read_mimg_source_word(SpirvCompute& b, RegState& rs, int reg);
int mimg_address_vgpr(RegState& rs, const Rdna2Inst& in, uint32_t index);
void mark_vector_loop_carried(RegState& rs, const std::vector<Rdna2Inst>& ins, uint32_t lo,
                              uint32_t hi, std::map<int, std::set<uint64_t>>& carried,
                              std::map<int, uint32_t>& confined);
// Loop roots use physical scalar numbers 0..253 and disjoint vector numbers 256..511.
inline constexpr int kVectorBlanketBase = 256;
ScalarSourceMarks carried_word_marks(const RegState& rs, int word);

}   // namespace prosper::gpu
