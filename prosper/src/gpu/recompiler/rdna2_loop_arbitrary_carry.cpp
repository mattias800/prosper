// See rdna2_loop_arbitrary_carry.hpp.
#include "gpu/recompiler/rdna2_loop_arbitrary_carry.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <algorithm>
#include <cstdint>

namespace prosper::gpu {
namespace {

bool any_in(const std::set<uint64_t>& roots, const std::set<uint64_t>& set) {
    return std::any_of(roots.begin(), roots.end(),
                       [&](uint64_t root) { return set.contains(root); });
}

}   // namespace

LoopRootVerdicts loop_root_verdicts(const SpirvCompute& b, const RegState& rs, uint32_t header_pc,
                                    const std::map<int, std::set<uint64_t>>& carried,
                                    const std::map<int, uint32_t>& confined,
                                    const std::set<int>& arbitrary, bool weak_allowed) {
    LoopRootVerdicts verdict;
    for (bool grew = true; grew;) {
        grew = false;
        for (const auto& entry : carried) {
            const int word = entry.first;
            const uint64_t root = blanket_root(header_pc, word);
            if (verdict.violated.contains(root)) continue;
            if (arbitrary_outside_root(root)) {
                // CONFIDENCE: MED -- a confined arbitrary header word's other lanes: the back edge
                // keeps them arbitrary unless it brings in a hard lane (#5193).
                const auto back = carried_word_marks(rs, word - kArbitraryOutsideRootOffset);
                if (back.placeholder && ((back.hard && !back.arbitrary) ||
                                         any_in(back.blanket_roots, verdict.violated)))
                    grew = verdict.violated.insert(root).second || grew;
                continue;
            }
            const auto back = carried_word_marks(rs, word);
            if (!back.placeholder) continue;
            const auto owner = confined.find(word);
            const bool defined =
                !back.hard ||
                (owner != confined.end() && vector_word_defined_in_exec(rs, back, owner->second));
            const bool arbitrary_back = word >= kVectorBlanketBase && back.hard && back.arbitrary;
            bool violated = any_in(back.blanket_roots, verdict.violated);
            bool weak = false;
            if (!violated && arbitrary.contains(word)) {
                // CONFIDENCE: MED -- the header already promised only "defined or arbitrary"
                // (#5193): a defined or arbitrary back edge keeps it, a hard lane breaks it.
                violated = !defined && !arbitrary_back;
            } else if (!violated) {
                // CONFIDENCE: MED -- the header promised DEFINED. An arbitrary back edge, or one
                // defined only under a weakly held root, keeps the weaker promise every read that
                // needed no more than "no hard lane" relied on (fresh predicates, bounded reads).
                weak = defined ? any_in(back.blanket_roots, verdict.weak) : arbitrary_back;
                violated = !defined && !arbitrary_back;
                // A scalar word holds no arbitrary value, and a strong read (an unbounded or
                // writing address, a vector value read into a scalar) needed the definedness.
                if (weak && (word < kVectorBlanketBase || !weak_allowed ||
                             b.loop_blanket_roots_strong.contains(root))) {
                    weak = false;
                    violated = true;
                }
            }
            if (violated) {
                verdict.weak.erase(root);
                grew = verdict.violated.insert(root).second || grew;
            } else if (weak) {
                grew = verdict.weak.insert(root).second || grew;
            }
        }
    }
    return verdict;
}

void strip_arbitrary_outside_roots(ScalarSourceMarks& marks, const std::set<uint64_t>& violated) {
    bool broken = false;
    const auto drop = [&](uint64_t root) {
        const bool hit = arbitrary_outside_root(root) && violated.contains(root);
        broken = broken || hit;
        return hit;
    };
    std::erase_if(marks.blanket_roots, drop);
    std::erase_if(marks.scoped_blanket_roots, drop);
    if (!broken) return;
    // The lanes outside an owner may be hard after all; the owner's own proof is untouched.
    marks.arbitrary = false;
    if (marks.hard && !marks.outside_exec) marks.blanket_roots.clear();
}

void weaken_vector_word_roots(ScalarSourceMarks& marks, const std::set<uint64_t>& weak) {
    if (!marks.placeholder || !any_in(marks.blanket_roots, weak)) return;
    if (marks.hard && !marks.outside_exec) return;   // already hard, or arbitrary with obligations
    // CONFIDENCE: MED -- a soft mark's lanes were defined under the roots, now defined or
    // arbitrary; a confined hard mark's active domain relied on them the same way, which its
    // confined form cannot express, so it loses the owner (its outside lanes decide arbitrary).
    // Clean data loaded at an address under such a root keeps its mark: the address read was
    // bounded (note_memory_address made every other read strong), so the value is one hardware
    // could have loaded.
    const bool arbitrary = !marks.hard || marks.arbitrary;
    marks.hard = true;
    marks.arbitrary = arbitrary;
    marks.outside_exec = 0;
    marks.scoped_complete_word = false;
    marks.blanket_roots.insert(marks.scoped_blanket_roots.begin(),
                               marks.scoped_blanket_roots.end());
    marks.scoped_blanket_roots.clear();
    if (!arbitrary) marks.blanket_roots.clear();
}

bool loop_may_hold_weakly(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi) {
    constexpr uint32_t kVop1ReadFirstLane = 0x02;
    constexpr uint32_t kVop3ReadFirstLane = 0x182;
    constexpr uint32_t kVop3ReadLane = 0x360;
    return std::none_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return in.pc >= lo && in.pc < hi &&
               ((in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1ReadFirstLane) ||
                (in.fmt == Rdna2Format::VOP3 &&
                 (in.opcode == kVop3ReadFirstLane || in.opcode == kVop3ReadLane)));
    });
}

std::set<int> retry_gate(SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                         const LoopExecSubset& proof) {
    std::set<int> reads;
    if (!b.is_compute || proof.admitted() || loop.bottom_tested) return reads;
    // CONFIDENCE: HIGH (conservative) -- keeps the existing refusal that selects the retry.
    b.strict_loop_condition = {loop.header_pc, loop.exit_branch_pc};
    for (const auto& in : ins) {
        if (in.pc < loop.header_pc || in.pc >= loop.exit_branch_pc) continue;
        for (uint32_t source = 0; source < in.n_src; ++source)
            for (uint32_t word = 0; word < vector_provenance_source_span(in, source); ++word)
                reads.insert(in.src[source].value + static_cast<int>(word));
    }
    return reads;
}

}   // namespace prosper::gpu
