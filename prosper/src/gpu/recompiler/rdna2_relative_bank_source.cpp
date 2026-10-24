#include "gpu/recompiler/rdna2_relative_bank_source.hpp"
#include "gpu/recompiler/rdna2_cfg_local_index_waterfall.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_relative_index_values.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

namespace prosper::gpu {
namespace {
struct Packet {
    int index;
    uint32_t cohort;
};
std::optional<Packet> current_packet(const SpirvCompute& b, const RegState& rs,
                                     const Rdna2Inst& in) {
    if (!b.is_fragment || b.is_compute || b.is_fragment_packet() || b.ngg_workgroup_shell ||
        b.ngg_one_lane || b.wave_size != 64 || rs.vector_wave_size != 64 ||
        in.fmt != Rdna2Format::VOP1 || in.opcode != 0x43 || in.has_modifier || in.has_dpp ||
        in.has_sdwa || !scalar_scope_local_index_waterfall(b, in))
        return std::nullopt;
    auto instructions = b.saved_mask_liveness_instructions;
    if (instructions.empty())
        rdna2_walk(b.saved_mask_original_words.data(), b.saved_mask_original_words.size(),
                   instructions);
    const uint32_t read_pc = local_index_waterfall_read_pc(instructions, in);
    if (read_pc == UINT32_MAX) return std::nullopt;
    const uint32_t latch = local_index_waterfall_latch(instructions, read_pc);
    const auto read = std::find_if(instructions.begin(), instructions.end(),
                                   [&](const auto& item) { return item.pc == read_pc; });
    const auto branch = std::find_if(instructions.begin(), instructions.end(),
                                     [&](const auto& item) { return item.pc == latch; });
    if (read == instructions.end() || branch == instructions.end() ||
        branch + 1 == instructions.end())
        return std::nullopt;
    const auto saved = rs.sreg_bool.find((branch + 1)->src[0].value);
    if (saved == rs.sreg_bool.end() || !saved->second || !exec_within(rs, rs.exec, saved->second))
        return std::nullopt;
    return Packet{read->dst.value, saved->second};
}

bool same_index(const RegState& rs, int index, uint32_t cohort, uint32_t value) {
    const auto* original = current_scalar_exec_scope(rs, index);
    const auto* m0 = current_scalar_exec_scope(rs, 124);
    if (!original || !m0 || original->outside_exec != cohort || m0->outside_exec != rs.exec ||
        !vector_word_defined_in_exec(rs, *original, cohort) ||
        !vector_word_defined_in_exec(rs, *m0, rs.exec))
        return false;
    return rs.sreg.at(index) == value && rs.sreg.at(124) == value &&
           rs.sreg_memory_pattern.contains(index) == rs.sreg_memory_pattern.contains(124);
}
} // namespace

std::optional<RelativeBankSourceProof> prove_relative_bank_source(const SpirvCompute& b,
                                                                  const RegState& rs,
                                                                  const Rdna2Inst& in,
                                                                  RelativeBankRootReader roots) {
    if (!roots) return std::nullopt;
    const auto packet = current_packet(b, rs, in);
    const auto m0 = rs.sreg.find(124);
    if (!packet || m0 == rs.sreg.end() ||
        !same_index(rs, packet->index, packet->cohort, m0->second))
        return std::nullopt;
    const auto offsets = relative_index_values(b, rs, m0->second, packet->cohort);
    if (!offsets || offsets->empty() || in.src[0].value < 0) return std::nullopt;
    RelativeBankSourceProof proof;
    proof.source.memory = rs.sreg_memory_pattern.contains(124);
    for (int reg : {packet->index, 124}) {
        if (const auto* scope = current_scalar_exec_scope(rs, reg)) {
            const auto& required = roots(rs, *scope, reg == 124 ? rs.exec : packet->cohort);
            proof.source.blanket_roots.insert(required.begin(), required.end());
        }
    }
    for (uint32_t offset : *offsets) {
        const uint64_t selected = static_cast<uint64_t>(in.src[0].value) + offset;
        if (selected > 255) return std::nullopt;
        const int reg = static_cast<int>(selected);
        const auto value = rs.vreg.find(reg);
        const auto found = rs.vreg_source_marks.find(reg);
        const auto marks =
            found == rs.vreg_source_marks.end() ? ScalarSourceMarks{} : found->second;
        if (value == rs.vreg.end() || !value->second ||
            !vector_word_defined_in_exec(rs, marks, packet->cohort))
            return std::nullopt;
        proof.source.memory = proof.source.memory || marks.memory;
        const auto& required = roots(rs, marks, packet->cohort);
        proof.source.blanket_roots.insert(required.begin(), required.end());
    }
    proof.source.placeholder = !proof.source.blanket_roots.empty();
    // The closed packet executes once for each cohort lane's own index. Every selected word
    // was independently proved in that entire cohort; no new EXEC ancestry is published.
    proof.publication = {in.pc,
                         in.dst.value,
                         rs.exec,
                         packet->cohort,
                         m0->second,
                         packet->index,
                         rs.sreg_memory_pattern.contains(124)};
    return proof;
}

void publish_relative_bank_word(RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source,
                                const RelativeBankPublication& publication) {
    if (!publication.cohort || in.fmt != Rdna2Format::VOP1 || in.opcode != 0x43 ||
        in.has_modifier || in.has_dpp || in.has_sdwa || rs.vector_wave_size != 64 ||
        in.pc != publication.pc || in.dst.value != publication.destination ||
        rs.exec != publication.input_exec || source.hard ||
        publication.m0_memory != rs.sreg_memory_pattern.contains(124) ||
        !same_index(rs, publication.index_reg, publication.cohort, publication.index_value))
        return;
    const auto result = rs.vreg_source_marks.find(in.dst.value);
    const auto value = rs.vreg.find(in.dst.value);
    if (result == rs.vreg_source_marks.end() || !result->second.placeholder ||
        value == rs.vreg.end() || !value->second)
        return;
    // One owner field also retains the legacy hard-word coverage. Do not narrow an
    // existing wider or unrelated view merely to add this independent cohort view.
    if (result->second.outside_exec &&
        !exec_within(rs, result->second.outside_exec, publication.cohort))
        return;
    // The ordinary writer already retained old lanes, roots and MEMORY. This independent
    // complete-word view covers only the checked cohort, including after its EXEC restore.
    result->second.outside_exec = publication.cohort;
    result->second.scoped_complete_word = true;
    result->second.scoped_blanket_roots = source.blanket_roots;
}
} // namespace prosper::gpu
