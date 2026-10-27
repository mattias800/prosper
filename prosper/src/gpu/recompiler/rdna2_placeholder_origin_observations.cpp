// Existing native Input and resource publications terminate a diagnostic dependency slice.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <optional>
#include <cstdint>
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_relative_index_values.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
namespace {
const PlaceholderOriginProgram* program(const SpirvCompute& b, const RegState& rs) {
    return rs.placeholder_origin_program ? rs.placeholder_origin_program.get()
                                         : b.placeholder_origin_program.get();
}
void relative_words(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                    PlaceholderOriginObservations& observations) {
    const auto relative = rdna2_instruction_effects(in).m0_relative;
    if ((relative != Rdna2M0Relative::Source && relative != Rdna2M0Relative::Both) || !in.n_src ||
        in.src[0].kind != OperandKind::VGPR)
        return;
    const auto m0 = rs.sreg.find(124);
    const auto* scope = current_scalar_exec_scope(rs, 124);
    const bool genuine = !sreg_word_may_be_fabricated(rs, 124) ||
                         (scope && vector_word_defined_in_exec(rs, *scope, rs.exec));
    const auto values = m0 != rs.sreg.end() && genuine
                            ? relative_index_values(b, rs, m0->second, rs.exec)
                            : std::nullopt;
    if (!values || values->empty()) {
        observations.relative_words.erase(in.pc);
        observations.unbounded_relative.insert(in.pc);
        return;
    }
    if (observations.unbounded_relative.contains(in.pc)) return;
    for (auto offset : *values) {
        const uint64_t reg = static_cast<uint64_t>(in.src[0].value) + offset;
        if (reg > 255) {
            observations.relative_words.erase(in.pc);
            observations.unbounded_relative.insert(in.pc);
            return;
        }
        observations.relative_words[in.pc].insert(static_cast<int>(reg));
    }
}
} // namespace
void note_placeholder_resources(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    // A dispatcher reload reconstructs value state; retain only this diagnostic input handle.
    if (!rs.placeholder_origin_program)
        rs.placeholder_origin_program = b.placeholder_origin_program;
    const auto* p = program(b, rs);
    if (p && p->observations && !p->observations->native_entry_checked) {
        p->observations->native_entry_checked = true;
        // This observes the existing complete-entry theorem, never publishes mask storage.
        if (native_compute_has_complete_entry(b))
            p->observations->supplied_special_words.insert({126, 127});
    }
    if (p && p->observations) relative_words(b, rs, in, *p->observations);
    if (!p || !p->observations ||
        (in.fmt != Rdna2Format::MIMG && in.fmt != Rdna2Format::MUBUF &&
         in.fmt != Rdna2Format::MTBUF))
        return;
    const bool genuine = arbitrary_read_resource_genuine(rs, in);
    auto [at, inserted] = p->observations->genuine_resources.emplace(in.pc, genuine);
    if (!inserted) at->second = at->second && genuine;
}

void note_placeholder_native_input(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    const auto* p = program(b, rs);
    if (p && p->observations) p->observations->native_inputs.insert(in.pc);
}

void note_placeholder_image_addresses(const RegState& rs, const Rdna2Inst& in) {
    const auto* p = rs.placeholder_origin_program.get();
    if (!p || !p->observations || in.fmt != Rdna2Format::MIMG || rs.mimg_sources.pc != in.pc ||
        !rs.mimg_sources.enabled || rs.mimg_sources.words.empty())
        return;
    auto& words = p->observations->image_addresses[in.pc];
    words.insert(rs.mimg_sources.words.begin(), rs.mimg_sources.words.end());
}
} // namespace prosper::gpu
