#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <set>
#include <vector>
#include <cstdint>
#include "gpu/recompiler/rdna2_placeholder_origin_trace.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_legacy_vcc_writes.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include <algorithm>

namespace prosper::gpu::detail {
std::set<int> PlaceholderOriginTrace::image_addresses(const Rdna2Inst& in) const {
    if (program.observations) {
        const auto observed = program.observations->image_addresses.find(in.pc);
        if (observed != program.observations->image_addresses.end()) return observed->second;
    }
    std::set<int> words;
    for (uint32_t n = 0; n < rdna2_mimg_read_address_count(in); ++n) {
        const int reg = rdna2_mimg_address_vgpr(in, n);
        if (reg < 0) return {};
        words.insert(reg);
    }
    return words;
}

bool PlaceholderOriginTrace::writes(const Rdna2Inst& in, PlaceholderWord word) const {
    if (word.file == PlaceholderFile::Scc)
        return rdna2_writes_scc(in.fmt, in.opcode) == Rdna2Answer::Yes;
    if (word.file == PlaceholderFile::Spill) {
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361u)
            return in.dst.value == word.reg &&
                   (word.lane < 0 || in.src[1].kind != OperandKind::InlineInt ||
                    in.src[1].value == word.lane);
        return writes(in, {PlaceholderFile::Vector, word.reg});
    }
    bool found = false;
    if (word.file == PlaceholderFile::Vector) {
        if (!vector_write_leaves_old_word(in, program.native_interpolation))
            for_each_possible_vector_write(in, 255, [&](int reg) { found |= reg == word.reg; });
    } else {
        if ((word.reg == 106 || (word.reg == 107 && program.wave_size != 32)) &&
            legacy_valu_writes_vcc(in))
            found = true;
        for_each_scalar_write(
            in,
            [&](int base, uint32_t width) {
                found |= word.reg >= base && word.reg - base < static_cast<int>(width);
            },
            program.wave_size == 32);
    }
    return found;
}

std::vector<PlaceholderWord> PlaceholderOriginTrace::sources(const Rdna2Inst& in,
                                                             PlaceholderWord result) const {
    std::vector<PlaceholderWord> words;
    if (program.observations && program.observations->native_inputs.contains(in.pc)) return words;
    const bool genuine_resource = program.observations &&
                                  program.observations->genuine_resources.contains(in.pc) &&
                                  program.observations->genuine_resources.at(in.pc);
    const auto add = [&](PlaceholderFile file, int base, uint32_t width) {
        for (uint32_t n = 0; n < std::min(width, 256u); ++n)
            words.push_back({file, base + static_cast<int>(n)});
    };
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto& source = in.src[k];
        if (source.kind == OperandKind::VGPR) {
            const auto relative = rdna2_instruction_effects(in).m0_relative;
            if (k == 0 &&
                (relative == Rdna2M0Relative::Source || relative == Rdna2M0Relative::Both)) {
                const auto found = program.observations->relative_words.find(in.pc);
                if (found != program.observations->relative_words.end()) {
                    for (int reg : found->second) words.push_back({PlaceholderFile::Vector, reg});
                } else if (source.value >= 0 && source.value <= 255) {
                    // An out-of-range source can alias v0; retain that possible input too.
                    add(PlaceholderFile::Vector, 0, 1);
                    add(PlaceholderFile::Vector, source.value, 256u - source.value);
                }
            } else if (in.fmt == Rdna2Format::MIMG && k == 0) {
                for (int reg : image_addresses(in)) words.push_back({PlaceholderFile::Vector, reg});
            } else if (((in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360u) ||
                        (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02u)) &&
                       k == 0 && writelane_destinations.contains(source.value)) {
                const int lane =
                    in.fmt == Rdna2Format::VOP3 && in.src[1].kind == OperandKind::InlineInt
                        ? in.src[1].value
                        : -1;
                words.push_back({PlaceholderFile::Spill, source.value, lane});
            } else {
                add(PlaceholderFile::Vector, source.value, vector_provenance_source_span(in, k));
            }
        } else if (source.kind == OperandKind::SGPR ||
                   (source.kind == OperandKind::Special && source.value <= 127 &&
                    source.value != 125)) {
            if (genuine_resource &&
                ((in.fmt == Rdna2Format::MIMG && (k == 1 || k == 2)) ||
                 ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
                  (k == 1 || k == 2))))
                continue;
            auto width = scalar_alu_source_words(in, k, program.wave_size);
            if (width == UINT32_MAX) continue;
            // Legacy scalar descriptor accounting, shared with CFG merge-deadness. These
            // format fields are scalar reads even though the ALU-width query returns zero.
            if (in.fmt == Rdna2Format::MIMG && k == 1) width = in.mimg_r128 ? 4 : 8;
            if (in.fmt == Rdna2Format::MIMG && k == 2)
                // The existing resource check uses the legacy MIMG sampler boundary.
                width = in.opcode >= 0x20u ? 4 : 0;
            if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) && k == 1) width = 4;
            if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) && k == 2) width = 1;
            if (in.fmt == Rdna2Format::FLAT && k == 1) width = 2;
            add(PlaceholderFile::Scalar, source.value, width);
        } else if (source.kind == OperandKind::Special && source.value == 253) {
            words.push_back({PlaceholderFile::Scc});
        }
    }
    const auto& effects = rdna2_instruction_effects(in);
    if (effects.implicit_reads & kRdna2ImplicitM0) add(PlaceholderFile::Scalar, 124, 1);
    if (effects.implicit_reads & kRdna2ImplicitVcc)
        add(PlaceholderFile::Scalar, 106, program.wave_size == 32 ? 1 : 2);
    if (effects.implicit_reads & kRdna2ImplicitScc) words.push_back({PlaceholderFile::Scc});
    if (result.file == PlaceholderFile::Scalar)
        add(PlaceholderFile::Scalar, in.dst.value, scalar_implicit_destination_read_width(in));
    return words;
}

PlaceholderRoots PlaceholderOriginTrace::entry(PlaceholderWord word) const {
    if (word.file == PlaceholderFile::Spill) return {{PlaceholderOrigin::Spill, word}};
    if (word.file == PlaceholderFile::Scalar && program.observations &&
        program.observations->supplied_special_words.contains(word.reg))
        return {};
    if (word.file == PlaceholderFile::Vector) {
        if (word.reg < 0 || word.reg > 255) return {{PlaceholderOrigin::UnmodelledTransfer, word}};
        const auto launch = std::find_if(program.vectors.begin(), program.vectors.end(),
                                         [&](const auto& x) { return x.vgpr == word.reg; });
        if (launch != program.vectors.end() && launch->state != LaunchVgprState::Reserved &&
            (launch->state == LaunchVgprState::Partial ||
             !program.supplied_vectors.contains(word.reg)))
            return {{PlaceholderOrigin::LaunchInput, word}};
        if (program.supplied_vectors.contains(word.reg)) return {};
        return {{program.vector_profile_known ? PlaceholderOrigin::Unloaded
                                              : PlaceholderOrigin::UnknownLaunch,
                 word}};
    }
    if (word.file == PlaceholderFile::Scalar && program.supplied_scalars.contains(word.reg))
        return {};
    if (word.file == PlaceholderFile::Scalar && program.loaded_scalars.contains(word.reg))
        return {{PlaceholderOrigin::LaunchInput, word}};
    // The compute request records the enabled user/system prefix, but not SCRATCH_EN.
    // Hardware can append one scratch-offset SGPR (RDNA2 ISA 3.12.2); that word remains
    // unknown. The prefix also says nothing about special-register initialization.
    const int possible_scratch =
        program.loaded_scalars.empty() ? 0 : *program.loaded_scalars.rbegin() + 1;
    const bool unloaded = program.scalar_profile_known && word.file == PlaceholderFile::Scalar &&
                          word.reg >= 0 && word.reg <= 105 && word.reg != possible_scratch;
    return {{unloaded ? PlaceholderOrigin::Unloaded : PlaceholderOrigin::UnknownLaunch, word}};
}
}   // namespace prosper::gpu::detail
