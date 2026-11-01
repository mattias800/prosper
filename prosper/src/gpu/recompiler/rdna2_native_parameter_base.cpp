#include "gpu/recompiler/rdna2_native_parameter_base.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <algorithm>
#include <optional>
#include <set>

namespace prosper::gpu {
struct NativeParameterBasePlan {
    Rdna2Inst load;
    std::vector<Rdna2Inst> original;
};

std::optional<uint32_t> pixel_parameter_sgpr(const PixelParameterLaunch& launch) {
    if (!launch.canonical() || !launch.available) return {};
    namespace P = prosper::agc::Pm4;
    const uint32_t count = ((launch.value >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) &
                            P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MASK) |
                           (((launch.value >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT) &
                             P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_MASK)
                            << 5u);
    // AMD RDNA2 ISA 70648 section 3.12.2: 0..32 user words, then the automatic PS word.
    // Later optional system words do not change this first system word's location.
    return count <= 32 ? std::optional<uint32_t>{count} : std::nullopt;
}

namespace {
bool native_fragment(const SpirvCompute& b) {
    return b.is_fragment && !b.is_compute && !b.is_vertex && !b.ngg_workgroup_shell &&
           !b.ngg_one_lane && b.guest_stage == GuestShaderStage::Fragment &&
           b.physical_domain == PhysicalExecutionDomain::Invocation;
}

bool overlaps(int base, uint32_t width, int word) {
    return width && base <= word && word - base < static_cast<int>(width);
}

bool reads_word(const Rdna2Inst& in, int word) {
    // ISA 70648 section 12.2: a non-null SDST supplies part of the wait threshold.
    // These SOPK forms have no explicit source operand in the decoded array.
    if (in.fmt == Rdna2Format::SOPK && in.opcode >= kSopkOpcodeWaitcntVscnt &&
        in.opcode <= kSopkOpcodeWaitcntLgkmcnt && in.dst.value == word)
        return true;
    if (overlaps(in.dst.value, scalar_implicit_destination_read_width(in), word)) return true;
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto source = in.src[k];
        if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
            overlaps(source.value, scalar_alu_source_words(in, k), word))
            return true;
    }
    return false;
}

bool writes_word(const Rdna2Inst& in, int word) {
    bool found = false;
    for_each_scalar_write(in,
                          [&](int base, uint32_t width) { found |= overlaps(base, width, word); });
    return found;
}

bool parameter_load(const Rdna2Inst& in, int word) {
    return !in.synthetic_terminator && in.fmt == Rdna2Format::SOP1 &&
           in.opcode == kSop1OpcodeMovB32 && in.dst.value == 124 &&
           in.src[0].kind == OperandKind::SGPR && in.src[0].value == word;
}

bool replaces_entry_word(const Rdna2Inst& in, int word) {
    // These scalar moves overwrite the addressed word even under empty EXEC. A generic write
    // inventory also includes conditional/partial writes, which cannot establish this lifetime.
    return in.fmt == Rdna2Format::SOP1 &&
           (in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64) &&
           overlaps(in.dst.value, in.opcode == kSop1OpcodeMovB64 ? 2u : 1u, word);
}

bool closed_control(const std::vector<Rdna2Inst>& instructions) {
    if (instructions.empty() || instructions.front().pc != 0 || !instructions.back().is_end ||
        !std::is_sorted(instructions.begin(), instructions.end(),
                        [](const auto& a, const auto& b) { return a.pc < b.pc; }))
        return false;
    std::set<uint32_t> pcs;
    for (const auto& in : instructions)
        if (in.synthetic_terminator || !in.len_dwords || !pcs.insert(in.pc).second) return false;
    for (const auto& in : instructions) {
        const auto& effects = rdna2_instruction_effects(in);
        if (effects.status == Rdna2EffectsStatus::Unknown ||
            effects.status == Rdna2EffectsStatus::Unmodeled ||
            effects.status == Rdna2EffectsStatus::Reserved ||
            effects.control >= Rdna2ControlFlow::Trap)
            return false;
        if (in.fmt != Rdna2Format::SOPP || !sopp_opcode_is_direct_branch(in.opcode)) continue;
        const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
        // A tail outside this original instruction inventory has no M0-use proof here.
        if (target < 0 || target > UINT32_MAX || !pcs.contains(static_cast<uint32_t>(target)))
            return false;
    }
    return true;
}

bool unsupported_m0_use(const Rdna2Inst& in) {
    if (reads_word(in, 124)) return true;
    if (in.fmt == Rdna2Format::VINTRP) return in.opcode > 2;
    // Keep numerical LDS/GDS and relative addressing outside this service. In particular,
    // untracking M0 must not turn its unknown numerical base into a default ADDTID base.
    if (in.fmt == Rdna2Format::DS || in.mubuf_lds || in.flat_lds) return true;
    const auto& effects = rdna2_instruction_effects(in);
    // Store/atomic SDATA is also a source encoded in dst, outside the explicit source array.
    // Keep these payloads outside this closed service; consume only legacy effects here.
    if (in.fmt == Rdna2Format::SMEM &&
        (effects.smem == Rdna2ScalarMemory::Store || effects.smem == Rdna2ScalarMemory::Atomic))
        return true;
    if (effects.implicit_reads & kRdna2ImplicitM0) return true;
    // LDS_DIRECT is a dynamic operand spelling, rather than an unconditional VALU effect.
    // Unmodeled Special selectors cannot inherit interpolation-only M0 authority either.
    for (uint32_t k = 0; k < in.n_src; ++k)
        if (in.src[k].kind == OperandKind::Special && in.src[k].value >= 128 &&
            in.src[k].value != 253)
            return true;
    return false;
}
}   // namespace

std::shared_ptr<const NativeParameterBasePlan>
native_parameter_base_plan(const PixelParameterLaunch& launch,
                           const std::vector<Rdna2Inst>& instructions) {
    const auto word = pixel_parameter_sgpr(launch);
    if (!word || !closed_control(instructions)) return {};
    std::optional<Rdna2Inst> load;
    bool prefix = true, replaced = false, entry_untouched = true, interpolation = false;
    for (const auto& in : instructions) {
        if (in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in)) return {};
        const bool supplies_m0 = parameter_load(in, *word);
        if (supplies_m0) {
            if (load || !prefix || !entry_untouched) return {};
            load = in;
        } else if (writes_word(in, 124) || (!replaced && reads_word(in, static_cast<int>(*word)))) {
            return {};
        }
        if (unsupported_m0_use(in)) return {};
        if (in.fmt == Rdna2Format::VINTRP) {
            if (!load) return {};
            interpolation = true;
        }
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
            // A later visit to the entry load could read a reused SGPR lifetime.
            if (load && target <= load->pc) return {};
            prefix = false;
        }
        if (prefix && replaces_entry_word(in, static_cast<int>(*word))) replaced = true;
        if (writes_word(in, static_cast<int>(*word))) entry_untouched = false;
        if (in.is_end) prefix = false;
    }
    if (!load || !interpolation) return {};
    return std::make_shared<const NativeParameterBasePlan>(
        NativeParameterBasePlan{*load, instructions});
}

bool native_parameter_base_load(const NativeParameterBasePlan& plan, const Rdna2Inst& in) {
    return !in.synthetic_terminator && in == plan.load;
}

bool native_parameter_base_region(const NativeParameterBasePlan& plan,
                                  const std::vector<Rdna2Inst>& region) {
    if (!std::is_sorted(region.begin(), region.end(),
                        [](const auto& a, const auto& b) { return a.pc < b.pc; }))
        return false;
    for (const auto& in : region) {
        if (in.synthetic_terminator && in.is_end && in.fmt == Rdna2Format::SOPP && in.opcode == 1)
            continue;
        const auto original =
            std::lower_bound(plan.original.begin(), plan.original.end(), in.pc,
                             [](const auto& candidate, uint32_t pc) { return candidate.pc < pc; });
        if (original == plan.original.end() || in != *original) return false;
    }
    return true;
}

bool native_parameter_base_region(SpirvCompute& b, const std::vector<Rdna2Inst>& region) {
    if (!b.native_parameter_base) return true;
    if (native_fragment(b) && native_parameter_base_region(*b.native_parameter_base, region))
        return true;
    log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                             "native-parameter-base context differs from original program");
    return false;
}

bool emit_native_parameter_base_load(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    if (!b.native_parameter_base || !native_fragment(b) ||
        !native_parameter_base_load(*b.native_parameter_base, in))
        return false;
    // AMD 70648 sections 3.12.2 and 10.4.2: the automatic word selects the wave's parameter
    // block and primitive. The existing native VINTRP Input is that per-primitive service.
    // No numerical observer exists in this admitted program; never publish SGPR/M0 bits.
    rs.sreg.erase(124);
    rs.sreg_srt.erase(124);
    rs.sreg_entry_m0.erase(124);
    return true;
}
}   // namespace prosper::gpu
