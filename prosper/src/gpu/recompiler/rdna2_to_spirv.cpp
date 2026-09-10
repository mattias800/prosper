// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

namespace prosper::gpu {

namespace {


} // namespace

void record_recompile_reject_reason_for_test(const RecompileDiagnosticContext& diagnostic,
                                             const char* tag, const char* role,
                                             const char* payload) {
    log_recompile_diagnostic(diagnostic, tag, role, "%s", payload);
}

std::string last_terminal_reject_reason(uint64_t program_address) {
    std::lock_guard lock(terminal_reject_mutex());
    const auto& reasons = terminal_reject_reasons();
    const auto found = reasons.find(program_address);
    return found == reasons.end() ? std::string() : found->second;
}

void log_compute_recompile_skip_diagnostic(const RecompileDiagnosticContext& diagnostic) {
    log_recompile_diagnostic(diagnostic, "compute-recompile-reject", "consequent",
                             "reason=empty-result dispatch-skipped");
}

namespace {


// The f16 bit pattern an inline float constant supplies in a 16-bit operand position (ISA Table 10
// lists per-width encodings: "0.5 ... half: 0x3800" etc.). Only 1/(2*pi) (code 248, 0x3118) differs
// from rounding the f32 value — the f32 table entry 0.15915494 would round to a different last bit
// than the documented operand, so 16-bit consumers must use these bits, not the f32 constant.

}  // namespace

uint32_t fragment_consumed_attribute_mask(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    // Decoded instruction by instruction rather than through `rdna2_walk`, and this is the whole
    // point of the function. `rdna2_walk` stops at the first `is_end` AND at the first Unknown
    // encoding (rdna2_decode.cpp), which is exactly the prefix `fragment_interpolation_layout`
    // already sees -- so routing through it would make this mask EQUAL to `attribute_mask`, not a
    // superset, and the margin the caller's safety argument depends on would be zero. Stepping past
    // both terminators is what buys the margin.
    //
    // The bias is deliberate and one-directional. Decoding past a terminator can decode data as
    // instructions, so a spurious VINTRP hit is possible; that costs one dead output varying.
    // MISSING an attribute would hand the fragment stage an input the vertex stage no longer
    // exports, which is the regression this analysis must never cause. Over-report, never under.
    const size_t span = rdna2_recompile_code_span(code, dwords);
    uint32_t mask = 0;
    for (size_t pc = 0; pc < span;) {
        const Rdna2Inst instruction = rdna2_decode_one(code + pc, span - pc);
        if (instruction.fmt == Rdna2Format::VINTRP && instruction.vintrp_attr < 32)
            mask |= 1u << instruction.vintrp_attr;
        if (!instruction.len_dwords) break;   // safety: never advance 0
        pc += instruction.len_dwords;
    }
    return mask;
}

bool dead_varying_elimination_enabled() {
    static const bool disabled = getenv("PROSPER_NO_DEAD_VARYING_ELIM") != nullptr;
    return !disabled;
}

void apply_fragment_consumption(PixelInputMapping& mapping,
                                const uint32_t* fragment_code, size_t dwords) {
    if (!dead_varying_elimination_enabled() || !mapping.valid_mask || !fragment_code || !dwords)
        return;
    mapping.consumed_mask = fragment_consumed_attribute_mask(fragment_code, dwords);
    mapping.consumed_known = true;
}

std::vector<uint32_t> recompile_interpolation_geometry(
        const FragmentInterpolationLayout& layout, bool capture_position,
        bool synthesize_rect) {
    SpirvCompute builder;
    return builder.build_interpolation_geometry(layout, capture_position, synthesize_rect);
}

namespace {
}

namespace {
// Defined after the scalar-writer inventory it depends on; used by detect_forward_ifs above it.
}
// allow_vcc: also accept a forward s_cbranch_vccz/vccnz. Fragment reduces the per-lane bool through an
// enforced native wave64 subgroup; vertex retains the guarded per-invocation representation. Compute
// routes accepted VCC/EXEC branches through the CFG dispatcher, whose workgroup scratch reduction spans
// the configured 32/64-lane guest wave independently of the implementation-defined host subgroup width.
// code/dwords: the raw stream, so a branch target past the first s_endpgm can be decoded and
// verified to be a genuine early-out (see below) instead of blanket-clamped.
ForwardIf detect_forward_if(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                            const uint32_t* code, size_t dwords,
                            const std::unordered_set<uint32_t>* skip = nullptr) {
    ForwardIf F; const Rdna2Inst* br = nullptr; int nbranch = 0; uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        switch (in.opcode) {
            case 0x02: case 0x08: case 0x09:                         // s_branch / execz / execnz -> reject
                return F;
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (!allow_vcc) return F;                            // compute: needs a wave-uniform VCC test
                br = &in; nbranch++; break;
            case 0x04: case 0x05:                                    // scc0 / scc1 (SCC is scalar/wave-uniform)
                if (skip && skip->count(in.pc)) break;               // alpha-test kill-mask branch: handled by
                                                                     // straight-line linearization, not a struct-if
                br = &in; nbranch++; break;
            default: break;                                          // hints (nop/waitcnt/…) are fine
        }
    }
    if (nbranch != 1 || !br) return F;
    uint32_t tgt = branch_target(*br);
    // Early-out: a forward branch PAST s_endpgm skips the rest of the shader — but only if execution
    // at the target immediately terminates. Decode from tgt and require the first non-s_nop
    // instruction to be s_endpgm (the compiled early-out shape `s_cbranch L; work; s_endpgm;
    // L: s_endpgm`). Anything else is REAL reachable code (an else-block) that the old blanket clamp
    // silently discarded — valid SPIR-V, wrong semantics (#129) — so reject instead (the caller's
    // straight-line fallback also rejects the branch, loudly). When verified, clamp the merge to
    // end_pc so the conditional block is [branch_pc+1, end_pc) and s_endpgm is emitted after the merge.
    bool early = false;
    if (tgt > end_pc && tgt != UINT32_MAX) {
        if (!code || tgt >= dwords) return F;    // target outside the decode window: can't verify
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + tgt, dwords - tgt, tail);
        bool ends_immediately = false;
        for (const auto& ti : tail) {
            if (ti.is_end) { ends_immediately = true; break; }
            if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;   // s_nop padding
            break;                               // real instruction at the target -> not an early-out
        }
        if (!ends_immediately) return F;
        tgt = end_pc; early = true;
    }
    if (tgt <= br->pc || tgt > end_pc) return F;                     // must be forward, within the stream
    F.found = true; F.branch_pc = br->pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
    F.on_scc0 = (br->opcode == 0x04 || br->opcode == 0x06);         // scc0/vccz: skip block when flag==0
    F.on_vcc  = (br->opcode == 0x06 || br->opcode == 0x07);
    return F;
}

namespace {


// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

// S_MOV_B64 from VCC publishes both an exact Bool-domain saved mask and its two ballot words under
// native Wave64. The compact structured emitter preserves both views through its SSA/PHI machinery,
// but unlike the dispatcher it previously had no lifetime tag telling S_FF1/S_BCNT which view owns
// the pair at an exact consumer. Compute a small forward MUST analysis over the decoded scalar CFG:
// a saved-mask fact is generated by an exact VCC copy reached from a proved mask-domain VCC or by a
// SAVEEXEC destination (which receives OLD_EXEC independently of its logical source). Every
// overlapping scalar write kills it, and joins retain it only when every reachable predecessor
// agrees. Indirect PC updates have no successor, so they cannot manufacture a dominance fact beyond
// an unknown transfer.

} // namespace

namespace {


} // namespace

bool rdna2_specialize_pcrel_dispatch(std::vector<Rdna2Inst>& instructions,
                                     const PcrelDispatchInfo& info,
                                     uint32_t selected_target) {
    return specialize_pcrel_dispatch(instructions, info, selected_target);
}

bool compute_trip_witness_active(uint64_t program_address) {
    const ComputeTripBoundSettings settings = compute_trip_bound_settings();
    // Mirrors every arming rule that does not need the program's bytes. The GDS-use refusal is the
    // one exception and lives with the caller that already decodes the program (see gpu_executor's
    // uses_gds), so this must not be treated as the complete predicate.
    if (!settings.bound) return false;
    if (settings.only_phase == ComputeTripBoundSettings::kAllPhases) return false;
    if (!settings.only_program) return true;
    return program_address == settings.only_program;
}

bool fragment_vcc_branch_is_wave_uniform_for_test(
        const uint32_t* code, size_t dwords, uint32_t branch_pc) {
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    return vcc_exit_is_wave_uniform(instructions, branch_pc);
}

bool compute_shader_prefers_native_multiwave(const std::vector<Rdna2Inst>& ins,
                                             const uint32_t* code, size_t dwords,
                                             RecompileDiagnosticContext diagnostic) {
    bool low = false;
    bool high = false;
    bool guest_barrier = false;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3) {
            low |= in.opcode == 0x365;
            high |= in.opcode == 0x366;
        }
        guest_barrier |= in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a;
        if (low && high) return true;
    }
    if (!guest_barrier || !code) return false;

    // The guarded phase splitter's shared proof is stronger than the whole-stream CFG shape:
    // branches and loops inside one barrier-free phase do not make the outer uniform guard or its
    // barriers divergent. Such kernels need an exact native subgroup for nested guest-wave votes.
    if (const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
        phased.found && phased.guarded)
        return true;

    // Mirror the conservative, acyclic subset of emit_body's structured-compute admission. Counting
    // raw VCC/EXEC opcodes is insufficient: kill-mask branches may be safely linearized, loop exits
    // are owned by another emitter, and rejected CFGs never reach guest_wave_any. Requiring the same
    // accepted ForwardIf regions proves that portable lowering really emits two scratch barriers per
    // counted vote and that exact-subgroup lowering removes them. Loops deliberately stay behind the
    // explicit experiment until their additional compute guards are shared with this analysis.
    auto safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    const std::vector<DivLoop> loops =
        detect_divergent_loops(ins, safe, /*fragment*/false, diagnostic, "multiwave-probe");
    if (!loops.empty()) return false;

    bool rejected = false;
    const std::vector<ForwardIf> branches = detect_forward_ifs(
        ins, /*allow_vcc*/false, code, dwords, &safe, nullptr, &rejected,
        /*compute_wave_branches*/true, diagnostic);
    if (rejected) return false;

    auto top_level_pc = [&](uint32_t pc) {
        for (const ForwardIf& parent : branches) {
            const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
            if (parent.branch_pc < pc && pc < parent_end) return false;
        }
        return true;
    };
    const bool barriers_are_top_level = std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a || top_level_pc(in.pc);
    });
    if (!barriers_are_top_level) return false;

    size_t structured_wave_votes = 0;
    for (const ForwardIf& branch : branches) {
        if (!branch.on_exec && !branch.on_vcc) continue;
        if (!top_level_pc(branch.branch_pc)) return false;
        ++structured_wave_votes;
    }
    // Four proven scratch-emulated votes keep the default narrower than the all-multi-wave experiment.
    return structured_wave_votes >= 4;
}

bool compute_shader_prefers_native_multiwave(const uint32_t* code, size_t dwords,
                                             RecompileDiagnosticContext diagnostic) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    return compute_shader_prefers_native_multiwave(ins, code, dwords, diagnostic);
}

// See FlatLoadInfo / analyze_flat_loads in the header (#1171). A `base + offset` flat address VGPR pair
// v[N:N+1] is computed as vN = add(base_lo_sgpr, offset_lo) [carry-out] and v(N+1) =
// add(base_hi_sgpr, offset_hi, carry) [carry-in], where s[base_lo:base_hi] are consecutive user SGPRs
// holding a 64-bit guest pointer (the low kernel-arg pointer dword feeds the low address dword). We
// identify the base by the NEAREST prior definition of each address dword being an integer add that
// reads a user-range SGPR, and require the low dword to add the SGPR one below the high dword's.
// Anything else (a non-add producer, a non-user SGPR, a store/atomic/LDS/global-with-saddr form) leaves
// the load unresolved so the caller keeps failing visibly. flat_access_info lives in this TU's
// anonymous namespace but is visible here (internal linkage is still TU-wide).
// Resolution is over LINEAR program order (not CFG-aware). That is exact for the target decode kernels
// (the address adds sit in the same block immediately before the load) and the base is a loop-invariant
// kernel-arg pointer, so the resolved base is stable across loop iterations; the executor's
// guest_readable_mapping_containing validation is the runtime backstop against a bogus base.
FlatLoadAnalysis analyze_flat_loads(const uint32_t* code, size_t dwords, uint32_t user_sgpr_count) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    FlatLoadAnalysis out;

    auto writes_vgpr = [](const Rdna2Inst& d, uint32_t reg) {
        return d.dst.kind == OperandKind::VGPR && static_cast<uint32_t>(d.dst.value) == reg;
    };
    // If `d` is an integer add that could produce a 64-bit-pointer dword, return the user-range SGPR it
    // adds (else -1). Recognizes the add_co / add_co_ci / add_nc forms a compiler emits for pointer
    // arithmetic: VOP2 0x25 (add_nc) / 0x28 (add_co_ci), VOP3 0x30F (add_co) / 0x125 (add_nc) / 0x128
    // (add_co_ci).
    auto add_user_sgpr = [&](const Rdna2Inst& d) -> int32_t {
        const bool is_add =
            (d.fmt == Rdna2Format::VOP2 && (d.opcode == 0x25 || d.opcode == 0x28)) ||
            (d.fmt == Rdna2Format::VOP3 &&
             (d.opcode == 0x30F || d.opcode == 0x125 || d.opcode == 0x128));
        if (!is_add) return -1;
        for (int s = 0; s < 2; ++s)
            if (d.src[s].kind == OperandKind::SGPR && d.src[s].value >= 0 &&
                static_cast<uint32_t>(d.src[s].value) < user_sgpr_count)
                return d.src[s].value;
        return -1;
    };
    // The user-range SGPR added by the nearest prior writer of `reg` (or -1 if that writer is not a
    // user-SGPR add). We trust only the immediate definition — a later non-add redefinition breaks the
    // pattern and must not be skipped over.
    auto prior_add_sgpr = [&](uint32_t reg, size_t before) -> int32_t {
        for (size_t j = before; j-- > 0;)
            if (!ins[j].is_end && writes_vgpr(ins[j], reg))
                return add_user_sgpr(ins[j]);
        return -1;
    };

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::FLAT) continue;
        if (in.flat_segment == 1u) continue;   // scratch spill: analyze_static_scratch owns it
        out.any = true;
        const FlatAccessInfo access = flat_access_info(in.opcode);
        FlatLoadInfo info;
        info.load_pc = in.pc;
        info.vaddr_lo_reg = in.src[0].kind == OperandKind::VGPR
                                ? static_cast<uint32_t>(in.src[0].value) : 0u;
        info.dst_reg = static_cast<uint32_t>(in.dst.value);
        info.bits = access.bits;
        info.components = access.components;
        info.sign_extend = access.sign_extend;
        // Resolvable only as a plain LOAD with a VGPR address pair and a null SADDR (true flat segment).
        const bool shape = access.valid && !access.store && !in.flat_lds &&
                           in.src[0].kind == OperandKind::VGPR &&
                           in.src[1].kind == OperandKind::Special && in.src[1].value == 125;
        if (shape) {
            const uint32_t lo = info.vaddr_lo_reg, hi = lo + 1;
            const int32_t hi_base = prior_add_sgpr(hi, i);
            const int32_t lo_base = prior_add_sgpr(lo, i);
            if (hi_base > 0 && lo_base == hi_base - 1)
                info.base_sgpr = lo_base;
        }
        if (info.base_sgpr < 0) out.all_resolved = false;
        out.loads.push_back(info);
    }
    return out;
}

std::vector<uint32_t> safe_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> s = safe_execz_branches(ins);
    return std::vector<uint32_t>(s.begin(), s.end());
}

std::vector<uint32_t> structured_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    bool rejected = false;
    const auto branches = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords, &safe,
                                             nullptr, &rejected,
                                             /*compute_wave_branches*/true);
    std::vector<uint32_t> pcs;
    if (!rejected) for (const auto& branch : branches) pcs.push_back(branch.branch_pc);
    return pcs;
}

std::vector<uint32_t> cselect_b64_low_only_pcs_for_test(
        const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto proven = proven_cselect_b64_low_only_pcs(ins);
    std::vector<uint32_t> result(proven.begin(), proven.end());
    std::sort(result.begin(), result.end());
    return result;
}

uint32_t fragment_color_export_mask(const uint32_t* code, size_t dwords) {
    uint32_t packed = 0;
    // One nibble per MRT, MRT0..MRT7. Sized 2 until 2026-08-15, which silently forced
    // `write_mask &= 0` for slots 2..7 at gpu_execute.hpp's EXP.EN gate -- so a shader exporting to
    // MRT2+ had those attachments dropped no matter what CB_TARGET_MASK and CB_SHADER_MASK said.
    std::array<bool, kFragmentColorOutputs> realized{};
    // This property needs no retained instructions. Match rdna2_walk's bounded decode and
    // termination rules without allocating/copying a complete instruction vector on every draw.
    for (size_t pc = 0; pc < dwords;) {
        const Rdna2Inst in = rdna2_decode_one(code + pc, dwords - pc);
        if (in.is_end || in.fmt == Rdna2Format::Unknown) break;
        if (in.fmt == Rdna2Format::EXP && in.exp_target < realized.size() &&
            !realized[in.exp_target] && in.exp_en != 0) {
            packed |= (in.exp_en & 0xFu) << (in.exp_target * 4u);
            realized[in.exp_target] = true;
        }
        if (in.len_dwords == 0) break;
        pc += in.len_dwords;
    }
    return packed;
}

uint32_t fragment_spirv_required_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.FragmentSubgroupSize=";
    bool legacy_arithmetic_marker = false;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2 &&
            spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
            legacy_arithmetic_marker = true;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(
                        text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length && (value == 32 || value == 64))
                        return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // Compatibility for cached/captured modules produced before explicit module metadata.
    return legacy_arithmetic_marker ? 64u : 0u;
}

uint32_t compute_spirv_min_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.ComputeSubgroupMin=";
    uint32_t required = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length &&
                        (value == 4 || value == 16 || value == 32 || value == 64))
                        required = std::max(required, static_cast<uint32_t>(value));
                }
            }
        }
        offset += words;
    }
    return required;
}

namespace {


}  // namespace

uint32_t fragment_spirv_required_subgroup_reasons(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return UINT32_MAX;
    constexpr char prefix[] = "Prosper.FragmentSubgroupWhy=";
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return UINT32_MAX;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value =
                        std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length) return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // No marker: the module predates #2147. UINT32_MAX rather than 0, so a caller cannot read
    // 'unknown' as 'nothing required it'.
    return UINT32_MAX;
}

uint32_t fragment_spirv_required_subgroup_features(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    uint32_t features = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2) {
            if (spirv[offset + 1] == Cap_GroupNonUniformVote)
                features |= kFragmentSubgroupVote;
            else if (spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
                features |= kFragmentSubgroupArithmetic;
            else if (spirv[offset + 1] == Cap_GroupNonUniformShuffle)
                features |= kFragmentSubgroupShuffle;
            else if (spirv[offset + 1] == Cap_GroupNonUniformBallot)
                features |= kFragmentSubgroupBallot;
        }
        offset += words;
    }
    return features;
}

bool fragment_spirv_uses_internal_gds(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return false;
    std::unordered_map<uint32_t, uint32_t> sets, bindings;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return false;
        if (opcode == Op_Decorate && words == 4) {
            if (spirv[offset + 2] == Dec_DescriptorSet)
                sets[spirv[offset + 1]] = spirv[offset + 3];
            else if (spirv[offset + 2] == Dec_Binding)
                bindings[spirv[offset + 1]] = spirv[offset + 3];
        }
        offset += words;
    }
    for (const auto& [variable, set] : sets) {
        auto binding = bindings.find(variable);
        if (set == 1 && binding != bindings.end() && binding->second == 0) return true;
    }
    return false;
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.

} // namespace

} // namespace prosper::gpu
