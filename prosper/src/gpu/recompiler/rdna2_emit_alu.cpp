// rdna2_emit_alu.cpp — emit_alu: the RDNA2 instruction-family translator, split out of
// rdna2_to_spirv.cpp. Shared state lives in gpu/recompiler/rdna2_to_spirv_internal.hpp.
#include <array>
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/cfg/exec_value_census.hpp"
#include "gpu/recompiler/ir/lower/buffer_address_lowering.hpp"
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/recompiler/ir/lower/smem_case_routes.hpp"
#include "gpu/recompiler/ir/lower/flat_case_routes.hpp"
#include "gpu/recompiler/ir/lower/vop3_case_routes.hpp"
#include "gpu/recompiler/ir/lower/vop1_case_routes.hpp"
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/flat_window_lowering.hpp"
#include "gpu/recompiler/ir/lower/ds_address_lowering.hpp"
#include "gpu/recompiler/ir/lower/ds_case_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_coordinates_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_d16_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_mip_clamp_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_cube_coordinates_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_operand_coordinates_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_gather_layer_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_store_values_lowering.hpp"
#include "gpu/recompiler/ir/lower/image_storage_routes.hpp"
#include "gpu/recompiler/ir/lower/buffer_format_values.hpp"
#include "gpu/recompiler/ir/lower/buffer_store_values.hpp"
#include "gpu/recompiler/ir/lower/buffer_access_routes.hpp"
#include "gpu/recompiler/ir/lower/ds_cross_lane_lowering.hpp"
#include "gpu/recompiler/ir/lower/carry_mask_write.hpp"
#include "gpu/recompiler/ir/lower/native_half_fma_lowering.hpp"
#include "gpu/recompiler/ir/lower/half_fma_finite.hpp"
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/storage_dst_sel.hpp"
#include "gpu/recompiler/ir/lower/vop3p_integer_dot_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3p_mixed_float.hpp"
#include "gpu/recompiler/ir/lower/vop3p_packed_values.hpp"
#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/ir/lower/vop3_sad_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_permute_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3p_packed_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad64_lowering.hpp"
#include "gpu/recompiler/rdna2_mimg_store.hpp"
#include "gpu/recompiler/rdna2_movrels.hpp"
#include "gpu/recompiler/rdna2_sampled_array.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <type_traits>
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
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_mask_write.hpp"
#include "gpu/recompiler/rdna2_partial_vcc_word.hpp"
#include "gpu/recompiler/ir/lower/scalar_alu_lowering.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_output_modifiers.hpp"
#include "gpu/recompiler/ir/lower/vop3_unary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_subword_to_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_float_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_bitfield_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer64_shift_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_align_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_float_modifiers.hpp"
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_ldexp_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mask_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering.hpp"
#include <functional>
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_float_source.hpp"
#include "gpu/recompiler/ir/lower/vop1_dpp_source_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_half_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_unary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_unary_f16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_cross_lane_lowering.hpp"

namespace prosper::gpu {


inline bool sreg_range_written(const RegState& rs, int base, uint32_t words) {
    for (uint32_t word = 0; word < words; ++word)
        if (rs.sreg_written.count(base + static_cast<int>(word))) return true;
    return false;
}

// An SRT tag describes the complete hardware descriptor, not merely its base SGPR. Accept it only
// while every word still carries the same provenance; any partial overwrite makes the descriptor
// unrepresentable even when the base word itself was untouched.
inline bool sreg_srt_range_tag(const RegState& rs, int base, uint32_t words, uint32_t& tag) {
    auto first = rs.sreg_srt.find(base);
    if (first == rs.sreg_srt.end()) return false;
    tag = first->second;
    for (uint32_t word = 1; word < words; ++word) {
        auto it = rs.sreg_srt.find(base + static_cast<int>(word));
        if (it == rs.sreg_srt.end() || it->second != tag) return false;
    }
    return true;
}

// A DIRECT descriptor staged into `base` by copying it word-for-word out of entry-time user data
// (#1773). Succeeds only for a FAITHFUL WHOLE-DESCRIPTOR move: every word `base + i` must alias
// exactly `origin + i` for one origin. A partial, permuted or partly-recomputed copy is a descriptor
// the shader assembled itself, which this must not resolve -- binding the wrong resource renders
// silently wrong texels, which is strictly worse than declining the draw.
//
// The load-bearing condition is applied where the alias is ESTABLISHED, not here: the source must
// still have been entry-time user data at the moment of the copy (`record_scalar_write`). It is
// deliberately NOT re-checked at consumption, because a copy captures bits -- a later write to the
// SOURCE register cannot change what the DESTINATION already holds. Re-checking it would reject the
// exact shape #1773 documents: Earthion stages s[9:16] into s[20:27] and then immediately reuses
// s[12:19] for the second descriptor, so five of the origin words are overwritten before the sample.
//
// Two invariants carry that instead, and both live elsewhere: a write to the DESTINATION expires
// the alias (`record_scalar_write`), and every control-flow join meets the two edges' claims
// (`merge_ud_alias`). Neither is visible here, which is why they are named here.
inline bool sreg_range_ud_alias(const RegState& rs, int base, uint32_t words, int& origin) {
    auto first = rs.sreg_ud_alias.find(base);
    if (first == rs.sreg_ud_alias.end()) return false;
    origin = first->second;
    for (uint32_t word = 0; word < words; ++word) {
        auto it = rs.sreg_ud_alias.find(base + static_cast<int>(word));
        if (it == rs.sreg_ud_alias.end() ||
            it->second != origin + static_cast<int>(word)) return false;
    }
    return true;
}

namespace {
}

namespace {
// Defined after the scalar-writer inventory it depends on; used by detect_forward_ifs above it.
}

namespace {

// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

} // namespace


// Emit one ALU instruction (VOP1/2/C/3 or SOP1/2) into `b`, updating `rs`. Returns true if `in` is an
// ALU format handled here; sets ok=false if it is an ALU op this stage doesn't support yet. Non-ALU
// formats (EXP/memory/...) return false so the stage-specific caller can handle them.
static bool instruction_reads_scc_as_scalar_data(const Rdna2Inst& in) {
    return rdna2_reads_scc_as_data(in.fmt, in.opcode) != Rdna2Answer::No;
}

bool emit_alu(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_exec_update,
              const std::unordered_set<uint32_t>* safe_execz, bool allow_smem,
              const ShaderResourceTable* rt, bool allow_wave) {
    const ir::cfg::ExecCensusObservation exec_census(b, rs, in);   // step 7a shadow; observes only
    rs.native_interpolation_publication = {};
    if (b.is_fragment && instruction_reads_scc_as_scalar_data(in))
        b.mark_fragment_wave_scalar_use(rs.scc);
    auto& vreg = rs.vreg; uint32_t& vcc = rs.vcc;
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o, &ok); };
    // SDWA/DPP forms carry a sub-dword select or cross-lane control word we don't model. The decoder
    // flags them (and gets their length right); reject here rather than compute with a wrong operand.
    PortableMaskHalfInstruction portable_halves(b, rs, in, ok);
    if (!ok || portable_halves.handled()) return true;
    VccMaskViewDrop drop(rs);   // takes VCC's lane view away at exit when armed
    if (refuse_mask_write(b, rs, in, drop, ok) || emit_exec_cmov(b, rs, in, ok)) return true;
    switch (in.fmt) {
        case Rdna2Format::SOP1: {
            return ir::lower_sop1(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::SOP2: {
            return ir::lower_sop2(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::VOP3P: {
            if (ir::lower_vop3p_packed_integer(b, rs, in, ok, allow_wave)) return true;
            if (auto routed = ir::lower_vop3p_packed_values(b, rs, in, ok, allow_wave))
                return *routed;
            if (ir::lower_vop3p_integer_dot(b, rs, in, ok, allow_wave)) return true;
            return ir::lower_vop3p_mixed_float(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::SOPC: {
            return ir::lower_sopc(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::SOPK: {
            return ir::lower_sopk(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::VOP1: {
            return ir::Vop1CaseRoutes{ir::LoweringContext{b, rs, in, ok, allow_wave}, vreg,
                                      std::cref(val)}
                .lower();
        }
        case Rdna2Format::VOP2: {
            return ir::lower_vop2(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::VOPC: {                                     // v_cmp_* -> VCC; v_cmpx_* also -> EXEC
            return ir::lower_vopc(b, rs, in, ok, allow_wave, allow_exec_update);
        }
        case Rdna2Format::VOP3: {
            if (auto routed = ir::lower_vop3_cross_lane(b, rs, in, ok, allow_wave)) return *routed;
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            bool clamp_routed = false;
            ir::Vop3FloatModifiers modifiers{b, in, val, clamp_routed, ok};
            auto fv = [&](int k) { return modifiers.fv(k); };
            auto fresult = [&](uint32_t bits) { return modifiers.fresult(bits); };
            ir::Vop3CaseRoutes{
                {b, rs, in, ok, allow_wave}, vreg, old_d, std::cref(val), std::cref(fv),
                std::cref(fresult)}
                .lower();
            // A set CLAMP bit on an opcode that does not route through fresult means unmodeled
            // INTEGER saturation (ISA 6.5: "for integer operations, it clamps the result to the
            // largest and smallest representable value") or an unhandled pack/select combination
            // (v_cvt_pkrtz, v_cndmask_e64, the VOP3B carry ops) — reject fail-visibly rather than
            // silently emitting the wrapping/unclamped result. OMOD on integer results is
            // architecturally ignored, so only CLAMP gates here.
            if (ok && in.clamp && !clamp_routed) ok = false;
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::SOPP: {
            // Control flow. The only branch we can safely linearize today is the common forward
            // s_cbranch_execz "skip the EXEC-predicated block if no lane is active" idiom. Branches
            // on SCC/VCC (or EXECNZ) can skip code for reasons not represented by EXEC predication,
            // so accepting them as no-ops would silently execute the wrong path.
            switch (in.opcode) {
                // Hints / sync with no effect in our synchronous SSA model — safe no-ops.
                case 0x10:   // s_sendmsg: no-op, except GS_ALLOC_REQ in the NGG subgroup shell
                    if (b.ngg_alloc_request) ok = b.ngg_alloc_request(rs, in);
                    break;
                case 0x00:   // s_nop
                case 0x0c:   // s_waitcnt        (no async memory latency to wait on)
                case 0x16:   // s_ttracedata     (sends M0 to the thread-trace stream — a profiling side
                             //                   channel with no architectural effect: it writes no SGPR,
                             //                   VGPR, or memory and does not branch. Prosper models no
                             //                   thread trace, so emitting nothing IS the correct lowering.
                             //                   Observed live in Greak (PPSA02849) vertex shaders, which
                             //                   ship it at pc=2 and were rejected whole. CONFIDENCE: HIGH
                             //                   — round-tripped against llvm-mc gfx1030.)
                case 0x17:   // s_cbranch_cdbgsys (Prosper exposes no attached GPU system debugger, so
                             //                    COND_DBG_SYS is permanently clear and the branch falls through)
                case 0x20:   // s_inst_prefetch  (I-cache hint)
                case 0x21:   // s_clause         (memory-clause scheduling hint)
                case 0x22:   // s_wait_idle
                    break;   // (s_waitcnt_vscnt is SOPK on gfx10, not SOPP 0x7d — see the SOPK case)
                case 0x08:                                          // s_cbranch_execz
                    if (in.simm16 < 0) ok = false;                 // backward = loop -> unsupported
                    else if (rs.exec_narrowed && (!safe_execz || !safe_execz->count(in.pc))) ok = false;
                    break;                                          // forward = no-op (predication covers it)
                case 0x0a:                                          // s_barrier
                    if (b.is_compute || b.ngg_private_lds) b.barrier();
                    else ok = false;                                // barrier only meaningful in compute
                    break;
                case 0x04: case 0x05:                              // s_cbranch_scc0 / scc1
                    // An alpha-test / clip() kill-mask early-out (SCC = "any lane survives", set by a 64-bit
                    // wave-mask op — see mask_test_branches) is a pure wave optimization: per-invocation the
                    // survivor mask narrows EXEC in the block below and the export OpKills the failed lanes,
                    // so the branch LINEARIZES away exactly like a forward s_cbranch_execz. Only these
                    // recognized branches are dropped; any other SCC branch is still a real uniform-if we
                    // can't model straight-line, so it rejects.
                    if (safe_execz && safe_execz->count(in.pc)) break;   // recognized kill-mask branch -> no-op
                    ok = false;
                    break;
                case 0x06: case 0x07:                              // s_cbranch_vccz / vccnz
                    // A byte-exact Astro NGG terminal suffix may skip only trailing PARAM exports
                    // after POS has already been exported. In the one-lane projection emitting those
                    // otherwise-unused values is harmless; the gate proof records only that bounded
                    // branch in safe_execz. General VCC branches remain unsupported below.
                    if (safe_execz && safe_execz->count(in.pc)) break;
                    ok = false;
                    break;
                case 0x09:                                         // s_cbranch_execnz
                    ok = false;
                    break;
                default: ok = false;   // s_branch / s_sendmsg / s_setreg / etc. -> reject
            }
            return true;
        }
        case Rdna2Format::SMEM: {
            // Scalar memory load. Modeled as a load from a single bound constant buffer indexed by the
            // immediate byte offset (>>2 -> dword index); SBASE/descriptor base is folded into the
            // binding. N consecutive dwords -> SDATA..SDATA+N-1. Only the compute path binds the cbuf,
            // so reject in graphics stages (allow_smem=false). Register-offset SMEM not yet handled.
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n = 1;  break;   // s_load_dword     / s_buffer_load_dword
                case 0x1: case 0x9: n = 2;  break;   // s_load_dwordx2   / s_buffer_load_dwordx2
                case 0x2: case 0xA: n = 4;  break;   // s_load_dwordx4   / s_buffer_load_dwordx4
                case 0x3: case 0xB: n = 8;  break;   // s_load_dwordx8   / s_buffer_load_dwordx8
                case 0x4: case 0xC: n = 16; break;   // s_load_dwordx16  / s_buffer_load_dwordx16
                default: ok = false; return true;    // stores / others not yet
            }
            if (emit_owned_raw_window(b, rs, in, n, ok)) return true;
            ir::SmemCaseRoutes smem_routes{{b, rs, in, ok, allow_wave},
                                           allow_smem,
                                           rt,
                                           sreg_range_written,
                                           sreg_srt_range_tag};
            if (auto routed = smem_routes.pcrel(n)) return *routed;
            if (auto routed = smem_routes.resolved(n)) return *routed;
            return smem_routes.memory(n);
        }
        case Rdna2Format::FLAT: {
            const FlatAccessInfo access = flat_access_info(in.opcode);
            return ir::FlatCaseRoutes{{b, rs, in, ok, allow_wave}, access, rt, std::cref(val)}
                .lower();
        }
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF: {
            // Untyped buffer LOAD — the per-lane fetch mechanism (vertex fetch et al.). Modeled as a
            // per-lane load from the bound constant buffer: byte addr = (offen ? VADDR : 0) + SOFFSET
            // + inst-offset; index = addr>>2; N dwords -> VDATA..+N-1. Descriptor (SRSRC), idxen*stride,
            // and the format-converting buffer_load_format_* variants are deferred. Compute-only (cbuf).
            // LDS bit set = transfer between LDS and memory instead of VGPRs (ISA Table 98). The
            // VDATA VGPRs stay untouched on hardware and the data lands in LDS — translating it as
            // a VGPR load would clobber a live register AND drop the LDS write. Reject until a
            // buffer->LDS model exists.
            if (in.fmt == Rdna2Format::MUBUF && in.mubuf_lds) { ok = false; return true; }
            // MTBUF opcodes 8..15 pack D16 results/inputs two per VGPR. Reusing the ordinary or raw
            // MUBUF cases below would silently apply the wrong register layout, so fail closed until
            // that packing is modeled.
            if (in.fmt == Rdna2Format::MTBUF && in.opcode >= 8u) { ok = false; return true; }
            // TFE appends a fault/status result after the data VGPRs. Dropping that write can corrupt
            // later register consumers, so reject until status-return semantics are implemented.
            if ((in.fmt == Rdna2Format::MUBUF && in.mubuf_tfe) ||
                (in.fmt == Rdna2Format::MTBUF && in.mtbuf_tfe)) {
                ok = false;
                return true;
            }
            uint32_t n = 0; bool is_format = false, is_store = false, is_atomic = false;
            bool is_atomic_x2 = false, is_atomic_fminmax = false, atomic_fmin = false;
            uint32_t atomic_op = 0;   // SPIR-V atomic RMW opcode for is_atomic (set by the switch)
            uint32_t atomic_x2_record_count = 0;
            bool raw_subword = false, raw_signed = false;
            uint32_t raw_bits = 32;
            switch (in.opcode) {
                case 0x8: n = 1; raw_subword = true; raw_bits = 8; break;   // buffer_load_ubyte
                case 0x9: n = 1; raw_subword = true; raw_signed = true; raw_bits = 8; break; // sbyte
                case 0xA: n = 1; raw_subword = true; raw_bits = 16; break;  // buffer_load_ushort
                case 0xB: n = 1; raw_subword = true; raw_signed = true; raw_bits = 16; break; // sshort
                case kMubufOpcodeLoadDword:   n = 1; break;
                case kMubufOpcodeLoadDwordX2: n = 2; break;
                case kMubufOpcodeLoadDwordX4: n = 4; break;
                case kMubufOpcodeLoadDwordX3: n = 3; break; // x3 sorts after x4 (llvm-mc gfx1010)
                case 0x0: n = 1; is_format = true; break;   // buffer_load_format_x  (vertex fetch)
                case 0x1: n = 2; is_format = true; break;   // buffer_load_format_xy
                case 0x2: n = 3; is_format = true; break;   // buffer_load_format_xyz
                case 0x3: n = 4; is_format = true; break;   // buffer_load_format_xyzw
                case kMubufOpcodeStoreDword: n = 1; is_store = true; break;
                case kMubufOpcodeStoreDwordX2: n = 2; is_store = true; break;
                case kMubufOpcodeStoreDwordX4: n = 4; is_store = true; break;
                case kMubufOpcodeStoreDwordX3: n = 3; is_store = true; break;
                case 0x4: n = 1; is_format = true; is_store = true; break;   // buffer_store_format_x
                case 0x5: n = 2; is_format = true; is_store = true; break;   // buffer_store_format_xy
                case 0x6: n = 3; is_format = true; is_store = true; break;   // buffer_store_format_xyz
                case 0x7: n = 4; is_format = true; is_store = true; break;   // buffer_store_format_xyzw
                // 32-bit atomic RMW family (RDNA2 MUBUF opcodes 0x30..0x3b). Each does
                // mem = mem OP VDATA and returns the PRE-op value in VDATA. They all share the generic
                // OpAtomic<Op>(ptr, Device, AcqRel, value) shape emitted by cbuf_atomic_rtn. VDATA is one
                // dword. CMPSWAP (0x31, two-operand), CSUB (0x34, conditional-subtract), INC/DEC
                // (0x3c/0x3d, wrap semantics), and most x2 64-bit variants stay deferred
                // (fail-visible) via the default case below. Opcodes 0x50/0x5a have separately
                // guarded GTA V lowerings: one true qword RMW, never two ordinary 32-bit atomics.
                case kMubufOpcodeAtomicSwap:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicExchange; break;
                case kMubufOpcodeAtomicAdd:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicIAdd; break;
                case kMubufOpcodeAtomicSub:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicISub; break;
                case kMubufOpcodeAtomicSmin:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicSMin; break;
                case kMubufOpcodeAtomicUmin:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicUMin; break;
                case kMubufOpcodeAtomicSmax:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicSMax; break;
                case kMubufOpcodeAtomicUmax:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicUMax; break;
                case kMubufOpcodeAtomicAnd:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicAnd; break;
                case kMubufOpcodeAtomicOr:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicOr; break;
                case kMubufOpcodeAtomicXor:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicXor; break;
                case kMubufOpcodeAtomicFmin:
                    n = 1; is_atomic = true; is_atomic_fminmax = true;
                    atomic_fmin = true; break;
                case kMubufOpcodeAtomicFmax:
                    n = 1; is_atomic = true; is_atomic_fminmax = true;
                    atomic_fmin = false; break;
                case kMubufOpcodeAtomicSwapX2:
                    n = 2; is_atomic = true; is_atomic_x2 = true;
                    atomic_op = Op_AtomicExchange; break;
                case kMubufOpcodeAtomicOrX2:
                    n = 2; is_atomic = true; is_atomic_x2 = true;
                    atomic_op = Op_AtomicOr; break;
                default: ok = false; return true;           // remaining typed/atomic opcodes deferred
            }
            // Per-instruction buffer-op disposition (PROSPER_DBG). A buffer op whose V# decodes to
            // NUM_RECORDS=0 — or that matches one of the no-backing markers below — is folded away:
            // loads become a constant, stores are dropped, and no backing buffer is declared. That is
            // architecturally correct for an empty descriptor. It is also, in the emitted SPIR-V,
            // INDISTINGUISHABLE from an instruction that never reached the emitter at all: both leave
            // no access chain, no load, no trace of any kind. Those two have completely different
            // causes, and offline dissection of a program whose buffer traffic has vanished cannot
            // separate them without this line (#2709 built an argument for a whole new memory model
            // on exactly that ambiguity, and was wrong).
            //
            // Reported from a scope guard rather than at each decision site so EVERY exit path emits
            // exactly one line. A path nobody classified prints `unclassified` rather than printing
            // nothing — so a census can detect its own incompleteness instead of silently
            // under-counting, which a per-site call cannot do. The three pre-switch rejects above are
            // deliberately outside it: they set ok=false, and a rejected program is already reported
            // by the recompile-reject census -- as is the switch's own `default:` arm, which is
            // also above this point. That is FOUR uncovered rejects, not three, and the `default:`
            // one matters most to a census: an unmodelled buffer opcode is literally the "never
            // reached the emitter" case this line exists to identify. So the tempting identity
            // "instructions in the stream == lines emitted" holds only for a program with none of
            // the four; it happened to hold for 0x413dc6700 (41 == 41) and that is not a guarantee.
            // Hoisting the guard above the switch is not the fix: n/is_store/is_atomic are unset
            // there, so it would print `n=0 store=0 atomic=0`, which is less honest than no line.
            //
            // `rt=` is on the line because `unclassified` alone does not mean "instrument hole".
            // recompile_coverage() translates on a table-less compute shell, so every buffer op it
            // sees takes a no-table path and prints `unclassified` for a reason that has nothing to
            // do with coverage. `rt=0` marks those. A hole is `rt=1 ... unclassified`.
            // `descriptor-resolved` IS NOT A SUCCESS CLAIM, and the name is the fix. It is set the
            // moment a V# resolves -- before the format decode runs -- and it reaches the line only
            // when no later path classified the outcome. It used to read `resolved`, and four reject
            // paths returned past it without touching it, so a refused buffer instruction printed the
            // same word as an emitted one (#3579). That is the one distinction this census exists to
            // make, inverted. They are named now (`reject-unknown-format`, `reject-badfmt`,
            // `reject-unaligned`, `reject-subword-int-store`, `reject-mtbuf-unknown-format`), and the
            // word stays deliberately narrow so a FIFTH reject path added later degrades to something
            // TRUE -- "a descriptor was resolved for this op" -- instead of to a false success.
            struct BufOpDisposition {
                bool on;
                const SpirvCompute& b;
                const Rdna2Inst& in;
                const uint32_t& n;
                const bool& is_store;
                const bool& is_atomic;
                const ShaderResourceTable* rt;
                const char* how = "unclassified";
                // The resolved descriptor's DST_SEL word, or UINT32_MAX when this op never reached a
                // resolved V# (#2869). It is on this line rather than behind its own switch so a
                // census of routed descriptors carries its own denominator: every MUBUF FORMAT fetch
                // that resolves prints a selector, so "no title routes a channel" is distinguishable
                // from "the instrument never ran", which a log of the non-identity case alone is not.
                uint32_t dst_sel_word = 0xFFFFFFFFu;
                ~BufOpDisposition() {
                    if (!on) return;
                    char sel[16] = "none";
                    if (dst_sel_word != 0xFFFFFFFFu)
                        std::snprintf(sel, sizeof(sel), "0x%03x", dst_sel_word & 0xfffu);
                    std::fprintf(stderr,
                                 "[buf-op] program=0x%llx pc=%u %s op=0x%x n=%u store=%d atomic=%d rt=%d dstsel=%s %s\n",
                                 (unsigned long long)b.diagnostic.program_address, in.pc,
                                 in.fmt == Rdna2Format::MUBUF ? "MUBUF" : "MTBUF", in.opcode, n,
                                 (int)is_store, (int)is_atomic, (int)(rt != nullptr), sel, how);
                }
            } buf_op{std::getenv("PROSPER_DBG") != nullptr, b, in, n, is_store, is_atomic, rt};
            uint32_t offset = in.literal & 0xFFFu;
            bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
            const bool indirect_pointer_descriptor_source_candidate =
                b.indirect_pointer_proof &&
                rdna2_indirect_pointer_source(*b.indirect_pointer_proof, in);
            // GLC/DLC ordinary loads must observe device-visible writes rather than a cached value;
            // GLC ordinary stores publish through the device cache. Atomics retain their existing
            // Device/AcquireRelease operands and GLC's separate return-pre-op-value contract below.
            const bool coherent_load = !is_store && !is_atomic &&
                                       (in.mubuf_glc || in.mubuf_dlc);
            const bool coherent_store = is_store && in.mubuf_glc;
            if (b.indirect_buffer_dispatch_validated &&
                rdna2_gta5_packed_pointer_atomic_site(in)) {
                // The exact producer at pc347 forces V# stride 256 and pc349..351 set
                // NUM_RECORDS=1, so pc355's qword at byte 24 is in bounds. Its scalar raw-pointer
                // resource was separately proven readable+writable through byte 31 and expanded to
                // that exact binding range. Emit one true qword OR; GLC=0 preserves v0:v1.
                if (!is_atomic_x2 || atomic_op != Op_AtomicOr || is_store || in.mubuf_glc ||
                    !b.storage_buffer_int64_atomics ||
                    b.indirect_buffer_atomic_binding == UINT32_MAX ||
                    b.indirect_buffer_atomic_byte_offset != offset ||
                    (offset & 7u) != 0u) {
                    buf_op.how = "reject-packed-pointer-atomic";
                    ok = false;
                    return true;
                }
                const auto lo = rs.vreg.find(in.dst.value);
                const auto hi = rs.vreg.find(in.dst.value + 1);
                if (lo == rs.vreg.end() || hi == rs.vreg.end()) {
                    buf_op.how = "reject-packed-pointer-atomic";
                    ok = false;
                    return true;
                }
                const uint32_t value = b.u64_from_lohi(lo->second, hi->second);
                const uint32_t access = rs.exec_narrowed ? rs.exec : b.btrue();
                (void)b.cbuf_atomic_x2_rtn(
                    Op_AtomicOr, b.uconst(offset / 8u), value,
                    b.indirect_buffer_atomic_binding, access, b.uconst64(0u));
                buf_op.how = "packed-pointer-atomic";
                return true;
            }
            // PC-relative EMBEDDED TABLE (#273): this load's V# was built from s_getpc_b64 and the
            // table bytes live inside the shader blob — detect_pcrel_tables already copied them out.
            // Fold to a compile-time constant lookup: dword index = (inst offset + offen VADDR) >> 2;
            // out-of-range indexes read 0 (the hardware's OOB contract for a bounded V#).
            //
            // The TYPED consumer of the same idiom (#2859) folds identically. Sonic Frontiers' three
            // Cyber Space scene kernels build the V# exactly as above and read it with
            // `tbuffer_load_format_x v, v, s[0:3], 0 offen` at BUF_FMT 22 (`32_FLOAT`), twice each --
            // so the untyped-only guard, not the idiom, is what refused them. detect_pcrel_tables
            // admits an MTBUF site only when its format stores 32 bits per component and its
            // component count matches the opcode, which makes the typed fetch a raw dword copy; that
            // proof is what licenses relaxing `is_format` here, and only for a pc it recorded.
            const bool mtbuf_pcrel_fold = in.fmt == Rdna2Format::MTBUF &&
                                          rs.mtbuf_pcrel_tables.count(in.pc) != 0;
            if ((!is_format || mtbuf_pcrel_fold) && !is_store && !is_atomic && !raw_subword) {
                const auto& pcrel_tables = mtbuf_pcrel_fold ? rs.mtbuf_pcrel_tables
                                                            : rs.mubuf_pcrel_tables;
                auto pt = pcrel_tables.find(in.pc);
                if (pt != pcrel_tables.end()) {
                    const std::vector<uint32_t>& tab = pt->second;
                    uint32_t addr = b.uconst(offset);
                    if (offen) { Operand ov{OperandKind::VGPR, in.src[0].value};
                                 addr = b.ibin(Op_IAdd, addr, val(ov)); }
                    uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
                    // ISA 70648 section 8.1: formatted loads are bounded all-or-nothing.
                    // Subtract on the host first so a large guest address cannot wrap the check.
                    // MTBUF uses identity DST_SEL (Table 31), so no SEL_1 exception applies here.
                    uint32_t format_in_bounds = 0;
                    if (mtbuf_pcrel_fold)
                        format_in_bounds = n <= tab.size()
                                               ? b.ucmp(Op_ULessThanEqual, addr,
                                                        b.uconst((uint32_t(tab.size()) - n) * 4u))
                                               : b.bfalse();
                    for (uint32_t k = 0; k < n; k++) {
                        int d = in.dst.value + (int)k;
                        uint32_t old = vreg_old(b, rs, d);
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        uint32_t acc = b.uconst(0);   // OOB -> 0
                        for (uint32_t t = 0; t < (uint32_t)tab.size(); t++)
                            acc = b.sel(b.ucmp(Op_IEqual, kidx, b.uconst(t)), b.uconst(tab[t]), acc);
                        if (mtbuf_pcrel_fold) acc = b.sel(format_in_bounds, acc, b.uconst(0));
                        rs.vreg[d] = acc;
                        predicate_write(b, rs, d, old);
                    }
                    buf_op.how = "pcrel-table";
                    return true;
                }
            }
            // A proven PC-relative table is self-contained shader data and needs no runtime resource
            // bindings. Only external buffer accesses require the stage's SMEM/MUBUF resource gate.
            // Keep this check after the fold so resource-free graphics shaders (Astro Bot's loading VS)
            // can consume their embedded lookup table without making unresolved V# accesses permissive.
            if (!allow_smem) { buf_op.how = "reject-no-smem"; ok = false; return true; }
            uint32_t binding = 2, stride = 0;   // overwritten by SRSRC resolution below whenever a resource
                                                // table is present (format AND raw ops); the binding-2 default
                                                // survives only on the table-less offline path (see below)
            // Format of the fetched components. Untyped buffer_load_dword* is raw 32-bit (comp_bytes=4);
            // buffer_load_format_* takes the format from the resolved V# descriptor.
            DataFormat fmt = DataFormat::Uint32;   // untyped default: raw dwords
            uint32_t fmt_ncomp = 0;    // the V#'s real component count (format loads only); 0 = don't default-fill
            bool dyn_vfetch = false;   // set when the V# came from by_fetch_pc — a const-folded per-vertex
                                       // attribute fetch, whose element address is exactly gl_VertexIndex*stride.
            bool instance_vfetch = false;
            bool folded_vfetch = false; // by-fetch V# base already includes OFFSET/SOFFSET
            const ShaderResource* resolved_buffer = nullptr;
            bool gta5_selected_sbuffer_consumer = false;
            bool indirect_pointer_descriptor_source = false;
            if (is_format) {
                // A format load reads a vertex/buffer attribute — it needs the V# descriptor for the
                // binding, stride, and data format. Resolve SRSRC (src[1]) via provenance: an s_load
                // tag (indirect) else the SGPR index (direct/user-data).
                const ShaderResource* res = nullptr;
                // A format load (vertex fetch) reads a VERTEX buffer — resolve the SRSRC SGPR to a vertex
                // buffer specifically (that SGPR may hold a constant-buffer V# at other points; the const-
                // fold-resolved vertex buffer is keyed by this SRSRC SGPR). Fall back to an s_load SRT tag.
                // Any write to the four-dword SRSRC range invalidates its entry-time direct V#.
                // Exact per-fetch and s_load provenance below may still identify the live descriptor,
                // but a missing/rejected dynamic result must never fall back to stale user data.
                //
                // Hoisted out of the `if (rt)` block below because the REJECT diagnostic needs it:
                // whether the shader built its own SRSRC is the single fact that separates "the
                // resource table has no entry for this fetch" from "it has one but direct-SGPR
                // provenance was deliberately suppressed", and those two point at different files
                // (#3137).
                const bool srsrc_rewritten = sreg_range_written(rs, in.src[1].value, 4);
                if (rt) {
                    // PER-FETCH first: a reloaded SRSRC holds a different V# per attribute, so match this
                    // exact fetch instruction's pc; fall back to untouched SGPR user data or an s_load
                    // SRT tag. A rewritten direct descriptor without either provenance stays unresolved.
                    // Only a VERTEX-buffer pc entry implies the vertex-index address model — a pc-keyed
                    // CONSTANT/structured buffer (a PS's per-lane table fetch, #273) keeps the faithful
                    // VADDR*stride+offset address below.
                    res = rt->by_fetch_pc(in.pc);
                    // A pc-keyed entry is not itself proof that OFFSET/SOFFSET was folded into the
                    // bound base. Shader mode deliberately binds DynFetch::unshifted_desc and must
                    // retain both VADDR terms (idxen+offen), OFFSET and SOFFSET exactly as encoded.
                    folded_vfetch = res && res->cls == ResourceClass::VertexBuffer &&
                                      res->fetch_index_mode != VertexFetchIndexMode::Shader;
                    // The NGG fetch-prologue shortcut applies only to an untouched ABI element index. It
                    // must NOT apply after the shader has selected or computed VADDR. DOLL lays out packed
                    // attributes as two/three descriptor records per vertex and computes
                    // 2*vertex+channel / 3*vertex+channel in v0/v4/v5/v6. DQ also uses a modeled
                    // v_cndmask merged-wave selector to choose instance_id for a per-instance transform
                    // lookup and vertex_id for positions. Replacing either shader value with
                    // gl_VertexIndex reads allocator metadata as transform indices and emits NaN/giant
                    // triangles.
                    // A pc-keyed VertexBuffer can also be a directly supplied structured V# whose SRSRC
                    // was never rewritten; retain the established non-v0 faithful-address behavior there.
                    if (res) {
                        const int vaddr = in.src[0].value;
                        switch (res->fetch_index_mode) {
                            case VertexFetchIndexMode::Vertex:
                                dyn_vfetch = true;
                                break;
                            case VertexFetchIndexMode::Instance:
                                instance_vfetch = true;
                                break;
                            case VertexFetchIndexMode::Shader:
                                break;
                            case VertexFetchIndexMode::Automatic:
                                // Backward-compatible fallback for metadata resources, hand-built
                                // tests, and captures predating explicit dynamic-fold provenance.
                                dyn_vfetch = res->cls == ResourceClass::VertexBuffer &&
                                             (vaddr == 0 || srsrc_rewritten);
                                break;
                        }
                    }
                    // MTBUF must resolve through the exact dynamic-use entry. The fold validates the
                    // live V# FORMAT != INVALID before publishing that pc. Falling back to an older
                    // metadata resource here can resurrect an unbound V# that happens to share its
                    // SGPR/SRT identity. MUBUF retains its established metadata fallbacks.
                    if (in.fmt != Rdna2Format::MTBUF) {
                        if (!res && !srsrc_rewritten)
                            res = rt->by_sgpr_base_cls(in.src[1].value, ResourceClass::VertexBuffer);
                        uint32_t srt_tag = 0;
                        if (!res && sreg_srt_range_tag(rs, in.src[1].value, 4, srt_tag))
                            res = rt->by_srt_offset(srt_tag);
                        // DIRECT user-data V# of any class (#273 — DOLL's title post PSes format-fetch
                        // through a V# the metadata labels a CONSTANT buffer sharp at s[24:27]): the class
                        // label doesn't change the descriptor's fields. Only when the SGPR was never
                        // REWRITTEN in-shader (no rs.sreg entry in its four-dword range) — a reloaded
                        // register no longer holds the seed-time sharp, and trusting it would fetch through
                        // a stale descriptor.
                        if (!res && !srsrc_rewritten)
                            res = rt->by_sgpr_base(in.src[1].value);
                    }
                }
                if (!res) {
                    if (getenv("PROSPER_DBG")) {   // which provenance step failed for this format load
                        uint32_t srt_tag = 0;
                        const bool has_srt_tag = sreg_srt_range_tag(
                            rs, in.src[1].value, 4, srt_tag);
                        const ShaderResource* tagged_res = has_srt_tag && rt
                                                               ? rt->by_srt_offset(srt_tag)
                                                               : nullptr;
                        // `pc_res` and `rewritten` are the two fields the RAW path
                        // (`[mubuf-raw-unresolved]`) has always printed and this one did not, and
                        // their absence is what made #3137 diagnosable only by re-running the title
                        // under PROSPER_DYNTRACE_FAIL: with `pc_res=null rewritten=1` the line says
                        // outright that the const-fold published no descriptor for this exact fetch
                        // AND that the shader assembles its own SRSRC, so the direct-SGPR fallbacks
                        // were correctly suppressed rather than tried and missed. That is a
                        // front-half/user-data verdict, not a recompiler one. `rewritten=0` with
                        // `pc_res=null` is the opposite verdict: the table simply has no entry at
                        // this SGPR.
                        const ShaderResource* pc_res = rt ? rt->by_fetch_pc(in.pc) : nullptr;
                        fprintf(stderr, "[mubuf-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x key_res=%s "
                                        "pc_res=%s rewritten=%d (%zu res)\n",
                                in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ",
                                has_srt_tag ? srt_tag : 0u,
                                !rt ? "no-table" : (tagged_res ? "yes" : "null"),
                                !rt ? "no-table" : (pc_res ? "yes" : "null"),
                                (int)srsrc_rewritten,
                                rt ? rt->resources.size() : 0u);
                    }
                    buf_op.how = "reject-unresolved";
                    ok = false; return true;
                }
                if (is_zero_record_raw_buffer(*res)) {
                    // The exact live V# is empty. The producer admitted only selectors whose OOB
                    // result is zero; re-check that contract so a malformed table cannot silently
                    // turn SQ_SEL_1 into zero. Apply the result only to active EXEC lanes, preserving
                    // the old destination in masked lanes, and never declare a backing buffer.
                    if (is_store) { buf_op.how = "reject-zero-record-format-store"; ok = false; return true; }
                    for (uint32_t k = 0; k < n; ++k) {
                        const uint32_t selector = res->swizzle[k];
                        if (selector != 0u && selector < 4u) { buf_op.how = "reject-zero-record-selector"; ok = false; return true; }
                        const int d = in.dst.value + static_cast<int>(k);
                        const uint32_t old = vreg_old(b, rs, d);
                        rs.vreg[d] = b.uconst(0);
                        predicate_write(b, rs, d, old);
                    }
                    buf_op.how = "zero-record";
                    return true;
                }
                buf_op.how = "descriptor-resolved";
                resolved_buffer = res;
                binding = res->binding;
                stride = res->stride;
                if (in.fmt == Rdna2Format::MTBUF) {
                    // Unlike MUBUF format ops, MTBUF owns the type in the instruction. gfx1030 uses
                    // the same combined 7-bit BUF_FMT table as Gen5 V# descriptors.
                    rdna2_buffer_format(in.mtbuf_format, &fmt, &fmt_ncomp);
                    if (fmt == DataFormat::Unknown || fmt_ncomp == 0) {
                        // Named separately from the MUBUF `reject-unknown-format` beside it: the two
                        // read the format from different places -- this one from the INSTRUCTION's
                        // 7-bit BUF_FMT, that one from the resolved descriptor -- so a census that
                        // merged them could not say which source was undecodable. Reached by every
                        // USCALED/SSCALED code, none of which `rdna2_buffer_format` has a case for.
                        buf_op.how = "reject-mtbuf-unknown-format";
                        ok = false; return true;
                    }
                } else {
                    fmt = res->format;
                    fmt_ncomp = res->num_components;   // format default-fill below (#368)
                }
            } else if (rt) {
                // RAW (untyped) MUBUF with a resource table: resolve SRSRC (src[1]) exactly like the
                // format path — a raw buffer op targets whatever buffer its V# describes, NOT a fixed
                // binding (#91: the old hardcoded binding-2 silently read/wrote the wrong buffer for
                // any other target). Raw ops don't imply a resource class the way a format load implies
                // VertexBuffer, so the direct-SGPR lookup is class-unrestricted (by_sgpr_base).
                // Provenance order mirrors the format path: exact fetch pc, then s_load SRT tag
                // (indirect), then user-data SGPR (direct).
                const ShaderResource* res = rt->by_fetch_pc(in.pc);
                uint32_t srt_tag = 0;
                const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 4, srt_tag);
                if (!res && has_srt_tag)
                    res = rt->by_srt_offset(srt_tag);
                if (!res && !sreg_range_written(rs, in.src[1].value, 4))
                    res = rt->by_sgpr_base(in.src[1].value);
                if (!res) {
                    if (getenv("PROSPER_DBG")) {
                        const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                        fprintf(stderr,
                                "[mubuf-raw-unresolved] rt=%p pc=%u srsrc=s%d srt_tag=%s0x%x "
                                "key_res=%s pc_res=%s rewritten=%d (%zu res)\n",
                                (const void*)rt,
                                in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ", srt_tag,
                                has_srt_tag && rt->by_srt_offset(srt_tag) ? "yes" : "null",
                                pp ? "yes" : "null",
                                sreg_range_written(rs, in.src[1].value, 4), rt->resources.size());
                    }
                    buf_op.how = "reject-raw-unresolved";
                    ok = false; return true;   // unresolvable V# -> reject; NEVER default to binding 2
                }
                if (indirect_pointer_descriptor_source_candidate) {
                    const bool zero_soffset =
                        (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                        (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    const bool exact_source =
                        b.indirect_pointer_proof->bound_kind ==
                            IndirectPointerBoundKind::DescriptorRange &&
                        b.indirect_pointer_proof->guard_kind ==
                            IndirectPointerGuardKind::None &&
                        b.indirect_pointer_proof->source_address_kind ==
                            IndirectBufferRelocationRecord::SourceAddressKind::
                                BufferDescriptorBase48 &&
                        in.fmt == Rdna2Format::MUBUF && !is_format && !is_store &&
                        !is_atomic && !raw_subword && n == 2u && idxen && !offen &&
                        offset == b.indirect_pointer_proof->pointer_byte_offset &&
                        zero_soffset && in.dst.kind == OperandKind::VGPR &&
                        in.dst.value >= 0 &&
                        static_cast<uint32_t>(in.dst.value) ==
                            b.indirect_pointer_proof->source_result_vgpr &&
                        in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 &&
                        static_cast<uint32_t>(in.src[0].value) ==
                            b.indirect_pointer_proof->source_record_index_vgpr &&
                        res->fetch_pc == b.indirect_pointer_proof->source_fetch_pc &&
                        is_indirect_pointer_relocation_resource(*res) &&
                        res->binding == b.indirect_pointer_binding &&
                        res->size == b.indirect_pointer_source_bytes &&
                        res->stride == b.indirect_pointer_source_stride &&
                        b.indirect_pointer_source_bytes % sizeof(uint32_t) == 0u &&
                        b.indirect_pointer_source_record_var &&
                        b.indirect_pointer_source_root_lo_var &&
                        b.indirect_pointer_source_root_hi_var;
                    if (!exact_source) {
                        buf_op.how = "reject-indirect-pointer-source";
                        ok = false;
                        return true;
                    }
                    indirect_pointer_descriptor_source = true;
                }
                if (is_gta5_cf9200_no_backing_marker_candidate(*res)) {
                    const GtaCf9200NoBackingAccess access =
                        rdna2_gta5_cf9200_no_backing_site(in);
                    if (!b.gta5_cf9200_no_backing_dispatch_validated ||
                        !is_proven_gta5_cf9200_no_backing(*res) ||
                        res->fetch_pc != in.pc || is_format || is_atomic ||
                        access == GtaCf9200NoBackingAccess::None ||
                        (access == GtaCf9200NoBackingAccess::LoadZero && is_store) ||
                        (access == GtaCf9200NoBackingAccess::DropStore && !is_store)) {
                        buf_op.how = "reject-cf9200";
                        ok = false;
                        return true;
                    }
                    if (access == GtaCf9200NoBackingAccess::LoadZero) {
                        const int destination = in.dst.value;
                        const uint32_t old = vreg_old(b, rs, destination);
                        rs.vreg[destination] = b.uconst(0u);
                        predicate_write(b, rs, destination, old);
                    }
                    buf_op.how = "cf9200-no-backing";
                    return true;
                }
                if (is_proven_null_nullable_raw_buffer(*res)) {
                    // The final compile boundary already re-established the exact production bytes,
                    // s7-sized x256 launch, marker set, and retained +0x20 zero-qword witness. Keep the
                    // instruction half fail-closed too: only one dword load may synthesize zero and
                    // only the three exact dword stores may disappear.
                    const Gta5NullableOutputAccess access =
                        rdna2_gta5_nullable_output_site(in);
                    if (res->fetch_pc != in.pc || is_format || is_atomic ||
                        access == Gta5NullableOutputAccess::None ||
                        (access == Gta5NullableOutputAccess::LoadDword && is_store) ||
                        (access == Gta5NullableOutputAccess::StoreDword && !is_store)) {
                        buf_op.how = "reject-proven-null-nullable";
                        ok = false;
                        return true;
                    }
                    if (access == Gta5NullableOutputAccess::LoadDword) {
                        const int d = in.dst.value;
                        const uint32_t old = vreg_old(b, rs, d);
                        rs.vreg[d] = b.uconst(0);
                        predicate_write(b, rs, d, old);
                    }
                    buf_op.how = "proven-null-nullable";
                    return true;
                }
                if (is_optional_null_raw_load_buffer(*res)) {
                    // Unlike NUM_RECORDS=0, this marker records an application-level optional
                    // entry, not an architectural empty descriptor. Only the exact admitted RAW
                    // dword load may consume it; stores, atomics, wider loads, and packet variants
                    // stay fail-visible rather than inheriting drop-write behavior.
                    if (!rdna2_optional_null_raw_load_shape(in)) {
                        buf_op.how = "reject-optional-null-load";
                        ok = false;
                        return true;
                    }
                    const int d = in.dst.value;
                    const uint32_t old = vreg_old(b, rs, d);
                    rs.vreg[d] = b.uconst(0);
                    predicate_write(b, rs, d, old);
                    buf_op.how = "optional-null-load";
                    return true;
                }
                if (is_proven_null_guarded_raw_store(*res)) {
                    // The front half proved the exact pc74/76/78 consumer lies in GTA V's
                    // dispatch-specialized null-output region. Accept only the raw store family;
                    // a load, typed store, or atomic at the same pc must remain fail-visible rather
                    // than inheriting a no-op from stale/malformed resource metadata.
                    const bool supported_store = in.fmt == Rdna2Format::MUBUF && is_store &&
                        !is_format && !is_atomic &&
                        (in.opcode == kMubufOpcodeStoreDword ||
                         in.opcode == kMubufOpcodeStoreDwordX2 ||
                         in.opcode == kMubufOpcodeStoreDwordX4 ||
                         in.opcode == kMubufOpcodeStoreDwordX3) &&
                        rdna2_gta5_null_guarded_raw_store_site(in);
                    if (!supported_store || res->fetch_pc != in.pc) {
                        buf_op.how = "reject-null-guarded-store";
                        ok = false;
                        return true;
                    }
                    buf_op.how = "null-guarded-store";
                    return true;
                }
                if (is_zero_record_raw_buffer(*res)) {
                    // Safe at the TOP here, unlike the format path: none of this block's three arms
                    // (atomic, store-drop, load-fold) sets ok=false, so no later outcome can
                    // contradict the label. A fail-visible guard added below must move this down.
                    buf_op.how = "zero-record";
                    // The front half proved all four live V# words at this exact instruction and
                    // decoded NUM_RECORDS=0. RDNA2's OOB contract returns zero for every raw load
                    // component, drops raw stores, and performs no memory operation for an atomic.
                    // For atomics GLC=0 leaves VDATA untouched; GLC=1 returns the pre-op value, which
                    // is zero for an empty descriptor. Apply that return only to active EXEC lanes,
                    // exactly like an ordinary predicated VGPR write. Never declare or touch a dummy
                    // binding: one empty operation cannot then leak state into another.
                    if (is_atomic) {
                        if (in.mubuf_glc) {
                            const int d = in.dst.value;
                            const uint32_t old = vreg_old(b, rs, d);
                            rs.vreg[d] = b.uconst(0);
                            predicate_write(b, rs, d, old);
                        }
                        return true;
                    }
                    if (is_store) return true;
                    for (uint32_t k = 0; k < n; ++k) {
                        const int d = in.dst.value + static_cast<int>(k);
                        const uint32_t old = vreg_old(b, rs, d);
                        rs.vreg[d] = b.uconst(0);
                        predicate_write(b, rs, d, old);
                    }
                    return true;
                }
                const bool exact_consumer =
                    (in.pc == 156u && in.opcode == kMubufOpcodeLoadDwordX4 && n == 4u &&
                     in.words[0] == 0xe0382000u && in.words[1] == 0x80020006u) ||
                    (in.pc == 158u && in.opcode == kMubufOpcodeLoadDwordX2 && n == 2u &&
                     in.words[0] == 0xe0342010u && in.words[1] == 0x80020406u);
                if (b.gta5_selected_sbuffer_dispatch_validated && exact_consumer) {
                    const bool exact_operands = in.fmt == Rdna2Format::MUBUF && !is_format &&
                        !is_store && !is_atomic && !raw_subword && in.len_dwords == 2u &&
                        in.dst.kind == OperandKind::VGPR &&
                        in.dst.value == (in.pc == 156u ? 0 : 4) &&
                        in.src[0].kind == OperandKind::VGPR && in.src[0].value == 6 &&
                        in.src[1].kind == OperandKind::SGPR && in.src[1].value == 8 &&
                        in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0;
                    if (!exact_operands) {
                        buf_op.how = "reject-selected-sbuffer-operands";
                        ok = false;
                        return true;
                    }
                    gta5_selected_sbuffer_consumer = true;
                }
                buf_op.how = "descriptor-resolved";
                resolved_buffer = res;
                binding = res->binding;
                stride  = res->stride;
                // fmt stays raw Uint32: untyped ops move raw dwords regardless of the V#'s declared format.
            }
            if (raw_subword) {
                if (raw_bits == 8) fmt = raw_signed ? DataFormat::Sint8 : DataFormat::Uint8;
                else               fmt = raw_signed ? DataFormat::Sint16 : DataFormat::Uint16;
            }
            // The checkpoint array ABI models read-only raw buffer accesses. Typed MUBUF default-fill
            // and descriptor swizzle semantics need per-selected-entry treatment; stores and atomics
            // additionally need writeback authority. Reject all three before any helper can form an
            // access through the array variable.
            if (resolved_buffer && resolved_buffer->table_index_count != 0u &&
                (is_format || is_store || is_atomic)) {
                ok = false;
                return true;
            }
            if (is_atomic_x2) {
                const bool zero_soffset =
                    (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                    (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                const bool exact_shape = b.storage_buffer_int64_atomics && resolved_buffer &&
                    resolved_buffer->fetch_pc == in.pc &&
                    resolved_buffer->table_index_count == 0u &&
                    resolved_buffer->stride == 8u &&
                    resolved_buffer->atomic_x2_record_count != 0u &&
                    resolved_buffer->atomic_x2_record_count <= 0x02000000u &&
                    static_cast<uint64_t>(resolved_buffer->atomic_x2_record_count) * 8u ==
                        resolved_buffer->size &&
                    (resolved_buffer->gpu_addr & 7u) == 0u && idxen && !offen &&
                    offset == 0u && zero_soffset && !in.mubuf_dlc &&
                    !in.mubuf_lds && (in.words[0] & 0x00020000u) == 0u &&
                    (in.words[1] & 0x00e00000u) == 0u &&
                    in.src[0].kind == OperandKind::VGPR;
                if (!exact_shape) { ok = false; return true; }
                atomic_x2_record_count = resolved_buffer->atomic_x2_record_count;
            }
            // else (rt == nullptr): table-less offline compute shell (recompile_valu without a resource
            // table — the unit-test harness). Keep the legacy single-cbuf convention: binding 2, stride 0.
            // The live graphics path can never reach here table-less: recompile_vertex/recompile_fragment
            // set allow_smem = (rt != nullptr), so MUBUF already rejected above when rt is null there.
            const bool packed_10_11_11 = fmt == DataFormat::Float10_11_11;
            const bool packed_2_10_10_10 =
                fmt == DataFormat::Unorm2_10_10_10 || fmt == DataFormat::Snorm2_10_10_10 ||
                fmt == DataFormat::Uint2_10_10_10  || fmt == DataFormat::Sint2_10_10_10;
            const bool packed_word = packed_10_11_11 || packed_2_10_10_10;
            const uint32_t comp_bytes = data_format_bytes(fmt);
            if (!packed_word && comp_bytes == 0) {                            // unknown / unsupported
                buf_op.how = "reject-unknown-format"; ok = false; return true;
            }
            // Per-component decode. 4-byte formats (Float32/Uint32/Sint32) are a raw dword load — no
            // conversion in our bit model. Sub-dword formats are unpacked: UNORM/SNORM normalize an
            // integer field, Float16 unpacks a packed half. num_components components pack tightly.
            const bool packed = packed_word || comp_bytes < 4;
            bool is_snorm = (fmt == DataFormat::Snorm8 || fmt == DataFormat::Snorm16 ||
                             fmt == DataFormat::Snorm2_10_10_10);
            bool is_half  = (fmt == DataFormat::Float16);
            // Integer sub-dword formats deliver the raw (un-normalized) INTEGER in the VGPR — the
            // hardware's UINT/SINT format-load contract. DOLL's skinned scene VS fetches its bone
            // indices as Uint8 x4 (stride 8, paired with Unorm8 weights); rejecting them dropped
            // every scene-geometry draw (#273). Zero-/sign-extend the field; no normalization.
            bool is_uint = (fmt == DataFormat::Uint8 || fmt == DataFormat::Uint16 ||
                            fmt == DataFormat::Uint2_10_10_10);
            bool is_sint = (fmt == DataFormat::Sint8 || fmt == DataFormat::Sint16 ||
                            fmt == DataFormat::Sint2_10_10_10);
            // A buffer_load_FORMAT whose V# type is a sub-dword integer (Uint8/Sint8/Uint16/Sint16)
            // reads tightly-packed integer components at a RUNTIME byte address — exactly like the raw
            // buffer_load_ubyte/ushort path, NOT the descriptor-defined statically-aligned packing the
            // norm/half formats use. DOLL's post-process LUT compute kernels index a stride-1 Uint8
            // table this way (`buffer_load_format_x v, vIDX, V#`, idxen), which the aligned packed path
            // below wrongly rejected as unaligned. Handle it with a runtime byte/halfword extract.
            const bool int_subword = is_format && !raw_subword && (is_uint || is_sint) &&
                                     (comp_bytes == 1 || comp_bytes == 2);
            const bool zero_soffset =
                (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
            // Vulkan exposes storage buffers as u32 arrays in Prosper's portable ABI. Admit a two-byte
            // guest range only for Plucky Squire's exact one-record UINT16/FLOAT16 scalar format fetch:
            // the original element index itself is checked against zero, the sole load is constant
            // dword 0, and the host supplies a zero upper half. No raw/store/atomic/offset variant can
            // inherit this contract.
            const StorageBufferTailSemantic one_record_tail_semantic = !resolved_buffer
                ? StorageBufferTailSemantic::None
                : resolved_buffer->format == DataFormat::Uint16
                    ? StorageBufferTailSemantic::Uint16
                    : resolved_buffer->format == DataFormat::Float16
                        ? StorageBufferTailSemantic::Float16
                        : StorageBufferTailSemantic::None;
            const bool one_record_16bit_tail =
                in.fmt == Rdna2Format::MUBUF && in.opcode == 0u && n == 1u &&
                one_record_tail_semantic != StorageBufferTailSemantic::None &&
                resolved_buffer->num_components == 1u && resolved_buffer->stride == 2u &&
                resolved_buffer->size == 2u && idxen && !offen && offset == 0u && zero_soffset &&
                !folded_vfetch && in.src[0].kind == OperandKind::VGPR;
            // V# DST_SEL channel routing (#2869). The descriptor names, per RETURNED channel, which
            // STORED component it takes -- SQ_SEL 4/5/6/7 = R/G/B/A -- or that the channel is the
            // constant 0 (SQ_SEL_0) or the constant 1 (SQ_SEL_1); 2 and 3 are reserved. WHICH ops
            // apply it is not a matter of inference: RDNA2 ISA (document 70648) sec. 8.1.4 "Buffer
            // Data" states "Dst_sel comes from the resource, but is ignored for many operations", and
            // its Table 31 "Buffer Instructions" gives the whole partition in one column --
            //
            //     BUFFER_LOAD_FORMAT_* / BUFFER_STORE_FORMAT_*  ->  DST SEL = resource
            //     TBUFFER_LOAD_FORMAT_* / TBUFFER_STORE_FORMAT_* -> DST SEL = identity
            //     BUFFER_LOAD_<type> / BUFFER_STORE_<type> / BUFFER_ATOMIC_* -> DST SEL = identity
            //
            // -- where the same table defines identity as "X000, XY00, XYZ0, or XYZW" by the data
            // format's component count. So MTBUF is genuinely exempt, which is why this is gated on
            // MUBUF and not merely on `is_format`, and a raw load ignores the field entirely. That
            // also reconciles the two comments #2869 was filed over: both were true, of different
            // instruction families. Until this landed no format lowering consulted `res->swizzle` at
            // all (and no V# ever populated it), so a routed MUBUF descriptor returned the STORED
            // order -- a plausible wrong picture, with no reject and no diagnostic.
            uint32_t dst_sel[4] = {4u, 5u, 6u, 7u};
            bool dst_sel_routed = false;   // some requested channel is not its identity source
            if (is_format && in.fmt == Rdna2Format::MUBUF && resolved_buffer) {
                for (uint32_t k = 0; k < 4u; ++k) dst_sel[k] = resolved_buffer->swizzle[k];
                for (uint32_t k = 0; k < n && k < 4u; ++k)
                    if (dst_sel[k] != k + 4u) dst_sel_routed = true;
                buf_op.dst_sel_word = (dst_sel[0] & 7u) | ((dst_sel[1] & 7u) << 3) |
                                      ((dst_sel[2] & 7u) << 6) | ((dst_sel[3] & 7u) << 9);
            }
            if (dst_sel_routed) {
                // SQ_SEL 2 and 3 are reserved. Fail visible rather than guess a meaning for them.
                for (uint32_t k = 0; k < n && k < 4u; ++k)
                    if (dst_sel[k] == 2u || dst_sel[k] == 3u) {
                        buf_op.how = "reject-dst-sel-reserved";
                        ok = false; return true;
                    }
                // Table 31 says BUFFER_STORE_FORMAT_* takes DST SEL from the resource too, so the
                // reject below is a known GAP rather than a settled contract -- it is the DIRECTION
                // the document does not spell out that stops it. For a store the mapping runs the
                // other way (which VDATA channel supplies each stored component), and a constant
                // selector has no stated meaning on the way OUT at all. Rejecting drops the store
                // visibly; applying the load's mapping unchanged would write components in the wrong
                // order into guest memory, which nothing downstream could detect (#3549).
                // CONFIDENCE: LOW on store routing.
                //
                // It is narrower than `dst_sel_routed` on purpose, and the difference is the common
                // case rather than a corner: only a selector among components that PHYSICALLY EXIST
                // can change the stored bytes. A three-component format's canonical selector is
                // X/Y/Z/1 (0x3AC) -- the routine encoding every driver emits, and the shape GTA V's
                // tables carry (#2481) -- whose SQ_SEL_1 in the W slot names no stored component at
                // all. Gating the store on `dst_sel_routed` would have rejected every
                // buffer_store_format_xyzw through an ordinary three-component descriptor, turning
                // this fix into a draw-dropping regression wearing a fail-visible label.
                if (is_store) {
                    const uint32_t stored = fmt_ncomp < n ? fmt_ncomp : n;
                    for (uint32_t k = 0; k < stored && k < 4u; ++k)
                        if (dst_sel[k] != k + 4u) {
                            buf_op.how = "reject-dst-sel-store";
                            ok = false; return true;
                        }
                }
            }
            return ir::BufferAccessRoutes{
                {b, rs, in, ok, allow_wave},
                {.fmt = fmt,
                 .n = n,
                 .comp_bytes = comp_bytes,
                 .stride = stride,
                 .offset = offset,
                 .binding = binding,
                 .atomic_x2_record_count = atomic_x2_record_count,
                 .atomic_op = atomic_op,
                 .fmt_ncomp = fmt_ncomp,
                 .raw_bits = raw_bits,
                 .one_record_tail_semantic = one_record_tail_semantic,
                 .packed = packed,
                 .packed_word = packed_word,
                 .is_half = is_half,
                 .is_uint = is_uint,
                 .is_sint = is_sint,
                 .raw_subword = raw_subword,
                 .folded_vfetch = folded_vfetch,
                 .idxen = idxen,
                 .offen = offen,
                 .is_store = is_store,
                 .int_subword = int_subword,
                 .dyn_vfetch = dyn_vfetch,
                 .instance_vfetch = instance_vfetch,
                 .indirect_pointer_descriptor_source = indirect_pointer_descriptor_source,
                 .coherent_load = coherent_load,
                 .gta5_selected_sbuffer_consumer = gta5_selected_sbuffer_consumer,
                 .is_atomic_x2 = is_atomic_x2,
                 .is_atomic = is_atomic,
                 .is_atomic_fminmax = is_atomic_fminmax,
                 .atomic_fmin = atomic_fmin,
                 .coherent_store = coherent_store,
                 .is_format = is_format,
                 .packed_10_11_11 = packed_10_11_11,
                 .is_snorm = is_snorm,
                 .dst_sel_routed = dst_sel_routed,
                 .one_record_16bit_tail = one_record_16bit_tail,
                 .raw_signed = raw_signed,
                 .dst_sel = dst_sel,
                 .buf_op = {buf_op.how}}}
                .lower();
        }
        case Rdna2Format::MIMG: {
            // Image op. Needs the resource table for the binding, so gated on allow_smem + rt. Two paths,
            // selected by the resolved resource's class: a STORAGE image (image_load/image_store, no
            // sampler — compute copy/blit) or a sampled TEXTURE (image_sample* / image_load via a combined
            // image+sampler). Other opcodes / NSA / gradient / compare variants are rejected (deferred).
            if (!allow_smem || !rt) { ok = false; return true; }
            const uint32_t SQ_DIM_2D = 1u;
            const uint32_t SQ_DIM_3D = 2u;
            const uint32_t SQ_DIM_2D_ARRAY = 5u;   // (x, y, layer) -- #2265
            begin_mimg_source_proof(rs, in.pc);
            auto vread = [&](int r) { return read_mimg_source_word(b, rs, r); };

            // IMAGE_BVH_INTERSECT_RAY (GFX10 opcode 0xe6) has an image encoding but consumes a
            // four-dword BVH descriptor and eleven NSA VGPR operands. Vulkan ray-query support is
            // not available on every backend/device Prosper supports, so lower the exact RTIP 1.1
            // contract used by Astro Bot to ordinary read-only SSBO loads and scalar ALU. The
            // front-half admits only TYPE=8, triangle-return-mode=1 descriptors; keeping
            // the instruction gate equally narrow makes every unverified variant fail visibly.
            if (in.opcode == 0xe6u) {
                const ShaderResource* bvh = rt->by_fetch_pc(in.pc);
                if (in.words[0] != 0xf1989f07u || in.len_dwords != 5u ||
                    in.mimg_dmask != 0xfu || !in.mimg_unorm || in.mimg_dim != 0u ||
                    in.mimg_glc || in.src[2].value != 0 || (in.words[4] & 0xffff0000u) != 0u ||
                    !bvh || bvh->cls != ResourceClass::ConstantBuffer ||
                    bvh->format != DataFormat::Uint32 || bvh->num_components != 1u ||
                    bvh->size < 64u || (bvh->size & 3u) != 0u) {
                    ok = false;
                    return true;
                }

                // The front-half only creates this marker when one mapped zero qword load taints all
                // four descriptor words and the ray instruction is dominated by an EXEC-narrowing
                // guard with no external region entry. This is an explicit empty acceleration
                // structure for this dispatch, not a fallback for an unresolved descriptor.
                if (is_proven_null_bvh(*bvh)) {
                    for (uint32_t k = 0; k < 4; ++k) {
                        const int vd = in.dst.value + static_cast<int>(k);
                        const uint32_t old = vreg_old(b, rs, vd);
                        rs.vreg[vd] = b.uconst(0xffffffffu);
                        predicate_write(b, rs, vd, old);
                    }
                    return true;
                }

                auto addr_vgpr = [&](uint32_t k) { return mimg_address_vgpr(rs, in, k); };
                uint32_t a[11]{};
                for (uint32_t k = 0; k < 11; ++k) a[k] = vread(addr_vgpr(k));

                const uint32_t node = a[0];
                const uint32_t node_type = b.ibin(Op_BitwiseAnd, node, b.uconst(7u));
                const uint32_t node_offset = b.ibin(Op_BitwiseAnd, node, b.uconst(~7u));
                const uint32_t word_base = b.ibin(Op_ShiftLeftLogical, node_offset, b.uconst(1u));
                const uint32_t dword_count = bvh->size / 4u;
                const uint32_t valid64 = b.ucmp(
                    Op_ULessThanEqual, node_offset, b.uconst((bvh->size - 64u) / 8u));
                const uint32_t valid128 = bvh->size < 128u ? b.bfalse() : b.ucmp(
                    Op_ULessThanEqual, node_offset, b.uconst((bvh->size - 128u) / 8u));

                // Every speculative load is independently clamped to dword zero. Results from a
                // malformed/out-of-range pointer are discarded below, but the clamp also prevents
                // an invalid guest pointer from becoming an out-of-bounds Vulkan SSBO access.
                auto load_node = [&](uint32_t rel) {
                    const uint32_t idx = rel
                        ? b.ibin(Op_IAdd, word_base, b.uconst(rel)) : word_base;
                    const uint32_t safe_idx = b.sel(
                        b.ucmp(Op_ULessThan, idx, b.uconst(dword_count)), idx, b.uconst(0u));
                    return b.cbuf_load(safe_idx, bvh->binding);
                };
                uint32_t w[28]{};
                for (uint32_t k = 0; k < 28; ++k) w[k] = load_node(k);

                const uint32_t zero = b.uconst(0u);
                const uint32_t one = b.uconst(fbits(1.0f));
                const uint32_t invalid = b.uconst(0xffffffffu);
                const uint32_t is_tri0 = b.ucmp(Op_IEqual, node_type, b.uconst(0u));
                const uint32_t is_tri1 = b.ucmp(Op_IEqual, node_type, b.uconst(1u));
                const uint32_t is_box16 = b.ucmp(Op_IEqual, node_type, b.uconst(4u));
                const uint32_t is_box32 = b.ucmp(Op_IEqual, node_type, b.uconst(5u));
                const uint32_t tri_valid = b.land(b.lor(is_tri0, is_tri1), valid64);
                const uint32_t box_valid = b.lor(b.land(is_box16, valid64),
                                                 b.land(is_box32, valid128));

                auto fadd = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FAdd, x, y); };
                auto fsub = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FSub, x, y); };
                auto fmul = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FMul, x, y); };
                auto dot3 = [&](const uint32_t x[3], const uint32_t y[3]) {
                    return fadd(fadd(fmul(x[0], y[0]), fmul(x[1], y[1])), fmul(x[2], y[2]));
                };
                auto cross3 = [&](const uint32_t x[3], const uint32_t y[3], uint32_t out[3]) {
                    out[0] = fsub(fmul(x[1], y[2]), fmul(x[2], y[1]));
                    out[1] = fsub(fmul(x[2], y[0]), fmul(x[0], y[2]));
                    out[2] = fsub(fmul(x[0], y[1]), fmul(x[1], y[0]));
                };

                // Triangle node types 0 and 1 share a 64-byte quad. Type 0 selects V0,V1,V2;
                // type 1 selects V1,V3,V2. The four hardware return values are the unnormalised
                // t numerator, determinant, I numerator and J numerator (mode 1).
                uint32_t tv0[3] = {
                    b.sel(is_tri1, w[3], w[0]),
                    b.sel(is_tri1, w[4], w[1]),
                    b.sel(is_tri1, w[5], w[2]),
                };
                uint32_t tv1[3] = {
                    b.sel(is_tri1, w[9], w[3]),
                    b.sel(is_tri1, w[10], w[4]),
                    b.sel(is_tri1, w[11], w[5]),
                };
                uint32_t tv2[3] = {w[6], w[7], w[8]};
                uint32_t edge1[3], edge2[3], from_v0[3];
                for (uint32_t k = 0; k < 3; ++k) {
                    edge1[k] = fsub(tv1[k], tv0[k]);
                    edge2[k] = fsub(tv2[k], tv0[k]);
                    from_v0[k] = fsub(a[2u + k], tv0[k]);
                }
                uint32_t ray_dir[3] = {a[5], a[6], a[7]};
                uint32_t s1[3], s2[3];
                cross3(ray_dir, edge2, s1);
                cross3(from_v0, edge1, s2);
                const uint32_t t_num = dot3(edge2, s2);
                const uint32_t t_denom = dot3(s1, edge1);
                const uint32_t i_num = dot3(from_v0, s1);
                const uint32_t j_num = dot3(ray_dir, s2);
                const uint32_t t = b.fbin(Op_FDiv, t_num, t_denom);
                const uint32_t bary_i = b.fbin(Op_FDiv, i_num, t_denom);
                const uint32_t bary_j = b.fbin(Op_FDiv, j_num, t_denom);
                uint32_t tri_miss = b.lor(
                    b.fcmp(Op_FOrdLessThan, bary_i, zero),
                    b.fcmp(Op_FOrdGreaterThan, bary_i, one));
                tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdLessThan, bary_j, zero));
                tri_miss = b.lor(tri_miss,
                    b.fcmp(Op_FOrdGreaterThan, fadd(bary_i, bary_j), one));
                tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdLessThan, t, zero));
                uint32_t tri_out[4] = {
                    b.sel(tri_miss, b.uconst(0x7f800000u), t_num),
                    b.sel(tri_miss, one, t_denom),
                    i_num,
                    j_num,
                };

                // Undo the pair-compression vertex rotation encoded in the final triangle-node
                // dword. The two 2-bit fields select from {1-I-J, I, J}; valid RTIP 1.1 nodes never
                // use selector 3, whose conservative fallback here is 1-I-J.
                const uint32_t bary0_num = fsub(fsub(tri_out[1], i_num), j_num);
                const uint32_t tri_shift = b.ibin(
                    Op_ShiftLeftLogical,
                    b.ibin(Op_BitwiseAnd, node_type, b.uconst(3u)), b.uconst(3u));
                const uint32_t i_selector = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, w[15], tri_shift), b.uconst(3u));
                const uint32_t j_selector = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, w[15],
                           b.ibin(Op_IAdd, tri_shift, b.uconst(2u))),
                    b.uconst(3u));
                auto swizzled_bary = [&](uint32_t selector) {
                    return b.sel(b.ucmp(Op_IEqual, selector, b.uconst(1u)), i_num,
                        b.sel(b.ucmp(Op_IEqual, selector, b.uconst(2u)), j_num,
                              bary0_num));
                };
                tri_out[2] = swizzled_bary(i_selector);
                tri_out[3] = swizzled_bary(j_selector);

                // Convert each FP16 box payload to the same six-float representation as an FP32
                // box, then apply the slab test to all four children. BOX_SORT_EN orders valid hits
                // by increasing near intersection time; misses receive +INF and therefore remain at
                // the end. Strict compare-swaps retain physical child order for equal-time hits.
                uint32_t box_out[4]{};
                uint32_t box_key[4]{};
                uint32_t ray_inv[3] = {a[8], a[9], a[10]};
                for (uint32_t child = 0; child < 4; ++child) {
                    const uint32_t hbase = 4u + child * 3u;
                    uint32_t fp16_bounds[6] = {
                        b.unpack_half(w[hbase], 0u), b.unpack_half(w[hbase], 1u),
                        b.unpack_half(w[hbase + 1u], 0u), b.unpack_half(w[hbase + 1u], 1u),
                        b.unpack_half(w[hbase + 2u], 0u), b.unpack_half(w[hbase + 2u], 1u),
                    };
                    const uint32_t fbase = 4u + child * 6u;
                    uint32_t bmin[3], bmax[3];
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        bmin[axis] = b.sel(is_box16, fp16_bounds[axis], w[fbase + axis]);
                        bmax[axis] = b.sel(is_box16, fp16_bounds[3u + axis], w[fbase + 3u + axis]);
                    }
                    uint32_t near_axis[3], far_axis[3];
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        const uint32_t t0 = fmul(fsub(bmin[axis], a[2u + axis]), ray_inv[axis]);
                        const uint32_t t1 = fmul(fsub(bmax[axis], a[2u + axis]), ray_inv[axis]);
                        const uint32_t positive = b.fcmp(Op_FOrdGreaterThanEqual, ray_inv[axis], zero);
                        near_axis[axis] = b.sel(positive, t0, t1);
                        far_axis[axis] = b.sel(positive, t1, t0);
                    }
                    const uint32_t near_unclamped = b.fext2(
                        Glsl_FMax, b.fext2(Glsl_FMax, near_axis[0], near_axis[1]), near_axis[2]);
                    const uint32_t far_unclamped = b.fext2(
                        Glsl_FMin, b.fext2(Glsl_FMin, far_axis[0], far_axis[1]), far_axis[2]);
                    const uint32_t near_t = b.fext2(Glsl_FMax, near_unclamped, zero);
                    const uint32_t far_t = b.fext2(Glsl_FMin, far_unclamped, a[1]);
                    const uint32_t nan_interval = b.lor(
                        b.fcmp(Op_FUnordNotEqual, near_unclamped, near_unclamped),
                        b.fcmp(Op_FUnordNotEqual, far_unclamped, far_unclamped));
                    const float box_multiplier = 1.0f +
                        static_cast<float>(bvh->bvh_box_grow) * (1.0f / 16777216.0f);
                    const uint32_t hit = b.land(
                        b.logical_not(nan_interval),
                        b.fcmp(Op_FOrdLessThanEqual, near_t,
                               fmul(far_t, b.uconst(std::bit_cast<uint32_t>(box_multiplier)))));
                    const uint32_t valid_hit = b.land(box_valid, hit);
                    box_out[child] = b.sel(valid_hit, w[child], invalid);
                    box_key[child] = b.sel(valid_hit, near_t, b.uconst(0x7f800000u));
                }
                if (bvh->bvh_sort_enabled) {
                    auto compare_swap = [&](uint32_t left, uint32_t right) {
                        const uint32_t left_key = box_key[left];
                        const uint32_t right_key = box_key[right];
                        const uint32_t left_out = box_out[left];
                        const uint32_t right_out = box_out[right];
                        const uint32_t swap = b.fcmp(
                            Op_FOrdGreaterThan, left_key, right_key);
                        box_key[left] = b.sel(swap, right_key, left_key);
                        box_key[right] = b.sel(swap, left_key, right_key);
                        box_out[left] = b.sel(swap, right_out, left_out);
                        box_out[right] = b.sel(swap, left_out, right_out);
                    };
                    compare_swap(0, 1);
                    compare_swap(2, 3);
                    compare_swap(0, 2);
                    compare_swap(1, 3);
                    compare_swap(1, 2);
                }

                for (uint32_t k = 0; k < 4; ++k) {
                    const uint32_t result = b.sel(tri_valid, tri_out[k], box_out[k]);
                    const int vd = in.dst.value + static_cast<int>(k);
                    const uint32_t old = vreg_old(b, rs, vd);
                    rs.vreg[vd] = result;
                    predicate_write(b, rs, vd, old);
                }
                return true;
            }

            // Which resource class THIS operation can accept, decided before any lookup runs.
            // image_store plus EVERY integer image atomic. This was the THIRD copy of the 0x08/0x0f/0x11
            // triple -- the others are the storage-image classifier in gpu_executor.cpp and the emitter's
            // own is_atomic test -- and all three must agree for an image atomic to compile. They did
            // agree, on the wrong set: exactly the ops the emitter happened to support.
            //
            // The failure mode when they DISAGREE is the instructive part. Widening the classifier alone
            // made this gate reject harder: the resource correctly became a StorageImage, and the clause
            // below then read "0x17 is not a storage op, so it must want a Texture; it is not a Texture"
            // and discarded it. A half-applied fix was worse than none (#2275).
            //
            // Opcodes verified by assembling each with llvm-mc, positive control image_atomic_add ->
            // word0 0xf0440128, byte-identical to the dword decoded from the guest here.
            const bool storage_only_op = in.opcode == 0x08 || in.opcode == 0x09 ||
                                         in.opcode == 0x0f ||
                                         (in.opcode >= 0x11 && in.opcode <= 0x1a && in.opcode != 0x13);
            const ImageResourceRequirement image_requirement =
                storage_only_op            ? ImageResourceRequirement::StorageOnly
              : (in.opcode == 0x00 ||
                 in.opcode == 0x0e)        ? ImageResourceRequirement::Either
                                           : ImageResourceRequirement::SampledOnly;

            // Resolve the T#/U# via SRSRC (src[1]) provenance: s_load tag (indirect) else user-data SGPR.
            //
            // Every lookup is CLASS-FILTERED. It used to take the first hit at each key and then null it
            // out on class further down, which is not the same thing and cost this file two distinct
            // failures: (a) an image resource sharing a key with an earlier buffer-class resource could
            // never be found -- one live vertex stage carries a ConstantBuffer and two VertexBuffers all
            // at sgpr_base 8, so the collision is real, not hypothetical -- and (b) a wrong-class hit on
            // the SRT route left `res` non-null, which SUPPRESSED the `!res` guard on the SGPR fallback,
            // so a route that could have resolved was never tried. Filtering inside the lookup makes the
            // search continue past a wrong-class entry instead of stopping on it.
            //
            // The ACCEPTED SET is unchanged -- `image_resource_class_satisfies` is exactly the old
            // post-check, verified by enumerating all 256x5 (opcode, class) pairs: 258 accepted, zero
            // disagreements, with a deliberately mis-mapped 0x0e control showing the comparison can
            // detect one. So this can never bind a class the op cannot use.
            //
            // What it CAN change, and the first version of this comment wrongly denied: ROUTE
            // PRECEDENCE. Where the first entry at a `fetch_pc` is buffer-class and a later one at
            // the same pc is image-class, the old code took the buffer hit, failed the class test and
            // fell through to the SRT/SGPR routes; this one lets the pc route win. That is the
            // intended contract -- exact per-use provenance outranks a table key, because a sample
            // and a store may share one key while needing different Vulkan classes -- but it means
            // the outcome can be a DIFFERENT, still class-valid resource, not only a rejection turned
            // into a resolution.
            //
            // NO REACHABLE INSTANCE IS KNOWN, and the first version of this comment claimed one that
            // cannot exist: "reachability needs the BVH piggyback to land on a texture-publishing pc".
            // It cannot -- that piggyback keys a ConstantBuffer by an IMAGE_BVH_INTERSECT_RAY pc, and
            // opcode 0xe6 returns from its own block above, before this lookup ever runs. The sentence
            // arrived verbatim from a review comment, carrying a file:line, and was promoted into
            // shipped code without anyone opening the cited path -- the #2049 -> #2052 shape CLAUDE.md
            // records, repeated. State the precedence change, which is real and provable from the
            // diff; do not attach a reachability story to it that has not been traced. #3126/#1634.
            const ShaderResource* res = rt->image_by_fetch_pc(in.pc, image_requirement);
            if (!res) {
                uint32_t srt_tag = 0;
                if (sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag))
                    res = rt->image_by_srt_offset(srt_tag, image_requirement);
                if (!res && !sreg_range_written(rs, in.src[1].value, 8))
                    res = rt->image_by_sgpr_base(in.src[1].value, image_requirement);
                // The SRSRC range reads as written when the shader STAGED a direct descriptor into
                // it with s_mov_b32, so the lookup above is skipped even though the bits are still
                // the driver's. Resolve through the copy to where those words actually live (#1773).
                if (int ud_origin = 0; !res && sreg_range_ud_alias(rs, in.src[1].value, 8, ud_origin))
                    res = rt->image_by_sgpr_base(ud_origin, image_requirement);
            }
            if (!res) {
                // Resolution-failure diagnostic: which provenance step failed for this image op.
                //
                // Ungated, and deduped per (pc, srsrc). It fires only when an image op has already
                // failed to resolve, so its volume is bounded by the defect it reports. Behind
                // PROSPER_DBG it was unreachable in practice on any routed boot.
                //
                // #3143: this report used to sit downstream of the PROSPER_MIMG_SOFT early return
                // below, so arming that switch to see whether softening changes a frame ALSO deleted
                // the one line that names which descriptor failed -- the switch was least informative
                // exactly when it was armed for its intended purpose ("softening changed nothing"
                // became indistinguishable from "there was nothing here to soften"). It now runs
                // before either the soft or the reject path decides what to DO about the failure, so
                // both keep the same descriptor identification; the soft path still gets its own
                // confirmation line below ([mimg-soft]) once the constant is written.
                static std::mutex mimg_mutex;
                // Keyed by PROGRAM as well as (pc, srsrc): every shader has a pc 16, so a
                // pc-only key lets the first program to reach one silence all later programs and
                // attribute its line to a shader the reader is not looking at.
                static std::set<std::tuple<uint64_t, uint32_t, int>> mimg_reported;
                bool first_report = false;
                {
                    std::lock_guard<std::mutex> lock(mimg_mutex);
                    first_report = mimg_reported.emplace(b.diagnostic.program_address, in.pc,
                                                         in.src[1].value).second;
                }
                if (first_report) {
                    uint32_t srt_tag = 0;
                    const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag);
                    const ShaderResource* pk = has_srt_tag ? rt->by_srt_offset(srt_tag) : nullptr;
                    const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                    // Report the copy-alias step too. Without it a staged descriptor (#1773) and a
                    // genuinely absent one print the identical line, which is the output-vs-input
                    // confusion that cost #1590 several sessions -- name which provenance route failed.
                    int ud_origin = 0;
                    const bool has_ud_alias = sreg_range_ud_alias(rs, in.src[1].value, 8, ud_origin);
                    const ShaderResource* pa = has_ud_alias ? rt->by_sgpr_base(ud_origin) : nullptr;
                    // sgpr_res: the DIRECT user-data route's own outcome, which this line did not
                    // report and which is the ONLY route that can fire for the commonest failure
                    // shape -- `written=0`, i.e. the shader never touched SRSRC, so by construction
                    // there is no s_load tag and no copy alias for the other three fields to carry.
                    // For such a site `srt_tag=NONE key_res=null ud_alias=NONE alias_res=null` is a
                    // restatement of `written=0`, not evidence: those routes did not run. Five sites
                    // on Stray (PPSA02101, #3126) printed exactly that quadruple while three of them
                    // had a resource sitting at the requested SGPR the whole time -- of the wrong
                    // class, discarded in silence. Print the class-BLIND hit so "nothing is here"
                    // and "something is here and it is a cbuf" stop looking identical. #3126/#1634.
                    //
                    // ...and say whether that route was CONSULTED, which is the same trap inverted
                    // (trap 254). The SGPR route is guarded by `!sreg_range_written` above, so on a
                    // `written=1` site it never runs -- and printing a bare `sgpr_res=tex` there
                    // tells the reader a Texture was found at the requested SGPR and rejected, when
                    // in fact nothing consulted it and the descriptor is not the entry user data at
                    // all. That is live on this title: `0x30131d0000` pc=77 is `written=1`. Print the
                    // status and the contents, never the contents alone.
                    const bool srsrc_written = sreg_range_written(rs, in.src[1].value, 8);
                    const ShaderResource* ps_direct = rt->by_sgpr_base(in.src[1].value);
                    auto cls_name = [](const ShaderResource* r) {
                        if (!r) return "null";
                        switch (r->cls) {
                            case ResourceClass::Texture:        return "tex";
                            case ResourceClass::StorageImage:   return "simg";
                            case ResourceClass::ConstantBuffer: return "cbuf";
                            case ResourceClass::VertexBuffer:   return "vbuf";
                            case ResourceClass::Sampler:        return "samp";
                        }
                        return "other-cls";
                    };
                    const char* need =
                        image_requirement == ImageResourceRequirement::StorageOnly ? "storage"
                      : image_requirement == ImageResourceRequirement::SampledOnly ? "sampled"
                                                                                   : "either";
                    // `not-consulted(<cls>)` keeps BOTH facts on the line: that the route did not
                    // run, and what happens to sit at that SGPR anyway. Dropping the second half
                    // would trade one misreading for a different one.
                    const std::string sgpr_res =
                        srsrc_written ? std::string("not-consulted(") + cls_name(ps_direct) + ")"
                                      : std::string(cls_name(ps_direct));
                    fprintf(stderr, "[mimg-unresolved] program=0x%llx pc=%u op=0x%02x storage=%d need=%s srsrc=s%d srt_tag=%s0x%x key_res=%s pc_res=%s ud_alias=%s%d alias_res=%s sgpr_res=%s written=%d (%zu res)\n",
                            (unsigned long long)b.diagnostic.program_address,
                            in.pc, in.opcode, (int)storage_only_op, need,
                            in.src[1].value, has_srt_tag ? "" : "NONE ",
                            has_srt_tag ? srt_tag : 0u,
                            cls_name(pk), cls_name(pp),
                            has_ud_alias ? "s" : "NONE ", has_ud_alias ? ud_origin : 0,
                            cls_name(pa), sgpr_res.c_str(),
                            srsrc_written ? 1 : 0,
                            rt->resources.size());
                    // #3126: say what WAS available, not only what failed. "key_res=null pc_res=null"
                    // tells you the lookups missed; it does not tell you whether the table holds the
                    // descriptor under a different key, holds it classified as something else, or does
                    // not hold it at all -- and those are three different bugs. Print every resource's
                    // class and provenance so the miss can be diagnosed from one line instead of a
                    // rebuild. Bounded, and printed once per site regardless of whether the site is
                    // then softened or rejected (#3143).
                    //
                    // Print EVERY key a resource carries, not one of them. The old form printed
                    // `srt=0x%x` whenever sgpr_base was unset, so a resource keyed only by fetch_pc
                    // -- which is how the const-fold publishes a seed/table-recovered T# -- rendered
                    // as `srt=0xffffffff` and read as "this entry has no lookup key at all". Four of
                    // the nine resources in Stray's failing pixel stage printed that way while being
                    // perfectly well keyed, just not by a key this reader could see (#3126).
                    std::string avail;
                    size_t shown = 0;
                    for (const ShaderResource& r : rt->resources) {
                        if (shown++ >= 16) { avail += " ..."; break; }
                        char one[128];
                        char keys[80];
                        int used = 0;
                        if (r.sgpr_base != 0xFFFFFFFFu)
                            used += snprintf(keys + used, sizeof keys - (size_t)used, " s%u", r.sgpr_base);
                        if (r.srt_offset != 0xFFFFFFFFu)
                            used += snprintf(keys + used, sizeof keys - (size_t)used, " srt=0x%x", r.srt_offset);
                        if (r.fetch_pc != 0xFFFFFFFFu)
                            used += snprintf(keys + used, sizeof keys - (size_t)used, " pc=%u", r.fetch_pc);
                        if (!used) snprintf(keys, sizeof keys, " unkeyed");
                        snprintf(one, sizeof one, " [b%u %s%s]", r.binding, cls_name(&r), keys);
                        avail += one;
                    }
                    fprintf(stderr, "[mimg-unresolved]   available:%s\n",
                            avail.empty() ? " (none)" : avail.c_str());
                }

                // PROSPER_MIMG_SOFT=1 -- diagnostic only, and deliberately WRONG output.
                //
                // Placed BEFORE the reject gate below on purpose. That gate fires once per
                // (program, pc, srsrc); downstream of it, only the FIRST compile of each site would be
                // softened and every later one -- reachable through ordinary shader-cache eviction --
                // would take the reject path unchanged. The recorded result would then be
                // indistinguishable from "nothing was softened", which is exactly the reading this
                // diagnostic exists to produce. When an image
                // op cannot resolve its descriptor, write a constant instead of failing the stage, so a
                // frame can be seen that would otherwise be discarded entirely. It answers one question
                // nothing else here can: how much of the picture is riding on THIS resolution failure.
                // On Stray it showed the answer was "none" -- the composite compiled and the frame did
                // not change -- which is what redirected #3126 away from the recompiler.
                //
                // Never acceptable in a run that produces progression evidence: every unresolved sample
                // reads a flat value.
                if (!res && compiler_choice(CompilerChoice::MimgSoft) != 0) {
                    const uint32_t soft = b.uconst(fbits(0.5f));
                    uint32_t comps = 0;
                    for (uint32_t m = 0; m < 4; ++m) if (in.mimg_dmask & (1u << m)) ++comps;
                    if (!comps) comps = 1;
                    for (uint32_t k = 0; k < comps; ++k) {
                        const int dst = in.dst.value + static_cast<int>(k);
                        if (dst < 0) break;
                        const uint32_t old = vreg_old(b, rs, dst);
                        rs.vreg[dst] = soft;
                        predicate_write(b, rs, dst, old);
                    }
                    static std::mutex smx; static std::set<uint64_t> sseen;
                    std::lock_guard<std::mutex> lk(smx);
                    if (sseen.insert(((uint64_t)b.diagnostic.program_address << 16) | in.pc).second)
                        fprintf(stderr, "[mimg-soft] program=0x%llx pc=%u -> constant %u comps\n",
                                (unsigned long long)b.diagnostic.program_address, in.pc, comps);
                    return true;
                }
                if (!first_report) { ok = false; return true; }
            }
            if (!res) { ok = false; return true; }
            const bool uint_texture = res->format == DataFormat::Uint8 ||
                                      res->format == DataFormat::Uint16 ||
                                      res->format == DataFormat::Uint32;
            // D16 packs two floating-point result components into each VDATA dword. The image
            // helpers expose normalized/float resources as ordinary f32 bits, so the back-half
            // must explicitly convert and pack those values rather than writing one f32 per VGPR.
            // Likewise an IMAGE_STORE D16 consumes consecutive packed halves, not four full-width
            // VGPRs. GTA V's FSR1 RCAS pass depends on both halves of this contract: its dmask-f
            // R11G11B10F load writes v[12:13], and its dmask-f RGBA8 stores read v[14:15].
            // Unsigned D16 stores have a separate packed-integer path; integer loads stay refused.
            const bool d16_float_resource =
                res->format == DataFormat::Float32 || res->format == DataFormat::Float16 ||
                res->format == DataFormat::Unorm16 || res->format == DataFormat::Snorm16 ||
                res->format == DataFormat::Unorm8 || res->format == DataFormat::Snorm8 ||
                res->format == DataFormat::Float10_11_11 ||
                res->format == DataFormat::Unorm2_10_10_10 ||
                res->format == DataFormat::Snorm2_10_10_10;
            auto write_mimg_results = [&](const uint32_t out[4]) -> bool {
                const int vd = in.dst.value;
                if (!in.mimg_d16) {
                    int written = 0;
                    for (uint32_t component = 0; component < 4; ++component) {
                        if (!(in.mimg_dmask & (1u << component))) continue;
                        const uint32_t old = vreg_old(b, rs, vd + written);
                        rs.vreg[vd + written] = out[component];
                        predicate_write(b, rs, vd + written, old);
                        ++written;
                    }
                    return true;
                }
                if (!d16_float_resource) return false;

                int word_index = 0;
                auto write_packed = [&](uint32_t value) {
                    const uint32_t old = vreg_old(b, rs, vd + word_index);
                    rs.vreg[vd + word_index] = value;
                    predicate_write(b, rs, vd + word_index, old);
                    ++word_index;
                };
                ir::lower_image_d16_results(b, in, out, write_packed);
                return true;
            };

            // --- Storage-image path: image_load (0x00), image_store[_mip] (0x08/0x09), and R32_UINT
            if (res->cls == ResourceClass::StorageImage) {
                return ir::ImageStorageRoutes{{b, rs, in, ok, allow_wave},
                                              res,
                                              uint_texture,
                                              d16_float_resource,
                                              SQ_DIM_2D,
                                              SQ_DIM_3D,
                                              SQ_DIM_2D_ARRAY,
                                              std::cref(vread),
                                              std::cref(write_mimg_results)}
                    .lower();
            }

            // image_get_resinfo (0x0e): sampled-image dimensions at the integer LOD in VADDR. The
            // DOLL UE4 volume initializer uses dim:3D, dmask:xyz to bounds-check its 8x8x8 dispatch
            // before loading and writing the volume. Array/cube queries remain deferred with their
            // corresponding sampled-image representations.
            const auto array_contract = sampled_array_contract(in, *res, b.is_compute);
            const bool res_arrayed = array_contract.resource_arrayed;
            const bool graphics_float_array = array_contract.graphics_float_array;
            const bool declared_arrayed = array_contract.declared_arrayed;
            if (in.opcode == 0x0e) {
                uint32_t dim;
                if (in.mimg_dim == 0u) dim = Dim_1D;
                else if (in.mimg_dim == 1u) dim = Dim_2D;
                else if (in.mimg_dim == 2u) dim = Dim_3D;
                // #2790: 2D_ARRAY (dim 5) queries as SPIR-V Dim_2D with Arrayed set, which is the
                // representation #325 already uploads and declares for these resources -- and
                // `image_get_resinfo` already branches on `tex_is_arrayed(binding)` to ask for the
                // ivec3 form and report its third component as the layer COUNT, which is exactly
                // what GET_RESINFO's third result means for a 2D_ARRAY T#. The lowering was widened
                // for the array case and this dispatch is the site that still declined it -- the
                // same lag #2265 records for the atomic coverage predicate.
                //
                // The backend-specific sampled_array_contract selects ivec2 or ivec3. Graphics
                // fallback views retain layer count one; compute Float16/Float32 DIM5 queries
                // observe the full selected array, consistently with their gather declaration.
                else if (in.mimg_dim == 3u) dim = Dim_Cube;
                else if (in.mimg_dim == 5u) dim = Dim_2D;
                else { ok = false; return true; }
                if (res->cls != ResourceClass::Texture) { ok = false; return true; }
                if (!b.declare_texture(res->binding, dim, uint_texture, declared_arrayed)) {
                    ok = false; return true;
                }
                uint32_t out[4]; b.image_get_resinfo(res->binding, dim, vread(in.src[0].value), out);
                int vd = in.dst.value, w = 0;
                for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) {
                    uint32_t old = vreg_old(b, rs, vd + w); rs.vreg[vd + w] = out[c];
                    predicate_write(b, rs, vd + w, old); w++;
                }
                return true;
            }

            // IMAGE_GET_LOD (0x60): rdna2_emit_image_lod.cpp.
            if (in.opcode == 0x60) {
                ok = emit_image_get_lod(b, rs, in, *res, uint_texture);
                return true;
            }

            // --- Sampled-texture path: image_sample* (0x20/0x24/0x25/0x27) / image_gather4_lz (0x47) /
            // image_gather4 (0x40, admitted only where LOD 0 is exact -- see the gather arm) /
            // image_load (0x00). 2D (any LOD variant) or 3D (implicit-LOD or LOD-0 sample); NSA allowed
            // (coords gathered below). image_sample = 0x20 (implicit-LOD), image_sample_l = 0x24
            // (explicit LOD in last coord), image_sample_b = 0x25 (implicit-LOD + BIAS in FIRST vaddr),
            // image_sample_lz = 0x27 (LOD 0), image_gather4_lz = 0x47 (2x2 single-channel gather,
            // base level), image_load = 0x00 (integer texel fetch). Opcodes round-trip-verified
            // via llvm-mc gfx1010 (#273).
            // op 0xa0 = the high-bit sibling of IMAGE_SAMPLE (0x20): the decoder builds the 8-bit
            // MIMG opcode as ((word0&1)<<7)|bits[24:18], so 0xa0 = 0x80|0x20. GTA V's (PPSA04263,
            // RAGE) intro/composite pipeline (es=0x2042d6a200 / ps=0x2042d83c00) issues it as a plain
            // 2D texture sample; rejecting it dropped that pipeline's draws and blacked the whole
            // frame (#1140). Lowered as an ordinary implicit-LOD sample it renders the animated
            // Rockstar Games intro logo correctly (live PPSA04263 capture). CONFIDENCE: MED — the base
            // op and the "samples a 2D texture" behavior are live-title evidence; the exact high-bit
            // family (an RDNA2 gfx10.3 sample variant) is not yet llvm-mc round-trip-verified, so a
            // future title exercising its distinguishing modifier may need a dedicated lowering.
            const bool is_sample = (in.opcode == 0x20) || (in.opcode == 0xa0);
            const bool is_zero_mip_load = in.opcode == 0x01;
            const bool is_load = in.opcode == 0x00 || is_zero_mip_load;
            const bool is_sample_l = (in.opcode == 0x24), is_sample_lz = (in.opcode == 0x27);
            const bool is_sample_b = (in.opcode == 0x25), is_gather_lz = (in.opcode == 0x47);
            // image_gather4 = 0x40: gather with implicit LOD. Core SPIR-V OpImageGather only samples
            // base level (LOD 0, identical to _lz). This lowering is exact when the resource has only
            // one mip level (res->declared_mip_levels == 1u, CONFIDENCE: HIGH). In compute shaders,
            // prosper resolves implicit LOD to base level (CONFIDENCE: MED). Multi-level textures
            // in fragment shaders are refused fail-visibly to prevent shimmering/aliasing.
            const bool is_gather = (in.opcode == 0x40);
            const bool is_sample_c_lz = (in.opcode == 0x2f);
            // image_gather4_lz_o = 0x57 (gather at base level with the _o packed-offset operand in the
            // FIRST vaddr — llvm-mc gfx1030 round-trip on live DOLL bytes: 0xf15c0808 "image_gather4_lz_o
            // v[4:7], [v0, v18, v19], ..." — coords follow the offset, matching image_sample_b's
            // modifier-first vaddr convention). DOLL's FXAA/upsample pass PS (#294).
            const bool is_gather_lz_o = (in.opcode == 0x57);
            // image_sample_lz_o = 0x37 (LOD-0 sample with the _o packed-offset FIRST vaddr — llvm-mc
            // gfx1010 round-trip on live DOLL FXAA bytes: 0xf0dc0808/0xf0dc080a "image_sample_lz_o";
            // the offset-adjust folds into the normalized coords, see image_sample_lz_offset_2d).
            const bool is_sample_lz_o = (in.opcode == 0x37);
            // image_sample_d = 0x22: sample with EXPLICIT gradients. For 2D the vaddr packs four
            // derivative dwords FIRST — [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy] — then the (u,v) coords, per ISA
            // 8.2.4's "{derivative}{body}" order. Blue Prince (PPSA25009) issues it as a plain 2D texture
            // sample in a fragment shader (ps=0x2011d60100); rejecting it dropped that whole pipeline's
            // draws. Preserve those explicit derivatives with OpImageSampleExplicitLod + Grad: projected,
            // wrapped, atlas, and uniform coordinates do not share the screen quad's implicit derivative.
            // CONFIDENCE: HIGH — operand order is ISA-defined and an execution regression distinguishes the
            // requested gradient-selected mip from the implicit-derivative result.
            //
            // image_sample_cd = 0x68 takes the same operands in the same order (llvm-mc gfx1030:
            // 0xf1a00f08 "image_sample_cd v[6:9], v[0:5], ..." beside 0xf0880f08 "image_sample_d"),
            // contiguous and NSA, so it takes the same Grad lowering. Only this opcode: the siblings
            // 0x69-0x6f (_cd_cl, _c_cd, _c_cd_cl, _cd_o, _cd_cl_o, _c_cd_o, _c_cd_cl_o) add a clamp,
            // a dref or a packed offset to the address, so the exact-opcode test below keeps them
            // refused. The Pathless (PPSA01826, six fragment programs), Little Nightmares II
            // (PPSA02154) and Sonic Origins (PPSA05325, NSA) issue it as a plain 2D sample; with no
            // lowering every one of those draws was dropped (#4808).
            // The lowering is stage-agnostic, like _d's: explicit Grad needs no quad, so compute and
            // vertex programs issuing 0x68 are admitted too. Only fragment use is seen or tested.
            // CONFIDENCE: MED -- the "C" (coarse) form selects one LOD per 2x2 quad from the supplied
            // derivatives; Grad evaluates it per invocation. They agree when the derivatives are
            // quad-uniform (what ddx_coarse supplies; not checked against any title here), and a
            // quad whose lanes differ may get a neighbouring mip per pixel. How the hardware reduces
            // a quad to one LOD (lane 0, an average) is not established. Only quad-uniform is tested.
            const bool is_sample_d = (in.opcode == 0x22) || (in.opcode == 0x68);
            // image_sample_c (0x28) / image_sample_o (0x30) on a one-level 2D texture (#4808):
            // mimg_single_level_implicit_form in rdna2_alu_support.hpp has the contract.
            const bool is_sample_c = in.opcode == 0x28 && mimg_single_level_implicit_form(in, *res);
            const bool is_sample_o = in.opcode == 0x30 && mimg_single_level_implicit_form(in, *res);
            // These array lowerings preserve the guest's layer operand. The older generic
            // bias/gradient/sample-offset helpers default arrays to layer zero, so do not admit
            // those forms for the newly supported Float32 graphics representation.
            if (graphics_float_array && in.mimg_dim == 5u &&
                (in.mimg_a16 ||
                 (!is_load && !is_sample && !is_sample_l && !is_sample_lz && !is_sample_c_lz &&
                  !is_gather_lz && !is_gather && !is_gather_lz_o))) {
                ok = false;
                return true;
            }
            const bool float_array_gather = array_contract.float_array_gather;
            // Only full-width coordinates and results are admitted here. Cache/status and
            // reserved controls cannot be admitted by widening the array opcode list alone.
            if (float_array_gather && (in.mimg_a16 || in.mimg_d16 || in.mimg_r128 || in.mimg_tfe ||
                in.mimg_lwe || in.mimg_dlc || in.mimg_glc || in.mimg_slc ||
                in.mimg_reserved || in.has_modifier)) {
                ok = false;
                return true;
            }
            // #325: supported 2D_ARRAY resources can be uploaded as arrays. BC graphics
            // declarations remain resource-wide; an ordinary instruction on one reads layer 0.
            // Layered Float32 graphics resources can instead expose a 2D base-slice view when
            // every consumer of the binding declares it that way.
            const bool dim2d = (in.mimg_dim == 1u || in.mimg_dim == 5u), dim3d = (in.mimg_dim == 2u);
            const bool dim_msaa = in.mimg_dim == 6u;
            const bool dimcube = (in.mimg_dim == 3u);   // CUBE: stacked-face 2D lowering (#273, below)
            if (in.mimg_dim == 5u && getenv("PROSPER_GFXLOG"))
                fprintf(stderr, "[recompile] 2D_ARRAY image op: resource %s an uploaded array (#325)\n",
                        res_arrayed ? "IS" : "is NOT");
            if ((!is_sample && !is_load && !is_sample_l && !is_sample_lz && !is_sample_b &&
                 !is_sample_c_lz && !is_gather_lz && !is_gather && !is_gather_lz_o &&
                 !is_sample_lz_o && !is_sample_d && !is_sample_c && !is_sample_o) ||
                (!dim2d && !dim3d && !dimcube && !dim_msaa)) {
                ok = false;
                return true;
            }
            if (res->cls != ResourceClass::Texture) { ok = false; return true; }
            // #3048: the guest's mip selector no longer has to be discarded. When the compute
            // backend materializes this resource's whole declared chain -- one derivation,
            // `shader_resource_compute_mip_chain_levels`, read by BOTH this lowering and
            // live_compute's image creation so the two cannot disagree about how many levels exist
            // -- the operand reaches OpImageFetch's Lod.
            //
            // That agreement is about the COUNT ONLY, and reading it as more is a documented
            // mistake (#2818): live_compute applies a SECOND, independent SHAPE test that this
            // lowering never consults -- `compute_binding_mip_chain_materializable`, which still
            // refuses `img_dim != 1` -- so a resource admitted here can still have its dispatch
            // dropped downstream at `[compute-mip-chain]`. A non-zero level count from this
            // function is NOT a promise that the backend will build the chain.
            // This is the only honest answer for the
            // dynamic case: Sonic Frontiers' three scene-width stage kernels issue IMAGE_LOAD_MIP
            // against a 12-level 2048x2048 R32G32_FLOAT surface with the mip NOT provably zero, so
            // the specialization below can never admit them.
            //
            // Compute only, deliberately: the graphics texture path still uploads one level (its
            // generated-chain gate in the render backend is narrower than this one), so a fragment
            // stage reaching here would fetch a level that does not exist.
            uint32_t dynamic_mip_vgpr = 0;
            const uint32_t materialized_mip_levels =
                prosper::gpu::shader_resource_compute_mip_chain_levels(*res);
            const bool dynamic_mip_load =
                is_zero_mip_load && b.is_compute && materialized_mip_levels > 1u &&
                rdna2_mimg_dynamic_mip_shape(in, &dynamic_mip_vgpr) &&
                in.mimg_dim == SQ_DIM_2D &&
                prosper::gpu::shader_resource_uses_ordinary_2d_image(*res, true, false, false) &&
                // `res->unnormalized` used to be refused here. It is NOT a property of this
                // operation (#2818). FORCE_UNNORMALIZED is S# WORD0[15] -- a SAMPLER bit that
                // decides how a sample's coordinates are interpreted -- and `IMAGE_LOAD_MIP`
                // binds no sampler at all: it addresses texels by integer coordinate, so there
                // are no normalized coordinates for the bit to describe. The zero-mip
                // specialization immediately below never consulted it, so the two IMAGE_LOAD_MIP
                // routes disagreed about whether a sampler flag gates a load; this is the half
                // that was wrong.
                //
                // Measured: Sonic Frontiers' three scene-width Cyber Space kernels declined with
                // `ord2d=1 res_unnorm=1 depth_cmp=0 dim_is_2d=1 dyn_shape=1 materialized_mips=12`
                // -- every other term of THIS gate satisfied, the chain PLAN resolving twelve
                // levels, and this one sampler bit holding the lowering. CONFIDENCE: HIGH -- the
                // operand is an integer texel coordinate by ISA definition, and the accepting
                // sibling path already ignores the bit.
                //
                // `materialized_mips` is a PLAN count, NOT "the backend built the chain" -- see
                // the note on `shader_resource_compute_mip_chain_levels` above. For this title's
                // 2D_ARRAY resource the dispatch is still declined downstream; admitting it here
                // moves the reject, it does not open the path.
                //
                // `depth_compare` is NOT removed alongside it. It is equally meaningless for a
                // load, but it is only ever set for IMAGE_SAMPLE_C* (see its declaration), so no
                // measured resource reaches here with it set and widening it would be an
                // unexercised change. Note the standalone `!res->depth_compare` below is
                // redundant belt-and-braces: `shader_resource_uses_ordinary_2d_image` already
                // tests it (shader_resources.hpp). Kept deliberately so this gate reads as a
                // complete statement of its own terms rather than depending on a helper's
                // internals.
                //
                // `in_mip_tail` used to be refused here as well. It no longer is (#3134): a
                // tail-packed selected level is a MODELLED placement, and the level count above is
                // the authority on it -- `shader_resource_mip_chain_plan` admits the tail only
                // after proving its own in-block coordinates equal the ones the descriptor decode
                // published, and reports one level otherwise. Repeating the test here would have
                // kept refusing the whole small-texture class (Stray's 32x32 six-level pyramid is
                // smaller than one 64 KiB macroblock, so EVERY one of its levels is in the tail)
                // for a placement the backend now uploads.
                !res->depth_compare &&
                res->sample_count == 1u && !res->compression_enabled;
            // IMAGE_LOAD_MIP's final address is a real guest mip selector. Specialize it away only
            // after the per-use fold and the materialized-resource checks agree. The 2D_ARRAY form
            // retains its slice coordinate; only the separately proven mip operand is discarded.
            if (is_zero_mip_load && !dynamic_mip_load &&
                (!rdna2_mimg_zero_mip_shape(in) || !res->proven_zero_mip ||
                 res->img_dim != in.mimg_dim || res->sample_count != 1u ||
                 res->declared_mip_levels != 1u || res->in_mip_tail ||
                 res->compression_enabled || (in.mimg_dim == 5u && !b.is_compute))) {
                // Name the sub-condition. Seven terms collapse into one
                // `mode=unresolved-operand`, and an investigation cannot act on that: "the mip is
                // not proven zero" and "the resource declares compression" are different pieces of
                // work. Ungated and deduped per pc, so it costs one line per declining site --
                // PROSPER_DBG, the usual home for this, produces a ~1.5 GB log and desyncs the pad
                // script badly enough that the route never reaches the phase being diagnosed.
                //
                // Measured on GTA V's 0x2042f49a00, a compute pass that reads the main depth and
                // stencil and writes two 4K storage images the frame goes on to sample:
                //   pc=16 shape=1 proven_zero_mip=0 ... compressed=1
                // so exactly two terms hold it, and one of them describes guest bytes that this
                // resource does not read -- 0x2052ac0000 resolves through the sampled-depth bridge,
                // whose pixels come from a retained Vulkan image rather than from compressed memory.
                static std::mutex mip_mutex;
                // Same program-scoped key as [mimg-mip-why]; see the note there.
                static std::set<std::pair<uint64_t, uint32_t>> mip_reported;
                bool first_mip = false;
                {
                    std::lock_guard<std::mutex> lock(mip_mutex);
                    first_mip = mip_reported.emplace(b.diagnostic.program_address, in.pc).second;
                }
                if (first_mip)
                    std::fprintf(stderr,
                                 "[mimg-mip] program=0x%llx image_load_mip declined pc=%u shape=%d "
                                 "proven_zero_mip=%d img_dim=%u/%u samples=%u mips=%u mip_tail=%d "
                                 "compressed=%d array_in_gfx=%d addr=0x%llx %ux%ux%u "
                                 "dataformat=%d ncomp=%u "
                                 "tile=%u dmask=0x%x unorm=%u glc=%u layer_stride=%u "
                                 "dyn_shape=%d nsa=%u materialized_mips=%u compute=%d "
                                 "ord2d=%d res_unnorm=%u depth_cmp=%u dim_is_2d=%d\n",
                                 (unsigned long long)b.diagnostic.program_address, in.pc,
                                 (int)rdna2_mimg_zero_mip_shape(in),
                                 (int)res->proven_zero_mip, res->img_dim, in.mimg_dim,
                                 res->sample_count, res->declared_mip_levels,
                                 (int)res->in_mip_tail, (int)res->compression_enabled,
                                 (int)(in.mimg_dim == 5u && !b.is_compute),
                                 (unsigned long long)res->gpu_addr, res->width, res->height,
                                 res->depth, (int)res->format, res->num_components,
                                 res->tile_mode, in.mimg_dmask,
                                 (unsigned)in.mimg_unorm, (unsigned)in.mimg_glc,
                                 res->layer_stride_bytes,
                                 // The DYNAMIC route's own inputs. Without these the line names
                                 // only the zero-mip specialization's terms, and an investigation
                                 // reading it cannot tell "the address encoding is not modelled"
                                 // (#3134's original state) from "the chain is not materialized on
                                 // this stage" -- which are different pieces of work.
                                 (int)rdna2_mimg_dynamic_mip_shape(in), in.mimg_nsa,
                                 materialized_mip_levels, (int)b.is_compute,
                                 // The remaining terms of the DYNAMIC gate. Without these the line
                                 // reports `dyn_shape=1 materialized_mips=12 compute=1` -- every
                                 // headline precondition satisfied -- and still says "declined",
                                 // which names no work. `res_unnorm` and `depth_cmp` are the
                                 // RESOURCE's own fields; the `unorm=` field above is the
                                 // INSTRUCTION's `mimg_unorm` and is a different thing.
                                 (int)prosper::gpu::shader_resource_uses_ordinary_2d_image(
                                     *res, true, false, false),
                                 res->unnormalized, (unsigned)res->depth_compare,
                                 (int)(in.mimg_dim == SQ_DIM_2D));
                ok = false;
                return true;
            }
            // Guest 2D_MSAA IMAGE_LOAD is represented exactly as a host single-sample 2D array: the
            // guest sample coordinate selects the array layer. Asterix's resolve PS mixes the ordinary
            // consecutive-vaddr packet with three one-extra-dword NSA packets; llvm-mc gfx1030 confirms
            // those NSA addresses are [x,y,sample] in bytes {VADDR, words[2].lo, words[2].byte1}.
            // Accept only those two exact address shapes. Nonzero unused NSA bytes could carry further
            // operands, and every other MSAA op/count/address shape stays fail-visible.
            const bool msaa_address_shape = in.len_dwords == 2u ||
                (in.len_dwords == 3u && (in.words[2] & 0xffff0000u) == 0u);
            const bool msaa_array_fetch = dim_msaa && is_load && msaa_address_shape &&
                !in.mimg_unorm && !in.has_modifier && res->img_dim == 6u &&
                res->sample_count == 4u && res->depth == 1u &&
                res->declared_mip_levels == 1u && !res->depth_compare;
            if (dim_msaa && !msaa_array_fetch) { ok = false; return true; }
            // S# FORCE_UNNORMALIZED makes only the spatial sample coordinates texel-space. Vulkan's
            // native unnormalized sampler cannot represent the guest contract in general (R-Type's
            // live S# combines it with wrap addressing and a nonzero max LOD), so retain the ordinary
            // sampler and convert texels to normalized coordinates in the shader. Array layers, cube
            // faces, explicit LOD/bias, DREF and packed offsets are not spatial coordinates. Explicit
            // gradients receive the same per-axis scale so mip selection remains identical.
            const bool normalize_sampler_coordinates =
                res->unnormalized &&
                compiler_choice(CompilerChoice::NormalizeSamplerCoordinates) != 0;
            ir::LegacyImageSpatialNormalizer normalized_spatial{b, ok,
                                                                normalize_sampler_coordinates};
            // UNRM=1 supplies unnormalized (texel-space) coordinates (Table 100). Compilers set it
            // only on loads/stores/atomics — and our fetch paths are already texel-space — but a
            // SAMPLER op with UNRM set would treat texel coords as normalized and sample a wildly
            // wrong location. Reject the sampler forms until a live title exercises one.
            if (in.mimg_unorm && !is_load) { ok = false; return true; }
            // coords (normalized float for sample, integer texel for load). Non-NSA: consecutive from VADDR.
            // NSA (len>2): coord0 = VADDR, coord k>=1 = byte (k-1) of the extra address dwords words[2..3].
            const bool nsa = in.len_dwords > 2;
            // GTA V's FSR1 RCAS packet uses ordinary 2D IMAGE_LOAD with A16: unsigned x/y
            // texel coordinates occupy the low/high 16-bit halves of ONE VADDR VGPR (LLVM prints
            // `image_load ..., v0, ... a16 d16`, not `v[0:1]`). Sample-family A16 carries fp16
            // addresses, and NSA/array/MSAA have different operand shapes; retain only the exact
            // integer-load contract until those forms have independent evidence.
            if (in.mimg_a16 &&
                (in.opcode != 0x00 || in.mimg_dim != SQ_DIM_2D ||
                 res->img_dim != SQ_DIM_2D || nsa)) {
                ok = false;
                return true;
            }
            auto cvg = [&](uint32_t k) { return mimg_address_vgpr(rs, in, k); };
            uint32_t out[4];
            if (dimcube) {
                // Cube-processed coordinates encode face selection and a normalized in-face pair;
                // no axis is a plain texel coordinate. FORCE_UNNORMALIZED on this combination has no
                // representable generic meaning, so keep it fail-visible instead of scaling a face.
                if (normalize_sampler_coordinates) { ok = false; return true; }
                // CUBE sample (#273 — DOLL's title post PSes sample their reflection probes /
                // skybox with `image_sample_l ..., dim:CUBE`). The compiled coords are the standard
                // AMD cube-processed form (Mesa ac_prepare_cube_coords): vaddr = { sc*rcp(|ma|)+1.5,
                // tc*rcp(|ma|)+1.5, face_id [, lod] } — face coords centered at 1.5 (span [1,2]),
                // face id an integral float from v_cubeid. Our texture backend uploads the cube as
                // SIX FACES STACKED VERTICALLY in one 2D image (w x 6h — see the live_renderer cube
                // path), so the sample lowers to a plain 2D sample at u = x-1, v = (face + clamp
                // (y-1)) / 6 at base LOD (mips aren't uploaded; a >0 LOD clamps to the one level).
                // The in-face clamp stops bilinear bleed across face seams. CONFIDENCE: MED — the
                // coordinate convention is Mesa-verified; face memory layout is validated visually.
                if (!is_sample && !is_sample_l && !is_sample_lz && !is_sample_b &&
                    !is_sample_c_lz) { ok = false; return true; }
                // Modifier-first order: _b carries bias in vaddr0; _c_lz carries DREF there.
                const uint32_t ci = (is_sample_b || is_sample_c_lz) ? 1u : 0u;
                const auto cube = ir::lower_image_cube_coordinates(
                    b, ci, [&](uint32_t k) { return vread(cvg(k)); });
                const uint32_t uf = cube[0], v6 = cube[3];
                if (is_sample_c_lz) {
                    // Point-light shadow maps use the same cube-processed coordinates as ordinary
                    // samples, with DREF prepended: [dref, x, y, face]. The renderer uploads cube
                    // faces as one vertical 2D stack, so compare the transformed face coordinate
                    // manually through its ordinary non-compare sampler (#1167/#1169).
                    if (uint_texture || !res->depth_compare) { ok = false; return true; }
                    if (!b.declare_texture(res->binding, Dim_2D, false, declared_arrayed)) {
                        ok = false; return true;
                    }
                    b.image_sample_dref_manual_2d(res->binding, uf, v6, vread(cvg(0)),
                                                  res->depth_compare_func,
                                                  res->mag_filter != 0u, res->addr_uvw[0],
                                                  res->addr_uvw[1], res->border_color_type, out);
                } else {
                    if (!b.declare_texture(res->binding, Dim_2D, uint_texture, declared_arrayed)) {
                        ok = false; return true;
                    }
                    b.image_sample_lod_2d(res->binding, uf, v6, b.uconst(0), out);
                }
            } else if (dim3d) {
                // 3D: implicit-LOD / explicit-LOD / LOD-0 sample, or an integer texel FETCH
                // (image_load — DOLL's color-grade 3D LUT, #273). Explicit LOD lowers with the
                // guest level, as the 2D path does: a single-level view's image holds one mip,
                // so a nonzero LOD clamps to the selected level, while a materialized chain
                // addresses each declared level (#4842). Integer FETCH out-of-range semantics
                // stay distinct from sampling clamps.
                if (!is_sample && !is_sample_l && !is_sample_lz && !is_load) {
                    ok = false;
                    return true;
                }
                if (!b.declare_texture(res->binding, Dim_3D, uint_texture)) {
                    ok = false; return true;
                }
                // ir/ops/image_coordinates.cpp; the reader keeps the MIMG source proof per slot.
                const auto coords3 =
                    ir::lower_image_coordinates(b, ok,
                                                {3,
                                                 ir::ImageCoordinateForm::Separate,
                                                 !is_load && normalize_sampler_coordinates,
                                                 {res->width, res->height, res->depth}},
                                                [&](uint32_t k) { return vread(cvg(k)); });
                const uint32_t cu = coords3[0], cv = coords3[1], cw = coords3[2];
                if (is_sample)      b.image_sample_3d(res->binding, cu, cv, cw, out);
                else if (is_load)   b.image_fetch_3d(res->binding, cu, cv, cw, out);
                else if (is_sample_l)
                    b.image_sample_lod_3d(res->binding, cu, cv, cw, vread(cvg(3)), out);
                else                b.image_sample_lod_3d(res->binding, cu, cv, cw,
                                                          b.uconst(0), out);   // _lz: base level
            } else if (is_sample_c_lz || is_sample_c) {
                // Astro Bot shadow/visibility packet (opcode 0x2f, dim 2D_ARRAY, NSA). ISA 8.2.5
                // vaddr order is "{offset}{bias}{z-compare}{derivative}{body}" — the z-compare
                // reference PRECEDES the coordinates, so SAMPLE_C_LZ 2D_ARRAY reads
                // [dref, u, v, slice] (the same modifier-first rule the _b/_o paths already use).
                // The earlier [u,v,layer,dref] mapping rotated every operand (#883).
                // EVIDENCE: no rendered-frame validation exists for either order — Astro crashes
                // in engine init before it reaches shadow rendering (#825), and the prior
                // "[u,v,layer,dref]" comment was an un-round-tripped interpretation of the packet,
                // not a captured/disassembled fact. This order is grounded in the ISA plus an
                // llvm-mc gfx1030 round-trip of a canonical c_lz 2D_ARRAY NSA packet
                // (image_sample_c_lz v5, [v10,v11,v12,v13], ... = 0xf0bc012a 0x0040050a 0x000d0c0b:
                // four consecutive address VGPRs, dref at slot 0 per 8.2.5) and LLVM's own
                // llvm.amdgcn.image.sample.c.lz lowering (dref before coords). CONFIDENCE: MED —
                // the #825 lane must re-validate against a real rendered shadow once it lights up.
                // Integer views are not legal for this lowering, and a non-shadow S# means the
                // provenance drifted; reject rather than silently turning a comparison sample into
                // an ordinary color read.
                if (uint_texture || !res->depth_compare) { ok = false; return true; }
                if (in.mimg_dim == 5u) {
                    if (b.is_compute) {
                        // Compute has a backend-reflected 2D-array path. Preserve [dref,u,v,slice]
                        // and perform the comparison manually over an ordinary color sampler, so no
                        // compare-sampler/Dref Vulkan contract is required. Astro Bot's world-map
                        // visibility shader selects among sixteen depth layers here.
                        if (!b.declare_texture(res->binding, Dim_2D, false, true)) {
                            ok = false; return true;
                        }
                        b.image_sample_dref_manual_2d(
                            res->binding,
                            normalized_spatial(vread(cvg(1)), res->width),
                            normalized_spatial(vread(cvg(2)), res->height),
                            vread(cvg(0)),
                            res->depth_compare_func, res->mag_filter != 0u,
                            res->addr_uvw[0], res->addr_uvw[1], res->border_color_type,
                            out, true, vread(cvg(3)));
                    } else {
                        // #325: the graphics fallback used to declare a base-slice 2D image here,
                        // "until its resource uploader can create matching array views". It now
                        // does -- but ONLY for resources the uploader actually uploads as arrays,
                        // which is what res_arrayed answers. A DIM=5 instruction is not sufficient:
                        // this title's shadow maps are declared img_dim 5 with **depth 1**, so they
                        // are uploaded as plain 2D, and declaring them Arrayed produced
                        // VUID-vkCmdDraw-viewType-07752 (caught by tools/vkval, not by ctest --
                        // validation errors do not fail a test).
                        if (!b.declare_texture(res->binding, Dim_2D, false, declared_arrayed)) {
                            ok = false; return true;
                        }
                        b.image_sample_dref_manual_2d(
                            res->binding,
                            normalized_spatial(vread(cvg(1)), res->width),
                            normalized_spatial(vread(cvg(2)), res->height),
                            vread(cvg(0)),
                            res->depth_compare_func, res->mag_filter != 0u,
                            res->addr_uvw[0], res->addr_uvw[1], res->border_color_type,
                            out, declared_arrayed, vread(cvg(3)));
                    }
                } else if (in.mimg_dim == 1u) {
                    // Plain 2D form (Blue Prince's lit-material PSes, #1271: 436 rejects/run, all
                    // op 0x2f dim 1 dmask 0x1 — the entire shadowed lighting pass dropped and the
                    // scene rendered unattenuated/blown-out). Same 8.2.5 vaddr order with no array
                    // slice: [dref, u, v]. Lowered as a manual compare against the color-sampled
                    // shadow map (see image_sample_dref_manual_2d for why not a compare sampler).
                    if (!b.declare_texture(res->binding, Dim_2D, false, declared_arrayed)) {
                        ok = false; return true;
                    }
                    b.image_sample_dref_manual_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(1)), res->width),
                        normalized_spatial(vread(cvg(2)), res->height),
                        vread(cvg(0)), res->depth_compare_func,
                        res->mag_filter != 0u, res->addr_uvw[0],
                        res->addr_uvw[1], res->border_color_type, out);
                } else { ok = false; return true; }
            } else if (is_gather_lz || is_gather || is_gather_lz_o) {
                if (is_gather && !(res->declared_mip_levels == 1u || !b.is_fragment)) {
                    ok = false;
                    return true;
                }
                // gather4 dmask selects ONE channel (must be a single bit); the result is always the
                // four texels of that channel, with gather order preserved. D16 changes only the
                // physical VDATA layout: the four fp16 results occupy two consecutive VGPRs. GTA V's
                // FSR1 EASU shaders consume those packed halves with VOP3P; writing four f32 VGPRs
                // clobbers the following temporaries and turns an otherwise intact scene into bands.
                uint32_t dm = in.mimg_dmask;
                if (dm != 1u && dm != 2u && dm != 4u && dm != 8u) { ok = false; return true; }
                uint32_t comp = dm == 1u ? 0u : dm == 2u ? 1u : dm == 4u ? 2u : 3u;
                if (!b.declare_texture(res->binding, Dim_2D, uint_texture, declared_arrayed)) {
                    ok = false; return true;
                }
                // Floating-point arrays carry an independent layer after the spatial
                // coordinates. cvg preserves both consecutive and NSA address layouts. The
                // gather's signed spatial offset must not translate or normalize that layer.
                ir::ImageOperandCoordinateNormalizer coord{b, ok, normalize_sampler_coordinates};
                const uint32_t layer =
                    ir::lower_image_gather_layer(b, float_array_gather, is_gather_lz_o,
                                                 [&](uint32_t slot) { return vread(cvg(slot)); });
                if (is_gather_lz_o)   // vaddr order: [packed offset, u, v, optional layer]
                    b.image_gather_offset_2d(res->binding, coord(vread(cvg(1)), res->width),
                                             coord(vread(cvg(2)), res->height), comp, vread(cvg(0)),
                                             out, layer);
                else
                    b.image_gather_2d(res->binding, coord(vread(cvg(0)), res->width),
                                      coord(vread(cvg(1)), res->height), comp, out, layer);
                int vd = in.dst.value;
                if (in.mimg_d16) {
                    if (!d16_float_resource) { ok = false; return true; }
                    for (uint32_t word = 0; word < 2; ++word) {
                        const uint32_t old = vreg_old(b, rs, vd + static_cast<int>(word));
                        rs.vreg[vd + static_cast<int>(word)] =
                            ir::lower_image_d16_gather_word(b, out, word);
                        predicate_write(b, rs, vd + static_cast<int>(word), old);
                    }
                } else {
                    for (uint32_t c = 0; c < 4; c++) {
                        uint32_t old = vreg_old(b, rs, vd + static_cast<int>(c));
                        rs.vreg[vd + static_cast<int>(c)] = out[c];
                        predicate_write(b, rs, vd + static_cast<int>(c), old);
                    }
                }
                return true;
            } else {
                const bool load_2d_array = (b.is_compute || graphics_float_array) &&
                    in.opcode == 0x00 && in.mimg_dim == 5u && res->img_dim == 5u;
                const bool mip_load_2d_array = b.is_compute && is_zero_mip_load &&
                    in.mimg_dim == 5u && res->img_dim == 5u;
                // #325: array SAMPLE was restricted to compute. Nothing about an array slice is
                // compute-specific -- the restriction recorded where it was first needed, not a
                // constraint -- and graphics paid for it: Tomb Raider I-III Remastered textures its
                // entire world from ONE 256-slice array, so every world surface sampled slice 0 and
                // rendered flat. Measured on a captured Croft Manor frame, zeroing slices 1..255
                // changed 0.0% of pixels while zeroing all 256 changed 62.9%.
                // `res_arrayed` and not `img_dim == 5` alone: the uploader arrays a resource only
                // when it is ALSO multi-layer with a supported BC/Float32 format. Keying only
                // on img_dim would declare Arrayed for depth-1 or unsupported-format resources
                // that receive a plain 2D view -- the same failure as the
                // hardcoded flag two branches up, at a different site. Compute keeps its own path:
                // it picks the view from the SPIR-V reflection, so it is not bound by that.
                const bool array_sample = in.mimg_dim == 5u &&
                    (b.is_compute || res_arrayed) &&
                    res->img_dim == 5u && (is_sample || is_sample_l || is_sample_lz);
                const bool host_array = load_2d_array || mip_load_2d_array ||
                    array_sample || msaa_array_fetch;
                if (!b.declare_texture(res->binding, Dim_2D, uint_texture,
                                       host_array || declared_arrayed)) {
                    ok = false; return true;
                }
                ir::ImageOperandCoordinateNormalizer operand_spatial{b, ok,
                                                                     normalize_sampler_coordinates};
                if (msaa_array_fetch || load_2d_array || mip_load_2d_array) {
                    b.image_fetch_2d_array(
                        res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), out);
                } else if (array_sample) {
                    // All three retain the 2D-array slice in the SPIR-V coordinate. SAMPLE_LZ
                    // resolves level zero and SAMPLE_L carries its own LOD, so both take the
                    // explicit-LOD form. Plain SAMPLE must NOT: in a fragment stage its LOD comes
                    // from quad derivatives, and forcing level zero there would sample the base mip
                    // across every textured surface -- correct in compute, wrong on a wall.
                    // image_sample_2d_array picks implicit or explicit by stage exactly as the
                    // non-array image_sample_2d does, so compute behaviour is preserved bit for bit.
                    const uint32_t au = operand_spatial(vread(cvg(0)), res->width);
                    const uint32_t av = operand_spatial(vread(cvg(1)), res->height);
                    const uint32_t layer = vread(cvg(2));
                    if (is_sample)
                        b.image_sample_2d_array(res->binding, au, av, layer, out);
                    else
                        b.image_sample_lod_2d_array(
                            res->binding, au, av, layer,
                            is_sample_l ? vread(cvg(3)) : b.uconst(fbits(0.0f)), out);
                } else if (is_sample_b) {      // vaddr order for _b: [bias, u, v]
                    b.image_sample_bias_2d(res->binding, operand_spatial(vread(cvg(1)), res->width),
                                           operand_spatial(vread(cvg(2)), res->height),
                                           vread(cvg(0)), out);
                } else if (is_sample_o) {   // vaddr order for _o: [packed offset, u, v]
                    b.image_sample_offset_2d(
                        res->binding, operand_spatial(vread(cvg(1)), res->width),
                        operand_spatial(vread(cvg(2)), res->height), vread(cvg(0)), out);
                } else if (is_sample_lz_o) {   // vaddr order for _o: [packed offset, u, v]
                    b.image_sample_lz_offset_2d(
                        res->binding, operand_spatial(vread(cvg(1)), res->width),
                        operand_spatial(vread(cvg(2)), res->height), vread(cvg(0)), out);
                } else if (
                    is_sample_d) {   // vaddr order for _d: [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy, u, v]
                    b.image_sample_grad_2d(res->binding, operand_spatial(vread(cvg(4)), res->width),
                                           operand_spatial(vread(cvg(5)), res->height),
                                           operand_spatial(vread(cvg(0)), res->width),
                                           operand_spatial(vread(cvg(1)), res->height),
                                           operand_spatial(vread(cvg(2)), res->width),
                                           operand_spatial(vread(cvg(3)), res->height), out);
                } else {
                    const auto coords2 = ir::lower_image_coordinates(
                        b, ok,
                        {2,
                         in.mimg_a16 ? ir::ImageCoordinateForm::Packed16
                                     : ir::ImageCoordinateForm::Separate,
                         !is_load && normalize_sampler_coordinates,
                         {res->width, res->height, 0u}},
                        [&](uint32_t k) { return vread(cvg(k)); });
                    const uint32_t cu = coords2[0], cv = coords2[1];
                    if (is_sample)         b.image_sample_2d(res->binding, cu, cv, out);
                    else if (is_sample_lz) b.image_sample_lod_2d(res->binding, cu, cv, b.uconst(0), out);      // LOD 0
                    // Explicit-LOD plain 2D uses [u,v,lod]. Graphics keeps the established
                    // base-slice view for DIM=5, whose address is [u,v,slice,lod].
                    else if (is_sample_l)  b.image_sample_lod_2d(res->binding, cu, cv,
                        vread(cvg(in.mimg_dim == 5u && res->img_dim == 5u ? 3u : 2u)), out);
                    else if (dynamic_mip_load) {
                        // See build_image_mip_clamp for the preserved value contract.
                        // The register the shape predicate itself identified, not a second
                        // derivation of it: for this packet family they are the same VGPR, and
                        // reading it from one place keeps them that way.
                        const uint32_t requested = vread(static_cast<int>(dynamic_mip_vgpr));
                        const uint32_t lod =
                            ir::lower_image_mip_clamp(b, requested, materialized_mip_levels);
                        b.image_fetch_2d_lod(res->binding, cu, cv, lod, out);
                    }
                    else                   b.image_fetch_2d (res->binding, cu, cv, out);
                }
            }
            // D16 changes the VDATA layout, not the logical image result: selected components are
            // still in dmask order, but two binary16 values share each consecutive VGPR.
            if (!write_mimg_results(out)) { ok = false; return true; }
            return true;
        }
        case Rdna2Format::DS: {
            return ir::lower_ds_case(b, rs, in, ok, allow_wave);
        }
        case Rdna2Format::VINTRP: {
            // Pixel-shader attribute interpolation. The rasterizer performs the interpolation; v_interp_p1
            // (opcode 0) initializes and is a no-op here, while v_interp_p2 (1) and v_interp_mov (2)
            // deliver the interpolated attribute component from the matching Input varying. Fragment-only.
            if (!b.is_fragment) { ok = false; return true; }
            if (in.opcode == 0) return true;   // p1: no-op (p2/mov produce the value)
            uint32_t old = vreg_old(b, rs, in.dst.value);
            if (in.opcode == 2) {
                const uint32_t parameter = b.interp_parameter(
                    in.vintrp_attr, in.vintrp_chan, in.src[0].value);
                if (!parameter) { ok = false; return true; }
                rs.vreg[in.dst.value] = parameter;
            } else {
                rs.vreg[in.dst.value] = b.interp_read(in.vintrp_attr, in.vintrp_chan);
            }
            predicate_write(b, rs, in.dst.value, old);
            record_native_interpolation_publication(b, rs, in);
            return true;
        }
        default: return false;   // not a VALU format
    }
}

namespace {

} // namespace

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
