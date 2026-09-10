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


size_t rdna2_specialize_shader_constant_branches(
        std::vector<Rdna2Inst>& instructions) {
    return specialize_shader_constant_branches(instructions);
}

size_t rdna2_specialize_proven_null_bvh_paths(
        std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
        uint32_t wave_size) {
    return specialize_proven_null_bvh_exits(instructions, resources, wave_size);
}

size_t rdna2_specialize_zero_record_execz_paths(
        std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
        uint32_t wave_size) {
    return specialize_zero_record_execz_exits(instructions, resources, wave_size);
}

// RDNA waves order their own LDS instructions, but the portable compute shell represents guest
// wave lanes as independent Vulkan invocations. A plain OpStore performed by lane 0 therefore does
// not publish its value to a following cross-lane atomic merely because every invocation reaches
// the atomic later in program order. GTA V's BVH bounds kernel uses this exact wave-synchronous
// idiom: select EXEC=1, initialize six adjacent dwords with one B64 and one B128 write, restore
// EXEC=-1, set vaddr=0, then issue three DS_MIN_F32 and three DS_MAX_F32 operations.
//
// Preserve the original byte-exact adjacent packet as a fast path: insert emitter-only S_BARRIERs
// before and after its six atomics. For a separated packet in one guest wave, ask emit_body to route
// every float atomic through the dispatcher's synchronized common phase. A proved lane-zero writer can
// retain ordinary stores; otherwise each preceding store becomes an atomic exchange, matching RDNA's
// serialized indexed bank conflicts without introducing a Vulkan write/write data race. The first
// common-phase barrier publishes those writes; each trailing barrier completes that atomic before the
// next dispatcher iteration or later gather. AcquireRelease on an individual atomic orders memory but
// is not an arrival barrier, so neither edge can be omitted. Every multi-wave separated shape rejects
// visibly. A real guest barrier, or an atomic with no preceding ordinary store in its phase, remains
// architectural and needs no synthesized edge.
struct LdsFminmaxSynchronization {
    bool needs_dispatcher = false;
    std::unordered_set<uint32_t> atomicized_store_pcs;
};

bool prepare_lds_fminmax_synchronization(std::vector<Rdna2Inst>& ins,
                                         RecompileDiagnosticContext diagnostic,
                                         bool at_most_one_guest_wave,
                                         LdsFminmaxSynchronization* synchronization = nullptr) {
    if (synchronization) *synchronization = {};
    auto ordinary_lds_store = [](const Rdna2Inst& in) {
        if (in.fmt != Rdna2Format::DS || in.ds_gds) return false;
        return in.opcode == 0x0d || in.opcode == 0x0e || in.opcode == 0x4d ||
               in.opcode == 0x4e ||
               in.opcode == 0xb0 || in.opcode == 0xde || in.opcode == 0xdf;
    };
    auto float_lds_atomic = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::DS && !in.ds_gds &&
               (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32);
    };
    auto store_data_registers = [](const Rdna2Inst& in) {
        std::vector<int> registers;
        auto append = [&](int first, uint32_t count) {
            for (uint32_t word = 0; word < count; ++word)
                registers.push_back(first + static_cast<int>(word));
        };
        switch (in.opcode) {
            case 0x0d: case 0xb0: append(in.src[1].value, 1); break;
            case 0x0e:
                append(in.src[1].value, 1);
                if ((in.literal & 0xffu) != ((in.literal >> 8u) & 0xffu))
                    append(in.src[2].value, 1);
                break;
            case 0x4d: append(in.src[1].value, 2); break;
            case 0x4e:
                append(in.src[1].value, 2);
                if ((in.literal & 0xffu) != ((in.literal >> 8u) & 0xffu))
                    append(in.src[2].value, 2);
                break;
            case 0xde: append(in.src[1].value, 3); break;
            case 0xdf: append(in.src[1].value, 4); break;
            default: break;
        }
        return registers;
    };
    // Atomic exchange is equivalent to RDNA's indexed-bank serialization only when colliding lanes
    // write the same bits. Prove the narrow but generic form this family needs: every stored dword's
    // last writer is a plain v_mov from a scalar/literal source, and no control/EXEC edge can let a
    // store lane bypass that writer. Different addresses remain independent; equal addresses then
    // have identical candidate values, so the exchange winner is immaterial.
    auto store_data_are_wave_uniform = [&](size_t store_index) {
        const Rdna2Inst& store = ins[store_index];
        for (int reg : store_data_registers(store)) {
            size_t writer_index = ins.size();
            for (size_t j = store_index; j-- > 0;) {
                if (writes_vgpr(ins[j], reg)) {
                    writer_index = j;
                    break;
                }
            }
            if (writer_index == ins.size()) return false;
            const Rdna2Inst& writer = ins[writer_index];
            if (writer.fmt != Rdna2Format::VOP1 || writer.opcode != 0x01 ||
                writer.has_modifier || writer.has_sdwa || writer.has_dpp ||
                writer.src[0].kind == OperandKind::VGPR)
                return false;
            for (size_t j = writer_index + 1; j < store_index; ++j) {
                const Rdna2Inst& between = ins[j];
                if (rdna2_instruction_may_change_exec(between) ||
                    (between.fmt == Rdna2Format::SOPP && between.opcode >= 0x02u &&
                     between.opcode <= 0x12u && between.opcode != 0x03u &&
                     between.opcode != 0x0cu))
                    return false;
            }
            for (const Rdna2Inst& edge : ins) {
                if (edge.fmt == Rdna2Format::SOP1 && edge.opcode >= 0x20u &&
                    edge.opcode <= 0x22u)
                    return false;
                if (edge.fmt != Rdna2Format::SOPP || edge.opcode < 0x02u ||
                    edge.opcode > 0x09u || edge.opcode == 0x03u)
                    continue;
                const uint32_t target = branch_target(edge);
                const bool source_outside = edge.pc < writer.pc || edge.pc >= store.pc;
                if (source_outside && target > writer.pc && target <= store.pc)
                    return false;
            }
        }
        return true;
    };
    auto words_are = [](const Rdna2Inst& in, uint32_t word0, uint32_t word1) {
        return in.words[0] == word0 && in.words[1] == word1;
    };

    static constexpr uint32_t kAtomicWord0[6] = {
        0xd8480000u, 0xd8480004u, 0xd8480008u,
        0xd84c000cu, 0xd84c0010u, 0xd84c0014u,
    };
    static constexpr uint32_t kAtomicWord1[6] = {
        0x00000900u, 0x00000a00u, 0x00000b00u,
        0x00000600u, 0x00000700u, 0x00000800u,
    };

    std::vector<size_t> phase_stores;
    bool phase_stores_are_single_lane = true;
    bool exec_is_single_lane = false;
    bool dispatcher_initializer_exec = false;
    uint32_t dispatcher_initializer_pc = UINT32_MAX;
    uint32_t phase_dispatcher_initializer_pc = UINT32_MAX;
    std::vector<size_t> synth_before;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) {
            phase_stores.clear();
            phase_stores_are_single_lane = true;
            phase_dispatcher_initializer_pc = UINT32_MAX;
            continue;
        }
        // These byte-exact scalar mask operations are the two lane-zero forms used by the known
        // initializers. A control-flow edge or any other EXEC writer ends the straight-line proof;
        // the dispatcher supplies ordering, not ownership of racing ordinary stores.
        if (in.words[0] == 0xbeea2481u || in.words[0] == 0xbefe0481u) {
            exec_is_single_lane = true; // s_and_saveexec_b64 vcc,1 / s_mov_b64 exec,1
            dispatcher_initializer_exec = in.words[0] == 0xbeea2481u;
            dispatcher_initializer_pc = dispatcher_initializer_exec ? in.pc : UINT32_MAX;
        } else if (rdna2_instruction_may_change_exec(in) ||
                   (in.fmt == Rdna2Format::SOPP &&
                    (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09) ||
                     in.opcode == 0x12))) {
            exec_is_single_lane = false;
            dispatcher_initializer_exec = false;
            dispatcher_initializer_pc = UINT32_MAX;
        }
        if (ordinary_lds_store(in)) {
            phase_stores.push_back(i);
            phase_stores_are_single_lane &= exec_is_single_lane;
            if (!dispatcher_initializer_exec ||
                (phase_dispatcher_initializer_pc != UINT32_MAX &&
                 phase_dispatcher_initializer_pc != dispatcher_initializer_pc))
                phase_stores_are_single_lane = false;
            else
                phase_dispatcher_initializer_pc = dispatcher_initializer_pc;
            continue;
        }
        if (!float_lds_atomic(in) || phase_stores.empty()) continue;

        bool exact = i >= 4 && i + 9 < ins.size() && phase_stores.size() == 2 &&
            phase_stores[0] == i - 4 && phase_stores[1] == i - 3 &&
            words_are(ins[i - 4], 0xd9340010u, 0x0000040cu) &&
            words_are(ins[i - 3], 0xdb7c0000u, 0x0000000cu) &&
            ins[i - 2].words[0] == 0xbefe04c1u && // s_mov_b64 exec, -1
            ins[i - 1].words[0] == 0x7e000280u;   // v_mov_b32 v0, 0
        for (size_t atomic = 0; exact && atomic < 6; ++atomic)
            exact = words_are(ins[i + atomic], kAtomicWord0[atomic], kAtomicWord1[atomic]);
        exact = exact &&
            ins[i + 6].words[0] == 0xbefe0481u && // pc81 s_mov_b64 exec, 1
            ins[i + 7].words[0] == 0x7e080280u && // pc82 v_mov_b32 v4, 0
            words_are(ins[i + 8], 0xdbfc0000u, 0x00000004u) && // pc83 ds_read_b128 v[0:3],v4
            words_are(ins[i + 9], 0xd9d80010u, 0x04000004u);   // pc85 ds_read_b64 v[4:5],v4

        bool found_lane0 = false;
        uint32_t lane0_pc = UINT32_MAX;
        if (exact) {
            for (size_t j = i - 4; j-- > 0;) {
                const Rdna2Inst& prefix = ins[j];
                if (prefix.words[0] == 0xbefe0481u) { // s_mov_b64 exec, 1
                    found_lane0 = true;
                    lane0_pc = prefix.pc;
                    for (size_t k = j + 1; k < i - 2; ++k) {
                        const Rdna2Inst& between = ins[k];
                        if (between.fmt == Rdna2Format::SOPP ||
                            (rdna2_instruction_may_change_exec(between) &&
                             between.words[0] != 0xbefe04c1u)) {
                            found_lane0 = false;
                            break;
                        }
                    }
                    break;
                }
                if (prefix.fmt == Rdna2Format::SOPP ||
                    rdna2_instruction_may_change_exec(prefix))
                    break;
            }
        }
        if (found_lane0) {
            const uint32_t last_store_pc = ins[i - 3].pc;
            for (const Rdna2Inst& edge : ins) {
                if (edge.pc >= lane0_pc || edge.fmt != Rdna2Format::SOPP ||
                    edge.opcode < 0x02 || edge.opcode > 0x09 || edge.opcode == 0x03)
                    continue;
                const uint32_t target = branch_target(edge);
                // The exact EXEC=1 writer must dominate both initializer stores. An edge from its
                // prefix may target the writer itself, but entering after it can leave EXEC full
                // and turn the supposedly single-writer OpStores into same-address races.
                if (target > lane0_pc && target <= last_store_pc) {
                    found_lane0 = false;
                    break;
                }
            }
        }
        if (!exact || !found_lane0) {
            bool dispatcher_initializer_dominates =
                phase_dispatcher_initializer_pc != UINT32_MAX;
            if (dispatcher_initializer_dominates) {
                const uint32_t last_store_pc = ins[phase_stores.back()].pc;
                for (const Rdna2Inst& edge : ins) {
                    // The deferred proof deliberately owns only a linear initializer region. Any
                    // scalar edge in that region could skip one of its lane-zero operations, while
                    // a later edge back into it could revisit a store after EXEC was restored full.
                    // Reject both, independent of whether the branch condition happens to look
                    // constant in this shader. GTA's captured initializer has no such edges.
                    if (edge.fmt == Rdna2Format::SOPP && edge.opcode >= 0x02 &&
                        edge.opcode <= 0x09 && edge.opcode != 0x03) {
                        const uint32_t target = branch_target(edge);
                        if (edge.pc <= last_store_pc || target <= last_store_pc) {
                            dispatcher_initializer_dominates = false;
                            break;
                        }
                    }
                    // An indirect PC update has no statically bounded target, so it cannot prove
                    // that the single-writer initializer is never re-entered.
                    if (edge.fmt == Rdna2Format::SOP1 && edge.opcode >= 0x20u &&
                        edge.opcode <= 0x22u) {
                        dispatcher_initializer_dominates = false;
                        break;
                    }
                }
            }
            if (synchronization) {
                if (!at_most_one_guest_wave) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-recompile-reject", "terminal",
                        "pc=%u reason=multiwave-lds-fminmax-dispatcher", in.pc);
                    return false;
                }
                synchronization->needs_dispatcher = true;
                // A proven lane-zero initializer has one writer and can retain ordinary OpStores.
                // For a general one-wave initializer whose data is identical across active lanes,
                // make each exact preceding DS write an atomic exchange. RDNA serializes indexed
                // bank conflicts; equal colliding values make Vulkan's exchange winner immaterial.
                if (!phase_stores_are_single_lane || !dispatcher_initializer_dominates) {
                    if (!std::all_of(phase_stores.begin(), phase_stores.end(),
                                     store_data_are_wave_uniform)) {
                        log_recompile_diagnostic(
                            diagnostic, "compute-recompile-reject", "terminal",
                            "pc=%u reason=nonuniform-lds-store-before-ds-fminmax", in.pc);
                        return false;
                    }
                    for (size_t store : phase_stores)
                        synchronization->atomicized_store_pcs.insert(ins[store].pc);
                }
                phase_stores.clear();
                phase_stores_are_single_lane = true;
                phase_dispatcher_initializer_pc = UINT32_MAX;
                continue;
            }
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=unsynchronized-lds-store-before-ds-fminmax", in.pc);
            return false;
        }
        // EXEC=1 selects lane zero independently in every guest wave. The exact initializer uses
        // ordinary same-address stores, so more than one wave would race even though each wave has
        // only one active lane. Keep the title-observed single-wave workgroup admissible and leave a
        // different launch shape fail-visible until its cross-wave ownership can be proved.
        if (!at_most_one_guest_wave) {
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=multiwave-lds-fminmax-initializer", in.pc);
            return false;
        }

        synth_before.push_back(i);
        synth_before.push_back(i + 6);
        phase_stores.clear();
        phase_stores_are_single_lane = true;
        phase_dispatcher_initializer_pc = UINT32_MAX;
        i += 9; // both synthesized boundaries belong to this complete live atomic/gather group
    }

    std::vector<uint32_t> synth_pcs;
    synth_pcs.reserve(synth_before.size());
    for (size_t index : synth_before) synth_pcs.push_back(ins[index].pc);
    for (auto it = synth_before.rbegin(); it != synth_before.rend(); ++it) {
        Rdna2Inst barrier;
        barrier.pc = ins[*it].pc; // boundary immediately before the atomic at this guest PC
        barrier.fmt = Rdna2Format::SOPP;
        barrier.opcode = 0x0a;
        barrier.words[0] = 0xbf8a0000u;
        barrier.len_dwords = 0;   // emitter-only marker; never part of the guest byte stream
        ins.insert(ins.begin() + static_cast<std::ptrdiff_t>(*it), barrier);
    }

    // Prove each synthesized boundary is top-level even when the compact structurizer (rather than
    // the phase dispatcher) owns the program. Branches and loops may finish before the boundary,
    // but no edge may skip it, enter it from the far side, or carry only part of a workgroup back
    // across it. Traps/indirect PC changes before it also fail closed. This is the same edge
    // invariant used by the barrier-phase route without its unrelated >2-branch selection policy.
    for (uint32_t barrier_pc : synth_pcs) {
        bool uniform = true;
        for (const Rdna2Inst& in : ins) {
            if (in.pc < barrier_pc &&
                (in.is_end ||
                 (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12) ||
                 (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)))
                uniform = false;
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a)
                continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                uniform = false;
        }
        if (!uniform) {
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=lds-fminmax-publication-barrier-not-workgroup-uniform",
                barrier_pc);
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt, uint32_t lds_bytes,
                                     uint32_t compute_pgm_rsrc1,
                                     bool force_cfg_for_test,
                                     uint32_t local_x_for_test,
                                     uint32_t threads_x_for_test) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // This shell publishes register state after the structured region, so both terminal arms can
    // converge in the host SPIR-V without inventing further guest execution. Graphics exports keep
    // separate side-effect bookkeeping and remain on the conservative reject path for this shape.
    (void)extend_terminating_if_else(code, dwords, ins);
    // The synthetic test shell is one Wave64 workgroup, matching the live GTA dispatch.
    LdsFminmaxSynchronization lds_fminmax_synchronization;
    if (!prepare_lds_fminmax_synchronization(
            ins, {}, true, &lds_fminmax_synchronization))
        return {};
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    b.atomicized_lds_store_pcs = lds_fminmax_synchronization.atomicized_store_pcs;
    b.compute_pgm_rsrc1 = compute_pgm_rsrc1;
    // Size the LDS array from the shader's real allocation when known (#130): bytes -> dwords, at
    // least the ds ops need, clamped to the RDNA2 64 KB (16384-dword) max. 0 keeps the 16 KB default.
    if (lds_bytes) {
        uint32_t dw = (lds_bytes + 3) / 4;
        b.lds_dwords = dw > 16384u ? 16384u : (dw ? dw : 1u);
    }
    if (!local_x_for_test || local_x_for_test > 1024) return {};
    b.begin(num_inputs ? num_inputs : 1, rt, local_x_for_test, 1, 1, 64, 0);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);
    // Compute kernels have no EXP output; reject if one appears.
    const auto no_export = [](RegState&, const Rdna2Inst&){ return false; };
    const uint32_t initial_active =
        force_cfg_for_test && threads_x_for_test &&
                threads_x_for_test % local_x_for_test != 0
            ? b.invocation_within_extent(threads_x_for_test, 1, 1)
            : 0;
    const bool emitted = force_cfg_for_test
        ? emit_cfg_state_machine(
              b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
              /*allow_smem*/true, no_export, code, dwords, initial_active, false)
        : emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                    /*allow_smem*/true, no_export, code, dwords,
                    nullptr, true, 0, false, lds_fminmax_synchronization.needs_dispatcher);
    if (!emitted) return {};
    auto it = rs.vreg.find((int)out_vgpr);
    uint32_t outbits = it == rs.vreg.end() ? b.uconst(0) : it->second;
    // If EXEC is still narrowed (a v_cmpx with no restore), masked-off lanes keep the output slot's prior
    // value; if it was restored to all-lanes-on, every lane stores.
    if (!rs.exec_narrowed) b.store_output(outbits);
    else                   b.store_output_pred(outbits, rs.exec);
    return b.finish();
}

std::vector<uint32_t> recompile_compute(const uint32_t* code, size_t dwords,
                                        const ShaderResourceTable* rt,
                                        const ComputeShaderConfig& config,
                                        RecompileDiagnosticContext diagnostic) {
    const bool has_null_guarded_raw_store = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_proven_null_guarded_raw_store);
    const bool has_nullable_output_raw_buffer = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_nullable_raw_buffer_marker_candidate);
    const bool has_selected_sbuffer_descriptor = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_selected_sbuffer_marker_candidate);
    const bool has_gta5_packed_pointer = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_packed_pointer_marker_candidate);
    const bool has_indirect_pointer_relocation = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_indirect_pointer_relocation_marker_candidate);
    const bool has_gta5_cf9200_no_backing = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_cf9200_no_backing_marker_candidate);
    const ShaderResource* selected_sbuffer_descriptor =
        has_selected_sbuffer_descriptor ? rt->by_fetch_pc(153u) : nullptr;
    // A resource table is externally constructible and can outlive the shader bytes or dispatch
    // that produced it. Re-establish the complete static guard and dynamic null-entry contract at
    // the final translation boundary before any marker is permitted to erase a real store.
    if (has_null_guarded_raw_store &&
        !rdna2_gta5_null_guarded_raw_store_dispatch(
            code, dwords, config.user_sgprs.data(), config.user_sgprs.size()))
        return {};
    if (has_nullable_output_raw_buffer &&
        !rdna2_gta5_nullable_output_dispatch(code, dwords, config, *rt))
        return {};
    if (has_selected_sbuffer_descriptor &&
        !rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, *rt))
        return {};
    if (has_gta5_packed_pointer &&
        !rdna2_gta5_packed_pointer_dispatch(code, dwords, config, *rt))
        return {};
    IndirectPointerRelocationProof indirect_pointer_proof;
    IndirectBufferRelocationInfo indirect_pointer_info;
    if (has_indirect_pointer_relocation &&
        !validate_rdna2_indirect_pointer_relocations(
            code, dwords, config, *rt,
            &indirect_pointer_proof, &indirect_pointer_info))
        return {};
    if (has_gta5_cf9200_no_backing &&
        !rdna2_gta5_cf9200_no_backing_dispatch(code, dwords, config, *rt))
        return {};
    const uint32_t local_x = std::max(1u, config.local_x);
    const uint32_t local_y = std::max(1u, config.local_y);
    const uint32_t local_z = std::max(1u, config.local_z);
    const uint32_t wave_size = config.wave_size == 32 ? 32u : 64u;
    const uint64_t local_count = static_cast<uint64_t>(local_x) * local_y * local_z;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // A dispatch-scoped proven-null BVH can collapse an exact no-hit exit and fully matched empty-stack
    // traversal cycle before generic shader-byte constant folding. Resource identity (including null
    // marker + fetch PC) is already part of the compute module cache key, so a later non-null dispatch
    // receives a distinct module.
    (void)rdna2_specialize_proven_null_bvh_paths(ins, rt, config.wave_size);
    (void)rdna2_specialize_zero_record_execz_paths(ins, rt, config.wave_size);
    (void)rdna2_specialize_shader_constant_branches(ins);
    // See recompile_valu: compute has no branch-external EXP state, so the common host-shell merge is
    // only a place to finish the invocation after either guest arm has terminated.
    (void)extend_terminating_if_else(code, dwords, ins);
    LdsFminmaxSynchronization lds_fminmax_synchronization;
    if (!prepare_lds_fminmax_synchronization(
            ins, diagnostic, local_count <= wave_size, &lds_fminmax_synchronization))
        return {};
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    b.atomicized_lds_store_pcs = lds_fminmax_synchronization.atomicized_store_pcs;
    b.diagnostic = diagnostic;
    b.gta5_selected_sbuffer_dispatch_validated = has_selected_sbuffer_descriptor;
    if (selected_sbuffer_descriptor && has_selected_sbuffer_descriptor)
        b.gta5_selected_sbuffer_soffset = selected_sbuffer_descriptor->selected_sbuffer_soffset;
    b.gta5_cf9200_no_backing_dispatch_validated = has_gta5_cf9200_no_backing;
    b.indirect_buffer_dispatch_validated = has_gta5_packed_pointer;
    if (has_gta5_packed_pointer) {
        const ShaderResource* packed = rt->by_fetch_pc(kGta5PackedPointerSourcePc);
        if (!packed || !is_gta5_packed_pointer_resource(*packed)) return {};
        b.indirect_buffer_binding = packed->binding;
        b.indirect_buffer_source_bytes = static_cast<uint32_t>(packed->size);
        b.indirect_buffer_slot_count = packed->indirect_buffer_slot_count;
        b.indirect_buffer_contract_tag = packed->indirect_buffer_contract_tag;
        b.indirect_buffer_header_bytes = packed->indirect_buffer_header_bytes;
        b.indirect_buffer_slot_bytes = packed->indirect_buffer_slot_bytes;
        const ShaderResource* atomic = rt->by_fetch_pc(kGta5PackedPointerAtomicSourcePc);
        if (!atomic || atomic->size != kGta5PackedPointerAtomicBindingBytes ||
            (atomic->gpu_addr & 7u) != 0u)
            return {};
        b.indirect_buffer_atomic_binding = atomic->binding;
        b.indirect_buffer_atomic_byte_offset = kGta5PackedPointerAtomicByteOffset;
    }
    if (has_indirect_pointer_relocation) {
        const ShaderResource* relocated = rt->by_fetch_pc(
            indirect_pointer_proof.source_fetch_pc);
        const auto& marker = relocated
            ? relocated->indirect_pointer_relocation
            : IndirectPointerRelocationBinding{};
        const uint32_t expected_carrier_version =
            indirect_pointer_proof.bound_kind ==
                    IndirectPointerBoundKind::StaticFootprint
                ? kIndirectPointerStaticFootprintLayout.version
                : indirect_pointer_proof.bound_kind ==
                        IndirectPointerBoundKind::DescriptorRange
                    ? kIndirectPointerDescriptorRangeLayout.version
                    : 0u;
        const uint64_t record_directory_offset =
            static_cast<uint64_t>(indirect_pointer_info.source_bytes) +
            kIndirectBufferRelocationHeaderBytes;
        const uint64_t segment_directory_offset = record_directory_offset +
            static_cast<uint64_t>(indirect_pointer_info.records.size()) *
                kIndirectBufferRelocationRecordBytes;
        if (!relocated || !is_indirect_pointer_relocation_resource(*relocated) ||
            !expected_carrier_version || marker.carrier_version != expected_carrier_version ||
            marker.record_count != indirect_pointer_info.records.size() ||
            marker.record_count != indirect_pointer_proof.record_count ||
            marker.segment_count != indirect_pointer_info.segments.size() ||
            indirect_pointer_info.source_bytes != relocated->size ||
            record_directory_offset > UINT32_MAX ||
            segment_directory_offset > UINT32_MAX ||
            marker.segment_directory_byte_offset != segment_directory_offset)
            return {};
        b.indirect_pointer_proof = &indirect_pointer_proof;
        b.indirect_pointer_binding = relocated->binding;
        b.indirect_pointer_source_bytes = indirect_pointer_info.source_bytes;
        b.indirect_pointer_record_count = marker.record_count;
        b.indirect_pointer_record_directory_byte_offset =
            static_cast<uint32_t>(record_directory_offset);
        b.indirect_pointer_segment_count = marker.segment_count;
        b.indirect_pointer_segment_directory_byte_offset =
            marker.segment_directory_byte_offset;
        b.indirect_pointer_payload_byte_offset =
            indirect_pointer_info.payload_byte_offset;
        b.indirect_pointer_carrier_bytes = marker.binding_bytes;
        b.indirect_pointer_source_stride = indirect_pointer_proof.source_stride;
        b.indirect_pointer_source_pointer_byte_offset =
            indirect_pointer_proof.pointer_byte_offset;
    }
    if (config.lds_bytes) {
        uint32_t dw = (config.lds_bytes + 3) / 4;
        b.lds_dwords = std::min(16384u, std::max(1u, dw));
    }
    const bool has_partial_workgroup = config.threads_x % local_x != 0 ||
                                       config.threads_y % local_y != 0 ||
                                       config.threads_z % local_z != 0;
    const BarrierPhasedCompute barrier_phases = analyze_barrier_phased_compute(ins);
    const bool partial_barrier_phases = config.exact_thread_extent && has_partial_workgroup &&
        barrier_phases.found && !barrier_phases.guarded;
    const bool exact_partial_dispatcher = config.exact_thread_extent &&
        has_partial_workgroup && (b.gta5_selected_sbuffer_dispatch_validated ||
                                  b.indirect_buffer_dispatch_validated ||
                                  has_indirect_pointer_relocation);
    b.native_subgroup_size = config.native_subgroup_size == wave_size &&
        local_count <= UINT32_MAX && local_count % wave_size == 0 ? wave_size : 0u;
    // A partial guest wave needs the portable dispatcher's per-lane ACTIVE bit. Native subgroup
    // operations cannot be entered by only the real prefix of the final host subgroup.
    if (partial_barrier_phases || exact_partial_dispatcher)
        b.native_subgroup_size = 0;
    // PROSPER_DBG: report the inputs to that decision, not just its outcome (#2429).
    //
    // Every wave-width-dependent lowering in this file gates on `b.native_subgroup_size` -- the
    // VCC-as-scalar-data path at :5604 most consequentially, since when it is 0 the guest's
    // `s_add_u32 sN, sM, vcc_lo` never resolves, the descriptor never lands, and the draw is
    // skipped. Nothing printed any of this, under any variable, so "was that path active on this
    // device?" could only be inferred from the adapter's advertised width.
    //
    // `config.native_subgroup_size` is NOT the adapter's advertised width -- it is the OUTPUT of
    // select_native_compute_subgroup_size() (gpu_executor.cpp), an adoption decision with THREE
    // `return 0` sites comprising 22 clauses -- 25 if `adoptable`'s four ANDed device checks are
    // counted individually, which is defensible since each is independently sufficient. It spans
    // device features, queue support, workgroup limits and the dispatch's own dimensions, and
    // yields 0 when it declines. Zero therefore means "no native width was adopted", NOT "the
    // device is narrower than wave_size", and the three cases below are distinguished for that
    // reason.
    //
    // Counted rather than estimated, because two lanes published two different guesses at it on the
    // same day (#2483 "~14", #2484 "roughly eight") and neither had derived the number.
    //
    // This line dedupes on the exact input tuple, so it answers "which combinations exist" cheaply --
    // a couple of lines for a whole boot. For a per-dispatch CENSUS (how many dispatches fall in
    // each category, which this instrument's dedupe destroys by design) use PROSPER_SUBGROUP_LOG in
    // gpu_executor.cpp instead.
    //
    // That inference is WRONG, and reporting only the effective value would preserve the error:
    // the expression above is zero for THREE independent reasons -- the device width not matching
    // `wave_size`, an implausible `local_count`, or a workgroup that is not a whole number of waves
    // (`local_count % wave_size`). A dispatch with a partial final wave disables the path on an
    // adapter whose width matches perfectly. #2429 attributes it entirely to the first cause, and
    // that is checkable only if all three inputs are printed.
    //
    // Deduplicated on the exact tuple rather than rate-limited, because the interesting event is a
    // DISTINCT combination appearing, not the hundredth repeat of one -- and a kernel that disables
    // the path for a different reason than its predecessors is exactly what a rate limit would drop.
    if (getenv("PROSPER_DBG")) {
        static std::mutex mx;
        // Keyed on the EXACT inputs, local_count included. An earlier revision packed
        // `local_count % wave_size` instead, which collapsed dispatches that differ only in
        // workgroup shape -- and the line prints `local=`, so one row then named whichever
        // instance arrived first and stood silently for the rest. Measured on GTA V's six
        // native=0 shapes, that key produced three rows, one of which represented 1024, 256,
        // 256 and 256 while printing only 1024 -- and the 256-wide ones were the multi-wave
        // case that mattered. A diagnostic may aggregate, but it must not name one member of
        // a bucket as though it were the bucket.
        static std::set<std::tuple<uint32_t, uint32_t, uint64_t, uint32_t>> seen;
        std::lock_guard<std::mutex> lk(mx);
        if (seen.insert(std::make_tuple(config.native_subgroup_size, wave_size,
                                        local_count, b.native_subgroup_size)).second)
            std::fprintf(stderr,
                         "[subgroup-width] device=%u wave=%u local=%llu local%%wave=%llu -> "
                         "native_subgroup_size=%u (%s)\n",
                         config.native_subgroup_size, wave_size,
                         (unsigned long long)local_count,
                         (unsigned long long)(local_count % (wave_size ? wave_size : 1u)),
                         b.native_subgroup_size,
                         b.native_subgroup_size
                             ? "width-dependent lowerings ENABLED"
                             : (partial_barrier_phases
                                    ? "DISABLED: partial barrier phases require the portable "
                                      "dispatcher"
                                : (config.native_subgroup_size == 0
                                    ? "DISABLED: no native width adopted -- "
                                      "select_native_compute_subgroup_size() declined"
                                    : (config.native_subgroup_size != wave_size
                                           ? "DISABLED: adopted width != wave_size"
                                           : (local_count > UINT32_MAX
                                                  ? "DISABLED: local_count exceeds the plausibility "
                                                    "guard"
                                                  : "DISABLED: workgroup is not a whole number "
                                                    "of waves")))));
    }
    b.native_storage_format_support = config.native_storage_format_support;
    b.storage_buffer_int64_atomics = config.storage_buffer_int64_atomics;
    b.packed_r11_storage = config.packed_r11_storage;
    b.compute_pgm_rsrc1 = config.compute_pgm_rsrc1;
    b.begin(1, rt, local_x, local_y, local_z, wave_size,
            static_cast<uint32_t>(config.user_sgprs.size()));
    b.allow_b32_masks = wave_size == 32;
    if (has_indirect_pointer_relocation &&
        indirect_pointer_proof.bound_kind ==
            IndirectPointerBoundKind::DescriptorRange)
        b.declare_indirect_pointer_descriptor_capture();
    b.declare_guest_scratch(scratch);
    uint32_t initial_dispatch_active = 0;
    if (partial_barrier_phases || exact_partial_dispatcher)
        initial_dispatch_active = b.invocation_within_extent(
            config.threads_x, config.threads_y, config.threads_z);
    else if (config.exact_thread_extent && has_partial_workgroup)
        b.guard_invocation_extent(config.threads_x, config.threads_y, config.threads_z);

    RegState rs;
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.exec = b.btrue();
    // Inline descriptors are represented by the resource table, not scalar SSA values. Leaving
    // their SGPR range absent also preserves the existing direct-provenance rule: a format MUBUF may
    // fall back to by_sgpr_base only while its SRSRC has not been overwritten by shader code.
    std::set<uint32_t> descriptor_sgprs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            uint32_t words = (resource.cls == ResourceClass::Texture ||
                              resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; word++)
                descriptor_sgprs.insert(resource.sgpr_base + word);
        }
    }
    for (size_t i = 0; i < config.user_sgprs.size(); i++) {
        const uint32_t value = b.load_push_constant(static_cast<uint32_t>(i));
        if (descriptor_sgprs.count(static_cast<uint32_t>(i)))
            rs.sreg_input[static_cast<int>(i)] = value;
        else
            rs.sreg[static_cast<int>(i)] = value;
    }

    rs.vreg[0] = b.localid_comp[0];
    if (config.tidig_comp_cnt >= 1) rs.vreg[1] = b.localid_comp[1];
    if (config.tidig_comp_cnt >= 2) rs.vreg[2] = b.localid_comp[2];

    int system_sgpr = static_cast<int>(config.user_sgprs.size());
    if (config.tgid_x_en) rs.sreg[system_sgpr++] = b.groupid[0];
    if (config.tgid_y_en) rs.sreg[system_sgpr++] = b.groupid[1];
    if (config.tgid_z_en) rs.sreg[system_sgpr++] = b.groupid[2];
    if (config.tg_size_en)
        rs.sreg[system_sgpr] = b.uconst(local_x * local_y * local_z);

    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/true, [](RegState&, const Rdna2Inst&) { return false; },
                   code, dwords, nullptr, true, initial_dispatch_active, false,
                   lds_fminmax_synchronization.needs_dispatcher))
        return {};
    // Exact dispatch contracts execute their partial final wave through the CFG dispatcher's ACTIVE
    // bit. Padded Vulkan lanes stay in the dispatcher and its synthesized workgroup barriers, but
    // cannot execute guest memory effects. The full program, launch, and resource proof above is the
    // authority boundary for extending the selected-SBUFFER path to the packed-pointer program.
    if (exact_partial_dispatcher && b.uses_barrier)
        b.partial_barrier_phases_emitted = true;
    // The entry guard is intentionally divergent only in the final partial workgroup. Vulkan requires
    // every workgroup invocation to participate uniformly in OpControlBarrier, including barriers the
    // recompiler synthesizes for wave operations. Reject this uncommon combination instead of emitting
    // a module that could deadlock or observe undefined workgroup-memory behavior.
    if (has_partial_workgroup && b.uses_barrier && !b.partial_barrier_phases_emitted) {
        log_recompile_diagnostic(
            b.diagnostic, "compute-recompile-reject", "terminal",
            "reason=partial-workgroup-barrier threads=%ux%ux%u local=%ux%ux%u",
            config.threads_x, config.threads_y, config.threads_z,
            local_x, local_y, local_z);
        return {};
    }
    return b.finish();
}

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords,
                                     std::vector<RecompileUnsupportedSite>* sites) {
    if (sites) sites->clear();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    uint32_t synthetic_branch_pc = UINT32_MAX;
    (void)extend_terminating_if_else(code, dwords, ins, nullptr, &synthetic_branch_pc);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // A scratch builder/state so emit_alu can run; its emitted code is discarded — we only want `ok`.
    SpirvCompute b; b.begin(1);
    b.cselect_b64_low_only_pcs = proven_cselect_b64_low_only_pcs(ins);
    b.cselect_b64_low_only_analysis_done = true;
    // Coverage is deliberately a context-free instruction census. Preserve its historical
    // CSELECT exception, but do not let a newly recognized whole-CFG logical lifetime poison the
    // scratch compute state and make later instructions look unsupported. Real compute emission
    // runs the complete proof in emit_body.
    b.vcc_b32_low_only_pcs =
        proven_wave64_vcc_b32_low_only_pcs(ins, /*include_logical*/false);
    b.vcc_b32_low_only_analysis_done = true;
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // emit_alu is a per-instruction check and rejects control-flow branches, but the whole-stream emit_body
    // RECONSTRUCTS a counted loop and a forward uniform-if. Credit the branches emit_body consumes (loop
    // back-edge + exit, and the forward-if branch) as handled, so coverage matches what actually recompiles
    // (previously the MSAA-resolve loop shaders 031-034 were mis-flagged "blocked" at their s_cbranch_scc0).
    const CountedLoop cL = detect_counted_loop(ins);
    std::vector<ForwardIf> cFs;
    if (cL.found) {
        // Match emit_body's counted-loop composition: inspect the truncated prefix independently
        // from the loop exit/back-edge, then inspect the recursively-emitted suffix. Feeding the
        // complete stream to detect_forward_ifs makes the canonical exit look like an IF whose
        // alleged else terminator is the backward loop branch, so otherwise-valid prefix/postfix
        // branches are incorrectly counted as unsupported.
        std::vector<Rdna2Inst> prefix;
        for (const auto& in : ins) {
            if (in.pc >= cL.header_pc) break;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                branch_target(in) >= cL.header_pc) continue;
            prefix.push_back(in);
        }
        Rdna2Inst prefix_end;
        prefix_end.pc = cL.header_pc;
        prefix_end.is_end = true;
        prefix.push_back(prefix_end);
        bool prefix_rejected = false;
        std::vector<ForwardIf> prefix_ifs = detect_forward_ifs(
            prefix, /*allow_vcc*/false, code, dwords, &safe_branches, nullptr,
            &prefix_rejected, /*compute_wave_branches*/true);
        const bool prefix_crosses_loop = std::any_of(
            prefix_ifs.begin(), prefix_ifs.end(), [&](const ForwardIf& branch) {
                return branch.early_out ||
                    (branch.has_else ? branch.merge_pc : branch.target_pc) > cL.header_pc;
            });
        if (!prefix_rejected && !prefix_crosses_loop)
            cFs.insert(cFs.end(), prefix_ifs.begin(), prefix_ifs.end());

        std::vector<Rdna2Inst> suffix;
        for (const auto& in : ins) if (in.pc >= cL.exit_pc) suffix.push_back(in);
        std::vector<ForwardIf> suffix_ifs = detect_forward_ifs(
            suffix, /*allow_vcc*/false, code, dwords, &safe_branches, nullptr, nullptr,
            /*compute_wave_branches*/true);
        cFs.insert(cFs.end(), suffix_ifs.begin(), suffix_ifs.end());
    } else {
        cFs = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords,
                                 nullptr, nullptr, nullptr,
                                 /*compute_wave_branches*/true);   // matches the compute shell (#590)
    }
    auto cf_reconstructed = [&](const Rdna2Inst& i) {
        if (cL.found && (i.pc == cL.backedge_pc || i.pc == cL.exit_branch_pc)) return true;
        for (const auto& F : cFs)
            if (i.pc == F.branch_pc || (F.has_else && i.pc == F.sb_pc)) return true;
        return false;
    };

    RecompileCoverage cov;
    for (const auto& in : ins) {
        if (in.is_end) break;
        // The rewritten first s_endpgm is compiler-only control flow. It lets the structured emitter
        // reuse its ordinary if/else path, but coverage describes decoded guest instructions and has
        // always excluded s_endpgm, so do not inflate total/ALU with this synthetic arm skip.
        if (in.pc == synthetic_branch_pc) continue;
        cov.total++;
        if (in.fmt == Rdna2Format::EXP) { cov.exports++; continue; }   // handled by the stage recompilers
        bool ok = true;
        const SavedB64MaskSnapshot saved_masks = snapshot_saved_b64_masks(rs, in);
        const bool emitted = emit_alu(
            b, rs, in, ok, /*allow_exec_update*/true, &safe_branches,
            /*allow_smem*/true, /*rt*/nullptr, /*allow_wave*/true);
        if (emitted && ok)
            record_scalar_write(
                rs, in,
                allows_compute_scalar_vcc_bridge(b), saved_masks);
        bool handled = cf_reconstructed(in) || (emitted && ok);
        // Shapes the recompiler handles only in context (a resource table for MIMG sample/load/LOD/store
        // and buffer_load/store_format; a fragment stage for VINTRP). This table-less compute-shell pass
        // rejects them, so count them apart from truly-unsupported (cross-lane, etc.). Instruction-aware
        // for MIMG so deferred variants (NSA multi-dword addr; arrayed/cube/MSAA dims) are NOT overcounted
        // as recompilable — they still land in `unsupported`, matching what the recompiler actually accepts.
        auto table_dependent = [](const Rdna2Inst& i) {
            switch (i.fmt) {
                case Rdna2Format::MIMG: {
                    // Storage load/store handle 1D/2D/3D + 1D/2D_ARRAY (dims 0,1,2,4,5) and NSA.
                    // Sampled 2D_MSAA IMAGE_LOAD is narrower: only the exact consecutive-address or
                    // one-extra NSA [x,y,sample] shapes accepted by emit_alu are table-dependent.
                    // Do not credit dim7 or unused NSA address bytes merely because a T# could exist.
                    const bool st_dim = i.mimg_dim <= 2u || i.mimg_dim == 4u || i.mimg_dim == 5u;
                    if (i.opcode == 0x00u) {
                        if (st_dim) return true;
                        const bool msaa_address_shape = i.len_dwords == 2u ||
                            (i.len_dwords == 3u && (i.words[2] & 0xffff0000u) == 0u);
                        return i.mimg_dim == 6u && !i.mimg_unorm && !i.has_modifier &&
                               msaa_address_shape;
                    }
                    if (i.opcode == 0x01u || i.opcode == 0x09u)
                        return rdna2_mimg_zero_mip_shape(i);
                    if (i.opcode == 0x08u) return st_dim;                       // image_store (no per-sample MSAA store)
                    if (i.opcode == 0x0fu || i.opcode == 0x11u)   // image_atomic_swap/add R32_UINT 2D / 2D_ARRAY
                        // #2265: 2D_ARRAY (dim 5) admitted alongside 2D. This is the COVERAGE
                        // predicate -- it decides whether the instruction counts as supported for
                        // the census, and it was the last of the four sites still reporting the
                        // arrayed form as unsupported after #2272 widened the lowering. A 2D_ARRAY
                        // arrayed atomic reaches its layer through the address VGPRs, not through a
                        // longer ENCODING: CrossWorlds' own instruction is `dim=5 ... len=2`, so
                        // `len_dwords` is the NSA-vs-packed encoding length and pinning it to 3 for
                        // dim 5 would reject exactly the instruction this admits. Matches the
                        // lowering gate, which rejects only `len_dwords < 2`.
                        return (i.mimg_dim == 1u || i.mimg_dim == 5u) && i.mimg_dmask == 1u &&
                               !i.mimg_unorm && i.len_dwords >= 2u;
                    // image_get_resinfo 1D/2D/3D, plus 2D_ARRAY (dim 5), which lowers as Dim_2D with
                    // Arrayed and reports the layer count as its third result (#2790). CUBE (dim 3) is
                    // still declined: its stacked-face lowering (#273) would have to promise face-count
                    // semantics for the third result that this has not established.
                    if (i.opcode == 0x0eu) return i.mimg_dim <= 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x60u)                                     // fragment image_get_lod 2D
                        return i.mimg_dim == 1u && i.len_dwords == 2u &&
                               !mimg_get_lod_has_unmodeled_controls(i) &&
                               (i.mimg_dmask & 0x3u) && !(i.mimg_dmask & ~0x3u);
                    // sample*: 2D (NSA ok); plus implicit-LOD image_sample (0x20) / LOD-0 image_sample_lz
                    // (0x27) from a 3D texture; sample_b (0x25) and gather4_lz (0x47) are 2D. 2D_ARRAY (dim 5)
                    // is accepted for all sample paths and handled as its base 2D slice (array index dropped,
                    // #325) — so array-sampling draws recompile+render instead of being skipped.
                    // 0xa0 is the high-bit sibling of image_sample (0x20), lowered identically (GTA V, #1145).
                    if (i.opcode == 0x20u || i.opcode == 0x27u || i.opcode == 0xa0u)
                        return i.mimg_dim == 1u || i.mimg_dim == 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x22u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    if (i.opcode == 0x2fu)
                        return i.mimg_dim == 1u || i.mimg_dim == 3u || i.mimg_dim == 5u;
                    if (i.opcode == 0x24u || i.opcode == 0x25u || i.opcode == 0x47u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    return false;
                }
                case Rdna2Format::MUBUF:  return i.opcode <= 0x07u ||                    // load/store_format_*
                                                 (i.opcode >= 0x0Cu && i.opcode <= 0x0Fu);  // load_dword/x2/x4/x3 (need the V#)
                case Rdna2Format::MTBUF:  return i.opcode <= 0x07u && !i.mtbuf_tfe;
                // Wide scalar loads are descriptor-table fetches. A real resource table lets the
                // emitter preserve their provenance without reading a fallback buffer, including the
                // register-offset bindless form; the table-less coverage shell must not call that a
                // newly unsupported instruction.
                case Rdna2Format::SMEM:   return i.opcode == 0x02u || i.opcode == 0x03u;
                case Rdna2Format::VINTRP: return true;               // handled in the fragment shell
                default: return false;
            }
        };
        if (handled) { cov.alu++; }
        else if (table_dependent(in)) { cov.table_dependent++; }
        else {
            cov.unsupported++;
            if (cov.first_bad_fmt < 0) {
                cov.first_bad_fmt = (int)in.fmt;
                cov.first_bad_op = in.opcode;
                cov.first_bad_pc = in.pc;
            }
            // Recorded from the SAME branch that increments the counter, so the enumeration cannot
            // disagree with `unsupported` about which instructions are in the class.
            if (sites) sites->push_back({(int)in.fmt, in.opcode, in.pc});
        }
    }
    return cov;
}

namespace {


}  // namespace

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
