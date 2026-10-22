#pragma once
#include "gpu/recompiler/rdna2_cfg_trip_settings.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_must_dataflow.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_wave_uniform.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_barrier_phases.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_scalar_merge.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_forward_admission.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_uniform_branch.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_divergent_detector.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_loop_admission.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_counted_loop.hpp" // IWYU pragma: export

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_cfg_support.hpp — helpers shared by rdna2_emit_cfg.cpp and rdna2_to_spirv.cpp.
// INTERNAL to src/gpu/recompiler/.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
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
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_pcrel_tables.hpp" // IWYU pragma: export

namespace prosper::gpu {

uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in);
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source);
// The VALU sources that are lane masks: SRC2 of v_cndmask_b32_e64 and the VOP3B carry-in forms.
bool valu_lane_mask_source(const Rdna2Inst& in, uint32_t source);
// scalar_alu_source_words for a program of a known wave width: a Wave32 lane mask is ONE SGPR,
// where the width-free form charges every lane mask the Wave64 pair (#5202). Any other wave_size
// value (64, or unknown) keeps the pair.
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);
// The words the source MARKS charge (scalar_source_marks and the dispatcher's
// spilled_vector_source_marks): the wave-aware width, except that a Wave32 VCC_LO lane mask still
// charges VCC_HI, the only witness the marks have of an unprovided entry VCC.
uint32_t scalar_source_mark_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);
bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
// Same proof, applied to a scalar branch's SCC instead of a VCC lane mask (#3713).
bool scc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);

// The header EXEC is a subset of its preheader only when every possible iteration preserves or
// narrows EXEC. This code-only certificate does not authenticate any carried register's data.
bool structured_loop_only_narrows_exec(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);
bool narrowing_loop_exec_admission(const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                                   const RegState& entry);
// A wave-scalar word the loop [header, back edge] may write that is live at its exit: an SGPR,
// VCC or M0 word (a mask pair counts, as scalar data), or kLoopLiveScc. -1 when there is none.
// The narrowing certificate requires -1; rdna2_loop_wave_scalar_route.cpp routes the others.
constexpr int kLoopLiveScc = 253;
int loop_live_wave_scalar(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);
// The launch and stream a fresh exact-dispatcher attempt of this program would compile: an
// authentic native Wave64 compute entry whose original words are `code`.
bool native_exact_retry_authentic(const SpirvCompute& b, const uint32_t* code, size_t dwords);
// A refused native store-loop header may select the existing exact dispatcher. This proves
// only eligibility to try a fresh builder, never DATA, scalar, resource or effect admission.
bool closed_compute_store_loop_data_retry(const SpirvCompute& b, const RegState& rs,
                                          const DivLoop& loop, const Rdna2Inst& header,
                                          uint32_t preheader_exec, const uint32_t* code,
                                          size_t dwords);
uint32_t make_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                      std::size_t& patch, bool admitted);
// The header EXEC phi is a subset of the preheader EXEC when every back edge carries a subset of
// it (induction over iterations). Two certificates grant that fact: the code-only compute one
// (narrowing_loop_exec_admission), and a checked one for fragment loops whose EXEC writes are
// all narrowing forms: assumed at the header, then verified at the back edge from the
// EXEC subset facts the body actually recorded. A failed check refuses the loop (#5039).
// Implemented in rdna2_loop_exec_subset.cpp.
class LoopExecSubset {
public:
    LoopExecSubset(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                   const RegState& entry);
    bool admitted() const { return certified_ || checked_; }
    void open(const RegState& preheader) {
        preheader_exec_ = preheader.exec;
        preheader_full_ = !preheader.exec_narrowed;
    }
    // At the back edge, with `rs` holding the body's final EXEC.
    bool holds(SpirvCompute& b, const RegState& rs) const;

private:
    bool certified_ = false;
    bool checked_ = false;
    uint32_t preheader_exec_ = 0;
    bool preheader_full_ = false;
};
// The EXEC phi at a one-armed structured if's merge. When the taken arm only narrowed the skipped
// edge's EXEC, the merge is a subset of it too (rdna2_loop_exec_subset.cpp).
uint32_t merge_exec_arm(SpirvCompute& b, RegState& rs, uint32_t skipped_exec,
                        uint32_t skipped_block, uint32_t taken_exec, uint32_t taken_block);
template <class PhiRecords>
void open_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                  PhiRecords& phis, LoopExecSubset& proof) {
    std::size_t patch;
    proof.open(rs);
    const uint32_t phi = make_narrowing_loop_exec_phi(b, rs, preheader, patch, proof.admitted());
    phis.push_back({0, 4, phi, patch});
}

inline bool reg_operand(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

inline uint32_t scalar_branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

}   // namespace prosper::gpu
