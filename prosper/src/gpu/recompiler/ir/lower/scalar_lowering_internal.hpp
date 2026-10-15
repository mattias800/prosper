// scalar_lowering_internal.hpp — the scalar-ALU lowering classes shared by ir/lower/sop*.cpp.
//
// Each SALU format (SOP1, SOP2, SOPC, SOPK) is lowered by one class. `lower()` tries the format's
// ROUTES in a fixed order and then falls through to the plain scalar-data path, which goes through
// the IR (ir/ops/scalar_alu.hpp). A route is one shape of the instruction that the IR does not model
// yet — a wave mask kept in the Bool domain, a VCC/EXEC bridge, a proof-carrying special case. The
// routes were moved verbatim out of emit_alu (ADR 0012); each returns the value emit_alu returned
// for that shape, or nullopt when the instruction is not its shape. The order of the routes is the
// order of the old `if` chain and is load-bearing: an earlier route wins.
//
// The members are emit_alu's locals under the same names, so the moved text compiles unchanged.
#pragma once

#include "gpu/recompiler/ir/lower/scalar_alu_lowering.hpp"
#include "gpu/recompiler/ir/ops/scalar_alu.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_movrels.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_mask_write.hpp"
#include "gpu/recompiler/rdna2_partial_vcc_word.hpp"
#include <cstdio>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>

namespace prosper::gpu::ir {

struct ScalarLoweringContext {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    bool& ok;
    bool allow_wave;
    uint32_t val(const Operand& o) { return operand_bits(b, rs, in, o, &ok); }
};

class Sop1Lowering : ScalarLoweringContext {
public:
    explicit Sop1Lowering(const ScalarLoweringContext& context) : ScalarLoweringContext(context) {}
    bool lower();

private:
    bool lower_data();
    std::optional<bool> route_cmov_b64_odd_pair();
    std::optional<bool> route_wqm_numeric_special_destination();
    std::optional<bool> route_wqm_numeric();
    std::optional<bool> route_quadmask_b64();
    std::optional<bool> route_ff1_b64_native64();
    std::optional<bool> route_ff1_b32_data();
    std::optional<bool> route_ff1_b32_wave32();
    std::optional<bool> route_exec_half_from_vcc_half();
    std::optional<bool> route_exec_half_immediate();
    std::optional<bool> route_mov_b32_mask();
    std::optional<bool> route_mov_b32_exec_lo_full();
    std::optional<bool> route_saveexec_b32();
    std::optional<bool> route_wqm_b32();
    std::optional<bool> route_not_b32_mask();
    std::optional<bool> route_bcnt1_b64();
    std::optional<bool> route_b64_mask_ops();
    std::optional<bool> route_not_vcc_half_self();
    std::optional<bool> route_getpc_b64();
    std::optional<bool> route_bcnt1_b64_one_lane();
    std::optional<bool> route_ff1_b64_one_lane();
    std::optional<bool> route_flbit();
    std::optional<bool> route_bitset();
    std::optional<bool> route_brev_exec_lo_wave32();
    std::optional<bool> route_bitreplicate_b64_b32();
    std::optional<bool> route_reject_exec_half_data();
    std::optional<bool> route_entry_m0_save();
    std::optional<bool> route_entry_m0_restore();
};

class Sop2Lowering : ScalarLoweringContext {
public:
    explicit Sop2Lowering(const ScalarLoweringContext& context) : ScalarLoweringContext(context) {}
    bool lower();

private:
    bool lower_data();
    // emit_alu's case-scope predicates, evaluated by lower() at the point the old chain did.
    bool gtav_wave32_vcchi_scalar_packet = false;
    bool mixed_wave32_mask_scalar_data = false;
    // emit_alu's case-scope lambdas, now members (sop2.cpp).
    bool wave32_mask_operand(const Operand& source) const;
    bool wave32_live_mask_operand(const Operand& source) const;
    bool exact_wave32_scalar_dword(const Operand& source) const;
    bool scalar_data_operand(const Operand& source) const;
    // The instruction's plain data operation through its IR definition (ir/ops/scalar_alu.hpp).
    ScalarOpIds emit_data_op(uint32_t s0, uint32_t s1);

    std::optional<bool> route_wave64_vcc_lo_cselect();
    std::optional<bool> route_b32_mask_cselect();
    std::optional<bool> route_cselect_b32_vcc_lo_bridge();
    std::optional<bool> route_exec_lo_logical_data();
    std::optional<bool> route_exec_lo_bfm_data();
    std::optional<bool> route_wave32_b32_mask_logical();
    std::optional<bool> route_pack_halfwords();
    std::optional<bool> route_bfm_b64();
    std::optional<bool> route_cselect_b64();
    std::optional<bool> route_b64_mask_logical();
    std::optional<bool> route_lshr_b64_exec_pack();
    std::optional<bool> route_shift_b64();
    std::optional<bool> route_bfe_u64();
    std::optional<bool> route_vcc_half_b32_logical();
};

class SopcLowering : ScalarLoweringContext {
public:
    explicit SopcLowering(const ScalarLoweringContext& context) : ScalarLoweringContext(context) {}
    bool lower();

private:
    bool lower_data();
    bool lower_compare_u64(uint32_t a, uint32_t c);
    std::optional<bool> route_wave32_mask_compare();
    std::optional<bool> route_b64_mask_compare();
};

class SopkLowering : ScalarLoweringContext {
public:
    explicit SopkLowering(const ScalarLoweringContext& context) : ScalarLoweringContext(context) {}
    bool lower();

private:
    bool lower_data(const ScalarOpDef& def);
};

} // namespace prosper::gpu::ir
