// vopc_lowering_internal.hpp — the VOPC lowering class shared by ir/lower/vopc*.cpp.
//
// A compare is two halves. The VALUE is the per-lane test result, a Bool computed from the two
// source words; it is opcode-specific (vopc_compare_values.cpp, and ir/ops for the opcodes that
// have a definition). The MASK WRITE is what every compare then does with that Bool, and it is
// the same for all of them (vopc_mask_write.cpp). The members are emit_alu's locals of the same
// names; the order lower() calls the pieces in is the old case body's order and is load-bearing.
#pragma once

#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>

namespace prosper::gpu::ir {

class VopcLowering : LoweringContext {
public:
    VopcLowering(const LoweringContext& context, bool allow_exec_update)
        : LoweringContext(context), allow_exec_update(allow_exec_update) {}
    bool lower();

private:
    struct Integer64Halves { uint32_t lo = 0, hi = 0; };

    bool allow_exec_update;
    uint32_t ra = 0, rc = 0;   // the raw source words
    uint32_t a = 0, c = 0;     // the sources after modifiers / SDWA selects
    uint32_t eff = 0;          // the v_cmp_* opcode (a v_cmpx_* maps to its compare)
    bool is_cmpx = false;
    bool integer64_compare = false;
    bool compare_wave_uniform = false;
    bool guest_compare_fact = false;
    uint32_t zero_cmp = 0;
    uint32_t magnitude_cmp = 0, magnitude_word = 0;
    Integer64Halves a64, c64;

    bool scalar_data_operand(const Operand& source) const;
    bool scalar_integer64_operand(const Operand& source) const;
    void zero_compare();
    void magnitude_compare();
    void source_modifiers();
    void integer_sdwa_sources();
    Integer64Halves integer64_halves(const Operand& operand, uint32_t lo);
    void record_guest_compare_facts(uint32_t cmp);

    // The comparison value (vopc_compare_values.cpp). Clears `ok` for an opcode it lacks.
    uint32_t compare_value();
    uint32_t compare_integer16();
    uint32_t compare_integer64_signed();
    uint32_t compare_integer64_unsigned();
    uint32_t compare_class_f32();
    uint32_t compare_f16();
};

} // namespace prosper::gpu::ir
