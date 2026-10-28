// Wave32 one-word masks have independent lifetimes, including saved EXEC in VCC_HI.
#include "gpu/recompiler/rdna2_scalar_vcc_retirement.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"

namespace prosper::gpu {
void retire_scalar_vcc_mask_view(RegState& state, const Rdna2Inst& instruction) {
    const auto& effects = rdna2_instruction_effects(instruction);
    const bool single_word =
        state.vector_wave_size == 32 && effects.known() && effects.sdst_dwords == 1 &&
        (instruction.fmt == Rdna2Format::SOP1 || instruction.fmt == Rdna2Format::SOP2) &&
        (instruction.dst.value == 106 || instruction.dst.value == 107);
    const int untouched = single_word ? 213 - instruction.dst.value : -1;
    // CONFIDENCE: HIGH -- a B32 destination writes one physical word (ISA 70648, 5.3/5.4).
    // Only an existing one-word lifetime survives; a B64 alias still overlaps the destination.
    if (untouched != 106 || !state.sreg_bool_b32.contains(106)) state.vcc = 0;
    for (int word : {106, 107}) {
        if (word == untouched && state.sreg_bool_b32.contains(word)) continue;
        state.sreg_bool.erase(word);
        state.sreg_bool_narrowed.erase(word);
        state.sreg_bool_b32.erase(word);
    }
}
}   // namespace prosper::gpu
