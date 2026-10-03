#include "gpu/recompiler/fragment_packet_exports_internal.hpp"
#include "gpu/recompiler/fragment_packet_exports.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
bool emit_packet_architectural_export(SpirvCompute& b, RegState& current, const Rdna2Inst& in,
                                      uint32_t base, uint32_t stride, uint32_t enabled) {
    const uint32_t source_mask = fragment_packet_export_source_mask(in.exp_en, in.exp_compr);
    const uint32_t fields[] = {b.uconst(1),
                               b.sel(current.exec, b.uconst(1), b.uconst(0)),
                               enabled,
                               b.uconst(in.exp_target),
                               b.uconst(in.exp_en),
                               b.uconst(in.exp_compr),
                               b.uconst((in.words[0] >> 11) & 1u),
                               b.uconst((in.words[0] >> 12) & 1u)};
    for (uint32_t field = 0; field < 8; ++field)
        b.store_output_word(fields[field], stride, base + field, 0);
    for (uint32_t word = 0; word < 4; ++word) {
        if (!(source_mask & (1u << word))) continue;
        bool valid = true;
        const auto bits = operand_bits(b, current, in, in.src[word], &valid);
        if (!valid) return false;
        // Eager internal storage is NOT guest entry authority. Runtime per-lane validity checks
        // the original active read; EXEC-off transport zero has observation bit CLEAR.
        b.store_output_word(b.sel(current.exec, bits, b.uconst(0)), stride, base + 8 + word, 0);
    }
    b.store_output_word(b.uconst(in.pc), stride, base + 12, 0);
    b.store_output_word(b.sel(current.exec, b.uconst(source_mask), b.uconst(0)), stride, base + 13,
                        0);
    return true;
}
}   // namespace prosper::gpu
