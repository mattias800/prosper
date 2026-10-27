// A memory result has its own input/safety obligation, independently of its address roots.
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_placeholder_origin_memory.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
void note_placeholder_scalar_load(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SMEM) return;
    bool supplied = true, writes = false;
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        writes = true;
        for (uint32_t k = 0; k < width; ++k) {
            const int word = base + static_cast<int>(k);
            const auto roots = rs.sreg_loop_blanket.find(word);
            supplied = supplied && rs.sreg.contains(word) && rs.sreg.at(word) != 0 &&
                       !sreg_word_may_be_fabricated(rs, word) &&
                       !rs.unavailable_saved_mask_words.contains(word) &&
                       (roots == rs.sreg_loop_blanket.end() || roots->second.empty());
        }
    });
    // Called after successful emission and scalar-write bookkeeping. Scalar loads publish
    // resolved resource/embedded-table data; their address is not the loaded word's value.
    const auto* p = rs.placeholder_origin_program ? rs.placeholder_origin_program.get()
                                                  : b.placeholder_origin_program.get();
    if (!p || !p->observations) return;
    auto [at, inserted] =
        p->observations->arbitrary_memory_results.emplace(in.pc, writes && supplied);
    if (!inserted) at->second = at->second && writes && supplied;
    if (at->second)
        note_placeholder_native_input(b, rs, in);
    else
        p->observations->native_inputs.erase(in.pc);
}

void note_placeholder_memory_contract(const SpirvCompute& b, const RegState& rs,
                                      const Rdna2Inst& in, const ScalarSourceMarks& source) {
    const auto* p = rs.placeholder_origin_program ? rs.placeholder_origin_program.get()
                                                  : b.placeholder_origin_program.get();
    if (!p || !p->observations) return;
    const bool memory = in.fmt == Rdna2Format::DS || in.fmt == Rdna2Format::FLAT ||
                        in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF ||
                        in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::SMEM;
    if (!memory || in.fmt == Rdna2Format::SMEM) return;
    // A genuine address without borrowed loop assumptions supplies the existing result.
    const bool genuine_address =
        !source.placeholder && source.blanket_roots.empty() && source.scoped_blanket_roots.empty();
    bool contract = arbitrary_read_form(in) && arbitrary_read_resource_genuine(rs, in);
    if (in.fmt == Rdna2Format::DS) contract = contract && (b.lds_bound_reads || genuine_address);
    if (in.fmt == Rdna2Format::FLAT)
        contract = !rdna2_instruction_may_write_memory(in) &&
                   (genuine_address ||
                    dispatcher_arbitrary_read(in, {}, {}, false, b.indirect_pointer_proof));
    auto [at, inserted] = p->observations->arbitrary_memory_results.emplace(in.pc, contract);
    if (!inserted) at->second = at->second && contract;
}
} // namespace prosper::gpu
