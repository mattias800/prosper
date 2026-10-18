// #4900: integer D16 stores consume packed unsigned halves, not floating-point half values.
#include "gpu/recompiler/rdna2_mimg_store.hpp"
#include "gpu/recompiler/ir/lower/image_store_values_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
bool unsigned_d16_store_supported(const Rdna2Inst& instruction) {
    // RDNA2 ISA Table41 / section8.2.5: D16 stores convert packed 16-bit elements according to
    // the resource's numeric format. CONFIDENCE: HIGH on packing and legal integer stores.
    // Sparse D16 store-mask placement is not established for this generation; only contiguous
    // low components are newly admitted. A16/status/cache variants need separate authority.
    const uint32_t mask = instruction.mimg_dmask;
    return instruction.opcode == 8u && mask && !(mask & (mask + 1u)) && !instruction.mimg_a16 &&
           !instruction.mimg_r128 && !instruction.mimg_tfe && !instruction.mimg_lwe &&
           !instruction.mimg_dlc && !instruction.mimg_glc && !instruction.mimg_slc &&
           !instruction.mimg_reserved && !instruction.has_modifier;
}

uint32_t mimg_d16_store_component(SpirvCompute& builder, uint32_t packed, uint32_t half,
                                  bool unsigned_data) {
    return ir::lower_image_store_d16_word(builder, packed, half, unsigned_data);
}
}   // namespace prosper::gpu
