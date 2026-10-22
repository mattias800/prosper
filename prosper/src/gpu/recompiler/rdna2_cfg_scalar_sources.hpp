// Scalar input marks and saved-mask snapshots; private to the recompiler.
#pragma once
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <cstdint>

namespace prosper::gpu {
struct Rdna2Inst;
struct RegState;
struct SpirvCompute;
struct SavedB64MaskSnapshot;
bool s_movrels_b32_result_is_scalar_data(const Rdna2Inst& in);
uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in);
ScalarSourceMarks scalar_source_marks(const RegState& rs, const Rdna2Inst& in);
SavedB64MaskSnapshot snapshot_saved_b64_masks(const RegState& rs, const Rdna2Inst& in);
SavedB64MaskSnapshot snapshot_saved_b64_masks(SpirvCompute& b, const RegState& rs,
                                              const Rdna2Inst& in);
uint32_t scalar_source_mark_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);
}  // namespace prosper::gpu
