// Semantic vector inputs/old-VDST dependencies, separate from conservative storage inventories.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>

namespace prosper::gpu {
uint32_t vector_provenance_source_span(const Rdna2Inst& in, uint32_t source);
bool vector_exec_is_full_after_write(const Rdna2Inst& in, bool full_before,
                                     uint32_t wave_size = 64);
bool vector_write_leaves_old_word(const Rdna2Inst& in, bool native_interpolation);
bool vector_write_depends_on_old_word(const Rdna2Inst& in, bool full_exec);
} // namespace prosper::gpu
