// split_t8_proof.hpp -- may the scalar const-fold publish an image descriptor assembled from two
// scalar loads? The definition explains the proof.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

bool mapped_split_t8_reaches_use(const uint32_t* code, size_t dwords, uint32_t use_pc,
                                 int tbase, const std::array<uint32_t, 8>& source_pc,
                                 const std::array<uint64_t, 8>& source_addr,
                                 const uint32_t* user_sgprs, uint32_t nsgpr,
                                 uint32_t user_sgpr_base);

} // namespace prosper::gpu
