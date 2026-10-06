// split_t8_proof.hpp -- may the scalar const-fold publish an image descriptor assembled from two
// scalar loads? The definition explains the proof.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Guest address range a storage-image instruction at `pc` can write, as a CONSERVATIVE superset
// derived from the image's own T#. See storage_image_write_extent.
struct ImageWriteExtent {
    uint32_t pc = 0;
    uint64_t lo = 0, hi = 0;
};

// Upper bound of the guest memory an image_store / image atomic through `t8` can touch: the base,
// plus the surface padded to 256x256 tiles, all slices and samples, doubled when it has a mip chain.
// Returns false when no sound bound is known (unknown format, compressed surface with metadata, a 3D
// surface whose thick tiling pads depth, a 64-bit overflow), and the caller then keeps treating the
// write as able to alias anything.
// CONFIDENCE: MED. The padding and the mip factor are bounds, not the layout; the layout is not needed
// to show two ranges are disjoint, only that this one does not underestimate the footprint. A linear
// surface is covered because GFX10 aligns its row pitch to 256 bytes, so a row never spans more than
// pad256(width) texels.
bool storage_image_write_extent(const std::array<uint32_t, 8>& t8, uint64_t& lo, uint64_t& hi);

// `image_writes` lists the earlier storage-image uses whose extents are known. A write through one of
// them that cannot reach the descriptor bytes does not revoke the proof; any other writer still does.
bool mapped_split_t8_reaches_use(const uint32_t* code, size_t dwords, uint32_t use_pc, int tbase,
                                 const std::array<uint32_t, 8>& source_pc,
                                 const std::array<uint64_t, 8>& source_addr,
                                 const uint32_t* user_sgprs, uint32_t nsgpr,
                                 uint32_t user_sgpr_base,
                                 const std::vector<ImageWriteExtent>& image_writes = {});

} // namespace prosper::gpu
