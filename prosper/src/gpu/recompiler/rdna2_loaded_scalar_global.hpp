#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Immutable original-code facts only. These do not authorize reading guest memory or
// initialize launch registers. Realization must own the parent and bounded target together.
struct LoadedScalarGlobalRead {
    uint32_t parent_pc = 0, global_pc = 0;
    uint32_t entry_pointer_sgpr = 0, loaded_pointer_sgpr = 0, address_vgpr = 0;
    uint32_t parent_offset = 0, window_offset = 0, window_bytes = 0;
    int32_t global_offset = 0;
    bool operator==(const LoadedScalarGlobalRead&) const = default;
};

// Initial bounded graphics contract: immediate x2 entry-pointer load, then DWORDX4 GLOBAL
// through its complete scalar pair and a finite, whole-dword VALU address. Both loads need
// immediate full WAIT completion; intervening hints are allowed. Every instruction
// in the original contiguous terminated body participates in control/writer exclusion.
std::vector<LoadedScalarGlobalRead> rdna2_loaded_scalar_global_reads(
    const std::vector<Rdna2Inst>& original);

} // namespace prosper::gpu
