#pragma once

#include "gpu/recompiler/rdna2_loaded_scalar_global.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/fold_reader.hpp"
#include <optional>
#include <span>

namespace prosper::gpu {

// The caller supplies the source-associated hardware user count and physical word presence.
// These facts authorize only the requested user pointer words, never system or wave inputs.
struct LoadedGlobalUserPrefix {
    std::optional<uint32_t> declared_count;
    uint32_t sgpr_base = 0;
    std::span<const uint32_t> words;
    uint32_t physical_word_mask = 0;
};

// Code-only obligations for the canonical owned reader; no guest addresses or byte authority.
std::vector<RawNestedWideChain> loaded_scalar_global_chains(
    const std::vector<LoadedScalarGlobalRead>&);

// Before folding, use the same proof-owned reader to snapshot parent and bounded target.
// The canonical reader owns the current producer/physical-alias checks and publication.
// Ordinary readers, absent pointer words, and incomplete observations refuse before folding.
bool prepare_loaded_scalar_globals(FoldReader&, const std::vector<LoadedScalarGlobalRead>&,
                                   const LoadedGlobalUserPrefix&);

} // namespace prosper::gpu
