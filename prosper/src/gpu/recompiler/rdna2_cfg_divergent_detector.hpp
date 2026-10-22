// Divergent-loop detection; private to the recompiler.
#pragma once
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
// Returns the loops in header-pc order, or {} when any backward branch doesn't fit the shape.
// `safe` carries the linearized branches (waterfalls etc.) which are not loop back-edges.
//
// An empty result has TWO causes that the caller cannot tell apart, and neither can the log: no
// eligible backward branch was found at all (the `out.empty()` early return below, which reaches
// none of the reject sites), or a shape check refused. Callers conflate them deliberately --
// `(cf_rejected || Ls.empty())` -- so this distinction matters only to a reader of the diagnostic,
// which is why the reject reporter below prints on exactly one of those two paths.
std::vector<DivLoop> detect_divergent_loops(const std::vector<Rdna2Inst>& ins,
                                            const std::unordered_set<uint32_t>& safe,
                                            bool exact_fragment_wave_breaks = false,
                                            const RecompileDiagnosticContext& diagnostic = {},
                                            const char* caller_role = "emit");
}  // namespace prosper::gpu
