// Counted-loop EXEC guards: which wave-empty `s_cbranch_execz` guards around or inside a counted loop
// the per-invocation lowering may linearize.
#pragma once

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// Adds the pc of every proven guard branch to `safe` (emit_alu then treats it as a no-op under the
// narrowed EXEC it guards). Returns true when a proven guard is entered before the loop header and
// restores only after the loop exit, so the loop itself runs under that guard's narrowed EXEC.
bool mark_counted_loop_exec_guards(const std::vector<Rdna2Inst>& ins, const CountedLoop& L,
                                   std::unordered_set<uint32_t>& safe);

}  // namespace prosper::gpu
