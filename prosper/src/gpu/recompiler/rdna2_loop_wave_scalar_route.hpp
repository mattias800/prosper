// Which route lowers a compute EXEC loop that a per-invocation lowering cannot reproduce
// (#5201, #5206).
//
// A structured compute loop runs per invocation, so each invocation leaves it on its own trip.
// Hardware runs the wave until its LAST lane leaves: a scalar the body writes ends at the wave's
// final iteration, a mask the body writes holds zero for lanes that left earlier, and SCC is the
// final iteration's. So a body-written wave scalar that is live at the exit has no faithful
// structured lowering. Nor does a body that can re-enable a lane that already left
// (rdna2_loop_exec_widening.hpp): hardware runs that lane again, the left invocation does not.
// The exact dispatcher runs the wave's control flow and holds both.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

class LoopExecSubset;
struct DivLoop;
struct SpirvCompute;

// Called once the structured walk has lowered the loop. True means REFUSE: the loop is a compute
// EXEC loop without the narrowing certificate whose body writes a wave scalar live at its exit
// (loop_live_wave_scalar), or may re-enable a lane that left at an exit test
// (loop_widening_exec_pc). The refusal names the cause at the loop header, and grants the fresh
// exact-dispatcher attempt (SpirvCompute::compute_store_loop_data_retry) when the program has an
// authentic native Wave64 entry; otherwise it stays a refusal. A certified loop, a loop that
// only narrows EXEC and whose written scalars are dead after it, and a VCC/SCC loop (whose exit
// is wave-uniform) keep the structured route.
bool refuse_per_invocation_loop(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                const DivLoop& loop, const LoopExecSubset& proof,
                                const uint32_t* code, size_t dwords);

}   // namespace prosper::gpu
