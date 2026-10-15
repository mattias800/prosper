// Whether a loop's body can re-enable a lane that already left it (#5206).
//
// A structured compute EXEC loop lets each invocation leave at the first exit test where its own
// EXEC bit is clear. Hardware keeps that lane in the wave, inactive, until the whole wave leaves.
// The two agree only while no EXEC write in the loop can set the lane's bit again. This is a
// code-only proof over the loop's instructions; the emitter state plays no part in it.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <vector>

namespace prosper::gpu {

struct DivLoop;

// The pc of the first EXEC write in [header, back edge] whose result is not provably within the
// EXEC of the most recent exit test (an execz/execnz leaving the loop), or -1 when every write is.
// "Within" is tracked per scalar word: EXEC after a test, a compare or a save of EXEC, and AND/OR/
// XOR/ANDN2/CSELECT combinations of such words stay within; anything else (a constant other than
// 0, a loaded or computed scalar, a complement, a mask saved before the test) does not. A forward
// dataflow meets paths by intersection; the outer back edge restarts from EXEC alone. Unmodelled
// control flow inside the loop counts as widening.
int loop_widening_exec_pc(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);

}   // namespace prosper::gpu
