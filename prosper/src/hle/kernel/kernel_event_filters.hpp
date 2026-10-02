// Event filters prosper posts onto guest equeues that more than one translation unit must agree on.
// The remaining filters (VIDEO_OUT, GRAPHICS_CORE, USER, TIMER, HRTIMER) are still local to
// hle_kernel_time.cpp; move one here when a second file needs it.
#pragma once
#include <cstdint>

namespace prosper {

// SCE_KERNEL_EVFILT_AMPR. The guest DOES read it on some titles: Assassin's Creed Black Flag
// Resynced's IOCompleteEventQueue consumer (eboot+0xacb370) dispatches on `filter == 0xffe7` (-25)
// and treats every other APR event as not its own, so a completion posted with any other filter was
// dropped and the title's boot stalled with every thread idle. The value was -24 while the guest was
// assumed never to read it; -25 also matches a secondary implementation (verification only).
constexpr int16_t EVFILT_AMPR = -25;

}  // namespace prosper
