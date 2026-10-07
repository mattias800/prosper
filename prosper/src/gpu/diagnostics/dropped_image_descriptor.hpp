// dropped_image_descriptor.hpp -- name the T# words an image instruction lost, the first time.
//
// The descriptor fold publishes every image use it can prove as a texture keyed by the consuming
// pc. When it declines one -- a direct user-data T# that fails the plausibility gate, or a recovered
// T# the image checks reject -- the instruction can only resolve through the metadata table, and
// when that misses too the recompiler prints `[mimg-unresolved] ... pc_res=null` and refuses the
// stage. That line cannot show the descriptor: the recompiler never holds the guest words. So a
// run that hit the refusal kept no record of WHY the words were unusable, and the case cannot be
// decided offline (#4700: Kena refused three pixel programs in one run and in none of four others).
//
// This prints one `[t8-dropped]` line per (program, pc, reason): the raw eight words plus the
// fields the gates read. Always on and bounded (kMaxReports distinct sites per process); it is a
// diagnostic and changes nothing the guest sees. A line does NOT mean the draw was lost: the
// metadata table may still resolve the same instruction by its SRSRC register.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper::gpu {

inline constexpr size_t kDroppedImageDescriptorMaxReports = 64;

// The report line, without the trailing newline. Pure, for tests.
std::string format_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                            const std::array<uint32_t, 8>& words,
                                            const char* reason);

// Print the line once per (program, pc, reason) until kDroppedImageDescriptorMaxReports distinct
// sites have been reported. Returns whether this call printed. Thread-safe.
bool note_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                   const std::array<uint32_t, 8>& words, const char* reason);

}   // namespace prosper::gpu
