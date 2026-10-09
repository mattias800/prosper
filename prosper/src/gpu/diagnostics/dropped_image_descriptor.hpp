// dropped_image_descriptor.hpp -- name the T# words an image instruction lost, the first time.
//
// The descriptor fold publishes every image use it can prove as a texture keyed by the consuming
// pc. When it declines one, the instruction can only resolve through the metadata table, and when
// that misses too the recompiler prints `[mimg-unresolved] ... pc_res=null` and refuses the stage.
// That line cannot show the descriptor: the recompiler never holds the guest words. So a run that
// hit the refusal kept no record of WHY the words were unusable, and the case could not be decided
// offline (#4700: Kena refused three pixel programs in one run and in none of five others).
//
// This prints one `[t8-dropped]` line per (program, pc, reason): the raw eight words plus the
// fields the gates read. Always on and bounded (kDroppedImageDescriptorMaxReports distinct sites
// per process); after the cap a call costs one relaxed atomic load. A diagnostic: it changes
// nothing the guest sees. A line does NOT mean the draw was lost -- the metadata table may still
// resolve the same instruction by its SRSRC register.
//
// Coverage, so an absence can be read: every image-use decline in resolve_dynamic_fetch_fold's T#
// admission (fold_t8_decline_reason below) and every `continue` in build_stage_table's texture
// publication. NOT covered: the fold's whole-program early return when a checked source is no
// longer current, which declines before any instruction is examined. A `[mimg-unresolved]
// pc_res=null` with no `[t8-dropped]` line for that program and pc points there, or at the cap.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper::gpu {

inline constexpr size_t kDroppedImageDescriptorMaxReports = 64;

// The inputs of the fold's T# admission for one image use. Named fields, because the predicate
// they feed is the one whose every decline this diagnostic must be able to report.
struct FoldT8Admission {
    bool words_known = false;        // all eight SRSRC words are known at the instruction
    bool have_t8 = false;            // they were snapshotted from a table load
    bool mapped_t8 = false;          // they descend from one proven mapped range
    bool branchy_x16 = false;   // an x16 bundle under control flow, needing the mapped proof
    bool seed_provenance = false;   // untouched (or consecutively copied) entry user data
    bool seed_publishable = false;   // such a seed is a plausible T# or a provable null image
};

// Why the fold declines this use, or nullptr when it admits it. This IS the admission predicate:
// resolve_dynamic_fetch_fold publishes exactly when it returns nullptr, so the reasons cannot
// drift from the decision. Reasons: words-unknown, branchy-x16-unproven, no-provenance,
// direct-seed-implausible. For words-unknown the printed words are not authoritative.
const char* fold_t8_decline_reason(const FoldT8Admission& admission);

// The report line, without the trailing newline. Pure, for tests. pc UINT32_MAX prints `none`.
std::string format_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                            const std::array<uint32_t, 8>& words,
                                            const char* reason);

// Print the line once per (program, pc, reason) until kDroppedImageDescriptorMaxReports distinct
// sites have been reported. `reason` must be a string literal (or otherwise outlive the process):
// it is kept by pointer as part of the dedupe key. Returns whether this call printed. Thread-safe.
bool note_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                   const std::array<uint32_t, 8>& words, const char* reason);

// #4775: the counterpart for a use that was NOT dropped. When an unusable T# is read only by an
// instruction the program can skip, build_stage_table binds a null image instead of refusing the
// program, and prints one `[t8-unbound]` line per (program, pc, reason) with the same fields as
// `[t8-dropped]`. Same bound and dedupe discipline, separate cap, so neither line can starve the
// other. It is how a reader tells "this arm's slot was stale and bound null" apart from a resolved
// texture: a run whose picture is missing a sampled term should look here first.
std::string format_unbound_image_descriptor(uint64_t program, uint32_t pc,
                                            const std::array<uint32_t, 8>& words,
                                            const char* reason);
bool note_unbound_image_descriptor(uint64_t program, uint32_t pc,
                                   const std::array<uint32_t, 8>& words, const char* reason);

}   // namespace prosper::gpu
