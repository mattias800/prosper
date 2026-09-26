#pragma once
// WHY prosper copies a colour target back to the CPU, counted per reason.
//
// Readback is the dominant cost inside a backend call -- measured 2026-09-26 on The Messenger,
// 68.5% of `measured` on prosper-app and 78.5% under tools/screenshot -- and the synchronization
// line shows one of every two flushes is forced BY it. That makes it the reason the submit blocks,
// so it is upstream of the "adopt a timeline semaphore" idea rather than parallel to it: a wait
// whose purpose is to deliver bytes the next draw reads cannot be removed by changing how the wait
// is expressed.
//
// `readback_policy.hpp` already distinguishes three ways a slot arrives at "yes", and they have
// nothing in common and nothing to do with each other:
//
//   no-color-target        the caller passed no target object -- a caller-shape question
//   explicit-request       something downstream asked for bytes -- a consumer question
//   bound-non-persistent   a bound target whose pixels are not retained -- a RESIDENCY question
//
// Only the third is addressable by making more targets persistent, which is the obvious first
// move; whether it is the right one depends entirely on which of the three dominates, and the
// aggregate millisecond figure cannot say. This counts them.
//
// NOTE FOR A FUTURE READER: this is the fourth census in this tree with the same shape (atomic
// per-reason counters plus a register_exit_report line) -- see draw_disposition, pass_break_census
// and worker_spawn_census. A shared helper is warranted and deliberately not done here, because
// retrofitting three already-tested instruments is a separable mechanical change and does not
// belong in the same commit as a new measurement.
//
// One line at end of run via register_exit_report (NOT std::atexit -- #3353).
// `PROSPER_NO_READBACK_REASON_CENSUS=1` silences it.

#include <cstdint>

namespace prosper::diagnostics {

// Mirrors prosper::frontend::ColorReadbackReason, kept as a plain count index so this header does
// not pull the frontend policy into every translation unit that reports.
// Order MATCHES prosper::frontend::ColorReadbackReason value-for-value, and the static_asserts in
// the implementation enforce it. The first version of this enum listed the interesting reasons
// first and NotWanted last, which reads better and was wrong: the call site casts one enum to the
// other, so every reason would have been recorded as its neighbour. The asserts caught it before
// a single number was published.
enum class ReadbackReasonSlot : unsigned char {
    NotWanted = 0, NoColorTarget, ExplicitRequest, BoundNonPersistent, Count
};

// `bytes` is the slot's readback extent when known, 0 otherwise. Counting bytes as well as
// occurrences matters: a rare reason covering a 4K target costs more than a frequent one covering
// a 64x64 scratch surface, and an occurrence count alone would rank them backwards.
void note_readback_reason(ReadbackReasonSlot slot, uint64_t bytes);

// The call site casts prosper::frontend::ColorReadbackReason straight to ReadbackReasonSlot, which
// is only sound while the two enumerations agree value-for-value. They are declared in different
// layers on purpose -- the policy is frontend, the census is diagnostics, and neither should pull
// the other in -- so the agreement is pinned by a static_assert in the census implementation
// rather than by a comment. Reorder either enum and the build stops.

}  // namespace prosper::diagnostics
