#pragma once
// WHY the backend requests a colour-target readback, counted per reason.
//
// The earlier claim that this dominates shipped rendering was withdrawn: the two harnesses used
// for that comparison both lacked the app's active GPU-present path. See the correction in
// docs/RENDERER_ARCHITECTURE_GAPS_2026_09_25.md. The reason census still distinguishes:
//
//   no-color-target        the caller passed no target object -- a caller-shape question
//   explicit-request       something downstream asked for bytes -- a consumer question
//   bound-non-persistent   a bound target whose pixels are not retained -- a residency question
//
// The backend also has an independent caller gate. A depth-only pass may have no colour target
// yet request no colour result. The census records the effective request after that gate, and
// reports requested extent rather than claiming the copy completed.
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

// `bytes` is the slot's requested readback extent when known, 0 otherwise. This is counted before
// submission, so it is not a completed-transfer measurement.
void note_readback_reason(ReadbackReasonSlot slot, uint64_t bytes);

// The call site casts prosper::frontend::ColorReadbackReason straight to ReadbackReasonSlot, which
// is only sound while the two enumerations agree value-for-value. They are declared in different
// layers on purpose -- the policy is frontend, the census is diagnostics, and neither should pull
// the other in -- so the agreement is pinned by a static_assert in the census implementation
// rather than by a comment. Reorder either enum and the build stops.

}  // namespace prosper::diagnostics
