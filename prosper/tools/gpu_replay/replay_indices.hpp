#pragma once
// Three index spaces over one replayed submit, as distinct types.
//
// WHY THESE EXIST. A replayed submit is indexed three different ways, and until this header they
// were all bare integers, so every one of them converted silently into every other:
//
//   DrawIndex       the semantic draw ordinal the CAPTURE assigned. Counts every draw, including
//                   ones replay did not realize.
//   ItemIndex       an index into GpuReplayFrame::items, which holds ONLY realized draws. An
//                   unrealized draw leaves a HOLE, so this diverges from DrawIndex from the first
//                   unrealized draw onward.
//   OperationIndex  an index into GpuReplayFrame::operations, which interleaves draws and computes.
//
// The divergence is not theoretical. Matching an operation by ITEM index silently resolved to an
// EARLIER draw's operation, so a truncated prefix looked executed and the guard that depended on it
// never fired — reintroducing the void comparison that guard exists to prevent (#2739 seam 2). It
// was invisible in the evidence at the time because that submit realized all 30 of its operations,
// which is exactly the case where the two coincide.
//
// A second instance of the same family cost a whole survey on 2026-08-21: `--output-target-after`
// takes an OPERATION index while `--inspect-only` prints DRAW indices, and they differ by the number
// of interleaved computes — measured at +5 in one submit and +17 in another on one GTA V frame.
//
// Scoped enums rather than a struct wrapper: they are distinct types with no implicit conversion in
// either direction, they cost nothing at runtime, and they deliberately have no arithmetic. Code
// that wants to do sums or comparisons has to say `raw(...)` and is thereby visible in review, which
// is the point — the bug above was an invisible conversion.
#include <cstddef>
#include <cstdint>

namespace prosper::tools {

enum class DrawIndex : uint64_t {};
enum class ItemIndex : size_t {};
enum class OperationIndex : size_t {};

inline constexpr uint64_t raw(DrawIndex value) { return static_cast<uint64_t>(value); }
inline constexpr size_t raw(ItemIndex value) { return static_cast<size_t>(value); }
inline constexpr size_t raw(OperationIndex value) { return static_cast<size_t>(value); }

// "no such index". Named per space so a not-found item cannot be compared against a not-found
// operation, which is the same conversion this header exists to stop.
inline constexpr ItemIndex kNoItemIndex{SIZE_MAX};
inline constexpr OperationIndex kNoOperationIndex{SIZE_MAX};

}  // namespace prosper::tools
