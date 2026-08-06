#pragma once

#include <cstdint>
#include <optional>

namespace prosper::gpu {

enum class GuestWriterKind : uint8_t {
    ColorTarget,
    ComputeBuffer,
    DmaData,
    WriteData,
    // Keep last. The per-kind recorded counts below are sized from this, so a new kind added above
    // is counted and named automatically. A kind that is armed and recording but missing from the
    // summary would be invisible in exactly the line that exists to say which recorders were armed.
    Count,
};

struct GuestWriteEvent {
    uint64_t sequence = 0;
    GuestWriterKind kind = GuestWriterKind::ColorTarget;
    uint64_t addr = 0;
    uint64_t size = 0;
    uint64_t submit = 0;
    uint64_t item = 0;
    uint64_t order = 0;
    uint64_t identity = 0;
    uint32_t width = 0, height = 0;
};

// The opt-in live history is enabled by either the existing dimension probe or its explicit alias.
bool writer_provenance_enabled();
// Explicit provenance requests retain even tiny label/pointer writes. The dimension probes keep
// their historical large-resource filter so their diagnostic overhead does not unexpectedly grow.
bool writer_provenance_full_enabled();
const char* guest_writer_kind_name(GuestWriterKind kind);

uint64_t record_guest_write(GuestWriterKind kind, uint64_t addr, uint64_t size,
                            uint64_t submit = 0, uint64_t item = 0, uint64_t order = 0,
                            uint64_t identity = 0,
                            uint32_t width = 0, uint32_t height = 0);
std::optional<GuestWriteEvent> last_guest_write_overlap(uint64_t addr, uint64_t size,
                                                        uint64_t before_sequence = UINT64_MAX);

// How many write events the history currently holds, and how many of each kind have ever been
// recorded. A caller that reports "no writer overlaps X" MUST report these too: the recorders are
// per-kind and independently gated, so an empty (or kind-incomplete) history makes a negative a
// statement about the instrument rather than about the guest. Without this the two are
// indistinguishable in the log — which is exactly how a void result gets read as a measurement.
size_t guest_write_history_size();
// Cumulative per-kind recorded counts, indexed by GuestWriterKind. Never evicted, unlike the
// bounded event map, so a kind that recorded and then aged out still reports non-zero.
uint64_t guest_write_recorded_count(GuestWriterKind kind);
// Compact "which recorders actually fired" summary for diagnostics, e.g.
// "color=0 compute-buffer=12 dma-data=3 write-data=0". The returned pointer is to a thread-local
// buffer valid until the next call on the same thread.
//
// SCOPE — what this history does NOT contain, so a negative is read correctly. These are not
// filters on the four kinds; they are guest writes that reach memory under no kind at all, and a
// summary showing four healthy counters says nothing about them:
//   * guest CPU writes — nothing hooks them, so any value the guest stores directly is invisible;
//   * RELEASE_MEM / EOP label writes and EVENT_WRITE timestamps (command_processor.cpp) — GPU-side,
//     so the CPU caveat above does not cover them, and they are 4-8 byte writes to exactly the kind
//     of address a PROSPER_PROVENANCE_ADDR watch is pointed at;
//   * colour targets beyond MRT1 — RenderState carries only color0_base/color1_base, so a third or
//     later target has no base to record;
//   * colour targets after the first draw to a given base — deduped deliberately, so the history
//     holds one representative event per range rather than a write log;
//   * a SKIPPED compute dispatch — it executes nothing, so it records nothing however the switches
//     are set;
//   * DMA destinations written by the capture/replay execute_ordered_items overload, which does not
//     record while the live path does.
// Do not try to make any one diagnostic line enumerate this; state scope and cite this list. #2111.
const char* guest_write_recorder_summary();

// Test-only reset for the process-global bounded history.
void reset_guest_write_history_for_test();

} // namespace prosper::gpu
