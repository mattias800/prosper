#include "gpu/capture/writer_provenance.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"

#include <cstdio>
#include <string>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } } while (0)

int main() {
    reset_guest_write_history_for_test();
    const uint64_t first = record_guest_write(
        GuestWriterKind::DmaData, 0x1000, 0x100, 4, 7, 40, 0xaaa);
    const uint64_t second = record_guest_write(
        GuestWriterKind::ComputeBuffer, 0x1080, 0x80, 5, 2, 50, 0xbbb);
    record_guest_write(GuestWriterKind::WriteData, UINT64_MAX - 7, 16, 6, 1, 60, 0xccc);

    auto latest = last_guest_write_overlap(0x10f0, 0x40);
    CHECK(latest && latest->kind == GuestWriterKind::ComputeBuffer && latest->sequence == second,
          "latest overlapping writer wins");
    CHECK(latest && latest->submit == 5 && latest->item == 2 && latest->order == 50 &&
          latest->identity == 0xbbb,
          "writer metadata survives the range history");
    auto earlier = last_guest_write_overlap(0x10f0, 0x40, second);
    CHECK(earlier && earlier->kind == GuestWriterKind::DmaData && earlier->sequence == first,
          "before-sequence query returns the preceding writer");
    CHECK(!last_guest_write_overlap(0x2000, 0x20), "disjoint ranges do not match");
    auto wrapped = last_guest_write_overlap(UINT64_MAX - 3, 4);
    CHECK(wrapped && wrapped->kind == GuestWriterKind::WriteData,
          "overflowing address spans clamp and still overlap safely");
    CHECK(std::string(guest_writer_kind_name(GuestWriterKind::ColorTarget)) == "color",
          "writer kind has a stable diagnostic name");

    // A negative from last_guest_write_overlap() means "nothing recorded overlaps this", which is a
    // statement about the guest ONLY if the relevant recorder was actually armed. The recorders are
    // per-kind and independently gated, so a caller reporting a negative has to be able to say which
    // kinds ever fired — otherwise "no writer wrote it" and "nobody was watching" are the same log
    // line. These accessors are what make that distinction expressible.
    CHECK(guest_write_history_size() == 3, "history size counts the retained events");
    CHECK(guest_write_recorded_count(GuestWriterKind::DmaData) == 1 &&
          guest_write_recorded_count(GuestWriterKind::ComputeBuffer) == 1 &&
          guest_write_recorded_count(GuestWriterKind::WriteData) == 1,
          "per-kind counts follow the recorders that fired");
    CHECK(guest_write_recorded_count(GuestWriterKind::ColorTarget) == 0,
          "a recorder that never fired reports zero, which is what makes a negative readable");
    CHECK(std::string(guest_write_recorder_summary()) ==
              "color=0 compute-buffer=1 dma-data=1 write-data=1",
          "recorder summary names every kind and its count");

    // Re-recording an existing (kind, addr) key REPLACES the event rather than adding one — this is
    // key replacement, not eviction, and the two are tested separately below because only the second
    // one proves the counters outlive the bounded map.
    record_guest_write(GuestWriterKind::DmaData, 0x1000, 0x100, 9, 9, 90, 0xddd);
    CHECK(guest_write_history_size() == 3,
          "re-recording an existing (kind, addr) key replaces rather than grows the history");
    CHECK(guest_write_recorded_count(GuestWriterKind::DmaData) == 2,
          "the cumulative count still advances when an event is replaced");

    // Rejected writes must not be counted: a zero address or size records nothing, so counting it
    // would report a recorder as armed on a run where it contributed no queryable event.
    record_guest_write(GuestWriterKind::ColorTarget, 0, 0x40);
    record_guest_write(GuestWriterKind::ColorTarget, 0x9000, 0);
    CHECK(guest_write_recorded_count(GuestWriterKind::ColorTarget) == 0,
          "a rejected zero address/size does not count as the recorder firing");

    // EVICTION, which is the property that actually matters: the event map is bounded, so a write
    // recorded early in a long boot is gone by the time anything queries it. The cumulative counts
    // must survive that, or "did this recorder ever fire" degrades to "did it fire recently" — and a
    // negative would then be unreadable again for exactly the runs that are long enough to matter.
    reset_guest_write_history_for_test();
    const uint64_t before_flood = guest_write_recorded_count(GuestWriterKind::ColorTarget);
    record_guest_write(GuestWriterKind::ColorTarget, 0x50000, 0x10, 1, 1, 1, 0xeee);
    // Overfill with a different kind so the colour event is the eviction candidate by sequence.
    for (uint64_t i = 0; i < 70000; ++i)
        record_guest_write(GuestWriterKind::DmaData, 0x100000 + i * 0x10, 0x10);
    CHECK(guest_write_history_size() <= 65536,
          "the event map stays inside its bound while being overfilled");
    CHECK(!last_guest_write_overlap(0x50000, 0x10),
          "the early event really was evicted (otherwise the next check proves nothing)");
    CHECK(guest_write_recorded_count(GuestWriterKind::ColorTarget) == before_flood + 1,
          "a cumulative count survives eviction of the event it counted");

    reset_guest_write_history_for_test();
    CHECK(guest_write_history_size() == 0 &&
          guest_write_recorded_count(GuestWriterKind::DmaData) == 0,
          "test reset clears the events and the cumulative counts together");

    // Kena's recognizable forest was retained in a third 3200x1800 color target, but the
    // provenance probe previously recorded only MRT0/1 and falsely answered "no prior programmed
    // color target" for later pixel-stage resource-table candidates. Drive the actual GpuState
    // diagnostic with programmed MRT2 and MRT7; a direct call to record_guest_write would not
    // cover that omission.
    GpuState mrt_state;
    constexpr uint64_t mrt_bases[] = {
        0x180000000ull, 0x190000000ull, 0x1a0000000ull, 0, 0, 0, 0, 0x1f0000000ull};
    for (uint32_t slot = 0; slot < 8; ++slot) {
        if (!mrt_bases[slot]) continue;
        mrt_state.cx[P::CB_COLOR0_BASE + slot * 0xfu] =
            static_cast<uint32_t>(mrt_bases[slot] >> 8);
        mrt_state.cx[P::CB_COLOR0_BASE_EXT + slot] =
            static_cast<uint32_t>(mrt_bases[slot] >> 40);
        mrt_state.cx[P::CB_COLOR0_INFO + slot * 0xfu] = 0xau;
        mrt_state.cx[P::CB_COLOR0_ATTRIB2 + slot] =
            ((3200u - 1u) << P::CB_COLOR0_ATTRIB2_MIP0_WIDTH_SHIFT) |
            ((1800u - 1u) << P::CB_COLOR0_ATTRIB2_MIP0_HEIGHT_SHIFT);
    }
    GpuState::Draw mrt_draw;
    mrt_draw.command_order = 77;
    mrt_state.draws.push_back(mrt_draw);
    diagnose_resource_provenance(mrt_state, 8080);
    for (uint32_t slot : {0u, 1u, 2u, 7u}) {
        const auto writer = last_guest_write_overlap(mrt_bases[slot], 4);
        CHECK(writer && writer->kind == GuestWriterKind::ColorTarget &&
              writer->addr == mrt_bases[slot] && writer->submit == 8080 &&
              writer->item == 0 && writer->order == 77 &&
              writer->width == 3200 && writer->height == 1800,
              "the GpuState provenance probe records every programmed MRT slot");
    }
    CHECK(!last_guest_write_overlap(0x1b0000000ull, 4),
          "an unprogrammed MRT slot creates no provenance event");

    if (fails) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
