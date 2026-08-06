#include "../src/gpu/writer_provenance.hpp"

#include <cstdio>
#include <string>

using namespace prosper::gpu;

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

    if (fails) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
