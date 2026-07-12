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

    if (fails) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
