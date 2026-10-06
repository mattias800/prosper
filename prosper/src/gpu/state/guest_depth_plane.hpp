#pragma once
// The guest's own depth plane, as distinct from the host image that stands for it.
//
// prosper renders every guest depth format into a D32 host image, so nothing about the host image
// says how many bytes of guest memory the plane occupies. DB_Z_INFO.FORMAT does. The retained
// depth cache needs it to decide whether a guest write lands in the plane: sizing a Z_16 plane at
// the host's four bytes a texel doubles its range, and the second half is never that surface. It
// is whatever the guest allocated next, or nothing at all -- and a range that runs off its mapping
// cannot be placed by the mapping table, which every write then counts against. MOUSE: P.I. For
// Hire's Z_16 shadow atlases lost their depth to every 4-byte label write that way (#4556).

#include "gpu/pm4/pm4_registers.hpp"
#include "host/memory/guest_memory_topology.hpp"

#include <cstdint>

namespace prosper::gpu {

// DB_Z_INFO as bytes per texel of the guest depth plane: a single-sample Z_16 plane is two, and
// everything else that names a plane is four. Format 0 is Z_INVALID, which names no plane, and
// reports 0 for "unknown".
//
// Four is the old assumption, not a measurement, for the multisampled formats: a plane with more
// than one sample is larger than that, and the range was already short for it. The one thing this
// must not do is make such a range shorter still, so a multisampled Z_16 plane keeps four.
constexpr uint32_t guest_depth_texel_bytes(uint32_t db_z_info) {
    const uint32_t format = (db_z_info >> prosper::agc::Pm4::DB_Z_INFO_FORMAT_SHIFT) &
                            prosper::agc::Pm4::DB_Z_INFO_FORMAT_MASK;
    const uint32_t samples_log2 = (db_z_info >> prosper::agc::Pm4::DB_Z_INFO_NUM_SAMPLES_SHIFT) &
                                  prosper::agc::Pm4::DB_Z_INFO_NUM_SAMPLES_MASK;
    return format == 1u && !samples_log2 ? 2u : format ? 4u : 0u;
}

// Record on a retained entry what a pass attaching it programmed. A pass that names no plane
// changes nothing, so the last pass to describe the plane decides.
template <typename Entry>
constexpr void note_guest_depth_format(Entry& entry, uint32_t db_z_info) {
    if (const uint32_t bytes = guest_depth_texel_bytes(db_z_info))
        entry.guest_depth_texel_bytes = bytes;
}

// Bytes of guest memory that `pixels` texels of the entry's depth plane occupy. An entry no pass
// has described is sized at four bytes a texel: nothing licenses a shorter range.
template <typename Entry>
constexpr uint64_t guest_depth_plane_bytes(const Entry& entry, uint64_t pixels) {
    return pixels * (entry.guest_depth_texel_bytes ? entry.guest_depth_texel_bytes : 4u);
}

// A process that replays a capture sets this, once, before it renders anything.
//
// Whether a guest write can land in a retained plane is a question for the guest mapping table:
// distinct guest addresses can name the same physical bytes, so only a proven-disjoint answer
// keeps the plane. A replay process has no mapping table. Every such query is then Unknown, and
// every write discarded every retained depth image, wherever the write was (#4619). A capture
// carries guest addresses and nothing about what backs them, so there the addresses are all
// there is to compare.
inline bool& guest_plane_alias_by_address() {
    static bool by_address = false;
    return by_address;
}

// Can a guest write of `write_bytes` at `write` land in the plane of `plane_bytes` at `plane`?
// By address, a range of no bytes is answered "it may", as the mapping table answers it. The
// table also says so for a zero address or a range that overflows; by address those are compared
// like any other, which no retained plane can reach (its caller refuses a zero base first).
// What by address cannot see is two guest addresses backed by the same bytes: a replay that
// writes a plane through such an alias keeps the image. That is the trade, and the alternative
// was to keep none.
inline bool guest_write_may_alias_plane(uint64_t write, uint64_t write_bytes, uint64_t plane,
                                        uint64_t plane_bytes) {
    if (!guest_plane_alias_by_address())
        return prosper::guest_memory_topology_relation(write, write_bytes, plane, plane_bytes) !=
               prosper::GuestMemoryTopologyRelation::Disjoint;
    if (!write_bytes || !plane_bytes) return true;
    const uint64_t write_end = write_bytes > UINT64_MAX - write ? UINT64_MAX : write + write_bytes;
    const uint64_t plane_end = plane_bytes > UINT64_MAX - plane ? UINT64_MAX : plane + plane_bytes;
    return write < plane_end && plane < write_end;
}

}  // namespace prosper::gpu
