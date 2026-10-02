// gpu_breadcrumbs.hpp -- where did the GPU stop? (Vulkan-free core)
//
// THE FAILURE THIS ANSWERS. A VK_ERROR_DEVICE_LOST names the submission that OBSERVES the loss, not
// the one that caused it (instrument trap 170): a graphics hang was read for days as "a compute
// dispatch hung the GPU" purely because compute was the only side that reported it. RADV's own `hang`
// dump and the last-recorded-serial log say what was SUBMITTED, never where the command processor
// actually stopped. Breadcrumbs say where: a marker is written to the GPU before and after every
// instrumented draw and dispatch, and after the loss the last marker that reached memory names the
// window the GPU stopped inside.
//
// THIS FILE IS THE VULKAN-FREE HALF. It owns the identity of a site (D8 in docs/DIAGNOSTICS_ROADMAP.md:
// submit ordinal, pass, draw index, program, pipeline key), the ordered table that turns a marker value
// back into that identity, and the arithmetic that turns "the last two markers the GPU wrote" into a
// verdict. Everything the GPU writes is a 32-bit value, so none of this needs a device and all of it is
// unit-testable on a synthetic stream (tests/gpu/diagnostics/test_gpu_breadcrumbs.cpp). The emitter that
// actually writes the markers is gpu_breadcrumbs_vk.hpp.
//
// MARKER ENCODING. Each site gets an id (1, 2, 3, ...; 0 means "none"). The marker written BEFORE the
// site is `id`; the marker written AFTER it is `id | kBreadcrumbAfterBit`. Two slots are enough on
// hardware that overwrites one location (AMD): the last "before" and the last "after" that executed.
// Hardware that reports a set of reached checkpoints (NV) is reduced to the same two values.
//
// WHAT THE VERDICT DOES AND DOES NOT PROVE -- read this before quoting one.
//   * "Started" is when the command processor reached the site; "finished" is when everything before
//     it had drained. Those are different moments, so the verdict is a WINDOW of sites, not one
//     suspect: the sites numbered (last_finished, last_started] were in flight together. A single
//     site is named only when the window holds exactly one. The cause may also be a site the window
//     DEPENDS on (a draw that samples what an earlier draw wrote), or work that carries no marker.
//   * Ids are assigned in RECORDING order. If command buffers are submitted in a different order than
//     they were recorded, a window can span sites that ran far apart. The table cannot see submit order.
//   * It is blind to everything that is not instrumented: transfers, barriers, presents, and any
//     queue other than the one the markers were written on.
//   * "Nothing recorded" means no marker reached memory. That is a measurement about the markers, not
//     about the hang: it is equally what an inactive extension, a hang before the first site and a
//     driver that did not flush its writes look like.
#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

namespace prosper::gpu {

enum class BreadcrumbKind : uint8_t { draw = 1, dispatch = 2 };

// One instrumented site. Every field is a number or a hash -- never a guest string -- so a verdict
// can be quoted into a public tracker. 0 in a field means "the call site did not know".
struct BreadcrumbSite {
    uint32_t id = 0;               // 0 marks an empty slot
    BreadcrumbKind kind = BreadcrumbKind::draw;
    uint64_t submit_no = 0;        // run-local submit ordinal
    // Graphics: the draw's offset within the backend pass it was recorded in. Pass-local, so it differs
    // from draw_index whenever a submit is split into several passes; 0 for dispatches.
    uint32_t pass_local_index = 0;
    uint32_t draw_index = 0;       // draw index (stable semantic index), or dispatch index within its submit
    uint64_t program_addr = 0;     // guest shader program address
    uint64_t pipeline_hash = 0;    // hash of the pipeline key
};

constexpr uint32_t kBreadcrumbAfterBit = 0x80000000u;
constexpr uint32_t kBreadcrumbMaxId = 0x7fffffffu;
// Power of two. A marker naming a site older than the ring is reported as such, never mis-resolved.
constexpr uint32_t kBreadcrumbRingSites = 4096;

constexpr uint32_t breadcrumb_before_value(uint32_t id) { return id; }
constexpr uint32_t breadcrumb_after_value(uint32_t id) { return id | kBreadcrumbAfterBit; }

// Thread-safe, bounded, and cheap enough to sit on the draw-recording path when ARMED. Not armed by
// default: nothing calls begin_site unless the switch is on.
class BreadcrumbTable {
public:
    // Assigns the next id to `site` (its `id` field is ignored), stores it, and returns the id. Ids
    // start at 1 and never carry the after bit; after 2^31 - 1 sites the counter wraps to 1, which a
    // lookup tells apart from the older site only by the ring still holding the right one.
    uint32_t begin_site(BreadcrumbSite site);
    // True and fills `out` when the ring still holds the site with this exact id.
    bool lookup(uint32_t id, BreadcrumbSite& out) const;
    uint64_t sites_recorded() const;
    void reset();

private:
    mutable std::mutex mutex_;
    std::array<BreadcrumbSite, kBreadcrumbRingSites> ring_{};
    uint32_t next_id_ = 1;
    uint64_t recorded_ = 0;
};

struct BreadcrumbVerdict {
    enum class State : uint8_t {
        nothing_recorded,   // no marker reached memory: see "WHAT THE VERDICT DOES AND DOES NOT PROVE"
        between_sites,      // every started site also finished: the stop was outside instrumented work
        in_flight,          // a window of sites started and had not finished
        inconsistent,       // the two markers contradict each other (finished a site never started)
    };
    State state = State::nothing_recorded;
    uint32_t last_started_id = 0;    // 0 = none
    uint32_t last_finished_id = 0;   // 0 = none
    // Number of sites in the in-flight window, last_started - last_finished. 1 means one suspect.
    uint32_t window_sites = 0;
    bool first_unfinished_known = false;   // the table still holds the window's first site
    bool last_started_known = false;       // ... and its last
    bool last_finished_known = false;      // ... and the last site that finished
    BreadcrumbSite first_unfinished;       // site last_finished_id + 1
    BreadcrumbSite last_started;
    BreadcrumbSite last_finished;
};

// `last_started_value` is the last "before" marker that reached memory, `last_finished_value` the last
// "after" marker, exactly as read from the slot (so the after marker carries the after bit); 0 means
// none, so a caller can pass raw buffer contents. A value of the wrong polarity -- a "before" slot that
// carries the after bit, or an "after" slot that lacks it -- is treated as absent rather than trusted.
BreadcrumbVerdict resolve_breadcrumbs(const BreadcrumbTable& table, uint32_t last_started_value,
                                      uint32_t last_finished_value);

// Reduce a SET of reached checkpoint values (what an API that lists reached checkpoints returns) to the
// two values above: the greatest before id and the greatest after id.
void reduce_checkpoint_values(const uint32_t* values, size_t count, uint32_t& last_started_value,
                              uint32_t& last_finished_value);

// One report, `[gpu-breadcrumb] ...`, naming the window, its sites and the markers read. Numbers only.
std::string format_breadcrumb_verdict(const BreadcrumbVerdict& verdict, uint32_t raw_started_value,
                                      uint32_t raw_finished_value);

// The process-wide table the shipped renderer and the live compute backend both write into.
BreadcrumbTable& breadcrumb_table();

} // namespace prosper::gpu
