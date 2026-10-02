// gpu_breadcrumbs.cpp -- see gpu_breadcrumbs.hpp.
#include "gpu/diagnostics/gpu_breadcrumbs.hpp"

#include <algorithm>
#include <cstdio>

namespace prosper::gpu {

uint32_t BreadcrumbTable::begin_site(BreadcrumbSite site) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t id = next_id_;
    next_id_ = (next_id_ >= kBreadcrumbMaxId) ? 1 : next_id_ + 1;
    site.id = id;
    ring_[id % kBreadcrumbRingSites] = site;
    ++recorded_;
    return id;
}

bool BreadcrumbTable::lookup(uint32_t id, BreadcrumbSite& out) const {
    if (id == 0 || id > kBreadcrumbMaxId) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const BreadcrumbSite& slot = ring_[id % kBreadcrumbRingSites];
    // The slot may hold a NEWER site that reused the index; only an exact id match is this site.
    if (slot.id != id) return false;
    out = slot;
    return true;
}

uint64_t BreadcrumbTable::sites_recorded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recorded_;
}

void BreadcrumbTable::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_ = {};
    next_id_ = 1;
    recorded_ = 0;
}

BreadcrumbVerdict resolve_breadcrumbs(const BreadcrumbTable& table, uint32_t last_started_value,
                                      uint32_t last_finished_value) {
    BreadcrumbVerdict v;
    // Polarity: a "before" marker never carries the after bit and an "after" marker always does. A
    // value of the wrong polarity is treated as absent rather than trusted, because reading a marker
    // slot that was never written must not be able to invent a site.
    const uint32_t started = (last_started_value & kBreadcrumbAfterBit) ? 0 : last_started_value;
    const uint32_t finished = (last_finished_value & kBreadcrumbAfterBit)
                                  ? (last_finished_value & ~kBreadcrumbAfterBit) : 0;
    v.last_started_id = started;
    v.last_finished_id = finished;
    if (started == 0 && finished == 0) {
        v.state = BreadcrumbVerdict::State::nothing_recorded;
        return v;
    }
    // A site cannot finish before it started, so this is a corrupt read or a marker the table did not
    // issue. It is reported as such: guessing which of the two to believe would invent a suspect.
    if (finished > started) {
        v.state = BreadcrumbVerdict::State::inconsistent;
        return v;
    }
    v.last_started_known = table.lookup(started, v.last_started);
    if (finished != 0) v.last_finished_known = table.lookup(finished, v.last_finished);
    if (finished == started) {
        v.state = BreadcrumbVerdict::State::between_sites;
        return v;
    }
    v.state = BreadcrumbVerdict::State::in_flight;
    v.window_sites = started - finished;
    v.first_unfinished_known = table.lookup(finished + 1, v.first_unfinished);
    return v;
}

void reduce_checkpoint_values(const uint32_t* values, size_t count, uint32_t& last_started_value,
                              uint32_t& last_finished_value) {
    last_started_value = 0;
    last_finished_value = 0;
    if (!values) return;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = values[i];
        if (value == 0) continue;
        if (value & kBreadcrumbAfterBit) last_finished_value = std::max(last_finished_value, value);
        else                             last_started_value = std::max(last_started_value, value);
    }
}

namespace {

const char* kind_name(BreadcrumbKind kind) {
    return kind == BreadcrumbKind::dispatch ? "dispatch" : "draw";
}

void append_site(std::string& out, const BreadcrumbSite& s) {
    char text[160];
    std::snprintf(text, sizeof text, "%s submit=%llu local=%u index=%u program=0x%llx pipeline=0x%llx",
                  kind_name(s.kind), static_cast<unsigned long long>(s.submit_no), s.pass_local_index,
                  s.draw_index, static_cast<unsigned long long>(s.program_addr),
                  static_cast<unsigned long long>(s.pipeline_hash));
    out += text;
}

} // namespace

std::string format_breadcrumb_verdict(const BreadcrumbVerdict& v, uint32_t raw_started_value,
                                      uint32_t raw_finished_value) {
    char head[128];
    std::snprintf(head, sizeof head, "[gpu-breadcrumb] markers read: started=0x%x finished=0x%x: ",
                  raw_started_value, raw_finished_value);
    std::string out = head;
    switch (v.state) {
    case BreadcrumbVerdict::State::nothing_recorded:
        out += "NO marker reached memory. This says nothing about the hang: an inactive extension, a "
               "hang before the first instrumented site and an unflushed write all look like this";
        return out;
    case BreadcrumbVerdict::State::inconsistent:
        out += "the markers contradict each other (a site finished that never started), so no site is "
               "named; treat the read as corrupt";
        return out;
    case BreadcrumbVerdict::State::between_sites:
        out += "every started site also finished, so the GPU stopped outside instrumented work "
               "(a transfer, a barrier, a present, or another queue). Last finished: ";
        if (v.last_finished_known) append_site(out, v.last_finished);
        else out += "(no longer in the ring)";
        return out;
    case BreadcrumbVerdict::State::in_flight:
        break;
    }
    char window[96];
    if (v.window_sites == 1)
        std::snprintf(window, sizeof window, "the GPU stopped inside ONE site: ");
    else
        std::snprintf(window, sizeof window,
                      "the GPU stopped inside a WINDOW of %u sites, started together. First unfinished: ",
                      v.window_sites);
    out += window;
    if (v.first_unfinished_known) append_site(out, v.first_unfinished);
    else out += "(no longer in the ring)";
    if (v.window_sites > 1) {
        out += "; last started: ";
        if (v.last_started_known) append_site(out, v.last_started);
        else out += "(no longer in the ring)";
    }
    out += "; last finished: ";
    if (v.last_finished_known) append_site(out, v.last_finished);
    else out += v.last_finished_id == 0 ? "none" : "(no longer in the ring)";
    out += ". A window, not a proven cause: see gpu_breadcrumbs.hpp";
    return out;
}

BreadcrumbTable& breadcrumb_table() {
    static BreadcrumbTable table;
    return table;
}

} // namespace prosper::gpu
