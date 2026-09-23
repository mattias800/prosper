#include "shared/diagnostics/compute_present_probe.hpp"

#include <cstdio>
#include <atomic>
#include <thread>
#include <vector>

int main() {
    using namespace prosper::frontend;
    int failures = 0;
    const auto check = [&](bool ok, const char* why) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); ++failures; }
    };
    check(!parse_compute_present_probe(nullptr).requested, "unset stays off");
    const auto spec = parse_compute_present_probe("0x300b220000:ms:30000:16");
    check(spec.armed && spec.code == 0x300b220000 && spec.after_ms == 30000 &&
          spec.stride == 16, "exact bounded selector parses");
    check(compute_probe_modes_compatible(false, false) &&
          !compute_probe_modes_compatible(true, false) &&
          !compute_probe_modes_compatible(false, true),
          "event probe refuses a competing one-shot trace or broad readback");
    check(compute_probe_targets_valid({1, 2, 3, 4}) &&
          !compute_probe_targets_valid({1, 2, 2, 4}) &&
          !compute_probe_targets_valid({1, 2, 3}),
          "four stage targets must be distinct and ordered by caller");
    for (const char* invalid : {"", "300b220000:ms:30000:16", "0x0:ms:30000:16",
                                "0x1:ms:-1:16", "0x1:ms:120001:16", "0x1:ms:0:0",
                                "0x1:ms:0:65", "0x1:ms:0:16:extra",
                                "0xfffffffffffffffff:ms:0:1"})
        check(!parse_compute_present_probe(invalid).armed, "malformed selector refuses");

    ComputePresentProbe probe;
    probe.configure(spec);
    ComputePresentEvent event;
    event.code = spec.code;
    event.submit = 17;
    event.address = 0x1000;
    event.bytes = 0x100;
    event.nonzero_channels = 9;
    event.guest_nonzero_bytes = 7;
    event.published_ms = 5000;
    check(!probe.accepts(spec.code) && !probe.publish(event), "compute cannot arm phase itself");
    check(!probe.should_scan(spec.code), "pre-phase guest census is refused");
    check(!probe.arm_phase(29999) && !probe.phase_armed(), "renderer threshold not yet crossed");
    check(probe.arm_phase(30000) && probe.phase_armed(), "renderer threshold arms once");
    check(probe.publishable(), "first producer slot becomes available after renderer arms");
    check(probe.should_scan(spec.code) && !probe.should_scan(0x42),
          "guest census admits only exact code while slot is idle");
    check(!probe.arm_phase(30001), "phase does not rearm");
    check(!probe.expire_no_producer(39999), "no-producer window remains open through nine seconds");
    check(!probe.accepts(0x42), "wrong producer program refused");
    const auto first = probe.publish(event);
    check(first && first->id == 1 && first->submit == 17, "first completed producer latches");
    check(!probe.publishable() && !probe.should_scan(spec.code),
          "latched event stops later dispatch guest scans");
    check(!probe.expire_no_producer(40001), "latched event is not a no-producer miss");
    check(first && compute_probe_binding_matches(*first, 0x55, 0x55, 0x1000) &&
          !compute_probe_binding_matches(*first, 0x56, 0x55, 0x1000) &&
          !compute_probe_binding_matches(*first, 0x55, 0x55, 0x1001),
          "only actual reflected PS/address binding admits the producer event");
    event.submit = 18;
    check(!probe.publish(event) && probe.ignored() == 1, "later producer ignored while pending");
    check(probe.pending() && probe.pending()->submit == 17, "pending is an immutable copy");
    check(!probe.expire(7000, 256), "age and callback bounds include endpoints");
    check(!probe.invalidate_write(0x2000, 8), "disjoint GPU write does not invalidate");
    check(probe.invalidate_write(0x10ff, 1).has_value(), "overlapping notified GPU write invalidates");
    check(!probe.pending() && !probe.finish(1), "invalidated event cannot be selected");
    check(!probe_ranges_overlap(UINT64_MAX - 1, 4, 0x1000, 4),
          "overflowing range refuses closed");
    check(!probe_ranges_overlap(0x1000, 0x100, 0x1100, 1), "touching endpoints do not overlap");
    ComputePresentProbe overflow_write;
    overflow_write.configure(spec);
    overflow_write.arm_phase(30000);
    overflow_write.publish(event);
    check(overflow_write.invalidate_write(UINT64_MAX - 1, 4).has_value(),
          "invalid notified write range invalidates rather than wrapping past the event");
    check(compute_probe_sample_due(1, 16, 0) &&
          !compute_probe_sample_due(2, 16, 1) &&
          compute_probe_sample_due(17, 16, 1) &&
          !compute_probe_sample_due(49, 16, 4),
          "four scene probes respect matching-callback stride");
    uint64_t raw_budget = 0;
    check(compute_probe_charge(raw_budget, 23'040'000, 160ull << 20) &&
          !compute_probe_charge(raw_budget, UINT64_MAX, 160ull << 20) &&
          raw_budget == 23'040'000,
          "aggregate byte cap fails closed without wrap or extra charge");
    check(compute_probe_bmp_bytes(3, 2) == 78 &&
          !compute_probe_bmp_bytes(UINT32_MAX, 2),
          "BMP artifact cap charges row padding and refuses header overflow");

    ComputePresentProbe aged;
    aged.configure(spec);
    aged.arm_phase(30000);
    aged.publish(event);
    check(aged.expire(7001, 0).has_value(), "time expiry is explicit");
    check(!aged.pending(), "expired event cannot be selected");

    ComputePresentProbe callbacks;
    callbacks.configure(spec);
    callbacks.arm_phase(30000);
    callbacks.publish(event);
    for (unsigned i = 0; i < 256; ++i) callbacks.admitted_callback();
    check(!callbacks.expire(5001, 256), "callback cap includes endpoint");
    callbacks.admitted_callback();
    check(callbacks.expire(5001, 257).has_value(), "callback overrun expires");

    ComputePresentProbe no_producer;
    no_producer.configure(spec);
    no_producer.arm_phase(30000);
    check(no_producer.expire_no_producer(40000) && !no_producer.publishable() &&
          !no_producer.expire_no_producer(40001),
          "missing producer emits one bounded selector miss");

    ComputePresentProbe finished;
    finished.configure(spec);
    finished.arm_phase(30000);
    finished.publish(event);
    check(finished.finish(1) && !finished.finish(1) && !finished.pending(),
          "one-shot completion is final");

    auto concurrent_spec = spec;
    concurrent_spec.stride = 1;
    ComputePresentProbe concurrent;
    concurrent.configure(concurrent_spec);
    concurrent.arm_phase(30000);
    concurrent.publish(event);
    std::atomic<unsigned> admitted_samples{0};
    std::atomic<unsigned> raw_charges{0};
    std::vector<std::thread> peers;
    for (unsigned i = 0; i < 16; ++i)
        peers.emplace_back([&] {
            admitted_samples += concurrent.admit_scene_probe(1).has_value();
            raw_charges += concurrent.charge_raw(1, 16ull << 20);
        });
    for (auto& peer : peers) peer.join();
    check(admitted_samples == 4, "four-probe cap is shared across renderer threads");
    check(raw_charges == 10, "160 MiB raw cap is shared across renderer threads");
    return failures ? 1 : 0;
}
