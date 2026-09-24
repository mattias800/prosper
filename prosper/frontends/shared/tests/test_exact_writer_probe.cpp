#include "shared/diagnostics/exact_writer_probe.hpp"

#include <atomic>
#include <cstdio>
#include <thread>

using namespace prosper::frontend;

int main() {
    const auto spec = parse_exact_writer_probe("0x300bbc0000:0x30a9830000:0x309abc0000:ms:30000");
    if (!spec.armed || spec.ps != 0x300bbc0000ull ||
        spec.input != 0x30a9830000ull || spec.output != 0x309abc0000ull ||
        spec.after_ms != 30000) return 1;
    const char* invalid[] = {
        "", "300bbc0000:0x30a9830000:0x309abc0000:ms:30000",
        "0x300bbc0000:0x30a9830000:0x30a9830000:ms:30000",
        "0x300bbc0000:0x30a9830000:0x309abc0000:ms:+30000",
        "0x300bbc0000:0x30a9830000:0x309abc0000:ms:120001",
        "0x300bbc0000:0x30a9830000:0x309abc0000:ms:30000junk",
        "0x300bbc0000:0x30a9830000:0x309abc0000:30000",
        "0x300bbc0000:0x30a9830000:auto:0x1800:ms:30000",
        "0x300bbc0000:0x30a9830000:auto:3200x0:ms:30000",
        "0x300bbc0000:0x30a9830000:auto:3200x1800junk:ms:30000",
        "0x300bbc0000:0x30a9830000:auto:8193x1800:ms:30000",
        "0x300bbc0000:0x30a9830000:auto:3200x1800:ms:120001",
    };
    for (const char* value : invalid)
        if (parse_exact_writer_probe(value).armed) return 2;
    const auto second = parse_exact_writer_second_input("37:0x30ad4b0000");
    if (!second.requested || !second.armed || second.binding != 37 ||
        second.address != 0x30ad4b0000ull ||
        parse_exact_writer_second_input(nullptr).requested) return 13;
    const char* bad_second[] = {
        "", "37", "37:30ad4b0000", "x:0x30ad4b0000", "37:0x0",
        "37:0x30ad4b0000junk", "4294967296:0x30ad4b0000",
        "37:0xfffffffffffffffff", "37:0x30ad4b0000:extra",
    };
    for (const char* value : bad_second)
        if (parse_exact_writer_second_input(value).armed) return 14;
    if (exact_writer_second_input_verdict(false, true, 12, 4) !=
            ExactWriterSecondInputVerdict::MissingBinding ||
        exact_writer_second_input_verdict(true, false, 12, 4) !=
            ExactWriterSecondInputVerdict::MissingImage ||
        exact_writer_second_input_verdict(true, true, 12, 0) !=
            ExactWriterSecondInputVerdict::NoVisibleRgb ||
        exact_writer_second_input_verdict(true, true, 0, 0) !=
            ExactWriterSecondInputVerdict::NoVisibleRgb ||
        exact_writer_second_input_verdict(true, true, 12, 4) !=
            ExactWriterSecondInputVerdict::Ready) return 15;
    if (exact_writer_match(spec.ps, spec.output, 3200, 1800, 1, 1, true, spec) !=
            ExactWriterMatch::Exact ||
        exact_writer_match(spec.ps + 1, spec.output, 3200, 1800, 1, 1, true, spec) !=
            ExactWriterMatch::WrongPs ||
        exact_writer_match(spec.ps, spec.output + 1, 3200, 1800, 1, 1, true, spec) !=
            ExactWriterMatch::WrongTarget ||
        exact_writer_match(spec.ps, spec.output, 3200, 1800, 2, 2, true, spec) !=
            ExactWriterMatch::MultiDraw ||
        exact_writer_match(spec.ps, spec.output, 3200, 1800, 1, 2, true, spec) !=
            ExactWriterMatch::MultiDraw ||
        exact_writer_match(spec.ps, spec.output, 3200, 1800, 1, 1, false, spec) !=
            ExactWriterMatch::NoColorWrite) return 3;
    const auto by_extent = parse_exact_writer_probe(
        "0x300bbc0000:0x30a9830000:auto:3200x1800:ms:30000");
    if (!by_extent.armed || !by_extent.output_by_extent || by_extent.output ||
        by_extent.output_width != 3200 || by_extent.output_height != 1800 ||
        exact_writer_match(by_extent.ps, 0x309afc0000, 3200, 1800, 1, 1, true,
                           by_extent) != ExactWriterMatch::Exact ||
        exact_writer_match(by_extent.ps, 0x309afc0000, 3200, 1080, 1, 1, true,
                           by_extent) != ExactWriterMatch::WrongTarget ||
        exact_writer_match(by_extent.ps, by_extent.input, 3200, 1800, 1, 1, true,
                           by_extent) != ExactWriterMatch::WrongTarget ||
        exact_writer_match(by_extent.ps, 0, 3200, 1800, 1, 1, true,
                           by_extent) != ExactWriterMatch::WrongTarget) return 16;
    if (!exact_writer_file_override_selected(true, true, 348, 348) ||
        exact_writer_file_override_selected(true, true, 347, 348) ||
        exact_writer_file_override_selected(true, true, 348, UINT64_MAX) ||
        exact_writer_file_override_selected(true, false, 348, 348) ||
        !exact_writer_file_override_selected(false, false, 347, UINT64_MAX)) return 17;
    const uint8_t same_a[] = {0, 0, 0, 255};
    const uint8_t same_b[] = {0, 0, 0, 255};
    const uint8_t different[] = {1, 0, 0, 255};
    if (exact_writer_scene_input_visible(1, 0) ||
        !exact_writer_scene_input_visible(1, 1) ||
        exact_writer_output_change(same_a, same_b) !=
            ExactWriterOutputChange::Identical ||
        exact_writer_output_change(same_a, different) !=
            ExactWriterOutputChange::Changed ||
        exact_writer_output_change({}, same_b) !=
            ExactWriterOutputChange::Unknown) return 3;

    ExactWriterProbe probe;
    probe.configure(spec);
    for (int i = 0; i < 16; ++i) {
        if (claim_exact_writer(probe, ExactWriterMatch::WrongPs, 30000) ||
            claim_exact_writer(probe, ExactWriterMatch::WrongTarget, 30000) ||
            claim_exact_writer(probe, ExactWriterMatch::MultiDraw, 30000)) return 4;
    }
    if (probe.attempts()) return 4;
    if (claim_exact_writer(probe, ExactWriterMatch::Exact, 29999)) return 4;
    std::atomic<int> winners{0};
    std::thread a([&] { if (probe.claim(30000)) winners++; });
    std::thread b([&] { if (probe.claim(30000)) winners++; });
    a.join(); b.join();
    if (winners != 1 || probe.attempts() != 1) return 5;
    if (!probe.charge_raw(ExactWriterProbe::kRawCap) ||
        probe.charge_raw(1) ||
        !probe.charge_bmp(ExactWriterProbe::kBmpCap) ||
        probe.charge_bmp(1)) return 6;
    probe.finish_failed_attempt();
    for (uint32_t attempt = 2; attempt <= 4; ++attempt) {
        if (probe.claim(30000) != attempt) return 7;
        probe.finish_failed_attempt();
    }
    if (probe.claim(30000) || probe.complete() || !probe.exhausted()) return 8;
    probe.configure(spec);
    if (!probe.claim(30000)) return 9;
    if (!probe.charge_raw(2) || !probe.charge_bmp(3) ||
        !probe.mark_bracketed() || !probe.bracketed() || probe.complete() ||
        probe.claim(30001) || !probe.charge_bmp(4) ||
        !probe.finish_final(true) || !probe.complete() || probe.claim(30002)) return 10;
    probe.configure(spec);
    if (!probe.expired(40000) || probe.claim(40001)) return 11;
    probe.configure(spec);
    if (!probe.claim(30000) || !probe.mark_bracketed() ||
        !probe.charge_bmp(8) || probe.finish_final(false) ||
        probe.complete() || !probe.exhausted() || probe.claim(30002) ||
        probe.charge_bmp(1)) return 12;
    std::puts("exact writer probe policy passed");
    return 0;
}
