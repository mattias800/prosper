#include "fixtures/compute_runner.h"
#include "fixtures/fragment_packet_fixture.hpp"
#include "fixtures/fragment_packet_wqm_fixture.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
int checks = 0, failures = 0, dispatches = 0;
void check(bool value, const std::string& name) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "[FAIL] %s\n", name.c_str()); }
}
std::vector<uint32_t> execute(const prosper::gpu::FragmentPacketProgram& program) {
    // The shared runner's float vectors are byte containers here, never host FP values.
    // Both the emitted storage interfaces and the oracle operate on raw uint words.
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::vector<float> input(program.input_words.size());
    std::memcpy(input.data(), program.input_words.data(), input.size() * sizeof(float));
    ++dispatches;
    const auto output = prosper::test::run_compute(program.spirv, input, 64,
        static_cast<uint32_t>(program.output_words.size()));
    std::vector<uint32_t> words(output.size());
    if (!output.empty()) std::memcpy(words.data(), output.data(), output.size() * sizeof(float));
    return words;
}
void compare(const std::vector<uint32_t>& actual, const std::vector<uint32_t>& expected,
             const std::string& name) {
    check(actual.size() == expected.size(), name + " complete raw record extent");
    if (actual.size() != expected.size()) return;
    check(actual == expected, name + " all64 raw EXP records");
    for (size_t word = 0; word < expected.size(); ++word)
        check(actual[word] == expected[word], name + " word=" + std::to_string(word));
}
} // namespace

int main() {
    using namespace prosper::test::fragment_packet;
    const auto subgroup = prosper::test::default_compute_subgroup_properties();
    if (!subgroup.size) {
        std::fprintf(stderr, "[SKIP] fragment_packet_exec: no Vulkan compute device\n");
        return 77;
    }
    std::printf("fragment_packet_exec: selected-device default subgroup size=%u; "
                "owned logical64 workgroup, no required native Wave64\n", subgroup.size);
    // This Vulkan 1.1 runner publishes no float-control capability. Packet words use only
    // integer operations; do not inherit a property published by an unrelated device owner.
    prosper::gpu::reset_float_controls_support_for_test();
    uint32_t ordinal = 0;
    for (uint32_t selected : {63u, 31u, 64u})
        for (uint32_t variant = 0; variant < 8; ++variant) {
            Case c;
            c.selected_lane = selected;
            c.inactive_source = variant >= 1 && variant <= 3;
            c.leave_source_inactive = variant == 2 || variant == 3;
            c.second_export = variant == 3 || variant == 7;
            c.scalar_selector = variant == 6;
            c.data_base = variant == 1 ? 0x7f800001u : 0x51000000u + ordinal * 0x1000;
            auto input = packet(c);
            if (variant == 4) input.exec_mask = uint64_t(1) << 63;
            if (variant == 5) input.exec_mask = 0xffffffffu;
            input.export_enabled[7] = 0; // eligibility must not end a wave participant
            const auto program = prosper::gpu::recompile_fragment_packet(input,
                {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x40930000u + ordinal});
            const std::string name = "packet_" + std::to_string(ordinal++);
            check(!program.spirv.empty() && program.rejection.empty(), name + " actual guest emits");
            if (program.spirv.empty()) continue;
            compare(execute(program), expected(c, input), name);
        }

    // A legitimate guest-instruction mutation changes the LIVE read source from lane40 to
    // lane8. The good expected data must reject it, and a separately specified lane8 result
    // must accept it. This is neither a host-data flip nor a hand-written SPIR-V substitute.
    auto changed = packet({});
    const auto read = std::find(changed.guest_code.begin(), changed.guest_code.end(), 0xd760000fu);
    check(read != changed.guest_code.end(), "control finds actual guest READLANE");
    if (read != changed.guest_code.end()) {
        const size_t pc = static_cast<size_t>(read - changed.guest_code.begin());
        changed.guest_code[pc + 1] = 264u | (136u << 9); // VGPR8, inline integer lane8
        const auto program = prosper::gpu::recompile_fragment_packet(changed,
            {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4093ffffu});
        check(!program.spirv.empty() && program.rejection.empty(), "READLANE8 control emits");
        if (!program.spirv.empty()) {
            const auto actual = execute(program);
            const auto good = expected({}, changed);
            check(!actual.empty() && actual != good, "live READLANE mutation rejects good oracle");
            auto lane8 = good;
            for (uint32_t lane = 0; lane < 64; ++lane) lane8[lane * 12 + 10] = 0x51000000u + 8 * 17;
            compare(actual, lane8, "actual READLANE8 control");
        }
    }
    // These are supplied logical quads, not inferred raster ownership. Execute the actual WQM
    // lowering on hardware as well as in the CPU SOURCE interpreter. Every source bit must widen
    // its three initially EXEC-off neighbors, including all quads in both native wave32 halves.
    namespace w = prosper::test::fragment_packet::wqm;
    uint32_t wqm_ordinal = 0;
    const auto run_wqm = [&](const w::Case& c, const std::string& name) {
        const auto program = prosper::gpu::recompile_fragment_packet(w::packet(c),
            {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x40990000u + wqm_ordinal++});
        check(!program.spirv.empty() && program.rejection.empty(), name + " actual WQM guest emits");
        if (!program.spirv.empty()) compare(execute(program), w::expected(c), name);
    };
    for (uint32_t lane = 0; lane < 64; ++lane) {
        w::Case c; c.exec = uint64_t(1) << lane; c.initial_scc = lane & 1u;
        run_wqm(c, "wqm_exec_bit_" + std::to_string(lane));
    }
    for (const auto source : {w::Source::Exec, w::Source::Vcc, w::Source::SavedVcc,
                              w::Source::ScalarPair, w::Source::Empty, w::Source::Full})
        for (uint32_t destination : {126u, 106u, 16u, 30u}) {
            w::Case c; c.source = source; c.destination = destination;
            c.scalar = (uint64_t(1) << 63) | (uint64_t(1) << 35) | 2;
            c.vcc = (uint64_t(1) << 40) | 8;
            c.initial_scc = source == w::Source::Empty;
            run_wqm(c, "wqm_source_destination_" + std::to_string(wqm_ordinal));
        }
    for (uint64_t mask : {uint64_t(0), UINT64_MAX, uint64_t(0x8000000100000001ull),
                          uint64_t(0xaaaaaaaa55555555ull)}) {
        w::Case c; c.exec = mask; c.source = w::Source::Vcc;
        c.vcc = ~mask; c.second = mask; c.initial_scc = true;
        run_wqm(c, "wqm_independent_masks_" + std::to_string(wqm_ordinal));
    }
    // Restored active pixels in the opposite half consume the scalar SCC: a native32 or per-quad
    // vote can otherwise appear correct at the widened pixels. Repeated WQM sites also change the
    // source to lane40, so an old event/result cannot silently satisfy the last EXP record.
    for (uint32_t source_lane : {0u, 40u, 63u}) {
        w::Case c; c.source = w::Source::Vcc; c.vcc = uint64_t(1) << source_lane;
        c.exec = uint64_t(1) << (source_lane < 32 ? 63u : 0u);
        run_wqm(c, "wqm_opposite_half_scc_" + std::to_string(source_lane));
    }
    for (bool scc : {false, true}) {
        const auto program = prosper::gpu::recompile_fragment_packet(w::skipped_scc_packet(scc));
        check(!program.spirv.empty() && program.rejection.empty(), "known-topology skipped WQM emits");
        if (!program.spirv.empty()) compare(execute(program), w::skipped_scc_expected(scc),
            scc ? "skipped_prior_scc_true" : "skipped_prior_scc_false");
    }
    check(wqm_ordinal == 95, "all64 WQM bits,24 source/destination,4 mask and3 SCC cases compiled");
    check(dispatches == 122, "all24 good packets, live mutation,95 WQM and2 skipped WQM dispatched");
    std::printf("fragment_packet_exec: dispatches=%d checks=%d failures=%d "
                "(owned packets only; no raster/game claim)\n", dispatches, checks, failures);
    return failures ? 1 : 0;
}
