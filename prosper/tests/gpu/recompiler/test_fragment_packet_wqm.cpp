#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_wqm_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {
int failures = 0, checks = 0;
void check(bool value, const std::string& name) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "[FAIL] %s\n", name.c_str()); }
}
void refuse(const prosper::gpu::FragmentInvocationPacket& p, const std::string& reason,
            const std::string& name) {
    const auto r = prosper::gpu::recompile_fragment_packet(
        p, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4099});
    check(r.spirv.empty() && r.input_words.empty() && r.output_words.empty(),
          name + " transactional refusal");
    check(r.rejection == reason, name + " precise reason: " + r.rejection);
}
void dump(const std::filesystem::path& directory, const std::string& name,
          const std::vector<uint32_t>& words) {
    if (directory.empty()) return;
    std::ofstream file(directory / (name + ".spv"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
    file.close(); check(bool(file), name + " strict-validation artifact");
}
void run(const prosper::test::fragment_packet::wqm::Case& c, const std::string& name,
         const std::filesystem::path& directory, bool oracle_control = false) {
    namespace w = prosper::test::fragment_packet::wqm;
    const auto input = w::packet(c);
    const auto decoded = prosper::gpu::rdna2_decode_one(
        input.guest_code.data() + 2, input.guest_code.size() - 2);
    check(decoded.fmt == prosper::gpu::Rdna2Format::SOP1 && decoded.opcode == 10 &&
          decoded.dst.value == int32_t(c.destination), name + " actual B64 WQM encoding");
    const auto r = prosper::gpu::recompile_fragment_packet(
        input, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4099});
    check(!r.spirv.empty() && r.rejection.empty(), name + " actual guest emits: " + r.rejection);
    if (r.spirv.empty()) return;
    bpermute_oracle::Interpreter vm(r.spirv);
    const auto actual = vm.run_packet(r.input_words, r.output_words), want = w::expected(c);
    check(vm.error.empty(), name + " typed SOURCE: " + vm.error);
    check(actual == want, name + " all raw logical64 EXP records");
    if (actual.size() == want.size()) for (size_t word = 0; word < want.size(); ++word)
        check(actual[word] == want[word], name + " word=" + std::to_string(word));
    dump(directory, name, r.spirv);
    if (oracle_control) {
        // Deliberately omit every actual Boolean OR's right input while preserving its type and
        // framing. This valid typed mutation must change live raw output, not merely a notice.
        auto changed = r.spirv;
        uint32_t ors = 0;
        for (size_t pc = 5; pc < changed.size();) {
            const uint32_t n = changed[pc] >> 16, op = changed[pc] & 0xffffu;
            if (!n || n > changed.size() - pc) break;
            if (op == 166 && n == 5) { changed[pc + 4] = changed[pc + 3]; ++ors; }
            pc += n;
        }
        check(ors != 0, name + " actual logical OR mutation targets");
        bpermute_oracle::Interpreter mutant(changed);
        const auto wrong = mutant.run_packet(r.input_words, r.output_words);
        check(mutant.error.empty() && wrong != want, name + " live quad/vote omission discriminator");
        dump(directory, name + "_oracle_no_or", changed);
    }
}
} // namespace
int main(int argc, char** argv) {
    namespace w = prosper::test::fragment_packet::wqm;
    using prosper::gpu::FragmentPacketQuadTopology;
    std::filesystem::path directory;
    if (argc == 3 && std::string(argv[1]) == "--dump-directory") {
        directory = argv[2]; std::filesystem::create_directories(directory);
    } else if (argc != 1) { std::fprintf(stderr, "unexpected arguments\n"); return 2; }
    // Every bit position, especially 31/32/63. Three initially EXEC-off neighbors must actually
    // write after expansion; the third record reuses the service with a different source.
    for (uint32_t lane = 0; lane < 64; ++lane) {
        w::Case c; c.exec = uint64_t(1) << lane; c.initial_scc = lane & 1u;
        run(c, "exec_bit_" + std::to_string(lane), directory, lane == 63);
    }
    uint32_t ordinal = 0;
    for (const auto source : {w::Source::Exec, w::Source::Vcc, w::Source::SavedVcc,
                              w::Source::ScalarPair, w::Source::Empty, w::Source::Full})
        for (uint32_t destination : {126u, 106u, 16u, 30u}) {
            w::Case c; c.source = source; c.destination = destination;
            c.scalar = (uint64_t(1) << 63) | (uint64_t(1) << 35) | 2;
            c.vcc = (uint64_t(1) << 40) | 8;
            c.initial_scc = source == w::Source::Empty;
            run(c, "source_destination_" + std::to_string(ordinal++), directory);
        }
    for (uint64_t mask : {uint64_t(0), UINT64_MAX, uint64_t(0x8000000100000001ull),
                          uint64_t(0xaaaaaaaa55555555ull)}) {
        w::Case c; c.exec = mask; c.source = w::Source::Vcc;
        c.vcc = ~mask; c.second = mask; c.initial_scc = true;
        run(c, "independent_masks_" + std::to_string(ordinal++), directory);
    }
    // Restored active lanes in the OPPOSITE half/quad consume the scalar SCC. A per-quad or
    // native32 SCC reduction can look correct at the widened pixels but fails this live sink.
    for (uint32_t source_lane : {0u, 40u, 63u}) {
        w::Case c; c.source = w::Source::Vcc; c.vcc = uint64_t(1) << source_lane;
        c.exec = uint64_t(1) << (source_lane < 32 ? 63u : 0u);
        run(c, "opposite_half_scc_" + std::to_string(source_lane), directory);
    }
    for (bool scc : {false, true}) {
        const auto r = prosper::gpu::recompile_fragment_packet(w::skipped_scc_packet(scc));
        check(!r.spirv.empty() && r.rejection.empty(), "known-topology skipped WQM emits");
        if (!r.spirv.empty()) {
            bpermute_oracle::Interpreter vm(r.spirv);
            const auto actual = vm.run_packet(r.input_words, r.output_words);
            check(vm.error.empty() && actual == w::skipped_scc_expected(scc),
                  "unrequested WQM service preserves live prior SCC and EXEC");
            dump(directory, scc ? "skipped_prior_scc_true" : "skipped_prior_scc_false", r.spirv);
        }
    }
    auto good = w::packet({});
    // The same source words without topology do not become a false raster/quad authority.
    auto p = good; p.quad_topology = FragmentPacketQuadTopology::Unknown;
    refuse(p, "packet-quad-topology-unavailable", "unknown topology");
    p = good; p.quad_topology = static_cast<FragmentPacketQuadTopology>(2);
    refuse(p, "packet-quad-topology-invalid", "malformed topology");
    p = good; p.slots_available[62] = false;
    refuse(p, "packet-invocation-state-unavailable", "missing initially inactive quad neighbor");
    p = good; p.mask_state_available = false;
    refuse(p, "packet-invocation-state-unavailable", "unknown EXEC/VCC/SCC state");
    p = good; p.guest_code[2] = w::sop1(9, 126, 126);
    refuse(p, "packet-wqm-b32-unimplemented", "unsupported B32 ownership width");
    for (uint32_t src : {129u, 193u + 1u, 240u, 255u, 127u, 124u, 31u}) {
        p = good; p.guest_code[2] = w::sop1(10, 126, src);
        if (src == 255) p.guest_code.insert(p.guest_code.begin() + 3, 0);
        refuse(p, "packet-wqm-source-form-unimplemented", "unsupported source " + std::to_string(src));
    }
    for (uint32_t dst : {17u, 107u, 124u, 127u}) {
        p = good; p.guest_code[2] = w::sop1(10, dst, 126);
        refuse(p, "packet-wqm-destination-form-unimplemented", "unsupported destination " + std::to_string(dst));
    }
    w::Case scalar; scalar.source = w::Source::ScalarPair;
    p = w::packet(scalar);
    std::erase_if(p.sgprs, [](const auto& value) { return value.first == 31; });
    refuse(p, "packet-sgpr-input-unavailable", "no invented numeric high word");
    // Initial scalar words must not reappear after the pair becomes a saved mask. A one-word
    // overwrite destroys that complete mask without defining its untouched low data word.
    // The matching NOP twin retains the saved VCC mask and must reach the same live raw sinks.
    for (bool overwrite_high : {false, true}) {
        p = w::packet(scalar);
        p.guest_code.insert(p.guest_code.begin() + 2,
            {w::sop1(4, 30, 106), overwrite_high
                ? w::sop1(prosper::gpu::kSop1OpcodeMovB32, 31, 24) : 0xbf800000u});
        const auto changed = prosper::gpu::rdna2_decode_one(
            p.guest_code.data() + 3, p.guest_code.size() - 3);
        check(overwrite_high
            ? changed.fmt == prosper::gpu::Rdna2Format::SOP1 &&
              changed.opcode == 3 && changed.dst.kind == prosper::gpu::OperandKind::SGPR &&
              changed.dst.value == 31 && changed.src[0].kind == prosper::gpu::OperandKind::SGPR &&
              changed.src[0].value == 24
            : changed.fmt == prosper::gpu::Rdna2Format::SOPP && changed.opcode == 0,
            "actual high-word overwrite / NOP encoding");
        const auto r = prosper::gpu::recompile_fragment_packet(
            p, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4099});
        if (overwrite_high) {
            check(r.spirv.empty() && r.input_words.empty() && r.output_words.empty(),
                  "expired saved-mask low word never becomes initial data or zero");
            check(r.rejection.starts_with("packet-guest-emission-refused:") &&
                  r.rejection.find("pc=4 reason=packet-wqm-source-state-unavailable") !=
                    std::string::npos,
                  "expired WQM source has a named emission cause: " + r.rejection);
        } else {
            check(!r.spirv.empty() && r.rejection.empty(), "live saved-mask NOP twin emits");
            if (!r.spirv.empty()) {
                bpermute_oracle::Interpreter vm(r.spirv);
                const auto actual = vm.run_packet(r.input_words, r.output_words);
                auto saved = scalar; saved.source = w::Source::Vcc;
                check(vm.error.empty() && actual == w::expected(saved),
                      "live saved-mask NOP twin reaches actual WQM EXP sinks");
                dump(directory, "live_saved_mask_nop_twin", r.spirv);
            }
        }
    }
    // Two genuine scalar definitions replace the whole expired mask lifetime and can be consumed
    // numerically by WQM. Do not turn the one-word refusal into a permanent register allow/denylist.
    p = w::packet(scalar);
    p.guest_code.insert(p.guest_code.begin() + 2, {w::sop1(4, 30, 106),
        w::sop1(prosper::gpu::kSop1OpcodeMovB32, 30, 24),
        w::sop1(prosper::gpu::kSop1OpcodeMovB32, 31, 25)});
    const auto replaced = prosper::gpu::recompile_fragment_packet(p);
    check(!replaced.spirv.empty() && replaced.rejection.empty(), "two real scalar words replace mask");
    if (!replaced.spirv.empty()) {
        bpermute_oracle::Interpreter vm(replaced.spirv);
        const auto actual = vm.run_packet(replaced.input_words, replaced.output_words);
        auto numeric = scalar;
        numeric.scalar = uint64_t(prosper::test::fragment_packet::branch_true) |
            (uint64_t(prosper::test::fragment_packet::branch_false) << 32);
        check(vm.error.empty() && actual == w::expected(numeric),
              "replacement scalar pair reaches actual WQM EXP sinks");
        dump(directory, "replaced_scalar_pair", replaced.spirv);
    }
    // A WQM result is a complete Bool-domain mask, not a materialized low SGPR word. Keep a later
    // physical-word read live in EXP: the real count twin emits, but the mask word must still refuse.
    for (uint32_t word : {12u, 30u}) {
        w::Case saved; saved.destination = 30;
        p = w::packet(saved);
        const auto first_exp = std::find(p.guest_code.begin(), p.guest_code.end(), 0xf800180fu);
        check(first_exp != p.guest_code.end(), "WQM physical-word control retains actual EXP");
        if (first_exp == p.guest_code.end()) continue;
        const size_t pc = size_t(first_exp - p.guest_code.begin());
        p.guest_code.insert(p.guest_code.begin() + pc, 0x7e0a0200u | word); // V_MOV v5,sWORD
        p.guest_code[pc + 2] = (p.guest_code[pc + 2] & ~0xffu) | 5u; // live channel0 consumes v5
        const auto r = prosper::gpu::recompile_fragment_packet(
            p, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4099});
        if (word == 30) {
            check(r.spirv.empty() && r.input_words.empty() && r.output_words.empty(),
                  "WQM mask word never becomes supplied initial data or zero");
            check(r.rejection.starts_with("packet-guest-emission-refused:") &&
                  r.rejection != "packet-guest-emission-refused:no-cause-recorded",
                  "later physical WQM mask read keeps a named cause: " + r.rejection);
        } else {
            check(!r.spirv.empty() && r.rejection.empty(), "defined WQM count word twin emits");
            if (!r.spirv.empty()) {
                bpermute_oracle::Interpreter vm(r.spirv);
                const auto actual = vm.run_packet(r.input_words, r.output_words);
                auto want = w::expected(saved);
                const uint64_t wide = w::expand(w::source_mask(saved));
                for (uint32_t lane = 0; lane < 64; ++lane)
                    if (((wide >> lane) & 1u) && !((saved.exec >> lane) & 1u))
                        want[lane * 36 + 12 + 8] = w::population(wide);
                check(vm.error.empty() && actual == want,
                      "defined count twin reaches actual live EXP channel");
                dump(directory, "defined_count_word_twin", r.spirv);
            }
        }
    }
    // Even branch-skipped WQM needs a declared topology. Its NOP twin with identical control
    // shape must emit without one; ordinary integer packets retain the Unknown/default contract.
    for (bool wqm : {false, true}) {
        p = prosper::test::fragment_packet::packet({});
        p.guest_code.insert(p.guest_code.begin(), {0xbf820001u,
            wqm ? w::sop1(10, 126, 126) : 0xbf800000u});
        if (wqm) refuse(p, "packet-quad-topology-unavailable", "dead WQM topology obligation");
        else {
            const auto r = prosper::gpu::recompile_fragment_packet(p);
            check(!r.spirv.empty(), "unknown topology non-quad NOP twin emits");
            p.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
            const auto tagged = prosper::gpu::recompile_fragment_packet(p);
            check(tagged.spirv == r.spirv && tagged.input_words == r.input_words &&
                  tagged.output_words == r.output_words,
                  "unused topology does not change existing integer SOURCE or owned buffers");
            if (!r.spirv.empty()) {
                bpermute_oracle::Interpreter vm(r.spirv);
                const auto raw = vm.run_packet(r.input_words, r.output_words);
                check(vm.error.empty() && raw == prosper::test::fragment_packet::expected({}, p),
                      "unknown topology non-quad actual raw sinks unchanged");
            }
        }
    }
    std::printf("fragment_packet_wqm: checks=%d failures=%d (owned logical quads, no raster claim)\n",
                checks, failures);
    return failures ? 1 : 0;
}
