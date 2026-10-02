#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_mbcnt_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {
int checks = 0, failures = 0;
void check(bool value, const std::string& name) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "[FAIL] %s\n", name.c_str()); }
}
void compare(const std::vector<uint32_t>& actual, const std::vector<uint32_t>& want,
             const std::string& name) {
    check(actual.size() == want.size(), name + " complete raw record extent");
    check(actual == want, name + " all logical64 raw EXP words");
    if (actual.size() == want.size()) for (size_t word = 0; word < want.size(); ++word)
        check(actual[word] == want[word], name + " word=" + std::to_string(word));
}
std::vector<uint32_t> run(const prosper::gpu::FragmentInvocationPacket& p,
                        const std::vector<uint32_t>& want, const std::string& name,
                        const std::filesystem::path& directory) {
    const auto r = prosper::gpu::recompile_fragment_packet(
        p, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4110});
    check(!r.spirv.empty() && r.rejection.empty(), name + " original guest emits: " + r.rejection);
    if (r.spirv.empty()) return {};
    bpermute_oracle::Interpreter vm(r.spirv);
    const auto actual = vm.run_packet(r.input_words, r.output_words);
    check(vm.error.empty(), name + " typed actual SOURCE: " + vm.error);
    compare(actual, want, name);
    if (!directory.empty()) {
        std::ofstream file(directory / (name + ".spv"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(r.spirv.data()), std::streamsize(r.spirv.size() * 4));
        file.close(); check(bool(file), name + " strict-validation artifact");
    }
    return actual;
}
void refuse(const prosper::gpu::FragmentInvocationPacket& p, const std::string& reason,
            const std::string& name, bool exact = true) {
    const auto r = prosper::gpu::recompile_fragment_packet(
        p, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4110});
    check(r.spirv.empty() && r.input_words.empty() && r.output_words.empty(),
          name + " transactional refusal");
    check(exact ? r.rejection == reason : r.rejection.find(reason) != std::string::npos,
          name + " precise cause: " + r.rejection);
}
} // namespace

int main(int argc, char** argv) {
    namespace m = prosper::test::fragment_packet::mbcnt;
    std::filesystem::path directory;
    if (argc == 3 && std::string(argv[1]) == "--dump-directory") {
        directory = argv[2]; std::filesystem::create_directories(directory);
    } else if (argc != 1) return 2;
    uint32_t ordinal = 0;
    for (const auto& c : m::cases()) {
        const auto p = m::packet(c);
        std::vector<prosper::gpu::Rdna2Inst> decoded;
        prosper::gpu::rdna2_walk(p.guest_code.data(), p.guest_code.size(), decoded);
        uint32_t mbcnts = 0;
        for (const auto& in : decoded)
            if (in.fmt == prosper::gpu::Rdna2Format::VOP3 &&
                (in.opcode == 0x365 || in.opcode == 0x366)) {
                ++mbcnts;
                check(in.n_src == 2 && in.dst.kind == prosper::gpu::OperandKind::VGPR,
                      "actual MBCNT has two raw sources and vector destination");
            }
        check(mbcnts == 4, "four original guest MBCNT sites decoded");
        if (c.skip_all && !decoded.empty()) {
            const auto& branch = decoded.front();
            check(branch.pc == 0 && branch.fmt == prosper::gpu::Rdna2Format::SOPP &&
                  branch.opcode == 0x02 && branch.len_dwords == 1 && decoded.back().is_end &&
                  branch.pc + branch.len_dwords + branch.simm16 == decoded.back().pc,
                  "all-ended original guest branches directly to S_ENDPGM");
            check(std::any_of(decoded.begin(), decoded.end(), [](const auto& in) {
                      return in.pc == 9 && in.fmt == prosper::gpu::Rdna2Format::SOP1 &&
                             in.opcode == prosper::gpu::kSop1OpcodeBcnt1I32B64 &&
                             in.words[0] == 0xbe8c106au;
                  }), "all-ended original guest retains the unreachable BCNT at pc9");
            auto unowned = p; unowned.slots_available[31] = false;
            refuse(unowned, "packet-invocation-state-unavailable",
                   "all-ended guest still requires complete owned slots");
            auto unsupported = p; unsupported.guest_code[3] |= 1u << 8;
            refuse(unsupported, "packet-modifier-unimplemented",
                   "all-ended guest still inventories unsupported original modifiers");
        }
        run(p, m::expected(c), "mbcnt_" + std::to_string(ordinal++), directory);
    }
    check(ordinal == 100, "nine source kinds, all64 bits, six addend kinds, skipped/ended and replacement");

    // A supplied true SCC takes this conditional edge to S_ENDPGM at execution time, but
    // the CFG must still retain its fallthrough arm. Replace exactly one original S_MOV:
    // a complete saved VCC definition is legal; a B32 DATA write expires the high mask half.
    auto conditional = m::packet({});
    conditional.guest_code.insert(conditional.guest_code.begin() + 2, 0xbe94046au);
    conditional.guest_code.insert(conditional.guest_code.begin(),
                                  0xbf850000u | uint32_t(conditional.guest_code.size() - 1));
    check(conditional.scc, "conditional ended control really supplies true SCC");
    auto conditional_expired = conditional;
    conditional_expired.guest_code[3] = 0xbe940318u; // S_MOV_B32 s20,s24, not a saved mask pair
    for (const auto* p : {&conditional, &conditional_expired}) {
        std::vector<prosper::gpu::Rdna2Inst> decoded;
        prosper::gpu::rdna2_walk(p->guest_code.data(), p->guest_code.size(), decoded);
        check(!decoded.empty(), "conditional reachability original instructions decode");
        if (decoded.empty()) continue;
        const auto& branch = decoded.front();
        check(branch.pc == 0 && branch.fmt == prosper::gpu::Rdna2Format::SOPP &&
              branch.opcode == 0x05 && branch.len_dwords == 1 && decoded.back().is_end &&
              branch.pc + branch.len_dwords + branch.simm16 == decoded.back().pc,
              "actual S_CBRANCH_SCC1 has both end and fallthrough CFG successors");
        const auto save = std::find_if(decoded.begin(), decoded.end(), [](const auto& in) {
            return in.pc == 3;
        });
        const bool expired = p == &conditional_expired;
        check(save != decoded.end() && save->fmt == prosper::gpu::Rdna2Format::SOP1 &&
              save->len_dwords == 1 && save->dst.kind == prosper::gpu::OperandKind::SGPR &&
              save->dst.value == 20 && save->n_src == 1 &&
              save->opcode == (expired ? prosper::gpu::kSop1OpcodeMovB32 :
                                        prosper::gpu::kSop1OpcodeMovB64) &&
              save->src[0].value == (expired ? 24 : 106) &&
              save->src[0].kind == (expired ? prosper::gpu::OperandKind::SGPR :
                                            prosper::gpu::OperandKind::Special),
              "one decoded conditional-arm instruction distinguishes DATA from live saved mask");
    }
    uint32_t conditional_changes = 0;
    for (size_t word = 0; word < conditional.guest_code.size(); ++word)
        if (conditional.guest_code[word] != conditional_expired.guest_code[word]) {
            ++conditional_changes;
            check(word == 3, "conditional lifetime control changes only the saved-mask definition");
        }
    check(conditional_changes == 1, "conditional reachability pair changes one guest instruction word");
    m::Case ended; ended.skip_all = true;
    run(conditional, m::expected(ended), "conditional_reachable_saved_mask", directory);
    refuse(conditional_expired, "mbcnt-unproven-saved-mask",
           "runtime-taken SCC1 cannot hide its reachable expired-mask arm", false);

    // A legal original-instruction mutation changes the first numeric MBCNT from LO to HI.
    // Walk both actual instruction streams; changing the oracle's first_is_high flag alone is
    // not evidence that a legal guest opcode changed. All other guest words remain identical.
    m::Case good; good.source = m::Source::Literal; good.mask = UINT64_MAX;
    m::Case changed = good; changed.first_is_high = true;
    const auto original = m::packet(good);
    auto mutated = original;
    constexpr uint32_t first_mbcnt_pc = 2; // two one-dword S_MOV_B64 saves precede it
    mutated.guest_code[first_mbcnt_pc] ^= (0x365u ^ 0x366u) << 16;
    check(m::packet(changed).guest_code == mutated.guest_code,
          "shared native HI fixture is this exact original opcode mutation");
    uint32_t changed_words = 0;
    for (size_t word = 0; word < original.guest_code.size(); ++word)
        if (original.guest_code[word] != mutated.guest_code[word]) {
            ++changed_words;
            check(word == first_mbcnt_pc, "only the first original MBCNT opcode word changes");
        }
    check(changed_words == 1, "one original guest instruction mutation");
    const auto decoded_witness = [&](const auto& p, uint32_t opcode) {
        std::vector<prosper::gpu::Rdna2Inst> decoded;
        prosper::gpu::rdna2_walk(p.guest_code.data(), p.guest_code.size(), decoded);
        const auto first = std::find_if(decoded.begin(), decoded.end(), [](const auto& in) {
            return in.fmt == prosper::gpu::Rdna2Format::VOP3 &&
                   (in.opcode == 0x365 || in.opcode == 0x366);
        });
        check(first != decoded.end(), "actual original/mutated MBCNT decodes");
        if (first == decoded.end()) return;
        check(first->pc == first_mbcnt_pc && first->opcode == opcode && first->len_dwords == 3 &&
              first->n_src == 2 && first->dst.kind == prosper::gpu::OperandKind::VGPR &&
              first->dst.value == 1 && first->src[0].kind == prosper::gpu::OperandKind::Literal &&
              first->literal == UINT32_MAX && first->src[1].kind == prosper::gpu::OperandKind::VGPR &&
              first->src[1].value == 7, "exact decoded first site retains its literal and v7 addend");
    };
    decoded_witness(original, 0x365);
    decoded_witness(mutated, 0x366);
    const auto baseline = run(original, m::expected(good), "original_lo_control", directory);
    const auto actual = run(mutated, m::expected(changed), "original_hi_control", directory);
    check(!actual.empty() && actual != m::expected(good), "legal guest LO-to-HI mutation reddens LO oracle");
    // Independently of expected(first_is_high): with all source bits set, first LO/HI counts
    // are min(N,32)/max(N-32,0). Their modular difference is -N below32, N-64 above32.
    // The second MBCNT carries that change into v2; the later pair resets both destinations.
    if (baseline.size() == 64 * 36 && actual.size() == baseline.size()) {
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t delta = lane < 32 ? 0u - lane : lane - 64u;
            for (uint32_t word = 0; word < 36; ++word) {
                const bool carries = word == 8 || word == 9 || word == 20 || word == 21;
                check(actual[lane * 36 + word] - baseline[lane * 36 + word] == (carries ? delta : 0u),
                      "analytic mutation delta lane=" + std::to_string(lane) + " word=" + std::to_string(word));
            }
        }
        for (uint32_t lane : {0u, 1u, 4u, 5u, 31u, 32u, 33u, 63u}) {
            const uint32_t addend = uint32_t(0xfffffff0ull + 3ull * lane);
            const uint32_t lo = lane < 32 ? lane : 32u, hi = lane < 32 ? 0u : lane - 32u;
            for (uint32_t export_word : {0u, 12u}) for (uint32_t channel = 0; channel < 2; ++channel) {
                const uint32_t word = lane * 36 + export_word + 8 + channel;
                check(baseline[word] == addend + lo + channel * hi &&
                      actual[word] == addend + hi + channel * hi,
                      "independent LO/HI boundary raw values lane=" + std::to_string(lane));
            }
            check(actual[lane * 36 + 32] == addend + lo &&
                  actual[lane * 36 + 33] == addend + lo + hi,
                  "later EXP boundary values are reset lane=" + std::to_string(lane));
        }
        check(baseline[4 * 36 + 8] == 0u && actual[4 * 36 + 8] == 0xfffffffcu,
              "original LO count wraps while mutated first HI adds zero");
    }

    // Operand KINDS must survive decoding: v106/v126 are numeric columns, not architectural
    // VCC/EXEC merely because their decoded register numbers overlap those scalar encodings.
    for (uint32_t reg : {106u, 126u}) {
        m::Case c; c.source = m::Source::Vgpr;
        auto p = m::packet(c);
        auto column = p.vgprs[4]; column.reg = reg; p.vgprs.push_back(column);
        p.guest_code[3] = (p.guest_code[3] & ~511u) | (256u + reg);
        run(p, m::expected(c), "numeric_v" + std::to_string(reg), directory);
    }
    m::Case scalar; scalar.source = m::Source::Scalar;
    auto missing = m::packet(scalar);
    std::erase_if(missing.sgprs, [](const auto& value) { return value.first == 25; });
    refuse(missing, "packet-sgpr-input-unavailable", "missing raw high source word");
    m::Case addend; addend.first_accumulator = m::Accumulator::Scalar;
    missing = m::packet(addend);
    std::erase_if(missing.sgprs, [](const auto& value) { return value.first == 27; });
    refuse(missing, "packet-sgpr-input-unavailable", "missing raw scalar accumulator");
    m::Case varying; varying.source = m::Source::Vgpr;
    missing = m::packet(varying);
    std::erase_if(missing.vgprs, [](const auto& value) { return value.reg == 5; });
    refuse(missing, "packet-vgpr-input-unavailable", "missing supplied vector source");
    m::Case only63 = varying; only63.exec = uint64_t(1) << 63;
    const auto owned_inactive = m::packet(only63);
    check(!(owned_inactive.exec_mask & (uint64_t(1) << 31)) &&
          !owned_inactive.export_enabled[31] && owned_inactive.slots_available[31],
          "paired slot31 is supplied, genuinely EXEC-off and export-ineligible");
    run(owned_inactive, m::expected(only63), "owned_inactive_slot31", directory);
    missing = owned_inactive; missing.slots_available[31] = false;
    refuse(missing, "packet-invocation-state-unavailable", "missing EXEC-off/export-ineligible slot");
    missing = m::packet(varying);
    std::erase_if(missing.vgprs, [](const auto& value) { return value.reg == 1; });
    refuse(missing, "packet-vgpr-input-unavailable", "missing predicated old destination");

    const auto operand_control = [&](uint32_t source, bool accumulator, const std::string& reason,
                                     const std::string& name, bool exact = true) {
        auto p = m::packet({});
        const uint32_t shift = accumulator ? 9u : 0u;
        p.guest_code[3] = (p.guest_code[3] & ~(511u << shift)) | (source << shift);
        refuse(p, reason, name, exact);
    };
    operand_control(127, false, "packet-mbcnt-mask-half-unimplemented", "LO cannot alias EXEC_HI");
    operand_control(107, false, "packet-mbcnt-mask-half-unimplemented", "LO cannot alias VCC_HI");
    operand_control(124, false, "packet-mbcnt-source-kind-unimplemented", "M0 is not owned MBCNT data");
    operand_control(126, true, "packet-mbcnt-accumulator-kind-unimplemented", "Bool EXEC is not a raw accumulator");
    operand_control(20, true, "packet-mbcnt-accumulator-state-unavailable",
                    "saved Bool is not a fabricated raw accumulator", false);
    auto modifiers = m::packet({}); modifiers.guest_code[2] |= 1u << 11;
    refuse(modifiers, "packet-mbcnt-modifier-unimplemented", "reserved OPSEL is not dropped");
    modifiers = m::packet({}); modifiers.guest_code[2] |= 1u << 8;
    refuse(modifiers, "packet-modifier-unimplemented", "ABS is not integer mask authority");

    // Saving VCC destroys the packet's old raw s20:s21 words. A later definite B32 write
    // replaces only its addressed word, never fabricates the untouched half or revives VCC.
    auto expired = m::packet({});
    std::vector<uint32_t> overwrite;
    prosper::test::fragment_packet::smov(overwrite, 20, 0x40000081u);
    expired.guest_code.insert(expired.guest_code.begin() + 2, overwrite.begin(), overwrite.end());
    refuse(expired, "mbcnt-unproven-saved-mask", "expired saved high half", false);
    // Same overwrite, then an actual complete mask definition: the exact MBCNT pair is legal again.
    expired.guest_code.insert(expired.guest_code.begin() + 4, 0xbe94046au);
    run(expired, m::expected({}), "saved_mask_redefined", directory);
    auto opposite = m::packet({}); opposite.guest_code[3] = (opposite.guest_code[3] & ~511u) | 21u;
    refuse(opposite, "mbcnt-unproven-saved-mask", "saved high word cannot alias a low Bool root", false);

    // One forward arm retains a Bool mask, the other definitely replaces both words with DATA.
    // Neither path's representation is a MUST fact at their join. A same-shape control has
    // definite DATA before the split and retains it on both arms.
    const auto join = [&](bool defined_before_split) {
        m::Case c; c.numeric_replacement = defined_before_split;
        auto p = m::packet(c);
        std::vector<uint32_t> arms{0xbf850005u};
        prosper::test::fragment_packet::smov(arms, 20, uint32_t(c.scalar));
        prosper::test::fragment_packet::smov(arms, 21, uint32_t(c.scalar >> 32));
        arms.push_back(0xbf820000u);
        p.guest_code.insert(p.guest_code.begin() + (defined_before_split ? 6 : 2),
                            arms.begin(), arms.end());
        return p;
    };
    refuse(join(false), "wave64-ambiguous-mask-read", "mask/DATA join cannot borrow either arm", false);
    m::Case replaced; replaced.numeric_replacement = true;
    run(join(true), m::expected(replaced), "definite_numeric_join", directory);
    std::printf("fragment_packet_mbcnt: cases=%u checks=%d failures=%d (owned packets, no raster claim)\n",
                ordinal, checks, failures);
    return failures ? 1 : 0;
}
