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
        run(p, m::expected(c), "mbcnt_" + std::to_string(ordinal++), directory);
    }
    check(ordinal == 100, "nine source kinds, all64 bits, six addend kinds, skipped/ended and replacement");

    // A legal original-instruction mutation changes the first numeric MBCNT from LO to HI.
    // Its differently derived expected output must pass, and the good LO output must reject it.
    m::Case good; good.source = m::Source::Literal; good.mask = UINT64_MAX;
    m::Case changed = good; changed.first_is_high = true;
    const auto actual = run(m::packet(changed), m::expected(changed), "original_hi_control", directory);
    check(!actual.empty() && actual != m::expected(good), "legal guest LO-to-HI mutation reddens LO oracle");

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
    missing = m::packet(varying); missing.slots_available[31] = false;
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
