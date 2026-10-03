// The ORIGINAL guest SMEM/P1/P2/ADD/MUL/explicit sample/EXP chain must consume owned inputs.
// A compiler admission or a success status alone is not the oracle: evaluate actual typed SOURCE
// and compare every raw sink, then use the production transactional decoder. No guest raster ABI.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_resource_packet_fixture.hpp"
#include "fixtures/fragment_special_f32_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_resource_packet;
void check(bool value, const std::string& name) {
    EXPECT_TRUE(value) << name;
}
std::filesystem::path dump_directory() {
    // Author explicitly supplies an owned dump root; ordinary registered tests create no files.
    const char* root = std::getenv("PROSPER_RESOURCE_PACKET_SPV_DIRECTORY");
    if (!root || !*root) return {};
    std::filesystem::path directory(root);
    std::filesystem::create_directories(directory);
    return directory;
}
FragmentResourcePacketProgram compile(const FragmentResourcePacket& p) {
    return recompile_fragment_resource_packet(p, {RecompileDiagnosticStage::Fragment, 0x4192});
}
std::vector<uint32_t> evaluate(const FragmentResourcePacketProgram& program, std::string& error,
                             const std::vector<uint32_t>* alternate_module = nullptr) {
    bpermute_oracle::Interpreter vm(alternate_module ? *alternate_module : program.packet.spirv);
    for (uint32_t i = 0; i < program.images.size(); ++i)
        for (const auto& mip : program.images[i].mips)
            vm.sampled_images[16 + i].push_back({mip.width, mip.height, mip.texels});
    auto words = vm.run_packet(program.packet.input_words, program.packet.output_words);
    error = vm.error;
    return words;
}
void dump(const FragmentResourcePacketProgram& p, const std::string& name, const std::filesystem::path& directory) {
    if (directory.empty() || p.packet.spirv.empty()) return;
    std::ofstream f(directory / (name + ".spv"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(p.packet.spirv.data()), p.packet.spirv.size() * 4);
    f.close(); check(bool(f), name + " SOURCE retained");
}
std::vector<uint32_t> sink(const FragmentResourcePacket& p, const std::vector<uint32_t>& expected,
                          const std::string& name, const std::filesystem::path& directory) {
    const auto program = compile(p);
    check(!program.packet.spirv.empty() && program.packet.rejection.empty(), name + " emits: " + program.packet.rejection);
    if (program.packet.spirv.empty()) return {};
    std::string error; const auto words = evaluate(program, error);
    check(error.empty(), name + " actual typed SOURCE: " + error);
    const auto result = decode_fragment_resource_packet(program, words, true, p.device.device_identity);
    check(result.rejection.empty(), name + " production status consumer: " + result.rejection);
    check(result.exports == expected, name + " exact all64 raw EXP");
    if (result.exports.size() == expected.size())
        for (size_t word = 0; word < expected.size(); ++word)
            check(result.exports[word] == expected[word], name + " raw word=" + std::to_string(word));
    dump(program, name, directory);
    return words;
}
void reject(const FragmentResourcePacket& p, const std::string& reason, const std::string& name) {
    const auto r = compile(p);
    check(r.packet.spirv.empty() && r.packet.input_words.empty() && r.packet.output_words.empty(), name + " transactional compile refusal");
    check(r.packet.rejection == reason, name + " exact reason=" + r.packet.rejection);
}
void runtime_reject(const FragmentResourcePacket& p, FragmentPacketRuntimeFailure reason,
                    const std::string& name, const std::filesystem::path& directory,
                    uint32_t first_pc, uint32_t lane = 0) {
    const auto program = compile(p);
    check(!program.packet.spirv.empty(), name + " runtime obligation emitted: " + program.packet.rejection);
    if (program.packet.spirv.empty()) return;
    std::string error; const auto words = evaluate(program, error);
    check(error.empty(), name + " completed typed all-worker execution: " + error);
    const auto r = decode_fragment_resource_packet(program, words, true, p.device.device_identity);
    check(r.exports.empty() && !r.rejection.empty() && r.failure == reason && r.lane == lane,
          name + " whole packet refused with deterministic lane/first reason");
    check(r.pc == first_pc, name + " original first failing guest PC=" + std::to_string(first_pc) +
          " actual=" + std::to_string(r.pc));
    dump(program, name, directory);
}
std::vector<uint32_t> arithmetic_expected(uint32_t raw) {
    std::vector<uint32_t> words(64 * 12, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t header[]{1, 1, 1, 0, 1, 0, 1, 1};
        std::copy(std::begin(header), std::end(header), words.begin() + lane * 12);
        words[lane * 12 + 8] = raw;
    }
    return words;
}
void update_image_pc(FragmentResourcePacket& p) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(p.invocation.guest_code.data(), p.invocation.guest_code.size(), ins);
    for (const auto& in : ins) if (in.fmt == Rdna2Format::MIMG) p.images[0].pc = in.pc;
}
} // namespace

TEST(FragmentResourcePacket, OriginalOwnedChainsAndTransactionalConsumer) {
    const auto directory = dump_directory();
    for (bool lod : {false, true}) for (bool inactive : {false, true}) for (uint32_t variant : {0u, 1u}) {
        const auto p = fixture::chain(lod, inactive, variant);
        const auto name = "chain_" + std::to_string(lod) + "_inactive" + std::to_string(inactive) + "_data" + std::to_string(variant);
        const auto words = sink(p, fixture::expected_chain(p, lod, variant), name, directory);
        if (words.empty()) continue;
        const auto program = compile(p);
        check(decode_fragment_resource_packet(program, words, false, p.device.device_identity).exports.empty(), name + " no completion publishes nothing");
        check(decode_fragment_resource_packet(program, words, true, p.device.device_identity + 1).exports.empty(), name + " wrong executing owner publishes nothing");
        auto bad = words; bad.pop_back();
        check(decode_fragment_resource_packet(program, bad, true, p.device.device_identity).exports.empty(), name + " truncation publishes nothing");
        bad = words; bad[program.status_offset + 63 * 3] ^= 1;
        check(decode_fragment_resource_packet(program, bad, true, p.device.device_identity).exports.empty(), name + " last lane magic required");
        bad = words; bad[63 * 36 + 33] = 0xfeed0001;
        check(decode_fragment_resource_packet(program, bad, true, p.device.device_identity).rejection == "packet-disabled-export-payload-invalid",
              name + " malformed last disabled EXP channel publishes nothing");
        bad = words; bad[program.status_offset + 63 * 3 + 1] = 0x12345678; bad[program.status_offset + 63 * 3 + 2] = 3;
        check(decode_fragment_resource_packet(program, bad, true, p.device.device_identity).rejection == "packet-status-record-invalid", name + " invalid guest-PC refuses");
        // A valid typed arithmetic omission must alter the executed SOURCE, not just the oracle's
        // expected array. The first live uint64 multiply carries real nonzero P10 or numeric data.
        auto mutant = program.packet.spirv;
        uint32_t wide = 0, boolean = 0, narrow = 0; size_t site = 0, direction_select = 0;
        std::unordered_map<uint32_t, uint32_t> comparisons;
        for (size_t pc = 5; pc < mutant.size();) {
            const auto n = mutant[pc] >> 16, op = mutant[pc] & 0xffff;
            if (!n || n > mutant.size() - pc) break;
            if (op == 21 && n == 4 && mutant[pc + 2] == 64) wide = mutant[pc + 1];
            if (op == 21 && n == 4 && mutant[pc + 2] == 32 && mutant[pc + 3] == 0) narrow = mutant[pc + 1];
            if (op == 20 && n == 2) boolean = mutant[pc + 1];
            if ((op == 174 || op == 176) && n == 5) comparisons[mutant[pc + 2]] = op;
            // The packer's normal/subnormal shift direction selects BOOL comparisons, not
            // uint bits. This is the actual invalid OpSelect caught by the first native run.
            if (op == 169 && n == 6 && mutant[pc + 1] == boolean &&
                comparisons[mutant[pc + 3]] == 174 && comparisons[mutant[pc + 4]] == 174 &&
                comparisons[mutant[pc + 5]] == 176 && !direction_select) direction_select = pc;
            if (op == 132 && n == 5 && mutant[pc + 1] == wide && !site) site = pc;
            pc += n;
        }
        check(site && boolean, name + " live typed uint64 producer found");
        if (site) {
            mutant[site] = (5u << 16) | 128u;
            std::string error; const auto changed = evaluate(program, error, &mutant);
            check(!error.empty() || changed != words, name + " actual multiply omission is detected");
            mutant = program.packet.spirv; mutant[site + 1] = boolean;
            (void)evaluate(program, error, &mutant);
            check(!error.empty(), name + " wrong uint64 result type fails closed");
        }
        check(direction_select && narrow, name + " actual Boolean rounding-direction Select found");
        if (direction_select) {
            mutant = program.packet.spirv; mutant[direction_select + 1] = narrow;
            std::string error; (void)evaluate(program, error, &mutant);
            check(error == "Select type mismatch", name + " Boolean arms with uint result fail closed");
        }
    }
}
TEST(FragmentResourcePacket, LargerRectangularMipAndResourceChanges) {
    const auto directory = dump_directory();
    std::vector<uint32_t> baseline_spirv, baseline_sink;
    for (uint32_t variant : {0u, 1u}) {
        const auto p = fixture::rectangular_chain(variant);
        const auto words = sink(p, fixture::expected_rectangular_chain(variant),
            "rectangular_mip4_data" + std::to_string(variant), directory);
        const auto module = compile(p).packet.spirv;
        if (!variant) { baseline_spirv = module; baseline_sink = words; }
        else {
            EXPECT_EQ(module, baseline_spirv) << "owned texel words are not baked into SOURCE";
            EXPECT_NE(words, baseline_sink) << "changed owned buffer/image reaches actual raw EXP";
        }
    }
}
TEST(FragmentResourcePacket, FiniteArithmeticAllRoundingAndDenormalModes) {
    const auto directory = dump_directory();
    // Independent hand-derived IEEE bit rails; integer SOURCE does not inherit device FTZ or RTE.
    for (uint32_t denorm = 0; denorm < 4; ++denorm) for (uint32_t round = 0; round < 4; ++round) {
        const auto mode = static_cast<uint8_t>((denorm << 4) | round);
        const auto suffix = std::to_string(mode);
        sink(fixture::arithmetic(3, 0x3f800000, 0x33800000, mode),
             arithmetic_expected(round == 1 ? 0x3f800001 : 0x3f800000), "halfway_positive_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3f800001, 0x33800000, mode),
             arithmetic_expected(round == 0 || round == 1 ? 0x3f800002 : 0x3f800001), "halfway_odd_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3fffffff, 0x33800000, mode),
             arithmetic_expected(round == 0 || round == 1 ? 0x40000000 : 0x3fffffff), "halfway_carry_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x80000000, 0x80000000, mode),
             arithmetic_expected(0x80000000), "same_negative_zeros_" + suffix, directory);
        sink(fixture::arithmetic(3, 0, 0x80000000, mode),
             arithmetic_expected(round == 2 ? 0x80000000 : 0), "opposite_signed_zeros_" + suffix, directory);
        sink(fixture::arithmetic(3, 0xbf800000, 0xb3800000, mode),
             arithmetic_expected(round == 2 ? 0xbf800001 : 0xbf800000), "halfway_negative_" + suffix, directory);
        sink(fixture::arithmetic(8, 0x00800000, 0x3f000000, mode),
             arithmetic_expected(denorm & 2 ? 0x00400000 : 0), "normal_to_subnormal_" + suffix, directory);
        sink(fixture::arithmetic(3, 1, 1, mode),
             arithmetic_expected(denorm == 3 ? 2 : 0), "input_subnormal_" + suffix, directory);
        sink(fixture::arithmetic(8, 0x80000001, 0x3f000000, mode),
             arithmetic_expected(denorm == 3 && round == 2 ? 0x80000001 : 0x80000000), "negative_tiny_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3f800000, 0xbf800000, mode),
             arithmetic_expected(round == 2 ? 0x80000000 : 0), "cancellation_zero_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3f800000, 0x2b800000, mode),
             arithmetic_expected(round == 1 ? 0x3f800001 : 0x3f800000), "distant_positive_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3f800000, 0xab800000, mode),
             arithmetic_expected(round == 2 || round == 3 ? 0x3f7fffff : 0x3f800000), "distant_subtract_" + suffix, directory);
        sink(fixture::arithmetic(3, 0xbf800000, 0x2b800000, mode),
             arithmetic_expected(round == 1 || round == 3 ? 0xbf7fffff : 0xbf800000), "distant_negative_subtract_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x00800000, 1, mode),
             arithmetic_expected(denorm & 1 ? 0x00800001 : 0x00800000), "normal_subnormal_exact_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3f800000, 1, mode),
             arithmetic_expected((denorm & 1) && round == 1 ? 0x3f800001 : 0x3f800000),
             "saturated_alignment_positive_" + suffix, directory);
        sink(fixture::arithmetic(3, 0x3f800000, 0x80000001, mode),
             arithmetic_expected((denorm & 1) && (round == 2 || round == 3) ? 0x3f7fffff : 0x3f800000),
             "saturated_alignment_subtract_" + suffix, directory);
    }
}
TEST(FragmentResourcePacket, RuntimeFailureIsStickyAndSuppressesWholePacket) {
    const auto directory = dump_directory();
    runtime_reject(fixture::arithmetic(8, 0x7f7fffff, 0x40000000, 0x30), FragmentPacketRuntimeFailure::FiniteOverflow, "finite_overflow", directory, 0);
    runtime_reject(fixture::arithmetic(3, 0x7fc00001, 0x3f800000, 0x30), FragmentPacketRuntimeFailure::NonFinite, "nonfinite_no_fake_canonical_nan", directory, 0);
    auto p = fixture::chain();
    p.invocation.vgprs[0].words[63] = 0x7fc00001;
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(p.invocation.guest_code.data(), p.invocation.guest_code.size(), decoded);
    const auto first_p1 = std::find_if(decoded.begin(), decoded.end(),
        [](const auto& in) { return in.fmt == Rdna2Format::VINTRP; });
    ASSERT_NE(first_p1, decoded.end());
    ASSERT_EQ(first_p1->pc, 4u); ASSERT_EQ(first_p1->opcode, 0u);
    // This same lane subsequently fails P2 and sampling. Neither may overwrite first PC4.
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "lane63_whole_packet_rejection", directory, 4, 63);
    p = fixture::chain(); p.invocation.sgprs.back().second = 0; // actual S# high word mismatch
    // sampler word3 is already zero; mutate the actual first S# word instead.
    for (auto& [reg, value] : p.invocation.sgprs) if (reg == 12) value ^= 1;
    runtime_reject(p, FragmentPacketRuntimeFailure::DescriptorMismatch, "actual_sampler_identity", directory, p.images[0].pc);
    p = fixture::chain(); for (auto& [reg, value] : p.invocation.sgprs) if (reg == 16) value ^= 1;
    runtime_reject(p, FragmentPacketRuntimeFailure::M0Mismatch, "actual_original_m0_writer", directory, 4);
    p = fixture::chain(); for (auto& t : p.parameter_cache.parameters) { t.p0 = 0x3f800000; t.p10 = 0x33800000; }
    p.invocation.vgprs[0].words.fill(0x3f800000);
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "interpolation_halfway_not_exact", directory, 4);
    p = fixture::chain(); for (auto& t : p.parameter_cache.parameters) t.p10 = 0x00800000;
    p.invocation.vgprs[0].words.fill(0x3f000000);
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "interpolation_underflow_not_normal", directory, 4);
    p = fixture::chain(true); p.buffers[0].words[2] = fixture::bits(0.0f);
    runtime_reject(p, FragmentPacketRuntimeFailure::SampleCoordinateDomain, "original_explicit_lod_changes_coordinate_obligation", directory, p.images[0].pc);
    p = fixture::chain(true); p.buffers[0].words[2] = fixture::bits(0.5f);
    runtime_reject(p, FragmentPacketRuntimeFailure::SampleLodDomain, "fractional_lod_named_not_rounded", directory, p.images[0].pc);
    p = fixture::chain();
    const auto sample_pc = p.images[0].pc;
    p.invocation.guest_code.insert(p.invocation.guest_code.begin() + sample_pc,
        {0x7e1402ffu, fixture::bits(0.125f) + 1}); // original v10 <- literal adjacent ULP
    update_image_pc(p);
    runtime_reject(p, FragmentPacketRuntimeFailure::SampleCoordinateDomain, "adjacent_ulp_not_exact_texel_center", directory, p.images[0].pc);
    p = fixture::chain();
    for (auto& parameter : p.parameter_cache.parameters) parameter.p0 = parameter.p10 = parameter.p20 = 0;
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "interpolation_zero_sign_authority_unavailable", directory, 4);
}
TEST(FragmentResourcePacket, CompletionAndOwnershipAreCompileTimeObligations) {
    auto p = fixture::chain();
    for (uint32_t destination : {17u, 18u, 19u}) {
        auto bad = p;
        bad.invocation.guest_code[1] = (bad.invocation.guest_code[1] & ~(127u << 6)) | (destination << 6);
        std::vector<Rdna2Inst> decoded;
        rdna2_walk(bad.invocation.guest_code.data(), bad.invocation.guest_code.size(), decoded);
        ASSERT_GT(decoded.size(), 1u);
        EXPECT_EQ(decoded[1].fmt, Rdna2Format::SMEM);
        EXPECT_EQ(decoded[1].dst.value, static_cast<int>(destination));
        reject(bad, "packet-smem-destination-alignment-invalid", "actual x4 unaligned SGPR destination");
    }
    p.invocation.guest_code[3] = 0xbf800000;
    reject(p, "packet-smem-result-read-before-wait", "SMEM original completion required");
    p = fixture::chain(); p.invocation.guest_code[3] = 0xbf850001;
    p.invocation.guest_code.insert(p.invocation.guest_code.begin() + 4, 0xbf8c0000); update_image_pc(p);
    reject(p, "packet-smem-result-read-before-wait", "WAIT on one SCC arm not MUST completion");
    p = fixture::chain(); p.invocation.guest_code[p.images[0].pc + 2] = 0xbf800000;
    reject(p, "packet-image-result-read-before-wait", "MIMG original completion required");
    p = fixture::chain(); p.buffers.clear(); reject(p, "packet-buffer-readpoint-unavailable", "missing owned readpoint");
    p = fixture::chain(); p.parameter_cache.available = false;
    reject(p, "packet-parameter-cache-or-quad-ownership-unavailable", "missing exact parameter owner");
    p = fixture::chain(); p.invocation.guest_code[0] = 0xbf800000;
    reject(p, "packet-parameter-m0-read-before-definition", "allocated M0 is not defined M0");
    p = fixture::chain(); p.device.shader_int64_enabled = false;
    reject(p, "packet-enabled-int64-device-unavailable", "device must actually enable shaderInt64");
    p = fixture::chain(); p.invocation.float_mode = {};
    reject(p, "packet-f32-launch-mode-or-flags-unavailable", "unknown mode not mode0");
    p = fixture::chain(); p.launch_rsrc1 = {};
    reject(p, "packet-f32-launch-rsrc1-unavailable", "missing complete original launch word");
    p = fixture::chain(); p.launch_rsrc1.value ^= 1u << 12;
    reject(p, "packet-f32-launch-rsrc1-association-mismatch", "mode must join actual raw launch word");
    p = fixture::chain(); p.invocation.guest_code[p.images[0].pc] ^= (0x27u ^ 0x20u) << 18;
    reject(p, "packet-image-derivative-or-effect-unimplemented", "implicit SAMPLE not replaced with LOD0");
    p = fixture::chain(); p.images[0].sampler[2] = 2u << 26;
    reject(p, "packet-sampler-domain-unimplemented", "linear mip filter is not nearest");
    p = fixture::chain(); p.device.rgba32_sfloat_sampled = false;
    reject(p, "packet-enabled-image-format-device-unavailable", "format witness not inferred from Int64");
    p = fixture::chain(); p.images[0].mips[0].texels.resize(262145);
    reject(p, "packet-resource-input-budget", "owned payload budget not convenient image dimension");
    p = fixture::chain(); p.invocation.guest_code[4] &= ~(255u << 18); // actual P1 VDST=VSRC=v0
    reject(p, "packet-parameter-p1-alias-mode-unavailable", "P1 alias cannot assume unknown HALF_LDS mode");
}
TEST(FragmentResourcePacket, IntegerWaveExtensionsDoNotExpandResourceInputAuthority) {
    const auto good = fixture::arithmetic(3u, fixture::bits(0.25f), fixture::bits(0.5f), 0x30u);
    ASSERT_FALSE(compile(good).packet.spirv.empty());
    auto bad = good;
    bad.invocation.raw_windows.emplace_back();
    reject(bad, "packet-resource-raw-window-domain-unimplemented",
           "raw-window authority does not mix with resource packet ownership");
    bad = good;
    std::erase_if(bad.invocation.vgprs, [](const auto& column) { return column.reg == 2u; });
    reject(bad, "packet-vgpr-input-unavailable",
           "resource packet still requires supplied destination storage despite a guest writer");
    for (uint32_t opcode : {0x04u, 0x16u, 0x1bu}) {
        bad = fixture::arithmetic(opcode, fixture::bits(0.25f), fixture::bits(0.5f), 0x30u);
        reject(bad, "packet-valu-op-unimplemented",
               "integer live bridge opcode does not widen the closed resource packet slice");
    }
    // The independently published resource services retain their special-F32 support. These
    // exact rails are hand-derived: reciprocal(4)=.25, sqrt(4)=2, reciprocal-sqrt(4)=.5.
    const auto directory = dump_directory();
    for (const auto [opcode, result] :
         {std::pair{0x2au, 0x3e800000u}, std::pair{0x33u, 0x40000000u},
          std::pair{0x2eu, 0x3f000000u}}) {
        std::array<uint32_t, 64> inputs;
        inputs.fill(0x40800000u);
        auto special = prosper::test::fragment_special_f32::packet(opcode, inputs);
        std::vector<uint32_t> wanted(64u * 24u, 0u);
        for (uint32_t lane = 0; lane < 64u; ++lane)
            for (uint32_t event = 0; event < 2u; ++event) {
                const uint32_t header[]{1u, 1u, 1u, 0u, 1u, 0u, 1u, 1u};
                const auto offset = lane * 24u + event * 12u;
                std::copy(std::begin(header), std::end(header), wanted.begin() + offset);
                wanted[offset + 8u] = result;
            }
        const auto words =
            sink(special, wanted, "resource_special_" + std::to_string(opcode), directory);
        ASSERT_EQ(words.size(), wanted.size() + 64u * kFragmentResourceStatusWords);
        const auto integer = recompile_fragment_packet(special.invocation);
        EXPECT_TRUE(integer.spirv.empty());
        EXPECT_EQ(integer.rejection, "packet-valu-op-unimplemented")
            << "resource-only special service does not grant generic Owned64 FP authority";
        bad = special;
        bad.entry_facts = {};
        reject(bad, "packet-special-f32-producing-rsrc2-unavailable",
               "new math support still needs the producing exception/trap observation");
        bad = special;
        bad.invocation.raw_windows.emplace_back();
        reject(bad, "packet-resource-raw-window-domain-unimplemented",
               "special math does not merge resource and live raw-window authority");
    }
}
TEST(FragmentResourcePacket, AlignedSmemWriterAndActualReachingP2Destination) {
    const auto directory = dump_directory();
    auto p = fixture::chain();
    p.invocation.guest_code[1] = (p.invocation.guest_code[1] & ~(127u << 6)) | (20u << 6);
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(p.invocation.guest_code.data(), p.invocation.guest_code.size(), decoded);
    for (const auto& in : decoded) {
        if ((in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2) &&
            in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 16 && in.src[0].value <= 18)
            p.invocation.guest_code[in.pc] = (p.invocation.guest_code[in.pc] & ~511u) | uint32_t(in.src[0].value + 4);
    }
    sink(p, fixture::expected_chain(p, false), "aligned_smem_s20", directory);
    p = fixture::chain();
    // An original V_MOV overwrites the P1 destination before P2. P2 must consume the reaching
    // VDST, not remember P1 or substitute a host-final interpolant. Sampling remains valid.
    p.invocation.guest_code.insert(p.invocation.guest_code.begin() + 5, {0x7e1402ffu, fixture::bits(0.125f)});
    update_image_pc(p);
    auto wanted = fixture::expected_chain(p, false);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t i = lane & 1, j = (lane >> 1) & 1;
        wanted[lane * 36 + 8] = fixture::bits(0.125f + 0.25f * float(j));
        for (uint32_t c = 0; c < 4; ++c)
            wanted[lane * 36 + 20 + c] = fixture::bits(float(1 + j + 4 * (i + 2 * j) + c));
    }
    sink(p, wanted, "p2_actual_overwritten_destination", directory);
}
