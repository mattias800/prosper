// The ORIGINAL guest SMEM/P1/P2/ADD/MUL/explicit sample/EXP chain must consume owned inputs.
// A compiler admission or a success status alone is not the oracle: evaluate actual typed SOURCE
// and compare every raw sink, then use the production transactional decoder. No guest raster ABI.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_resource_packet_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_resource_packet;
int checks = 0, failures = 0;
void check(bool value, const std::string& name) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "[FAIL] %s\n", name.c_str()); }
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
                    const std::string& name, const std::filesystem::path& directory, uint32_t lane = 0) {
    const auto program = compile(p);
    check(!program.packet.spirv.empty(), name + " runtime obligation emitted: " + program.packet.rejection);
    if (program.packet.spirv.empty()) return;
    std::string error; const auto words = evaluate(program, error);
    check(error.empty(), name + " completed typed all-worker execution: " + error);
    const auto r = decode_fragment_resource_packet(program, words, true, p.device.device_identity);
    check(r.exports.empty() && !r.rejection.empty() && r.failure == reason && r.lane == lane,
          name + " whole packet refused with deterministic lane/first reason");
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

int main(int argc, char** argv) {
    std::filesystem::path directory;
    if (argc == 3 && std::string(argv[1]) == "--dump-directory") {
        directory = argv[2]; std::filesystem::create_directories(directory);
    } else if (argc != 1) return 2;
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
        bad = words; bad[program.status_offset + 63 * 3 + 1] = 0x12345678; bad[program.status_offset + 63 * 3 + 2] = 3;
        check(decode_fragment_resource_packet(program, bad, true, p.device.device_identity).rejection == "packet-status-record-invalid", name + " invalid guest-PC refuses");
        // A valid typed arithmetic omission must alter the executed SOURCE, not just the oracle's
        // expected array. The first live uint64 multiply carries real nonzero P10 or numeric data.
        auto mutant = program.packet.spirv;
        uint32_t wide = 0, boolean = 0; size_t site = 0;
        for (size_t pc = 5; pc < mutant.size();) {
            const auto n = mutant[pc] >> 16, op = mutant[pc] & 0xffff;
            if (!n || n > mutant.size() - pc) break;
            if (op == 21 && n == 4 && mutant[pc + 2] == 64) wide = mutant[pc + 1];
            if (op == 20 && n == 2) boolean = mutant[pc + 1];
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
    }
    // Independent hand-derived IEEE bit rails; integer SOURCE does not inherit device FTZ or RTE.
    for (uint32_t denorm = 0; denorm < 4; ++denorm) for (uint32_t round = 0; round < 4; ++round) {
        const auto mode = static_cast<uint8_t>((denorm << 4) | round);
        const auto suffix = std::to_string(mode);
        sink(fixture::arithmetic(3, 0x3f800000, 0x33800000, mode),
             arithmetic_expected(round == 1 ? 0x3f800001 : 0x3f800000), "halfway_positive_" + suffix, directory);
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
    }
    runtime_reject(fixture::arithmetic(8, 0x7f7fffff, 0x40000000, 0x30), FragmentPacketRuntimeFailure::FiniteOverflow, "finite_overflow", directory);
    runtime_reject(fixture::arithmetic(3, 0x7fc00001, 0x3f800000, 0x30), FragmentPacketRuntimeFailure::NonFinite, "nonfinite_no_fake_canonical_nan", directory);
    auto p = fixture::chain();
    p.invocation.vgprs[0].words[63] = 0x7fc00001;
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "lane63_whole_packet_rejection", directory, 63);
    p = fixture::chain(); p.invocation.sgprs.back().second = 0; // actual S# high word mismatch
    // sampler word3 is already zero; mutate the actual first S# word instead.
    for (auto& [reg, value] : p.invocation.sgprs) if (reg == 12) value ^= 1;
    runtime_reject(p, FragmentPacketRuntimeFailure::DescriptorMismatch, "actual_sampler_identity", directory);
    p = fixture::chain(); for (auto& [reg, value] : p.invocation.sgprs) if (reg == 16) value ^= 1;
    runtime_reject(p, FragmentPacketRuntimeFailure::M0Mismatch, "actual_original_m0_writer", directory);
    p = fixture::chain(); for (auto& t : p.parameter_cache.parameters) { t.p0 = 0x3f800000; t.p10 = 0x33800000; }
    p.invocation.vgprs[0].words.fill(0x3f800000);
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "interpolation_halfway_not_exact", directory);
    p = fixture::chain(); for (auto& t : p.parameter_cache.parameters) t.p10 = 0x00800000;
    p.invocation.vgprs[0].words.fill(0x3f000000);
    runtime_reject(p, FragmentPacketRuntimeFailure::InterpolationNotExact, "interpolation_underflow_not_normal", directory);
    p = fixture::chain(); p.invocation.guest_code[3] = 0xbf800000;
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
    p = fixture::chain(); p.invocation.guest_code[p.images[0].pc] ^= (0x27u ^ 0x20u) << 18;
    reject(p, "packet-image-derivative-or-effect-unimplemented", "implicit SAMPLE not replaced with LOD0");
    std::printf("fragment_resource_packet checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
