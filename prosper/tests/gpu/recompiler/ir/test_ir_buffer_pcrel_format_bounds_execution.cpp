// PC-relative formatted loads use instruction-wide bounds; raw dword-x4 retains per-word bounds.
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {

constexpr uint32_t kTable[] = {0x11223344u, 0x22334455u, 0x33445566u, 0x44556677u};

bool save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    return out.good();
}

std::filesystem::path artifacts() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable.
    if (const char* configured = std::getenv("PROSPER_PCREL_FORMAT_BOUNDS_ARTIFACTS"))
        return configured;
    return prosper_test::test_scratch_dir() / "pcrel-format-bounds";
}

LaneProgram program(bool typed, uint32_t wave) {
    const uint32_t load = typed ? 0xea6b1000u : 0xe0381000u;
    LaneProgram p = lane_program({load, 0x80050102u, 0xbf8c3f70u}, wave, wave);
    p.code.resize(p.code.size() - 3);   // replace the output store and END after the EXEC restore
    for (uint32_t component = 0; component < 4; ++component)
        p.code.insert(p.code.end(), {0xe0702000u | (component * wave * 4),
                                     0x80020000u | ((1u + component) << 8)});
    p.code.push_back(0xbf810000u);

    // s_getpc_b64 + constant address build; the V# and four-word table are embedded after END.
    const std::vector<uint32_t> prefix = {0xbe941f00u, 0x801414ffu, 0u,         0x82151580u,
                                          0xbe960390u, 0xbe9703ffu, 0x31005facu};
    p.code.insert(p.code.begin(), prefix.begin(), prefix.end());
    p.code[2] = static_cast<uint32_t>(p.code.size() * sizeof(uint32_t)) - 4u;
    p.code.insert(p.code.end(), std::begin(kTable), std::end(kTable));
    p.config.native_subgroup_size = wave;
    // PC-relative typed entries are MTBUF, whose format selectors are instruction-defined identity;
    // the current PC-relative MUBUF entries are raw dword loads. This fixture does not exercise a
    // descriptor SEL_1 exception, which is not available on either of these routes.
    return p;
}

void check(bool typed, bool past_end, uint32_t arm) {
    const std::filesystem::path root = artifacts();
    std::filesystem::create_directories(root);
    for (const uint32_t wave : {32u, 64u}) {
        SCOPED_TRACE(::testing::Message()
                     << "typed=" << typed << " past_end=" << past_end << " wave=" << wave);
        LaneProgram p = program(typed, wave);
        const Rdna2Inst in = rdna2_decode_one(p.code.data() + 17, p.code.size() - 17);
        ASSERT_EQ(in.fmt, typed ? Rdna2Format::MTBUF : Rdna2Format::MUBUF);
        if (typed) ASSERT_EQ(in.mtbuf_format, 77u);
        const uint64_t address = 0x5383000000ull + arm * 0x1000 + wave * 0x10;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, address});
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);

        std::vector<uint32_t> buffer;
        for (uint32_t array = 0; array < 4; ++array)
            for (uint32_t lane = 0; lane < wave; ++lane)
                buffer.push_back(array == 0   ? (past_end ? 12u : 0u)
                                 : array == 1 ? 0x72000000u + lane
                                 : array == 2 ? 0xd0000000u + lane
                                              : 1u);
        std::vector<uint32_t> expected;
        for (uint32_t component = 0; component < 4; ++component)
            for (uint32_t lane = 0; lane < wave; ++lane)
                expected.push_back(typed && past_end ? 0u
                                   : past_end        ? (component == 0 ? kTable[3] : 0u)
                                                     : kTable[component]);

        const std::string name = std::string(typed ? "typed-" : "raw-") +
                                 (past_end ? "tail-" : "inrange-") + std::to_string(wave);
        const auto stem = root / name;
        ASSERT_TRUE(save(stem.string() + ".bin", p.code));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", address, &p.table,
                                                 p.config, "synthetic witness (no live refusal)"));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_TRUE(replay.produced) << replay.reject << " incomplete=" << replay.incomplete;
        ASSERT_EQ(replay.spirv, module);
        ASSERT_TRUE(save(stem.string() + ".spv", module));

        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = wave;
        const auto actual = vm.run_buffers(wave, {{3, buffer}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_TRUE(save(stem.string() + ".buffer", buffer));
        ASSERT_TRUE(save(stem.string() + ".actual", actual));
        ASSERT_TRUE(save(stem.string() + ".expected", expected));
        EXPECT_EQ(actual, expected) << "component-wide bounds contract";
    }
}

// Each case uses a real four-register load, including registers that overlap the harness inputs.
// Snapshot the expected old words before the instruction so inactive lanes must retain all of them.
void check_lane_matrix(bool typed, uint32_t components, uint32_t format, uint32_t table_words,
                       uint32_t wave, bool partial, uint32_t selectors) {
    auto p = program(typed, wave);
    p.code[4] = 0xbe960380u | (table_words * 4u);   // s_mov_b32 s22, byte bound
    p.code[6] = (p.code[6] & ~0xfffu) | selectors;
    constexpr uint32_t raw_opcodes[] = {0xcu, 0xdu, 0xfu, 0xeu};   // x3 sorts after x4
    p.code[17] = typed ? 0xe8001000u | ((components - 1u) << 16u) | (format << 19u)
                       : 0xe0001000u | (raw_opcodes[components - 1u] << 18u);
    const auto instruction = rdna2_decode_one(p.code.data() + 17, p.code.size() - 17);
    ASSERT_EQ(instruction.fmt, typed ? Rdna2Format::MTBUF : Rdna2Format::MUBUF);
    EXPECT_EQ(instruction.opcode, typed ? components - 1u : raw_opcodes[components - 1u]);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    ASSERT_FALSE(module.empty());
    const uint32_t offsets[] = {0u, 4u, 8u, 12u, 16u, 20u, 0xfffffffcu, 0xfffffff0u};
    std::vector<uint32_t> buffer, expected;
    for (uint32_t array = 0; array < 4; ++array)
        for (uint32_t lane = 0; lane < wave; ++lane)
            buffer.push_back(array == 0                      ? offsets[lane % std::size(offsets)]
                             : array == 1                    ? 0x72000000u + lane
                             : array == 2                    ? 0xd0000000u + lane
                             : (!partial || lane % 3u != 0u) ? 1u
                                                             : 0u);
    for (uint32_t component = 0; component < 4; ++component)
        for (uint32_t lane = 0; lane < wave; ++lane) {
            // v1=old_d; v2=address; v3=S1; v4=active, all loaded before EXEC narrows.
            const uint32_t old_array[] = {2u, 0u, 1u, 3u};
            uint32_t value = buffer[old_array[component] * wave + lane];
            if (buffer[3u * wave + lane] && component < components) {
                const uint32_t first = buffer[lane] / 4u;
                const bool whole_fits =
                    components <= table_words && first <= table_words - components;
                value = (typed ? whole_fits : first + component < table_words)
                            ? kTable[first + component]
                            : 0u;
            }
            expected.push_back(value);
        }
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = wave;
    const auto got = vm.run_buffers(wave, {{3, buffer}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(got, expected);
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable.
    const char* directory = std::getenv("PROSPER_IR_WITNESS_DIR");
    if (directory && !partial) {
        const auto dump = std::filesystem::path(directory) / "refused_shaders_pcrel_format_bounds";
        std::filesystem::create_directories(dump);
        const uint32_t key = (uint32_t(typed) << 20u) | (components << 16u) | (format << 8u) |
                             (table_words << 4u) | uint32_t(wave == 64u);
        const uint64_t address = 0x5d53830000ull + uint64_t(key) * 0x100u;
        const auto stem = dump / ("cs_" + std::to_string(address) + "_" + std::to_string(key));
        ASSERT_TRUE(save(stem.string() + ".bin", p.code));
        ASSERT_TRUE(save(stem.string() + ".spv", module));
        ASSERT_TRUE(save(stem.string() + ".buffer", buffer));
        ASSERT_TRUE(save(stem.string() + ".expected", expected));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", address, &p.table,
                                                 p.config, "synthetic PC-relative bounds witness"));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        ASSERT_TRUE(replay.produced) << replay.reject;
        ASSERT_EQ(replay.spirv, module);
    }
}

TEST(IrBufferPcrelFormatBoundsExecution, MixedLaneBoundsRetainRawComponentsAndInactiveWords) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t n = 1; n <= 4; ++n)
            for (uint32_t bound = 1; bound <= 4; ++bound)
                for (bool partial : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << wave << ':' << n << ':' << bound << ':' << partial);
                    check_lane_matrix(false, n, 0, bound, wave, partial, 0xfacu);
                }
}
TEST(IrBufferPcrelFormatBoundsExecution, EveryAdmittedFormatUsesWholeBoundsAndIdentitySelectors) {
    // The twelve format anchors are the already-admitted 32-bit uint/sint/float forms.
    const uint32_t formats[4][3] = {{20, 21, 22}, {62, 63, 64}, {72, 73, 74}, {75, 76, 77}};
    for (uint32_t wave : {32u, 64u})
        for (uint32_t n = 1; n <= 4; ++n)
            for (const uint32_t format : formats[n - 1])
                for (uint32_t bound = 1; bound <= 4; ++bound)
                    for (bool partial : {false, true}) {
                        SCOPED_TRACE(::testing::Message() << wave << ':' << n << ':' << format
                                                          << ':' << bound << ':' << partial);
                        // All four resource selectors SEL_1: MTBUF must ignore these (ISA Table 31).
                        check_lane_matrix(true, n, format, bound, wave, partial, 0x249u);
                    }
}

TEST(IrBufferPcrelFormatBoundsExecution, RawInRangeControl) {
    check(false, false, 0);
}
TEST(IrBufferPcrelFormatBoundsExecution, RawTailKeepsPerComponentBounds) {
    check(false, true, 1);
}
TEST(IrBufferPcrelFormatBoundsExecution, FormattedInRangeControl) {
    check(true, false, 2);
}
TEST(IrBufferPcrelFormatBoundsExecution, FormattedTailZerosTheWholeLoad) {
    check(true, true, 3);
}

}   // namespace
