// #4292: exercise default evidence at the real draw/dispatch refusal boundaries. The first-byte
// and address stay unchanged on rewrites; deleting/moving the live producer call must lose evidence.
// Project-owned ISA only, no device/guest boot. Byte oracles read actual dump files, not cache hashes.
#include "fixtures/test_scratch.h"
#include "gpu/execute/compute_program_facts.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
namespace fs = std::filesystem;

namespace {

class RefusedShaderProducer : public testing::Test {
protected:
    void SetUp() override {
        const fs::path root = prosper_test::test_scratch_dir() / "refused-producer";
        std::error_code error;
        fs::create_directories(root, error);
        ASSERT_FALSE(error) << error.message();
        reset_refused_shader_dump_for_test(root.string());
        clear_shader_recompile_cache();
        clear_shader_analysis_cache();
        reset_compute_program_facts_for_test();
    }
};

std::vector<std::vector<uint32_t>> recorded_words() {
    std::vector<std::vector<uint32_t>> result;
    const fs::path directory = refused_shader_dump_directory();
    if (directory.empty()) return result;
    for (const auto& file : fs::directory_iterator(directory)) {
        if (file.path().extension() != ".bin") continue;
        std::ifstream input(file.path(), std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
        EXPECT_EQ(bytes.size() % sizeof(uint32_t), 0u);
        std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
        if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
        result.push_back(std::move(words));
    }
    return result;
}

// The constructor relocates small SDK pointers from the pointer field. A contiguous retained
// blob works in low-address non-PIE executables as well as high-address hosts.
struct ComputeShaderBlob {
    AgcShaderHeader header;
    ShaderReg registers[2];
};

bool register_compute(const uint32_t* words, size_t dwords, ComputeShaderBlob& blob) {
    auto& header = blob.header;
    auto& registers = blob.registers;
    prosper::register_agc_hle();
    const auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
    if (!create_shader) return false;
    registers[0] = {P::COMPUTE_PGM_LO, 0};
    registers[1] = {P::COMPUTE_PGM_HI, 0};
    header = {};
    header.file_header = 0x34333231u;
    header.version = 0x18;
    header.sh_registers = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(registers) - reinterpret_cast<uintptr_t>(&header.sh_registers));
    header.shader_size = static_cast<uint32_t>(dwords * sizeof(uint32_t));
    header.num_sh_registers = 2;
    void* registered = nullptr;
    return create_shader(reinterpret_cast<uint64_t>(&registered),
                         reinterpret_cast<uint64_t>(&header), reinterpret_cast<uint64_t>(words), 0,
                         0, 0) == 0 &&
           registered == &header;
}

GpuState compute_state(const ShaderReg (&registers)[2], uint32_t threads) {
    GpuState state;
    for (const auto& reg : registers) state.sh[reg.offset] = reg.value;
    state.sh[P::COMPUTE_NUM_THREAD_X] = 64;
    state.sh[P::COMPUTE_NUM_THREAD_Y] = state.sh[P::COMPUTE_NUM_THREAD_Z] = 1;
    GpuState::Dispatch dispatch;
    dispatch.threads_x = threads;
    dispatch.threads_y = dispatch.threads_z = 1;
    dispatch.modifier = 1ull << P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT;
    dispatch.command_order = 4292;
    dispatch.state = std::make_shared<GpuState>(state);
    state.dispatches.push_back(std::move(dispatch));
    return state;
}

void expect_compute_refusal(const GpuState& state, uint64_t address) {
    std::vector<OperationRealizationFailure> failures;
    const auto items = realize_compute_dispatches(state, 4292, &failures);
    EXPECT_TRUE(items.empty());
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures[0].reason, RealizationFailureReason::ShaderRecompile);
    ASSERT_EQ(failures[0].stages.size(), 1u);
    EXPECT_EQ(failures[0].stages[0].program_addr, address);
}

alignas(256) constexpr uint32_t vertex_words[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
    0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};

GpuState graphics_state(const uint32_t* fragment) {
    GpuState state;
    const auto program = [&](uint32_t lo, uint32_t hi, const uint32_t* words) {
        const auto address = reinterpret_cast<uint64_t>(words);
        state.sh[lo] = static_cast<uint32_t>(address >> 8);
        state.sh[hi] = static_cast<uint32_t>((address >> 40) & 0xffu);
    };
    program(P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, vertex_words);
    program(P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, fragment);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = 0xf;
    state.cx[P::CB_COLOR_CONTROL] = P::CB_COLOR_CONTROL_MODE_NORMAL
                                    << P::CB_COLOR_CONTROL_MODE_SHIFT;
    return state;
}

}   // namespace

TEST_F(RefusedShaderProducer, ComputeRewriteKeepsFullOriginalAndWarmWorkBounded) {
    // A crossing branch makes this partial-workgroup barrier impossible to lift. The registered
    // original includes a post-END tail: evidence must retain it, not the specialized/decoded prefix.
    alignas(256) static uint32_t code[] = {
        0xbf060000u, 0xbf840002u, 0xbf8a0000u, 0x7e040282u,
        0x7e040281u, 0xbf810000u, 0x11223344u, 0x55667788u,
    };
    static ComputeShaderBlob blob;
    code[3] = 0x7e040282u;
    ASSERT_TRUE(register_compute(code, std::size(code), blob));
    ASSERT_EQ(blob.header.sh_registers, blob.registers);
    ASSERT_EQ(blob.header.code, code);
    ASSERT_EQ(prosper_agc_shader_header_for_code(reinterpret_cast<uint64_t>(code)), &blob.header);
    ASSERT_EQ(blob.registers[0].value,
              static_cast<uint32_t>(reinterpret_cast<uint64_t>(code) >> 8));
    ASSERT_EQ(blob.registers[1].value,
              static_cast<uint32_t>((reinterpret_cast<uint64_t>(code) >> 40) & 0xffu));
    const auto state = compute_state(blob.registers, 65);
    const auto address = reinterpret_cast<uint64_t>(code);
    const std::vector<uint32_t> before(std::begin(code), std::end(code));
    expect_compute_refusal(state, address);
    ASSERT_EQ(recorded_words(), (std::vector<std::vector<uint32_t>>{before}));
    for (size_t i = 0; i != 16; ++i) expect_compute_refusal(state, address);
    const auto warm = refused_shader_dump_stats();
    EXPECT_EQ(warm.content_records, 1u);
    EXPECT_EQ(warm.hash_evaluations, 1u);
    EXPECT_EQ(warm.hashed_dwords, std::size(code));
    const uint32_t first_word = code[0];
    code[3] = 0x7e040283u;
    ASSERT_EQ(code[0], first_word);
    const std::vector<uint32_t> after(std::begin(code), std::end(code));
    expect_compute_refusal(state, address);
    const auto files = recorded_words();
    ASSERT_EQ(files.size(), 2u);
    EXPECT_NE(std::find(files.begin(), files.end(), before), files.end());
    EXPECT_NE(std::find(files.begin(), files.end(), after), files.end());
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 2u)
        << "the old once-per-address log gate must not hide a new original version";
    EXPECT_EQ(refused_shader_dump_stats().hashed_dwords, 2u * std::size(code));
}

TEST_F(RefusedShaderProducer, SuccessfulComputeCreatesNoRefusalEvidence) {
    alignas(256) static const uint32_t code[] = {0x7e000280u, 0xbf810000u};
    static ComputeShaderBlob blob;
    ASSERT_TRUE(register_compute(code, std::size(code), blob));
    ASSERT_EQ(blob.header.sh_registers, blob.registers);
    ASSERT_EQ(blob.header.code, code);
    ASSERT_EQ(prosper_agc_shader_header_for_code(reinterpret_cast<uint64_t>(code)), &blob.header);
    ASSERT_EQ(blob.registers[0].value,
              static_cast<uint32_t>(reinterpret_cast<uint64_t>(code) >> 8));
    ASSERT_EQ(blob.registers[1].value,
              static_cast<uint32_t>((reinterpret_cast<uint64_t>(code) >> 40) & 0xffu));
    std::vector<OperationRealizationFailure> failures;
    const auto items =
        realize_compute_dispatches(compute_state(blob.registers, 64), 4292, &failures);
    EXPECT_TRUE(failures.empty());
    ASSERT_EQ(items.size(), 1u);
    EXPECT_FALSE(items[0].spirv.empty());
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 0u);
    EXPECT_TRUE(refused_shader_dump_directory().empty());
}

TEST_F(RefusedShaderProducer, GraphicsRewriteKeepsOnlyRefusedStageAndSuccessfulNeighbor) {
    // S_BARRIER is not admitted in a normal fragment stage. Both byte versions must still refuse.
    alignas(256) uint32_t fragment[std::size(vertex_words)] = {
        0x7e000280u,
        0x7e020282u,
        0xbf8a0000u,
        0xbf810000u,
    };
    auto state = graphics_state(fragment);
    const std::vector<uint32_t> before(fragment, fragment + 4);
    GpuState::Draw packet;
    packet.index_count = 3;
    packet.command_order = 4292;
    for (bool shared : {false, true, false, true}) {
        DrawItem item;
        OperationRealizationFailure failure;
        ASSERT_FALSE(realize_draw_item(state, &packet, 3, std::size(vertex_words), false, item,
                                       &failure, shared));
        EXPECT_EQ(failure.reason, RealizationFailureReason::ShaderRecompile);
        ASSERT_EQ(failure.stages.size(), 2u);
        EXPECT_TRUE(failure.stages[0].recompiled);
        EXPECT_FALSE(failure.stages[1].recompiled);
        EXPECT_EQ(failure.stages[1].program_addr, reinterpret_cast<uint64_t>(fragment));
    }
    ASSERT_EQ(recorded_words(), (std::vector<std::vector<uint32_t>>{before}));
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 1u);
    fragment[1] = 0x7e020283u;   // same address/first word, changed later original instruction
    const std::vector<uint32_t> after(fragment, fragment + 4);
    DrawItem item;
    ASSERT_FALSE(realize_draw_item(state, &packet, 3, std::size(vertex_words), false, item));
    const auto files = recorded_words();
    ASSERT_EQ(files.size(), 2u);
    EXPECT_NE(std::find(files.begin(), files.end(), before), files.end());
    EXPECT_NE(std::find(files.begin(), files.end(), after), files.end());
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 2u);
    constexpr uint32_t valid_fragment[] = {
        0x7e000280u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    std::copy(std::begin(valid_fragment), std::end(valid_fragment), fragment);
    EXPECT_TRUE(realize_draw_item(state, &packet, 3, std::size(vertex_words), false, item));
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 2u);
    EXPECT_EQ(recorded_words().size(), 2u) << "a successful rewritten neighbor is never evidence";
}

TEST_F(RefusedShaderProducer, OwnedWaveGateRefusalKeepsTheProgramAndItsReason) {
    // #4555. A fragment program with a numeric raw-wide load and a v_readfirstlane is routed to
    // the owned-wave path, and that path's gate can refuse the draw before any recompile is
    // attempted. The drop is counted as shader-recompile/fragment, but nothing used to keep the
    // program: GTA V lost ~2,500 draws a run to one such program and left no copy of it.
    //   0  v_mov_b32 v1, 0
    //   1  s_load_dwordx4 s[16:19], s[28:29], 0xf0     numeric: s18 is read by a v_mov below
    //   3  s_buffer_load_dwordx4 s[8:11], s[16:19], 0xc0
    //   5  v_mov_b32 v0, s18
    //   6  v_cmp_*_sdwa s[16:17], 0, s10 ; s_mov_b64 vcc, s[16:17] ; s_cbranch_vccz +1 ; s_branch 1
    //  11  v_readfirstlane vcc_lo, v1 ; s_endpgm
    alignas(256) static const uint32_t fragment[] = {
        0x7e020280u, 0xf408040eu, 0xfa0000f0u, 0xf4280208u, 0xfa0000c0u, 0x7e000212u, 0x7c1a14f9u,
        0x86869080u, 0xbeea0410u, 0xbf860001u, 0xbf82fff6u, 0x7ed40501u, 0xbf810000u,
    };
    {
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(fragment, std::size(fragment), decoded), std::size(fragment));
        ASSERT_FALSE(rdna2_raw_wave_wide_data_loads(decoded).empty())
            << "the fixture must be a program the classifier routes to owned waves";
    }
    static ComputeShaderBlob blob;   // two SH registers and a header: the PS pair fits as well
    {
        prosper::register_agc_hle();
        const auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
        ASSERT_TRUE(create_shader);
        blob.registers[0] = {P::SPI_SHADER_PGM_LO_PS, 0};
        blob.registers[1] = {P::SPI_SHADER_PGM_HI_PS, 0};
        blob.header = {};
        blob.header.file_header = 0x34333231u;
        blob.header.version = 0x18;
        blob.header.sh_registers =
            reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(blob.registers) -
                                          reinterpret_cast<uintptr_t>(&blob.header.sh_registers));
        blob.header.shader_size = static_cast<uint32_t>(sizeof fragment);
        blob.header.num_sh_registers = 2;
        void* registered = nullptr;
        ASSERT_EQ(create_shader(reinterpret_cast<uint64_t>(&registered),
                                reinterpret_cast<uint64_t>(&blob.header),
                                reinterpret_cast<uint64_t>(fragment), 0, 0, 0),
                  0u);
        ASSERT_EQ(registered, &blob.header);
    }
    const auto address = reinterpret_cast<uint64_t>(fragment);
    ASSERT_TRUE(graphics_program_requires_owned_waves(address));
    // No fragment launch state is set, so the gate refuses on its first register check.
    auto state = graphics_state(fragment);
    GpuState::Draw packet;
    packet.index_count = 3;
    packet.command_order = 4555;
    for (int repeat = 0; repeat < 3; ++repeat) {
        DrawItem item;
        OperationRealizationFailure failure;
        ASSERT_FALSE(
            realize_draw_item(state, &packet, 3, std::size(vertex_words), false, item, &failure));
        EXPECT_EQ(failure.reason, RealizationFailureReason::ShaderRecompile);
    }
    const std::vector<uint32_t> expected(std::begin(fragment), std::end(fragment));
    ASSERT_EQ(recorded_words(), (std::vector<std::vector<uint32_t>>{expected}))
        << "one copy of the refused fragment program, however many draws it lost";
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 1u);
    std::ifstream index(fs::path(refused_shader_dump_directory()) / "index.txt");
    const std::string lines((std::istreambuf_iterator<char>(index)), {});
    EXPECT_NE(lines.find("ps addr=0x"), std::string::npos) << lines;
    EXPECT_NE(lines.find("refusal=draw-wave-known-fragment64-launch-unavailable"),
              std::string::npos)
        << lines;
}

TEST_F(RefusedShaderProducer, PreKeyGraphicsGuardsStillKeepOriginalEvidence) {
    const uint32_t code[] = {0x7e000280u, 0x7e020282u, 0xbf810000u};
    RefusedShaderSource original;
    EXPECT_FALSE(recompile_graphics_shader_cached_shared(
        ShaderProgramStage::Fragment, code, std::size(code), nullptr, nullptr, nullptr, nullptr,
        false, 0, false, {}, FragmentFloatMode{false, 1}, {}, {}, {}, &original));
    EXPECT_FALSE(original.words) << "canonical-profile guard precedes creation of a compile key";
    const auto address = reinterpret_cast<uint64_t>(code);
    const std::vector<uint32_t> expected(std::begin(code), std::end(code));
    note_refused_draw_shaders(
        {{}, original, {}, 0, address, 0, 4292, std::size(code), 0, 0, 0, false, true});
    ASSERT_EQ(recorded_words(), (std::vector<std::vector<uint32_t>>{expected}));
    RefusedShaderSource chain;
    const FloatTransportConfig invalid{static_cast<FloatTransportProfile>(0xff)};
    EXPECT_FALSE(recompile_vertex_chain_cached_shared(code, std::size(code), code, std::size(code),
                                                      nullptr, nullptr, nullptr, 0, false, invalid,
                                                      &chain));
    EXPECT_FALSE(chain.words);
    note_refused_draw_shaders(
        {chain, {}, {}, address, 0, address, 4293, std::size(code), 0, 0, 0, true, false});
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 2u);
    EXPECT_EQ(recorded_words().size(), 2u) << "the same source refused at VS and PS is distinct";
}
