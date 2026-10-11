// #4901: graphics must preserve a genuine folded empty scalar-buffer proof as an exact-PC
// constant-zero resource, rather than refusing a load whose architectural result needs no backing.
// All fixtures use actual AGC registration, folding, stage materialization and stage compilation.
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/empty_scalar_buffer.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "fixtures/graphics_empty_sbuffer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::graphics_empty_sbuffer;
struct StageFixture {
    AgcShaderHeader header{};
    std::vector<uint32_t> words;
    std::array<uint32_t, 20> descriptor_table{};
    std::vector<uint32_t> load_pcs;
    uint64_t registration_result = UINT64_MAX;
    const void* registered = nullptr;
    bool pixel = true;
};

StageFixture& make_stage(bool pixel, uint32_t opcode = 0x8u, uint32_t stride = 0u,
                         uint32_t records = 0u, uint32_t offset = 4u, bool missing_word = false,
                         bool keyed_pair = false, uint32_t soffset = 125u,
                         uint32_t initial_soffset = UINT32_MAX) {
    // The AGC registry retains header/code pointers. Keep each synthetic source alive and distinct
    // even when a developer executes the whole GoogleTest binary in one process.
    static std::vector<std::unique_ptr<StageFixture>> sources;
    auto owned = std::make_unique<StageFixture>();
    auto& source = *owned;
    source.pixel = pixel;
    source.words.reserve(40u);
    if (keyed_pair) {
        source.descriptor_table[17] = stride << 16u;
        source.descriptor_table[18] = records;
        const auto address = reinterpret_cast<uint64_t>(source.descriptor_table.data());
        fixture::scalar_literal(source.words, 32u, static_cast<uint32_t>(address));
        fixture::scalar_literal(source.words, 33u, static_cast<uint32_t>(address >> 32u));
        source.words.push_back(0xf4080610u);   // s_load_dwordx4 s[24:27],s[32:33],0x40
        source.words.push_back(0xfa000040u);
    } else {
        source.words = fixture::program(pixel, opcode, stride, records, offset, missing_word,
                                        soffset, initial_soffset);
        source.load_pcs.push_back((missing_word ? 6u : 8u) +
                                  (initial_soffset != UINT32_MAX ? 2u : 0u));
    }
    const auto add_load = [&](uint32_t destination) {
        source.load_pcs.push_back(static_cast<uint32_t>(source.words.size()));
        source.words.push_back(0xf4000000u | (opcode << 18u) | (destination << 6u) | 12u);
        source.words.push_back((soffset << 25u) | offset);
    };
    if (keyed_pair) {
        add_load(0u);
        add_load(4u);
        source.words.push_back(0x7e000200u);   // v_mov_b32 v0,s0
        source.words.push_back(0x7e020204u);   // v_mov_b32 v1,s4 (observe the second load too)
        source.words.push_back(pixel ? 0xf800080fu : 0xf80008cfu);
        source.words.push_back(0x00000100u);
        source.words.push_back(0xbf810000u);
    }
    source.header.file_header = 0x34333231u;
    source.header.version = 0x18u;
    source.header.shader_size = static_cast<uint32_t>(source.words.size() * sizeof(uint32_t));
    source.header.type = pixel ? 1u : 2u;
    prosper::register_agc_hle();
    const auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
    if (create_shader) {
        source.registration_result =
            create_shader(reinterpret_cast<uint64_t>(&source.registered),
                          reinterpret_cast<uint64_t>(&source.header),
                          reinterpret_cast<uint64_t>(source.words.data()), 0u, 0u, 0u);
    }
    sources.push_back(std::move(owned));
    return source;
}

std::vector<SrtUse> folded_uses(const StageFixture& source) {
    std::vector<SrtUse> uses;
    resolve_dynamic_fetch(source.words.data(), source.words.size(), nullptr, 0u, 0u, &uses);
    return uses;
}

std::shared_ptr<ShaderResourceTable> stage_table(const StageFixture& source) {
    return build_stage_table(GpuState{}, reinterpret_cast<uint64_t>(source.words.data()),
                             source.pixel, 1u);
}

std::vector<uint32_t> compile(const StageFixture& source, const ShaderResourceTable* table) {
    const auto address = reinterpret_cast<uint64_t>(source.words.data());
    return source.pixel ? recompile_fragment(source.words.data(), source.words.size(), table,
                                             nullptr, UINT32_MAX, nullptr, false,
                                             {RecompileDiagnosticStage::Fragment, address})
                        : recompile_vertex(source.words.data(), source.words.size(), table, nullptr,
                                           false, 0u, {RecompileDiagnosticStage::Vertex, address});
}

void expect_zero_stage(bool pixel, uint32_t opcode, uint32_t stride = 0u, uint32_t offset = 4u) {
    const auto& source = make_stage(pixel, opcode, stride, 0u, offset);
    ASSERT_EQ(source.registration_result, 0u);
    ASSERT_EQ(source.registered, &source.header);
    const auto uses = folded_uses(source);
    ASSERT_EQ(uses.size(), 1u);
    ASSERT_TRUE(uses[0].zero_record_raw) << "All four live V# words and the offset prove zero";
    EXPECT_EQ(uses[0].required_size, 0u);
    const auto table = stage_table(source);
    ASSERT_NE(table, nullptr);
    const auto* marker = table->by_fetch_pc(source.load_pcs[0]);
    ASSERT_NE(marker, nullptr) << "The actual stage builder must retain the fold's PC proof";
    EXPECT_TRUE(is_zero_record_raw_buffer(*marker));
    EXPECT_EQ(marker->srt_offset, UINT32_MAX);
    EXPECT_EQ(marker->sgpr_base, UINT32_MAX);
    const auto module = compile(source, table.get());
    ASSERT_FALSE(module.empty());
    const auto reflected = validate_spirv_descriptor_interface(
        module, table.get(), pixel ? 1u : 0u,
        pixel ? SpirvShaderStage::Fragment : SpirvShaderStage::Vertex, false);
    EXPECT_TRUE(reflected.ok());
    EXPECT_TRUE(reflected.descriptors.empty()) << "An architectural zero needs no dummy SSBO";
}

void expect_no_zero(const StageFixture& source) {
    ASSERT_EQ(source.registration_result, 0u);
    ASSERT_EQ(source.registered, &source.header);
    const auto uses = folded_uses(source);
    EXPECT_FALSE(std::any_of(uses.begin(), uses.end(),
                             [](const SrtUse& use) { return use.zero_record_raw; }));
    const auto table = stage_table(source);
    if (table) {
        EXPECT_FALSE(std::any_of(table->resources.begin(), table->resources.end(),
                                 is_zero_record_raw_buffer));
    }
    const ShaderResourceTable empty;
    EXPECT_TRUE(compile(source, table ? table.get() : &empty).empty())
        << "Unavailable backing must not gain constant-zero authority";
    const auto reason =
        last_terminal_reject_reason(reinterpret_cast<uint64_t>(source.words.data()));
    EXPECT_NE(reason.find("pc=" + std::to_string(source.load_pcs[0]) + " "), std::string::npos)
        << "The negative must reach the actual scalar consumer: " << reason;
    EXPECT_NE(reason.find("op=0x8"), std::string::npos) << reason;
}

TEST(GraphicsEmptyScalarBuffer, PixelDword) {
    expect_zero_stage(true, 0x8u);
}
TEST(GraphicsEmptyScalarBuffer, PixelDwordX2) {
    expect_zero_stage(true, 0x9u);
}
TEST(GraphicsEmptyScalarBuffer, PixelDwordX4) {
    expect_zero_stage(true, 0xau);
}
TEST(GraphicsEmptyScalarBuffer, PixelDwordX8) {
    expect_zero_stage(true, 0xbu);
}
TEST(GraphicsEmptyScalarBuffer, PixelDwordX16) {
    expect_zero_stage(true, 0xcu);
}
TEST(GraphicsEmptyScalarBuffer, VertexDword) {
    expect_zero_stage(false, 0x8u);
}
TEST(GraphicsEmptyScalarBuffer, VertexDwordX2) {
    expect_zero_stage(false, 0x9u);
}
TEST(GraphicsEmptyScalarBuffer, VertexDwordX4) {
    expect_zero_stage(false, 0xau);
}
TEST(GraphicsEmptyScalarBuffer, VertexDwordX8) {
    expect_zero_stage(false, 0xbu);
}
TEST(GraphicsEmptyScalarBuffer, VertexDwordX16) {
    expect_zero_stage(false, 0xcu);
}

TEST(GraphicsEmptyScalarBuffer, EmptyStridedDescriptorProvesZeroAtOffsetZero) {
    for (const bool pixel : {false, true}) expect_zero_stage(pixel, 0x8u, 16u, 0u);
}

TEST(GraphicsEmptyScalarBuffer, KeyedEmptyDescriptorPublishesBothConsumerPcs) {
    for (const bool pixel : {false, true}) {
        const auto& source = make_stage(pixel, 0x8u, 0u, 0u, 4u, false, true);
        ASSERT_EQ(source.registration_result, 0u);
        const auto uses = folded_uses(source);
        ASSERT_EQ(uses.size(), 2u);
        ASSERT_EQ(uses[0].key, 0x40u);
        ASSERT_EQ(uses[1].key, 0x40u);
        ASSERT_TRUE(uses[0].zero_record_raw && uses[1].zero_record_raw);
        const auto table = stage_table(source);
        ASSERT_NE(table, nullptr);
        for (const uint32_t pc : source.load_pcs) {
            const auto* marker = table->by_fetch_pc(pc);
            ASSERT_NE(marker, nullptr);
            EXPECT_TRUE(is_zero_record_raw_buffer(*marker));
            EXPECT_EQ(marker->srt_offset, UINT32_MAX);
        }
        EXPECT_FALSE(compile(source, table.get()).empty());
    }
}

TEST(GraphicsEmptyScalarBuffer, UnknownSoffsetKeepsTheExistingMinimumOffsetProof) {
    const auto& source = make_stage(true, 0x8u, 0u, 0u, 4u, false, false, 56u);
    const auto uses = folded_uses(source);
    ASSERT_EQ(uses.size(), 1u);
    ASSERT_TRUE(uses[0].zero_record_raw);
    EXPECT_FALSE(uses[0].scalar_oob_offset_known);
    const auto table = stage_table(source);
    ASSERT_NE(table, nullptr);
    const auto* marker = table->by_fetch_pc(source.load_pcs[0]);
    ASSERT_NE(marker, nullptr);
    EXPECT_TRUE(is_zero_record_raw_buffer(*marker));
    EXPECT_FALSE(compile(source, table.get()).empty());
}

TEST(GraphicsEmptyScalarBuffer, KnownSoffsetCarriesItsExactOobProof) {
    for (const bool pixel : {false, true}) {
        const auto& source = make_stage(pixel, 0x8u, 0u, 0u, 0u, false, false, 56u, 4u);
        const auto uses = folded_uses(source);
        ASSERT_EQ(uses.size(), 1u);
        ASSERT_TRUE(uses[0].zero_record_raw);
        ASSERT_TRUE(uses[0].scalar_oob_offset_known);
        ASSERT_EQ(uses[0].scalar_oob_byte_offset, 4u);
        const auto table = stage_table(source);
        ASSERT_NE(table, nullptr);
        const auto* marker = table->by_fetch_pc(source.load_pcs[0]);
        ASSERT_NE(marker, nullptr);
        EXPECT_TRUE(is_zero_record_raw_buffer(*marker));
        EXPECT_FALSE(compile(source, table.get()).empty());
        auto bad = uses[0];
        bad.scalar_oob_byte_offset = 0u;
        EXPECT_FALSE(empty_scalar_buffer_resource(bad, source.words.data(), source.words.size()));
        bad = uses[0];
        bad.scalar_oob_offset_known = false;
        EXPECT_FALSE(empty_scalar_buffer_resource(bad, source.words.data(), source.words.size()));
    }
}

TEST(GraphicsEmptyScalarBuffer, EmptyStrideZeroOffsetZeroIsNotAWholeAccessProof) {
    for (const bool pixel : {false, true}) expect_no_zero(make_stage(pixel, 0x8u, 0u, 0u, 0u));
}

TEST(GraphicsEmptyScalarBuffer, NonemptyLowBaseNeverManufacturesZero) {
    for (const bool pixel : {false, true}) expect_no_zero(make_stage(pixel, 0x8u, 0u, 16u, 4u));
}

TEST(GraphicsEmptyScalarBuffer, MissingLiveDescriptorWordNeverManufacturesZero) {
    for (const bool pixel : {false, true}) {
        // The paired complete descriptor uses the same s[64:67], outside either entry seed.
        const auto& complete = make_stage(pixel);
        const auto known = folded_uses(complete);
        ASSERT_EQ(known.size(), 1u);
        ASSERT_TRUE(known[0].zero_record_raw);
        expect_no_zero(make_stage(pixel, 0x8u, 0u, 0u, 4u, true));
    }
}

TEST(GraphicsEmptyScalarBuffer, MaterializerRechecksObservationAndConsumer) {
    const auto& source = make_stage(true);
    const auto uses = folded_uses(source);
    ASSERT_EQ(uses.size(), 1u);
    ASSERT_TRUE(empty_scalar_buffer_resource(uses[0], source.words.data(), source.words.size()));
    auto bad = uses[0];
    bad.zero_record_raw = false;
    EXPECT_FALSE(empty_scalar_buffer_resource(bad, source.words.data(), source.words.size()));
    bad = uses[0];
    bad.v4[2] = 16u;
    EXPECT_FALSE(empty_scalar_buffer_resource(bad, source.words.data(), source.words.size()));
    bad = uses[0];
    bad.required_size = 4u;
    EXPECT_FALSE(empty_scalar_buffer_resource(bad, source.words.data(), source.words.size()));
    bad = uses[0];
    bad.use_pc = 0u;   // S_MOV does not supply a scalar-buffer marker's authority.
    EXPECT_FALSE(empty_scalar_buffer_resource(bad, source.words.data(), source.words.size()));
    EXPECT_FALSE(
        empty_scalar_buffer_resource(uses[0], source.words.data(), source.load_pcs[0] + 1u));
}
}   // namespace
}   // namespace prosper::gpu
