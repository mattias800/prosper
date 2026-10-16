// Exact graphics launch modes must survive offline input capture. Old NGG sidecars can use
// their observed physical GS word; a native vertex selector alone cannot prove that bank.
#include "gpu/diagnostics/refused_graphics_inputs.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/graphics_float_mode.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <set>
#include <stdexcept>
#include <vector>

using namespace prosper::gpu;
namespace {
RefusedGraphicsInputs input(RefusedGraphicsCompiler selector) {
    RefusedGraphicsInputs g;
    g.selector = selector;
    g.raw = {0x7e000280u, 0xf80008c1u, 0u, 0xbf810000u};
    g.invocation.code = g.raw;
    g.invocation.expected_reject = "synthetic terminal";
    return g;
}
std::vector<uint8_t> encode(const RefusedGraphicsInputs& g) {
    const auto path = prosper_test::test_scratch_dir() / "graphics-float-inputs.inputs";
    EXPECT_TRUE(write_refused_graphics_inputs(path, "vs", g));
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
}
uint32_t schema(const std::vector<uint8_t>& bytes) {
    uint32_t result = 0;
    for (unsigned i = 0; i < 4; ++i) result |= uint32_t(bytes.at(8 + i)) << (8 * i);
    return result;
}
void checksum(std::vector<uint8_t>& bytes) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i + 8 < bytes.size(); ++i) hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    for (unsigned i = 0; i < 8; ++i) bytes.at(bytes.size() - 8 + i) = uint8_t(hash >> (8 * i));
}
std::vector<uint8_t> as_legacy(std::vector<uint8_t> bytes) {
    // Schema5's two three-byte carriers are append-only before the checksum.
    bytes.erase(bytes.end() - 14, bytes.end() - 8);
    bytes.at(8) = 3;
    checksum(bytes);
    return bytes;
}
}   // namespace
TEST(GraphicsFloatInputs, NewSchemaRetainsEveryCarrierAndProducingIdentity) {
    constexpr std::array modes{
        GraphicsFloatMode{}, GraphicsFloatMode{true, 0, false}, GraphicsFloatMode{true, 0, true},
        GraphicsFloatMode{true, 0xab, false}, GraphicsFloatMode{true, 0xab, true}};
    std::set<uint64_t> identities;
    for (const auto vertex : modes)
        for (const auto ngg : modes) {
            auto g = input(RefusedGraphicsCompiler::Vertex);
            g.vertex_float_mode = vertex;
            g.shell.float_mode = ngg;
        // Force the new schema even when both compiler inputs are explicitly unknown.
            g.launch.observed = true;
            g.launch.rsrc1_available = 1;
            g.launch.rsrc1[0] = 0x008ff000;
            const auto bytes = encode(g);
            EXPECT_EQ(schema(bytes), 5u);
            const auto decoded = decode_refused_shader_inputs(bytes);
            ASSERT_TRUE(decoded.graphics);
            EXPECT_EQ(decoded.graphics->vertex_float_mode, vertex);
            EXPECT_EQ(decoded.graphics->shell.float_mode, ngg);
            EXPECT_EQ(refused_graphics_inputs_identity(*decoded.graphics),
                      refused_graphics_inputs_identity(g));
            identities.insert(refused_graphics_inputs_identity(g));
        }
    EXPECT_EQ(identities.size(), modes.size() * modes.size());
}
TEST(GraphicsFloatInputs, LegacyGsWordIsAuthoritativeOnlyForNggAndOnlyWhenObserved) {
    for (const auto selector :
         {RefusedGraphicsCompiler::Vertex, RefusedGraphicsCompiler::VertexChain,
          RefusedGraphicsCompiler::NggSubgroup}) {
        for (bool observed : {false, true})
            for (uint32_t word : {0u, 0x008a5000u}) {
                auto g = input(selector);
                g.launch.observed = observed;
                g.launch.rsrc1_available = 1;
                g.launch.rsrc1[0] = word;
                const auto modern = encode(g);
                const auto current = decode_refused_shader_inputs(modern);
                ASSERT_TRUE(current.graphics);
                EXPECT_EQ(current.graphics->shell.float_mode, GraphicsFloatMode{});
                const auto decoded = decode_refused_shader_inputs(as_legacy(modern));
                ASSERT_TRUE(decoded.graphics);
                EXPECT_EQ(decoded.graphics->vertex_float_mode, GraphicsFloatMode{});
                const GraphicsFloatMode expected =
                    observed && selector == RefusedGraphicsCompiler::NggSubgroup
                        ? graphics_float_mode_from_rsrc1(word)
                        : GraphicsFloatMode{};
                EXPECT_EQ(decoded.graphics->shell.float_mode, expected);
            }
    }
    auto absent = input(RefusedGraphicsCompiler::NggSubgroup);
    absent.launch.observed = true;
    absent.launch.rsrc1[0] = 0x008a5000;
    const auto decoded = decode_refused_shader_inputs(encode(absent));
    ASSERT_TRUE(decoded.graphics);
    EXPECT_EQ(decoded.graphics->shell.float_mode, GraphicsFloatMode{});
}
TEST(GraphicsFloatInputs, InvalidAndTruncatedNewTailRefusesWithAValidChecksum) {
    auto g = input(RefusedGraphicsCompiler::NggSubgroup);
    g.shell.float_mode = {true, 0, false};
    const auto bytes = encode(g);
    ASSERT_EQ(schema(bytes), 5u);
    const size_t tail = bytes.size() - 14;
    for (size_t field : {tail, tail + 2, tail + 3, tail + 5}) {
        auto bad = bytes;
        bad.at(field) = 2;
        checksum(bad);
        EXPECT_THROW(decode_refused_shader_inputs(bad), std::runtime_error);
    }
    auto bad = bytes;
    bad.at(tail + 1) = 1; // unavailable native mode with nonzero payload
    checksum(bad);
    EXPECT_THROW(decode_refused_shader_inputs(bad), std::runtime_error);
    for (size_t end = tail; end < tail + 6; ++end) {
        bad = bytes;
        bad.erase(bad.begin() + ptrdiff_t(end), bad.end() - 8);
        checksum(bad);
        EXPECT_THROW(decode_refused_shader_inputs(bad), std::runtime_error);
    }
    g.vertex_float_mode = {false, 1, false};
    const auto path = prosper_test::test_scratch_dir() / "graphics-float-invalid.inputs";
    EXPECT_FALSE(write_refused_graphics_inputs(path, "vs", g));
}
