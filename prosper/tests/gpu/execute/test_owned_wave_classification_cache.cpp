// The immutable MAY classification must reuse the exact decoded original version, not repeat its
// dataflow on every draw or retain a stale answer after same-address code replacement. No GPU runs.
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);

#if defined(PROSPER_TEST_WRAP_OWNED_WAVE_CLASSIFIER)
static size_t classifier_calls = 0;
std::vector<uint32_t> real_classifier(const std::vector<Rdna2Inst>&) asm(
    "__real__ZN7prosper3gpu30rdna2_raw_wave_wide_data_loadsERKSt6vectorINS0_9Rdna2InstESaIS2_EE");
std::vector<uint32_t> wrap_classifier(const std::vector<Rdna2Inst>& original) asm(
    "__wrap__ZN7prosper3gpu30rdna2_raw_wave_wide_data_loadsERKSt6vectorINS0_9Rdna2InstESaIS2_EE");
std::vector<uint32_t> wrap_classifier(const std::vector<Rdna2Inst>& original) {
    ++classifier_calls;
    return real_classifier(original);
}
#endif

namespace {
#if defined(PROSPER_TEST_OWNED_WAVE_CACHE_BYPASS)
#define CACHE_SUITE OwnedWaveClassificationUncached
constexpr bool kBypass = true;
#elif defined(PROSPER_TEST_OWNED_WAVE_CLASSIFICATION_BYPASS)
#define CACHE_SUITE OwnedWaveClassificationOnlyUncached
constexpr bool kBypass = false;
#else
#define CACHE_SUITE OwnedWaveClassificationCache
constexpr bool kBypass = false;
#endif
#if defined(PROSPER_TEST_OWNED_WAVE_CLASSIFICATION_BYPASS)
constexpr bool kMayBypass = true;
#else
constexpr bool kMayBypass = false;
#endif
class DecodeCacheScope {
public:
    DecodeCacheScope() {
        EXPECT_EQ(std::getenv("PROSPER_NO_SHADER_DECODE_CACHE") != nullptr, kBypass)
            << "cache-off coverage requires a separate process before the frozen flag is read";
        EXPECT_EQ(std::getenv("PROSPER_NO_OWNED_WAVE_CLASSIFICATION_CACHE") != nullptr, kMayBypass);
        clear_shader_decode_cache();
    }
    ~DecodeCacheScope() { clear_shader_decode_cache(); }
};

struct Program {
    // Project-owned READFIRST selector followed by a raw numeric x4 load; no descriptor or data
    // address is dereferenced by this MAY query. Header/code owners survive registry retention.
    alignas(256) std::array<uint32_t, 8> code{0x7e040500u, 0x8f148402u, 0xf4080600u, 0x28000000u,
                                              0x7e00021bu, 0xbf810000u, 0xbf800000u, 0xbf800000u};
    AgcShaderHeader header{};
    std::array<uint32_t, 4> sh_registers{prosper::agc::Pm4::SPI_SHADER_PGM_LO_PS, 0,
                                         prosper::agc::Pm4::SPI_SHADER_PGM_HI_PS, 0};
    uint64_t address() const { return reinterpret_cast<uint64_t>(code.data()); }
};

Program* registered_program() {
    static std::vector<std::unique_ptr<Program>> owners;
    auto program = std::make_unique<Program>();
    program->header.file_header = 0x34333231u;
    program->header.version = 0x18u;
    program->header.shader_size = sizeof(program->code);
    program->header.type = 1u;
    // SDK pointer fields are self-relative below 4 GiB, regardless of the host allocation address.
    const auto register_offset = reinterpret_cast<uintptr_t>(program->sh_registers.data()) -
                                 reinterpret_cast<uintptr_t>(&program->header.sh_registers);
    if (!register_offset || register_offset >= 0x100000000ull) return nullptr;
    program->header.sh_registers = reinterpret_cast<const void*>(register_offset);
    program->header.num_sh_registers = 2;
    register_builtin_hle();
    const auto create = Hle::lookup("f3dg2CSgRKY");
    if (!create) return nullptr;
    void* registered = nullptr;
    if (create(reinterpret_cast<uint64_t>(&registered),
               reinterpret_cast<uint64_t>(&program->header), program->address(), 0, 0, 0) != 0 ||
        registered != &program->header || program->header.code != program->code.data() ||
        program->header.sh_registers != program->sh_registers.data() ||
        program->sh_registers[1] != uint32_t(program->address() >> 8) ||
        program->sh_registers[3] != uint32_t((program->address() >> 40) & 0xffu) ||
        prosper_agc_shader_header_for_code(program->address()) != &program->header)
        return nullptr;
    owners.push_back(std::move(program));
    return owners.back().get();
}

bool classify_original(const Program& program) {
    const auto original = registered_graphics_original(program.address());
    if (!original) return false;
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(original->data(), original->size(), decoded);
    return !rdna2_raw_wave_wide_data_loads(decoded).empty();
}
}   // namespace

TEST(CACHE_SUITE, HotQueriesReuseClassification) {
    DecodeCacheScope scope;
    auto* program = registered_program();
    ASSERT_NE(program, nullptr);
#if defined(PROSPER_TEST_WRAP_OWNED_WAVE_CLASSIFIER)
    const auto before = classifier_calls;
#endif
    ASSERT_TRUE(graphics_program_requires_owned_waves(program->address()));
    const auto cold = shader_decode_cache_stats();
    EXPECT_EQ(cold.misses, kBypass ? 0u : 1u);
    EXPECT_EQ(cold.bypasses, kBypass ? 1u : 0u);
    for (int query = 0; query < 8; ++query)
        EXPECT_TRUE(graphics_program_requires_owned_waves(program->address()));
    EXPECT_EQ(shader_decode_cache_stats().hits, cold.hits + (kBypass ? 0u : 8u));
#if defined(PROSPER_TEST_WRAP_OWNED_WAVE_CLASSIFIER)
    EXPECT_EQ(classifier_calls - before, (kBypass || kMayBypass) ? 9u : 1u)
        << "eight hot MAY queries must not rerun the out-of-line dataflow classifier";
#endif
}

TEST(CACHE_SUITE, SameAddressReplacementInvalidatesMayFact) {
    DecodeCacheScope scope;
    auto* program = registered_program();
    ASSERT_NE(program, nullptr);
    ASSERT_TRUE(graphics_program_requires_owned_waves(program->address()));
    const uint32_t original = program->code[0];
    program->code[0] = 0xbf810000u;   // genuine empty prefix, not a different allocation
    EXPECT_FALSE(graphics_program_requires_owned_waves(program->address()));
    program->code[0] = original;
    EXPECT_TRUE(graphics_program_requires_owned_waves(program->address()));
    EXPECT_EQ(shader_decode_cache_stats().invalidations, kBypass ? 0u : 2u);
}

TEST(CACHE_SUITE, ConsumedPrefixKeepsTerminatorAndUnknownBoundary) {
    DecodeCacheScope scope;
    auto* program = registered_program();
    ASSERT_NE(program, nullptr);
    ASSERT_TRUE(graphics_program_requires_owned_waves(program->address()));
    program->code[6] = 0xffffffffu;   // beyond the original END, never a consumed classifier input
    EXPECT_TRUE(graphics_program_requires_owned_waves(program->address()));
    EXPECT_EQ(shader_decode_cache_stats().invalidations, 0u);
    program->code[0] = 0xffffffffu;   // now the consumed prefix itself is unknown
    EXPECT_EQ(graphics_program_requires_owned_waves(program->address()),
              classify_original(*program));
    EXPECT_FALSE(graphics_program_requires_owned_waves(program->address()));
    program->code[0] = 0x7e040500u;
    program->header.shader_size = 3u * sizeof(uint32_t);   // incomplete two-word SMEM at the bound
    clear_shader_decode_cache();
    EXPECT_EQ(graphics_program_requires_owned_waves(program->address()),
              classify_original(*program));
    bool may = true;
    EXPECT_FALSE(registered_graphics_original(0, &may));
    EXPECT_FALSE(may);
    program->header.shader_size = 0;
    may = true;
    EXPECT_FALSE(registered_graphics_original(program->address(), &may));
    EXPECT_FALSE(may);
}

TEST(CACHE_SUITE, BypassMatchesOriginalAndReanalyzesEachQuery) {
    DecodeCacheScope scope;
    auto* program = registered_program();
    ASSERT_NE(program, nullptr);
    const bool expected = classify_original(*program);
    ASSERT_TRUE(expected) << "the hand-built positive must actually require owned waves";
#if defined(PROSPER_TEST_WRAP_OWNED_WAVE_CLASSIFIER)
    const auto before = classifier_calls;
#endif
    for (int query = 0; query < 3; ++query)
        EXPECT_EQ(graphics_program_requires_owned_waves(program->address()), expected);
    EXPECT_EQ(shader_decode_cache_stats().bypasses, kBypass ? 4u : 0u);
#if defined(PROSPER_TEST_WRAP_OWNED_WAVE_CLASSIFIER)
    EXPECT_EQ(classifier_calls - before, (kBypass || kMayBypass) ? 3u : 0u);
#endif
    const auto read_source = registered_graphics_read_source(program->address());
    ASSERT_TRUE(read_source.words);
    ASSERT_TRUE(read_source.chains);
    EXPECT_FALSE(read_source.words.owner_before(read_source.chains));
    EXPECT_FALSE(read_source.chains.owner_before(read_source.words));
    EXPECT_EQ(read_source.requires_owned_waves, !kMayBypass);
    bool may = false;
    const auto original = registered_graphics_original(program->address(), &may);
    ASSERT_TRUE(original);
    EXPECT_EQ(*original, *read_source.words);
    EXPECT_EQ(may, read_source.requires_owned_waves);
    if (!kBypass) {
        EXPECT_FALSE(original.owner_before(read_source.words));
        EXPECT_FALSE(read_source.words.owner_before(original));
    }
    const auto absent_source = registered_graphics_read_source(0);
    EXPECT_FALSE(absent_source.words);
    EXPECT_FALSE(absent_source.chains);
    EXPECT_FALSE(absent_source.requires_owned_waves);
    program->code[0] = 0xbf810000u;
    EXPECT_FALSE(graphics_program_requires_owned_waves(program->address()));
    EXPECT_EQ(graphics_program_requires_owned_waves(program->address()),
              classify_original(*program));
}
