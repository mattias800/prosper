// #5108: unknown indirect group counts do not erase authentic complete local Wave64 entry.
// Original entry words need this local/control proof and their current producer participation;
// count/extent authority stays absent, so the proof cannot manufacture global bounds or work.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr uint32_t local_words =
    ComputeDispatchLaunchState::kInitiator | ComputeDispatchLaunchState::kNumThreadX |
    ComputeDispatchLaunchState::kNumThreadY | ComputeDispatchLaunchState::kNumThreadZ;
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t mov(uint32_t dst, uint32_t src, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | src;
}
ComputeShaderConfig canonical() {
    ComputeShaderConfig config;
    config.local_x = config.wave_size = config.native_subgroup_size = 64;
    config.local_y = config.local_z = 1;
    config.threads_x = config.threads_y = config.threads_z = 0;
    config.dispatch_launch.known = local_words;
    config.dispatch_launch.initiator = 1;
    config.dispatch_launch.num_thread = {64, 1, 1};
    return config;
}
ComputeShaderConfig direct() {
    auto config = canonical();
    prosper::test::complete_compute_launch(config);
    return config;
}
std::vector<uint32_t> program(bool asymmetric = false) {
    std::vector<uint32_t> words;
    if (asymmetric) words.push_back(0x7d8800a1u); // VCC selects local lanes 0..32
    words.insert(words.end(), {mov(20, asymmetric ? 106u : 126u, true), 0x7e020214u,
                               0x7e040215u, // both current physical words -> v1:v2
                               0xe0742000u, 0x80020100u, end}); // output[local x] = both words
    return words;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> result;
    rdna2_walk(words.data(), words.size(), result);
    return result;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    ComputeShaderConfig config;
    explicit Emitter(ComputeShaderConfig launch = canonical()) : config(std::move(launch)) {
        b.native_subgroup_size = config.native_subgroup_size;
        b.begin(4, nullptr, config.local_x, config.local_y, config.local_z, config.wave_size, 0,
                true, true);
        b.native_logical_uniform_launch = true;
        bind();
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.bfalse();
        rs.vreg[0] = b.localid_comp[0];
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
    }
    void bind() { prosper::test::bind_checked_compute_launch(b, config); }
    void prove(const std::vector<uint32_t>& words, uint32_t active = 0) {
        initialize_saved_mask_word_proof(b, decode(words), words.data(), words.size(), active, &rs);
    }
};
size_t ballots(const std::vector<uint32_t>& module) {
    size_t count = 0;
    for (size_t at = 5; at < module.size();) {
        const size_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return SIZE_MAX;
        count += (module[at] & 0xffffu) == 339u;
        at += width;
    }
    return count;
}
struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};
void retain_module(const std::vector<uint32_t>& module, uint64_t address) {
    // Test-only SOURCE observation; unset writes no files and changes no compilation or assertion.
    // NOLINTNEXTLINE(concurrency-mt-unsafe): configured before the single-threaded test runner starts.
    const char* directory = std::getenv("PROSPER_TEST_NATIVE_INDIRECT_SPV_DIRECTORY");
    if (!directory || !*directory || module.empty()) return;
    std::filesystem::create_directories(directory);
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    ASSERT_NE(test, nullptr);
    const auto path = std::filesystem::path(directory) /
                      (std::string(test->name()) + "-" + std::to_string(address) + ".spv");
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.good());
    out.write(reinterpret_cast<const char*>(module.data()),
              static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    ASSERT_TRUE(out.good());
}
Compilation compile(const ComputeShaderConfig& config, bool asymmetric = false) {
    static std::atomic<uint64_t> next{0x5108000000ull};
    const uint64_t address = next.fetch_add(0x100u, std::memory_order_relaxed);
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 2;
    output.binding = 3;
    output.stride = 8;
    output.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(output);
    const auto words = program(asymmetric);
    auto module = recompile_compute(words.data(), words.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    retain_module(module, address);
    return {std::move(module), last_terminal_reject_reason(address)};
}
void exact_values(const ComputeShaderConfig& config, bool asymmetric = false) {
    const auto result = compile(config, asymmetric);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(ballots(result.module), 2u) << "both physical words must come from current ballots";
    bpermute_oracle::Interpreter vm(result.module);
    // This synthetic invocation of one workgroup is independently supplied to the oracle. It is
    // not a reconstructed indirect count, an original SGPR value or proof of useful captured work.
    vm.native_subgroup_width = 64;
    const auto actual = vm.run(64, std::vector<uint32_t>(128, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> expected;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        expected.push_back(UINT32_MAX);
        expected.push_back(asymmetric ? 1u : UINT32_MAX);
    }
    EXPECT_EQ(actual, expected);
}
void refuses(const ComputeShaderConfig& config) {
    // Identical program and genuine canonical launch control. Pin the numeric-mask consumer so
    // a malformed unrelated instruction or resource cannot pass as a launch-authority negative.
    ASSERT_FALSE(compile(canonical()).module.empty());
    const auto result = compile(config);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("saved-mask"), std::string::npos) << result.reason;
}
void bare_refuses(const ComputeShaderConfig& config) {
    Emitter e;
    e.config = config;
    e.bind();
    EXPECT_FALSE(native_compute_has_complete_entry(e.b));
    e.prove(program());
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, decode(program())[0]));
}
}   // namespace

TEST(NativeIndirectEntry, AuthenticLocalWordsProveOnlyConditionalEntry) {
    Emitter e;
    const auto before = e.b.guest_compute_launch;
    ASSERT_TRUE(native_compute_has_complete_entry(e.b));
    EXPECT_FALSE(compute_dispatch_has_complete_groups(e.config));
    EXPECT_EQ(e.b.guest_compute_launch, before);
    EXPECT_EQ(e.config.dispatch_launch.known, 15u);
    EXPECT_EQ(e.config.dispatch_launch.dimensions, (std::array<uint32_t, 3>{}));
    EXPECT_EQ(e.b.guest_compute_launch.threads, (std::array<uint32_t, 3>{}));
    for (const auto local : {std::array{8u, 8u, 1u}, std::array{16u, 4u, 2u}}) {
        auto config = canonical();
        config.local_x = local[0];
        config.local_y = local[1];
        config.local_z = local[2];
        config.dispatch_launch.num_thread = local;
        Emitter shaped(config);
        EXPECT_TRUE(native_compute_has_complete_entry(shaped.b));
        EXPECT_FALSE(compute_dispatch_has_complete_groups(config));
    }
}
TEST(NativeIndirectEntry, OriginalPrefixRetainsItsExistingCodeAndProducerProof) {
    Emitter e;
    const auto words = program();
    e.prove(words);
    EXPECT_FALSE(e.b.native_compute_physical_entry_expired);
    EXPECT_TRUE(saved_mask_physical_launch_matches_original(e.b));
    EXPECT_TRUE(saved_mask_producer_has_full_participation(e.b, decode(words)[0]));
}
TEST(NativeIndirectEntry, PublicCompilerPreservesBothOriginalEntryWords) {
    exact_values(canonical());
}
TEST(NativeIndirectEntry, PublicCurrentCompareHasAsymmetricPhysicalWords) {
    exact_values(canonical(), true);
}
TEST(NativeIndirectEntry, CompleteDirectLaunchRetainsBothValueControls) {
    EXPECT_TRUE(compute_dispatch_has_complete_groups(direct()));
    exact_values(direct());
    exact_values(direct(), true);
}
TEST(NativeIndirectEntry, OrdinaryControlsKeepTheSameConditionalLocalEntry) {
    for (uint64_t flag : {4u, 8u, 16u, 64u, 1024u, 2048u}) {
        Emitter e;
        e.config.dispatch_launch.initiator |= flag;
        e.bind();
        EXPECT_TRUE(native_compute_has_complete_entry(e.b)) << flag;
        EXPECT_FALSE(compute_dispatch_has_complete_groups(e.config));
    }
}
TEST(NativeIndirectEntry, DisabledPartialFieldsDoNotChangeLocalExtent) {
    auto config = canonical();
    config.dispatch_launch.num_thread = {(31u << 16u) | 64u, (9u << 16u) | 1u, (7u << 16u) | 1u};
    Emitter e(config);
    EXPECT_TRUE(native_compute_has_complete_entry(e.b));
    exact_values(config);
}
TEST(NativeIndirectEntry, EachMissingControlOrLocalWordStaysRefused) {
    for (uint32_t bit : {1u, 2u, 4u, 8u}) {
        auto config = canonical();
        config.dispatch_launch.known &= ~bit;
        SCOPED_TRACE(bit);
        bare_refuses(config);
        refuses(config);
    }
}
TEST(NativeIndirectEntry, LegacyUnknownLaunchCannotAcquireEntry) {
    auto config = canonical();
    config.dispatch_launch = {};
    bare_refuses(config);
    refuses(config);
}
TEST(NativeIndirectEntry, PartialDimensionPresenceAndUnknownBitsStayRefused) {
    for (uint32_t bit : {16u, 32u, 64u, 128u}) {
        auto config = canonical();
        config.dispatch_launch.known |= bit;
        SCOPED_TRACE(bit);
        bare_refuses(config);
        refuses(config);
    }
}
TEST(NativeIndirectEntry, StaleAbsentDimensionStorageCannotBorrowEntry) {
    for (size_t axis = 0; axis < 3; ++axis) {
        auto config = canonical();
        config.dispatch_launch.dimensions[axis] = 1;
        SCOPED_TRACE(axis);
        bare_refuses(config);
        refuses(config);
    }
}
TEST(NativeIndirectEntry, NonzeroNormalizedExtentsStayOutsideTheUnknownCountRoute) {
    for (size_t axis = 0; axis < 3; ++axis) {
        auto config = canonical();
        if (axis == 0) config.threads_x = 64;
        if (axis == 1) config.threads_y = 1;
        if (axis == 2) config.threads_z = 1;
        SCOPED_TRACE(axis);
        bare_refuses(config);
        refuses(config);
    }
}
TEST(NativeIndirectEntry, MissingDimensionDirectProfilesRemainRefused) {
    for (uint32_t bit : {16u, 32u, 64u}) {
        auto config = direct();
        config.dispatch_launch.known &= ~bit;
        SCOPED_TRACE(bit);
        bare_refuses(config);
        refuses(config);
    }
}
TEST(NativeIndirectEntry, EveryFullDimensionMustBeNonzeroAndMatchLocalSize) {
    for (size_t axis = 0; axis < 3; ++axis)
        for (uint32_t value : {0u, 2u}) {
            auto config = canonical();
            config.dispatch_launch.num_thread[axis] = value;
            SCOPED_TRACE(axis);
            SCOPED_TRACE(value);
            bare_refuses(config);
            refuses(config);
        }
}
TEST(NativeIndirectEntry, PartialThreadTunnelRestoreWave32AndUpperControlsStayRefused) {
    for (uint64_t flag : {uint64_t{2}, uint64_t{32}, uint64_t{4096}, uint64_t{8192},
                          uint64_t{16384}, uint64_t{32768}, uint64_t{1} << 32u}) {
        auto config = canonical();
        config.dispatch_launch.initiator |= flag;
        SCOPED_TRACE(flag);
        bare_refuses(config);
        refuses(config);
    }
}
TEST(NativeIndirectEntry, ShaderEnableAndInexactModeMustBeAuthentic) {
    auto config = canonical();
    config.dispatch_launch.initiator = 0;
    bare_refuses(config);
    refuses(config);
    config = canonical();
    config.exact_thread_extent = true;
    bare_refuses(config);
    refuses(config);
}
TEST(NativeIndirectEntry, Native32AndPortableContextsCannotBorrowThisProof) {
    for (uint32_t width : {0u, 32u}) {
        Emitter e;
        e.b.native_subgroup_size = width;
        EXPECT_FALSE(native_compute_has_complete_entry(e.b));
        e.prove(program());
        EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, decode(program())[0]));
    }
    Emitter e;
    e.b.wave_size = e.b.guest_compute_launch.wave_size = 32;
    e.b.native_subgroup_size = 32;
    e.config.dispatch_launch.initiator |= 1u << 15u;
    e.bind();
    EXPECT_FALSE(native_compute_has_complete_entry(e.b));
}
TEST(NativeIndirectEntry, AComplete65WorkerGroupIsNotACompleteGuestWave) {
    Emitter e;
    e.config.local_x = e.config.dispatch_launch.num_thread[0] = 65;
    e.b.guest_local_size[0] = e.b.local_count = 65;
    e.bind();
    EXPECT_FALSE(native_compute_has_complete_entry(e.b));
    e.prove(program());
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, decode(program())[0]));
}
TEST(NativeIndirectEntry, LocalProductOverflowCannotMatchAWrappedBuilderCount) {
    Emitter e;
    e.config.local_x = e.config.local_y = e.config.local_z = 65535;
    e.config.dispatch_launch.num_thread = {65535, 65535, 65535};
    e.b.guest_local_size = {65535, 65535, 65535};
    e.b.local_count = 65535u * 65535u * 65535u;
    e.bind();
    EXPECT_FALSE(native_compute_has_complete_entry(e.b));
}
TEST(NativeIndirectEntry, GuestStagePhysicalDomainAndShellIdentityRemainNecessary) {
    for (uint32_t change = 0; change < 8; ++change) {
        Emitter e;
        if (change == 0) e.b.guest_stage = GuestShaderStage::Standalone;
        if (change == 1) e.b.physical_domain = PhysicalExecutionDomain::Invocation;
        if (change == 2) e.b.is_compute = false;
        if (change == 3) e.b.is_fragment = true;
        if (change == 4) e.b.is_vertex = true;
        if (change == 5) e.b.ngg_workgroup_shell = true;
        if (change == 6) e.b.ngg_one_lane = true;
        if (change == 7) e.b.guest_local_size = {32, 2, 1};
        SCOPED_TRACE(change);
        EXPECT_FALSE(native_compute_has_complete_entry(e.b));
    }
}
TEST(NativeIndirectEntry, NativeLocalCountAndGuestWaveMustAgreeWithObservedProfile) {
    Emitter e;
    ++e.b.local_count;
    EXPECT_FALSE(native_compute_has_complete_entry(e.b));
    e.b.local_count = 64;
    e.b.guest_compute_launch.wave_size = 32;
    EXPECT_FALSE(native_compute_has_complete_entry(e.b));
}
TEST(NativeIndirectEntry, ChangedProfileExpiresAuthorityEvenIfLocalExtentStillMatches) {
    Emitter e;
    const auto words = program();
    const auto in = decode(words)[0];
    e.prove(words);
    ASSERT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
    e.config.dispatch_launch.num_thread[0] |= 7u << 16u;
    e.bind();
    EXPECT_TRUE(native_compute_has_complete_entry(e.b));
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
    e.prove(words);
    EXPECT_TRUE(e.b.native_compute_physical_entry_expired);
    e.config = canonical();
    e.bind();
    e.prove(words);
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
}
TEST(NativeIndirectEntry, RewrittenOrSlicedCodeCannotReseedOriginalEntry) {
    for (bool sliced : {false, true}) {
        Emitter e;
        auto words = program();
        e.prove(words);
        const auto original = decode(words)[0];
        ASSERT_TRUE(saved_mask_producer_has_full_participation(e.b, original));
        if (sliced)
            words.erase(words.begin());
        else
            words[1] = 0x7e020215u;
        e.prove(words);
        EXPECT_TRUE(e.b.native_compute_physical_entry_expired);
        EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, original));
    }
}
TEST(NativeIndirectEntry, InitialActiveAndInvocationGuardsCannotGrantPhysicalWords) {
    for (bool guard : {false, true}) {
        Emitter e;
        if (guard) e.b.invocation_guard_merge = 1;
        e.prove(program(), guard ? 0u : e.b.btrue());
        EXPECT_TRUE(e.b.native_compute_physical_entry_expired);
        EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, decode(program())[0]));
    }
}
TEST(NativeIndirectEntry, MissingDataSourceStillCannotBeMadeGenuineByEntry) {
    const std::vector<uint32_t> words{mov(20, 22, true), 0x7e020214u, end};
    Emitter e;
    e.prove(words);
    EXPECT_TRUE(native_compute_has_complete_entry(e.b));
    EXPECT_FALSE(native_uniform_mov_entry_sources(&e.rs, decode(words)[0]));
}
}   // namespace prosper::gpu
