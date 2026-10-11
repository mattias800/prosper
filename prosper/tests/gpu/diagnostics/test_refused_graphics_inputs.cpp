// #4877: actual graphics compiler refusals retain their producing inputs. Bare code, later PM4
// state and synthetic descriptors cannot establish the live compiler decision. These CPU-only
// tests drive shipping cache/NGG boundaries and construct missing-source/choice negative arms.
#include "gpu/diagnostics/refused_graphics_inputs.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>

using namespace prosper::gpu;
namespace fs = std::filesystem;
namespace {
fs::path fresh(const char* name) {
    const auto root = prosper_test::test_scratch_dir() / name;
    std::error_code e;
    fs::remove_all(root, e);
    fs::create_directories(root, e);
    reset_refused_shader_dump_for_test(root.string());
    clear_shader_recompile_cache();
    clear_shader_analysis_cache();
    return root;
}
std::vector<uint32_t> vertex() {
    // A real partial POS0 export: the compiler names its unsupported EN=1 instead of inventing xyz/w.
    return {0x7e000280u, 0xf80008c1u, 0x00000000u, 0xbf810000u};
}
std::vector<uint32_t> pixel() {
    return {0x7e000280u, 0xbf810000u};
}   // actual no-export refusal
fs::path sidecar() {
    fs::path result;
    const auto directory = refused_shader_dump_directory();
    if (directory.empty()) return result;
    for (const auto& entry : fs::directory_iterator(directory))
        if (entry.path().extension() == ".inputs") {
            EXPECT_TRUE(result.empty());
            result = entry.path();
        }
    return result;
}
std::vector<uint32_t> dumped_words(const fs::path& path) {
    auto bin = path;
    bin.replace_extension(".bin");
    std::ifstream f(bin, std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(f), {}};
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}
RefusedShaderInputs produce(ShaderProgramStage stage, const std::vector<uint32_t>& code,
                            const ShaderResourceTable* table = nullptr) {
    PixelInputMapping pixel_inputs;
    pixel_inputs.valid_mask = 1;
    pixel_inputs.controls[0] = 3;
    pixel_inputs.consumed_known = true;
    pixel_inputs.consumed_mask = 1;
    PixelSystemInputMapping system{1, 1};
    auto module = recompile_graphics_shader_cached_shared(
        stage, code.data(), code.size(), table, &pixel_inputs,
        stage == ShaderProgramStage::Fragment ? &system : nullptr, nullptr,
        stage == ShaderProgramStage::Fragment, 128, true);
    EXPECT_TRUE(module && module->empty());
    const auto path = sidecar();
    EXPECT_FALSE(path.empty());
    if (path.empty()) return {};
    return read_refused_shader_inputs(path);
}
void expect_match(const RefusedShaderInputs& inputs) {
    ASSERT_TRUE(inputs.graphics);
    ASSERT_FALSE(inputs.live_reject.empty());
    const auto replay = replay_refused_shader_inputs(dumped_words(sidecar()), inputs);
    EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    EXPECT_FALSE(replay.produced);
    EXPECT_EQ(replay.reject, inputs.live_reject);
}
}   // namespace

TEST(RefusedGraphicsInputs, ShippingVertexCompilerKeepsActualSourceAndConfig) {
    fresh("refused_graphics_vertex");
    const auto code = vertex();
    const auto inputs = produce(ShaderProgramStage::Vertex, code);
    ASSERT_TRUE(inputs.graphics);
    EXPECT_EQ(inputs.stage, "vs");
    EXPECT_EQ(inputs.graphics->selector, RefusedGraphicsCompiler::Vertex);
    EXPECT_EQ(inputs.graphics->invocation.code, code);
    EXPECT_EQ(inputs.graphics->vertex_lds_dwords, 128u);
    EXPECT_TRUE(inputs.graphics->capture_position);
    EXPECT_TRUE(inputs.graphics->invocation.has_pixel_inputs);
    EXPECT_FALSE(inputs.graphics->launch.observed) << "an absent producing draw cannot be invented";
    expect_match(inputs);
}
TEST(RefusedGraphicsInputs, ShippingPixelCompilerKeepsActualMappingAndSpan) {
    fresh("refused_graphics_pixel");
    const auto code = pixel();
    const auto inputs = produce(ShaderProgramStage::Fragment, code);
    ASSERT_TRUE(inputs.graphics);
    EXPECT_EQ(inputs.stage, "ps");
    EXPECT_EQ(inputs.graphics->selector, RefusedGraphicsCompiler::Fragment);
    EXPECT_TRUE(inputs.graphics->invocation.wave32);
    EXPECT_TRUE(inputs.graphics->invocation.has_system_inputs);
    EXPECT_EQ(inputs.graphics->invocation.system_inputs.ena, 1u);
    EXPECT_EQ(inputs.graphics->invocation.pixel_inputs.controls[0], 3u);
    expect_match(inputs);
}
TEST(RefusedGraphicsInputs, ShippingChainCompilerRetainsBothSpansAndSelector) {
    fresh("refused_graphics_chain");
    const auto main = vertex();
    const std::vector<uint32_t> prolog{0xbfa00003u, 0xbe802006u};
    const auto module = recompile_vertex_chain_cached_shared(prolog.data(), prolog.size(),
                                                             main.data(), main.size());
    ASSERT_TRUE(module && module->empty());
    ASSERT_FALSE(sidecar().empty());
    const auto inputs = read_refused_shader_inputs(sidecar());
    ASSERT_TRUE(inputs.graphics);
    EXPECT_EQ(inputs.graphics->selector, RefusedGraphicsCompiler::VertexChain);
    EXPECT_EQ(inputs.graphics->invocation.code, prolog);
    EXPECT_EQ(inputs.graphics->chain, main);
    expect_match(inputs);
}
TEST(RefusedGraphicsInputs, ShippingNggShellRetainsActualLinkedCompilerSource) {
    fresh("refused_graphics_ngg");
    auto linked = vertex();
    linked.push_back(
        0x12345678u);   // outside the analysis' executable span, inside the actual call
    NggSubgroupDrawRequest request;
    request.linked_code = linked.data();
    request.dwords = linked.size();
    request.diagnostic = {RecompileDiagnosticStage::Vertex, uint64_t(uintptr_t(linked.data()))};
    request.limits.wave_lanes = 64;
    request.shell.user_sgprs = 24;
    request.shell.rsrc2_gs_lds_size = 2;
    std::string why;
    const auto result = compile_ngg_subgroup_stages(request, 0, &why);
    ASSERT_FALSE(result);
    ASSERT_FALSE(why.empty());
    ASSERT_FALSE(sidecar().empty());
    const auto inputs = read_refused_shader_inputs(sidecar());
    ASSERT_TRUE(inputs.graphics);
    EXPECT_EQ(inputs.graphics->selector, RefusedGraphicsCompiler::NggSubgroup);
    EXPECT_EQ(inputs.graphics->invocation.code, linked);
    EXPECT_EQ(inputs.graphics->shell.waves, 0u);
    EXPECT_TRUE(inputs.graphics->raw_is_source_prefix);
    EXPECT_EQ(inputs.graphics->raw, vertex());
    EXPECT_EQ(inputs.graphics->shell.user_sgprs, 24u);
    EXPECT_EQ(inputs.live_reject, why);
    expect_match(inputs);
}
TEST(RefusedGraphicsInputs, ProducingRegisterKnownnessDistinguishesZeroFromAbsent) {
    fresh("refused_graphics_registers");
    GpuState draw;
    namespace P = prosper::agc::Pm4;
    draw.sh[P::SPI_SHADER_USER_DATA_PS_0] = 0;
    draw.sh[P::SPI_SHADER_USER_DATA_PS_0 + 3] = 0x12345678;
    draw.sh[P::SPI_SHADER_PGM_RSRC2_PS] = 0;
    RefusedGraphicsLaunchScope scope(draw);
    const auto inputs = produce(ShaderProgramStage::Fragment, pixel());
    ASSERT_TRUE(inputs.graphics);
    EXPECT_TRUE(inputs.graphics->launch.observed);
    EXPECT_EQ(inputs.graphics->launch.available[2], 9u);
    EXPECT_EQ(inputs.graphics->launch.user_data[2][0], 0u);
    EXPECT_EQ(inputs.graphics->launch.user_data[2][3], 0x12345678u);
    EXPECT_EQ(inputs.graphics->launch.rsrc2_available, 4u);
    expect_match(inputs);   // Physical launch observations are not synthetic compiler inputs.
}
TEST(RefusedGraphicsInputs, MissingAndMismatchedSourceOrStageAreIncomplete) {
    fresh("refused_graphics_missing");
    auto inputs = produce(ShaderProgramStage::Vertex, vertex());
    ASSERT_TRUE(inputs.graphics);
    auto words = dumped_words(sidecar());
    auto wrong = words;
    wrong[0] ^= 1;
    EXPECT_NE(replay_refused_shader_inputs(wrong, inputs).incomplete.find("association"),
              std::string::npos);
    inputs.stage = "ps";
    EXPECT_NE(replay_refused_shader_inputs(words, inputs).incomplete.find("stage mismatch"),
              std::string::npos);
    inputs.stage = "vs";
    auto g = std::make_shared<RefusedGraphicsInputs>(*inputs.graphics);
    g->invocation.code.clear();
    inputs.graphics = g;
    EXPECT_FALSE(replay_refused_shader_inputs(words, inputs).incomplete.empty());
    g->invocation.code = words;
    g->selector = RefusedGraphicsCompiler::VertexChain;
    EXPECT_FALSE(replay_refused_shader_inputs(words, inputs).incomplete.empty());
}
TEST(RefusedGraphicsInputs, PrecompileDrawRefusalNeverInventsAnInvocation) {
    fresh("refused_graphics_precompile");
    const auto code = vertex();
    const auto analysis = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(analysis);
    // Pass original source through the shipping early-draw diagnostic; no compile has run.
    note_refused_draw_shaders({{},
                               {},
                               {},
                               uint64_t(uintptr_t(code.data())),
                               0,
                               0,
                               0,
                               code.size(),
                               0,
                               0,
                               0,
                               true,
                               false,
                               "owned-wave-entry-unavailable"});
    ASSERT_FALSE(sidecar().empty());
    const auto inputs = read_refused_shader_inputs(sidecar());
    ASSERT_TRUE(inputs.graphics);
    EXPECT_EQ(inputs.graphics->selector, RefusedGraphicsCompiler::Unavailable);
    EXPECT_FALSE(replay_refused_shader_inputs(dumped_words(sidecar()), inputs).incomplete.empty());
}
TEST(RefusedGraphicsInputs, OwnedResourceBytesAndMetadataSurviveLaterChanges) {
    fresh("refused_graphics_owned");
    ShaderResourceTable table;
    auto bytes = std::make_shared<std::vector<uint8_t>>(64, 0x31);
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.sgpr_base = 4;
    r.binding = 2;
    r.host_data = bytes->data();
    r.host_data_size = bytes->size();
    r.size = 64;
    r.format = DataFormat::Uint32;
    table.resources = {r};
    table.owned_host_data = {bytes};
    const auto inputs = produce(ShaderProgramStage::Vertex, vertex(), &table);
    ASSERT_TRUE(inputs.graphics);
    std::fill(bytes->begin(), bytes->end(), 0xee);
    table.resources[0].binding = 9;
    const auto& owned = inputs.graphics->invocation.resources.resources[0];
    ASSERT_NE(owned.host_data, nullptr);
    EXPECT_EQ(owned.host_data[0], 0x31);
    EXPECT_EQ(owned.binding, 2u);
    expect_match(inputs);
}
TEST(RefusedGraphicsInputs, PayloadBudgetOmitsBytesBeforeMakingAnOwnedCopy) {
    fresh("refused_graphics_payload");
    ShaderResourceTable table;
    auto bytes =
        std::make_shared<std::vector<uint8_t>>(kRefusedShaderInputsMaxPayloadBytes + 1, 0x42);
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.host_data = bytes->data();
    r.host_data_size = bytes->size();
    table.resources = {r};
    table.owned_host_data = {bytes};
    const auto inputs = produce(ShaderProgramStage::Vertex, vertex(), &table);
    ASSERT_TRUE(inputs.graphics);
    EXPECT_TRUE(inputs.payload_omitted);
    EXPECT_EQ(inputs.graphics->invocation.resources.resources[0].host_data, nullptr);
    for (const auto& blob : inputs.graphics->invocation.blobs) EXPECT_FALSE(blob.bytes);
    expect_match(
        inputs);   // No byte request: absence of irrelevant bytes is not automatically a failure.
    const auto& resource = inputs.graphics->invocation.resources.resources[0];
    CompilerResourceScope reads({{&resource, true, false, {}}});
    EXPECT_TRUE(compiler_resource_has_host_data(resource));
    EXPECT_THROW(compiler_resource_data(resource, 1), std::runtime_error)
        << "a requested omitted byte must never become a zero or a host-process read";
}
TEST(RefusedGraphicsInputs, ActualAbsentResourceTableIsACompleteArgument) {
    fresh("refused_graphics_null_table");
    auto code = vertex();
    code.insert(code.begin(), {0xf4200800u, 0xfa000000u});   // s_buffer_load_dword s32,s[0:3],0
    const auto inputs = produce(ShaderProgramStage::Vertex, code);
    ASSERT_TRUE(inputs.graphics);
    EXPECT_FALSE(inputs.graphics->invocation.has_resources);
    EXPECT_TRUE(inputs.incomplete.empty()) << inputs.incomplete;
    expect_match(inputs);
}

TEST(RefusedGraphicsInputs, OwnershipBudgetFailurePreservesActualCompilerRefusal) {
    fresh("refused_graphics_metadata_budget");
    ShaderResourceTable table;
    table.resources.resize(kCompileCaseMaxResources + 1);
    const auto inputs = produce(ShaderProgramStage::Vertex, vertex(), &table);
    ASSERT_TRUE(inputs.graphics);
    EXPECT_EQ(inputs.graphics->invocation.code, vertex());
    EXPECT_FALSE(inputs.live_reject.empty()) << "the original compiler must still run";
    EXPECT_EQ(inputs.incomplete, "compile-case resource count budget");
    const auto replay = replay_refused_shader_inputs(dumped_words(sidecar()), inputs);
    EXPECT_FALSE(replay.incomplete.empty());
    EXPECT_FALSE(replay.produced);
}

TEST(RefusedGraphicsInputs, ObserverNeverRetriesOrSwallowsTheActualCompilerException) {
    fresh("refused_graphics_compiler_exception");
    const auto code = vertex();
    const auto analysis = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(analysis);
    const RefusedShaderSource source{{analysis, &analysis->code}, &analysis->refused_shader_memo};
    RefusedGraphicsInputs g;
    g.selector = RefusedGraphicsCompiler::Vertex;
    g.invocation.code = code;
    unsigned calls = 0;
    EXPECT_THROW(observe_refused_graphics_compile("vs", source, std::move(g), nullptr,
                                                  [&calls]() -> std::vector<uint32_t> {
                                                      ++calls;
                                                      throw std::runtime_error("compiler failure");
                                                  }),
                 std::runtime_error);
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(sidecar().empty());
}
TEST(RefusedGraphicsInputs, GraphicsEnvelopeRejectsCorruptionAndTruncation) {
    fresh("refused_graphics_corrupt");
    produce(ShaderProgramStage::Fragment, pixel());
    std::ifstream f(sidecar(), std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f), {}};
    ASSERT_GT(bytes.size(), 24u);
    EXPECT_NO_THROW(decode_refused_shader_inputs(bytes));
    auto corrupt = bytes;
    corrupt[20] ^= 1;
    EXPECT_THROW(decode_refused_shader_inputs(corrupt), std::runtime_error);
    auto truncated = bytes;
    truncated.resize(bytes.size() / 2);
    EXPECT_THROW(decode_refused_shader_inputs(truncated), std::runtime_error);
}
TEST(RefusedGraphicsInputs, SetIdentityCoversArgumentsAndMetadataButNotBytes) {
    fresh("refused_graphics_identity");
    ShaderResourceTable table;
    auto bytes = std::make_shared<std::vector<uint8_t>>(4, 0x31);
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.host_data = bytes->data();
    r.host_data_size = 4;
    table.resources = {r};
    table.owned_host_data = {bytes};
    const auto inputs = produce(ShaderProgramStage::Vertex, vertex(), &table);
    ASSERT_TRUE(inputs.graphics);
    const auto& g = *inputs.graphics;
    const auto original = refused_graphics_inputs_identity(g);
    auto changed = g;
    changed.invocation.program_address += 0x1000;
    EXPECT_EQ(refused_graphics_inputs_identity(changed), original)
        << "relocating diagnostic-only code does not manufacture another input set";
    changed = g;
    changed.vertex_lds_dwords++;
    EXPECT_NE(refused_graphics_inputs_identity(changed), original);
    changed = g;
    changed.invocation.resources.resources[0].binding++;
    EXPECT_NE(refused_graphics_inputs_identity(changed), original);
    changed = g;
    changed.invocation.blobs[0].bytes = std::make_shared<std::vector<uint8_t>>(4, 0xee);
    restore_fragment_compile_case_backing(changed.invocation);
    EXPECT_EQ(refused_graphics_inputs_identity(changed), original);
}
TEST(RefusedGraphicsInputs, NestedProducingChoicesAreRecordedButReplayChecksAreExcluded) {
    CompilerChoiceTrace outer, inner;
    {
        CompilerChoiceScope a(outer);
        {
            CompilerChoiceScope b(inner);
            EXPECT_EQ(compiler_choice(CompilerChoice::FloatControls, [] { return int64_t(1); }), 1);
        }
        const CompilerChoiceTrace frozen = inner;
        {
            CompilerChoiceScope replay(frozen);
            EXPECT_EQ(compiler_choice(CompilerChoice::FloatControls, [] { return int64_t(0); }), 1);
            replay.finish_replay();
        }
    }
    ASSERT_EQ(outer.reads.size(), 1u);
    EXPECT_EQ(outer.reads, inner.reads);
}
TEST(RefusedGraphicsInputs, MissingSemanticChoiceTraceFailsClosed) {
    fresh("refused_graphics_choices");
    auto inputs = produce(ShaderProgramStage::Vertex, vertex());
    ASSERT_TRUE(inputs.graphics);
    auto g = std::make_shared<RefusedGraphicsInputs>(*inputs.graphics);
    // An extra unconsumed semantic choice is just as invalid as a missing one.
    g->invocation.choices.reads.push_back({CompilerChoice::ForcedArrayLayer, 0});
    inputs.graphics = g;
    EXPECT_FALSE(replay_refused_shader_inputs(dumped_words(sidecar()), inputs).incomplete.empty());
}
TEST(RefusedGraphicsInputs, WarmCacheDoesNotReadOrCopyDiagnosticInputs) {
    fresh("refused_graphics_warm");
    const auto code = vertex();
    const auto first = produce(ShaderProgramStage::Vertex, code);
    ASSERT_TRUE(first.graphics);
    const auto before = refused_shader_dump_stats();
    const auto cache_before = shader_recompile_cache_stats();
    PixelInputMapping mapping;
    mapping.valid_mask = 1;
    mapping.controls[0] = 3;
    mapping.consumed_known = true;
    mapping.consumed_mask = 1;
    for (int i = 0; i < 8; ++i)
        recompile_graphics_shader_cached_shared(ShaderProgramStage::Vertex, code.data(),
                                                code.size(), nullptr, &mapping, nullptr, nullptr,
                                                false, 128, true);
    const auto after = refused_shader_dump_stats();
    EXPECT_EQ(after.input_identity_evaluations, before.input_identity_evaluations);
    EXPECT_EQ(after.input_set_global_locks, before.input_set_global_locks);
    EXPECT_EQ(after.hash_evaluations, before.hash_evaluations);
    EXPECT_EQ(shader_recompile_cache_stats().hits - cache_before.hits, 8u);
    EXPECT_EQ(shader_recompile_cache_stats().misses, cache_before.misses);
}
TEST(RefusedGraphicsInputs, IdentityDistinguishesOpaquePresenceAndAliasing) {
    uint8_t a = 1, b = 2;
    ShaderResourceTable table;
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.host_data = &a;
    table.resources = {r, r};
    RefusedGraphicsInputs same;
    own_refused_graphics_resources(same, &table);
    table.resources[1].host_data = &b;
    RefusedGraphicsInputs separate;
    own_refused_graphics_resources(separate, &table);
    EXPECT_NE(refused_graphics_inputs_identity(same), refused_graphics_inputs_identity(separate));
    table.resources[1].host_data = nullptr;
    RefusedGraphicsInputs absent;
    own_refused_graphics_resources(absent, &table);
    // All replay pointers are null and all byte sizes are zero; only retained backing differs.
    EXPECT_EQ(separate.invocation.resources.resources[1].host_data, nullptr);
    EXPECT_EQ(absent.invocation.resources.resources[1].host_data, nullptr);
    EXPECT_NE(refused_graphics_inputs_identity(separate), refused_graphics_inputs_identity(absent));
}
TEST(RefusedGraphicsInputs, WriterReservesItsOwnTemporaryAndReportsInstallFailure) {
    const auto root = fresh("refused_graphics_writer");
    const auto inputs = produce(ShaderProgramStage::Vertex, vertex());
    ASSERT_TRUE(inputs.graphics);
    const auto path = root / "owned.inputs";
    const auto foreign = fs::path(path.string() + ".partial");
    {
        std::ofstream f(foreign);
        f << "foreign-marker";
    }
    EXPECT_TRUE(write_refused_graphics_inputs(path, "vs", *inputs.graphics));
    {
        std::ifstream f(foreign);
        std::string marker;
        f >> marker;
        EXPECT_EQ(marker, "foreign-marker");
    }
    EXPECT_NO_THROW(read_refused_shader_inputs(path));
    const auto directory = root / "occupied.inputs";
    fs::create_directory(directory);
    {
        std::ofstream f(directory / "marker");
        f << "original";
    }
    EXPECT_FALSE(write_refused_graphics_inputs(directory, "vs", *inputs.graphics));
    EXPECT_TRUE(fs::is_directory(directory));
    EXPECT_TRUE(fs::exists(directory / "marker"));
    for (const auto& entry : fs::directory_iterator(root))
        EXPECT_EQ(entry.path().filename().string().find(".partial-"), std::string::npos);
}
TEST(RefusedGraphicsInputs, OversizeOriginalAllocationIsOmittedBeforeOwnershipCopy) {
    ShaderResourceTable table;
    auto bytes = std::make_shared<std::vector<uint8_t>>(kCompileCaseMaxPayloadBytes + 1, 0x42);
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.host_data = bytes->data();
    r.host_data_size = bytes->size();
    table.resources = {r};
    table.owned_host_data = {bytes};
    RefusedGraphicsInputs g;
    EXPECT_NO_THROW(own_refused_graphics_resources(g, &table));
    EXPECT_TRUE(g.payload_omitted);
    EXPECT_TRUE(g.incomplete.empty()) << g.incomplete;
    ASSERT_EQ(g.invocation.blobs.size(), 1u);
    EXPECT_FALSE(g.invocation.blobs[0].bytes);
    EXPECT_EQ(g.invocation.blobs[0].size, bytes->size());
}
TEST(RefusedGraphicsInputs, MissingSidecarAndSaturationStopBeforeColdObservation) {
    fresh("refused_graphics_stop");
    const auto code = vertex();
    const auto first = produce(ShaderProgramStage::Vertex, code);
    ASSERT_TRUE(first.graphics);
    const auto analysis = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(analysis);
    const RefusedShaderSource source{{analysis, &analysis->code}, &analysis->refused_shader_memo};
    const uint64_t address = uint64_t(uintptr_t(code.data()));
    ASSERT_TRUE(refused_shader_inputs_needed("vs", source));
    for (size_t i = 0; i < kRefusedShaderMaxInputSetsCounted; ++i) {
        RefusedShaderInputSet set{i + 1, {}, {}};
        note_refused_shader_inputs("vs", address, source, "saturation control", set);
    }
    ASSERT_FALSE(refused_shader_inputs_needed("vs", source));
    const auto before = refused_shader_dump_stats();
    PixelInputMapping mapping;
    mapping.valid_mask = 1;
    mapping.controls[0] = 3;
    mapping.consumed_known = true;
    mapping.consumed_mask = 1;
    const auto trace = first.graphics->invocation.choices;
    {
        // A needless inner live observer would swallow these reads instead of consuming replay.
        CompilerChoiceScope replay(trace);
        const auto result = recompile_graphics_shader_cached_shared(
            ShaderProgramStage::Vertex, code.data(), code.size(), nullptr, &mapping, nullptr,
            nullptr, false, 129, true);
        ASSERT_TRUE(result && result->empty());
        EXPECT_NO_THROW(replay.finish_replay());
    }
    EXPECT_EQ(refused_shader_dump_stats().input_identity_evaluations,
              before.input_identity_evaluations);
    reset_refused_shader_dump_for_test(
        prosper_test::test_scratch_path("refused_graphics_no_sidecar").string());
    ASSERT_TRUE(note_refused_shader("vs", address, source, "no-sidecar control"));
    EXPECT_FALSE(refused_shader_inputs_needed("vs", source));
    {
        CompilerChoiceScope replay(trace);
        const auto result = recompile_graphics_shader_cached_shared(
            ShaderProgramStage::Vertex, code.data(), code.size(), nullptr, &mapping, nullptr,
            nullptr, false, 130, true);
        ASSERT_TRUE(result && result->empty());
        EXPECT_NO_THROW(replay.finish_replay());
    }
}
