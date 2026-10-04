// Real registered DrawItems exercise whole64 execution and normal attachment commit. Two waves
// deliberately select different current numeric words; an unavailable second wave must never
// publish the first. These synthetic offscreen tests make no title or console packing claim.
#include <gtest/gtest.h>
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "shared/live/live_renderer.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace prosper;
using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
constexpr uint64_t Page = 0x10000u;
constexpr uint32_t W = 16u, H = 8u;
constexpr uint32_t Fullscreen[]{0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u,
                                0x7e0a02f6u, 0x7e0c02f2u, 0x10020b01u, 0x08020d01u,
                                0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u};
struct Program {
    alignas(256) std::array<uint32_t, 128> code{};
    std::array<ShaderReg, 2> registers{};
    AgcShaderUserData user{};
    AgcShaderSpecials specials{};
    AgcShaderHeader header{};
};
struct Owners {
    std::array<Program, 24> programs;
};
static Owners* owners;
static size_t next_program;

// Opt-in retention uses only finite role/width/wave names, never guest identifiers. A supplied
// directory must be canonical and absolute; exclusive writes preserve preexisting destinations.
// Partial files survive an I/O failure as failed-run evidence and are never reported as fixtures.
static bool artifact_directory(std::filesystem::path& directory) {
    const char* requested = std::getenv("PROSPER_PERWAVE_ARTIFACT_DIR");
    if (!requested || !*requested) return true; // normal seven cases retain no files
    directory = requested;
    if (!directory.is_absolute() || directory.lexically_normal() != directory) return false;
    std::error_code error;
    const auto resolved = std::filesystem::weakly_canonical(directory, error);
    if (error || resolved != directory) return false;
    const bool exists = std::filesystem::exists(directory, error);
    if (error || (!exists && !std::filesystem::create_directory(directory, error)) || error)
        return false;
    const auto status = std::filesystem::symlink_status(directory, error);
    return !error && std::filesystem::is_directory(status) && !std::filesystem::is_symlink(status);
}
static bool retain_artifact(const std::filesystem::path& path, const void* bytes, size_t size) {
    if (!bytes || !size) return false;
    FILE* file = std::fopen(path.string().c_str(), "wbx");
    if (!file) return false;
    const size_t written = std::fwrite(bytes, 1u, size, file);
    const int closed = std::fclose(file);
    return written == size && closed == 0;
}
static thread_local std::vector<std::vector<uint32_t>> completed_modules;
static thread_local size_t completed_transactions = 0;
struct ScopedModuleRetention {
    bool enabled;
    prosper::test::OwnedGraphicsWaveModuleObserver previous;
    explicit ScopedModuleRetention(bool retain)
        : enabled(retain), previous(prosper::test::owned_graphics_wave_module_observer) {
        if (!enabled) return;
        completed_modules.clear();
        completed_transactions = 0;
        prosper::test::owned_graphics_wave_module_observer = [](const auto& programs) {
            ++completed_transactions;
            for (const auto& program : programs) completed_modules.push_back(program.spirv);
        };
    }
    ~ScopedModuleRetention() {
        if (enabled) prosper::test::owned_graphics_wave_module_observer = previous;
    }
};

// AGC retains registered addresses for process lifetime. The absolute mapped registry and real
// direct output allocations therefore stay alive until the test process/device has exited.
static uint64_t direct_page() {
    uint64_t physical = 0, address = 0;
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    if (!allocate || !map ||
        allocate(0, 0x200000000ull, Page, Page, 0, reinterpret_cast<uint64_t>(&physical)) != 0 ||
        map(reinterpret_cast<uint64_t>(&address), Page, 3u, 0u, physical, Page) != 0)
        return 0;
    std::memset(reinterpret_cast<void*>(address), 0, Page);   // actual committed CPU-readable input
    return address;
}
static Program* register_program(bool vertex, const std::vector<uint32_t>& code) {
    if (!owners || next_program >= owners->programs.size() || code.size() > 128u) return nullptr;
    auto& p = owners->programs[next_program++];
    std::copy(code.begin(), code.end(), p.code.begin());
    p.registers[0].offset = vertex ? P::SPI_SHADER_PGM_LO_ES : P::SPI_SHADER_PGM_LO_PS;
    p.registers[1].offset = vertex ? P::SPI_SHADER_PGM_HI_ES : P::SPI_SHADER_PGM_HI_PS;
    p.specials.user_data_range_end = 2u;   // observed loaded PS0 / merged-VS8 pointer pair
    p.header.file_header = 0x34333231u;
    p.header.version = 0x18u;
    p.header.user_data = &p.user;
    p.header.specials = &p.specials;
    p.header.sh_registers = p.registers.data();
    p.header.num_sh_registers = 2u;
    p.header.shader_size = uint32_t(code.size() * 4u);
    p.header.type = vertex ? 2u : 1u;
    const auto create = Hle::lookup("f3dg2CSgRKY");
    void* registered = nullptr;
    if (!create ||
        create(reinterpret_cast<uint64_t>(&registered), reinterpret_cast<uint64_t>(&p.header),
               reinterpret_cast<uint64_t>(p.code.data()), 0, 0, 0) != 0 ||
        registered != &p.header)
        return nullptr;
    EXPECT_EQ(p.registers[0].value, uint32_t(reinterpret_cast<uint64_t>(p.code.data()) >> 8u));
    EXPECT_EQ(p.registers[1].value,
              uint32_t((reinterpret_cast<uint64_t>(p.code.data()) >> 40u) & 255u));
    return &p;
}
static std::vector<uint32_t> vertex_code(bool per_wave, bool wide8) {
    std::vector<uint32_t> code;
    if (per_wave) {
        code = {0x7e280503u,
                0x87148114u,
                0x8f148414u,   // RF InstanceIndex -> AND1 -> offset16
                wide8 ? 0xf40c0604u : 0xf4080604u,
                0x28000000u,
                0x7e0e0200u | (wide8 ? 31u : 27u)};
    }
    code.insert(code.end(), std::begin(Fullscreen), std::end(Fullscreen));
    code.insert(code.end(), {0xf80008cfu, 0x04030201u});   // nondegenerate actual indexed POS0
    if (per_wave) code.insert(code.end(), {0xf800020fu, 0x04030307u});   // PARAM0=(numeric,0,0,1)
    code.push_back(0xbf810000u);
    return code;
}
static std::vector<uint32_t> fragment_code(bool per_wave, bool wide8) {
    if (!per_wave)
        return {0xc8000000u, 0xc8010001u,   // interpolate produced PARAM0.x
                0x7e0202ffu, 0x3f000000u, 0x7e0402ffu, 0x3f400000u,
                0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u};
    // The x8 load writes s[24:31]; its real loop counter must remain outside that destination.
    std::vector<uint32_t> code{0xbea80380u};   // genuine dominating s40=0
    const uint32_t event = uint32_t(code.size());
    code.insert(
        code.end(),
        {0x7e280501u, 0x90149714u, 0x87148314u, 0x8f148414u, wide8 ? 0xf40c0600u : 0xf4080600u,
         0x28000000u,
         0x7e000200u |
             (wide8 ? 31u : 27u),   // current RF FragCoord.y exponent selects last component
         0x80288128u,   // s40 += 1, real SCC definition
         0x060202ffu, 0x40800000u,   // v1 += 4.0: second RF must sample the CURRENT VGPR
         0xbf0a8228u});   // s_cmp_lt_u32 s40,2
    const int32_t displacement = int32_t(event) - int32_t(code.size() + 1u);
    code.push_back(0xbf850000u | uint16_t(displacement));
    // The raw packet ABI observes enabled EXP payload even in nonexporting helper slots. Restore
    // full EXEC and write the final current SMEM colour in every lane; helper export eligibility
    // remains independently false in the actual collected packet/attachment commit.
    code.insert(code.end(),
                {0xbefe04c1u, 0x7e000200u | (wide8 ? 31u : 27u), 0x7e0202ffu, 0x3f000000u,
                 0x7e0402ffu, 0x3f400000u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u});
    return code;
}

class OwnedGraphicsWaveLive : public ::testing::TestWithParam<bool> {
protected:
    static void SetUpTestSuite() {
        register_builtin_hle();
        const auto map = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
        uint64_t address = 0;
        ASSERT_TRUE(map);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&address), Page, 2u, 0u,
                      reinterpret_cast<uint64_t>("owned-wave-programs"), 0),
                  0);
        ASSERT_GE(address, 0x100000000ull);
        static_assert(sizeof(Owners) <= Page);
        owners = std::construct_at(reinterpret_cast<Owners*>(address));
        prosper::frontend::register_live_renderer(".", false);
    }
    void SetUp() override {
        ASSERT_TRUE(prosper::test::render_vk_ctx().ok);
        source = direct_page();
        output = direct_page();
        index_page = direct_page();
        ASSERT_NE(source, 0u);
        ASSERT_NE(output, 0u);
        ASSERT_NE(index_page, 0u);
        auto* indices = reinterpret_cast<uint32_t*>(index_page);
        for (uint32_t i = 0; i < 64u; ++i) indices[i] = i % 3u;
        auto* words = reinterpret_cast<uint32_t*>(source);
        std::fill_n(words, 32u, std::bit_cast<uint32_t>(0.125f));
        const uint32_t last = GetParam() ? 7u : 3u;
        words[last] = std::bit_cast<uint32_t>(0.25f);
        words[4u + last] = std::bit_cast<uint32_t>(0.75f);
        words[8u + last] = std::bit_cast<uint32_t>(0.25f);
    }
    GpuState state(const Program& vs, const Program& ps, bool owned_vs) {
        GpuState s;
        for (const Program* p : {&vs, &ps})
            for (const auto& reg : p->registers) s.sh[reg.offset] = reg.value;
        s.uc[P::VGT_PRIMITIVE_TYPE] = 4u;
        s.cx[P::CB_TARGET_MASK] = s.cx[P::CB_SHADER_MASK] = 15u;
        s.cx[P::SPI_PS_IN_CONTROL] = 0u;
        s.sh[P::SPI_SHADER_PGM_RSRC1_PS] = 0u;   // observed full word, independent of width/entry
        s.sh[P::SPI_SHADER_PGM_RSRC2_PS] = 2u << P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
        s.sh[P::SPI_SHADER_PGM_RSRC2_GS] = 2u << P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT;
        for (uint32_t base : {P::SPI_SHADER_USER_DATA_GS_0, P::SPI_SHADER_USER_DATA_PS_0}) {
            s.sh[base] = uint32_t(source);
            s.sh[base + 1u] = uint32_t(source >> 32u);
        }
        s.cx[P::SPI_PS_INPUT_ENA] = s.cx[P::SPI_PS_INPUT_ADDR] = owned_vs ? 2u : 0x300u;
        if (owned_vs) s.cx[P::SPI_PS_INPUT_CNTL_0] = 0u;
        s.cx[P::CB_COLOR0_BASE] = uint32_t(output >> 8u);
        s.cx[P::CB_COLOR0_BASE_EXT] = uint32_t(output >> 40u);
        s.cx[P::CB_COLOR0_INFO] = 0xau << P::CB_COLOR0_INFO_FORMAT_SHIFT;
        s.cx[P::CB_COLOR0_ATTRIB2] = ((W - 1u) << P::CB_COLOR0_ATTRIB2_MIP0_WIDTH_SHIFT) | (H - 1u);
        s.cx[P::CB_COLOR0_ATTRIB3] = 1u << P::CB_COLOR0_ATTRIB3_RESOURCE_TYPE_SHIFT;
        s.index_type = 1u;
        s.index_type_announced = true;
        GpuState::Draw draw;
        draw.indexed = owned_vs;
        draw.index_addr = owned_vs ? index_page : 0;
        draw.index_count = owned_vs ? 64u : 3u;
        draw.instance_count = owned_vs ? 2u : 1u;
        s.draws.push_back(draw);
        return s;
    }
    bool realize(GpuState& state, DrawItem& draw) {
        const auto status = graphics_producer_status();
        const GraphicsRawSnapshotContext complete{status.known && !status.pending};
        return complete.producers_complete &&
               realize_draw_item(state, &state.draws[0], state.draws[0].index_count, 64u, false,
                                 draw, nullptr, true, nullptr, &complete);
    }
    void pixels(const std::vector<uint8_t>& image, bool fragment) {
        ASSERT_EQ(image.size(), W * H * 4u);
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x) {
                // Sorted full quads form two64-slot waves. At the SECOND event y+.5 becomes
                // y+4.5: first wave selects offset16/.75, second offset32/.25.
                const std::array<int, 4> expected{fragment && y >= 4u ? 64 : 191, 128, 191, 255};
                for (uint32_t c = 0; c < 4u; ++c)
                    EXPECT_NEAR(int(image[(y * W + x) * 4u + c]), expected[c], 1)
                        << "x=" << x << " y=" << y << " c=" << c;
            }
    }
    void capture_parity(const DrawItem& draw, bool fragment) {
        std::filesystem::path directory;
        ASSERT_TRUE(artifact_directory(directory)) << "unsafe or unavailable artifact directory";
        if (!directory.empty())
            ASSERT_EQ(prosper::test::owned_graphics_wave_module_observer, nullptr);
        GpuCaptureMetadata metadata;
        metadata.width = W;
        metadata.height = H;
        GpuCaptureFile capture, decoded;
        GpuReplayFrame replay;
        std::vector<uint8_t> bytes;
        std::string error;
        const auto vs_original = registered_graphics_original(draw.vs_guest_addr);
        const auto fs_original = registered_graphics_original(draw.fs_guest_addr);
        ASSERT_TRUE(vs_original && !vs_original->empty());
        ASSERT_TRUE(fs_original && !fs_original->empty());
        const std::array originals{std::pair{draw.vs_guest_addr, vs_original},
                                   std::pair{draw.fs_guest_addr, fs_original}};
        for (const auto& [address, original] : originals) {
            ASSERT_NE(address, 0u);
            ASSERT_LE(original->size(), UINT64_MAX / sizeof(uint32_t));
            ASSERT_LE(address, UINT64_MAX - original->size() * sizeof(uint32_t));
        }
        // The reader's size is destination capacity. Return only the complete registered code
        // owner at its exact start, never a guest read, an interior span or fabricated padding.
        const CaptureMemoryReader original_reader = [originals](uint64_t address, uint8_t* dst,
                                                                size_t capacity) {
            for (const auto& [start, original] : originals) {
                if (address != start) continue;
                const size_t size = original->size() * sizeof(uint32_t);
                if (!dst || capacity < size) return size_t(0);
                std::memcpy(dst, original->data(), size);
                return size;
            }
            return size_t(0);
        };
        std::array<uint8_t, sizeof(uint32_t)> refused_code{};
        EXPECT_EQ(
            original_reader(draw.vs_guest_addr + 1u, refused_code.data(), refused_code.size()), 0u);
        EXPECT_EQ(original_reader(UINT64_MAX, refused_code.data(), refused_code.size()), 0u);
        EXPECT_EQ(original_reader(draw.vs_guest_addr, refused_code.data(), refused_code.size()),
                  0u);
        ASSERT_TRUE(capture_draw_items({draw}, metadata, original_reader, capture, error)) << error;
        ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
        ASSERT_TRUE(deserialize_gpu_capture(bytes, decoded, error)) << error;
        ASSERT_EQ(decoded.draws.size(), 1u);
        ASSERT_LT(decoded.draws[0].vs_raw_shader_index, decoded.raw_shader_versions.size());
        ASSERT_LT(decoded.draws[0].fs_raw_shader_index, decoded.raw_shader_versions.size());
        EXPECT_EQ(decoded.raw_shader_versions[decoded.draws[0].vs_raw_shader_index].words,
                  *vs_original);
        EXPECT_EQ(decoded.raw_shader_versions[decoded.draws[0].fs_raw_shader_index].words,
                  *fs_original);
        ASSERT_TRUE(materialize_gpu_replay(decoded, replay, error)) << error;
        ASSERT_EQ(replay.items.size(), 1u);
        ASSERT_TRUE(replay.items[0].owned_waves);
        EXPECT_EQ(replay.items[0].ps_entry, draw.ps_entry);
        EXPECT_EQ(replay.items[0].ps_launch_rsrc1, draw.ps_launch_rsrc1);
        auto* mutable_words = reinterpret_cast<uint32_t*>(source);
        for (uint32_t i = 0; i < 32u; ++i) mutable_words[i] = std::bit_cast<uint32_t>(0.0f);
        // Actual stored replay uses the same authenticated original-code owner. No GPU read or
        // restored guest VA is needed to retain A after guest source B changes.
        {
            // Observe the SAME stored-replay transaction this case already checks, after real
            // completion and whole-output validation. Copy module bytes, never GPU handles.
            ScopedModuleRetention retention(!directory.empty());
            pixels(render_submit_items(replay.items, W, H), fragment);
        }
        auto ownerless = decoded;
        ownerless.draws[0].owned_waves.reset();
        GpuReplayFrame refused;
        EXPECT_FALSE(materialize_gpu_replay(ownerless, refused, error));
        EXPECT_EQ(error,
                  fragment
                      ? "logical-wave replay lacks original exact-PC window owners stage=fs pc=5"
                      : "logical-wave replay lacks original exact-PC window owners stage=vs pc=3");
        EXPECT_TRUE(refused.items.empty());
        if (directory.empty() || HasFailure()) return;
        ASSERT_EQ(completed_transactions, 1u);
        ASSERT_EQ(completed_modules.size(), 2u);
        for (const auto& module : completed_modules) {
            ASSERT_FALSE(module.empty());
            ASSERT_EQ(module.front(), 0x07230203u);
        }
        const std::string role = std::string(fragment ? "ps-x" : "vs-x") + (GetParam() ? "8" : "4");
        ASSERT_TRUE(retain_artifact(directory / (role + ".prgcap"), bytes.data(), bytes.size()))
            << "exclusive capture retention failed for " << role;
        for (size_t wave = 0; wave < completed_modules.size(); ++wave) {
            const auto& module = completed_modules[wave];
            ASSERT_TRUE(
                retain_artifact(directory / (role + ".wave" + std::to_string(wave) + ".spv"),
                                module.data(), module.size() * sizeof(uint32_t)))
                << "exclusive completed-module retention failed for " << role << " wave=" << wave;
        }
        std::printf("owned-wave artifacts %s: capture-bytes=%zu completed-modules=%zu\n",
                    role.c_str(), bytes.size(), completed_modules.size());
    }
    uint64_t source = 0, output = 0, index_page = 0;
};

TEST_P(OwnedGraphicsWaveLive, IndexedInstancedVertexWavesCommitDistinctNumericOwners) {
    auto* vs = register_program(true, vertex_code(true, GetParam()));
    auto* ps = register_program(false, fragment_code(false, GetParam()));
    ASSERT_NE(vs, nullptr);
    ASSERT_NE(ps, nullptr);
    auto s = state(*vs, *ps, true);
    DrawItem draw;
    ASSERT_TRUE(realize(s, draw));
    ASSERT_TRUE(draw.owned_waves);
    const auto& plan = draw.owned_waves->vertex;
    ASSERT_EQ(plan.packets.size(), 2u);
    ASSERT_EQ(plan.invocations.size(), 2u);
    for (uint32_t lane : {0u, 31u, 32u, 63u}) {
        EXPECT_EQ(plan.invocations[0][lane].vertex_index, lane % 3u);
        EXPECT_EQ(plan.invocations[1][lane].vertex_index, lane % 3u);
        EXPECT_EQ(plan.invocations[0][lane].instance_index, 0u);
        EXPECT_EQ(plan.invocations[1][lane].instance_index, 1u);
    }
    prosper::test::BackendSubmissionBatch prior;
    GraphicsWaveOutputTransaction completed;
    GraphicsVertexExportCommit commit;
    std::string error;
    {
        prosper::test::BackendPersistentResourceGuard guard;
        ASSERT_TRUE(prosper::test::execute_owned_graphics_waves(prosper::test::render_vk_ctx(),
                                                                plan, prior, completed, error))
            << error;
    }
    ASSERT_TRUE(prepare_owned_vertex_export_commit(
        plan, completed, draw.has_pixel_inputs ? &draw.pixel_inputs : nullptr, commit, error))
        << error;
    ASSERT_EQ(commit.words.size(), 128u * 8u);
    EXPECT_EQ(commit.instances, 2u);
    for (uint32_t instance = 0; instance < 2u; ++instance)
        for (uint32_t occurrence = 0; occurrence < 64u; ++occurrence) {
            const uint32_t offset = (instance * 64u + occurrence) * 8u;
            const uint32_t index = occurrence % 3u;
            EXPECT_EQ(commit.words[offset], std::bit_cast<uint32_t>(index == 1u ? 3.0f : -1.0f));
            EXPECT_EQ(commit.words[offset + 1u],
                      std::bit_cast<uint32_t>(index == 2u ? 3.0f : -1.0f));
            EXPECT_EQ(commit.words[offset + 4u], std::bit_cast<uint32_t>(instance ? 0.75f : 0.25f));
        }
    pixels(render_submit_items({draw}, W, H), false);
    capture_parity(draw, false);
}

TEST_P(OwnedGraphicsWaveLive, FragmentWavesSampleCurrentReadfirstSourceOnEveryVisit) {
    auto* vs = register_program(true, vertex_code(false, GetParam()));
    auto* ps = register_program(false, fragment_code(true, GetParam()));
    ASSERT_NE(vs, nullptr);
    ASSERT_NE(ps, nullptr);
    auto s = state(*vs, *ps, false);
    DrawItem draw;
    ASSERT_TRUE(realize(s, draw));
    ASSERT_TRUE(draw.owned_waves);
    ASSERT_TRUE(draw.owned_waves->fragment_pending);
    EXPECT_TRUE(draw.owned_waves->fragment_raster_inputs->entry.observed);
    EXPECT_EQ(draw.owned_waves->fragment_raster_inputs->entry, draw.ps_entry);
    EXPECT_EQ(draw.system_inputs.ena, 0x300u);
    // Drive the actual registered-source DrawItem observation with the producing six-bit PS
    // prefix. Read s16 before any writer: physical UD16 must not become system s16 at N=16.
    auto prefix_code = fragment_code(true, GetParam());
    prefix_code.insert(prefix_code.begin() + 1u, 0x7e0a0210u);   // v5=s16, real entry read
    auto* prefix_ps = register_program(false, prefix_code);
    ASSERT_NE(prefix_ps, nullptr);
    const auto prefix_state = [&](uint32_t count) {
        auto observed = state(*vs, *prefix_ps, false);
        observed.sh[P::SPI_SHADER_PGM_RSRC2_PS] =
            ((count & 31u) << P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) |
            ((count >> 5u) << P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT);
        for (uint32_t reg = 2u; reg < 32u; ++reg)
            observed.sh[P::SPI_SHADER_USER_DATA_PS_0 + reg] = 0u;   // present zero is a value
        return observed;
    };
    const auto entry_packet = [&](const DrawItem& observed) {
        FragmentInvocationPacket packet;
        packet.guest_code = *observed.owned_waves->fragment_code;
        packet.sgprs = observed.owned_waves->fragment_scalars;
        packet.raw_windows = observed.owned_waves->fragment_windows;
        packet.slots_available.fill(true);
        packet.mask_state_available = true;
        FragmentPacketVgpr source_y;
        source_y.reg = 1u;
        source_y.words.fill(std::bit_cast<uint32_t>(0.5f));   // authenticated fixture input column
        packet.vgprs.push_back(source_y);
        return packet;
    };
    for (uint32_t count : {17u, 32u}) {
        auto observed = prefix_state(count);
        DrawItem prefix_draw;
        ASSERT_TRUE(realize(observed, prefix_draw));
        ASSERT_TRUE(prefix_draw.owned_waves);
        ASSERT_EQ(prefix_draw.owned_waves->fragment_scalars.size(), count);
        EXPECT_EQ(prefix_draw.owned_waves->fragment_scalars.back(), std::make_pair(count - 1u, 0u));
        auto packet = entry_packet(prefix_draw);
        std::string error;
        EXPECT_TRUE(complete_graphics_packet_locals(packet, error, false)) << error;
        if (count == 32u) {
            observed.sh.erase(P::SPI_SHADER_USER_DATA_PS_0 + 31u);
            DrawItem missing_unused;
            ASSERT_TRUE(realize(observed, missing_unused));
            ASSERT_TRUE(missing_unused.owned_waves);
            ASSERT_EQ(missing_unused.owned_waves->fragment_scalars.size(), 31u);
            EXPECT_EQ(missing_unused.owned_waves->fragment_scalars.back().first, 30u);
            packet = entry_packet(missing_unused);
            EXPECT_TRUE(complete_graphics_packet_locals(packet, error, false))
                << "absent s31 stays absent; this original either never reads it or overwrites it "
                   "first";
        }
    }
    for (bool system_word : {false, true}) {
        auto observed = prefix_state(system_word ? 16u : 17u);
        if (!system_word) observed.sh.erase(P::SPI_SHADER_USER_DATA_PS_0 + 16u);
        DrawItem missing_read;
        ASSERT_TRUE(realize(observed, missing_read));
        ASSERT_TRUE(missing_read.owned_waves);
        auto packet = entry_packet(missing_read);
        std::string error;
        EXPECT_FALSE(complete_graphics_packet_locals(packet, error, false));
        EXPECT_EQ(error, "stage-input-scalar-uninitialized:pc=1")
            << "missing prefix s16 and unproved system s16 cannot borrow physical UD16";
    }
    for (uint32_t count : {33u, 63u}) {
        auto observed = prefix_state(count);
        const GraphicsRawSnapshotContext complete{true};
        std::shared_ptr<const GraphicsOwnedWaveDraw> refused;
        std::vector<uint32_t> indices;
        std::string error;
        EXPECT_FALSE(prepare_draw_owned_waves(observed, &observed.draws[0],
                                              reinterpret_cast<uint64_t>(vs->code.data()),
                                              reinterpret_cast<uint64_t>(prefix_ps->code.data()),
                                              3u, {}, &complete, refused, indices, error));
        EXPECT_FALSE(refused);
        EXPECT_EQ(error, "draw-wave-fragment-user-sgpr-count-out-of-range");
    }
    pixels(render_submit_items({draw}, W, H), true);
    capture_parity(draw, true);
}

TEST_P(OwnedGraphicsWaveLive, UnavailableSecondWaveRefusesBeforeAttachmentPublication) {
    auto* vs = register_program(true, vertex_code(true, GetParam()));
    auto* ps = register_program(false, fragment_code(false, GetParam()));
    ASSERT_NE(vs, nullptr);
    ASSERT_NE(ps, nullptr);
    auto s = state(*vs, *ps, true);
    DrawItem draw;
    ASSERT_TRUE(realize(s, draw));
    ASSERT_TRUE(draw.owned_waves);
    auto invalid = std::make_shared<GraphicsOwnedWaveDraw>(*draw.owned_waves);
    ASSERT_EQ(invalid->vertex.packets.size(), 2u);
    ASSERT_FALSE(invalid->vertex.packets[1].raw_windows.empty());
    ASSERT_FALSE(invalid->vertex.packets[1].raw_windows[0].words.empty());
    invalid->vertex.packets[1].raw_windows[0].words.pop_back();
    GraphicsWaveOutputTransaction completed;
    prosper::test::BackendSubmissionBatch prior;
    std::string error;
    {
        prosper::test::BackendPersistentResourceGuard guard;
        EXPECT_FALSE(prosper::test::execute_owned_graphics_waves(
            prosper::test::render_vk_ctx(), invalid->vertex, prior, completed, error));
    }
    EXPECT_TRUE(completed.records.empty()) << "the first wave cannot publish a partial transaction";
    draw.owned_waves = std::move(invalid);
    const auto generation = prosper::test::backend_failed_publication_generation().load();
    EXPECT_TRUE(render_submit_items({draw}, W, H).empty());
    EXPECT_GT(prosper::test::backend_failed_publication_generation().load(), generation);
    EXPECT_TRUE(std::all_of(reinterpret_cast<const uint8_t*>(output),
                            reinterpret_cast<const uint8_t*>(output) + Page,
                            [](uint8_t b) { return b == 0; }))
        << "refusal preserves the original guest backing";
}
// graphics_program_requires_owned_waves is asked several times per draw, so it answers from the
// decode cache's once-per-byte-version classification instead of re-walking the stream (#4270
// made the re-derivation GTA V's dominant render-thread cost). The answer must still equal a
// from-scratch derivation, and a rewrite of the registered bytes must not be answered stale.
TEST_P(OwnedGraphicsWaveLive, OwnedWaveClassificationMatchesFullStreamDerivation) {
    const auto derived = [](const std::vector<uint32_t>& code) {
        std::vector<Rdna2Inst> walked;
        rdna2_walk(code.data(), code.size(), walked);
        return !rdna2_raw_wave_wide_data_loads(walked).empty();
    };
    for (const bool vertex : {true, false})
        for (const bool per_wave : {false, true}) {
            const auto code = vertex ? vertex_code(per_wave, GetParam())
                                     : fragment_code(per_wave, GetParam());
            auto* program = register_program(vertex, code);
            ASSERT_NE(program, nullptr);
            const uint64_t address = reinterpret_cast<uint64_t>(program->code.data());
            ASSERT_EQ(derived(code), per_wave) << "fixture must exercise both classifications";
            for (int query = 0; query < 3; ++query)
                EXPECT_EQ(graphics_program_requires_owned_waves(address), per_wave)
                    << (vertex ? "vertex" : "fragment") << " query " << query;
        }
    // Counter-arm: the same registered address rewritten from plain to per-wave code. A
    // classification keyed on the address alone would keep answering false here.
    const auto plain = fragment_code(false, GetParam());
    const auto owned = fragment_code(true, GetParam());
    auto* program = register_program(false, plain);
    ASSERT_NE(program, nullptr);
    const uint64_t address = reinterpret_cast<uint64_t>(program->code.data());
    EXPECT_FALSE(graphics_program_requires_owned_waves(address));
    std::copy(owned.begin(), owned.end(), program->code.begin());
    program->header.shader_size = uint32_t(owned.size() * 4u);
    EXPECT_TRUE(graphics_program_requires_owned_waves(address))
        << "rewritten registered bytes must be reclassified, not answered from a stale entry";
}
INSTANTIATE_TEST_SUITE_P(RawWidths, OwnedGraphicsWaveLive, ::testing::Values(false, true));

TEST(OwnedGraphicsWaveCompletion, IndeterminateCompletionRetainsRecordedGpuOwner) {
    // Exercise the exact cleanup branch used after an unproved submit, with real recorded Vulkan
    // owners. No command is submitted and no driver failure is fabricated. Only for that reason
    // may this isolated discriminator restore the process poison flag after checking retention.
    const auto& ctx = prosper::test::render_vk_ctx();
    ASSERT_TRUE(ctx.ok);
    prosper::test::BackendPersistentResourceGuard guard;
    ASSERT_FALSE(prosper::test::backend_has_unproven_submission());
    struct ClearUnsubmittedControlPoison {
        ~ClearUnsubmittedControlPoison() {
            prosper::test::backend_unproven_submission_storage().store(false);
        }
    } clear_poison;
    auto owner = std::make_shared<prosper::test::OwnedGraphicsWaveGpuOwner>();
    owner->context = &ctx;
    owner->buffers.push_back(prosper::test::acquire_render_host_buffer(ctx, Page));
    ASSERT_NE(owner->buffers.back().mapped, nullptr);
    owner->commands = prosper::test::acquire_render_command_pool(ctx.dev, ctx.qfi);
    ASSERT_NE(owner->commands.command, VK_NULL_HANDLE);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ASSERT_EQ(vkBeginCommandBuffer(owner->commands.command, &begin), VK_SUCCESS);
    ASSERT_EQ(vkEndCommandBuffer(owner->commands.command), VK_SUCCESS);
    std::weak_ptr<prosper::test::OwnedGraphicsWaveGpuOwner> retained = owner;
    bool cleanup_ran = false, speculative_valid = true;
    {
        prosper::test::BackendSubmissionBatch abandoned;
        abandoned.enqueue(owner->commands.command);
        abandoned.add_failure_cleanup([&] { speculative_valid = false; });
        abandoned.add_cleanup([owner, &cleanup_ran] { cleanup_ran = true; });
        owner.reset();
        abandoned.abandon_pending_resources();
        abandoned.complete();
        EXPECT_TRUE(abandoned.retains_pending_resources());
        EXPECT_FALSE(abandoned.pending());
        EXPECT_FALSE(speculative_valid);
        EXPECT_FALSE(cleanup_ran);
        EXPECT_FALSE(retained.expired());
        GraphicsWaveStagePlan plan;
        GraphicsWaveOutputTransaction transaction;
        prosper::test::BackendSubmissionBatch prior;
        std::string error;
        EXPECT_FALSE(
            prosper::test::execute_owned_graphics_waves(ctx, plan, prior, transaction, error));
        EXPECT_EQ(error, "graphics-wave-executing-device-unavailable");
        EXPECT_TRUE(transaction.records.empty());
    }
    EXPECT_FALSE(retained.expired()) << "batch destruction must not destroy unproved GPU owners";
}
}   // namespace
