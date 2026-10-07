// test_integer_color_export -- #4703: integer colour targets written through compressed exports.
//
// Kena: Bridge of Spirits copies stencil into UE4's lighting-channel texture with an R16_UINT
// target and SPI_SHADER_COL_FORMAT=0x7 (UINT16_ABGR). prosper used to decode every compressed
// export as two f16 halves and declare every output float, and the backend folded R16_UINT into
// RGBA8 UNORM, so the channel value 1 became 5.96e-8 and was stored as 0. The deferred sun light ANDs its
// channel mask with that texture, so the sun added nothing anywhere.
//
// The arms, and the mutation each one exists to catch (all were run; see the PR):
//   RoundTrip            -- decode a UINT16 half as f16 again (compressed_export_channel): reads 0.
//                           Drop R16_UINT from backend_integer_color_format: readback is 4 B/px.
//   ReaderBranch         -- Kena's reader: image_load the R16_UINT target as uint, AND it with the
//                           channel mask, v_cmpx_ne_u32 0, export. Any zeroed producer is discarded.
//   CacheKey             -- drop the export formats from ShaderCompileKey equality/hash: one module
//                           serves both a float and an integer target. Declare every output float
//                           again (recompile_fragment's color_output_class): the declared class
//                           check goes red, which RADV's execution arms alone cannot see.
//   LiveDrawKeysOnTarget -- pass {} instead of the derived formats at the live compile site
//                           (gpu_execute.hpp): the draw gets the f16, float-output module.
//   GuestAndBackendAgree -- fold any integer guest format into a float backend format (or the
//                           reverse): the uvec4/ivec4 output would no longer match its attachment.
//   Fp16Control          -- the positive control: with FP16 col_format the same program still
//                           unpacks halves into a float target, exactly as before #4703.
#include <gtest/gtest.h>

#include "fixtures/render_runner.h"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/state/fragment_export_state.hpp"
#include "gpu/state/render_state.hpp"
#include "gpu/state/vk_translate.hpp"
#include "shared/live/live_target_format.hpp"

#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <vector>

// No `using namespace prosper::gpu`: its VkFormat enum would make ::VkFormat ambiguous here.
namespace gpu = prosper::gpu;
using gpu::ColorExportFormat;
using gpu::DataFormat;
using gpu::FragmentExportFormats;
using gpu::FragmentOutputClass;
using gpu::make_fragment_export_formats;
using gpu::recompile_fragment;
using gpu::RecompileDiagnosticStage;
using gpu::ResolvedPipelineState;
using gpu::ResourceClass;
using gpu::ShaderProgramStage;
using gpu::ShaderResource;
using gpu::ShaderResourceTable;
using gpu::SpirvImageNumericClass;

namespace {

constexpr uint32_t W = 16, H = 16;
constexpr size_t kCenter = size_t(H / 2) * W + W / 2;   // texel index of the frame centre

// v0 = (r, g) and v1 = (b, a), then `exp mrt0 v0, v1 compr done vm` with all four channels.
std::vector<uint32_t> compressed_writer(uint32_t rg, uint32_t ba) {
    return {0x7e0002ffu, rg,   // v_mov_b32 v0, rg
            0x7e0202ffu, ba,   // v_mov_b32 v1, ba
            0xf8001c0fu, 0x00000100u,   // exp mrt0 v0, v1 compr done vm (en=0xF)
            0xbf810000u};   // s_endpgm
}

// Kena's reader, reduced: image_load v2 at (0,0) from the R16_UINT target as uint, AND the channel
// mask (1), v_cmpx_ne_u32 0 kills every lane whose channel is clear, then export v2 raw to MRT0.
const uint32_t kReaderPs[] = {
    0x7e000280u,   // v_mov_b32 v0, 0         x
    0x7e020280u,   // v_mov_b32 v1, 0         y
    0xf0000108u, 0x00020200u,   // image_load v2, v[0:1], s[8:15] dmask:x dim:2D
    0x36060481u,   // v_and_b32 v3, 1, v2     channel & LightingChannelMask
    0x7daa0680u,   // v_cmpx_ne_u32 0, v3     early-out of every zero channel
    0xf8001801u, 0x00000002u,   // exp mrt0 v2 done vm (en=R)
    0xbf810000u,   // s_endpgm
};

std::vector<uint32_t> fullscreen_vs() {
#include "../tools/boot_trace/refvs.inc"
    return std::vector<uint32_t>(std::begin(kRefVs), std::end(kRefVs));
}

FragmentExportFormats uint16_into_uint() {
    FragmentOutputClass classes[8]{};
    classes[0] = FragmentOutputClass::Uint;
    return make_fragment_export_formats(0x7u, classes);
}

ResolvedPipelineState pipeline_for(VkFormat format) {
    ResolvedPipelineState ps{};
    ps.topology = 3;   // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    ps.color_write_mask = 0xF;
    ps.color0_format = static_cast<uint32_t>(format);
    ps.color_targets[0].format = ps.color0_format;
    ps.color_targets[0].write_mask = 0xF;
    return ps;
}

// The numeric class of the fragment module's Location-`location` Output variable: what Vulkan
// matches against the attachment format. RADV happens to store a float-typed output's bits into an
// R16_UINT attachment correctly, so the execution arms cannot see a wrong declaration; this can.
FragmentOutputClass declared_output_class(const std::vector<uint32_t>& spirv, uint32_t location) {
    std::map<uint32_t, uint32_t> int_signedness, vector_component, pointer_pointee, var_type;
    std::map<uint32_t, uint32_t> locations;
    for (size_t i = 5; i < spirv.size();) {
        const uint32_t op = spirv[i] & 0xffffu, words = spirv[i] >> 16;
        if (!words || i + words > spirv.size()) break;
        const uint32_t* w = &spirv[i];
        if (op == 21) int_signedness[w[1]] = w[3];   // OpTypeInt
        if (op == 23) vector_component[w[1]] = w[2];   // OpTypeVector
        if (op == 32 && w[2] == 3) pointer_pointee[w[1]] = w[3];   // OpTypePointer Output
        if (op == 59 && w[3] == 3) var_type[w[2]] = w[1];   // OpVariable Output
        if (op == 71 && words >= 4 && w[2] == 30) locations[w[1]] = w[3];   // OpDecorate Location
        i += words;
    }
    for (const auto& [variable, pointer] : var_type) {
        if (!locations.count(variable) || locations[variable] != location) continue;
        const uint32_t component = vector_component[pointer_pointee[pointer]];
        if (!int_signedness.count(component)) return FragmentOutputClass::Float;
        return int_signedness[component] ? FragmentOutputClass::Sint : FragmentOutputClass::Uint;
    }
    return FragmentOutputClass::Float;
}

bool device_available() {
    return prosper::test::render_vk_ctx().ok;
}

// Render one fullscreen draw and return slot 0's native bytes.
std::vector<uint8_t> render(const std::vector<uint32_t>& fs, VkFormat format,
                            std::vector<prosper::test::FrameResource> resources = {},
                            prosper::test::BackendColorTarget* target = nullptr) {
    static const std::vector<uint32_t> vs = fullscreen_vs();
    const ResolvedPipelineState ps = pipeline_for(format);
    prosper::test::BackendDraw draw;
    draw.vs = vs;
    draw.fs = fs;
    draw.ps = &ps;
    draw.vcount = 3;
    draw.R = std::move(resources);
    return prosper::test::render_draws_rgba({draw}, W, H, nullptr, nullptr, false, target);
}

uint16_t u16_at_center(const std::vector<uint8_t>& px) {
    uint16_t value = 0;
    std::memcpy(&value, &px[kCenter * 2u], 2);
    return value;
}

uint32_t u32_at_center(const std::vector<uint8_t>& px) {
    uint32_t value = 0;
    std::memcpy(&value, &px[kCenter * 4u], 4);
    return value;
}

}   // namespace

TEST(IntegerColorExport, NormalizedKeyKeepsHistoricalValue) {
    FragmentOutputClass floats[8]{};
    // FP16 and every 32-bit code compile as they always did, so they normalize to the old key.
    for (uint32_t code : {0u, 1u, 2u, 3u, 4u, 9u})
        EXPECT_EQ(make_fragment_export_formats(code * 0x11111111u, floats), FragmentExportFormats{})
            << "col_format code " << code;
    const FragmentExportFormats kena = uint16_into_uint();
    EXPECT_EQ(kena.compressed_formats, 0x7u);
    EXPECT_EQ(kena.uint_outputs, 1u);
    EXPECT_EQ(kena.compressed_format(0), ColorExportFormat::Uint16Abgr);
    EXPECT_EQ(kena.compressed_format(1), ColorExportFormat::Fp16Abgr);
    EXPECT_TRUE(kena.canonical());
    FragmentExportFormats both = kena;
    both.sint_outputs = 1u;
    EXPECT_FALSE(both.canonical());
}

TEST(IntegerColorExport, KenaRegistersDeriveUint16) {
    // The writer's probed registers (#4703): CB_COLOR0_INFO FORMAT=2 (COLOR_16) NUMBER_TYPE=4
    // (UINT), SPI_SHADER_COL_FORMAT=0x7.
    ASSERT_EQ(gpu::vk_color_format(2u, 4u, 0u), gpu::VkFormat::R16_UINT);
    ResolvedPipelineState ps{};
    ps.color0_format = static_cast<uint32_t>(gpu::VkFormat::R16_UINT);
    ps.color_targets[0].format = ps.color0_format;
    ps.spi_shader_col_format = 0x7u;
    EXPECT_EQ(gpu::fragment_export_formats(ps), uint16_into_uint());
    // A float target with the FP16 export format is the historical key.
    ps.color0_format = static_cast<uint32_t>(gpu::VkFormat::R16G16B16A16_SFLOAT);
    ps.color_targets[0].format = ps.color0_format;
    ps.spi_shader_col_format = 0x4u;
    EXPECT_EQ(gpu::fragment_export_formats(ps), FragmentExportFormats{});
}

TEST(IntegerColorExport, GuestAndBackendAgree) {
    using Class = SpirvImageNumericClass;
    unsigned integer_formats = 0;
    for (uint32_t format = 0; format < 0x20u; ++format)
        for (uint32_t number_type = 0; number_type < 8u; ++number_type)
            for (uint32_t swap = 0; swap < 2u; ++swap) {
                const gpu::VkFormat guest = gpu::vk_color_format(format, number_type, swap);
                if (guest == gpu::VkFormat::Undefined) continue;
                const auto guest_vk = static_cast<::VkFormat>(guest);
                const auto host = prosper::test::backend_color_format(guest_vk);
                const Class backend = prosper::test::backend_image_numeric_class(host);
                const auto output = gpu::color_format_output_class(static_cast<uint32_t>(guest));
                if (output == FragmentOutputClass::Float) {
                    EXPECT_NE(backend, Class::Uint) << "guest VkFormat " << uint32_t(guest);
                    EXPECT_NE(backend, Class::Sint) << "guest VkFormat " << uint32_t(guest);
                    continue;
                }
                ++integer_formats;
                EXPECT_EQ(backend, output == FragmentOutputClass::Uint ? Class::Uint : Class::Sint)
                    << "guest VkFormat " << uint32_t(guest);
                // The attachment keeps the guest's texel size (0 = packed, not listed).
                const uint32_t guest_bytes =
                    prosper::frontend::live_target_guest_format_bytes(guest_vk);
                if (guest_bytes)
                    EXPECT_EQ(prosper::test::backend_color_bytes_per_pixel(host), guest_bytes)
                        << "guest VkFormat " << uint32_t(guest);
            }
    // Positive control on the loop itself: the CB table names 24 integer triples (COLOR_8,
    // COLOR_16, COLOR_8_8, COLOR_32, COLOR_16_16, COLOR_32_32, COLOR_16_16_16_16 and
    // COLOR_32_32_32_32 x UINT/SINT; COLOR_8_8_8_8 x UINT/SINT x STD/ALT; and UINT 2_10_10_10 x
    // STD/ALT under both of its codes, 0x8 and 0x9).
    EXPECT_EQ(integer_formats, 24u);
    EXPECT_EQ(prosper::test::backend_color_format(VK_FORMAT_R16_UINT), VK_FORMAT_R16_UINT);
    EXPECT_EQ(prosper::test::backend_color_bytes_per_pixel(VK_FORMAT_R16_UINT), 2u);
}

TEST(IntegerColorExport, CacheKey) {
    gpu::clear_shader_recompile_cache();
    const auto code = compressed_writer(0xbeef0001u, 0x00020003u);
    const auto fp16 = gpu::recompile_graphics_shader_cached(ShaderProgramStage::Fragment,
                                                            code.data(), code.size());
    const auto uint16 = gpu::recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, code.data(), code.size(), nullptr, nullptr, nullptr, nullptr,
        false, 0, false, {}, {}, {}, {}, {}, nullptr, uint16_into_uint());
    ASSERT_FALSE(fp16.empty());
    ASSERT_FALSE(uint16.empty());
    EXPECT_NE(fp16, uint16) << "a float and an integer target need different modules";
    EXPECT_EQ(declared_output_class(fp16, 0), FragmentOutputClass::Float);
    EXPECT_EQ(declared_output_class(uint16, 0), FragmentOutputClass::Uint)
        << "an R16_UINT attachment needs a uvec4 output";
    FragmentOutputClass sint_classes[8]{};
    sint_classes[0] = FragmentOutputClass::Sint;
    const auto sint16 =
        recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr, false,
                           {RecompileDiagnosticStage::Fragment, 0}, {}, nullptr, {}, {},
                           make_fragment_export_formats(0x8u, sint_classes));
    EXPECT_EQ(declared_output_class(sint16, 0), FragmentOutputClass::Sint);
    EXPECT_EQ(fp16, recompile_fragment(code.data(), code.size()));
    EXPECT_EQ(uint16, recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX,
                                         nullptr, false, {RecompileDiagnosticStage::Fragment, 0},
                                         {}, nullptr, {}, {}, uint16_into_uint()));
}

namespace {
// The live realization reads the program from guest memory, so it sits at a 256-aligned address.
alignas(256) constexpr uint32_t kLiveVertex[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
    0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr uint32_t kLiveWriter[] = {
    0x7e0002ffu, 0xbeef0001u, 0x7e0202ffu, 0x00020003u, 0xf8001c0fu, 0x00000100u, 0xbf810000u,
};

std::vector<uint32_t> realize_live(uint32_t cb_color0_info, uint32_t col_format) {
    namespace P = prosper::agc::Pm4;
    gpu::GpuState state;
    const auto set_program = [&](uint32_t lo, uint32_t hi, const void* program) {
        const auto address = reinterpret_cast<uint64_t>(program);
        state.sh[lo] = static_cast<uint32_t>(address >> 8);
        state.sh[hi] = static_cast<uint32_t>((address >> 40) & 0xffu);
    };
    set_program(P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kLiveVertex);
    set_program(P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kLiveWriter);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4u;
    state.cx[P::CB_TARGET_MASK] = 0xfu;
    state.cx[P::CB_COLOR_CONTROL] = P::CB_COLOR_CONTROL_MODE_NORMAL
                                    << P::CB_COLOR_CONTROL_MODE_SHIFT;
    state.cx[P::CB_COLOR0_INFO] = cb_color0_info;
    state.cx[P::SPI_SHADER_COL_FORMAT] = col_format;
    gpu::DrawItem draw;
    gpu::GpuState::Draw packet;
    packet.index_count = 3u;
    if (!gpu::realize_draw_item(state, &packet, 3u, std::size(kLiveVertex), false, draw)) return {};
    return draw.fs_words();
}
}   // namespace

TEST(IntegerColorExport, LiveDrawKeysOnTarget) {
    gpu::clear_shader_recompile_cache();
    const auto direct = [](FragmentExportFormats formats) {
        return recompile_fragment(kLiveWriter, std::size(kLiveWriter), nullptr, nullptr, UINT32_MAX,
                                  nullptr, false, {RecompileDiagnosticStage::Fragment, 0}, {},
                                  nullptr, {}, {}, formats);
    };
    // Kena's writer registers: CB_COLOR0_INFO 0x00050408 (COLOR_16, UINT), COL_FORMAT 0x7.
    const auto kena = realize_live(0x00050408u, 0x7u);
    ASSERT_FALSE(kena.empty());
    EXPECT_EQ(kena, direct(uint16_into_uint()));
    EXPECT_NE(kena, direct({}));
    // The same program into a float RGBA16F target with FP16 exports keeps the historical module.
    // COLOR_16_16_16_16 (0xC << 2) with NUMBER_TYPE FLOAT (7 << 8).
    const auto fp16 = realize_live((0xCu << 2) | (7u << 8), 0x4u);
    ASSERT_FALSE(fp16.empty());
    EXPECT_EQ(fp16, direct({}));
}

TEST(IntegerColorExport, RoundTrip) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan device";
    // (r, g) = (1, 0xBEEF): Kena's channel value, then a large value in the other half.
    for (const auto& [rg, expected] :
         {std::pair{0xbeef0001u, uint16_t(1)}, std::pair{0x0001beefu, uint16_t(0xbeef)}}) {
        const auto fs =
            recompile_fragment(compressed_writer(rg, 0x00020003u).data(), 7, nullptr, nullptr,
                               UINT32_MAX, nullptr, false, {RecompileDiagnosticStage::Fragment, 0},
                               {}, nullptr, {}, {}, uint16_into_uint());
        ASSERT_FALSE(fs.empty());
        const auto px = render(fs, VK_FORMAT_R16_UINT);
        ASSERT_EQ(px.size(), size_t(W) * H * 2u) << "R16_UINT is a 2-byte attachment";
        EXPECT_EQ(u16_at_center(px), expected) << std::hex << "rg=0x" << rg;
    }
}

TEST(IntegerColorExport, Fp16Control) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan device";
    // 0x3c00 = 1.0h in R, 0 in G; (b, a) = (0, 1.0h). Default formats: f16 into RGBA8 UNORM.
    const auto code = compressed_writer(0x00003c00u, 0x3c000000u);
    const auto fs = recompile_fragment(code.data(), code.size());
    ASSERT_FALSE(fs.empty());
    const auto px = render(fs, VK_FORMAT_R8G8B8A8_UNORM);
    ASSERT_EQ(px.size(), size_t(W) * H * 4u);
    const uint8_t* c = &px[kCenter * 4u];
    EXPECT_EQ(c[0], 255u);
    EXPECT_EQ(c[1], 0u);
    EXPECT_EQ(c[2], 0u);
    EXPECT_EQ(c[3], 255u);
}

TEST(IntegerColorExport, ReaderBranch) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan device";
    ShaderResourceTable rt;
    {
        ShaderResource image{};
        image.cls = ResourceClass::Texture;
        image.format = DataFormat::Uint16;
        image.num_components = 1;
        image.binding = 4;
        image.img_dim = 1;
        image.width = W;
        image.height = H;
        image.depth = 1;
        image.sgpr_base = 8;
        rt.resources.push_back(image);
    }
    const auto reader =
        recompile_fragment(kReaderPs, std::size(kReaderPs), &rt, nullptr, UINT32_MAX, nullptr,
                           false, {RecompileDiagnosticStage::Fragment, 0}, {}, nullptr, {}, {}, [] {
                               FragmentOutputClass classes[8]{};
                               classes[0] = FragmentOutputClass::Uint;
                               return make_fragment_export_formats(0x1u, classes);
                           }());
    ASSERT_FALSE(reader.empty());
    for (const auto& [rg, expected] : {std::pair{0x00000001u, 1u}, std::pair{0x0000beefu, 0xbeefu},
                                       std::pair{0x00000002u, 0u}}) {
        constexpr uint64_t kTarget = 0x4703000010000000ull;
        const auto writer = recompile_fragment(
            compressed_writer(rg, 0).data(), 7, nullptr, nullptr, UINT32_MAX, nullptr, false,
            {RecompileDiagnosticStage::Fragment, 0}, {}, nullptr, {}, {}, uint16_into_uint());
        ASSERT_FALSE(writer.empty());
        prosper::test::BackendColorTarget target;
        target.persistent_id = kTarget;
        target.load_existing = false;
        target.format = VK_FORMAT_R16_UINT;
        render(writer, VK_FORMAT_R16_UINT, {}, &target);
        ASSERT_NE(prosper::test::find_persistent_color_target(kTarget, W, H, VK_FORMAT_R16_UINT),
                  nullptr)
            << "the producer's R16_UINT target stays renderer-owned";

        prosper::test::FrameResource resource;
        resource.binding = 4;
        resource.set = 1;
        resource.tw = W;
        resource.th = H;
        resource.texture_format = VK_FORMAT_R16_UINT;
        resource.persistent_render_target_id = kTarget;
        resource.mag_filter = resource.min_filter = 0;
        const auto px = render(reader, VK_FORMAT_R32_UINT, {resource});
        ASSERT_EQ(px.size(), size_t(W) * H * 4u);
        EXPECT_EQ(prosper::test::backend_color_target_stats().sampled_hits, 1u)
            << "the reader bound the producer's GPU image, not an upload";
        // Channel 2 & mask 1 == 0: every lane is discarded and the cleared 0 survives.
        EXPECT_EQ(u32_at_center(px), expected) << std::hex << "rg=0x" << rg;
    }
}
