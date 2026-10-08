// A compute dispatch that samples a 16-bit depth surface reads it through a one-component UNORM16
// T#, which is how GFX10 views a Z16 plane. The live renderer keeps that plane as a retained D32
// Vulkan image and never writes depth back to guest memory, so compute must import the renderer's
// depth image: the guest bytes hold only what was there before the pass (the clear). UE4's
// volumetric fog samples its 16-bit shadow atlas this way; reading the clear made every froxel
// lit and washed Kena's title scene out (KENA_STATUS.md).
//
// Drives the real live renderer (producer + importer) and the real live compute backend.
// Mutation that turns the UNORM16 arm red: drop the `DataFormat::Unorm16` case from
// `depth_plane_view` (shared/rtt/depth_plane_view.hpp) -- the dispatch then reads the stale
// 0xFFFF guest bytes and observes 1.0 instead of the rendered 0.25.
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t Width = 64, Height = 8;
constexpr float RenderedDepth = 0.25f;

struct GuestMapping {
    uint64_t base = 0;
    size_t bytes = 0;
    ~GuestMapping() {
        if (!base) return;
        if (auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap")))
            unmap(base, bytes, 0, 0, 0, 0);
    }
};

// One compute item: image_load at (gid, 0) from the sampled T# in s[0:7], image_store of the
// loaded value to a one-component FLOAT32 storage image in s[8:15].
ComputeItem load_item(uint64_t source, DataFormat source_format, float* output,
                      const uint32_t* indices, const uint32_t* dummy) {
    ShaderResourceTable resources;
    for (uint32_t binding = 0; binding < 4; ++binding) {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = binding;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy : indices);
        buffer.size = binding ? 16 : Width * sizeof(uint32_t);
        resources.resources.push_back(buffer);
    }
    ShaderResource depth{};
    depth.cls = ResourceClass::Texture;
    depth.binding = 4;
    depth.sgpr_base = 0;
    depth.img_dim = 1;
    depth.width = Width;
    depth.height = Height;
    depth.depth = 1;
    depth.format = source_format;
    depth.num_components = 1;
    depth.gpu_addr = source;
    depth.size = Width * Height * (source_format == DataFormat::Unorm16 ? 2u : 4u);
    depth.swizzle[0] = 4;
    depth.swizzle[1] = 0;
    depth.swizzle[2] = 0;
    depth.swizzle[3] = 1;  // X001
    resources.resources.push_back(depth);
    ShaderResource out{};
    out.cls = ResourceClass::StorageImage;
    out.binding = 5;
    out.sgpr_base = 8;
    out.img_dim = 1;
    out.width = Width;
    out.height = 1;
    out.depth = 1;
    out.format = DataFormat::Float32;
    out.num_components = 1;
    out.gpu_addr = reinterpret_cast<uint64_t>(output);
    out.size = Width * sizeof(float);
    out.swizzle[0] = 4;
    out.swizzle[1] = 0;
    out.swizzle[2] = 0;
    out.swizzle[3] = 1;
    resources.resources.push_back(out);
    // v4 = gid (x), v5 = 0; image_load v[0:3] from s[0:7]; image_store v[0:3] to s[8:15].
    const uint32_t code[] = {
        0x7e080300u, 0x7e0a0280u, 0xf0000f08u, 0x00000004u,
        0xbf8c3f70u, 0xf0200f08u, 0x00020004u, 0xbf810000u,
    };
    ComputeItem item;
    item.spirv = recompile_valu(code, std::size(code), 1, 0, &resources);
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.threads_x = Width;
    item.launch.local_x = 64;
    item.launch.groups_x = Width / 64;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x16d16;
    return item;
}

} // namespace

TEST(LiveComputeDepthUnorm16, ComputeSamplesRendererDepthThroughUnorm16View) {
    prosper::register_builtin_hle();
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    ASSERT_TRUE(map);
    GuestMapping guest;
    guest.bytes = 0x200000;
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&guest.base), guest.bytes, 2, 0,
                  reinterpret_cast<uint64_t>("live-compute-depth-unorm16"), 0),
              0);
    ASSERT_NE(guest.base, 0u);
    const uint64_t depth = guest.base;              // rendered by the producer below
    const uint64_t untouched = guest.base + 0x80000;   // never rendered: guest bytes are authority
    // Every buffer the dispatch touches lives in the guest mapping, as in a real submit. A write
    // the mapping topology cannot place (a host heap address) conservatively revokes every
    // retained depth plane, which would turn each later arm into a test of that rule instead.
    auto* indices = reinterpret_cast<uint32_t*>(guest.base + 0x100000);
    auto* dummy = reinterpret_cast<uint32_t*>(guest.base + 0x110000);
    auto* output = reinterpret_cast<float*>(guest.base + 0x120000);
    for (uint32_t i = 0; i < Width; ++i) indices[i] = i;
    std::memset(dummy, 0, 16);
    // The guest bytes of both planes hold the clear, all ones: 1.0 through either view.
    constexpr size_t PlaneBytes = size_t(Width) * Height * 4;
    std::memset(reinterpret_cast<void*>(depth), 0xff, PlaneBytes);
    std::memset(reinterpret_cast<void*>(untouched), 0xff, PlaneBytes);

    const uint32_t vs_words[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u,
                              0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u,
                              0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    const uint32_t fs_words[]{0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
                              0xF800180Fu, 0x03020100u, 0xBF810000u};
    const auto vs = recompile_vertex(vs_words, std::size(vs_words));
    const auto fs = recompile_fragment(fs_words, std::size(fs_words));
    ASSERT_FALSE(vs.empty() || fs.empty());
    prosper::frontend::register_live_renderer(".", false);

    // A depth-only pass writing RenderedDepth over a Z16 plane, as a shadow-map pass does.
    DrawItem producer;
    producer.vs = vs;
    producer.fs = fs;
    producer.vertex_count = 3;
    producer.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    producer.ps.color_write_mask = 0;
    producer.ps.depth_test_enable = producer.ps.depth_write_enable = true;
    producer.ps.depth_compare_op = VK_COMPARE_OP_ALWAYS;
    producer.ps.depth_read_base = producer.ps.depth_write_base = depth;
    producer.ps.db_z_info = 1;   // Z_16
    producer.ps.db_depth_size_xy = (Width - 1) | ((Height - 1) << 16);
    producer.ps.has_viewport = true;
    producer.ps.viewport_w = float(Width);
    producer.ps.viewport_h = float(Height);
    producer.ps.min_depth = producer.ps.max_depth = RenderedDepth;
    (void)render_submit_items({producer}, Width, Height);

    // Positive control on the premise: the renderer holds the depth, the guest bytes do not.
    // Without this, a renderer that wrote depth back would make every arm below pass for a
    // reason that has nothing to do with the import.
    {
        std::vector<float> values;
        std::string error;   // the reader takes the backend resource lock itself
        ASSERT_EQ(prosper::test::read_persistent_ds_depth_array(depth, Width, Height, 0, 1, values,
                                                                error),
                  prosper::test::PersistentDsDepthArrayStatus::Ready)
            << error;
        ASSERT_EQ(values.size(), size_t(Width) * Height);
        EXPECT_EQ(values[0], RenderedDepth) << "the renderer retains the rendered depth";
        const auto* bytes = reinterpret_cast<const uint8_t*>(depth);
        bool stale = true;
        for (size_t i = 0; i < size_t(Width) * Height * 2; ++i) stale = stale && bytes[i] == 0xff;
        ASSERT_TRUE(stale) << "the guest bytes still hold the clear: only an import can see 0.25";
    }

    struct Arm {
        const char* name;
        uint64_t address;
        DataFormat format;
        float expected;
    };
    const Arm arms[] = {
        // The fix: the Z16 view of a renderer-held plane observes the rendered depth. A D16
        // surface would hold round(0.25 * 65535); the D32 image holds 0.25 exactly, which is
        // within that quantization step.
        {"UNORM16 view of a rendered Z16 plane", depth, DataFormat::Unorm16, RenderedDepth},
        // The established FLOAT32 view of the same plane, unchanged by the fix.
        {"FLOAT32 view of a rendered plane", depth, DataFormat::Float32, RenderedDepth},
        // A plane the renderer never drew keeps the guest bytes as its authority.
        {"UNORM16 view of an unrendered plane", untouched, DataFormat::Unorm16, 1.0f},
    };
    for (const Arm& arm : arms) {
        std::fill(output, output + Width, -99.0f);
        ComputeItem item = load_item(arm.address, arm.format, output, indices, dummy);
        ASSERT_FALSE(item.spirv.empty()) << arm.name;
        ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item})) << arm.name;
        bool all = true;
        for (uint32_t x = 0; x < Width; ++x)
            all = all && std::isfinite(output[x]) &&
                  std::fabs(output[x] - arm.expected) <= 1.0f / 65535.0f;
        EXPECT_TRUE(all) << arm.name << ": expected " << arm.expected << ", texel 0 = " << output[0]
                         << ", texel " << Width - 1 << " = " << output[Width - 1];
    }
}
