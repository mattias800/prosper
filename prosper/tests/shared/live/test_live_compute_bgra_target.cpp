// #4291: a CB_COLOR ALT (BGRA) render target is kept by the renderer as canonical RGBA8. Compute
// must still observe the GUEST's component order through every handoff the live renderer offers:
// the CPU snapshot, the in-place sampled view, the exact-bits copy and the storage seed. Drives the
// real live renderer and live compute backend, so each arm reaches the production path; the
// mutation that removes each path's order handling turns its arm red.
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t Width = 256;

struct GuestMapping {
    uint64_t base = 0;
    size_t bytes = 0;
    ~GuestMapping() {
        if (!base) return;
        if (auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap")))
            unmap(base, bytes, 0, 0, 0, 0);
    }
};

// One compute item: image_load RGBA at (gid, 0) from s[0:7], image_store it to s[8:15].
ComputeItem copy_item(uint64_t source, ResourceClass source_class, DataFormat source_format,
                      DataFormat output_format, void* output, std::vector<uint32_t>& indices,
                      std::vector<uint32_t>& dummy) {
    ShaderResourceTable resources;
    for (uint32_t binding = 0; binding < 4; ++binding) {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = binding;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy.data() : indices.data());
        buffer.size = binding ? 16 : Width * sizeof(uint32_t);
        resources.resources.push_back(buffer);
    }
    for (uint32_t binding : {4u, 5u}) {
        ShaderResource image{};
        image.cls = binding == 4 ? source_class : ResourceClass::StorageImage;
        image.binding = binding; image.sgpr_base = binding == 4 ? 0 : 8;
        image.img_dim = 1; image.width = Width; image.height = image.depth = 1;
        image.format = binding == 4 ? source_format : output_format;
        image.num_components = 4;
        image.gpu_addr = binding == 4 ? source : reinterpret_cast<uint64_t>(output);
        image.size = Width * (binding == 4 || output_format != DataFormat::Float32 ? 4 : 16);
        for (uint32_t c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;  // identity X,Y,Z,W
        resources.resources.push_back(image);
    }
    const uint32_t code[] = {
        0x7e080300u, 0x7e0a0280u, 0xf0000f08u, 0x00000004u, 0xbf8c3f70u,
        0xf0200f08u, 0x00020004u, 0xbf810000u,
    };
    ComputeItem item;
    item.spirv = recompile_valu(code, std::size(code), 1, 0, &resources);
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.threads_x = Width;
    item.launch.local_x = 64; item.launch.groups_x = Width / 64;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x4291;
    return item;
}

} // namespace

TEST(LiveComputeBgraTarget, ComputeObservesGuestComponentOrder) {
    prosper::register_builtin_hle();
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    ASSERT_TRUE(map);
    GuestMapping guest;
    guest.bytes = 0x200000;
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&guest.base), guest.bytes, 2, 0,
                  reinterpret_cast<uint64_t>("live-compute-bgra-target"), 0), 0);
    ASSERT_NE(guest.base, 0u);
    const uint64_t color = guest.base + 0x100000;

    // Fullscreen triangle writing the constant (1, 0.5, 0, 1): R and B differ, so any exchange
    // of the two is visible in every texel.
    const uint32_t vs_words[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    const uint32_t fs_words[]{0x7E0002F2u, 0x7E0202F0u, 0x7E040280u,
        0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    const auto vs = recompile_vertex(vs_words, std::size(vs_words));
    const auto fs = recompile_fragment(fs_words, std::size(fs_words));
    ASSERT_FALSE(vs.empty() || fs.empty());
    prosper::frontend::register_live_renderer(".", false);

    DrawItem producer;
    producer.vs = vs; producer.fs = fs; producer.vertex_count = 3;
    producer.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    producer.ps.color_write_mask = 0xf;
    producer.color0_base = color; producer.color0_width = Width; producer.color0_height = 1;
    producer.color_targets[0] = {color, Width, 1};
    producer.ps.color_targets[0].format = VK_FORMAT_B8G8R8A8_UNORM;  // CB_COLOR ALT
    producer.ps.color_targets[0].write_mask = 0xf;
    auto produce = [&] { (void)render_submit_items({producer}, Width, 1); };
    produce();

    // The renderer image itself is canonical RGBA: red first.
    std::vector<uint8_t> canonical;
    {
        std::string error;
        prosper::test::BackendPersistentResourceGuard guard;
        ASSERT_TRUE(prosper::test::readback_persistent_color_target(
            color, Width, 1, VK_FORMAT_R8G8B8A8_UNORM, canonical, error)) << error;
    }
    ASSERT_EQ(canonical.size(), size_t(Width) * 4);
    ASSERT_EQ(canonical[0], 255u);
    ASSERT_EQ(canonical[2], 0u);
    std::vector<uint8_t> guest_order = canonical;
    for (size_t i = 0; i < guest_order.size(); i += 4) std::swap(guest_order[i], guest_order[i + 2]);

    // Snapshot: compute decodes these bytes as guest memory, so they arrive B,G,R,A -- and
    // reading twice must not swap twice (the cached surface stays canonical).
    for (int read = 0; read < 2; ++read) {
        LiveTargetSnapshot snapshot;
        ASSERT_TRUE(read_live_render_target(color, snapshot) && snapshot.pixels);
        EXPECT_EQ(*snapshot.pixels, guest_order)
            << "snapshot read " << read << " hands compute the guest's BGRA byte order";
    }

    std::vector<uint32_t> indices(Width), dummy(4, 0);
    for (uint32_t i = 0; i < Width; ++i) indices[i] = i;
    struct Arm { const char* name; ResourceClass cls; DataFormat format; };
    const Arm arms[] = {
        // In-place sampled view: the selector must be composed with the R<->B swap.
        {"sampled UNORM view", ResourceClass::Texture, DataFormat::Unorm8},
        // Exact-bits copy: a raw copy cannot reorder, so it must decline to the snapshot.
        {"sampled UINT copy", ResourceClass::Texture, DataFormat::Uint8},
        // Storage: the identity storage view cannot be swizzled; the raw seed must decline.
        {"storage load", ResourceClass::StorageImage, DataFormat::Unorm8},
    };
    for (const Arm& arm : arms) {
        produce();  // re-establish GPU authority in case an earlier arm materialized a snapshot
        std::vector<uint8_t> output(size_t(Width) * 4, 0xee);
        ComputeItem item = copy_item(color, arm.cls, arm.format, arm.format, output.data(),
                                     indices, dummy);
        ASSERT_FALSE(item.spirv.empty()) << arm.name;
        ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item})) << arm.name;
        EXPECT_EQ(output, guest_order) << arm.name << ": compute observes the guest's BGRA order";
    }

    // Packed-10 view: the GPU conversion reads the renderer image's components in place, so it
    // must decline and let the CPU reconstruct from the guest-order snapshot. Observed as FLOAT32.
    {
        produce();
        std::vector<float> floats(size_t(Width) * 4, -99.0f);
        ComputeItem item = copy_item(color, ResourceClass::Texture, DataFormat::Unorm2_10_10_10,
                                     DataFormat::Float32, floats.data(), indices, dummy);
        ASSERT_FALSE(item.spirv.empty());
        ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
        bool all = true;
        for (size_t i = 0; i < floats.size(); ++i) {
            const uint32_t scale = i % 4 == 3 ? 3u : 1023u;
            const uint32_t q = (guest_order[i] * scale + 127u) / 255u;
            all = all && floats[i] == float(q) / float(scale);
        }
        EXPECT_TRUE(all) << "packed-10 view observes the guest's BGRA order; texel 0 = "
                         << floats[0] << "," << floats[1] << "," << floats[2] << "," << floats[3];
    }

    // Partial native-float storage write: EXEC limits the store to texels 0..63, so the rest of
    // the result must come from the old target contents. The renderer image cannot seed it by a
    // raw copy (canonical order); the guest-order snapshot must.
    produce();
    ShaderResourceTable storage_table;
    {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = 2;
        buffer.format = DataFormat::Uint32; buffer.num_components = 1;
        buffer.stride = 4; buffer.size = 16;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(dummy.data());
        ShaderResource image{};
        image.cls = ResourceClass::StorageImage;
        image.binding = 5; image.sgpr_base = 8;
        image.img_dim = 1; image.width = Width; image.height = image.depth = 1;
        image.format = DataFormat::Unorm8; image.num_components = 4;
        image.gpu_addr = color; image.size = Width * 4;
        for (uint32_t c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;
        storage_table.resources = {buffer, image};
    }
    const std::vector<uint32_t> partial_writer{
        0x7e080300u,               // v4 = thread x
        0x7e0a0280u,               // v5 = 0
        0x7e0002ffu, 0x3f800000u,  // v0 = 1.0
        0x7e020280u, 0x7e040280u,  // v1 = v2 = 0
        0x7e0602ffu, 0x3f800000u,  // v3 = 1.0
        0x7da808c0u,               // v_cmpx_gt_u32: EXEC = 64 > v4
        0xf0200f08u, 0x00020004u,  // image_store v[0:3] at (v4, v5) to s[8:15]
        0xbf810000u,
    };
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = Width; config.local_y = config.local_z = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Unorm8, 4);
    ComputeItem writer;
    writer.spirv = recompile_compute(partial_writer.data(), partial_writer.size(), &storage_table, config);
    ASSERT_FALSE(writer.spirv.empty());
    writer.resources = std::make_shared<ShaderResourceTable>(storage_table);
    writer.user_sgprs = config.user_sgprs;
    writer.code_addr = 0x42910;
    writer.launch.threads_x = writer.launch.local_x = Width;
    writer.launch.threads_y = writer.launch.threads_z = 1;
    writer.launch.local_y = writer.launch.local_z = 1;
    writer.launch.groups_x = writer.launch.groups_y = writer.launch.groups_z = 1;
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({writer}));
    std::vector<uint8_t> expected = guest_order;
    for (uint32_t x = 0; x < 64; ++x) {
        const uint8_t written[4] = {255, 0, 0, 255};
        std::memcpy(expected.data() + x * 4, written, 4);
    }
    LiveTargetSnapshot result;
    ASSERT_TRUE(read_live_render_target(color, result) && result.pixels);
    EXPECT_EQ(*result.pixels, expected)
        << "texels the partial write did not touch keep the guest's BGRA order";

    // The same partial write with a sampled sibling of the target, whose loaded value is what the
    // store writes back. The sibling borrows the renderer image in place (swizzled view); the
    // storage binding must not seed from that borrowed image by a raw copy.
    produce();
    ShaderResourceTable sibling_table = storage_table;
    {
        ShaderResource sampled = storage_table.resources[1];
        sampled.cls = ResourceClass::Texture;
        sampled.binding = 4; sampled.sgpr_base = 16;
        sibling_table.resources.insert(sibling_table.resources.begin() + 1, sampled);
    }
    const std::vector<uint32_t> sibling_writer{
        0x7e080300u, 0x7e0a0280u,               // v4 = thread x, v5 = 0
        0xf0000f08u, 0x00040c04u, 0xbf8c3f70u,  // image_load v[12:15] from s[16:23]
        0x7da808c0u,                            // EXEC = 64 > v4
        0xf0200f08u, 0x00020c04u,               // image_store v[12:15] to s[8:15]
        0xbf810000u,
    };
    writer.spirv = recompile_compute(sibling_writer.data(), sibling_writer.size(), &sibling_table, config);
    ASSERT_FALSE(writer.spirv.empty());
    writer.resources = std::make_shared<ShaderResourceTable>(sibling_table);
    writer.code_addr = 0x42911;
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({writer}));
    ASSERT_TRUE(read_live_render_target(color, result) && result.pixels);
    EXPECT_EQ(*result.pixels, guest_order)
        << "a sampled-sibling partial write keeps every texel in the guest's BGRA order";
}
