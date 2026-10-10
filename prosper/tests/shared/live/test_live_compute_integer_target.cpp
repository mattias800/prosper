// Integer graphics targets must reach compute as current producer bytes, rather than an
// unavailable live-target format or stale guest backing. The synthetic producer writes R16_UINT
// through the real renderer; CPU snapshot, sampled load and storage load observe the same words.
#include <gtest/gtest.h>

#include "fixtures/render_runner.h"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/state/fragment_export_state.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

namespace gpu = prosper::gpu;

namespace {

constexpr uint32_t Width = 64;
constexpr uint16_t ProducerWord = 0x5aa5;
constexpr uint16_t GuestWord = 0x1234;

class LiveComputeIntegerTarget : public testing::Test {
protected:
    uint64_t allocation = 0;
    uint64_t color = 0;
    gpu::DrawItem producer;

    void SetUp() override {
        prosper::register_builtin_hle();
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        ASSERT_TRUE(map);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&allocation), 0x200000, 2, 0,
                      reinterpret_cast<uint64_t>("integer-render-target"), 0),
                  0);
        ASSERT_NE(allocation, 0u);
        color = allocation + 0x100000;
        auto* guest = reinterpret_cast<uint16_t*>(color);
        std::fill_n(guest, Width, GuestWord);

        const uint32_t vs_words[] = {
            0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
            0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
            0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
        };
        const uint32_t ps_words[] = {
            0x7e0002ffu, uint32_t(ProducerWord), 0x7e020280u, 0xf8001c0fu, 0x00000100u, 0xbf810000u,
        };
        gpu::FragmentOutputClass classes[8]{};
        classes[0] = gpu::FragmentOutputClass::Uint;
        const auto formats = gpu::make_fragment_export_formats(0x7u, classes);
        producer.vs = gpu::recompile_vertex(vs_words, std::size(vs_words));
        producer.fs = gpu::recompile_fragment(
            ps_words, std::size(ps_words), nullptr, nullptr, UINT32_MAX, nullptr, false,
            {gpu::RecompileDiagnosticStage::Fragment, 0}, {}, nullptr, {}, {}, formats);
        ASSERT_FALSE(producer.vs.empty());
        ASSERT_FALSE(producer.fs.empty());
        producer.vertex_count = 3;
        producer.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        producer.ps.color_write_mask = 0xf;
        producer.color0_base = color;
        producer.color0_width = Width;
        producer.color0_height = 1;
        producer.color_targets[0] = {color, Width, 1};
        producer.ps.color_targets[0].format = VK_FORMAT_R16_UINT;
        producer.ps.color_targets[0].write_mask = 0xf;
        prosper::frontend::register_live_renderer(".", false);
        (void)gpu::render_submit_items({producer}, Width, 1);
        ASSERT_TRUE(gpu::is_live_render_target(color));
        ASSERT_EQ(guest[0], GuestWord) << "renderer-owned pixels must differ from guest backing";
    }

    void TearDown() override {
        if (allocation) {
            const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
            if (unmap) unmap(allocation, 0x200000, 0, 0, 0, 0);
        }
    }

    void ComputeReads(gpu::ResourceClass source_class) {
        std::vector<uint32_t> indices(Width), dummy(4, 0);
        for (uint32_t i = 0; i < Width; ++i) indices[i] = i;
        std::vector<uint16_t> output(Width, 0xeeee);
        gpu::ShaderResourceTable table;
        for (uint32_t binding = 0; binding < 4; ++binding) {
            gpu::ShaderResource buffer{};
            buffer.cls = gpu::ResourceClass::ConstantBuffer;
            buffer.binding = binding;
            buffer.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy.data() : indices.data());
            buffer.size = binding ? 16 : Width * sizeof(uint32_t);
            table.resources.push_back(buffer);
        }
        for (uint32_t binding : {4u, 5u}) {
            gpu::ShaderResource image{};
            image.cls = binding == 4 ? source_class : gpu::ResourceClass::StorageImage;
            image.binding = binding;
            image.sgpr_base = binding == 4 ? 0 : 8;
            image.img_dim = 1;
            image.width = Width;
            image.height = image.depth = 1;
            image.format = gpu::DataFormat::Uint16;
            image.num_components = 1;
            image.gpu_addr = binding == 4 ? color : reinterpret_cast<uint64_t>(output.data());
            image.size = Width * sizeof(uint16_t);
            image.swizzle[0] = 4;
            image.swizzle[1] = image.swizzle[2] = 0;
            image.swizzle[3] = 1;
            table.resources.push_back(image);
        }
        const uint32_t code[] = {
            0x7e080300u, 0x7e0a0280u, 0xf0000108u, 0x00000004u,
            0xbf8c3f70u, 0xf0200108u, 0x00020004u, 0xbf810000u,
        };
        gpu::ComputeItem item;
        item.spirv = gpu::recompile_valu(code, std::size(code), 1, 0, &table);
        ASSERT_FALSE(item.spirv.empty());
        item.resources = std::make_shared<gpu::ShaderResourceTable>(table);
        item.launch.threads_x = Width;
        item.launch.local_x = 64;
        item.launch.groups_x = Width / 64;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x4849;
        ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
        EXPECT_EQ(output, std::vector<uint16_t>(Width, ProducerWord));
    }
};

TEST_F(LiveComputeIntegerTarget, SnapshotPreservesCurrentProducerWords) {
    gpu::LiveTargetSnapshot snapshot;
    ASSERT_TRUE(gpu::read_live_render_target(color, snapshot));
    ASSERT_TRUE(snapshot.pixels);
    ASSERT_EQ(snapshot.pixels->size(), Width * sizeof(uint16_t));
    for (uint32_t i = 0; i < Width; ++i) {
        uint16_t actual = 0;
        std::memcpy(&actual, snapshot.pixels->data() + i * sizeof(uint16_t), sizeof(actual));
        EXPECT_EQ(actual, ProducerWord) << "texel " << i;
    }
}

TEST_F(LiveComputeIntegerTarget, SampledLoadReadsTheRendererProducer) {
    ComputeReads(gpu::ResourceClass::Texture);
}

TEST_F(LiveComputeIntegerTarget, SampledLoadReadsCpuOnlyProducerSnapshot) {
    gpu::LiveTargetSnapshot snapshot;
    ASSERT_TRUE(gpu::read_live_render_target(color, snapshot));
    ASSERT_TRUE(snapshot.pixels);
    // Republishing the real producer's bytes as a CPU-only target forces the snapshot upload arm.
    gpu::notify_live_render_target_image_written(
        {color, Width, 1, snapshot.format, snapshot.pixels});
    gpu::LiveTargetImageImport import;
    EXPECT_FALSE(gpu::import_live_render_target_image(color, {Width, 1}, import));
    ComputeReads(gpu::ResourceClass::Texture);
}

TEST_F(LiveComputeIntegerTarget, StorageLoadReadsTheRendererProducer) {
    ComputeReads(gpu::ResourceClass::StorageImage);
}

}   // namespace

TEST(LiveComputeIntegerTargetContract, TypedImportsRejectFloatAndSignedViews) {
    using gpu::DataFormat;
    using gpu::LiveTargetPixelFormat;
    EXPECT_TRUE(prosper::frontend::direct_sampled_rtt_compatible(
        DataFormat::Uint16, 1, LiveTargetPixelFormat::R16Uint, false));
    for (DataFormat alias : {DataFormat::Float16, DataFormat::Unorm16, DataFormat::Sint16}) {
        EXPECT_FALSE(prosper::frontend::direct_sampled_rtt_compatible(
            alias, 1, LiveTargetPixelFormat::R16Uint, false));
        EXPECT_FALSE(prosper::frontend::sampled_rtt_snapshot_byte_compatible(
            alias, 1, LiveTargetPixelFormat::R16Uint));
    }
    EXPECT_FALSE(prosper::frontend::direct_sampled_rtt_compatible(
        DataFormat::Uint16, 2, LiveTargetPixelFormat::R16Uint, false));
    EXPECT_FALSE(prosper::frontend::direct_sampled_rtt_compatible(
        DataFormat::Uint16, 0, LiveTargetPixelFormat::R16Uint, false));
    EXPECT_FALSE(prosper::frontend::direct_sampled_rtt_compatible(
        DataFormat::Float32, 1, LiveTargetPixelFormat::R32Uint, false));
}
