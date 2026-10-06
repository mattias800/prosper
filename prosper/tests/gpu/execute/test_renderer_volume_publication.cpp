// #4625: a compute dispatch that binds a renderer-claimed volume publishes that volume to guest
// memory first and then runs on the published bytes, instead of being skipped. The CPU cases pin the
// publication plan and its tiling; the Vulkan case runs a real 3D storage dispatch through the live
// compute backend (Kena's translucency-lighting clear binds four claimed 64^3 volumes this way).
#include "gpu/execute/renderer_volume_publication.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kW = 64, kH = 8, kD = 8, kBpe = 8;
constexpr uint32_t kMode = static_cast<uint32_t>(TileMode::Sw64KbRX);   // Kena's volume mode (27)

size_t tiled_bytes() {
    return tiled_volume_bytes(kW, kH, kD, kMode, kBpe);
}
size_t linear_bytes() {
    return static_cast<size_t>(kW) * kH * kD * kBpe;
}

std::vector<uint8_t> pattern(size_t bytes, uint32_t mul, uint32_t add) {
    std::vector<uint8_t> out(bytes);
    for (size_t i = 0; i < bytes / 2; ++i) {
        const uint16_t v = static_cast<uint16_t>((i * mul + add) & 0xffffu);
        std::memcpy(out.data() + i * 2, &v, 2);
    }
    return out;
}

VolumePublicationSource claimed(const std::vector<uint8_t>& guest) {
    VolumePublicationSource source;
    source.base = reinterpret_cast<uint64_t>(guest.data());
    source.claimed_bytes = guest.size();
    source.footprint_proven = true;
    source.renderer_image_valid = true;
    source.image_width = kW;
    source.image_height = kH;
    source.image_depth = kD;
    source.layout = {kW, kH, kD, kMode, kBpe};
    source.exact_representation = true;
    return source;
}

struct HookReset {
    ~HookReset() {
        set_unpublished_volume_query({});
        set_renderer_volume_publisher({});
    }
};

}   // namespace

TEST(RendererVolumePublication, PlanNamesEveryRefusal) {
    std::vector<uint8_t> guest(tiled_bytes());
    const auto base = claimed(guest);
    ASSERT_EQ(plan_volume_publication(base), VolumePublication::Published);

    auto source = base;
    source.claimed_bytes = 0;
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::NothingToPublish);
    source = base;
    source.renderer_image_valid = false;
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::NoRendererImage);
    source = base;
    source.image_depth = 0;   // a 2D alias replaced the retained volume
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::NoRendererImage);
    source = base;
    source.footprint_proven = false;
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::UnprovenFootprint);
    source = base;
    source.layout = {};
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::UnknownLayout);
    source = base;
    source.layout.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbZX);
    ASSERT_FALSE(tile_mode_supports_volume(source.layout.tile_mode));
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::UnknownLayout);
    source = base;
    source.image_width = kW * 2;   // a scaled image tiles to the wrong addresses
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::ExtentMismatch);
    source = base;
    source.claimed_bytes = guest.size() + 1;
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::ExtentMismatch);
    source = base;
    source.exact_representation = false;
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::FormatConversion);
    source = base;
    source.overlapping_alias = true;
    EXPECT_EQ(plan_volume_publication(source), VolumePublication::OverlappingAlias);
}

TEST(RendererVolumePublication, PublishTilesTheRetainedImageIntoTheNativeLayout) {
    const auto rendered = pattern(linear_bytes(), 313u, 7u);
    std::vector<uint8_t> guest(tiled_bytes(), 0x5a), expected(tiled_bytes(), 0);
    ASSERT_TRUE(
        tile_volume(expected.data(), expected.size(), rendered.data(), kW, kH, kD, kMode, kBpe));
    // The renderer's own aliases and every guest-memory cache learn of the write through the
    // guest-GPU write notification, which is what retires the retained image afterwards.
    std::vector<std::pair<uint64_t, uint64_t>> notified;
    set_guest_gpu_write_observer(
        [&](uint64_t addr, uint64_t size, const char*) { notified.emplace_back(addr, size); });
    const auto result = publish_volume_to_guest(
        claimed(guest),
        [&](std::vector<uint8_t>& linear) {
            linear = rendered;
            return true;
        },
        guest.data());
    ASSERT_EQ(result, VolumePublication::Published);
    EXPECT_EQ(guest, expected);
    EXPECT_EQ(notified, (std::vector<std::pair<uint64_t, uint64_t>>{
                            {reinterpret_cast<uint64_t>(guest.data()), guest.size()}}));
    notified.clear();

    // A refusal writes nothing.
    std::vector<uint8_t> untouched(tiled_bytes(), 0x5a);
    auto short_read = claimed(untouched);
    EXPECT_EQ(publish_volume_to_guest(
                  short_read,
                  [&](std::vector<uint8_t>& linear) {
                      linear.assign(rendered.begin(), rendered.end() - 1);
                      return true;
                  },
                  untouched.data()),
              VolumePublication::ReadbackFailed);
    EXPECT_EQ(publish_volume_to_guest(
                  short_read, [](std::vector<uint8_t>&) { return false; }, untouched.data()),
              VolumePublication::ReadbackFailed);
    short_read.exact_representation = false;
    bool read = false;
    EXPECT_EQ(publish_volume_to_guest(
                  short_read,
                  [&](std::vector<uint8_t>&) {
                      read = true;
                      return true;
                  },
                  untouched.data()),
              VolumePublication::FormatConversion);
    EXPECT_FALSE(read) << "a planned refusal must not pay for the readback";
    EXPECT_EQ(untouched, std::vector<uint8_t>(tiled_bytes(), 0x5a));
    EXPECT_TRUE(notified.empty()) << "a refusal notifies no write";
    set_guest_gpu_write_observer({});
}

TEST(RendererVolumePublication, ComputeGateAsksThePublisherAndRechecksTheClaim) {
    HookReset reset;
    bool claimed_volume = true;
    int asked = 0;
    VolumePublication answer = VolumePublication::Published;
    bool release = true;
    set_unpublished_volume_query([&](uint64_t, uint64_t) { return claimed_volume; });
    set_renderer_volume_publisher([&](uint64_t, uint64_t) {
        ++asked;
        if (answer == VolumePublication::Published && release) claimed_volume = false;
        return answer;
    });
    EXPECT_EQ(compute_renderer_volume_refusal(0x1000, 64, false), nullptr)
        << "a 2D binding the renderer serves directly needs no publication";
    EXPECT_EQ(asked, 0);
    EXPECT_EQ(compute_renderer_volume_refusal(0x1000, 64, true), nullptr);
    EXPECT_EQ(asked, 1);
    EXPECT_EQ(compute_renderer_volume_refusal(0x1000, 64, true), nullptr)
        << "a released claim no longer asks";
    EXPECT_EQ(asked, 1);

    claimed_volume = true;
    release = false;
    const char* why = compute_renderer_volume_refusal(0x1000, 64, true);
    ASSERT_NE(why, nullptr) << "a publisher that reports success but keeps the claim is refused";
    EXPECT_NE(std::strstr(why, "claim not released"), nullptr) << why;

    answer = VolumePublication::FormatConversion;
    why = compute_renderer_volume_refusal(0x1000, 64, true);
    ASSERT_NE(why, nullptr);
    EXPECT_NE(std::strstr(why, "format conversion"), nullptr) << why;

    set_renderer_volume_publisher({});
    why = compute_renderer_volume_refusal(0x1000, 64, true);
    ASSERT_NE(why, nullptr);
    EXPECT_NE(std::strstr(why, "no publisher"), nullptr) << why;
}

// The executed case. The guest bytes under the source volume are stale ("before the renderer drew
// it"); the renderer holds the current volume. The copy kernel moves row (x, 0, 0) of the source
// into the destination, so the destination row says which of the two the dispatch actually read.
TEST(RendererVolumePublication, ClaimedVolumeIsPublishedThenTheDispatchReadsIt) {
    HookReset reset;
    // v4 = x, v5 = y = 0, v6 = z = 0; image_load DIM=3D s[0:7] -> v[0:3]; image_store DIM=3D s[8:15].
    static const uint32_t image_copy_3d[] = {
        0x7E080300u, 0x7E0A0280u, 0x7E0C0280u, 0xF0000F10u, 0x00000004u,
        0xBF8C3F70u, 0xF0200F10u, 0x00020004u, 0xBF810000u,
    };
    std::vector<uint32_t> lane_index(kW);
    for (uint32_t i = 0; i < kW; ++i) lane_index[i] = i;
    std::vector<uint32_t> dummy(4, 0);
    const auto rendered = pattern(linear_bytes(), 313u, 7u);
    const auto stale = pattern(linear_bytes(), 101u, 3u);
    const auto dst_initial = pattern(linear_bytes(), 197u, 17u);
    std::vector<uint8_t> src(tiled_bytes()), dst(tiled_bytes());
    ASSERT_TRUE(tile_volume(src.data(), src.size(), stale.data(), kW, kH, kD, kMode, kBpe));
    ASSERT_TRUE(tile_volume(dst.data(), dst.size(), dst_initial.data(), kW, kH, kD, kMode, kBpe));

    ShaderResourceTable table;
    auto add_buffer = [&](uint32_t binding, void* data, uint32_t size) {
        ShaderResource b{};
        b.cls = ResourceClass::ConstantBuffer;
        b.binding = binding;
        b.gpu_addr = reinterpret_cast<uint64_t>(data);
        b.size = size;
        table.resources.push_back(b);
    };
    add_buffer(0, lane_index.data(), kW * sizeof(uint32_t));
    for (uint32_t binding = 1; binding <= 3; ++binding) add_buffer(binding, dummy.data(), 16);
    auto add_volume = [&](uint32_t binding, uint32_t sgpr, std::vector<uint8_t>& data) {
        ShaderResource im{};
        im.cls = ResourceClass::StorageImage;
        im.img_dim = 2;
        im.binding = binding;
        im.sgpr_base = sgpr;
        im.format = DataFormat::Unorm16;
        im.num_components = 4;
        im.width = kW;
        im.height = kH;
        im.depth = kD;
        im.tile_mode = kMode;
        im.gpu_addr = reinterpret_cast<uint64_t>(data.data());
        im.size = static_cast<uint32_t>(data.size());
        table.resources.push_back(im);
    };
    add_volume(4, 0, src);
    add_volume(5, 8, dst);
    const auto spirv = recompile_valu(image_copy_3d, std::size(image_copy_3d), 1, 0, &table);
    ASSERT_FALSE(spirv.empty());
    ComputeItem item;
    item.spirv = spirv;
    item.resources = std::make_shared<ShaderResourceTable>(table);
    item.launch.threads_x = kW;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_x = 64;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.code_addr = 0x4625000000ull;

    // Positive control first: unclaimed, the dispatch reads the guest bytes it is given. This also
    // warms the backend's retained source image, so the published arm below must be told the bytes
    // changed (the publication's write notification) or it would reuse the stale upload.
    if (!prosper::frontend::execute_live_compute_items({item}))
        GTEST_SKIP() << "no live compute device";
    std::vector<uint8_t> result(linear_bytes());
    ASSERT_TRUE(detile_volume(result.data(), dst.data(), dst.size(), kW, kH, kD, kMode, kBpe));
    const size_t row = static_cast<size_t>(kW) * kBpe;
    ASSERT_TRUE(std::equal(result.begin(), result.begin() + row, stale.begin()));
    ASSERT_TRUE(tile_volume(dst.data(), dst.size(), dst_initial.data(), kW, kH, kD, kMode, kBpe));
    prosper::gpu::notify_guest_gpu_write(reinterpret_cast<uint64_t>(dst.data()), dst.size());

    const uint64_t src_base = reinterpret_cast<uint64_t>(src.data());
    bool src_claimed = true;
    VolumePublicationSource renderer = claimed(src);
    int publications = 0;
    set_unpublished_volume_query([&](uint64_t addr, uint64_t bytes) {
        return src_claimed && addr < src_base + src.size() && src_base < addr + bytes;
    });
    set_renderer_volume_publisher([&](uint64_t, uint64_t) {
        ++publications;
        const auto result = publish_volume_to_guest(
            renderer,
            [&](std::vector<uint8_t>& linear) {
                linear = rendered;
                return true;
            },
            src.data());
        if (result == VolumePublication::Published) src_claimed = false;
        return result;
    });

    // Control: a publication the renderer must refuse keeps the dispatch skipped and both
    // volumes untouched -- guest bytes never stand in for the claimed volume.
    renderer.exact_representation = false;
    const auto src_before = src, dst_before = dst;
    EXPECT_FALSE(prosper::frontend::execute_live_compute_items({item}));
    EXPECT_EQ(publications, 1);
    EXPECT_EQ(src, src_before);
    EXPECT_EQ(dst, dst_before) << "a refused publication must skip the dispatch";

    renderer.exact_representation = true;
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}))
        << "the dispatch runs once its volume is published";
    EXPECT_EQ(publications, 2);
    EXPECT_FALSE(src_claimed);
    ASSERT_TRUE(detile_volume(result.data(), dst.data(), dst.size(), kW, kH, kD, kMode, kBpe));
    auto expected = dst_initial;
    std::copy_n(rendered.begin(), row, expected.begin());
    EXPECT_TRUE(std::equal(result.begin(), result.begin() + row, rendered.begin()))
        << "the row must come from the renderer's volume, not the stale guest bytes";
    EXPECT_FALSE(std::equal(result.begin(), result.begin() + row, stale.begin()));
    EXPECT_EQ(result, expected) << "every voxel outside the copied row is preserved";
}
