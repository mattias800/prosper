// #4625: a compute dispatch that binds a renderer-claimed volume publishes that volume to guest
// memory first and then runs on the published bytes, instead of being skipped. The CPU cases pin the
// publication plan and its tiling; the Vulkan case runs a real 3D storage dispatch through the live
// compute backend (Kena's translucency-lighting clear binds four claimed 64^3 volumes this way).
#include "gpu/execute/renderer_volume_publication.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "shared/rtt/volume_publication_source.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#if defined(__linux__)
#include <csignal>
#include <sys/ucontext.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
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

// The renderer's own claim, alias and release rules (live_renderer.cpp drives these with RttSurf).
struct FakeSurface {
    uint32_t w = 0, h = 0, volume_depth = 0;
    uint64_t volume_guest_bytes = 0;
    bool volume_footprint_proven = false;
    VolumeGuestLayout volume_layout;
    int format = 0, guest_format = 0;
    bool gpu_valid = false;
};

TEST(RendererVolumePublication, RendererClaimBecomesThePublicationSource) {
    FakeSurface claim{kW, kH, kD, 4096, true, {kW, kH, kD, kMode, kBpe}, 97, 97, true};
    auto source = prosper::frontend::volume_publication_source(0x5013f30000ull, claim);
    EXPECT_EQ(source.base, 0x5013f30000ull);
    EXPECT_EQ(source.claimed_bytes, 4096u);
    EXPECT_TRUE(source.footprint_proven && source.renderer_image_valid &&
                source.exact_representation);
    EXPECT_EQ(source.layout, claim.volume_layout);
    EXPECT_EQ(source.image_depth, kD);

    auto two_d_over_claim = claim;   // a later 2D pass at the base keeps the claim, not the volume
    two_d_over_claim.volume_depth = 0;
    EXPECT_FALSE(
        prosper::frontend::volume_publication_source(1, two_d_over_claim).renderer_image_valid);
    auto tombstone = claim;
    tombstone.gpu_valid = false;
    EXPECT_FALSE(prosper::frontend::volume_publication_source(1, tombstone).renderer_image_valid);
    auto canonicalised = claim;   // e.g. BGRA guest bytes in an RGBA renderer image
    canonicalised.guest_format = 44;
    EXPECT_FALSE(
        prosper::frontend::volume_publication_source(1, canonicalised).exact_representation);

    prosper::frontend::release_volume_claim(claim);
    EXPECT_EQ(claim.volume_guest_bytes, 0u);
    EXPECT_FALSE(claim.volume_footprint_proven);
    EXPECT_EQ(claim.volume_layout, VolumeGuestLayout{});
}

TEST(RendererVolumePublication, AliasRuleIsTheGuestWriteDrains) {
    constexpr uint64_t base = 0x5013f30000ull, claim = 4u << 20;
    std::map<uint64_t, FakeSurface> cache;
    cache[base] = FakeSurface{64, 64, 64, claim, true};
    // Kena's case: an unrelated 3200x1800 RGBA8 target, disjoint by address, whose physical
    // topology is unproven ("may overlap" for every pair). It must not block the publication.
    cache[0x5002dc0000ull] = FakeSurface{3200, 1800};
    const auto bytes = [](const FakeSurface& s) {
        return std::max<uint64_t>(uint64_t{s.w} * s.h * 4u, s.volume_guest_bytes);
    };
    int topology_asks = 0;
    const auto unproven_topology = [&](uint64_t, uint64_t, uint64_t, uint64_t) {
        ++topology_asks;
        return true;
    };
    auto alias =
        prosper::frontend::volume_publication_alias(cache, base, claim, bytes, unproven_topology);
    EXPECT_EQ(alias.first, 0u) << "a 2D surface is an alias by address only";
    EXPECT_EQ(topology_asks, 0);

    cache[base + 0x100000] = FakeSurface{256, 256};   // a 2D surface inside the footprint
    alias =
        prosper::frontend::volume_publication_alias(cache, base, claim, bytes, unproven_topology);
    EXPECT_EQ(alias.first, base + 0x100000);
    EXPECT_EQ(alias.second, 256u * 256u * 4u);
    cache.erase(base + 0x100000);

    cache[0x7000000000ull] = FakeSurface{64, 64, 64, claim, true};   // another claim, far by VA
    alias =
        prosper::frontend::volume_publication_alias(cache, base, claim, bytes, unproven_topology);
    EXPECT_EQ(alias.first, 0x7000000000ull) << "another claim is an alias by physical topology too";
    const auto disjoint = [](uint64_t, uint64_t, uint64_t, uint64_t) { return false; };
    EXPECT_EQ(
        prosper::frontend::volume_publication_alias(cache, base, claim, bytes, disjoint).first, 0u);
}

#if defined(__linux__)
namespace {
volatile sig_atomic_t g_watch_faults = 0;
void watch_fault(int signal, siginfo_t* info, void* context) {
    const bool write =
        context && (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2);
    if (signal == SIGSEGV && write && info && info->si_addr &&
        prosper::host::guest_write_watch_handle_fault(reinterpret_cast<uint64_t>(info->si_addr))) {
        g_watch_faults = g_watch_faults + 1;
        return;
    }
    _exit(86);
}
}   // namespace

// A publication is a host store into guest memory: it must open write-watched pages before it
// writes (no page fault per page), and the watch must still read Dirty afterwards.
TEST(RendererVolumePublication, PublicationAnnouncesItsHostWriteToTheWriteWatch) {
    static uint8_t alt_stack[256 * 1024];
    stack_t stack{}, old_stack{};
    stack.ss_sp = alt_stack;
    stack.ss_size = sizeof(alt_stack);
    struct sigaction action{}, old_action{};
    action.sa_sigaction = watch_fault;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    ASSERT_EQ(sigaltstack(&stack, &old_stack), 0);
    ASSERT_EQ(sigaction(SIGSEGV, &action, &old_action), 0);
    prosper::host::guest_write_watch_set_fault_onstack(true);
    struct Restore {
        stack_t stack;
        struct sigaction action;
        ~Restore() {
            prosper::host::guest_write_watch_set_fault_onstack(false);
            sigaction(SIGSEGV, &action, nullptr);
            sigaltstack(&stack, nullptr);
        }
    } restore{old_stack, old_action};

    prosper::register_builtin_hle();
    const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
    const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    const auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
    ASSERT_TRUE(allocate && map && unmap && release);
    const uint64_t mapping_bytes = (tiled_bytes() + 0xffffu) & ~uint64_t{0xffff};
    uint64_t physical = 0, guest_address = 0;
    ASSERT_EQ(allocate(0, 0x200000000ull, mapping_bytes, 0x10000, 0,
                       reinterpret_cast<uint64_t>(&physical)),
              0);
    struct Backing {
        prosper::HleFn unmap, release;
        uint64_t physical, guest, bytes;
        ~Backing() {
            if (guest) unmap(guest, bytes, 0, 0, 0, 0);
            release(physical, bytes, 0, 0, 0, 0);
        }
    } backing{unmap, release, physical, 0, mapping_bytes};
    ASSERT_EQ(
        map(reinterpret_cast<uint64_t>(&guest_address), mapping_bytes, 2, 0, physical, 0x10000), 0);
    backing.guest = guest_address;

    auto watch = prosper::host::GuestWriteWatch::create(guest_address, tiled_bytes());
    if (!watch) GTEST_SKIP() << "page-protection write watches are unavailable here";
    ASSERT_EQ(watch.query(), prosper::host::GuestWriteWatchQuery::Unchanged);
    const auto rendered = pattern(linear_bytes(), 313u, 7u);
    VolumePublicationSource source;
    source.base = guest_address;
    source.claimed_bytes = tiled_bytes();
    source.footprint_proven = source.renderer_image_valid = source.exact_representation = true;
    source.image_width = kW;
    source.image_height = kH;
    source.image_depth = kD;
    source.layout = {kW, kH, kD, kMode, kBpe};
    const auto faults_before = g_watch_faults;
    ASSERT_EQ(publish_volume_to_guest(
                  source,
                  [&](std::vector<uint8_t>& linear) {
                      linear = rendered;
                      return true;
                  },
                  reinterpret_cast<uint8_t*>(guest_address)),
              VolumePublication::Published);
    EXPECT_EQ(g_watch_faults, faults_before) << "the host write must open armed pages first";
    EXPECT_EQ(watch.query(), prosper::host::GuestWriteWatchQuery::Dirty);
    std::vector<uint8_t> expected(tiled_bytes());
    ASSERT_TRUE(
        tile_volume(expected.data(), expected.size(), rendered.data(), kW, kH, kD, kMode, kBpe));
    EXPECT_EQ(
        std::memcmp(reinterpret_cast<const void*>(guest_address), expected.data(), expected.size()),
        0);
}
#endif

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

    // Positive control first: unclaimed, the dispatch reads the guest bytes it is given. This arm
    // does not depend on the publication's write notification (mutation M3 leaves it green: the
    // backend re-uploads this small source); PublishTiles... pins the notification instead.
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
