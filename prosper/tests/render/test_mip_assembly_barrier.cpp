// test_mip_assembly_barrier — #3248: two transfer WRITES to the same image need a memory
// dependency, and the mip-assembly path had only an execution one.
//
// `assembled_target_mips` clears the whole destination image and then copies each level the guest
// actually rendered into it. Every barrier in that loop names the copy SOURCE, so nothing ordered
// the clear's write against the copies' writes. Synchronization validation reports it as
// SYNC-HAZARD-WRITE-AFTER-WRITE, four times, in `texture_mip_render`.
//
// WHAT THIS TEST CAN AND CANNOT DETECT:
//
//   * It CANNOT detect the defect from pixels, and the reason is worse than usual. Assembly clears
//     to BLACK on purpose so a level the guest never rendered stays unavailable rather than being
//     invented from a neighbour. A clear that lands after a copy therefore produces a black level —
//     which is exactly what a CORRECT run produces for a missing level. The corruption is
//     indistinguishable from the intended output, so no content assertion can separate them.
//     Measured, not assumed: `texture_mip_render` passes identically with and without the barrier.
//   * It DOES fail without the fix. The assertion is STRUCTURAL: arm 1 pins the barrier's masks,
//     layouts and subresource scope exactly, and arm 2 requires one to be RECORDED on the assembly
//     path. Measured: deleting the call leaves arm 2 at zero and the case goes red.
//   * It CANNOT see ORDER, and that limit is worth stating precisely because the arm above reads as
//     if it could. Arm 2 reads a process-wide counter, so it catches the barrier being DELETED and
//     nothing else: a barrier MOVED out from between the clear and the copies -- after the loop, or
//     into the wrong branch -- still increments it and passes all three arms with the hazard fully
//     restored. Pinning the position would need a command-buffer recorder this fixture does not
//     have. The oracle for ordering is the layer: `vk_validation_scan.py --sync` reports the hazard
//     on the unfixed tree and nothing on the fixed one, measured both ways. That scan is opt-in and
//     is NOT what CI runs (#3255), so a relocation would reach master unless somebody runs it.
//   * Arm 3 is the discriminator, not decoration. A texture upload that is NOT an assembled mip
//     chain must record none, so a counter bumped anywhere on the upload path would satisfy arm 2
//     while proving nothing.

#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using prosper::test::BackendDraw;
using prosper::test::FrameResource;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::printf("  [FAIL] %s\n", message); ++failures; } \
    else               { std::printf("  [ok]   %s\n", message); } \
} while (0)

namespace {

uint64_t waw_count() {
    return prosper::test::backend_transfer_waw_barrier_count().load(std::memory_order_relaxed);
}

// One draw sampling `resource`, so both arms differ only in what the texture upload IS.
std::vector<BackendDraw> sampling_draw(const FrameResource& resource) {
    BackendDraw draw;
    draw.vs.assign(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv) / 4);
    draw.fs.assign(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv) / 4);
    draw.vcount = 3;
    draw.R = {resource};
    return {std::move(draw)};
}

}  // namespace

int main() {
    std::printf("== test_mip_assembly_barrier ==\n");

    // --- Arm 1: the barrier descriptor itself (no device) --------------------------------------
    {
        auto* const image = reinterpret_cast<VkImage>(static_cast<uintptr_t>(0xABCD));
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 3, 0, 1};
        const prosper::test::TransferWriteAfterWriteBarrier waw =
            prosper::test::transfer_write_after_write_barrier_for(image, range);
        CHECK(waw.barrier.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
              "the barrier is a VkImageMemoryBarrier");
        CHECK(waw.barrier.srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT &&
              waw.barrier.dstAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT,
              "both halves are TRANSFER_WRITE -- this orders a write against a write");
        CHECK(waw.src_stages == VK_PIPELINE_STAGE_TRANSFER_BIT &&
              waw.dst_stages == VK_PIPELINE_STAGE_TRANSFER_BIT,
              "both stage masks are the transfer stage");
        CHECK(waw.barrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
              waw.barrier.newLayout == waw.barrier.oldLayout,
              "no layout transition: a transition here would discard the clear it is ordering");
        CHECK(waw.barrier.image == image, "the barrier names the DESTINATION image");
        CHECK(waw.barrier.subresourceRange.baseMipLevel == 0 &&
              waw.barrier.subresourceRange.levelCount == 3,
              "the scope is every level the clear wrote, not just the one being copied next");
        CHECK(waw.barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
              waw.barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED,
              "no ownership transfer is requested");
    }

    const uint32_t W = 32, H = 32;

    // Two independent renderer-owned single-level targets, which is what an assembled chain is made
    // of: the guest rendered level 0 and level 1 into separate CB_COLOR surfaces and never rendered
    // level 2.
    constexpr uint64_t kMipTarget0 = 0x3248001000ull;
    constexpr uint64_t kMipTarget1 = 0x3248002000ull;
    const uint8_t red[4 * 4 * 4] = {
        255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
        255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
        255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
        255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
    };
    const uint8_t green[2 * 2 * 4] = {
        0,255,0,255, 0,255,0,255, 0,255,0,255, 0,255,0,255,
    };
    auto publish_target = [&](uint64_t id, const uint8_t* pixels, uint32_t w, uint32_t h) {
        FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = pixels; resource.tw = w; resource.th = h;
        prosper::test::BackendColorTarget target;
        target.persistent_id = id;
        target.load_existing = false;
        target.readback = false;
        prosper::test::render_draws_rgba(
            sampling_draw(resource), w, h, nullptr, nullptr, false, &target,
            nullptr, nullptr, nullptr, nullptr, true, nullptr, false);
    };
    publish_target(kMipTarget0, red, 4u, 4u);
    publish_target(kMipTarget1, green, 2u, 2u);
    CHECK(prosper::test::find_persistent_color_target(kMipTarget0, 4u, 4u,
                                                      VK_FORMAT_R8G8B8A8_UNORM) &&
          prosper::test::find_persistent_color_target(kMipTarget1, 2u, 2u,
                                                      VK_FORMAT_R8G8B8A8_UNORM),
          "two renderer-owned mip levels are available to assemble from");

    // --- Arm 2: assembling a chain records the barrier ------------------------------------------
    FrameResource assembled;
    assembled.binding = 4; assembled.set = 1;
    assembled.tw = assembled.th = 4u;
    assembled.declared_mip_levels = 3u;
    assembled.persistent_render_target_id = kMipTarget0;
    assembled.persistent_render_target_mip_ids[0] = kMipTarget0;
    assembled.persistent_render_target_mip_ids[1] = kMipTarget1;
    assembled.persistent_render_target_mip_count = 3u;
    const uint64_t before_assembly = waw_count();
    const std::vector<uint8_t> assembled_px =
        prosper::test::render_draws_rgba(sampling_draw(assembled), W, H);
    const uint64_t after_assembly = waw_count();
    CHECK(assembled_px.size() == static_cast<size_t>(W) * H * 4,
          "the assembled-chain pass rendered a full frame");
    std::printf("  transfer write-after-write barriers recorded by the assembly: %llu\n",
                static_cast<unsigned long long>(after_assembly - before_assembly));
    CHECK(after_assembly > before_assembly,
          "assembling a mip chain orders the clear against the per-level copies");

    // --- Arm 3 (negative control): an ordinary upload records none -------------------------------
    // Same shape of call, same extent, one difference: the texture comes from host pixels instead of
    // renderer-owned targets, so there is no clear-then-copy pair to order. If the counter moved
    // here it would be counting uploads rather than write-after-write hazards.
    FrameResource plain;
    plain.binding = 4; plain.set = 1;
    plain.tex_rgba = red; plain.tw = 4; plain.th = 4;
    plain.declared_mip_levels = 3u;
    const uint64_t before_plain = waw_count();
    const std::vector<uint8_t> plain_px =
        prosper::test::render_draws_rgba(sampling_draw(plain), W, H);
    const uint64_t after_plain = waw_count();
    CHECK(plain_px.size() == static_cast<size_t>(W) * H * 4,
          "the ordinary-upload pass rendered a full frame");
    CHECK(after_plain == before_plain,
          "an upload with no clear-then-copy pair records no write-after-write barrier");

    std::printf("== %s ==\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
