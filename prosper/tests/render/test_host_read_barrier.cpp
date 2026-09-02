// test_host_read_barrier — #2944: a GPU readback the CPU maps and reads needs an AVAILABILITY
// operation into the host domain, and waiting a fence is not one.
//
// Waiting a fence (or vkQueueWaitIdle) orders EXECUTION. Making a transfer write available to the
// host additionally requires a dependency with dstStageMask = VK_PIPELINE_STAGE_HOST_BIT and
// dstAccessMask = VK_ACCESS_HOST_READ_BIT — equivalently, for non-coherent memory,
// vkInvalidateMappedMemoryRanges, which this header already does at every mapped read. The two are
// independent halves; the backend had only the second.
//
// WHAT THIS TEST CAN AND CANNOT DETECT, stated plainly because the honest answer is not "it proves
// the fix works":
//
//   * It CANNOT detect the defect by its output. Every readback buffer in the backend is backed by
//     HOST_VISIBLE memory that this platform's driver keeps coherent enough that the unfixed code
//     returns correct pixels. A test that reads back a triangle and checks the colour passes
//     identically with and without the barrier — that is precisely why the gap survived. So the
//     assertion here is STRUCTURAL: the barrier must be RECORDED.
//   * It DOES fail without the fix, on the path it drives. Deleting the
//     `record_host_read_barrier(cmd, rb)` call from the colour readback in render_draw_pass_rgba
//     leaves the counter at zero across arm 2 and the arm goes red.
//   * The negative control is the discriminator, not decoration. Arm 3 renders the SAME draw with
//     `want_color_readback = false`. No host read happens, so no barrier may be recorded. Without
//     it, a counter incremented unconditionally somewhere on the render path would satisfy arm 2
//     while proving nothing about the readback.
//   * It covers ONE of the seven recorded sites — the composited-frame readback. The DS-plane
//     transfer, the persistent colour-target readback, the storage-image writeback and the three
//     env-gated diagnostics are not reachable from a fixture-only test without a title's resources;
//     they share the same helper, and arm 1 pins that helper's masks.
//
// Arm 1 needs no device at all: `host_read_barrier_for` is pure, so the exact masks and stages are
// asserted directly rather than inferred from a driver's behaviour.

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

// The same draw render_triangle_rgba() builds, spelled out here because the negative-control arm
// needs the `want_color_readback` parameter that the convenience wrapper does not expose.
std::vector<BackendDraw> one_triangle() {
    BackendDraw draw;
    draw.vs.assign(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv) / 4);
    draw.fs.assign(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv) / 4);
    draw.vcount = 3;
    for (uint32_t set = 0; set < 2; ++set) {
        FrameResource b2; b2.binding = 2; b2.set = set; draw.R.push_back(std::move(b2));
        FrameResource b3; b3.binding = 3; b3.set = set; draw.R.push_back(std::move(b3));
    }
    return {std::move(draw)};
}

uint64_t barrier_count() {
    return prosper::test::backend_host_read_barrier_count().load(std::memory_order_relaxed);
}

}  // namespace

int main() {
    std::printf("== test_host_read_barrier ==\n");

    // --- Arm 1: the barrier descriptor itself (no device) --------------------------------------
    {
        auto* const buffer = reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x1234));
        const prosper::test::HostReadBarrier host_read =
            prosper::test::host_read_barrier_for(buffer);
        CHECK(host_read.barrier.sType == VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
              "the barrier is a VkBufferMemoryBarrier");
        CHECK(host_read.barrier.dstAccessMask == VK_ACCESS_HOST_READ_BIT,
              "dstAccessMask is exactly VK_ACCESS_HOST_READ_BIT");
        CHECK(host_read.dst_stages == VK_PIPELINE_STAGE_HOST_BIT,
              "dstStageMask is exactly VK_PIPELINE_STAGE_HOST_BIT");
        CHECK(host_read.barrier.srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT,
              "the default source is the transfer write the copy performed");
        CHECK(host_read.src_stages == VK_PIPELINE_STAGE_TRANSFER_BIT,
              "the default source stage is the transfer stage");
        CHECK(host_read.barrier.buffer == buffer, "the barrier names the buffer it was asked for");
        CHECK(host_read.barrier.offset == 0 && host_read.barrier.size == VK_WHOLE_SIZE,
              "the scope is the whole buffer, so every slot copied into it is covered");
        CHECK(host_read.barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
              host_read.barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED,
              "no ownership transfer is requested");
    }

    // The geometry probe reads transform-feedback output, which is not a transfer write. The source
    // scope is a parameter for exactly that reason; the host half must not change with it.
    {
        auto* const buffer = reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x5678));
        const prosper::test::HostReadBarrier host_read = prosper::test::host_read_barrier_for(
            buffer, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
            VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT);
        CHECK(host_read.barrier.srcAccessMask == VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT &&
              host_read.src_stages == VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
              "a non-transfer producer keeps its own source scope");
        CHECK(host_read.barrier.dstAccessMask == VK_ACCESS_HOST_READ_BIT &&
              host_read.dst_stages == VK_PIPELINE_STAGE_HOST_BIT,
              "the host half is the same whatever wrote the buffer");
    }

    const uint32_t W = 64, H = 64;

    // --- Arm 2: a real readback records one ----------------------------------------------------
    const uint64_t before_readback = barrier_count();
    const std::vector<uint8_t> pixels = prosper::test::render_draws_rgba(one_triangle(), W, H);
    const uint64_t after_readback = barrier_count();
    CHECK(pixels.size() == static_cast<size_t>(W) * H * 4,
          "the pass rendered and read back a full frame");
    std::printf("  host-read barriers recorded by the readback pass: %llu\n",
                static_cast<unsigned long long>(after_readback - before_readback));
    CHECK(after_readback > before_readback,
          "a colour readback records a host-read barrier before the host maps the buffer");

    // --- Arm 3 (negative control): no readback, no barrier --------------------------------------
    // Same draws, same extent, one flag. If the counter moved here it would be counting renders
    // rather than readbacks, and arm 2 would be worthless.
    const uint64_t before_no_readback = barrier_count();
    const std::vector<uint8_t> none = prosper::test::render_draws_rgba(
        one_triangle(), W, H, /*seed_rgba=*/nullptr, /*clear_rgba=*/nullptr,
        /*persist_depth_stencil=*/false, /*color_target=*/nullptr, /*seed_rgba1=*/nullptr,
        /*clear_rgba1=*/nullptr, /*out_rgba1=*/nullptr, /*submission_batch=*/nullptr,
        /*flush_submission_batch=*/true, /*mrt_outputs=*/nullptr, /*want_color_readback=*/false);
    const uint64_t after_no_readback = barrier_count();
    CHECK(none.empty(), "a pass that asks for no colour readback returns no pixels");
    CHECK(after_no_readback == before_no_readback,
          "a pass with no host read records no host-read barrier");

    std::printf("== %s ==\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
