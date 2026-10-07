// The ranges the live pipelining hook derives from a dispatch's bindings (ADR 0009 Stage 1, step 2).
// Pure data in, pure data out: it needs the Vulkan headers for the bound-resource types but creates
// no Vulkan object and runs no dispatch.
#include "shared/live/pipeline_observe_hook.hpp"

#include <gtest/gtest.h>

namespace {

// The ranges the live hook derives from a dispatch's bindings (pure; no Vulkan objects needed).
TEST(PipelineObserveHook, BindingRangesAreConservativeAboutReadsAndExactAboutWrites) {
    using namespace prosper::frontend;
    prosper::gpu::ShaderResource constants{}, output{}, replay{}, sampled{}, storage_out{};
    constants.gpu_addr = 0x1000; output.gpu_addr = 0x2000; sampled.gpu_addr = 0x3000;
    storage_out.gpu_addr = 0x4000; replay.gpu_addr = 0x5000;
    uint8_t blob[4] = {};
    replay.host_data = blob;   // a replayed resource has no guest backing to track
    std::vector<BoundBuffer> buffers(3);
    buffers[0].resource = &constants; buffers[0].guest_bytes = 64;
    buffers[1].resource = &output; buffers[1].guest_bytes = 128; buffers[1].writable = true;
    buffers[2].resource = &replay; buffers[2].guest_bytes = 32;
    std::vector<BoundImage> images(2);
    images[0].resource = &sampled; images[0].guest_bytes = 256;
    images[1].resource = &storage_out; images[1].guest_bytes = 512;
    images[1].storage = true; images[1].storage_write_only = true; images[1].storage_writeback = true;

    const auto ranges = dispatch_binding_ranges(buffers, images);
    ASSERT_EQ(ranges.size(), 4u);   // the replayed buffer is skipped
    EXPECT_TRUE(ranges[0].read && !ranges[0].write);      // read-only buffer
    EXPECT_TRUE(ranges[1].read && ranges[1].write);       // a writable buffer may read old contents
    EXPECT_TRUE(ranges[2].read && !ranges[2].write);      // sampled image
    EXPECT_TRUE(!ranges[3].read && ranges[3].write);      // write-only storage image is written, not read
    EXPECT_EQ(ranges[3].addr, 0x4000u);
    EXPECT_EQ(ranges[3].bytes, 512u);
}


}  // namespace
