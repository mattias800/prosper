// Exercise the production observer -> pending queue -> registered renderer query/submit callback.
// Replay seeds populate the real private CPU RTT cache. Backend records below contain metadata
// only: no fake Vulkan handles, draws or completion events are used as rendering evidence.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "shared/live/live_renderer.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
using prosper::frontend::guest_write_drain_work_for_thread;
int failures = 0;
void check(bool ok, const char* description) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", description);
    failures += !ok;
}
constexpr uint64_t A = 0x10000, B = 0x20000, Miss = 0x1000000;

void seed(uint64_t address, uint32_t width = 4, uint32_t height = 4) {
    GpuCaptureRttSeed value;
    value.guest_addr = address;
    value.width = width; value.height = height;
    value.format = GpuCaptureColorFormat::Rgba8Unorm;
    value.rgba.assign(size_t(width) * height * 4, 0x40);
    std::string error;
    check(restore_gpu_replay_rtt_seeds({value}, error), "publish a real renderer RTT seed");
}

void drain() { (void)is_live_render_target(Miss); }
void clear() {
    // Use the production overflow contract, not a test-only cache reset.
    for (size_t i = 0; i != 65537; ++i) notify_guest_gpu_write(Miss, 4);
    drain();
    prosper::test::persistent_color_target_cache().clear();
    prosper::test::persistent_ds_cache().clear();
}

void associate_metadata(uint64_t address, uint64_t metadata) {
    DrawItem item;
    item.vs.assign(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    item.fs.assign(std::begin(kTriFragSpv), std::end(kTriFragSpv));
    item.ps.topology = 3; // Triangle-list fixture; the default is point-list and needs PointSize.
    item.prt = std::make_shared<ShaderResourceTable>();
    ShaderResource resource;
    resource.cls = ResourceClass::Texture;
    resource.format = DataFormat::Unorm8;
    resource.num_components = 4;
    resource.width = resource.height = 4;
    resource.depth = 1;
    resource.img_dim = 1;
    resource.tile_mode = 27;
    resource.gpu_addr = address;
    resource.compression_enabled = true;
    resource.metadata_addr = metadata;
    check(gpu_capture_dcc_metadata_footprint(resource) != 0,
          "descriptor has a supported DCC metadata footprint");
    item.prt->resources.push_back(resource);
    // The valid fixture shaders do not read this descriptor. The real submit callback registers
    // metadata before draining. This is an association/invalidation state guard, not a pixel test.
    (void)render_submit_items({item}, 1, 1);
}
} // namespace

int main(int argc, char** argv) {
    const bool ordinary = argc == 2 && std::strcmp(argv[1], "--ordinary") == 0;
    prosper::frontend::register_live_renderer("", false);
    check(guest_write_drain_work_for_thread().observed, "census was armed at process startup");
    clear();
    seed(A); seed(B);
    const auto before = guest_write_drain_work_for_thread();
    for (unsigned i = 0; i != 4; ++i) notify_guest_gpu_write(Miss + i * 64, 4);
    drain();
    const auto after = guest_write_drain_work_for_thread();
    check(after.rtt_footprints - before.rtt_footprints == (ordinary ? 8u : 2u),
          "four writes calculate each footprint once in a batch, once per write in the control");
    check(after.prepared_drains - before.prepared_drains == (ordinary ? 0u : 1u),
          "the actual registered drain selects the expected path");
    check(is_live_render_target(A) && is_live_render_target(B), "missed writes preserve both owners");
    const auto empty_before = guest_write_drain_work_for_thread();
    drain();
    check(guest_write_drain_work_for_thread().rtt_footprints == empty_before.rtt_footprints,
          "an empty drain does not prepare or calculate footprints");
    notify_guest_gpu_write(A + 64, 4); // Exactly at the end of A.
    drain();
    check(is_live_render_target(A), "adjacent writes do not overlap");
    notify_guest_gpu_write(A + 63, 1);
    notify_guest_gpu_write(A, 64); // Must not dereference the entry erased by the first write.
    notify_guest_gpu_write(Miss, 4);
    drain();
    check(!is_live_render_target(A) && is_live_render_target(B),
          "partial and repeated color writes erase only the overlapping owner");

    // Rebuild an existing owner with a larger extent between drains.
    seed(A); seed(A, 8, 4);
    notify_guest_gpu_write(A + 96, 4); notify_guest_gpu_write(Miss, 4); drain();
    check(!is_live_render_target(A), "a new drain observes a replaced owner's larger footprint");
    seed(A); seed(A + 32);
    notify_guest_gpu_write(A + 40, 1); notify_guest_gpu_write(Miss, 4); drain();
    check(!is_live_render_target(A) && !is_live_render_target(A + 32),
          "one partial write invalidates every overlapping alias");
    seed(UINT64_MAX - 31);
    notify_guest_gpu_write(UINT64_MAX - 8, 64); notify_guest_gpu_write(Miss, 4); drain();
    check(!is_live_render_target(UINT64_MAX - 31), "overflowing ends retain saturating overlap semantics");
    notify_guest_gpu_write(0, 64); notify_guest_gpu_write(B, 0); drain();
    check(is_live_render_target(B), "zero-address and zero-size notifications remain ineffective");

    auto& colors = prosper::test::persistent_color_target_cache();
    const prosper::test::PersistentColorTargetKey color{B, 4, 4, VK_FORMAT_R8G8B8A8_UNORM};
    colors[color].valid = true;
    prosper::test::PersistentDsKey key;
    key.dr = 0x30000; key.sr = 0x40000; key.htile = 0x50000;
    key.w = key.h = 4;
    auto& ds = prosper::test::persistent_ds_cache();
    ds[key].depth_valid = ds[key].stencil_valid = true;
    notify_guest_gpu_write(key.dr, 4); notify_guest_gpu_write(B, 4); drain();
    check(!ds.at(key).depth_valid && ds.at(key).stencil_valid && !colors.at(color).valid,
          "CPU RTT preparation preserves independent backend color and depth invalidation");
    ds[key].depth_valid = true;
    notify_guest_gpu_write(key.sr, 1); notify_guest_gpu_write(Miss, 4); drain();
    check(ds.at(key).depth_valid && !ds.at(key).stencil_valid, "stencil writes preserve depth");
    ds[key].stencil_valid = true;
    seed(A); // Keep the prepared CPU RTT path active during both HTILE origin checks.
    ds[key].last_depth_write = 1;
    ds[key].last_depth_present = present_count();
    notify_guest_gpu_write_preserving_bytes(key.htile, 4);
    notify_guest_gpu_write(Miss, 4); drain();
    check(ds.at(key).depth_valid && ds.at(key).stencil_valid,
          "queued preserving origin retains the existing current-frame HTILE policy");
    notify_guest_gpu_write(key.htile, 4); notify_guest_gpu_write(Miss, 4); drain();
    check(!ds.at(key).depth_valid && !ds.at(key).stencil_valid,
          "an ordinary HTILE write invalidates both aspects after the preserving write");
    colors.clear(); ds.clear();

    clear(); seed(A); seed(B);
    constexpr uint64_t Metadata = 0x60000, NewMetadata = 0x80000;
    // Queue before the submit so the fresh descriptor association must precede preparation.
    notify_guest_gpu_write(Metadata, 4); notify_guest_gpu_write(Metadata + 4, 4);
    const auto dcc_before = guest_write_drain_work_for_thread();
    associate_metadata(A, Metadata);
    const auto dcc_after = guest_write_drain_work_for_thread();
    check(!is_live_render_target(A) && is_live_render_target(B) &&
              dcc_after.rtt_dcc - dcc_before.rtt_dcc == 2 &&
              dcc_after.rtt_erases == dcc_before.rtt_erases,
          "repeated DCC writes revoke pixels while retaining identity for the second write");
    seed(A);
    notify_guest_gpu_write(NewMetadata, 4); notify_guest_gpu_write(Miss, 4);
    associate_metadata(A, NewMetadata);
    check(!is_live_render_target(A), "metadata registered for a later drain replaces the old footprint");
    seed(A);
    notify_guest_gpu_write(Metadata, 4); notify_guest_gpu_write(Miss, 4); drain();
    check(is_live_render_target(A), "the replaced metadata range no longer invalidates the owner");
    const auto order_before = guest_write_drain_work_for_thread();
    notify_guest_gpu_write(NewMetadata, 4); notify_guest_gpu_write(A, 4);
    notify_guest_gpu_write(NewMetadata, 4); drain();
    const auto order_after = guest_write_drain_work_for_thread();
    check(!is_live_render_target(A) && order_after.rtt_dcc - order_before.rtt_dcc == 1 &&
              order_after.rtt_erases - order_before.rtt_erases == 1,
          "DCC then color then DCC retires the erased snapshot record immediately");
    seed(A); associate_metadata(A, A);
    const auto precedence_before = guest_write_drain_work_for_thread();
    notify_guest_gpu_write(A, 4); notify_guest_gpu_write(Miss, 4); drain();
    check(!is_live_render_target(A) &&
              guest_write_drain_work_for_thread().rtt_dcc == precedence_before.rtt_dcc,
          "color overlap takes precedence when the same bytes also name metadata");
    seed(A);
    clear();
    check(!is_live_render_target(A) && !is_live_render_target(B),
          "pending queue overflow still clears all CPU RTT authority");
    return failures ? 1 : 0;
}
