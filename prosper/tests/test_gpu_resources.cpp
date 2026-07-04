// test_gpu_resources — the resource-layer contract: create/track resources, look them up by handle,
// enforce handle validity. This is the front-half↔back-half integration point (gpu_resources.hpp);
// the test fixes its behavior so both halves can rely on it as we build the Vulkan backing.
#include "../src/gpu/gpu_resources.hpp"
#include <cstdio>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_gpu_resources ==\n");
    resource_reset();
    CHECK(resource_count() == 0, "registry starts empty");
    CHECK(resource_get(0) == nullptr && !resource_valid(0), "handle 0 is always invalid");

    // A render target, as the front-half would fill it from an AGC color-target descriptor.
    ResourceDesc rt{};
    rt.kind = ResourceKind::RenderTarget;
    rt.gpu_addr = 0xC0000000ull; rt.width = 1920; rt.height = 1080;
    rt.format = 0x0A; rt.usage = Usage_ColorAttachment;
    ResourceHandle h_rt = resource_create(rt);
    CHECK(h_rt != 0, "render-target create returns a non-zero handle");

    // A vertex buffer.
    ResourceDesc vb{};
    vb.kind = ResourceKind::Buffer; vb.gpu_addr = 0xD0000000ull; vb.size = 4096;
    vb.usage = Usage_Vertex;
    ResourceHandle h_vb = resource_create(vb);
    CHECK(h_vb != 0 && h_vb != h_rt, "buffer create returns a distinct non-zero handle");
    CHECK(resource_count() == 2, "two resources tracked");

    const ResourceDesc* g_rt = resource_get(h_rt);
    CHECK(g_rt && g_rt->kind == ResourceKind::RenderTarget && g_rt->width == 1920 &&
          g_rt->height == 1080 && g_rt->gpu_addr == 0xC0000000ull &&
          (g_rt->usage & Usage_ColorAttachment),
          "render-target descriptor round-trips through the registry");
    const ResourceDesc* g_vb = resource_get(h_vb);
    CHECK(g_vb && g_vb->kind == ResourceKind::Buffer && g_vb->size == 4096 &&
          (g_vb->usage & Usage_Vertex),
          "buffer descriptor round-trips through the registry");

    CHECK(resource_get(h_vb + 100) == nullptr && !resource_valid(h_vb + 100),
          "an out-of-range handle is invalid (no silent aliasing)");

    const ResourceDesc* pinned_rt = resource_get(h_rt);
    for (int i = 0; i < 4096; i++) {
        ResourceDesc tmp{};
        tmp.kind = ResourceKind::Buffer;
        tmp.gpu_addr = 0xE0000000ull + (uint64_t)i * 0x1000u;
        tmp.size = 256;
        tmp.usage = Usage_Storage;
        resource_create(tmp);
    }
    CHECK(pinned_rt == resource_get(h_rt) && pinned_rt && pinned_rt->width == 1920,
          "resource_get pointers remain stable as the registry grows");

    resource_reset();
    CHECK(resource_count() == 0 && !resource_valid(h_rt), "reset clears the registry");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
