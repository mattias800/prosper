// Live compute must sample a guest-backed one- or two-channel FP16 texture with its values intact.
// The historical RGBA8 conversion clamps to [0, 1]: a negative circle of confusion became 0, and
// UE4's depth of field then found no blur anywhere (Kena, KENA_STATUS.md).
//
// Mutation that turns both arms red: make sample_float16_natively() return false for guest-backed
// one/two-channel FP16 (shared/compute/sampled_float16_view.hpp) -- the dispatch then reads the
// clamped values (0 for negatives, 1 above one).
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_compute.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t Width = 64;

struct GuestMapping {
    uint64_t base = 0;
    size_t bytes = 0;
    ~GuestMapping() {
        if (!base) return;
        if (auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap")))
            unmap(base, bytes, 0, 0, 0, 0);
    }
};

uint16_t to_half(float value) {   // exact for the small multiples of 0.25 used below
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    if ((bits & 0x7fffffffu) == 0) return static_cast<uint16_t>(sign);
    const int exponent = int((bits >> 23) & 0xff) - 127 + 15;
    return static_cast<uint16_t>(sign | (uint32_t(exponent) << 10) | ((bits >> 13) & 0x3ff));
}

// image_load at (gid, 0) from the sampled FP16 T# in s[0:7], image_store of all four channels to
// an RGBA32F storage image in s[8:15].
ComputeItem load_item(uint64_t source, uint32_t components, float* output, const uint32_t* indices,
                      const uint32_t* dummy) {
    ShaderResourceTable resources;
    for (uint32_t binding = 0; binding < 4; ++binding) {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = binding;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy : indices);
        buffer.size = binding ? 16 : Width * sizeof(uint32_t);
        resources.resources.push_back(buffer);
    }
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.binding = 4;
    texture.sgpr_base = 0;
    texture.img_dim = 1;
    texture.width = Width;
    texture.height = texture.depth = 1;
    texture.format = DataFormat::Float16;
    texture.num_components = components;
    texture.gpu_addr = source;
    texture.size = Width * components * 2;
    // GCN's view of a narrow format: X, Y (or 0), 0, 1.
    texture.swizzle[0] = 4;
    texture.swizzle[1] = components == 2 ? 5 : 0;
    texture.swizzle[2] = 0;
    texture.swizzle[3] = 1;
    resources.resources.push_back(texture);
    ShaderResource out{};
    out.cls = ResourceClass::StorageImage;
    out.binding = 5;
    out.sgpr_base = 8;
    out.img_dim = 1;
    out.width = Width;
    out.height = out.depth = 1;
    out.format = DataFormat::Float32;
    out.num_components = 4;
    out.gpu_addr = reinterpret_cast<uint64_t>(output);
    out.size = Width * 16;
    for (uint32_t c = 0; c < 4; ++c) out.swizzle[c] = 4 + c;
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
    item.code_addr = 0xf16f16;
    return item;
}

} // namespace

TEST(LiveComputeNarrowFloat16, GuestBackedFloat16KeepsSignedAndHdrValues) {
    prosper::register_builtin_hle();
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    ASSERT_TRUE(map);
    GuestMapping guest;
    guest.bytes = 0x100000;
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&guest.base), guest.bytes, 2, 0,
                  reinterpret_cast<uint64_t>("live-compute-narrow-float16"), 0),
              0);
    ASSERT_NE(guest.base, 0u);
    auto* indices = reinterpret_cast<uint32_t*>(guest.base + 0x80000);
    auto* dummy = reinterpret_cast<uint32_t*>(guest.base + 0x90000);
    auto* output = reinterpret_cast<float*>(guest.base + 0xa0000);
    for (uint32_t i = 0; i < Width; ++i) indices[i] = i;
    std::memset(dummy, 0, 16);
    // A foreground CoC is negative; a background one can exceed 1.
    auto red = [](uint32_t x) { return -0.5f * float(x + 1); };
    auto green = [](uint32_t x) { return 1.25f * float(x + 1); };

    for (const uint32_t components : {2u, 1u}) {
        const uint64_t texture = guest.base + uint64_t(components) * 0x10000;
        auto* texels = reinterpret_cast<uint16_t*>(texture);
        for (uint32_t x = 0; x < Width; ++x) {
            texels[size_t(x) * components] = to_half(red(x));
            if (components == 2) texels[size_t(x) * 2 + 1] = to_half(green(x));
        }
        for (uint32_t i = 0; i < Width * 4; ++i) output[i] = -99.0f;
        ComputeItem item = load_item(texture, components, output, indices, dummy);
        ASSERT_FALSE(item.spirv.empty()) << components;
        ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item})) << components;
        bool all = true;
        for (uint32_t x = 0; x < Width; ++x) {
            const float* t = output + size_t(x) * 4;
            all = all && t[0] == red(x) && t[1] == (components == 2 ? green(x) : 0.0f) &&
                  t[2] == 0.0f && t[3] == 1.0f;
        }
        EXPECT_TRUE(all) << components << "-channel FP16: texel 0 = (" << output[0] << ", "
                         << output[1] << ", " << output[2] << ", " << output[3] << "), expected ("
                         << red(0) << ", " << (components == 2 ? green(0) : 0.0f) << ", 0, 1)";
    }
}
