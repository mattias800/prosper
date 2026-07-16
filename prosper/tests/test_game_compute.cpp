#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/tile.hpp"
#include "live_compute.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } } while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    // Dead Cells' bound startup fill kernel, copied verbatim from eboot.elf at runtime address
    // 0x401aec200. It stores s4-s7 to record `(TGID_X << 6) + local_id_x` through the V# in s0-s3.
    static const uint32_t code[] = {
        0xd7460004, 0x04010c08, 0x7e000204, 0x7e020205, 0x7e040206,
        0x7e060207, 0xe01c2000, 0x80000004, 0xbf810000,
    };

    constexpr uint32_t records = 130;
    std::vector<uint32_t> result(records * 4, 0xcccccccc);
    ShaderResourceTable rt;
    ShaderResource buffer;
    // Runtime metadata classifies compute direct type-1 V#s as ConstantBuffer even though MUBUF
    // format stores use them; both classes lower to storage buffers.
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.format = DataFormat::Uint32;
    buffer.num_components = 4;
    buffer.binding = 2;
    buffer.gpu_addr = (uint64_t)(uintptr_t)result.data();
    buffer.size = records * 4 * sizeof(uint32_t);
    buffer.stride = 4 * sizeof(uint32_t);
    buffer.sgpr_base = 0;
    rt.resources.push_back(buffer);

    ComputeShaderConfig config;
    config.user_sgprs = {
        0, 0, 0, 0,
        0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00,
    };
    config.local_x = 64;
    config.tidig_comp_cnt = 0;
    config.tgid_x_en = true;

    std::vector<uint32_t> spirv = recompile_compute(
        code, sizeof(code) / sizeof(code[0]), &rt, config);
    CHECK(!spirv.empty(), "real Dead Cells compute kernel recompiles");
    if (spirv.empty()) return 1;

    auto report = validate_spirv_descriptor_interface(
        spirv, &rt, 0, SpirvShaderStage::Compute);
    for (const auto& issue : report.issues) {
        if (issue.error)
            std::printf("descriptor error: %s binding=%u expected=%s actual=%s\n",
                        descriptor_issue_name(issue.code), issue.binding,
                        spirv_descriptor_kind_name(issue.expected),
                        spirv_descriptor_kind_name(issue.actual));
    }
    CHECK(report.ok(), "real compute descriptor interface validates");

    ComputeItem item;
    item.spirv = spirv;
    item.resources = std::make_shared<ShaderResourceTable>(rt);
    item.launch.threads_x = records;
    item.launch.local_x = 64;
    item.launch.groups_x = 3;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x401aec200;
    item.dispatch_index = 7;
    item.submit_no = 11;
    item.command_order = 70;
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production live backend executes the game kernel");
    CHECK(result.size() == records * 4, "compute resource retains its declared size");
    const uint32_t expected[4] = {0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00};
    bool all_filled = result.size() == records * 4;
    for (uint32_t record = 0; record < records; record++) {
        for (uint32_t component = 0; component < 4; component++) {
            if (result[record * 4 + component] != expected[component]) {
                std::printf("FAIL: record %u component %u = %08x, expected %08x\n",
                            record, component, result[record * 4 + component], expected[component]);
                all_filled = false;
                break;
            }
        }
        if (!all_filled) break;
    }
    CHECK(all_filled, "all 130 records are filled across three workgroups");

    std::vector<uint32_t> replay_owned(records * 4, 0xdddddddd);
    ShaderResource replay_buffer = buffer;
    replay_buffer.gpu_addr = 1; // Deliberately unreadable: replay must never dereference this identity.
    replay_buffer.host_data = reinterpret_cast<uint8_t*>(replay_owned.data());
    replay_buffer.host_data_size = replay_owned.size() * sizeof(uint32_t);
    item.resources = std::make_shared<ShaderResourceTable>();
    item.resources->resources.push_back(replay_buffer);
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production compute backend executes against replay-owned bytes");
    bool replay_filled = true;
    for (uint32_t record = 0; record < records && replay_filled; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            replay_filled &= replay_owned[record * 4 + component] == expected[component];
    CHECK(replay_filled, "compute writeback updates owned backing for a later replay operation");

    // --- #590: the live backend's storage-IMAGE path. The same 1D image-copy kernel that
    // test_storage_image_copy proves against the raw harness, executed through the PRODUCTION
    // backend with Unorm8x4 guest-style backing — exercising the full chain: channel unpack
    // (bytes -> float-bit uvec4 texels), the R32G32B32A32_UINT image contract, and the pack-back
    // writeback (float bits -> clamped bytes). Unorm8 pack(unpack(b)) == b exactly, so a bit-exact
    // dst==src is the correctness assertion.
    static const uint32_t image_copy[] = {
        0x7E080300u, 0xF0000F00u, 0x00000004u, 0xBF8C3F70u, 0xF0200F00u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_2d[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0xF0000F08u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F08u, 0x00020004u, 0xBF810000u,
    };
    const uint32_t W = 64;
    std::vector<uint32_t> lane_index(W);
    for (uint32_t i = 0; i < W; i++) lane_index[i] = i;      // shell input: v0 = input[gid] = gid
    std::vector<uint32_t> dummy(4, 0);
    std::vector<uint8_t> img_src(W * 4), img_dst(W * 4, 0xEE);
    for (uint32_t i = 0; i < W * 4; i++) img_src[i] = (uint8_t)(i * 37 + 5);
    ShaderResourceTable irt;
    auto add_buffer = [&](uint32_t binding, void* data, uint32_t size) {
        ShaderResource b{};
        b.cls = ResourceClass::ConstantBuffer;
        b.binding = binding;
        b.gpu_addr = (uint64_t)(uintptr_t)data;
        b.size = size;
        irt.resources.push_back(b);
    };
    add_buffer(0, lane_index.data(), W * sizeof(uint32_t));
    add_buffer(1, dummy.data(), 16);
    add_buffer(2, dummy.data(), 16);
    add_buffer(3, dummy.data(), 16);
    auto add_image = [&](uint32_t binding, uint32_t sgpr, void* data, uint32_t size) {
        ShaderResource im{};
        im.cls = ResourceClass::StorageImage;
        im.img_dim = 0;                     // 1D
        im.binding = binding;
        im.sgpr_base = sgpr;
        im.format = DataFormat::Unorm8;
        im.num_components = 4;
        im.width = W; im.height = 1;
        im.gpu_addr = (uint64_t)(uintptr_t)data;
        im.size = size;
        irt.resources.push_back(im);
    };
    add_image(4, 0, img_src.data(), W * 4);
    add_image(5, 8, img_dst.data(), W * 4);
    std::vector<uint32_t> image_spirv = recompile_valu(
        image_copy, sizeof(image_copy) / sizeof(image_copy[0]), 1, 0, &irt);
    CHECK(!image_spirv.empty(), "storage-image copy kernel recompiles against the game-style table");
    if (!image_spirv.empty()) {
        ComputeItem image_item;
        image_item.spirv = image_spirv;
        image_item.resources = std::make_shared<ShaderResourceTable>(irt);
        image_item.launch.threads_x = W;
        image_item.launch.local_x = 64;
        image_item.launch.groups_x = 1;
        image_item.launch.local_y = image_item.launch.local_z = 1;
        image_item.launch.groups_y = image_item.launch.groups_z = 1;
        image_item.code_addr = 0x590590;
        CHECK(prosper::frontend::execute_live_compute_items({image_item}),
              "live backend executes the storage-image copy dispatch (#590)");
        uint32_t bad = 0;
        for (uint32_t i = 0; i < W * 4; i++) bad += img_dst[i] != img_src[i];
        if (bad) std::printf("  image copy mismatched bytes = %u/%u (b0 src=%02x dst=%02x)\n",
                             bad, W * 4, img_src[0], img_dst[0]);
        CHECK(bad == 0, "Unorm8 unpack -> uvec4 copy -> pack writeback is byte-exact (#590)");
    }

    // The backend publishes ordinary tiled texels, not hardware-compressed blocks. Prove that a
    // DCC-enabled storage destination atomically becomes the uncompressed (0xff) metadata state,
    // including replay-owned metadata that a later sampled descriptor shares by logical address.
    const uint32_t dcc_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
    const size_t tiled_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 4);
    const size_t metadata_bytes = gfx10_dcc_metadata_bytes(W, 1, 1, dcc_tile, 4, true);
    std::vector<uint8_t> tiled_src(tiled_bytes, 0), tiled_dst(tiled_bytes, 0);
    std::vector<uint8_t> dcc_metadata(metadata_bytes, 0x40);
    tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
    ShaderResourceTable dcc_rt = irt;
    for (ShaderResource& resource : dcc_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.tile_mode = dcc_tile;
        resource.size = static_cast<uint32_t>(tiled_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? tiled_src.data() : tiled_dst.data());
        if (resource.binding == 5) {
            resource.compression_enabled = true;
            resource.write_compress_enabled = true;
            resource.meta_pipe_aligned = true;
            resource.metadata_addr = 0x207cef0000ull;
            resource.dcc_metadata_size = metadata_bytes;
            resource.dcc_metadata_host_data = dcc_metadata.data();
            resource.dcc_metadata_host_data_size = dcc_metadata.size();
        }
    }
    std::vector<uint32_t> dcc_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &dcc_rt);
    CHECK(!dcc_spirv.empty(), "2D storage copy recompiles for a DCC-enabled tiled destination");
    if (!dcc_spirv.empty()) {
        ComputeItem dcc_item;
        dcc_item.spirv = dcc_spirv;
        dcc_item.resources = std::make_shared<ShaderResourceTable>(dcc_rt);
        dcc_item.launch.threads_x = W;
        dcc_item.launch.local_x = 64;
        dcc_item.launch.groups_x = 1;
        dcc_item.launch.local_y = dcc_item.launch.local_z = 1;
        dcc_item.launch.groups_y = dcc_item.launch.groups_z = 1;
        dcc_item.code_addr = 0x719dcc;
        set_test_env("PROSPER_COMPUTE_TILED_2D_STORAGE", "1");
        CHECK(prosper::frontend::execute_live_compute_items({dcc_item}),
              "live backend writes the tiled DCC storage image");
        set_test_env("PROSPER_COMPUTE_TILED_2D_STORAGE", nullptr);
        std::vector<uint8_t> dcc_linear(W * 4, 0);
        detile_surface(dcc_linear.data(), tiled_dst.data(), W, 1, dcc_tile, 0, 4);
        CHECK(dcc_linear == img_src,
              "DCC storage writeback preserves the producer's tiled base texels");
        CHECK(std::all_of(dcc_metadata.begin(), dcc_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "DCC storage writeback publishes uniform uncompressed metadata");
    }

    // A renderer-owned color target is newer than its guest backing. Recompile the same copy kernel
    // with a sampled source, publish deliberately different live pixels, and prove the production
    // backend imports the immutable renderer snapshot instead of either skipping or reading stale RAM.
    std::vector<uint8_t> stale_rtt(W * 4, 0);
    auto live_rtt = std::make_shared<std::vector<uint8_t>>(W * 4);
    std::vector<uint8_t> live_dst(W * 4, 0xEE);
    for (uint32_t i = 0; i < W * 4; ++i) (*live_rtt)[i] = static_cast<uint8_t>(i * 29 + 11);
    ShaderResourceTable live_rt = irt;
    for (ShaderResource& resource : live_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.img_dim = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(stale_rtt.data());
            resource.host_data = nullptr;
            resource.host_data_size = 0;
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else if (resource.binding == 5) {
            resource.img_dim = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(live_dst.data());
            resource.host_data = nullptr;
            resource.host_data_size = 0;
        }
    }
    const uint64_t live_addr = reinterpret_cast<uint64_t>(stale_rtt.data());
    set_live_target_query([live_addr](uint64_t addr) { return addr == live_addr; });
    set_live_target_reader(
        [live_addr, live_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != live_addr) return false;
            snapshot.width = W;
            snapshot.height = 1;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = live_rtt;
            return true;
        });
    std::vector<uint32_t> live_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &live_rt);
    CHECK(!live_spirv.empty(), "sampled-image copy kernel recompiles for renderer RTT import");
    if (!live_spirv.empty()) {
        ComputeItem live_item;
        live_item.spirv = live_spirv;
        live_item.resources = std::make_shared<ShaderResourceTable>(live_rt);
        live_item.launch.threads_x = W;
        live_item.launch.local_x = 64;
        live_item.launch.groups_x = 1;
        live_item.launch.local_y = live_item.launch.local_z = 1;
        live_item.launch.groups_y = live_item.launch.groups_z = 1;
        live_item.code_addr = 0x590591;
        CHECK(prosper::frontend::execute_live_compute_items({live_item}),
              "live backend executes a dispatch sampling a renderer-owned RTT");
        CHECK(live_dst == *live_rtt,
              "sampled renderer RTT pixels reach storage-image writeback byte-exactly");
        CHECK(live_dst != stale_rtt, "renderer RTT import does not use stale guest backing");
    }
    set_live_target_reader({});
    set_live_target_query({});

    // UE4's exposure chain samples tiny tiled Float32 surfaces. Preserve those values natively:
    // normalizing through RGBA8 would clamp negative and HDR channels before the compute shader sees
    // them. Copy a tiled Float32x4 source through the production sampled-image path into the existing
    // raw-channel Float32 storage path and require exact bits at the guest destination.
    std::vector<float> float_src(W * 4), float_dst(W * 4, 0.0f);
    for (uint32_t i = 0; i < W * 4; ++i)
        float_src[i] = (static_cast<int32_t>(i % 17) - 8) * 0.375f;
    const size_t float_tiled_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 16);
    std::vector<uint8_t> float_tiled(float_tiled_bytes, 0);
    tile_surface(float_tiled.data(), reinterpret_cast<const uint8_t*>(float_src.data()),
                 W, 1, dcc_tile, 0, 16);
    ShaderResourceTable float_rt = irt;
    for (ShaderResource& resource : float_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.format = DataFormat::Float32;
        resource.num_components = 4;
        resource.width = W;
        resource.height = resource.depth = 1;
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.tile_mode = dcc_tile;
            resource.gpu_addr = reinterpret_cast<uint64_t>(float_tiled.data());
            resource.size = static_cast<uint32_t>(float_tiled.size());
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else {
            resource.tile_mode = 0;
            resource.gpu_addr = reinterpret_cast<uint64_t>(float_dst.data());
            resource.size = static_cast<uint32_t>(float_dst.size() * sizeof(float));
        }
    }
    std::vector<uint32_t> float_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &float_rt);
    CHECK(!float_spirv.empty(), "sampled-image copy kernel recompiles for tiled Float32 input");
    if (!float_spirv.empty()) {
        ComputeItem float_item;
        float_item.spirv = float_spirv;
        float_item.resources = std::make_shared<ShaderResourceTable>(float_rt);
        float_item.launch.threads_x = W;
        float_item.launch.local_x = 64;
        float_item.launch.groups_x = 1;
        float_item.launch.local_y = float_item.launch.local_z = 1;
        float_item.launch.groups_y = float_item.launch.groups_z = 1;
        float_item.code_addr = 0x590f32;
        CHECK(prosper::frontend::execute_live_compute_items({float_item}),
              "live backend executes a dispatch sampling a tiled Float32 image");
        CHECK(float_dst == float_src,
              "tiled Float32 sampled values preserve negative and HDR channels byte-exactly");
    }

    if (fails) {
        std::printf("== FAIL: %d ==\n", fails);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
