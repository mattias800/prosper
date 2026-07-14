#include "../src/gpu/agc_shader_layout.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "live_compute.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } } while (0)

static void make_tsharp(uint32_t t[8], uint64_t base, uint32_t width, uint32_t height,
                        uint32_t type, uint32_t depth) {
    std::memset(t, 0, 8 * sizeof(uint32_t));
    const uint64_t encoded_base = base >> 8;
    t[0] = static_cast<uint32_t>(encoded_base);
    t[1] = static_cast<uint32_t>((encoded_base >> 32) & 0xffu) |
           (56u << 20) | (((width - 1) & 0x3u) << 30); // IMG_FMT 56 = Unorm8x4
    t[2] = (((width - 1) >> 2) & 0xfffu) | (((height - 1) & 0x3fffu) << 14);
    t[3] = (type & 0xfu) << 28; // linear tile mode + real SQ_RSRC_IMG TYPE
    t[4] = (depth - 1) & 0x1fffu;
}

static bool build_storage_image_resource(ShaderResource& out, uint8_t* data,
                                         uint32_t width, uint32_t height,
                                         uint32_t type, uint32_t depth,
                                         uint32_t binding, uint32_t sgpr_base) {
    uint32_t sgprs[8];
    make_tsharp(sgprs, reinterpret_cast<uint64_t>(data), width, height, type, depth);
    AgcShaderSharp sharp[1]; sharp[0].bits = 0;
    AgcShaderUserData user_data{};
    user_data.sharp_resource_offset[0] = sharp;
    user_data.sharp_resource_count[0] = 1;
    AgcShaderHeader header{};
    header.file_header = 0x34333231u; header.version = 0x18; header.type = 1;
    header.user_data = &user_data;
    ShaderResourceTable decoded = build_shader_resources(header, sgprs, 8);
    if (decoded.resources.size() != 1) return false;
    out = decoded.resources.front();
    out.cls = ResourceClass::StorageImage; // image_store consumer selects this class in gpu_executor.
    out.binding = binding;
    out.sgpr_base = sgpr_base;
    return true;
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

    // #657: production-backend copies for every newly enabled layered shape. Unlike the old 1D
    // fixture, image metadata here is decoded from real 8-dword T#s via build_shader_resources.
    static const uint32_t copy_3d[] = {
        0x361000bfu, 0x7e120280u, 0x2c140086u,
        0xf0000f10u, 0x00000008u, 0xbf8c3f70u,
        0xf0200f10u, 0x00020008u, 0xbf810000u,
    };
    static const uint32_t copy_1d_array[] = {
        0x361000bfu, 0x2c120086u,
        0xf0000f20u, 0x00000008u, 0xbf8c3f70u,
        0xf0200f20u, 0x00020008u, 0xbf810000u,
    };
    static const uint32_t copy_2d_array[] = {
        0x361000bfu, 0x7e120280u, 0x2c140086u,
        0xf0000f28u, 0x00000008u, 0xbf8c3f70u,
        0xf0200f28u, 0x00020008u, 0xbf810000u,
    };
    auto run_layered_copy = [&](const uint32_t* code_words, size_t word_count,
                                uint32_t tsharp_type, uint32_t expected_dim,
                                const char* shape_name) {
        constexpr uint32_t width = 64, height = 1, slices = 3;
        constexpr size_t image_bytes = static_cast<size_t>(width) * height * slices * 4;
        std::vector<uint8_t> backing(image_bytes * 2 + 512);
        const uintptr_t first = (reinterpret_cast<uintptr_t>(backing.data()) + 255u) & ~uintptr_t(255u);
        uint8_t* layered_src = reinterpret_cast<uint8_t*>(first);
        uint8_t* layered_dst = layered_src + image_bytes; // image_bytes is 256-byte aligned.
        for (size_t i = 0; i < image_bytes; ++i) layered_src[i] = static_cast<uint8_t>(i * 29 + 11);
        std::memset(layered_dst, 0xee, image_bytes);

        const uint32_t threads = width * height * slices;
        std::vector<uint32_t> indices(threads);
        for (uint32_t i = 0; i < threads; ++i) indices[i] = i;
        std::vector<uint32_t> scratch(4, 0);
        ShaderResourceTable layered_table;
        auto add_layered_buffer = [&](uint32_t binding, void* ptr, uint32_t bytes) {
            ShaderResource resource{};
            resource.cls = ResourceClass::ConstantBuffer; resource.binding = binding;
            resource.gpu_addr = reinterpret_cast<uint64_t>(ptr); resource.size = bytes;
            layered_table.resources.push_back(resource);
        };
        add_layered_buffer(0, indices.data(), threads * sizeof(uint32_t));
        add_layered_buffer(1, scratch.data(), 16);
        add_layered_buffer(2, scratch.data(), 16);
        add_layered_buffer(3, scratch.data(), 16);
        ShaderResource src_image{}, dst_image{};
        const bool src_built = build_storage_image_resource(
            src_image, layered_src, width, height, tsharp_type, slices, 4, 0);
        const bool dst_built = build_storage_image_resource(
            dst_image, layered_dst, width, height, tsharp_type, slices, 5, 8);
        CHECK(src_built && dst_built, "layered storage images decode from real T# descriptors");
        CHECK(src_built && src_image.img_dim == expected_dim && src_image.depth == slices &&
              src_image.size == image_bytes,
              "layered T# carries the expected dimension, slice count, and backing size");
        if (!src_built || !dst_built) return;
        layered_table.resources.push_back(src_image);
        layered_table.resources.push_back(dst_image);

        std::vector<uint32_t> layered_spirv = recompile_valu(
            code_words, word_count, 1, 0, &layered_table);
        CHECK(!layered_spirv.empty(), "layered storage-image copy kernel recompiles");
        if (layered_spirv.empty()) return;
        ComputeItem layered_item;
        layered_item.spirv = std::move(layered_spirv);
        layered_item.resources = std::make_shared<ShaderResourceTable>(layered_table);
        layered_item.launch.threads_x = threads; layered_item.launch.local_x = 64;
        layered_item.launch.groups_x = threads / 64;
        layered_item.launch.local_y = layered_item.launch.local_z = 1;
        layered_item.launch.groups_y = layered_item.launch.groups_z = 1;
        layered_item.code_addr = 0x657000 + expected_dim;
        const bool executed = prosper::frontend::execute_live_compute_items({layered_item});
        if (!executed) std::printf("  layered backend rejected %s\n", shape_name);
        CHECK(executed, "production backend executes layered storage-image dispatch");
        CHECK(std::memcmp(layered_src, layered_dst, image_bytes) == 0,
              "layered upload -> copy -> writeback is byte-exact across every slice");
    };
    run_layered_copy(copy_3d, sizeof(copy_3d) / sizeof(copy_3d[0]), 10, 2, "3D");
    run_layered_copy(copy_1d_array, sizeof(copy_1d_array) / sizeof(copy_1d_array[0]), 12, 4, "1D_ARRAY");
    run_layered_copy(copy_2d_array, sizeof(copy_2d_array) / sizeof(copy_2d_array[0]), 13, 5, "2D_ARRAY");

    if (fails) {
        std::printf("== FAIL: %d ==\n", fails);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
