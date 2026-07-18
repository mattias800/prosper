#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/tile.hpp"
#include "live_compute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } } while (0)

int main() {
    // MinGW's lround dominates full-HD storage-image writeback. Prove the bounded integer path is
    // identical to the previous conversion across all half-float values, every UNORM threshold and
    // adjacent float, plus a deterministic million-value float32 sample.
    auto reference_unorm8 = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (std::isnan(value)) value = 0.0f;
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<uint8_t>(std::lround(value * 255.0f));
    };
    auto check_unorm8 = [&](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return prosper::frontend::storage_pack_unorm8(bits) == reference_unorm8(bits);
    };
    bool unorm8_matches = true;
    const float edge_values[] = {
        -std::numeric_limits<float>::infinity(), -1.0f, -0.0f, 0.0f,
        std::numeric_limits<float>::denorm_min(), 1.0f,
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
    };
    for (float value : edge_values) unorm8_matches &= check_unorm8(value);
    for (uint32_t i = 0; i < 255; ++i) {
        const float threshold = (static_cast<float>(i) + 0.5f) / 255.0f;
        unorm8_matches &= check_unorm8(std::nextafter(threshold, 0.0f));
        unorm8_matches &= check_unorm8(threshold);
        unorm8_matches &= check_unorm8(std::nextafter(threshold, 1.0f));
    }
    for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const float value = prosper::gpu::half_to_float(static_cast<uint16_t>(bits));
        unorm8_matches &= check_unorm8(value);
    }
    uint32_t random_bits = 0x6d2b79f5u;
    for (uint32_t i = 0; i < 1000000; ++i) {
        random_bits ^= random_bits << 13;
        random_bits ^= random_bits >> 17;
        random_bits ^= random_bits << 5;
        if (prosper::frontend::storage_pack_unorm8(random_bits) !=
            reference_unorm8(random_bits)) {
            unorm8_matches = false;
            break;
        }
    }
    CHECK(unorm8_matches, "fast UNORM8 pack is equivalent to the lround reference");

    // Dead Cells' bound startup fill kernel, copied verbatim from eboot.elf at runtime address
    // 0x401aec200. It stores s4-s7 to record `(TGID_X << 6) + local_id_x` through the V# in s0-s3.
    static const uint32_t code[] = {
        0xd7460004, 0x04010c08, 0x7e000204, 0x7e020205, 0x7e040206,
        0x7e060207, 0xe01c2000, 0x80000004, 0xbf810000,
    };

    constexpr uint32_t records = 130;
    constexpr uint32_t launched_records = 3 * 64;
    std::vector<uint32_t> result(launched_records * 4, 0xcccccccc);
    ShaderResourceTable rt;
    ShaderResource buffer;
    // Runtime metadata classifies compute direct type-1 V#s as ConstantBuffer even though MUBUF
    // format stores use them; both classes lower to storage buffers.
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.format = DataFormat::Uint32;
    buffer.num_components = 4;
    buffer.binding = 2;
    buffer.gpu_addr = (uint64_t)(uintptr_t)result.data();
    // Deliberately expose the padded Vulkan lanes as valid writable storage. Only the generated
    // exact-thread guard can keep records 130..191 unchanged; robust buffer access cannot hide a bug.
    buffer.size = launched_records * 4 * sizeof(uint32_t);
    buffer.stride = 4 * sizeof(uint32_t);
    buffer.sgpr_base = 0;
    rt.resources.push_back(buffer);

    ComputeShaderConfig config;
    config.user_sgprs = {
        0, 0, 0, 0,
        0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00,
    };
    config.local_x = 64;
    config.exact_thread_extent = true;
    config.threads_x = records;
    config.threads_y = config.threads_z = 1;
    config.tidig_comp_cnt = 0;
    config.tgid_x_en = true;

    std::vector<uint32_t> spirv = recompile_compute(
        code, sizeof(code) / sizeof(code[0]), &rt, config);
    CHECK(!spirv.empty(), "real Dead Cells compute kernel recompiles");
    if (spirv.empty()) return 1;
    ComputeShaderConfig alternate_config = config;
    alternate_config.user_sgprs[4] ^= 0xffffffffu;
    alternate_config.user_sgprs[7] ^= 0x13579bdfu;
    CHECK(recompile_compute(code, sizeof(code) / sizeof(code[0]), &rt, alternate_config) == spirv,
          "per-dispatch user SGPR values do not specialize the reusable SPIR-V module");

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
    item.user_sgprs = config.user_sgprs;
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
    CHECK(result.size() == launched_records * 4, "compute resource retains its padded declared size");
    const uint32_t expected[4] = {0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00};
    bool all_filled = result.size() == launched_records * 4;
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
    bool padded_lanes_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            padded_lanes_untouched &= result[record * 4 + component] == 0xcccccccc;
    CHECK(padded_lanes_untouched,
          "partial workgroup suppresses all 62 padded invocations without a guest bounds check");

    std::fill(result.begin(), result.end(), 0xeeeeeeee);
    item.user_sgprs = alternate_config.user_sgprs;
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "cached compute pipeline executes with updated user SGPR push constants");
    const uint32_t alternate_expected[4] = {
        alternate_config.user_sgprs[4], alternate_config.user_sgprs[5],
        alternate_config.user_sgprs[6], alternate_config.user_sgprs[7],
    };
    bool alternate_filled = true;
    for (uint32_t record = 0; record < records && alternate_filled; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            alternate_filled &= result[record * 4 + component] == alternate_expected[component];
    CHECK(alternate_filled,
          "cached compute pipeline observes per-dispatch user SGPR values");
    item.user_sgprs = config.user_sgprs;

    std::vector<uint32_t> replay_owned(launched_records * 4, 0xdddddddd);
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
    bool replay_padding_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            replay_padding_untouched &= replay_owned[record * 4 + component] == 0xdddddddd;
    CHECK(replay_padding_untouched, "captured/replay-owned backing preserves padded invocations");

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
        CHECK(prosper::frontend::execute_live_compute_items({dcc_item}),
              "live backend writes the tiled DCC storage image");
        std::vector<uint8_t> dcc_linear(W * 4, 0);
        detile_surface(dcc_linear.data(), tiled_dst.data(), W, 1, dcc_tile, 0, 4);
        CHECK(dcc_linear == img_src,
              "DCC storage writeback preserves the producer's tiled base texels");
        CHECK(std::all_of(dcc_metadata.begin(), dcc_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "DCC storage writeback publishes uniform uncompressed metadata");
    }

    // Two storage views may share base texels while only one carries DCC state. They are not safe
    // Vulkan-image aliases: collapsing the compressed view onto an uncompressed owner drops its
    // separate metadata writeback obligation and leaves later sampled users reading stale DCC.
    std::vector<uint8_t> mixed_alias_base = tiled_src;
    std::vector<uint8_t> mixed_alias_metadata(metadata_bytes, 0x40);
    ShaderResourceTable mixed_alias_rt = dcc_rt;
    for (ShaderResource& resource : mixed_alias_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.gpu_addr = reinterpret_cast<uint64_t>(mixed_alias_base.data());
        if (resource.binding == 5) {
            resource.metadata_addr = 0x207cf00000ull;
            resource.dcc_metadata_host_data = mixed_alias_metadata.data();
            resource.dcc_metadata_host_data_size = mixed_alias_metadata.size();
        }
    }
    const std::vector<uint32_t> mixed_alias_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &mixed_alias_rt);
    CHECK(!mixed_alias_spirv.empty(), "mixed DCC storage-alias kernel recompiles");
    if (!mixed_alias_spirv.empty()) {
        ComputeItem mixed_alias_item;
        mixed_alias_item.spirv = mixed_alias_spirv;
        mixed_alias_item.resources = std::make_shared<ShaderResourceTable>(mixed_alias_rt);
        mixed_alias_item.launch.threads_x = W;
        mixed_alias_item.launch.local_x = 64;
        mixed_alias_item.launch.groups_x = 1;
        mixed_alias_item.launch.local_y = mixed_alias_item.launch.local_z = 1;
        mixed_alias_item.launch.groups_y = mixed_alias_item.launch.groups_z = 1;
        mixed_alias_item.code_addr = 0x719dcd;
        CHECK(prosper::frontend::execute_live_compute_items({mixed_alias_item}),
              "mixed compressed/uncompressed storage views execute independently");
        CHECK(std::all_of(mixed_alias_metadata.begin(), mixed_alias_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "compressed storage alias retains its DCC writeback obligation");
    }

    // Exercise the same production path with a tiled 2D guest surface. The kernel copies row zero;
    // every untouched row must survive upload/readback/retiling exactly. This guards the inverse
    // address mapping used by live tiled StorageImage dispatches, not only the pure tile helper.
    constexpr uint32_t TILED_H = 19;
    constexpr uint32_t tiled2d_mode = static_cast<uint32_t>(TileMode::Sw4KbS);
    std::vector<uint8_t> tiled2d_src_linear(W * TILED_H * 4);
    std::vector<uint8_t> tiled2d_dst_initial(W * TILED_H * 4, 0x5a);
    for (size_t i = 0; i < tiled2d_src_linear.size(); ++i)
        tiled2d_src_linear[i] = static_cast<uint8_t>(i * 41 + 7);
    const size_t tiled2d_bytes = tiled_surface_bytes(W, TILED_H, tiled2d_mode, 0, 4);
    std::vector<uint8_t> tiled2d_src(tiled2d_bytes, 0), tiled2d_dst(tiled2d_bytes, 0);
    tile_surface(tiled2d_src.data(), tiled2d_src_linear.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);

    ShaderResourceTable tiled2d_rt = irt;
    for (ShaderResource& resource : tiled2d_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.width = W;
        resource.height = TILED_H;
        resource.depth = 1;
        resource.tile_mode = tiled2d_mode;
        resource.size = static_cast<uint32_t>(tiled2d_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? tiled2d_src.data() : tiled2d_dst.data());
    }
    const std::vector<uint32_t> tiled2d_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &tiled2d_rt);
    CHECK(!tiled2d_spirv.empty(), "tiled 2D storage-image copy kernel recompiles");
    if (!tiled2d_spirv.empty()) {
        ComputeItem tiled2d_item;
        tiled2d_item.spirv = tiled2d_spirv;
        tiled2d_item.resources = std::make_shared<ShaderResourceTable>(tiled2d_rt);
        tiled2d_item.launch.threads_x = W;
        tiled2d_item.launch.local_x = 64;
        tiled2d_item.launch.groups_x = 1;
        tiled2d_item.launch.local_y = tiled2d_item.launch.local_z = 1;
        tiled2d_item.launch.groups_y = tiled2d_item.launch.groups_z = 1;
        tiled2d_item.code_addr = 0x590592;
        CHECK(prosper::frontend::execute_live_compute_items({tiled2d_item}),
              "production backend executes a tiled 2D storage-image dispatch");

        std::vector<uint8_t> tiled2d_result(W * TILED_H * 4, 0);
        detile_surface(tiled2d_result.data(), tiled2d_dst.data(), W, TILED_H,
                       tiled2d_mode, 0, 4);
        std::vector<uint8_t> tiled2d_expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, tiled2d_expected.begin());
        CHECK(tiled2d_result == tiled2d_expected,
              "tiled 2D storage-image writeback matches the linear reference byte-exactly");
    }

    // Exercise the wider element mappings that become default-on with tiled storage. Float16 uses
    // finite values so half->float->half is exact; Float32 follows the backend's raw channel contract.
    auto wide_tiled_roundtrip = [&](DataFormat format, uint32_t bytes_per_texel,
                                    uint64_t code_addr) {
        const size_t linear_bytes = static_cast<size_t>(W) * TILED_H * bytes_per_texel;
        std::vector<uint8_t> src_linear(linear_bytes, 0);
        std::vector<uint8_t> dst_initial(linear_bytes, 0);
        if (format == DataFormat::Float16) {
            for (size_t i = 0; i < linear_bytes / 2; ++i) {
                const uint16_t src_half = float_to_half(
                    static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f);
                const uint16_t dst_half = float_to_half(
                    static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f);
                std::memcpy(src_linear.data() + i * 2, &src_half, sizeof(src_half));
                std::memcpy(dst_initial.data() + i * 2, &dst_half, sizeof(dst_half));
            }
        } else {
            for (size_t i = 0; i < linear_bytes / 4; ++i) {
                const uint32_t src_word = static_cast<uint32_t>(i * 2654435761u + 0x1020304u);
                const uint32_t dst_word = static_cast<uint32_t>(i * 2246822519u + 0x5060708u);
                std::memcpy(src_linear.data() + i * 4, &src_word, sizeof(src_word));
                std::memcpy(dst_initial.data() + i * 4, &dst_word, sizeof(dst_word));
            }
        }
        const size_t tiled_bytes_wide = tiled_surface_bytes(
            W, TILED_H, tiled2d_mode, 0, bytes_per_texel);
        std::vector<uint8_t> src_tiled(tiled_bytes_wide, 0);
        std::vector<uint8_t> dst_tiled(tiled_bytes_wide, 0);
        tile_surface(src_tiled.data(), src_linear.data(), W, TILED_H,
                     tiled2d_mode, 0, bytes_per_texel);
        tile_surface(dst_tiled.data(), dst_initial.data(), W, TILED_H,
                     tiled2d_mode, 0, bytes_per_texel);
        ShaderResourceTable wide_rt = tiled2d_rt;
        for (ShaderResource& resource : wide_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.format = format;
            resource.num_components = 4;
            resource.size = static_cast<uint32_t>(tiled_bytes_wide);
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? src_tiled.data() : dst_tiled.data());
        }
        const std::vector<uint32_t> wide_spirv = recompile_valu(
            image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
            &wide_rt);
        if (wide_spirv.empty()) return false;
        ComputeItem wide_item;
        wide_item.spirv = wide_spirv;
        wide_item.resources = std::make_shared<ShaderResourceTable>(wide_rt);
        wide_item.launch.threads_x = W;
        wide_item.launch.local_x = 64;
        wide_item.launch.groups_x = 1;
        wide_item.launch.local_y = wide_item.launch.local_z = 1;
        wide_item.launch.groups_y = wide_item.launch.groups_z = 1;
        wide_item.code_addr = code_addr;
        if (!prosper::frontend::execute_live_compute_items({wide_item})) return false;
        std::vector<uint8_t> result(linear_bytes, 0);
        detile_surface(result.data(), dst_tiled.data(), W, TILED_H,
                       tiled2d_mode, 0, bytes_per_texel);
        std::vector<uint8_t> expected = dst_initial;
        std::copy_n(src_linear.begin(), static_cast<size_t>(W) * bytes_per_texel,
                    expected.begin());
        return result == expected;
    };
    CHECK(wide_tiled_roundtrip(DataFormat::Float16, 8, 0x590595),
          "tiled Float16x4 storage writeback is byte-exact");
    CHECK(wide_tiled_roundtrip(DataFormat::Float32, 16, 0x590596),
          "tiled Float32x4 storage writeback is byte-exact");

    ShaderResourceTable unsupported_tiled_rt = tiled2d_rt;
    for (ShaderResource& resource : unsupported_tiled_rt.resources)
        if (resource.binding == 4 || resource.binding == 5) resource.tile_mode = 6;
    const std::vector<uint32_t> unsupported_tiled_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &unsupported_tiled_rt);
    CHECK(!unsupported_tiled_spirv.empty(), "unsupported tiled storage kernel still recompiles");
    if (!unsupported_tiled_spirv.empty()) {
        ComputeItem unsupported_tiled_item;
        unsupported_tiled_item.spirv = unsupported_tiled_spirv;
        unsupported_tiled_item.resources =
            std::make_shared<ShaderResourceTable>(unsupported_tiled_rt);
        unsupported_tiled_item.launch.threads_x = W;
        unsupported_tiled_item.launch.local_x = 64;
        unsupported_tiled_item.launch.groups_x = 1;
        unsupported_tiled_item.launch.local_y = unsupported_tiled_item.launch.local_z = 1;
        unsupported_tiled_item.launch.groups_y = unsupported_tiled_item.launch.groups_z = 1;
        unsupported_tiled_item.code_addr = 0x590594;
        CHECK(!prosper::frontend::execute_live_compute_items({unsupported_tiled_item}),
              "unknown nonzero tile mode is rejected before storage writeback");
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

    // A renderer target can become a writable storage image before its pixels have been materialized
    // in guest RAM. Seed the dispatch from the immutable renderer snapshot, modify row zero, and
    // prove writeback preserves every untouched row while publishing a cache-invalidation write.
    std::vector<uint8_t> writable_rtt_guest(tiled2d_bytes, 0);
    auto writable_rtt = std::make_shared<std::vector<uint8_t>>(W * TILED_H * 4);
    for (size_t i = 0; i < writable_rtt->size(); ++i)
        (*writable_rtt)[i] = static_cast<uint8_t>(i * 17 + 3);
    ShaderResourceTable writable_rt = tiled2d_rt;
    for (ShaderResource& resource : writable_rt.resources) {
        if (resource.binding == 5) {
            resource.gpu_addr = reinterpret_cast<uint64_t>(writable_rtt_guest.data());
            resource.size = static_cast<uint32_t>(writable_rtt_guest.size());
        }
    }
    const uint64_t writable_addr = reinterpret_cast<uint64_t>(writable_rtt_guest.data());
    set_live_target_query([writable_addr](uint64_t addr) { return addr == writable_addr; });
    set_live_target_reader(
        [writable_addr, writable_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != writable_addr) return false;
            snapshot.width = W;
            snapshot.height = TILED_H;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = writable_rtt;
            return true;
        });
    bool writable_rtt_published = false;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
        writable_rtt_published |= addr == writable_addr && size == writable_rtt_guest.size();
    });
    const std::vector<uint32_t> writable_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &writable_rt);
    CHECK(!writable_spirv.empty(), "writable renderer RTT copy kernel recompiles");
    if (!writable_spirv.empty()) {
        ComputeItem writable_item;
        writable_item.spirv = writable_spirv;
        writable_item.resources = std::make_shared<ShaderResourceTable>(writable_rt);
        writable_item.launch.threads_x = W;
        writable_item.launch.local_x = 64;
        writable_item.launch.groups_x = 1;
        writable_item.launch.local_y = writable_item.launch.local_z = 1;
        writable_item.launch.groups_y = writable_item.launch.groups_z = 1;
        writable_item.code_addr = 0x590593;
        CHECK(prosper::frontend::execute_live_compute_items({writable_item}),
              "live backend executes a dispatch writing a renderer-owned RTT");
        std::vector<uint8_t> writable_result(W * TILED_H * 4, 0);
        detile_surface(writable_result.data(), writable_rtt_guest.data(), W, TILED_H,
                       tiled2d_mode, 0, 4);
        std::vector<uint8_t> writable_expected = *writable_rtt;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, writable_expected.begin());
        CHECK(writable_result == writable_expected,
              "writable renderer RTT seeds from live pixels and preserves untouched rows");
        CHECK(writable_rtt_published,
              "writable renderer RTT publishes guest writeback for cache invalidation");
    }
    set_guest_gpu_write_observer({});
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
