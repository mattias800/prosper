// Execute guest storage writes through the production backend, then compare every
// tiled byte (including padding and untouched texels) to the established CPU walk.
#include "shared/live/gpu_retile.hpp"
#include "shared/live/live_compute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::frontend;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
static bool equation_covers_block(uint32_t mode, uint32_t bpe, uint32_t bx,
                                 uint32_t by, bool broken = false) {
    std::array<uint32_t, 16> equation{};
    uint32_t bw = 0, bh = 0;
    if (!(bpe == 2 ? tile64_paired16_equation(mode, equation, bw, bh)
                   : tile64_word_equation(mode, bpe, equation, bw, bh))) return false;
    if (broken) equation[15] = 0; // deliberately collapse one address bit
    std::vector<bool> seen(65536 / 4);
    size_t written = 0;
    for (uint32_t y = by * bh; y < (by + 1) * bh; ++y)
    for (uint32_t x = bx * bw; x < (bx + 1) * bw; x += bpe == 2 ? 2 : 1)
    for (uint32_t component = 0; component < std::max(1u, bpe / 4); ++component) {
        uint32_t offset = component * 4;
        for (uint32_t bit = 2; bit < 16; ++bit) {
            const uint32_t parity = std::popcount(
                (x & (equation[bit] & 65535)) ^ (y & (equation[bit] >> 16))) & 1u;
            offset |= parity << bit;
        }
        if (offset >= 65536 || (offset & 3) || seen[offset / 4]) return false;
        seen[offset / 4] = true; ++written;
    }
    return written == seen.size();
}
static bool volume_equation_covers_block(uint32_t mode, uint32_t bpe, bool collapse = false) {
    std::array<uint32_t, 16> equation{};
    uint32_t bw = 0, bh = 0, bd = 0, bits = 0;
    if (!tile_volume_word_equation(mode, bpe, equation, bw, bh, bd, bits)) return false;
    if (collapse) equation[bits - 1] = 0;
    std::vector<bool> seen(size_t{1} << (bits - 2));
    size_t written = 0;
    for (uint32_t z = 0; z < bd; ++z) for (uint32_t y = 0; y < bh; ++y)
    for (uint32_t x = 0; x < bw; ++x) for (uint32_t component = 0; component < bpe / 4; ++component) {
        uint32_t offset = component * 4;
        for (uint32_t bit = 2; bit < bits; ++bit) {
            const auto e = equation[bit];
            offset |= uint32_t(std::popcount((x & (e & 255)) ^
                (y & ((e >> 8) & 255)) ^ (z & (e >> 16))) & 1) << bit;
        }
        if ((offset & 3) || offset / 4 >= seen.size() || seen[offset / 4]) return false;
        seen[offset / 4] = true; ++written;
    }
    return written == seen.size();
}

int run_case(int argc, char** argv) {
    const bool mixed = argc == 2 && std::strstr(argv[1], "mixed");
    const bool volume = argc == 2 && std::strstr(argv[1], "volume");
    const bool array16 = argc == 2 && std::strstr(argv[1], "array16");
    const bool shape_refusal = array16 && std::strstr(argv[1], "shape-refusal");
    const bool cpu = argc == 2 && std::strstr(argv[1], "cpu");
    const bool clean = argc == 2 && std::strcmp(argv[1], "--clean") == 0;
    const bool mapping = argc == 2 && std::strstr(argv[1], "map-");
    const bool allocation = argc == 2 && std::strstr(argv[1], "alloc-");
    const bool fault_mode = mapping || allocation;
    const bool loss = fault_mode && std::strstr(argv[1], "loss");
    gpu_retile_poison_output_for_test().store(!cpu && !clean);
    VkPhysicalDeviceLimits limits{};
    limits.maxStorageBufferRange = UINT32_MAX;
    limits.maxComputeWorkGroupSize[0] = limits.maxComputeWorkGroupInvocations = 128;
    limits.maxComputeWorkGroupCount[0] = limits.maxComputeWorkGroupCount[1] =
        limits.maxComputeWorkGroupCount[2] = 65535;
    limits.maxPushConstantsSize = 128;
    GpuRetileParameters params;
    check(params.initialize(3840, 2160, 4, 27, limits) &&
              params.linear_bytes == 33177600 && params.tiled_bytes == 33423360 &&
              params.groups_x == 30 && params.groups_y == 2176,
          "measured 4K layout is admitted with exact linear and padded sizes");
    check(!params.initialize(UINT32_MAX, UINT32_MAX, 16, 27, limits) &&
          !params.initialize(0, 2160, 4, 27, limits) &&
          !params.initialize(3840, 2160, 3, 27, limits) &&
          !params.initialize(3840, 2160, 4, 5, limits),
          "overflow, degenerate texels and other layouts decline");
    for (unsigned field = 0; field < 5; ++field) {
        auto reduced = limits;
        if (field == 0) reduced.maxStorageBufferRange = 1024;
        if (field == 1) reduced.maxPushConstantsSize = 64;
        if (field == 2) reduced.maxComputeWorkGroupInvocations = 64;
        if (field == 3) reduced.maxComputeWorkGroupCount[0] = 1;
        if (field == 4) reduced.maxComputeWorkGroupCount[1] = 1;
        check(!params.initialize(3840, 2160, 4, 27, reduced), "optional device limit declines");
    }
    auto padded_limits = limits;
    padded_limits.maxComputeWorkGroupCount[1] = 2160;
    check(!params.initialize(3840, 2160, 4, 27, padded_limits),
          "workgroup limit applies to padded height, not just visible rows");
    padded_limits = limits; padded_limits.maxComputeWorkGroupCount[0] = 3;
    check(!params.initialize(129, 1, 8, 24, padded_limits),
          "workgroup limit applies to padded width, not just visible columns");
    for (uint32_t mode : {9u, 24u, 27u}) for (uint32_t bpe : {4u, 8u, 16u})
    for (auto [bx, by] : {std::pair{0u, 0u}, std::pair{1u, 1u},
                          std::pair{3u, 5u}, std::pair{17u, 33u}})
        check(equation_covers_block(mode, bpe, bx, by),
              "full block equation writes every output word exactly once, including nonzero block coordinates");
    check(!equation_covers_block(27, 4, 0, 0, true),
          "equation coverage guard rejects a manually collapsed address bit");
    for (uint32_t mode : {5u, 9u}) for (uint32_t bpe : {4u, 8u, 16u})
        check(volume_equation_covers_block(mode, bpe),
              "3D equation writes every physical word exactly once");
    check(!volume_equation_covers_block(9, 8, true),
          "3D coverage guard rejects a collapsed coordinate bit");
    // Metadata-only predicate checks, not execution of a multi-mip/MSAA image.
    // Real GPU/CPU integration below separately covers bytes, strides and widths.
    ShaderResource descriptor{};
    descriptor.width = descriptor.height = 512;
    check(gpu_retile_paired16_descriptor_supported(descriptor, 2, 1),
          "descriptor predicate accepts unspecified allocation hints for an ordinary base view");
    descriptor.mip_chain_element_width = descriptor.mip_chain_element_height = 512;
    descriptor.mip_chain_bytes_per_block = 2;
    check(gpu_retile_paired16_descriptor_supported(descriptor, 2, 1),
          "descriptor predicate accepts exact allocation hints");
    check(!gpu_retile_paired16_descriptor_supported(descriptor, 4, 1) &&
          !gpu_retile_paired16_descriptor_supported(descriptor, 2, 2),
          "descriptor predicate excludes other native widths and backend mip counts");
    uint8_t metadata_byte = 0xff;
    for (unsigned field = 0; field < 16; ++field) {
        auto changed = descriptor;
        switch (field) {
        case 0: changed.sample_count = 2; break;
        case 1: changed.declared_mip_levels = 2; break;
        case 2: changed.mip_chain_base_level = 1; break;
        case 3: changed.mip_chain_max_level = 1; break;
        case 4: changed.linear_row_pitch_bytes = 1024; break;
        case 5: changed.mip_tail_offset = 4; break;
        case 6: changed.mip_tail_bytes = 65536; break;
        case 7: changed.mip_tail_x = 1; break;
        case 8: changed.mip_tail_y = 1; break;
        case 9: changed.compression_enabled = true; break;
        case 10: changed.write_compress_enabled = true; break;
        case 11: changed.metadata_addr = reinterpret_cast<uint64_t>(&metadata_byte); break;
        case 12: changed.dcc_metadata_host_data = &metadata_byte; break;
        case 13: changed.mip_chain_element_width = 256; break;
        case 14: changed.mip_chain_element_height = 256; break;
        case 15: changed.mip_chain_bytes_per_block = 4; break;
        }
        check(!gpu_retile_paired16_descriptor_supported(changed, 2, 1),
              "descriptor-only sample/mip/metadata/pitch/hint change refuses paired retile");
    }
    check(params.initialize_paired16_array(512, 512, 6, 24, limits) &&
          params.linear_bytes == 3145728 && params.tiled_bytes == 3145728 &&
          params.words[23] == 131072 && params.words[24] == 131072 &&
          params.groups_x == 2 && params.groups_y == 512 && params.groups_z == 6,
          "six ordinary R16 layers have exact independent word strides");
    check(!params.initialize_paired16_array(511, 512, 6, 24, limits) &&
          !params.initialize_paired16_array(512, 512, 0, 24, limits) &&
          !params.initialize_paired16_array(0, 512, 6, 24, limits) &&
          !params.initialize_paired16_array(512, 0, 6, 24, limits) &&
          !params.initialize_paired16_array(512, 512, 6, 27, limits) &&
          !params.initialize_paired16_array(UINT32_MAX - 1, UINT32_MAX, UINT32_MAX, 24, limits) &&
          !params.initialize_paired16_array(512, 512, UINT32_MAX, 24, limits),
          "unpaired rows, unsupported modes, zero shape and overflow decline");
    for (unsigned field = 0; field < 7; ++field) {
        auto reduced = limits;
        if (field == 0) reduced.maxStorageBufferRange = 3145727;
        if (field == 1) reduced.maxPushConstantsSize = 103;
        if (field == 2) reduced.maxComputeWorkGroupInvocations = 127;
        if (field == 3) reduced.maxComputeWorkGroupCount[0] = 1;
        if (field == 4) reduced.maxComputeWorkGroupCount[1] = 511;
        if (field == 5) reduced.maxComputeWorkGroupCount[2] = 5;
        if (field == 6) reduced.maxComputeWorkGroupSize[0] = 127;
        check(!params.initialize_paired16_array(512, 512, 6, 24, reduced),
              "paired array respects every device range and dispatch limit");
    }
    for (auto [bx, by] : {std::pair{0u, 0u}, std::pair{1u, 1u},
                          std::pair{3u, 5u}, std::pair{17u, 33u}})
        check(equation_covers_block(24, 2, bx, by), "paired words cover each complete tile without overlap");
    check(!equation_covers_block(24, 2, 0, 0, true), "paired permutation guard detects a collapsed bit");
    struct Format { DataFormat format; uint32_t components, bpe; };
    const std::vector<Format> formats = array16 ? std::vector<Format>{{DataFormat::Uint16, 1, 2},
        {DataFormat::Unorm8, 2, 2}, {DataFormat::Float16, 1, 2}} : std::vector<Format>{ {DataFormat::Uint32, 1, 4}, {DataFormat::Uint8, 4, 4},
        {DataFormat::Unorm8, 4, 4}, {DataFormat::Float16, 4, 8}, {DataFormat::Float32, 4, 16} };
    check(params.initialize_volume(64, 64, 128, 8, 9, limits) &&
              params.linear_bytes == 4194304 && params.tiled_bytes == 4194304 &&
              params.groups_z == 128, "representative 4 MiB standard 3D RGBA16F volume is admitted");
    check(!params.initialize_volume(UINT32_MAX, UINT32_MAX, UINT32_MAX, 16, 9, limits) &&
          !params.initialize_volume(64, 64, 128, 2, 9, limits) &&
          !params.initialize_volume(64, 64, 128, 8, 27, limits),
          "overflow, sub-word texels and unsupported 3D layouts decline");
    auto depth_limits = limits; depth_limits.maxComputeWorkGroupCount[2] = 21;
    check(!params.initialize_volume(67, 19, 21, 8, 9, depth_limits),
          "3D workgroup limit applies to padded depth");
    const std::vector<uint32_t> modes = array16 ? std::vector<uint32_t>{24} : mixed ? std::vector<uint32_t>{9} : volume ? std::vector<uint32_t>{5, 9}
                                              : std::vector<uint32_t>{9, 24, 27};
    const std::vector<std::pair<uint32_t, uint32_t>> extents = array16
        ? (shape_refusal ? std::vector<std::pair<uint32_t, uint32_t>>{{257, 131}, {258, 131}}
                         : std::vector<std::pair<uint32_t, uint32_t>>{{258, 131}, {512, 512}}) : mixed
        ? std::vector<std::pair<uint32_t, uint32_t>>{{67, 19}} : volume
        ? std::vector<std::pair<uint32_t, uint32_t>>{{67, 19}, {128, 64}}
        : std::vector<std::pair<uint32_t, uint32_t>>{{257, 131}, {384, 256}};
    uint64_t code = 0x34070000;
    for (uint32_t mode : modes) for (auto format : formats)
    for (auto [width, height] : extents)
    for (uint32_t view = 0; view < ((volume || mixed || array16) ? 1u : 3u); ++view) {
        if (mixed && !array16 && format.format != DataFormat::Float16) continue;
        // Integer 3D images still use the existing raw interchange path. Its packing fallback
        // remains covered; this retile change does not broaden native storage declarations.
        const bool native_volume = native_storage_3d_format_support_bit(
            format.format, format.components) != 0;
        const bool gpu = !cpu && !shape_refusal && (!volume || native_volume);
        if (fault_mode && volume && !native_volume) continue;
        // A one-layer array descriptor may be consumed through either a plain
        // 2D instruction or an array instruction retaining its layer coordinate.
        const uint32_t resource_dim = volume ? 2u : (array16 || view) ? 5u : 1u;
        const uint32_t instruction_dim = volume ? 2u : (array16 || view == 2) ? 5u : 1u;
        const uint32_t depth = array16 ? 6u : volume ? (width == 67 ? 21u : 32u) : 1u;
        const size_t linear_bytes = size_t(width) * height * depth * format.bpe;
        const size_t tile_slice = tiled_surface_bytes(width, height, mode, 0, format.bpe);
        const size_t layer_stride = tile_slice + (shape_refusal && !(width & 1) ? 65536 : 0);
        const size_t tiled_bytes = volume ? tiled_volume_bytes(width, height, depth, mode, format.bpe)
                                         : layer_stride * (depth - 1) + tile_slice;
        auto tile = [&](uint8_t* dst, const uint8_t* src) {
            if (volume) check(tile_volume(dst, tiled_bytes, src, width, height, depth, mode, format.bpe),
                              "CPU volume layout supports the test resource");
            else for (uint32_t layer = 0; layer < depth; ++layer)
                tile_surface(dst + layer * layer_stride,
                             src + size_t(layer) * width * height * format.bpe,
                             width, height, mode, 0, format.bpe);
        };
        std::vector<uint8_t> source_linear(linear_bytes), expected_linear(linear_bytes);
        std::vector<uint8_t> source(tiled_bytes, 0xb7), destination(tiled_bytes + 64, 0xcc);
        auto fill = [&](std::vector<uint8_t>& bytes, uint32_t salt) {
            auto mixed = [&](uint32_t index) {
                uint32_t v = index + salt * 0x9e3779b9u;
                v ^= v >> 16; v *= 0x85ebca6bu;
                v ^= v >> 13; v *= 0xc2b2ae35u; return v ^ (v >> 16);
            };
            if (format.format == DataFormat::Float16) {
                for (size_t i = 0; i < bytes.size() / 2; ++i) {
                    const auto bits = mixed(uint32_t(i));
                    const uint16_t v = uint16_t((bits & 0x8000) |
                        ((1 + bits % 29) << 10) | (bits & 1023)); // finite normal halves
                    std::memcpy(bytes.data() + i * 2, &v, 2);
                }
            } else {
                for (size_t i = 0; i < bytes.size() / 4; ++i) {
                    uint32_t v = mixed(uint32_t(i));
                    if (format.format == DataFormat::Float32)
                        v = (v & 0x807fffffu) | ((100 + v % 50) << 23); // finite normal floats
                    std::memcpy(bytes.data() + i * 4, &v, 4);
                }
            }
        };
        fill(expected_linear, 173);
        tile(destination.data(), expected_linear.data());
        std::vector<uint32_t> coordinates(64), dummy(4);
        for (uint32_t lane = 0; lane < 64; ++lane) coordinates[lane] = lane;
        ShaderResourceTable resources;
        for (uint32_t binding = 0; binding < 4; ++binding) {
            ShaderResource r{}; r.cls = ResourceClass::ConstantBuffer; r.binding = binding;
            r.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy.data() : coordinates.data());
            r.size = binding ? 16 : uint32_t(coordinates.size() * 4);
            resources.resources.push_back(r);
        }
        for (uint32_t binding : {4u, 5u}) {
            ShaderResource r{}; r.cls = ResourceClass::StorageImage; r.binding = binding;
            r.sgpr_base = binding == 4 ? 0 : 8; r.img_dim = resource_dim;
            r.width = width; r.height = height; r.depth = depth;
            r.format = format.format; r.num_components = format.components; r.tile_mode = mode;
            r.size = uint32_t(tiled_bytes);
            if (array16) {
                // Both inferred and explicit exact strides occur in decoded resources.
                r.layer_stride_bytes = (width == 512 || shape_refusal) ? uint32_t(layer_stride) : 0;
                r.mip_chain_element_width = width; r.mip_chain_element_height = height;
                r.mip_chain_bytes_per_block = format.bpe;
            }
            r.gpu_addr = reinterpret_cast<uint64_t>(binding == 4 ? source.data() : destination.data());
            resources.resources.push_back(r);
        }
        const uint64_t output_images =
            std::getenv("PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP") ? 2 : 1;
        const auto before = gpu_retile_recordings().load();
        for (uint32_t round = 0; round < 3; ++round) {
            fill(source_linear, 7 + round * 31);
            tile(source.data(), source_linear.data());
            const uint32_t row = round == 0 ? 0 : round == 1 ? height / 2 : height - 1;
            // Odd starts/ends independently update the high and low halves of paired stores.
            const uint32_t first_x = round == 2 ? width - 64 : (array16 && round == 1 ? 1 : 0);
            const uint32_t slice = round == 0 ? 0 : round == 1 ? depth / 2 : depth - 1;
            const uint32_t shader[]{
                0x4A0800FFu, first_x, // v_add_nc_u32 v4, first_x, v0
                0x7E0A02FFu, row, // v5=selected y
                0x7E0C02FFu, slice, // v6=selected depth slice or array layer zero
                0xF0000F00u | (instruction_dim << 3), 0x00000004u, 0xBF8C3F70u,
                0xF0200F00u | (instruction_dim << 3), 0x00020004u, 0xBF810000u};
            ComputeItem item;
            ComputeShaderConfig config;
            config.user_sgprs.resize(16); config.local_x = 64;
            config.local_y = config.local_z = 1; config.tidig_comp_cnt = 0;
            config.native_storage_format_support = volume
                ? native_storage_3d_format_support_bit(format.format, format.components)
                : native_storage_format_support_bit(format.format, format.components);
            item.spirv = recompile_compute(shader, std::size(shader), &resources, config);
            item.resources = std::make_shared<ShaderResourceTable>(resources);
            item.user_sgprs = config.user_sgprs;
            const auto reflection = validate_spirv_descriptor_interface(
                item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
            size_t matching_views = 0;
            for (const auto& descriptor : reflection.descriptors)
                if (descriptor.kind == SpirvDescriptorKind::StorageImage &&
                    (descriptor.binding == 4 || descriptor.binding == 5) &&
                    descriptor.image_dim == (volume ? 2u : 1u) && descriptor.image_arrayed == (array16 || view == 2))
                    ++matching_views;
            check(reflection.ok() && matching_views == 2,
                  "both storage views reflect the intended ordinary or array shader type");
            const auto* source_access = find_spirv_descriptor_binding(reflection, 0, 4);
            const auto* destination_access = find_spirv_descriptor_binding(reflection, 0, 5);
            check(reflection.storage_image_writes_complete && source_access && destination_access &&
                      source_access->readable && !source_access->writable && !source_access->atomic_access &&
                      destination_access->writable,
                  "complete reflection proves one read-only source and one writable destination");
            item.launch.threads_x = item.launch.local_x = 64;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = code + round;
            const auto barriers_before = backend_host_read_barrier_count().load();
            const auto retile_before = gpu_retile_recordings().load();
            const bool inject = fault_mode && code == 0x34070000 && round == 0;
            const auto submits_before = live_compute_queue_submit_attempts();
            const auto guest_before = destination;
            if (inject) live_compute_fail_next_retile_memory_for_test(mapping, loss);
            const bool executed = !item.spirv.empty() && execute_live_compute_items({item});
            if (inject) check(!live_compute_retile_memory_fault_pending_for_test(),
                              "injected error reached the actual allocator driver boundary");
            if (inject && loss) {
                check(!executed && destination == guest_before &&
                      live_compute_queue_submit_attempts() == submits_before,
                      "allocation or eager-map device loss stops before submission and guest writeback");
                check(!execute_live_compute_items({item}) &&
                      live_compute_queue_submit_attempts() == submits_before,
                      "device loss remains sticky on the following dispatch");
                return failures ? 1 : 0;
            }
            check(executed, "partial guest image copy executes through live Vulkan");
            check(gpu_retile_recordings().load() - retile_before == (!gpu ? 0 : output_images - (inject ? 1 : 0)),
                  "every actual output takes the selected GPU/CPU path");
            if (code == 0x34070000 && round == 0 && !inject)
                check(backend_host_read_barrier_count().load() - barriers_before == output_images * (gpu ? 2 : 1),
                      "first dispatch records host availability for every linear and tiled output");
            for (uint32_t lane : coordinates) {
                const uint32_t x = lane + first_x;
                std::copy_n(source_linear.data() + ((size_t(slice) * height + row) * width + x) * format.bpe,
                            format.bpe, expected_linear.data() + ((size_t(slice) * height + row) * width + x) * format.bpe);
            }
            std::vector<uint8_t> expected(tiled_bytes, 0xcc);
            tile(expected.data(), expected_linear.data());
            check(std::equal(expected.begin(), expected.end(), destination.begin()),
                  "every guest byte matches CPU tiling, including untouched texels and padding");
            check(std::all_of(destination.begin() + tiled_bytes, destination.end(),
                             [](uint8_t v) { return v == 0xcc; }), "writeback keeps destination guard bytes");
        }
        std::printf("mode=%u bpe=%u format=%u extent=%ux%u resource-dim=%u instruction-dim=%u GPU retile dispatches=%llu\n",
            mode, format.bpe, unsigned(format.format), width, height, resource_dim, instruction_dim,
            static_cast<unsigned long long>(gpu_retile_recordings().load() - before));
        check(!gpu ? gpu_retile_recordings().load() == before : gpu_retile_recordings().load() == before + 3 * output_images - (fault_mode ? 1 : 0),
              !gpu ? "CPU fallback used" : "GPU tiler actually recorded; CPU equivalence alone is insufficient");
        if (fault_mode) return failures ? 1 : 0;
        code += 16;
    }
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--mixed") == 0) {
        // All three use the same retained live context and its immutable retile pipelines.
        for (const char* option : {"--mixed-2d", "--mixed-volume", "--mixed-array16", "--mixed-2d"}) {
            char* args[]{argv[0], const_cast<char*>(option)};
            if (run_case(2, args)) return 1;
        }
        return 0;
    }
    return run_case(argc, argv);
}
