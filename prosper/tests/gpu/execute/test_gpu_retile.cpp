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
int main(int argc, char** argv) {
    const bool cpu = argc == 2 && std::strcmp(argv[1], "--cpu") == 0;
    const bool mapping = argc == 2 && std::strncmp(argv[1], "--map-", 6) == 0;
    const bool allocation = argc == 2 && std::strncmp(argv[1], "--alloc-", 8) == 0;
    const bool fault_mode = mapping || allocation;
    const bool loss = fault_mode && std::strstr(argv[1], "loss");
    VkPhysicalDeviceLimits limits{};
    limits.maxStorageBufferRange = UINT32_MAX;
    limits.maxComputeWorkGroupSize[0] = limits.maxComputeWorkGroupInvocations = 128;
    limits.maxComputeWorkGroupCount[0] = limits.maxComputeWorkGroupCount[1] = 65535;
    limits.maxPushConstantsSize = 128;
    GpuRetileParameters params;
    check(params.initialize(3840, 2160, 4, 27, limits) &&
              params.linear_bytes == 33177600 && params.tiled_bytes == 33423360,
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
    struct Format { DataFormat format; uint32_t components, bpe; };
    constexpr Format formats[]{ {DataFormat::Uint32, 1, 4}, {DataFormat::Uint8, 4, 4},
        {DataFormat::Unorm8, 4, 4}, {DataFormat::Float16, 4, 8}, {DataFormat::Float32, 4, 16} };
    uint64_t code = 0x34070000;
    for (uint32_t mode : {9u, 24u, 27u}) for (auto format : formats)
    for (auto [width, height] : {std::pair{257u, 131u}, std::pair{384u, 256u}})
    for (uint32_t view : {0u, 1u, 2u}) {
        // A one-layer array descriptor may be consumed through either a plain
        // 2D instruction or an array instruction retaining its layer coordinate.
        const uint32_t resource_dim = view ? 5u : 1u;
        const uint32_t instruction_dim = view == 2 ? 5u : 1u;
        const size_t linear_bytes = size_t(width) * height * format.bpe;
        const size_t tiled_bytes = tiled_surface_bytes(width, height, mode, 0, format.bpe);
        std::vector<uint8_t> source_linear(linear_bytes), expected_linear(linear_bytes);
        std::vector<uint8_t> source(tiled_bytes), destination(tiled_bytes + 64, 0xcc);
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
        tile_surface(destination.data(), expected_linear.data(), width, height, mode, 0, format.bpe);
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
            r.width = width; r.height = height; r.depth = 1;
            r.format = format.format; r.num_components = format.components; r.tile_mode = mode;
            r.size = uint32_t(tiled_bytes);
            r.gpu_addr = reinterpret_cast<uint64_t>(binding == 4 ? source.data() : destination.data());
            resources.resources.push_back(r);
        }
        const auto before = gpu_retile_recordings().load();
        for (uint32_t round = 0; round < 3; ++round) {
            fill(source_linear, 7 + round * 31);
            tile_surface(source.data(), source_linear.data(), width, height, mode, 0, format.bpe);
            const uint32_t row = round == 0 ? 0 : round == 1 ? height / 2 : height - 1;
            const uint32_t first_x = round == 2 ? width - 64 : 0;
            const uint32_t shader[]{
                0x4A0800FFu, first_x, // v_add_nc_u32 v4, first_x, v0
                0x7E0A02FFu, row, // v5=selected y
                0x7E0C0280u, // v6=layer zero for the array instruction
                0xF0000F00u | (instruction_dim << 3), 0x00000004u, 0xBF8C3F70u,
                0xF0200F00u | (instruction_dim << 3), 0x00020004u, 0xBF810000u};
            ComputeItem item;
            ComputeShaderConfig config;
            config.user_sgprs.resize(16); config.local_x = 64;
            config.local_y = config.local_z = 1; config.tidig_comp_cnt = 0;
            config.native_storage_format_support = native_storage_format_support_bit(
                format.format, format.components);
            item.spirv = recompile_compute(shader, std::size(shader), &resources, config);
            item.resources = std::make_shared<ShaderResourceTable>(resources);
            item.user_sgprs = config.user_sgprs;
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
            check(gpu_retile_recordings().load() - retile_before == (cpu ? 0 : inject ? 1 : 2),
                  "both storage images take the selected GPU/CPU path, including the destination");
            if (code == 0x34070000 && round == 0 && !inject)
                check(backend_host_read_barrier_count().load() - barriers_before == (cpu ? 2 : 4),
                      "first dispatch records host availability for both linear and both tiled results");
            for (uint32_t lane : coordinates) {
                const uint32_t x = lane + first_x;
                std::copy_n(source_linear.data() + (size_t(row) * width + x) * format.bpe,
                            format.bpe, expected_linear.data() + (size_t(row) * width + x) * format.bpe);
            }
            std::vector<uint8_t> expected(tiled_bytes);
            tile_surface(expected.data(), expected_linear.data(), width, height, mode, 0, format.bpe);
            check(std::equal(expected.begin(), expected.end(), destination.begin()),
                  "every guest byte matches CPU tiling, including untouched texels and padding");
            check(std::all_of(destination.begin() + tiled_bytes, destination.end(),
                             [](uint8_t v) { return v == 0xcc; }), "writeback keeps destination guard bytes");
        }
        std::printf("mode=%u bpe=%u format=%u extent=%ux%u resource-dim=%u instruction-dim=%u GPU retile dispatches=%llu\n",
            mode, format.bpe, unsigned(format.format), width, height, resource_dim, instruction_dim,
            static_cast<unsigned long long>(gpu_retile_recordings().load() - before));
        check(cpu ? gpu_retile_recordings().load() == before : gpu_retile_recordings().load() == before + (fault_mode ? 5 : 6),
              cpu ? "CPU fallback used" : "GPU tiler actually recorded; CPU equivalence alone is insufficient");
        if (fault_mode) return failures ? 1 : 0;
        code += 16;
    }
    return failures ? 1 : 0;
}
