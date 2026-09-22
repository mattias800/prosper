// A one-level retained storage result must not replace a complete sampled mip upload.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t W = 256, H = 256, Bpt = 8, MaxMip = 8, ReadWidth = 32;
constexpr uint32_t BaseR = 0x3ec00000u, BaseG = 0x3f200000u;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ComputeItem compile(const std::vector<uint32_t>& code, const ShaderResourceTable& resources,
                    uint32_t width, uint32_t index) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    config.local_x = width;
    config.local_y = config.local_z = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Float32, 2);
    ComputeItem item;
    item.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    item.user_sgprs = config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.code_addr = 0x30480000ull + index * 0x100u;
    item.dispatch_index = index;
    item.command_order = index * 10u;
    item.launch.threads_x = item.launch.local_x = width;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    check(!item.spirv.empty(), "real producer/consumer shader compiles");
    return item;
}
} // namespace

int main() {
    const uint32_t tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
    const auto base = tiled_mip_level_layout(W, H, Bpt, tile, MaxMip, 0);
    const size_t bytes = tiled_mip_chain_bytes(W, H, Bpt, tile, MaxMip);
    check(base.supported && !base.in_tail && bytes, "level-zero identity has a complete placed chain");
    if (!base.supported || base.in_tail || !bytes) return 1;
    std::vector<uint8_t> allocation(bytes, 0);
    std::array<std::vector<uint32_t>, 2> expected;
    for (size_t arm = 0; arm < expected.size(); ++arm) {
        const uint32_t level = arm ? 3u : 1u;
        const auto layout = tiled_mip_level_layout(W, H, Bpt, tile, MaxMip, level);
        check(layout.supported, "checked mip has a proven ordinary or tail placement");
        if (!layout.supported) return 1;
        const uint32_t width = W >> level, height = H >> level;
        std::vector<uint32_t> pixels(size_t(width) * height * 2u);
        for (size_t i = 0; i < pixels.size() / 2u; ++i) {
            // Exact float bits, distinct both from mip zero and the other requested mip.
            pixels[i * 2u] = 0x3e800001u + static_cast<uint32_t>(i) + level * 0x10000u;
            pixels[i * 2u + 1u] = 0x3f000003u + static_cast<uint32_t>(i) + level * 0x10000u;
        }
        if (layout.in_tail)
            tile_surface_level(allocation.data(), allocation.size(),
                               reinterpret_cast<const uint8_t*>(pixels.data()), width, height,
                               tile, Bpt, layout.tail_x, layout.tail_y);
        else
            tile_surface(allocation.data() + layout.byte_offset,
                         reinterpret_cast<const uint8_t*>(pixels.data()), width, height, tile, 0, Bpt);
        expected[arm].assign(pixels.begin(), pixels.begin() + ReadWidth * 2u);
    }
    const auto guest_before = allocation;
    ShaderResource source{};
    source.cls = ResourceClass::StorageImage;
    source.binding = 5; source.sgpr_base = 8; source.img_dim = 1;
    source.format = DataFormat::Float32; source.num_components = 2;
    source.width = W; source.height = H; source.depth = 1;
    source.tile_mode = tile;
    source.gpu_addr = reinterpret_cast<uint64_t>(allocation.data() + base.byte_offset);
    source.size = static_cast<uint32_t>(tiled_surface_bytes(W, H, tile, 0, Bpt));
    for (uint32_t c = 0; c < 4; ++c) source.swizzle[c] = 4u + c;
    ShaderResourceTable writer_table; writer_table.resources = {source};
    std::vector<uint32_t> writer{0x7e080300u, 0x7e0002ffu, BaseR, 0x7e0202ffu, BaseG};
    for (uint32_t y = 0; y < H; ++y)
        writer.insert(writer.end(), {0x7e0a02ffu, y, 0xf0200308u, 0x00020004u});
    writer.push_back(0xbf810000u);
    std::vector<ComputeItem> items{compile(writer, writer_table, W, 1)};
    std::array<std::vector<uint32_t>, 3> outputs;
    for (uint32_t arm = 0; arm < outputs.size(); ++arm) {
        outputs[arm].assign(ReadWidth * 2u, 0xdeadbeefu);
        auto sampled = source;
        sampled.cls = ResourceClass::Texture;
        if (arm) {
            sampled.declared_mip_levels = MaxMip + 1u;
            sampled.mip_chain_element_width = W; sampled.mip_chain_element_height = H;
            sampled.mip_chain_bytes_per_block = Bpt; sampled.mip_chain_max_level = MaxMip;
            sampled.mip_chain_base_level = 0;
            check(shader_resource_compute_mip_chain_levels(sampled) == MaxMip + 1u,
                  "multi-level consumer declares all nine placed levels");
        }
        ShaderResource output{};
        output.cls = ResourceClass::ConstantBuffer; output.binding = 2; output.sgpr_base = 0;
        output.format = DataFormat::Uint32; output.num_components = 1; output.stride = Bpt;
        output.gpu_addr = reinterpret_cast<uint64_t>(outputs[arm].data());
        output.size = static_cast<uint32_t>(outputs[arm].size() * sizeof(uint32_t));
        ShaderResourceTable table; table.resources = {output, sampled};
        const uint32_t level = arm == 2u ? 3u : arm;
        std::vector<uint32_t> reader{
            0x7e080300u,                 // v4 = local x, also independent SSBO index
            0x7e020280u,                 // v1 = y = 0
            0x7e040280u + level,         // v2 = literal mip
            arm ? 0xf0040308u : 0xf0000308u, 0x00020800u, // LOAD[_MIP] xy -> v8,v9
            0xbf8c3f70u,
            0xe0702000u, 0x80000804u,    // buffer_store_dword v8, index v4, byte offset 0
            0xe0702004u, 0x80000904u,    // buffer_store_dword v9, same index, byte offset 4
            0xbf810000u,
        };
        items.push_back(compile(reader, table, ReadWidth, arm + 2u));
    }
    if (std::any_of(items.begin(), items.end(), [](const auto& item) { return item.spirv.empty(); })) return 1;
    std::vector<SubmitOperation> operations;
    for (const auto& item : items)
        operations.push_back({SubmitOperationKind::Dispatch, item.dispatch_index, item.command_order});
    uint32_t calls = 0;
    bool all_ok = true;
    const auto result = execute_ordered_items(operations, {}, items,
        [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
        [&](const std::vector<ComputeItem>& batch) {
            ++calls;
            const auto before = prosper::frontend::live_compute_storage_transfer_seeds();
            const bool ok = prosper::frontend::execute_live_compute_items(batch);
            all_ok &= ok;
            const auto after = prosper::frontend::live_compute_storage_transfer_seeds();
            check(batch.size() == 1u, "ordered journal retains separate producer/consumer callbacks");
            if (batch.size() == 1u && batch[0].dispatch_index == 2u)
                check(after > before, "single-level control actually seeds from retained native storage");
            if (batch.size() == 1u && batch[0].dispatch_index >= 3u)
                check(after == before, "complete mip chain refuses the one-level retained transfer");
            return ok;
        }, 1, 1);
    check(result.compute_executed && all_ok && calls == items.size(), "all ordered GPU dispatches execute");
    for (uint32_t x = 0; x < ReadWidth; ++x)
        check(outputs[0][2u*x] == BaseR && outputs[0][2u*x+1u] == BaseG,
              "single-level GPU transfer preserves both exact Float32 components");
    check(outputs[1] == expected[0], "mip1 nonzero exact float bits reach the independent SSBO");
    check(outputs[2] == expected[1], "mip3 distinct nonzero exact float bits reach the independent SSBO");
    // This layout stores the entire tail and higher levels before the selected level-zero span.
    check(base.byte_offset > 0 &&
              std::memcmp(allocation.data(), guest_before.data(), base.byte_offset) == 0,
          "base storage writer preserves actual higher-mip guest bytes including the tail");
    return failures ? 1 : 0;
}
