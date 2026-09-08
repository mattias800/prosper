// #3467: raw write-only residency preserves packed pixels, not canonical shader inputs.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t width = 64, height = 16, texels = width * height;
constexpr uint32_t raw_half = 0x3f000000u; // 0.5 packs to 128, but 128/255 is not 0.5.
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ComputeItem compile(const std::vector<uint32_t>& code, const ShaderResourceTable& table,
                    uint64_t address) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = width;
    config.local_y = config.local_z = 1;
    config.native_storage_format_support = 0; // Exercise the raw-uvec4 fallback explicitly.
    ComputeItem item;
    item.spirv = recompile_compute(code.data(), code.size(), &table, config);
    item.resources = std::make_shared<ShaderResourceTable>(table);
    item.user_sgprs = config.user_sgprs;
    item.code_addr = address;
    item.launch.threads_x = item.launch.local_x = width;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    check(!item.spirv.empty(), "fixture shader compiles");
    return item;
}
ShaderResource buffer(void* data, uint32_t bytes, uint32_t stride) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = 2;
    r.sgpr_base = 0;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.stride = stride;
    r.gpu_addr = reinterpret_cast<uint64_t>(data);
    r.size = bytes;
    return r;
}
void append_output(std::vector<uint32_t>& code, uint32_t row) {
    for (uint32_t c = 0; c < 4; ++c)
        code.insert(code.end(), {
            0xe0702000u | (row * width * 16u + c * 4u),
            0x80000004u | ((8u + c) << 8), // store v8..v11, v4 index, s[0:3]
        });
}
} // namespace

int main() {
    std::vector<uint8_t> guest(texels * 4, 17), expected(texels * 4, 128);
    std::vector<uint32_t> mask(width, 1), observed(texels * 4, 0xccccccccu);
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5;
    image.sgpr_base = 8;
    image.img_dim = 1;
    image.format = DataFormat::Unorm8;
    image.num_components = 4;
    image.width = width;
    image.height = height;
    image.depth = 1;
    image.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
    image.size = guest.size();
    ShaderResourceTable writer_table;
    writer_table.resources = {buffer(mask.data(), mask.size() * 4, 4), image};
    std::vector<uint32_t> writer{
        0x7e080300u,             // v4 = local x
        0xe0302000u, 0x80000804u, // v8 = mask[v4]
        0xbf8c3f70u,
        0x7daa1080u,             // v_cmpx_ne_u32 0, v8: whole-column mask
    };
    for (uint32_t c = 0; c < 4; ++c)
        writer.insert(writer.end(), {0x7e0002ffu + (c << 17), raw_half});
    for (uint32_t y = 0; y < height; ++y)
        writer.insert(writer.end(), {0x7e0a0280u + y, 0xf0200f08u, 0x00020004u});
    writer.push_back(0xbf810000u);
    const auto producer = compile(writer, writer_table, 0x34673001u);
    const auto writer_report = validate_spirv_descriptor_interface(
        producer.spirv, &writer_table, 0, SpirvShaderStage::Compute, false);
    const auto* output_image = find_spirv_descriptor_binding(writer_report, 0, 5);
    check(writer_report.ok() && output_image && output_image->writable &&
              !output_image->readable && !output_image->atomic_access &&
              output_image->image_numeric_class == SpirvImageNumericClass::Uint,
          "producer is an actual raw integer write-only storage image");

    // A sampled reader materializes canonical guest values, independently of the raw output image.
    ShaderResource sampled = image;
    sampled.cls = ResourceClass::Texture;
    for (uint32_t c = 0; c < 4; ++c) sampled.swizzle[c] = 4 + c;
    ShaderResourceTable reader_table;
    reader_table.resources = {buffer(observed.data(), observed.size() * 4, 16), sampled};
    std::vector<uint32_t> reader{0x7e080300u};
    for (uint32_t y = 0; y < height; ++y) {
        reader.insert(reader.end(), {0x7e0a0280u + y, 0xf0000f08u, 0x00020804u, 0xbf8c3f70u});
        append_output(reader, y);
    }
    reader.push_back(0xbf810000u);
    const auto consumer = compile(reader, reader_table, 0x34673002u);
    auto pixels = [&](const char* phase) {
        check(guest == expected, "every architectural packed guest channel is preserved");
        bool correct = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            const uint32_t canonical = std::bit_cast<uint32_t>(expected[i] / 255.0f);
            if (observed[i] != canonical) {
                std::fprintf(stderr, "%s channel %zu: GPU=%08x expected=%08x guest=%u\n",
                             phase, i, observed[i], canonical, guest[i]);
                correct = false;
                break;
            }
        }
        check(correct, "every independent GPU-consumed channel is canonical");
    };
    auto produce_and_read = [&](const char* phase) {
        std::printf("raw residency phase: %s\n", phase);
        std::fflush(stdout);
        check(prosper::frontend::execute_live_compute_items({producer}), "raw producer executes");
        std::fill(observed.begin(), observed.end(), 0xccccccccu);
        check(prosper::frontend::execute_live_compute_items({consumer}), "independent GPU reader executes");
        pixels(phase);
    };
    if (failures) return 1;
    check(std::bit_cast<uint32_t>(128.0f / 255.0f) != raw_half,
          "raw retained value differs from its canonical packed/unpacked representation");
    produce_and_read("full");
    produce_and_read("warm full");
    mask[0] = 0;
    produce_and_read("partial unchanged guest");
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t c = 0; c < 4; ++c)
            guest[(y * width) * 4 + c] = expected[(y * width) * 4 + c] = 65 + c;
    produce_and_read("partial changed guest");
    produce_and_read("warm partial");

    // Join a readable sibling to the same storage VkImage. It must receive canonical guest
    // inputs, even though the previous write-only raw image can still validate as packed bytes.
    ShaderResource readable = image;
    readable.binding = 6;
    readable.sgpr_base = 16;
    ShaderResourceTable mixed_table;
    mixed_table.resources = {buffer(observed.data(), observed.size() * 4, 16), image, readable};
    std::vector<uint32_t> mixed{0x7e080300u};
    for (uint32_t y = 0; y < height; ++y) {
        mixed.insert(mixed.end(), {0x7e0a0280u + y, 0xf0000f08u, 0x00040804u, 0xbf8c3f70u});
        append_output(mixed, y);
        mixed.insert(mixed.end(), {0xf0200f08u, 0x00020804u}); // preserve loaded canonical channels
    }
    mixed.push_back(0xbf810000u);
    const auto mixed_item = compile(mixed, mixed_table, producer.code_addr);
    const auto mixed_report = validate_spirv_descriptor_interface(
        mixed_item.spirv, &mixed_table, 0, SpirvShaderStage::Compute, false);
    const auto* mixed_writer = find_spirv_descriptor_binding(mixed_report, 0, 5);
    const auto* mixed_reader = find_spirv_descriptor_binding(mixed_report, 0, 6);
    check(mixed_report.ok() && mixed_writer && mixed_reader && mixed_writer->writable &&
              !mixed_writer->readable && mixed_reader->readable && !mixed_reader->writable,
          "same-backing transition has distinct actual write-only and readable aliases");
    if (failures) return 1;
    std::fill(observed.begin(), observed.end(), 0xccccccccu);
    std::puts("raw residency phase: readable alias transition");
    std::fflush(stdout);
    check(prosper::frontend::execute_live_compute_items({mixed_item}), "mixed alias dispatch executes");
    pixels("readable alias transition");
    std::printf("raw residency safety: %d failures\n", failures);
    return failures ? 1 : 0;
}
