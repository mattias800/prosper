// #3467: an unwritten SNORM minimum encoding is not an explicit store of -1.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t width = 64, height = 16, texels = width * height;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ComputeItem compile(const std::vector<uint32_t>& code, const ShaderResourceTable& resources,
                    uint64_t address) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    config.local_x = width;
    config.local_y = config.local_z = 1;
    config.native_storage_format_support = 0; // Explicitly exercise raw storage.
    ComputeItem item;
    item.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = config.user_sgprs;
    item.code_addr = address;
    item.launch.threads_x = item.launch.local_x = width;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    check(!item.spirv.empty(), "actual fixture shader compiles");
    return item;
}
ShaderResource buffer(void* data, uint32_t bytes) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = 2;
    r.sgpr_base = 0;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.stride = 4;
    r.gpu_addr = reinterpret_cast<uint64_t>(data);
    r.size = bytes;
    return r;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "8") && std::strcmp(argv[1], "16")) ||
        (std::strcmp(argv[2], "none") && std::strcmp(argv[2], "partial") &&
         std::strcmp(argv[2], "full"))) {
        std::fprintf(stderr, "usage: test_storage_snorm_untouched 8|16 none|partial|full\n");
        return 2;
    }
    const bool wide = !std::strcmp(argv[1], "16");
    const uint32_t bytes = wide ? 2 : 1;
    const uint32_t minimum = wide ? 0x8000u : 0x80u;
    const uint32_t stored = minimum + 1; // Explicit imageStore(-1) uses -32767 / -127.
    std::vector<uint8_t> guest(texels * bytes);
    std::vector<uint32_t> mask(width), expected(texels), observed(texels, 0xccccccccu);
    constexpr uint32_t unused_sentinel = 0x96a53cf0u;
    std::vector<uint32_t> unused(texels, unused_sentinel);
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5;
    image.sgpr_base = 8;
    image.img_dim = 1;
    image.format = wide ? DataFormat::Snorm16 : DataFormat::Snorm8;
    image.num_components = 1;
    image.width = width;
    image.height = height;
    image.depth = 1;
    image.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
    image.size = guest.size();
    image.linear_row_pitch_bytes = width * bytes;
    ShaderResourceTable producer_table;
    producer_table.resources = {buffer(mask.data(), mask.size() * 4), image};
    // Binding 6 is absent from the original module, but occupied in its resource
    // table. Injected instrumentation must not reuse max-reflected-binding + 1.
    auto unused_resource = buffer(unused.data(), unused.size() * 4);
    unused_resource.binding = 6;
    producer_table.resources.push_back(unused_resource);
    std::vector<uint32_t> writer{
        0x7e080300u,             // v4 = local X
        0xe0302000u, 0x80000804u, // v8 = mask[X]
        0xbf8c3f70u,
        0x7daa1080u,             // disable columns whose mask is zero
        0x7e0002ffu, 0xbf800000u, // v0 = float -1 bits
    };
    for (uint32_t y = 0; y < height; ++y)
        writer.insert(writer.end(), {0x7e0a0280u + y, 0xf0200108u, 0x00020004u});
    writer.push_back(0xbf810000u);
    const auto producer = compile(writer, producer_table, 0x34674001u);
    const auto report = validate_spirv_descriptor_interface(
        producer.spirv, &producer_table, 0, SpirvShaderStage::Compute, false);
    const auto* descriptor = find_spirv_descriptor_binding(report, 0, 5);
    check(report.ok() && descriptor && descriptor->writable && !descriptor->readable &&
          descriptor->image_numeric_class == SpirvImageNumericClass::Uint,
          "producer uses the raw integer write-only storage representation");

    // View the guest encoding as UINT, so the real GPU reader distinguishes the
    // two encodings that an SNORM sample would both report as the same -1 value.
    ShaderResource sampled = image;
    sampled.cls = ResourceClass::Texture;
    sampled.format = wide ? DataFormat::Uint16 : DataFormat::Uint8;
    for (uint32_t c = 0; c < 4; ++c) sampled.swizzle[c] = 4 + c;
    ShaderResourceTable reader_table;
    reader_table.resources = {buffer(observed.data(), observed.size() * 4), sampled};
    std::vector<uint32_t> reader{0x7e080300u};
    for (uint32_t y = 0; y < height; ++y)
        reader.insert(reader.end(), {
            0x7e0a0280u + y, 0xf0000108u, 0x00020804u, 0xbf8c3f70u,
            0xe0702000u | (y * width * 4u), 0x80000804u,
        });
    reader.push_back(0xbf810000u);
    const auto consumer = compile(reader, reader_table, 0x34674002u);
    if (failures) return 1;

    auto run = [&](const char* phase) {
        // Reset architectural bytes before every dispatch. Code, launch, SGPRs,
        // addresses and descriptors remain identical; only runtime mask data changes.
        for (uint32_t i = 0; i < texels; ++i) {
            guest[i * bytes] = static_cast<uint8_t>(minimum);
            if (wide) guest[i * bytes + 1] = static_cast<uint8_t>(minimum >> 8);
            expected[i] = mask[i % width] ? stored : minimum;
        }
        check(prosper::frontend::execute_live_compute_items({producer}), "SNORM producer executes");
        check(std::all_of(unused.begin(), unused.end(),
                          [](uint32_t value) { return value == unused_sentinel; }),
              "injected write mask never touches an occupied unreferenced resource binding");
        std::fill(observed.begin(), observed.end(), 0xccccccccu);
        check(prosper::frontend::execute_live_compute_items({consumer}), "UINT GPU reader executes");
        bool guest_ok = true;
        for (uint32_t i = 0; i < texels; ++i) {
            uint32_t actual = guest[i * bytes];
            if (wide) actual |= uint32_t(guest[i * bytes + 1]) << 8;
            if (actual != expected[i] || observed[i] != expected[i]) {
                std::fprintf(stderr, "%s SNORM%s (%u,%u): guest=%04x GPU=%04x expected=%04x\n",
                             phase, argv[1], i % width, i / width, actual, observed[i], expected[i]);
                guest_ok = false;
                break;
            }
        }
        check(guest_ok, "packed guest encoding and GPU-consumed encoding match exact store coverage");
        check(observed == expected, "all UINT GPU-consumed texels match the independent oracle");
    };
    if (std::strcmp(argv[2], "none")) std::fill(mask.begin(), mask.end(), 1);
    if (!std::strcmp(argv[2], "partial")) mask[0] = 0;
    run("first dispatch");
    run("unchanged mask");
    std::fill(mask.begin(), mask.end(), 1);
    run("same shader now explicitly stores -1 everywhere");
    return failures ? 1 : 0;
}
