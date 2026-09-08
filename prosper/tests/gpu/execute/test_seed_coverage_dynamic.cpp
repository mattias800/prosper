// #3467: observed full coverage need not survive a launch or input change.
// Run each mode in a fresh process. PROSPER_NO_SKIP_SEED=1 is an external control;
// neither mode treats corrupted output as an expected or passing result.
#include "gpu/execute/gpu_execute.hpp"
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
constexpr uint32_t width = 128, height = 8, texels = width * height;
constexpr uint32_t stored = 0x13579bdfu;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ComputeItem item(const std::vector<uint32_t>& code, const ShaderResourceTable& resources,
                 uint64_t address, uint32_t index) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    config.local_x = 64;
    config.local_y = config.local_z = 1;
    config.tidig_comp_cnt = 0;
    config.tgid_x_en = config.tgid_y_en = true;
    config.tgid_z_en = false;
    config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Uint32, 1);
    ComputeItem result;
    result.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    result.user_sgprs = config.user_sgprs;
    result.resources = std::make_shared<ShaderResourceTable>(resources);
    result.code_addr = address;
    result.dispatch_index = index;
    result.command_order = index * 10;
    result.launch.local_x = 64;
    result.launch.local_y = result.launch.local_z = 1;
    result.launch.groups_x = 2;
    result.launch.groups_y = result.launch.groups_z = 1;
    result.launch.threads_x = width;
    result.launch.threads_y = result.launch.threads_z = 1;
    return result;
}
} // namespace

int main(int argc, char** argv) {
    const bool launch = argc == 2 && std::strcmp(argv[1], "launch") == 0;
    const bool mask_mode = argc == 2 && std::strcmp(argv[1], "mask") == 0;
    const bool coordinates = argc == 2 && std::strcmp(argv[1], "coordinates") == 0;
    const bool layered = argc == 2 && std::strcmp(argv[1], "layers") == 0;
    if (!launch && !mask_mode && !coordinates && !layered) {
        std::fprintf(stderr, "usage: test_seed_coverage_dynamic launch|mask|coordinates|layers\n");
        return 2;
    }
    const uint32_t layers = layered ? 3 : 1;
    std::vector<uint32_t> guest(texels * layers), observed(texels * layers, 0xccccccccu);
    // The second half contains destinations for the coordinate-only transition.
    std::vector<uint32_t> mask(width * 2, 1), expected(texels * layers, stored);
    for (uint32_t x = 0; x < width; ++x) mask[width + x] = x;
    for (uint32_t i = 0; i < guest.size(); ++i) guest[i] = 0xabc00000u + i;
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5;
    image.sgpr_base = 8;
    image.img_dim = layered ? 5 : 1;
    image.format = DataFormat::Uint32;
    image.num_components = 1;
    image.width = width;
    image.height = height;
    image.depth = layers;
    image.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
    image.size = guest.size() * sizeof(uint32_t);
    if (layered) {
        image.linear_row_pitch_bytes = width * sizeof(uint32_t);
        image.layer_stride_bytes = texels * sizeof(uint32_t);
    }
    ShaderResourceTable producer_resources;
    producer_resources.resources.push_back(image);
    if (!launch) {
        ShaderResource input{};
        input.cls = ResourceClass::ConstantBuffer;
        input.binding = 2;
        input.sgpr_base = 0;
        input.format = DataFormat::Uint32;
        input.num_components = 1;
        input.stride = sizeof(uint32_t);
        input.gpu_addr = reinterpret_cast<uint64_t>(mask.data());
        input.size = mask.size() * sizeof(uint32_t);
        producer_resources.resources.push_back(input);
    }

    // Workgroup IDs follow the 16 user SGPRs: s16 = X, s17 = Y. Reject Y != 0
    // uniformly so groups (1,2,1) leave half the image unwritten without races.
    std::vector<uint32_t> writer{
        0xbf068011u,             // s_cmp_eq_u32 s17, 0
        0xbf840000u,             // s_cbranch_scc0 end (patched below)
        0xd7460004u, 0x04010c10u, // v4 = (s16 << 6) + v0: global X
    };
    if (layered) {
        writer.erase(writer.begin(), writer.begin() + 2);
        writer.insert(writer.end(), {
            0x7e0c0211u,             // v6 = s17: array layer
            0x7e0e0211u,             // v7 = s17: per-layer mask index
        });
    }
    if (!launch) {
        writer.insert(writer.end(), {
            0xe0302000u, layered ? 0x80000807u : 0x80000804u,
            0xbf8c3f70u,
            0x7daa1080u,             // v_cmpx_ne_u32 0, v8
        });
    }
    if (coordinates) {
        writer.insert(writer.end(), {
            0xe0302000u | (width * 4u), 0x80000904u, // destination[x] -> v9
            0xbf8c3f70u,
            0x7e080309u,             // v4 = destination; mask is unchanged when it moves
        });
    }
    writer.insert(writer.end(), {0x7e0002ffu, stored}); // v0 = stored integer
    for (uint32_t y = 0; y < height; ++y) {
        writer.insert(writer.end(), {
            0x7e0a0280u + y,         // v5 = row
            layered ? 0xf0200128u : 0xf0200108u, 0x00020004u,
        });
    }
    if (!layered) writer[1] |= static_cast<uint32_t>(writer.size() - 2);
    writer.push_back(0xbf810000u);
    const uint64_t address = launch ? 0x34671001u : 0x34672001u;
    ComputeItem producer = item(writer, producer_resources, address, 1);
    producer.launch.groups_y = producer.launch.threads_y = layers;
    const auto original_spirv = producer.spirv;
    const auto original_sgprs = producer.user_sgprs;

    ShaderResource sampled = image;
    sampled.cls = ResourceClass::Texture;
    for (unsigned c = 0; c < 4; ++c) sampled.swizzle[c] = 4 + c;
    ShaderResource output{};
    output.cls = ResourceClass::ConstantBuffer;
    output.binding = 2;
    output.sgpr_base = 0;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.stride = sizeof(uint32_t);
    output.gpu_addr = reinterpret_cast<uint64_t>(observed.data());
    output.size = observed.size() * sizeof(uint32_t);
    ShaderResourceTable consumer_resources;
    consumer_resources.resources = {output, sampled};
    std::vector<uint32_t> reader{0xd7460004u, 0x04010c10u}; // v4 = global X
    if (layered) {
        reader.insert(reader.end(), {
            0x7e0c0211u,             // v6 = s17: sampled array layer
            0xd7460007u, (0x104u << 18) | (138u << 9) | 17u,
                                    // v7 = (s17 << 10) + v4: SSBO layer base + X
        });
    }
    for (uint32_t y = 0; y < height; ++y) {
        reader.insert(reader.end(), {
            0x7e0a0280u + y,
            layered ? 0xf0000128u : 0xf0000108u, 0x00020804u,
            0xbf8c3f70u,
            0xe0702000u | (y * width * 4u), layered ? 0x80000807u : 0x80000804u,
        });
    }
    reader.push_back(0xbf810000u);
    ComputeItem consumer = item(reader, consumer_resources, address + 1, 2);
    consumer.launch.groups_y = consumer.launch.threads_y = layers;
    check(!producer.spirv.empty() && !consumer.spirv.empty(), "both actual shaders recompile");
    if (failures) return 1;

    auto consume = [&](bool require_retained) {
        std::fill(observed.begin(), observed.end(), 0xccccccccu);
        unsigned calls = 0;
        bool all_ok = true;
        const auto before = prosper::frontend::live_compute_storage_transfer_seeds();
        const auto result = execute_ordered_items(
            {{SubmitOperationKind::Dispatch, 1, 10}, {SubmitOperationKind::Dispatch, 2, 20}},
            {}, {producer, consumer},
            [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
            [&](const std::vector<ComputeItem>& items) {
                ++calls;
                const bool ok = prosper::frontend::execute_live_compute_items(items);
                all_ok &= ok;
                return ok;
            }, 1, 1);
        check(result.compute_executed && calls == 2 && all_ok,
              "producer and sampled-to-SSBO consumer execute in order");
        if (require_retained)
            check(prosper::frontend::live_compute_storage_transfer_seeds() > before,
                  "consumer uses a retained GPU image transfer");
    };
    auto pixels = [&](const char* phase) {
        for (uint32_t i = 0; i < guest.size(); ++i) {
            if (guest[i] != expected[i] || observed[i] != expected[i]) {
                std::fprintf(stderr, "%s %s witness (%u,%u,%u): guest=%08x GPU=%08x expected=%08x\n",
                             argv[1], phase, i % width, (i / width) % height, i / texels,
                             guest[i], observed[i], expected[i]);
                break;
            }
        }
        check(observed == expected, "all GPU-consumed texels match the independent expected image");
        check(guest == expected, "all architectural guest texels match the independent expected image");
    };
    check(prosper::frontend::execute_live_compute_items({producer}),
          "initial full writer establishes its coverage proof");
    // The array consumer still executes a real IMAGE_LOAD-to-SSBO shader, but
    // multi-layer native image borrowing is not this fixture's contract.
    consume(!layered);
    pixels("full");
    if (failures) return 1;

    if (!launch) {
        auto transition = [&](const char* phase, uint32_t salt) {
            check(producer.spirv == original_spirv && producer.user_sgprs == original_sgprs &&
                  producer.launch.groups_x == 2 && producer.launch.groups_y == layers &&
                  producer.launch.groups_z == 1 && producer.launch.local_x == 64,
                  "input transition preserves executable, user SGPRs and launch");
            for (uint32_t i = 0; i < guest.size(); ++i)
                guest[i] = 0x24680000u ^ (salt << 16) ^ i;
            expected = guest;
            for (uint32_t z = 0; z < layers; ++z)
                for (uint32_t x = 0; x < width; ++x) {
                    if (!mask[layered ? z : x]) continue;
                    const uint32_t destination = coordinates ? mask[width + x] : x;
                    check(destination < width, "fixture destination stays inside the image");
                    if (destination >= width) continue;
                    for (uint32_t y = 0; y < height; ++y)
                        expected[z * texels + y * width + destination] = stored;
                }
            consume(false); // An input change may need a correct fresh GPU upload.
            pixels(phase);
            consume(false);
            pixels("unchanged input repeat");
        };
        if (layered) {
            mask[0] = 1; mask[1] = mask[2] = 0;
            transition("Full to layer zero only", 1);
            mask[0] = 0; mask[2] = 1;
            transition("written layer moves to layer two", 2);
        } else {
            std::fill(mask.begin(), mask.begin() + width, 1);
            if (coordinates)
                std::fill(mask.begin() + width / 2, mask.begin() + width, 0);
            else
                mask[0] = 0;
            transition("Full to Partial", 1);
            if (coordinates) {
                for (uint32_t x = 0; x < width / 2; ++x) mask[width + x] = x + width / 2;
                transition("fixed active mask moves coordinates to right half", 2);
            }
        }
        std::fill(mask.begin(), mask.begin() + width, 0);
        transition("Partial to None", 3);
        std::fill(mask.begin(), mask.begin() + width, 1);
        for (uint32_t x = 0; x < width; ++x) mask[width + x] = x;
        transition("None to Full", 4);
        return failures ? 1 : 0;
    }

    producer.launch.groups_x = 1;
    producer.launch.groups_y = 2;
    producer.launch.threads_x = 64;
    producer.launch.threads_y = 2;
    check(producer.launch.groups_x * producer.launch.groups_y * producer.launch.local_x == width,
          "changed launch preserves the total invocation count");
    check(producer.spirv == original_spirv && producer.user_sgprs == original_sgprs,
          "coverage change preserves exact executable bytes and user SGPRs");
    for (uint32_t i = 0; i < texels; ++i) {
        if (i % width >= 64)
            guest[i] = expected[i] = 0x24680000u + i;
    }
    // Reproving partial coverage can repair poison into guest bytes and invalidate
    // the retained image. The first consumer must also accept that correct upload.
    consume(false);
    pixels("changed");
    if (failures) return 1;
    consume(true);
    pixels("stable partial repeat");
    return failures ? 1 : 0;
}
