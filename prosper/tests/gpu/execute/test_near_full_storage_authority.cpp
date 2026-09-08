// #3455: retained allocation identity cannot authorize an unwritten storage-image cutout.
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
constexpr uint32_t width = 64, height = 16, texels = width * height;
constexpr uint32_t stored = 0x13579bdfu;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ComputeItem item(const std::vector<uint32_t>& code, const ShaderResourceTable& resources,
                 uint64_t address, uint32_t index) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    config.local_x = width;
    config.local_y = config.local_z = 1;
    config.tidig_comp_cnt = 0;
    config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Uint32, 1);
    ComputeItem result;
    result.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    result.user_sgprs = config.user_sgprs;
    result.resources = std::make_shared<ShaderResourceTable>(resources);
    result.code_addr = address;
    result.dispatch_index = index;
    result.command_order = index * 10;
    result.launch.threads_x = result.launch.local_x = width;
    result.launch.threads_y = result.launch.threads_z = 1;
    result.launch.local_y = result.launch.local_z = 1;
    result.launch.groups_x = result.launch.groups_y = result.launch.groups_z = 1;
    return result;
}
} // namespace

int main(int argc, char** argv) {
    const bool full_transition = argc == 2 && std::strcmp(argv[1], "full-transition") == 0;
    const bool eviction = argc == 2 && std::strcmp(argv[1], "variant-eviction") == 0;
    const bool replacement = eviction ||
        (argc == 2 && std::strcmp(argv[1], "program-replacement") == 0);
    if (argc != 1 && !full_transition && !replacement) return 2;
    std::vector<uint32_t> guest(texels), observed(texels, 0xccccccccu);
    for (uint32_t i = 0; i < texels; ++i) guest[i] = 0xabc00000u + i;
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5;
    image.sgpr_base = 8;
    image.img_dim = 1;
    image.format = DataFormat::Uint32;
    image.num_components = 1;
    image.width = width;
    image.height = height;
    image.depth = 1;
    image.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
    image.size = texels * sizeof(uint32_t);
    ShaderResourceTable producer_resources;
    producer_resources.resources.push_back(image);

    // Each of 64 invocations writes its column's 16 rows, except (0,0). This is
    // 1023/1024 texels, inside the near-full threshold, with no invocation races.
    std::vector<uint32_t> writer{
        0x7e080300u,             // v4 = local x
        0x7e0002ffu, stored,     // v0 = stored integer
    };
    for (uint32_t y = 1; y < height; ++y) {
        writer.insert(writer.end(), {
            0x7e0a0280u + y,     // v5 = row
            0xf0200108u, 0x00020004u, // IMAGE_STORE x, s[8:15]
        });
    }
    writer.insert(writer.end(), {
        0x7daa0880u,             // v_cmpx_ne_u32 0, v4: preserve (0,0)
        0x7e0a0280u,             // v5 = 0
        0xf0200108u, 0x00020004u,
        0xbf810000u,
    });
    ComputeItem producer = item(writer, producer_resources, 0x34550001u, 1);

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
    output.size = texels * sizeof(uint32_t);
    ShaderResourceTable consumer_resources;
    consumer_resources.resources = {output, sampled};
    std::vector<uint32_t> reader{0x7e080300u}; // v4 = local x
    for (uint32_t y = 0; y < height; ++y) {
        reader.insert(reader.end(), {
            0x7e0a0280u + y,
            0xf0000108u, 0x00020804u, // IMAGE_LOAD x -> v8, s[8:15]
            0xbf8c3f70u,
            0xe0702000u | (y * width * 4u), 0x80000804u,
        });
    }
    reader.push_back(0xbf810000u);
    ComputeItem consumer = item(reader, consumer_resources, 0x34550002u, 2);
    check(!producer.spirv.empty() && !consumer.spirv.empty(), "both actual shaders recompile");
    if (producer.spirv.empty() || consumer.spirv.empty()) return 1;

    auto consume = [&](bool require_retained = true) {
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
              "producer and real sampled-to-SSBO consumer execute in order");
        if (require_retained)
            check(prosper::frontend::live_compute_storage_transfer_seeds() > before,
                  "consumer seeds from the retained GPU image rather than the guest fallback");
    };
    auto pixels = [&](uint32_t cutout, const char* message) {
        check(observed[0] == cutout, message);
        check(std::all_of(observed.begin() + 1, observed.end(),
                          [](uint32_t value) { return value == stored; }),
              "all 1023 written texels reach the independent SSBO consumer");
    };
    if (replacement) {
        constexpr uint32_t full_value = 0x31415926u, changed_cutout = 0x2468ace0u;
        const auto partial = producer;
        auto full_code = writer;
        full_code.erase(std::find(full_code.begin(), full_code.end(), 0x7daa0880u));
        full_code[2] = full_value;
        producer = item(full_code, producer_resources, partial.code_addr, 1);
        const auto full = producer;
        check(producer.spirv != partial.spirv, "replacement changes the executable module");
        check(prosper::frontend::execute_live_compute_items({producer}),
              "full writer proves coverage before any partial writer uses its key");
        consume();
        check(std::all_of(observed.begin(), observed.end(),
                          [](uint32_t value) { return value == full_value; }),
              "identical full program remains correct on a warm proof");
        guest[0] = changed_cutout;
        producer = partial; // Same code address, binding, extent and alias membership.
        // A fresh partial proof repairs guest bytes and invalidates the poisoned GPU
        // image. The real GPU reader must see the repair, including via a fresh upload.
        consume(false);
        std::printf("program replacement witness: guest=%08x GPU=%08x expected=%08x\n",
                    guest[0], observed[0], changed_cutout);
        pixels(changed_cutout, "replacement preserves the newly changed input cutout");
        check(guest[0] == changed_cutout, "replacement preserves the architectural guest cutout");
        consume();
        pixels(changed_cutout, "identical partial program remains correct on a warm proof");
        auto full_pixels = [&](uint32_t expected) {
            check(std::all_of(observed.begin(), observed.end(),
                              [&](uint32_t value) { return value == expected; }) &&
                  std::all_of(guest.begin(), guest.end(),
                              [&](uint32_t value) { return value == expected; }),
                  "returning full variant writes every GPU and guest texel");
        };
        producer = full;
        consume();
        full_pixels(full_value);
        guest[0] = changed_cutout;
        producer = partial;
        consume();
        pixels(changed_cutout, "returning partial variant preserves the changed cutout");
        check(guest[0] == changed_cutout, "returning partial variant preserves the guest cutout");
        if (eviction) {
            // Fill the four-variant budget, touch A, then insert E. B must be evicted,
            // whereas recently used A remains warm. Log assertions distinguish a
            // bounded LRU from both an unbounded map and a one-verdict cache.
            for (uint32_t value : {0x33333333u, 0x44444444u}) {
                full_code[2] = value;
                producer = item(full_code, producer_resources, partial.code_addr, 1);
                consume(false); // A newly proved variant need not retain its image yet.
                full_pixels(value);
            }
            producer = full;
            consume();
            full_pixels(full_value);
            full_code[2] = 0x55555555u;
            producer = item(full_code, producer_resources, partial.code_addr, 1);
            consume(false);
            full_pixels(0x55555555u);
            producer = full;
            consume();
            full_pixels(full_value);
            guest[0] = changed_cutout;
            producer = partial;
            consume(false); // B must safely re-prove after losing its cached identity.
            pixels(changed_cutout, "evicted partial variant re-proves and repairs the cutout");
            check(guest[0] == changed_cutout, "eviction preserves the guest cutout");
        }
        return failures ? 1 : 0;
    }
    // Proving can restore untouched guest texels after a poison run. Warm outside
    // the consumer scope first, then require real GPU transfer on every check.
    for (unsigned warm = 0; warm < 3; ++warm)
        check(prosper::frontend::execute_live_compute_items({producer}),
              "near-full producer proves coverage and retains its allocation");
    consume();
    pixels(0xabc00000u, "warm retained cutout preserves its original input");
    if (full_transition) {
        constexpr uint32_t full_value = 0x31415926u;
        auto full_code = writer;
        full_code.erase(std::find(full_code.begin(), full_code.end(), 0x7daa0880u));
        full_code[2] = full_value;
        // Coverage proofs do not key on the allocation. Prove on independent bytes
        // so the target's guest mirror cannot accidentally supply the next oracle.
        std::vector<uint32_t> proof_guest(texels, 0xeeeeeeeeu);
        auto proof_resources = producer_resources;
        proof_resources.resources[0].gpu_addr = reinterpret_cast<uint64_t>(proof_guest.data());
        auto full = item(full_code, proof_resources, 0x34550003u, 1);
        check(!full.spirv.empty() && prosper::frontend::execute_live_compute_items({full}),
              "independent full writer proves coverage");
        if (failures) return 1;
        full.resources = std::make_shared<ShaderResourceTable>(producer_resources);
        const auto partial = producer;
        producer = full;
        consume();
        check(std::all_of(observed.begin(), observed.end(),
                          [](uint32_t value) { return value == full_value; }),
              "real consumer observes every texel of the proven full writer");
        std::printf("full transition witness: guest=%08x GPU=%08x\n", guest[0], observed[0]);
        check(std::all_of(guest.begin(), guest.end(),
                          [](uint32_t value) { return value == full_value; }),
              "completed full writer publishes the architectural guest mirror");
        producer = partial;
        consume();
        pixels(full_value, "partial writer preserves the preceding full GPU result in its cutout");
        return failures ? 1 : 0;
    }
    guest[0] = 0x2468ace0u;
    guest[1] = 0xfedcba98u;
    consume();
    pixels(0x2468ace0u, "guest cutout mutation invalidates retained image contents");
    consume();
    pixels(0x2468ace0u, "stable warm repeat preserves the repaired cutout");
    if (failures) return 1; // Do not leave the one-shot failure hook armed after an earlier failure.
    guest[0] = 0x98765432u;
    guest[1] = 0x55555555u;
    const auto before_failure = guest;
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    bool failed = false, exported_after_failure = false;
    unsigned failure_calls = 0;
    execute_ordered_items(
        {{SubmitOperationKind::Dispatch, 1, 10}}, {}, {producer},
        [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
        [&](const std::vector<ComputeItem>& items) {
            ++failure_calls;
            const bool ok = prosper::frontend::execute_live_compute_items(items);
            failed = !ok;
            prosper::frontend::LiveComputeImageImport imported;
            exported_after_failure = prosper::frontend::import_live_compute_storage_image(
                sampled, image.size, imported);
            return ok;
        }, 1, 1);
    check(failure_calls == 1 && failed, "post-submit readback failure is reported");
    check(!exported_after_failure, "failed result has no submit-local image export authority");
    check(guest == before_failure, "failed readback preserves authoritative guest bytes");
    consume();
    pixels(0x98765432u, "failure retry reseeds the changed authoritative cutout");
    consume();
    pixels(0x98765432u, "recovered warm result retains the repaired cutout");
    std::printf("near-full storage authority: %d failures\n", failures);
    return failures ? 1 : 0;
}
