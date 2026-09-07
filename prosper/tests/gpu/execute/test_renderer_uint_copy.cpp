// #3407: a renderer-owned RGBA8 UNORM target sampled as UINT must preserve all byte values.
// Exercise the live compute backend, including pin release, typed sampling and guest writeback.
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "shared/live/live_compute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <memory>
#include <cmath>
#include <algorithm>

using namespace prosper::gpu;
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); ++failures; } } while (0)

int main() {
    constexpr uint32_t width = 256;
    std::vector<uint8_t> stale(width * 4, 0), output(width * 4, 0xee);
    auto expected = std::make_shared<std::vector<uint8_t>>(width * 4);
    const uint64_t address = reinterpret_cast<uint64_t>(stale.data());
    const auto& ctx = prosper::test::render_vk_ctx(); // publish before compute adopts a device
    if (!ctx.ok) return 1;

    ResolvedPipelineState state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.color_write_mask = 0; // retain the seeded pattern across the harmless draw
    prosper::test::BackendDraw draw;
    draw.vs.assign(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    draw.fs.assign(std::begin(kTriFragSpv), std::end(kTriFragSpv));
    draw.ps = &state;
    draw.vcount = 3;
    prosper::test::BackendColorTarget target{address, false, true};

    unsigned reads = 0, imports = 0, releases = 0;
    bool transfer_source = true, matching_extent = true;
    set_live_target_query([&](uint64_t addr) { return addr == address; });
    set_live_target_reader([&](uint64_t addr, LiveTargetSnapshot& snapshot) {
        if (addr != address) return false;
        ++reads;
        snapshot.width = width; snapshot.height = 1;
        snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
        snapshot.pixels = expected;
        return true;
    });
    set_live_target_image_importer(
        [&](uint64_t addr, const LiveTargetImageRequest&, LiveTargetImageImport& result) {
            if (addr != address) return false;
            prosper::test::BackendPersistentResourceGuard guard;
            auto* image = prosper::test::find_persistent_color_target(
                address, width, 1, VK_FORMAT_R8G8B8A8_UNORM);
            if (!image || !prosper::test::pin_persistent_color_target(
                    address, width, 1, VK_FORMAT_R8G8B8A8_UNORM)) return false;
            ++imports;
            result.width = matching_extent ? width : width / 2;
            result.height = 1;
            result.format = LiveTargetPixelFormat::Rgba8Unorm;
            result.image = image->image; result.device = ctx.dev;
            result.layout = image->layout;
            result.transfer_src = transfer_source;
            return true;
        },
        [&](uint64_t addr) {
            CHECK(addr == address, "release names the pinned renderer target");
            ++releases;
            prosper::test::BackendPersistentResourceGuard guard;
            prosper::test::unpin_persistent_color_target(
                address, width, 1, VK_FORMAT_R8G8B8A8_UNORM);
        });

    std::vector<uint32_t> indices(width), dummy(4, 0);
    for (uint32_t i = 0; i < width; ++i) indices[i] = i;
    ShaderResourceTable resources;
    for (uint32_t binding = 0; binding < 4; ++binding) {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = binding;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy.data() : indices.data());
        buffer.size = binding ? 16 : width * sizeof(uint32_t);
        resources.resources.push_back(buffer);
    }
    for (uint32_t binding : {4u, 5u}) {
        ShaderResource image{};
        image.cls = binding == 4 ? ResourceClass::Texture : ResourceClass::StorageImage;
        image.binding = binding; image.sgpr_base = binding == 4 ? 0 : 8;
        image.img_dim = 1; image.width = width; image.height = image.depth = 1;
        image.format = DataFormat::Uint8; image.num_components = 4;
        image.gpu_addr = binding == 4 ? address : reinterpret_cast<uint64_t>(output.data());
        image.size = width * 4;
        for (uint32_t c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;
        resources.resources.push_back(image);
    }
    // v4 = input[gid] (x), v5 = 0; image_load RGBA from s[0:7]; image_store to s[8:15].
    const uint32_t code[] = {
        0x7e080300u, 0x7e0a0280u, 0xf0000f08u, 0x00000004u, 0xbf8c3f70u,
        0xf0200f08u, 0x00020004u, 0xbf810000u,
    };
    ComputeItem item;
    item.spirv = recompile_valu(code, std::size(code), 1, 0, &resources);
    CHECK(!item.spirv.empty(), "UINT image copy kernel recompiles");
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.threads_x = width;
    item.launch.local_x = 64; item.launch.groups_x = width / 64;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x3407;

    for (unsigned round = 0; round < 4; ++round) {
        for (uint32_t i = 0; i < width * 4; ++i)
            (*expected)[i] = static_cast<uint8_t>((i / 4) * 37 + (i % 4) * 23 + round * 113 + 11);
        const auto rendered = prosper::test::render_draws_rgba(
            {draw}, width, 1, expected->data(), nullptr, false, &target);
        CHECK(rendered == *expected, "renderer holds the exact nontrivial RGBA8 pattern");
        // First two rounds replace content at the same identity. The remaining arms require the
        // old CPU route: neither a missing transfer usage nor a scaled image is an exact copy.
        transfer_source = round != 2;
        matching_extent = round != 3;
        std::fill(output.begin(), output.end(), 0xee);
        const unsigned reads_before = reads;
        const unsigned imports_before = imports;
        CHECK(prosper::frontend::execute_live_compute_items({item}), "live UINT copy executes");
        CHECK(output == *expected, "all 256 byte values survive UINT sampling and guest writeback");
        CHECK(imports > imports_before && imports == releases, "every acquired image pin is released");
        CHECK(round < 2 ? reads == reads_before : reads == reads_before + 1,
              "eligible copies avoid CPU readback; incompatible imports use the snapshot fallback");
    }
    // Two descriptors can pin the same renderer image. Check folded UINT aliases, then each
    // ordering of a direct UNORM borrow beside the UINT copy. The latter must preserve GENERAL
    // while the direct borrower owns it, then restore the renderer's original layout on release.
    transfer_source = matching_extent = true;
    for (unsigned order = 0; order < 3; ++order) {
        auto paired = resources;
        auto second = paired.resources[4];
        second.binding = 6; second.sgpr_base = 16;
        if (order == 1) paired.resources[4].format = DataFormat::Unorm8;
        if (order == 2) second.format = DataFormat::Unorm8;
        paired.resources.push_back(second);
        const uint32_t paired_code[] = {
            0x7e080300u, 0x7e0a0280u,
            0xf0000f08u, order == 2 ? 0x00040004u : 0x00000004u, 0xbf8c3f70u,
            0xf0000f08u, order == 2 ? 0x00000004u : 0x00040004u, 0xbf8c3f70u,
            0xf0200f08u, 0x00020004u, 0xbf810000u,
        };
        item.spirv = recompile_valu(paired_code, std::size(paired_code), 1, 0, &paired);
        CHECK(!item.spirv.empty(), "paired UNORM/UINT copy kernel recompiles");
        item.resources = std::make_shared<ShaderResourceTable>(paired);
        const unsigned reads_before = reads, imports_before = imports;
        std::fill(output.begin(), output.end(), 0xee);
        CHECK(prosper::frontend::execute_live_compute_items({item}), "paired image imports execute");
        CHECK(output == *expected, "alias folding and mixed view order preserve UINT values");
        CHECK(reads == reads_before && imports == imports_before + 2 && imports == releases,
              "both borrowed descriptors release their pins without a CPU snapshot");
        std::vector<uint8_t> retained;
        std::string error;
        prosper::test::BackendPersistentResourceGuard guard;
        CHECK(prosper::test::readback_persistent_color_target(
                  address, width, 1, VK_FORMAT_R8G8B8A8_UNORM, retained, error) && retained == *expected,
              "the renderer source retains its pixels and usable layout after both imports");
    }
    // Observe the sampled values in FLOAT32 output. Packed-to-packed output would
    // hide a wrong direct RGBA8 borrow by quantizing the values during storage.
    std::vector<float> floats(width * 4), second_floats(width * 4);
    auto packed_resources = resources;
    packed_resources.resources[4].format = DataFormat::Unorm2_10_10_10;
    auto& float_output = packed_resources.resources[5];
    float_output.format = DataFormat::Float32;
    float_output.gpu_addr = reinterpret_cast<uint64_t>(floats.data());
    float_output.size = floats.size() * sizeof(float);
    auto compile = [&](const ShaderResourceTable& table, const uint32_t* words, size_t n) {
        item.spirv = recompile_valu(words, n, 1, 0, &table);
        CHECK(!item.spirv.empty(), "packed UNORM sample into float output recompiles");
        item.resources = std::make_shared<ShaderResourceTable>(table);
    };
    auto check_values = [&](const std::vector<float>& values, bool quantized) {
        for (size_t i = 0; i < values.size(); ++i) {
            const uint32_t byte = (*expected)[i];
            const uint32_t scale = i % 4 == 3 ? 3u : 1023u;
            const uint32_t q = (byte * scale + 127u) / 255u;
            const float oracle = quantized ? float(q) / float(scale) : float(byte) / 255.0f;
            // Independent arithmetic plus an exact CPU-path comparison below. This
            // tolerance allows float normalization rounding, not an 8-bit-value bypass.
            CHECK(std::isfinite(values[i]) && std::abs(values[i] - oracle) <= 0.000001f,
                  "the guest observes the requested UNORM precision in every channel");
            CHECK(q == uint32_t(std::lround((float(byte) / 255.0f) * float(scale))),
                  "integer packing agrees with existing CPU reconstruction for every byte");
        }
    };
    compile(packed_resources, code, std::size(code));
    std::fill(stale.begin(), stale.end(), 0xa5);
    const auto poison = stale;
    for (unsigned round = 0; round < 2; ++round) {
        for (uint32_t i = 0; i < width * 4; ++i)
            (*expected)[i] = uint8_t((i / 4) * 37 + (i % 4) * 23 + round * 113 + 11);
        CHECK(prosper::test::render_draws_rgba(
            {draw}, width, 1, expected->data(), nullptr, false, &target) == *expected,
            "packed conversion source refreshes at the same renderer address");
        transfer_source = matching_extent = true;
        unsigned reads_before = reads, imports_before = imports;
        std::fill(floats.begin(), floats.end(), -99.0f);
        CHECK(prosper::frontend::execute_live_compute_items({item}), "GPU packed conversion executes");
        check_values(floats, true);
        const auto gpu_values = floats;
        CHECK(reads == reads_before && imports > imports_before && imports == releases,
              "packed conversion avoids CPU snapshots and releases every pin");
        CHECK(stale == poison, "renderer authority does not read or overwrite poisoned guest backing");
        for (unsigned refusal = 0; refusal < 2; ++refusal) {
            transfer_source = refusal != 0; matching_extent = refusal != 1;
            reads_before = reads;
            std::fill(floats.begin(), floats.end(), -99.0f);
            CHECK(prosper::frontend::execute_live_compute_items({item}), "packed CPU fallback executes");
            CHECK(floats == gpu_values, "GPU reconstruction exactly matches sampled CPU fallback values");
            CHECK(reads == reads_before + 1 && imports == releases,
                  "incompatible packed import uses one snapshot and balances its pin");
        }
    }
    transfer_source = matching_extent = true;
    for (unsigned order = 0; order < 3; ++order) {
        auto paired = packed_resources;
        auto second = paired.resources[4]; second.binding = 6; second.sgpr_base = 16;
        if (order == 1) paired.resources[4].format = DataFormat::Unorm8;
        if (order == 2) second.format = DataFormat::Unorm8;
        paired.resources.push_back(second);
        auto second_output = float_output;
        second_output.binding = 7; second_output.sgpr_base = 24;
        second_output.gpu_addr = reinterpret_cast<uint64_t>(second_floats.data());
        paired.resources.push_back(second_output);
        const uint32_t both_observed[] = {
            0x7e080300u, 0x7e0a0280u,
            0xf0000f08u, 0x00000004u, 0xbf8c3f70u,
            0xf0200f08u, 0x00020004u,
            0xf0000f08u, 0x00040004u, 0xbf8c3f70u,
            0xf0200f08u, 0x00060004u, 0xbf810000u,
        };
        compile(paired, both_observed, std::size(both_observed));
        const unsigned reads_before = reads, imports_before = imports;
        std::fill(floats.begin(), floats.end(), -99.0f);
        std::fill(second_floats.begin(), second_floats.end(), -99.0f);
        CHECK(prosper::frontend::execute_live_compute_items({item}), "paired packed/direct views execute");
        check_values(floats, order != 1); check_values(second_floats, order != 2);
        CHECK(reads == reads_before && imports == imports_before + 2 && imports == releases,
              "paired packed/direct views avoid snapshots and release both pins");
        std::vector<uint8_t> retained; std::string error;
        prosper::test::BackendPersistentResourceGuard guard;
        CHECK(prosper::test::readback_persistent_color_target(
            address, width, 1, VK_FORMAT_R8G8B8A8_UNORM, retained, error) && retained == *expected,
            "mixed packed/direct imports restore the renderer image and layout");
    }
    compile(packed_resources, code, std::size(code));
    std::fill(floats.begin(), floats.end(), -99.0f);
    const auto untouched = floats;
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    CHECK(!prosper::frontend::execute_live_compute_items({item}), "packed readback failure is reported");
    CHECK(floats == untouched && imports == releases,
          "failure after completed GPU work preserves guest output and releases pins");
    CHECK(prosper::frontend::execute_live_compute_items({item}), "packed conversion retries after failure");
    check_values(floats, true);
    // Match the measured normalized-sampling contract, including filtering AFTER
    // quantization. Alpha42/43 straddles the 2-bit boundary: quarter interpolation
    // yields 1/12, while incorrectly filtering the original bytes yields42.25/255.
    (*expected)[3] = 42; (*expected)[7] = 43;
    CHECK(prosper::test::render_draws_rgba(
        {draw}, width, 1, expected->data(), nullptr, false, &target) == *expected,
        "filtered packed source retains the alpha boundary pattern");
    auto sampled = packed_resources;
    auto& sampled_source = sampled.resources[4];
    sampled_source.sampler_sgpr_base = 8;
    sampled_source.unnormalized = 0;
    sampled_source.min_filter = sampled_source.mag_filter = 1;
    sampled_source.mip_filter = 0;
    sampled_source.addr_uvw[0] = sampled_source.addr_uvw[1] = 2;
    sampled_source.min_lod = sampled_source.max_lod = sampled_source.lod_bias = 0;
    sampled.resources[5].sgpr_base = 16;
    const uint32_t filtered_code[] = {
        0x7e020280u, 0x7e0802ffu, 0x3b400000u, 0x7e0a02ffu, 0x3f000000u,
        0xf09c0f08u, 0x00400c04u, 0xbf8c3f70u,
        0xf0200f08u, 0x00040c00u, 0xbf810000u,
    };
    compile(sampled, filtered_code, std::size(filtered_code));
    const auto reflected = validate_spirv_descriptor_interface(
        item.spirv, &sampled, 0, SpirvShaderStage::Compute, false);
    const auto* binding = find_spirv_descriptor_binding(reflected, 0, 4);
    CHECK(reflected.ok() && binding && binding->kind == SpirvDescriptorKind::CombinedImageSampler &&
        binding->sampled_float && binding->normalized_sampling && !binding->texel_access,
        "packed filter guard exercises normalized float sampling, not texel fetch");
    transfer_source = true;
    unsigned reads_before_filter = reads;
    std::fill(floats.begin(), floats.end(), -99.0f);
    CHECK(prosper::frontend::execute_live_compute_items({item}), "packed normalized filtering executes");
    const auto filtered_gpu = floats;
    for (size_t i = 3; i < floats.size(); i += 4)
        CHECK(std::abs(floats[i] - 1.0f / 12.0f) < 0.000002f,
              "linear filtering observes quantized 2-bit alpha, not original8-bit alpha");
    CHECK(reads == reads_before_filter && imports == releases,
          "normalized packed filtering avoids readback and balances pins");
    transfer_source = false;
    const unsigned reads_before_fallback = reads;
    std::fill(floats.begin(), floats.end(), -99.0f);
    CHECK(prosper::frontend::execute_live_compute_items({item}), "filtered packed CPU fallback executes");
    CHECK(floats == filtered_gpu, "all filtered channels exactly match the CPU reconstruction path");
    CHECK(reads == reads_before_fallback + 1 && imports == releases,
          "filtered CPU fallback takes one snapshot and balances its pin");
    transfer_source = true;
    const unsigned reads_before_oom = reads;
    prosper::frontend::live_compute_fail_next_packed_rtt_setup_for_test(false);
    std::fill(floats.begin(), floats.end(), -99.0f);
    CHECK(prosper::frontend::execute_live_compute_items({item}) && floats == filtered_gpu,
          "recoverable conversion admission failure uses the correct CPU fallback");
    CHECK(reads == reads_before_oom + 1 && imports == releases,
          "recoverable admission failure takes one snapshot and releases its pin");
    std::fill(floats.begin(), floats.end(), -99.0f);
    CHECK(prosper::frontend::execute_live_compute_items({item}) && floats == filtered_gpu &&
          reads == reads_before_oom + 1 && imports == releases,
          "conversion can be used again after recoverable admission failure");
    // Last: simulated device loss permanently latches this process's compute context.
    // Inject the admission result, not an actual failing driver allocation.
    const unsigned imports_before_loss = imports, reads_before_loss = reads;
    const uint64_t submits_before_loss = prosper::frontend::live_compute_queue_submit_attempts();
    std::fill(floats.begin(), floats.end(), -99.0f);
    const auto before_loss = floats;
    prosper::frontend::live_compute_fail_next_packed_rtt_setup_for_test(true);
    CHECK(!prosper::frontend::execute_live_compute_items({item}),
          "device loss in optional conversion admission stops execution");
    CHECK(floats == before_loss && reads == reads_before_loss &&
          imports == imports_before_loss + 1 && imports == releases &&
          prosper::frontend::live_compute_queue_submit_attempts() == submits_before_loss,
          "lost-device admission releases the pin without snapshot, submit or writeback");
    CHECK(!prosper::frontend::execute_live_compute_items({item}) &&
          imports == imports_before_loss + 1 && reads == reads_before_loss &&
          prosper::frontend::live_compute_queue_submit_attempts() == submits_before_loss,
          "latched device loss rejects later work before imports and submission");
    set_live_target_image_importer({}, {});
    set_live_target_reader({});
    set_live_target_query({});
    std::printf("renderer UINT/packed copy: %d failures\n", failures);
    return failures ? 1 : 0;
}
