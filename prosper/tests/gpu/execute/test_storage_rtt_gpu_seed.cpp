// #3407/#3467: standalone storage writes seed from current renderer pixels, not stale guest RAM.
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

using namespace prosper::gpu;
namespace {
constexpr uint32_t width = 64, height = 16, texels = width * height;
constexpr uint32_t guest_bytes = texels * 4;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ShaderResource buffer(void* data, uint32_t bytes, uint32_t binding = 2,
                      uint32_t sgpr = 0, uint32_t stride = 4) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = binding; r.sgpr_base = sgpr;
    r.format = DataFormat::Uint32; r.num_components = 1;
    r.stride = stride; r.size = bytes;
    r.gpu_addr = reinterpret_cast<uint64_t>(data);
    return r;
}
ComputeItem compile(const std::vector<uint32_t>& code, const ShaderResourceTable& table,
                    uint64_t address) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = width; config.local_y = config.local_z = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Unorm8, 4);
    ComputeItem item;
    item.spirv = recompile_compute(code.data(), code.size(), &table, config);
    item.resources = std::make_shared<ShaderResourceTable>(table);
    item.user_sgprs = config.user_sgprs;
    item.code_addr = address;
    item.launch.threads_x = item.launch.local_x = width;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    check(!item.spirv.empty(), "actual fixture shader compiles");
    return item;
}
enum class Miss { None, Usage, Extent, PixelFormat, NativeFormat, Device, Layout };
} // namespace

int main(int argc, char** argv) {
    if (argc > 2) return 2;
    const bool cpu_control = argc == 2 && std::strcmp(argv[1], "--cpu") == 0;
    const bool retention_off = argc == 2 && std::strcmp(argv[1], "--retention-off") == 0;
    const bool retention_test = retention_off ||
        (argc == 2 && std::strcmp(argv[1], "--retention") == 0);
    if (argc == 2 && !cpu_control && !retention_test) return 2;
    if (retention_off && !std::getenv("PROSPER_NO_RENDERER_SEEDED_RESULT_CACHE")) {
        std::fprintf(stderr, "--retention-off requires PROSPER_NO_RENDERER_SEEDED_RESULT_CACHE\n");
        return 2;
    }
    if (cpu_control && !std::getenv("PROSPER_NO_STANDALONE_RTT_SEED")) {
        std::fprintf(stderr, "--cpu requires PROSPER_NO_STANDALONE_RTT_SEED\n");
        return 2;
    }
    const auto& ctx = prosper::test::render_vk_ctx(); // Publish the real renderer device first.
    if (!ctx.ok) return 1;
    // Reserve a real wider backing for the value-equivalent RGBA16 sampled alias.
    // The writable RGBA8 view only owns the first half; the tail is a bounds guard.
    std::vector<uint8_t> guest(texels * 8, 0xc7), expected(guest_bytes);
    // The final retention arm needs a real complete tiled allocation plus an
    // independent tail. Reserve it now so its later resize preserves descriptor addresses.
    if (retention_test) guest.reserve(tiled_surface_bytes(
        width, height, static_cast<uint32_t>(TileMode::Sw64KbRX), 0, 4) + guest_bytes);
    auto source = std::make_shared<std::vector<uint8_t>>(texels * 4);
    std::vector<uint32_t> mask(width), observed(texels * 4, 0xccccccccu);
    std::vector<uint32_t> sampled_values(texels * 4, 0xccccccccu);
    const uint64_t address = reinterpret_cast<uint64_t>(guest.data());
    ResolvedPipelineState state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.color_write_mask = 0;
    prosper::test::BackendDraw draw;
    draw.vs.assign(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    draw.fs.assign(std::begin(kTriFragSpv), std::end(kTriFragSpv));
    draw.ps = &state; draw.vcount = 3;
    prosper::test::BackendColorTarget target{address, false, true};
    unsigned reads = 0, imports = 0, releases = 0, publications = 0;
    bool renderer_visible = true;
    Miss miss = Miss::None;
    VkImageLayout saved_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    set_live_target_query([&](uint64_t addr) { return renderer_visible && addr == address; });
    set_live_target_reader([&](uint64_t addr, LiveTargetSnapshot& result) {
        if (addr != address) return false;
        ++reads;
        result.width = width; result.height = height;
        result.format = LiveTargetPixelFormat::Rgba8Unorm;
        result.pixels = source;
        return true;
    });
    set_live_target_image_importer(
        [&](uint64_t addr, const LiveTargetImageRequest&, LiveTargetImageImport& result) {
            if (addr != address) return false;
            prosper::test::BackendPersistentResourceGuard guard;
            auto* image = prosper::test::find_persistent_color_target(
                address, width, height, VK_FORMAT_R8G8B8A8_UNORM);
            if (!image || !prosper::test::pin_persistent_color_target(
                    address, width, height, VK_FORMAT_R8G8B8A8_UNORM)) return false;
            ++imports;
            result.width = miss == Miss::Extent ? width / 2 : width;
            result.height = height;
            result.format = miss == Miss::PixelFormat ? LiveTargetPixelFormat::Rgba16Float
                                                       : LiveTargetPixelFormat::Rgba8Unorm;
            result.native_format = miss == Miss::NativeFormat ? VK_FORMAT_R8G8B8A8_UINT
                                                               : VK_FORMAT_R8G8B8A8_UNORM;
            result.image = image->image;
            // A foreign identity is metadata only: the guard must reject before any Vulkan use.
            result.device = miss == Miss::Device ? static_cast<void*>(&miss) : ctx.dev;
            result.layout = miss == Miss::Layout ? VK_IMAGE_LAYOUT_UNDEFINED : image->layout;
            result.transfer_src = miss != Miss::Usage;
            result.transfer_dst = false; // Neither companion arm may mirror writes into the source.
            return true;
        },
        [&](uint64_t addr) {
            check(addr == address, "release identifies the real pinned renderer image");
            ++releases;
            prosper::test::BackendPersistentResourceGuard guard;
            check(prosper::test::unpin_persistent_color_target(
                      address, width, height, VK_FORMAT_R8G8B8A8_UNORM),
                  "each imported source pin has one matching release");
        });
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t bytes, const char*) {
        if (addr != address || bytes != guest_bytes) return;
        ++publications;
        if (miss == Miss::None && !cpu_control) {
            prosper::test::BackendPersistentResourceGuard guard;
            const auto* image = prosper::test::find_persistent_color_target(
                address, width, height, VK_FORMAT_R8G8B8A8_UNORM);
            check(image && image->pin_count != 0,
                  "renderer source remains pinned through completed guest publication");
        }
    });
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5; image.sgpr_base = 8;
    image.img_dim = 1; image.width = width; image.height = height; image.depth = 1;
    image.format = DataFormat::Unorm8; image.num_components = 4;
    image.gpu_addr = address; image.size = guest_bytes;
    for (uint32_t c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;
    ShaderResourceTable table;
    table.resources = {buffer(mask.data(), mask.size() * 4), image};
    auto writer_words = [&](bool sampled_sibling) {
        std::vector<uint32_t> code{
            0x7e080300u, 0xe0302000u, 0x80000804u, 0xbf8c3f70u,
            0x7daa1080u, // EXEC selects runtime mask columns; code/launch stay identical.
            0x7e0002ffu, 0x3f800000u, 0x7e020280u,
            0x7e0402ffu, 0x3f800000u, 0x7e0602ffu, 0x3f800000u,
        };
        for (uint32_t y = 0; y < height; ++y) {
            code.push_back(0x7e0a0280u + y);
            if (sampled_sibling) {
                code.insert(code.end(), {0xf0000f08u, 0x00040c04u, 0xbf8c3f70u,
                                         0xbe9803ffu, y * width * 16u}); // s24 = output row offset
                for (uint32_t c = 0; c < 4; ++c)
                    code.insert(code.end(), {0xe0702000u | (c * 4u),
                                            0x18010004u | ((12u + c) << 8)});
            }
            code.insert(code.end(), {0xf0200f08u, 0x00020004u});
        }
        code.push_back(0xbf810000u);
        return code;
    };
    const auto producer = compile(writer_words(false), table, 0x340746701u);
    const auto report = validate_spirv_descriptor_interface(
        producer.spirv, &table, 0, SpirvShaderStage::Compute, false);
    const auto* storage = find_spirv_descriptor_binding(report, 0, 5);
    check(report.ok() && storage && storage->writable && !storage->readable &&
              storage->image_numeric_class == SpirvImageNumericClass::Float,
          "standalone producer has one native float storage image and no sampled sibling");
    check(std::count_if(report.descriptors.begin(), report.descriptors.end(), [](const auto& d) {
              return d.kind == SpirvDescriptorKind::StorageImage ||
                     d.kind == SpirvDescriptorKind::CombinedImageSampler;
          }) == 1, "primary regression really has a standalone image binding");
    auto sampled = image; sampled.cls = ResourceClass::Texture; sampled.format = DataFormat::Uint8;
    ShaderResourceTable reader_table;
    reader_table.resources = {buffer(observed.data(), observed.size() * 4, 2, 0, 16), sampled};
    std::vector<uint32_t> reader{0x7e080300u};
    for (uint32_t y = 0; y < height; ++y) {
        // A full cache-eligible image exceeds MUBUF's 12-bit immediate range.
        // Keep only the channel in the immediate and the full row offset in s24.
        reader.insert(reader.end(), {0x7e0a0280u + y, 0xf0000f08u, 0x00020804u, 0xbf8c3f70u,
                                     0xbe9803ffu, y * width * 16u});
        for (uint32_t c = 0; c < 4; ++c)
            reader.insert(reader.end(), {0xe0702000u | (c * 4u),
                                        0x18000004u | ((8u + c) << 8)});
    }
    reader.push_back(0xbf810000u);
    const auto consumer = compile(reader, reader_table, 0x340746702u);
    auto refresh = [&](unsigned round) {
        for (size_t i = 0; i < source->size(); ++i) (*source)[i] = uint8_t(i * 37 + round * 113 + 11);
        check(prosper::test::render_draws_rgba(
                  {draw}, width, height, source->data(), nullptr, false, &target) == *source,
              "real renderer image receives the new full-channel source pattern");
        std::fill(guest.begin(), guest.end(), 0xc7);
        check(!std::equal(source->begin(), source->end(), guest.begin()),
              "architectural backing is deliberately stale");
        prosper::test::BackendPersistentResourceGuard guard;
        const auto* retained = prosper::test::find_persistent_color_target(
            address, width, height, VK_FORMAT_R8G8B8A8_UNORM);
        check(retained && retained->pin_count == 0, "source starts without borrowed pins");
        if (retained) saved_layout = retained->layout;
    };
    auto source_preserved = [&] {
        prosper::test::BackendPersistentResourceGuard guard;
        const auto* retained = prosper::test::find_persistent_color_target(
            address, width, height, VK_FORMAT_R8G8B8A8_UNORM);
        check(retained && retained->pin_count == 0 && retained->layout == saved_layout,
              "source ownership and incoming layout are restored");
        std::vector<uint8_t> pixels; std::string error;
        check(prosper::test::readback_persistent_color_target(
                  address, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels, error) && pixels == *source,
              "private storage writes preserve every renderer source byte and its usable layout");
    };
    auto expected_producer = [&] {
        expected = *source;
        constexpr uint8_t written[4]{255, 0, 255, 255};
        for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x)
            if (mask[x]) for (uint32_t c = 0; c < 4; ++c)
                expected[(y * width + x) * 4 + c] = written[c];
    };
    auto observe_guest = [&] {
        expected_producer();
        check(std::equal(expected.begin(), expected.end(), guest.begin()),
              "every written and untouched guest channel uses the current renderer seed");
        check(std::all_of(guest.begin() + guest_bytes, guest.end(), [](uint8_t byte) { return byte == 0xc7; }),
              "storage writeback never crosses its guest view into the wider alias tail");
        std::fill(observed.begin(), observed.end(), 0xccccccccu);
        renderer_visible = false; // The independent observer must consume guest writeback.
        const bool executed = prosper::frontend::execute_live_compute_items({consumer});
        renderer_visible = true;
        check(executed, "independent UINT consumer executes against guest bytes");
        check(std::equal(observed.begin(), observed.end(), expected.begin()),
              "every independently GPU-consumed channel matches architectural writeback");
    };
    if (failures) return 1;
    for (unsigned round = 0; round < 4; ++round) {
        refresh(round);
        for (uint32_t x = 0; x < width; ++x)
            mask[x] = round == 0 ? 0 : round == 2 ? 1 : uint32_t(x % 3 != 0);
        const unsigned before_reads = reads, before_imports = imports, before_publications = publications;
        check(prosper::frontend::execute_live_compute_items({producer}), "standalone None/Partial/Full producer executes");
        check(reads == before_reads + unsigned(cpu_control),
              "eligible standalone seed avoids CPU snapshots; forced control takes exactly one");
        check(cpu_control ? imports == before_imports : imports == before_imports + 1,
              "standalone path acquires exactly its own source pin");
        check(imports == releases && publications == before_publications + 1,
              "completed standalone dispatch releases its pin and publishes guest output");
        observe_guest(); source_preserved();
    }
    if (!cpu_control) {
        for (Miss refusal : {Miss::Usage, Miss::Extent, Miss::PixelFormat, Miss::NativeFormat,
                             Miss::Device, Miss::Layout}) {
            refresh(7 + unsigned(refusal)); miss = refusal;
            const unsigned before_reads = reads, before_imports = imports;
            check(prosper::frontend::execute_live_compute_items({producer}), "incompatible standalone source takes CPU fallback");
            check(reads == before_reads + 1 && imports == before_imports + 1 && imports == releases,
                  "refused import releases its pin and performs one current snapshot");
            observe_guest(); source_preserved();
        }
        miss = Miss::None;
        refresh(19);
        auto short_backing = producer;
        short_backing.resources = std::make_shared<ShaderResourceTable>(*producer.resources);
        // Nonzero metadata passes reflection, then the real image materializer
        // refuses the short destination after importing its exact renderer seed.
        short_backing.resources->resources[1].size = 1;
        const auto before_setup_guest = guest;
        const unsigned before_setup_reads = reads, before_setup_imports = imports;
        const uint64_t before_setup_submits = prosper::frontend::live_compute_queue_submit_attempts();
        check(!prosper::frontend::execute_live_compute_items({short_backing}),
              "short storage backing declines after seed admission");
        check(guest == before_setup_guest && reads == before_setup_reads &&
                  imports == before_setup_imports + 1 && imports == releases &&
                  prosper::frontend::live_compute_queue_submit_attempts() == before_setup_submits,
              "setup failure releases its admitted source pin without snapshot, submit or guest write");
        source_preserved();
        refresh(20);
        const auto before_guest = guest;
        const unsigned before_reads = reads, before_imports = imports;
        prosper::frontend::live_compute_fail_next_storage_readback_for_test();
        check(!prosper::frontend::execute_live_compute_items({producer}), "completed-GPU readback failure is reported");
        check(guest == before_guest && reads == before_reads && imports == before_imports + 1 && imports == releases,
              "readback failure keeps guest bytes untouched and releases the standalone pin");
        source_preserved();
        check(prosper::frontend::execute_live_compute_items({producer}), "standalone seed retries after readback failure");
        observe_guest(); source_preserved();
    }
    // Both relative binding orders matter: a sampled borrower can already own GENERAL,
    // or can be materialized after the standalone seed has acquired its own pin.
    if (!cpu_control) for (unsigned arm = 0; arm < 3; ++arm) {
        const uint32_t sampled_binding = arm == 1 ? 6u : 4u;
        const bool wider_alias = arm == 2;
        refresh(30 + arm);
        std::fill(mask.begin(), mask.end(), 1);
        auto paired = table;
        auto sibling = image; sibling.cls = ResourceClass::Texture;
        sibling.binding = sampled_binding; sibling.sgpr_base = 16;
        if (wider_alias) {
            // byte*257/65535 == byte/255, so this existing direct sampled borrow
            // uses the same source image. Its guest format differs from storage:
            // standalone copy must preserve this prior borrower's GENERAL layout.
            sibling.format = DataFormat::Unorm16;
            sibling.size = guest.size();
        }
        paired.resources.push_back(sibling);
        paired.resources.push_back(buffer(sampled_values.data(), sampled_values.size() * 4, 3, 4, 16));
        const auto paired_item = compile(writer_words(true), paired, 0x340746710u + arm);
        std::fill(sampled_values.begin(), sampled_values.end(), 0xccccccccu);
        const unsigned before_reads = reads, before_imports = imports;
        check(prosper::frontend::execute_live_compute_items({paired_item}), "storage seed and sampled borrower execute in either binding order");
        check(reads == before_reads && imports == before_imports + (arm == 0 ? 1u : 2u) && imports == releases,
              "sampled sibling order avoids snapshots and balances every source pin");
        bool sampled_ok = true;
        for (size_t i = 0; i < sampled_values.size(); ++i) {
            const float value = std::bit_cast<float>(sampled_values[i]);
            sampled_ok &= std::isfinite(value) && std::abs(value - (*source)[i] / 255.0f) <= 0.000001f;
        }
        check(sampled_ok, "the sampled sibling observes every original renderer channel");
        observe_guest(); source_preserved();
    }
    if (retention_test) {
        // UINT aliases intentionally use a different sampled format in observe_guest().
        // This observer instead asks for the exact normalized format of the retained
        // storage result, and reads every channel independently on the GPU.
        auto normalized_table = reader_table;
        normalized_table.resources.back().format = DataFormat::Unorm8;
        auto normalized_consumer = compile(reader, normalized_table, 0x340746720u);
        normalized_consumer.dispatch_index = 102;
        normalized_consumer.command_order = 20;
        auto ordered_producer = producer;
        ordered_producer.dispatch_index = 101;
        ordered_producer.command_order = 10;
        const auto normalized_report = validate_spirv_descriptor_interface(
            normalized_consumer.spirv, &normalized_table, 0, SpirvShaderStage::Compute, false);
        const auto* normalized_binding = find_spirv_descriptor_binding(normalized_report, 0, 5);
        check(normalized_report.ok() && normalized_binding &&
                  normalized_binding->image_numeric_class == SpirvImageNumericClass::Float,
              "retention consumer reflects the producer's exact normalized sampled format");
        check(prosper::frontend::compute_image_cache_default_eligible(
                  guest_bytes, prosper::frontend::ComputeImageCacheClass::storage) &&
                  prosper::frontend::compute_image_cache_default_eligible(
                      guest_bytes, prosper::frontend::ComputeImageCacheClass::sampled),
              "retention fixture reaches both real image-cache size thresholds");
        // Each round replaces the renderer source while preserving the same addresses,
        // shader identities and launch. A cached completed result must never replace
        // that fresh source during the next storage setup.
        for (unsigned round = 0; round < 7; ++round) {
            refresh(60 + round);
            for (uint32_t x = 0; x < width; ++x)
                mask[x] = round == 0 ? 0 : round == 2 ? 1 : uint32_t(x % 3 != 0);
            expected_producer();
            const bool guest_mutation = round == 4;
            const bool readback_failure = round == 5;
            const auto before_guest = guest;
            const auto before_reads = reads, before_imports = imports;
            const auto before_publications = publications;
            const auto before_transfers = prosper::frontend::live_compute_storage_transfer_seeds();
            unsigned producers = 0, consumers = 0;
            std::fill(observed.begin(), observed.end(), 0xccccccccu);
            if (readback_failure) prosper::frontend::live_compute_fail_next_storage_readback_for_test();
            const auto result = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, ordered_producer.dispatch_index, 10},
                 {SubmitOperationKind::Dispatch, normalized_consumer.dispatch_index, 20}},
                {}, {ordered_producer, normalized_consumer},
                [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
                [&](const std::vector<ComputeItem>& items) {
                    for (const auto& item : items) {
                        if (item.dispatch_index == ordered_producer.dispatch_index) {
                            ++producers;
                            const bool ok = prosper::frontend::execute_live_compute_items({item});
                            check(ok != readback_failure, "ordered producer reports actual success or injected readback failure");
                            if (readback_failure) {
                                check(guest == before_guest && publications == before_publications,
                                      "failed completed result changes neither guest bytes nor publication");
                                expected.assign(before_guest.begin(), before_guest.begin() + guest_bytes);
                            } else {
                                check(std::equal(expected.begin(), expected.end(), guest.begin()),
                                      "ordered producer publishes every expected renderer-seeded channel");
                                if (guest_mutation) {
                                    // A real intervening write invalidates the submit journal.
                                    constexpr size_t changed = 13 * width * 4 + 7;
                                    guest[changed] ^= 0x5a;
                                    expected[changed] ^= 0x5a;
                                    notify_guest_gpu_write(address + changed, 1);
                                }
                            }
                            renderer_visible = false;
                            // Deliberately continue after this *expected* injected failure to
                            // inspect whether an invalid result was wrongly published. The
                            // independently asserted producer return remains the failure oracle.
                        } else {
                            ++consumers;
                            check(prosper::frontend::execute_live_compute_items({item}),
                                  "ordered normalized GPU consumer executes");
                        }
                    }
                    return true;
                }, 1, 1);
            renderer_visible = true;
            check(result.compute_executed && producers == 1 && consumers == 1,
                  "ordered regression executes exactly one producer then one consumer");
            check(reads == before_reads && imports == before_imports + 1 && imports == releases,
                  "retained output never bypasses fresh renderer seed or leaks its source pin");
            bool pixels_ok = true;
            for (size_t i = 0; i < observed.size(); ++i) {
                const float value = std::bit_cast<float>(observed[i]);
                pixels_ok &= std::isfinite(value) &&
                    std::abs(value - expected[i] / 255.0f) <= 0.000001f;
            }
            check(pixels_ok && std::equal(expected.begin(), expected.end(), guest.begin()),
                  "retained transfer or fallback preserves all GPU-observed and guest channels");
            const auto transfers = prosper::frontend::live_compute_storage_transfer_seeds() - before_transfers;
            const bool should_transfer = !retention_off && !guest_mutation && !readback_failure;
            check(transfers == uint64_t(should_transfer),
                  "only valid completed results take exactly one storage-to-sampled GPU transfer");
            check(std::all_of(guest.begin() + guest_bytes, guest.end(),
                              [](uint8_t byte) { return byte == 0xc7; }),
                  "ordered retained result preserves the independent guest tail");
            source_preserved();
        }
        // These descriptors overlap guest bytes but differ in extent, so they are
        // independent writable owners rather than one folded alias. A's private
        // completed image cannot represent the final A+B guest composite.
        // Hosted aliases keep their advertised range disjoint; writeback still
        // targets A through host_data. Distinct colors also invalidate a prior sampled result.
        for (bool hosted_alias : {false, true}) {
            std::fprintf(stderr, "[rtt-retention] owner-overlap hosted=%u\n", unsigned(hosted_alias));
            refresh(90 + unsigned(hosted_alias));
            std::vector<uint8_t> advertised(guest_bytes, 0x69);
            auto overlap_table = table;
            auto overlap_b = image;
            overlap_b.binding = 6; overlap_b.sgpr_base = 16;
            overlap_b.height = height / 2;
            overlap_b.size = guest_bytes / 2;
            if (hosted_alias) {
                overlap_b.gpu_addr = reinterpret_cast<uint64_t>(advertised.data());
                overlap_b.host_data = guest.data();
                overlap_b.host_data_size = overlap_b.size;
                check(overlap_b.gpu_addr + overlap_b.size <= address ||
                          address + guest_bytes <= overlap_b.gpu_addr,
                      "hosted output advertises a real disjoint GPU range");
            }
            overlap_table.resources.push_back(overlap_b);
            std::vector<uint32_t> overlap_words{
                0x7e080300u, 0x7e0002ffu, 0x3f800000u, 0x7e020280u,
                0x7e0402ffu, 0x3f800000u, 0x7e0602ffu, 0x3f800000u};
            for (uint32_t y = 0; y < height; ++y)
                overlap_words.insert(overlap_words.end(), {0x7e0a0280u + y, 0xf0200f08u, 0x00020004u});
            overlap_words.push_back(0x7e000280u);
            if (hosted_alias) overlap_words.push_back(0x7e020280u);
            else overlap_words.insert(overlap_words.end(), {0x7e0202ffu, 0x3f800000u});
            for (uint32_t y = 0; y < height / 2; ++y)
                overlap_words.insert(overlap_words.end(), {0x7e0a0280u + y, 0xf0200f08u, 0x00040004u});
            overlap_words.push_back(0xbf810000u);
            auto overlap_producer = compile(overlap_words, overlap_table, 0x340746721u);
            overlap_producer.dispatch_index = ordered_producer.dispatch_index;
            overlap_producer.command_order = 10;
            for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
                const uint8_t color[4]{uint8_t(y < height / 2 ? 0 : 255),
                                       uint8_t(y < height / 2 && !hosted_alias ? 255 : 0), 255, 255};
                for (uint32_t c = 0; c < 4; ++c) expected[(y * width + x) * 4 + c] = color[c];
            }
            std::fill(observed.begin(), observed.end(), 0xccccccccu);
            const auto before_overlap_transfers = prosper::frontend::live_compute_storage_transfer_seeds();
            unsigned overlap_producers = 0, overlap_consumers = 0;
            const auto overlap_result = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, overlap_producer.dispatch_index, 10},
                 {SubmitOperationKind::Dispatch, normalized_consumer.dispatch_index, 20}},
                {}, {overlap_producer, normalized_consumer},
                [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
                [&](const std::vector<ComputeItem>& items) {
                    for (const auto& item : items) {
                        const bool is_producer = item.dispatch_index == overlap_producer.dispatch_index;
                        check(prosper::frontend::execute_live_compute_items({item}),
                              "independent overlapping output and sampled consumer execute");
                        if (is_producer) {
                            ++overlap_producers;
                            check(std::equal(expected.begin(), expected.end(), guest.begin()),
                                  "ordinary overlapping writeback publishes B overlap plus A remainder");
                            renderer_visible = false;
                        } else ++overlap_consumers;
                    }
                    return true;
                }, 1, 1);
            renderer_visible = true;
            bool overlap_pixels_ok = true;
            for (size_t i = 0; i < observed.size(); ++i) {
                const float value = std::bit_cast<float>(observed[i]);
                overlap_pixels_ok &= std::isfinite(value) &&
                    std::abs(value - expected[i] / 255.0f) <= 0.000001f;
            }
            check(overlap_result.compute_executed && overlap_producers == 1 && overlap_consumers == 1 &&
                      overlap_pixels_ok && std::equal(expected.begin(), expected.end(), guest.begin()),
                  "GPU sampled observer sees final overlapping composite, never stale private A");
            check(prosper::frontend::live_compute_storage_transfer_seeds() == before_overlap_transfers,
                  "independent overlapping storage owner excludes completed-result promotion");
            check(std::all_of(guest.begin() + guest_bytes, guest.end(),
                              [](uint8_t byte) { return byte == 0xc7; }),
                  "overlapping storage views preserve the independent guest tail");
            check(std::all_of(advertised.begin(), advertised.end(),
                              [](uint8_t byte) { return byte == 0x69; }),
                  "hosted write targets the effective backing, never its advertised allocation");
            source_preserved();
        }

        // The producer's own metadata reset is another write to guest memory.
        // With DCC stored inside its base allocation, the private rendered image
        // represents the pre-reset pixels and must not become a sampled source.
        for (bool hosted_metadata : {false, true}) {
            std::fprintf(stderr, "[rtt-retention] metadata-overlap hosted=%u\n", unsigned(hosted_metadata));
            refresh(93 + unsigned(hosted_metadata));
            for (uint32_t x = 0; x < width; ++x)
                mask[x] = !hosted_metadata || x % 3 != 0;
            expected_producer();
            auto dcc_table = table;
            auto& dcc_image = dcc_table.resources.back();
            dcc_image.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
            const size_t dcc_tiled_bytes = tiled_surface_bytes(
                width, height, dcc_image.tile_mode, 0, 4);
            dcc_image.size = static_cast<uint32_t>(dcc_tiled_bytes);
            dcc_image.compression_enabled = true;
            dcc_image.write_compress_enabled = true;
            dcc_image.meta_pipe_aligned = true;
            std::vector<uint8_t> advertised_metadata(dcc_tiled_bytes, 0x69);
            dcc_image.metadata_addr = hosted_metadata
                ? reinterpret_cast<uint64_t>(advertised_metadata.data()) : address;
            const uint64_t metadata_bytes = gpu_capture_dcc_metadata_footprint(dcc_image);
            const bool bounded_metadata = metadata_bytes && metadata_bytes <= dcc_tiled_bytes;
            check(bounded_metadata, "production DCC footprint fits the real aliased base allocation");
            if (bounded_metadata) {
                dcc_image.dcc_metadata_size = metadata_bytes;
                if (hosted_metadata) {
                    dcc_image.dcc_metadata_host_data = guest.data();
                    dcc_image.dcc_metadata_host_data_size = metadata_bytes;
                    check(dcc_image.metadata_addr + metadata_bytes <= address ||
                              address + dcc_tiled_bytes <= dcc_image.metadata_addr,
                          "hosted DCC advertises a real disjoint metadata range");
                }
                guest.resize(dcc_tiled_bytes + guest_bytes, 0xc7);
                check(reinterpret_cast<uint64_t>(guest.data()) == address,
                      "DCC fixture resizing retains all resource and renderer addresses");
                std::vector<uint8_t> expected_tiled(guest.size(), 0xc7);
                tile_surface(expected_tiled.data(), expected.data(), width, height,
                             dcc_image.tile_mode, 0, 4);
                std::fill_n(expected_tiled.begin(), static_cast<size_t>(metadata_bytes), uint8_t{0xff});
                std::vector<uint8_t> expected_reset(guest_bytes);
                detile_surface(expected_reset.data(), expected_tiled.data(), width, height,
                               dcc_image.tile_mode, 0, 4);
                check(expected_reset != expected,
                      "own metadata reset changes visible texels and distinguishes stale private output");
                auto dcc_producer = compile(writer_words(false), dcc_table, 0x340746722u);
                dcc_producer.dispatch_index = ordered_producer.dispatch_index;
                dcc_producer.command_order = 10;
                auto dcc_reader_table = normalized_table;
                dcc_reader_table.resources.back().tile_mode = dcc_image.tile_mode;
                dcc_reader_table.resources.back().size = dcc_image.size;
                auto dcc_consumer = compile(reader, dcc_reader_table, 0x340746723u);
                dcc_consumer.dispatch_index = normalized_consumer.dispatch_index;
                dcc_consumer.command_order = 20;
                std::fill(observed.begin(), observed.end(), 0xccccccccu);
                const auto before_dcc_transfers = prosper::frontend::live_compute_storage_transfer_seeds();
                const auto before_dcc_reads = reads, before_dcc_imports = imports;
                unsigned dcc_producers = 0, dcc_consumers = 0;
                const auto dcc_result = execute_ordered_items(
                    {{SubmitOperationKind::Dispatch, dcc_producer.dispatch_index, 10},
                     {SubmitOperationKind::Dispatch, dcc_consumer.dispatch_index, 20}},
                    {}, {dcc_producer, dcc_consumer},
                    [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
                    [&](const std::vector<ComputeItem>& items) {
                        for (const auto& item : items) {
                            check(prosper::frontend::execute_live_compute_items({item}),
                                  "own-DCC-overlap producer and sampled observer execute");
                            if (item.dispatch_index == dcc_producer.dispatch_index) {
                                ++dcc_producers;
                                check(guest == expected_tiled,
                                      "real image writeback then metadata reset preserves every expected tiled byte");
                                renderer_visible = false;
                            } else ++dcc_consumers;
                        }
                        return true;
                    }, 1, 1);
                renderer_visible = true;
                bool reset_pixels_ok = true;
                for (size_t i = 0; i < observed.size(); ++i) {
                    const float value = std::bit_cast<float>(observed[i]);
                    reset_pixels_ok &= std::isfinite(value) &&
                        std::abs(value - expected_reset[i] / 255.0f) <= 0.000001f;
                }
                check(dcc_result.compute_executed && dcc_producers == 1 && dcc_consumers == 1 &&
                          reset_pixels_ok && guest == expected_tiled,
                      "normalized GPU observer sees actual metadata-reset pixels and the complete guest tail");
                check(reads == before_dcc_reads && imports == before_dcc_imports + 1 && imports == releases,
                      "DCC overlap case takes exact renderer seeding and releases its real source pin");
                check(prosper::frontend::live_compute_storage_transfer_seeds() == before_dcc_transfers,
                      "own overlapping DCC metadata reset excludes completed-result retention");
                check(std::all_of(advertised_metadata.begin(), advertised_metadata.end(),
                                  [](uint8_t byte) { return byte == 0x69; }),
                      "hosted reset targets the effective DCC backing, never its advertised allocation");
                source_preserved();
            }
        }
    }
    set_guest_gpu_write_observer({});
    set_live_target_image_importer({}, {});
    set_live_target_reader({}); set_live_target_query({});
    std::printf("storage RTT GPU seed: %d failures\n", failures);
    return failures ? 1 : 0;
}
