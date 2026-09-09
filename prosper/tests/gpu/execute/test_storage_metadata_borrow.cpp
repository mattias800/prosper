// #3482: a separately changed metadata plane must gate the public retained-image borrow.
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t W = 64, H = 16, N = W * H;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
ComputeItem compile(const std::vector<uint32_t>& code, const ShaderResourceTable& resources,
                    uint64_t address, uint32_t index) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = W; config.local_y = config.local_z = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Float16, 4) |
        native_storage_format_support_bit(DataFormat::Uint16, 1) |
        native_storage_format_support_bit(DataFormat::Uint32, 1);
    ComputeItem result;
    result.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    result.resources = std::make_shared<ShaderResourceTable>(resources);
    result.user_sgprs = config.user_sgprs;
    result.code_addr = address; result.dispatch_index = index; result.command_order = index * 20;
    result.launch.threads_x = result.launch.local_x = W;
    result.launch.threads_y = result.launch.threads_z = 1;
    result.launch.local_y = result.launch.local_z = 1;
    result.launch.groups_x = result.launch.groups_y = result.launch.groups_z = 1;
    check(!result.spirv.empty(), "guest shader compiles");
    return result;
}
struct Fixture {
    bool numeric_alias, transfer_test;
    ShaderResource image{};
    std::vector<uint8_t> pixels, metadata, hosted;
    std::vector<uint32_t> output = std::vector<uint32_t>(N * 4 + 16, 0xdeadbeefu);
    CompressionMetadataKind kind = CompressionMetadataKind::Dcc;
    size_t metadata_bytes = 0, pixel_bytes = 0;
    explicit Fixture(bool alias, bool transfer = false) : numeric_alias(alias), transfer_test(transfer) {
        image.cls = ResourceClass::StorageImage; image.binding = 5; image.sgpr_base = 8;
        image.img_dim = 1; image.width = W; image.height = H; image.depth = 1;
        image.format = transfer ? DataFormat::Uint32 : alias ? DataFormat::Uint16 : DataFormat::Float16;
        image.num_components = alias ? 1 : 4;
        for (unsigned c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;
        image.tile_mode = uint32_t(TileMode::Sw64KbRX);
        image.compression_enabled = image.write_compress_enabled = image.meta_pipe_aligned = true;
        image.alpha_is_on_msb = true;
        pixel_bytes = tiled_surface_bytes(W, H, image.tile_mode, 0, transfer_test ? 4 : numeric_alias ? 2 : 8);
        pixels.resize(pixel_bytes + 64, 0xc7);
        image.gpu_addr = reinterpret_cast<uint64_t>(pixels.data()); image.size = pixel_bytes;
        // Footprint calculation needs a nonzero address, not the eventual allocation contents.
        image.metadata_addr = image.gpu_addr;
        metadata_bytes = gpu_capture_dcc_metadata_footprint(image);
        check(metadata_bytes && metadata_bytes <= 65536 && metadata_bytes % (W * 4) == 0,
              "fixture has a bounded complete DCC plane writable in whole rows");
        metadata.resize(metadata_bytes + 64, 0x97);
        hosted.resize(metadata_bytes + 64, 0x98);
        std::fill_n(metadata.begin(), metadata_bytes, 0xff);
        std::fill_n(hosted.begin(), metadata_bytes, 0xff);
        image.metadata_addr = reinterpret_cast<uint64_t>(metadata.data());
        image.dcc_metadata_size = metadata_bytes;
    }
    ShaderResource sampled() const {
        auto r = image; r.cls = ResourceClass::Texture; r.format = transfer_test ? DataFormat::Float32 : DataFormat::Float16; return r;
    }
    ComputeItem writer() {
        ShaderResourceTable table; table.resources = {image};
        const uint32_t one = numeric_alias && !transfer_test ? 0x3c00u : 0x3f800000u;
        std::vector<uint32_t> code{0x7e080300u, 0x7e0002ffu, one, 0x7e020280u,
                                  0x7e0402ffu, one, 0x7e0602ffu, one};
        for (uint32_t y = 0; y < H; ++y)
            code.insert(code.end(), {0x7e0a0280u + y, numeric_alias ? 0xf0200108u : 0xf0200f08u, 0x00020004u});
        code.push_back(0xbf810000u);
        auto result = compile(code, table, numeric_alias ? 0x34820011u : 0x34820001u, 1);
        const auto reflection = validate_spirv_descriptor_interface(
            result.spirv, &table, 0, SpirvShaderStage::Compute, false);
        const auto* d = find_spirv_descriptor_binding(reflection, 0, 5);
        check(reflection.ok() && reflection.storage_image_writes_complete && d && d->writable &&
                  !d->readable && (numeric_alias ? d->image_numeric_class == SpirvImageNumericClass::Uint &&
                                      d->storage_image_format == (transfer_test ? kSpirvImageFormatR32ui : kSpirvImageFormatR16ui)
                                                : d->storage_float && d->storage_image_format == 0),
              "producer is the intended native float or integer storage image");
        return result;
    }
    ComputeItem metadata_writer() {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer; buffer.binding = 2; buffer.sgpr_base = 0;
        buffer.format = DataFormat::Uint32; buffer.num_components = 1; buffer.stride = 4;
        buffer.gpu_addr = image.metadata_addr; buffer.size = metadata_bytes;
        ShaderResourceTable table; table.resources = {buffer};
        std::vector<uint32_t> code{0x7e1002ffu, 0x40404040u};
        for (uint32_t row = 0; row < metadata_bytes / (W * 4); ++row)
            code.insert(code.end(), {0xbe9803ffu, row * W * 4, 0xe0702000u, 0x18000800u});
        code.push_back(0xbf810000u);
        return compile(code, table, 0x34820002u, 2);
    }
    ComputeItem reader() {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer; buffer.binding = 2; buffer.sgpr_base = 0;
        buffer.format = DataFormat::Uint32; buffer.num_components = 1; buffer.stride = 16;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(output.data()); buffer.size = N * 16;
        ShaderResourceTable table; table.resources = {buffer, sampled()};
        std::vector<uint32_t> code{0x7e080300u};
        for (uint32_t y = 0; y < H; ++y) {
            code.insert(code.end(), {0x7e0a0280u + y, 0xf0000f08u, 0x00020804u,
                                    0xbf8c3f70u, 0xbe9803ffu, y * W * 16});
            for (uint32_t c = 0; c < 4; ++c)
                code.insert(code.end(), {0xe0702000u | (c * 4u), 0x18000004u | ((8u + c) << 8)});
        }
        code.push_back(0xbf810000u);
        return compile(code, table, 0x34820003u, 3);
    }
    prosper::frontend::LiveComputeImageImport borrow(const ShaderResource& r, bool expected,
                                                     const char* message) {
        prosper::frontend::LiveComputeImageImport lease;
        const bool ok = prosper::frontend::import_live_compute_storage_image(r, r.size, lease);
        check(ok == expected && ok == lease.valid(), message);
        return lease;
    }
    std::vector<uint8_t> draw_lease(const prosper::frontend::LiveComputeImageImport& lease) {
        if (!lease.valid()) return {};
        auto r = sampled(); r.binding = 4;
        ShaderResourceTable table; table.resources = {r};
        const uint32_t fs[] = {0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u,
                              0xf0800f08u, 0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u};
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1; resource.tw = W; resource.th = H;
        resource.texture_format = static_cast<VkFormat>(lease.native_format);
        resource.borrowed_compute_image = lease.image;
        resource.borrowed_compute_device = lease.device;
        resource.borrowed_compute_image_layout = lease.layout;
        resource.borrowed_compute_image_lease = lease.lease;
        prosper::test::BackendDraw draw;
        draw.vs.assign(std::begin(kTriVertSpv), std::end(kTriVertSpv));
        draw.fs = recompile_fragment(fs, std::size(fs), &table);
        draw.vcount = 3; draw.R = {resource};
        return prosper::test::render_draws_rgba({draw}, W, H);
    }
    bool center(const std::vector<uint8_t>& rgba, bool colored) const {
        const size_t at = (H / 2 * W + W / 2) * 4;
        return rgba.size() == N * 4 && rgba[at] == (colored ? 255 : 0) &&
            rgba[at + 1] == 0 && rgba[at + 2] == (colored && !numeric_alias ? 255 : 0) && rgba[at + 3] == 255;
    }
    void base_oracle() {
        std::vector<uint8_t> linear(N * image.num_components * (transfer_test ? 4 : 2));
        detile_surface(reinterpret_cast<uint8_t*>(linear.data()), pixels.data(), W, H, image.tile_mode, 0, transfer_test ? 4 : numeric_alias ? 2 : 8);
        bool exact = true;
        for (size_t i = 0; i < N * image.num_components; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, linear.data() + i * (transfer_test ? 4 : 2), transfer_test ? 4 : 2);
            exact &= bits == (transfer_test ? 0x3f800000u : !numeric_alias && i % 4 == 1 ? 0u : 0x3c00u);
        }
        check(exact, "every native base texel has the exact expected component bits");
        check(std::all_of(pixels.begin() + pixel_bytes, pixels.end(), [](uint8_t x) { return x == 0xc7; }),
              "base allocation tail unchanged");
    }
    void guards() {
        auto r = sampled();
        borrow(r, true, "restored full-FF metadata permits a fresh borrow");
        kind = CompressionMetadataKind::Unknown;
        borrow(r, false, "unknown metadata kind cannot authorize retained base");
        kind = CompressionMetadataKind::Htile;
        borrow(r, false, "HTILE interpretation cannot authorize this DCC result");
        kind = CompressionMetadataKind::Dcc;
        const auto snapshot = guest_gpu_write_snapshot();
        metadata[metadata_bytes - 1] = 0x40; // Deliberately no notification for this first probe.
        check(guest_gpu_writes_since(snapshot, image.gpu_addr, pixel_bytes) == GuestGpuWriteQuery::Unchanged &&
              guest_gpu_writes_since(snapshot, image.metadata_addr, metadata_bytes) == GuestGpuWriteQuery::Unchanged,
              "unchanged journal does not observe this unnotified metadata mutation");
        borrow(r, false, "fresh full-plane proof rejects an unnotified metadata mutation");
        notify_guest_gpu_write(image.metadata_addr + metadata_bytes - 1, 1);
        borrow(r, false, "last metadata byte participates in complete-plane proof");
        metadata[metadata_bytes - 1] = 0xff;
        notify_guest_gpu_write(image.metadata_addr + metadata_bytes - 1, 1);
        auto h = r; h.dcc_metadata_host_data = hosted.data(); h.dcc_metadata_host_data_size = metadata_bytes;
        borrow(h, true, "complete hosted FF plane is accepted");
        hosted.back() = 0x98; hosted[metadata_bytes - 1] = 0x40;
        borrow(h, false, "hosted plane is authoritative over unchanged FF guest metadata");
        hosted[metadata_bytes - 1] = 0xff;
        h.dcc_metadata_host_data_size = metadata_bytes - 1;
        borrow(h, false, "short hosted plane cannot fall back to valid guest metadata");
        auto unsupported = r; unsupported.sample_count = 2;
        borrow(unsupported, false, "same-key multisample request cannot borrow single-sample compressed content");
        unsupported = r; unsupported.mip_chain_base_level = 1;
        borrow(unsupported, false, "same-key selected mip cannot borrow base-level DCC proof");
        unsupported = r; unsupported.metadata_addr = 0;
        borrow(unsupported, false, "missing metadata address refuses same pixel cache key");
        unsupported = r; unsupported.img_dim = 2; unsupported.depth = 2;
        borrow(unsupported, false, "unsupported compressed shape cannot authorize retained image");
        borrow(r, true, "failed metadata checks do not consume or invalidate the valid lease");
    }
    void run() {
        if (!metadata_bytes || metadata_bytes > 65536 || metadata_bytes % (W * 4)) return;
        set_metadata_kind_query([&](const MetadataKindRequest& request) {
            return request.resource_addr == image.gpu_addr && request.metadata_addr == image.metadata_addr &&
                   request.num_components == image.num_components && request.img_dim == 1 &&
                   (request.format == image.format || request.format == sampled().format)
                       ? kind : CompressionMetadataKind::Unknown;
        });
        auto producer = writer(), meta = metadata_writer(), consumer = reader();
        // Exercise repeated publication before testing public retained-result authority.
        check(prosper::frontend::execute_live_compute_items({producer}), "producer warmup executes");
        DrawItem initial, changed;
        initial.draw_index = 1; initial.command_order = 30;
        changed.draw_index = 2; changed.command_order = 50;
        prosper::frontend::LiveComputeImageImport held;
        std::vector<uint8_t> saved_pixels;
        unsigned draws = 0;
        uint64_t transfer_before = 0;
        std::vector<SubmitOperation> operations{
            {SubmitOperationKind::Dispatch, 1, 20}, {SubmitOperationKind::Draw, 1, 30},
            {SubmitOperationKind::Dispatch, 2, 40}, {SubmitOperationKind::Draw, 2, 50}};
        // The established sampled fast-clear decoder is RGBA16F only. The R16 numeric retry
        // exercises borrow refusal without inventing a decoded color for its compressed state.
        if (!numeric_alias || transfer_test) operations.push_back({SubmitOperationKind::Dispatch, 3, 60});
        const auto result = execute_ordered_items(
            operations, {initial, changed}, {producer, meta, consumer},
            [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                if (++draws == 1) {
                    base_oracle();
                    held = borrow(sampled(), true, "full-FF native result initially imports");
                    check(center(draw_lease(held), true), "real GPU draw reads initial imported native color");
                    saved_pixels = pixels;
                    if (transfer_test) {
                        const auto count = prosper::frontend::live_compute_storage_transfer_seeds();
                        auto first_consumer = consumer;
                        first_consumer.dispatch_index = 4; first_consumer.command_order = 32;
                        check(prosper::frontend::execute_live_compute_items({first_consumer}), "initial transfer consumer executes");
                        check(prosper::frontend::live_compute_storage_transfer_seeds() == count + 1,
                              "plain R32_UINT result admits the Float32 sampled transfer borrow");
                        bool red = true;
                        for (size_t i = 0; i < N * 4; ++i)
                            red &= output[i] == (i % 4 == 0 || i % 4 == 3 ? 0x3f800000u : 0u);
                        check(red, "every initial transfer consumer texel is red with alpha one");
                        std::fill(output.begin(), output.end(), 0xdeadbeefu);
                    }
                    uint32_t unrelated = 0;
                    notify_guest_gpu_write(reinterpret_cast<uint64_t>(&unrelated), sizeof(unrelated));
                    borrow(sampled(), true, "unrelated write preserves public import");
                    guards();
                } else {
                    check(pixels == saved_pixels, "separate metadata writer leaves complete base allocation unchanged");
                    check(std::all_of(metadata.begin(), metadata.begin() + metadata_bytes,
                                      [](uint8_t x) { return x == 0x40; }), "separate writer sets full changed DCC metadata plane");
                    if (transfer_test) {
                        // No supported fast-clear materializer applies to Float32. Keep the known
                        // base-byte fallback and test only whether a retained result is borrowed.
                        std::fill_n(metadata.begin(), metadata_bytes, 0xff);
                        metadata[metadata_bytes - 1] = 0x40;
                        notify_guest_gpu_write(image.metadata_addr, metadata_bytes);
                    }
                    transfer_before = prosper::frontend::live_compute_storage_transfer_seeds();
                    auto rejected = borrow(sampled(), false, "metadata-only write refuses stale public graphics import");
                    if (rejected.valid() && !numeric_alias) {
                        const auto stale = draw_lease(rejected);
                        std::fprintf(stderr, "stale-import-gpu-magenta=%d\n", center(stale, true));
                        check(center(stale, false), "accepted graphics lease must not expose stale magenta under black-clear metadata");
                    }
                    // Already-issued lease keeps the allocation alive; it is not a new authority grant.
                    check(center(draw_lease(held), true), "outstanding old lease remains usable while new borrow is refused");
                }
                return RenderedFrame{};
            }, [](const std::vector<ComputeItem>& items) { return prosper::frontend::execute_live_compute_items(items); }, W, H);
        check(result.compute_executed && draws == 2, "all separate dispatches and public consumer boundaries execute");
        bool black = true;
        for (size_t i = 0; i < N * 4; ++i) black &= output[i] == (i % 4 == 3 ? 0x3f800000u : 0u);
        if (transfer_test) {
            check(prosper::frontend::live_compute_storage_transfer_seeds() == transfer_before,
                  "changed nonuniform metadata prevents the sampled storage-transfer borrow");
            bool red = true;
            for (size_t i = 0; i < N * 4; ++i)
                red &= output[i] == (i % 4 == 0 || i % 4 == 3 ? 0x3f800000u : 0u);
            check(red, "unsupported metadata preserves existing Float32 base-byte fallback semantics");
        }
        if (!numeric_alias)
            check(black, "ordinary sampled compute reads every DCC-cleared texel as black with alpha one");
        check(std::all_of(output.begin() + N * 4, output.end(), [](uint32_t x) { return x == 0xdeadbeefu; }) &&
              std::all_of(metadata.begin() + metadata_bytes, metadata.end(), [](uint8_t x) { return x == 0x97; }) &&
              std::all_of(hosted.begin() + metadata_bytes, hosted.end(), [](uint8_t x) { return x == 0x98; }),
              "metadata and output tails remain untouched");
        held = {};
        set_metadata_kind_query({});
    }
};
}
int main() {
    if (!prosper::test::render_vk_ctx().ok) return 1;
    // Keep independent identities alive together; do not let allocator address reuse hide a miss.
    Fixture exact(false), numeric(true), transfer(true, true);
    exact.run(); numeric.run(); transfer.run();
    std::printf("storage_metadata_borrow: %d failures\n", failures);
    return failures ? 1 : 0;
}
