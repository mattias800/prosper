// test_compute_linear_row_pitch -- live compute dispatches over a padded linear 2D image and over a
// fast-cleared renderer target, through `execute_live_compute_items`.
//
// `test_linear_image_pitch` and `test_sampled_dcc_fast_clear` pin the helpers; these cases pin the
// call sites in live_compute.cpp that use them, so reverting a call site turns a case red.
//
// Uint32 at 100 px is a 400-byte row, which GFX10 pads to a 512-byte pitch: no image here states a
// pitch except StatedPitchOverridesTheRule, which registers one the way AvPlayer does.
//
// WHAT EACH TEST KILLS:
//   SampledUploadDropsRowPadding        the sampled upload reads a padded linear image as tight rows
//                                       (the row-repack / `remap` hunk), so rows 1.. shear
//   SampledSpanCoversThePaddedLastRow   the sampled backing span stays tight, so a write to the
//                                       last row's texels beyond `width*height*bpt` is not part of
//                                       the image's identity and a stale cached upload is reused
//   ComputeProducerSeedsComputeConsumer the producer and the consumer key the retained result by
//                                       different spans, so the borrow misses; also checks the
//                                       writeback's guest rows sit at the padded pitch
//   StorageSeedReadsPaddedRows          the storage seed reads a padded image as tight rows
//   ComputeLinearRowPitchDefaults.ProducerConsumerThroughGuestMemory
//                                       at default cache thresholds nothing is retained, so the
//                                       chain goes through guest memory: the storage writeback must
//                                       write rows at the padded pitch the consumer reads
//                                       (registered twice; see CMakeLists.txt)
//   StatedPitchOverridesTheRule         a registered pitch loses to the 256-byte alignment on either
//                                       the producer or the consumer side
//   RendererTouchedAddressKeepsOneLayout an address the renderer owns (seeded, or handed a compute
//                                       result) is written back tight while plain memory is padded,
//                                       so one address changes layout between dispatches
//   GraphicsStorageWriteComputeSample   a draw's storage image (UAV) writes a linear image back as
//                                       tight rows while compute reads it at the padded pitch (#4618)
//   ComputeStorageWriteGraphicsStorageRead
//                                       the reverse: a draw's storage image seeds tight from the
//                                       padded rows a compute pass wrote, or writes its own output
//                                       tight
//   GraphicsStorageAtomicKeepsThePitch  the typed R32_UINT storage path (image atomics) seeds tight
//                                       or writes back tight
//   SampledRendererTargetRegistersPlane a compute pass that samples a renderer target through a
//                                       DCC-compressed T# does not tell the renderer about the
//                                       control plane, so a later plane write leaves stale pixels
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "fixtures/render_runner.h"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t W = 100;   // 400-byte rows: not a multiple of 256
constexpr uint32_t H = 4;
constexpr uint32_t Bpt = 4;   // Uint32 x1
constexpr uint32_t Pitch = 512;   // GFX10 linear rows round up to 256 bytes
constexpr uint32_t PaddedSpan = Pitch * (H - 1u) + W * Bpt;
constexpr uint8_t Poison = 0xee;

// Every allocation outlives the process's compute caches, so no two cases share an address.
std::vector<uint8_t>& keep_alive(size_t bytes, uint8_t fill) {
    static std::vector<std::unique_ptr<std::vector<uint8_t>>> owned;
    owned.push_back(std::make_unique<std::vector<uint8_t>>(bytes, fill));
    return *owned.back();
}

// The guest STATES this allocation's row pitch, as AvPlayer does for its planes.
std::vector<uint8_t>& stated_pitch(std::vector<uint8_t>& backing, uint32_t pitch) {
    register_guest_linear_texture_layout(reinterpret_cast<uint64_t>(backing.data()), backing.size(),
                                         pitch);
    return backing;
}

uint32_t texel_value(uint32_t base, uint32_t x, uint32_t y) {
    return base + y * W + x;
}

ComputeItem compile(const std::vector<uint32_t>& code, const ShaderResourceTable& resources,
                    uint32_t index, uint32_t native_support) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    config.local_x = W;
    config.local_y = config.local_z = 1;
    config.native_storage_format_support = native_support;
    ComputeItem item;
    item.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    item.user_sgprs = config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.code_addr = 0x30490000ull + uint64_t{index} * 0x100u;
    item.dispatch_index = index;
    item.command_order = uint64_t{index} * 10u;
    item.launch.threads_x = item.launch.local_x = W;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    return item;
}

ShaderResource linear_image(const uint8_t* data, size_t bytes, ResourceClass cls) {
    ShaderResource r{};
    r.cls = cls;
    r.binding = 5;
    r.sgpr_base = 8;
    r.img_dim = 1;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.width = W;
    r.height = H;
    r.depth = 1;
    r.tile_mode = 0;
    r.declared_mip_levels = 1;
    r.gpu_addr = reinterpret_cast<uint64_t>(data);
    r.size = static_cast<uint32_t>(bytes);
    for (uint32_t c = 0; c < 4; ++c) r.swizzle[c] = 4u + c;
    return r;
}
ShaderResource linear_image(const std::vector<uint8_t>& backing, ResourceClass cls) {
    return linear_image(backing.data(), backing.size(), cls);
}

ShaderResource output_buffer(std::vector<uint32_t>& out) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = 2;
    r.sgpr_base = 0;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.stride = 4;
    r.gpu_addr = reinterpret_cast<uint64_t>(out.data());
    r.size = static_cast<uint32_t>(out.size() * sizeof(uint32_t));
    return r;
}

// One lane per column: for each row, image_load the texel at (x, y) and store it to out[y*W + x].
// clang-format off: one instruction per line, with its disassembly
std::vector<uint32_t> reader_program(uint32_t dmask = 1u) {
    std::vector<uint32_t> code{0x7e080300u};                 // v_mov_b32 v4, v0 (x)
    for (uint32_t y = 0; y < H; ++y)
        code.insert(code.end(), {
            0x7e0202ffu, y,                                  // v_mov_b32 v1, y
            0xf0000008u | (dmask << 8), 0x00020800u,         // image_load v8.., v[0:1], s[8:15] 2D
            0xbf8c3f70u,                                     // s_waitcnt vmcnt(0)
            0x4a0a08ffu, y * W,                              // v_add_nc_u32 v5, y*W, v4
            0xe0702000u, 0x80000805u,                        // buffer_store_dword v8, v5 idxen
        });
    code.push_back(0xbf810000u);                             // s_endpgm
    return code;
}
// clang-format on

// One lane per column: for each row, image_store base + y*W + x at (x, y).
// clang-format off: one instruction per line, with its disassembly
std::vector<uint32_t> writer_program(uint32_t base) {
    std::vector<uint32_t> code{0x7e080300u};                 // v_mov_b32 v4, v0 (x)
    for (uint32_t y = 0; y < H; ++y)
        code.insert(code.end(), {
            0x7e0a02ffu, y,                                  // v_mov_b32 v5, y
            0x4a0008ffu, base + y * W,                       // v_add_nc_u32 v0, base + y*W, v4
            0xf0200108u, 0x00020004u,                        // image_store v0, v[4:5], s[8:15] 2D
        });
    code.push_back(0xbf810000u);
    return code;
}
// clang-format on

void write_padded(std::vector<uint8_t>& backing, uint32_t base) {
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const uint32_t value = texel_value(base, x, y);
            std::memcpy(backing.data() + size_t(y) * Pitch + size_t(x) * Bpt, &value, Bpt);
        }
}

std::vector<uint32_t> expected_values(uint32_t base) {
    std::vector<uint32_t> values(size_t(W) * H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) values[size_t(y) * W + x] = texel_value(base, x, y);
    return values;
}

bool sample_into(const ShaderResource& image, std::vector<uint32_t>& out, uint32_t index) {
    out.assign(size_t(W) * H, 0xdeadbeefu);
    ShaderResourceTable table;
    table.resources = {output_buffer(out), image};
    const ComputeItem item = compile(reader_program(), table, index, 0u);
    if (item.spirv.empty()) return false;
    return prosper::frontend::execute_live_compute_items({item});
}

}   // namespace

TEST(ComputeLinearRowPitch, SampledUploadDropsRowPadding) {
    auto& backing = keep_alive(size_t(Pitch) * H, Poison);
    write_padded(backing, 0x10000000u);
    std::vector<uint32_t> out;
    ASSERT_TRUE(sample_into(linear_image(backing, ResourceClass::Texture), out, 1));
    EXPECT_EQ(out, expected_values(0x10000000u))
        << "every row of a 400-byte-wide linear image starts 512 bytes after the previous one";
}

TEST(ComputeLinearRowPitch, SampledSpanCoversThePaddedLastRow) {
    auto& backing = keep_alive(PaddedSpan, Poison);
    write_padded(backing, 0x20000000u);
    const ShaderResource image = linear_image(backing, ResourceClass::Texture);
    std::vector<uint32_t> out;
    ASSERT_TRUE(sample_into(image, out, 2));
    ASSERT_EQ(out, expected_values(0x20000000u));

    // The last row's final texel lives past `W*H*Bpt`: only a padded span sees this write.
    constexpr size_t last = size_t(Pitch) * (H - 1u) + size_t(W - 1u) * Bpt;
    static_assert(last >= size_t(W) * H * Bpt, "the probe must sit outside the tight span");
    const uint32_t changed = 0x2abcdef0u;
    std::memcpy(backing.data() + last, &changed, Bpt);
    ASSERT_TRUE(sample_into(image, out, 3));
    auto expected = expected_values(0x20000000u);
    expected.back() = changed;
    EXPECT_EQ(out, expected) << "a cached upload keyed by the tight span hid the last-row write";
}

namespace {

// Dispatch 1 writes base + y*W + x through a storage binding of `backing`; dispatch 2 samples the
// same padded linear image into `out`. Returns how many retained-result transfer seeds the
// consumer took (1 = it borrowed the producer's native image, 0 = it read guest memory).
uint64_t produce_then_consume(std::vector<uint8_t>& backing, uint32_t base,
                              std::vector<uint32_t>& out, uint32_t first_index) {
    const ShaderResource storage = linear_image(backing, ResourceClass::StorageImage);
    ShaderResourceTable writer_table;
    writer_table.resources = {storage};
    std::vector<ComputeItem> items{
        compile(writer_program(base), writer_table, first_index,
                native_storage_format_support_bit(DataFormat::Uint32, 1))};
    out.assign(size_t(W) * H, 0xdeadbeefu);
    ShaderResourceTable reader_table;
    reader_table.resources = {output_buffer(out), linear_image(backing, ResourceClass::Texture)};
    items.push_back(compile(reader_program(), reader_table, first_index + 1u, 0u));
    for (const auto& item : items) EXPECT_FALSE(item.spirv.empty());

    std::vector<SubmitOperation> operations;
    operations.reserve(items.size());
    for (const auto& item : items)
        operations.push_back(
            {SubmitOperationKind::Dispatch, item.dispatch_index, item.command_order});
    uint64_t consumer_seeds = ~0ull;
    bool all_ok = true;
    const auto result = execute_ordered_items(
        operations, {}, items,
        [](const std::vector<DrawItem>&, uint32_t, uint32_t) { return RenderedFrame{}; },
        [&](const std::vector<ComputeItem>& batch) {
            const auto before = prosper::frontend::live_compute_storage_transfer_seeds();
            const bool ok = prosper::frontend::execute_live_compute_items(batch);
            all_ok &= ok;
            if (batch.size() == 1u && batch[0].dispatch_index == first_index + 1u)
                consumer_seeds = prosper::frontend::live_compute_storage_transfer_seeds() - before;
            return ok;
        },
        1, 1);
    EXPECT_TRUE(result.compute_executed && all_ok);
    return consumer_seeds;
}

// The storage writeback must lay rows at the padded pitch and leave the padding alone.
void expect_padded_guest_rows(const uint8_t* backing, uint32_t base, uint8_t padding) {
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t value = 0;
            std::memcpy(&value, backing + size_t(y) * Pitch + size_t(x) * Bpt, Bpt);
            ASSERT_EQ(value, texel_value(base, x, y)) << "guest texel (" << x << ", " << y << ")";
        }
        if (y + 1u == H) break;   // the last row's padding is outside the image's span
        for (size_t b = size_t(W) * Bpt; b < Pitch; ++b)
            ASSERT_EQ(backing[size_t(y) * Pitch + b], padding)
                << "row " << y << " padding byte " << b;
    }
}
void expect_padded_guest_rows(const std::vector<uint8_t>& backing, uint32_t base, uint8_t padding) {
    expect_padded_guest_rows(backing.data(), base, padding);
}

}   // namespace

TEST(ComputeLinearRowPitch, ComputeProducerSeedsComputeConsumer) {
    auto& backing = keep_alive(size_t(Pitch) * H, Poison);
    constexpr uint32_t Base = 0x30000000u;
    std::vector<uint32_t> out;
    EXPECT_EQ(produce_then_consume(backing, Base, out, 4), 1u)
        << "the consumer must borrow the producer's retained native image, not re-read guest bytes";
    EXPECT_EQ(out, expected_values(Base));
    expect_padded_guest_rows(backing, Base, Poison);
}

TEST(ComputeLinearRowPitch, StorageSeedReadsPaddedRows) {
    // A storage binding that is only read: its seed must gather the rows from the padded pitch.
    auto& backing = keep_alive(size_t(Pitch) * H, Poison);
    constexpr uint32_t Base = 0x40000000u;
    write_padded(backing, Base);
    const std::vector<uint8_t> before = backing;
    std::vector<uint32_t> out(size_t(W) * H, 0xdeadbeefu);
    ShaderResourceTable table;
    table.resources = {output_buffer(out), linear_image(backing, ResourceClass::StorageImage)};
    const ComputeItem item = compile(reader_program(), table, 7,
                                     native_storage_format_support_bit(DataFormat::Uint32, 1));
    ASSERT_FALSE(item.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    EXPECT_EQ(out, expected_values(Base)) << "the storage seed read tight rows out of padded ones";
    EXPECT_EQ(backing, before) << "a read-only storage pass must leave the guest layout as it was";
}

// Registered a second time WITHOUT the zero cache minimums (CMakeLists.txt), so it runs at the
// default 4 KiB thresholds: a 1,600-byte image is never retained, the consumer cannot borrow, and
// the chain goes through guest memory -- the path every sub-4 KiB image takes on a default run.
TEST(ComputeLinearRowPitchDefaults, ProducerConsumerThroughGuestMemory) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read before any thread exists; single-threaded test
    if (std::getenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB") ||
        // NOLINTNEXTLINE(concurrency-mt-unsafe): as above
        std::getenv("PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB"))
        GTEST_SKIP()
            << "needs the default cache thresholds; see the _default_thresholds registration";
    auto& backing = keep_alive(size_t(Pitch) * H, Poison);
    constexpr uint32_t Base = 0x50000000u;
    std::vector<uint32_t> out;
    EXPECT_EQ(produce_then_consume(backing, Base, out, 8), 0u)
        << "control: below the cache minimum the consumer reads guest memory";
    expect_padded_guest_rows(backing, Base, Poison);
    EXPECT_EQ(out, expected_values(Base))
        << "the producer's guest layout and the consumer's read pitch disagree";
}

TEST(ComputeLinearRowPitch, SampledRendererTargetRegistersPlane) {
    // The replay RTT seed writer is how a test publishes a real renderer-owned target; the live
    // renderer installs it only when asked (see guest_write_drain's registration).
    // NOLINTNEXTLINE(concurrency-mt-unsafe): set before the renderer starts any thread
    ::setenv("PROSPER_GPU_REPLAY_RTT_SEEDS", "1", 1);
    prosper::frontend::register_live_renderer("", false);
    constexpr uint32_t Side = 4;
    auto& target = keep_alive(size_t(Side) * Side * 4u, 0);
    auto& miss = keep_alive(64, 0);
    const uint64_t target_addr = reinterpret_cast<uint64_t>(target.data());
    const uint64_t miss_addr = reinterpret_cast<uint64_t>(miss.data());
    const auto drain = [&] { (void)is_live_render_target(miss_addr); };

    GpuCaptureRttSeed seed;
    seed.guest_addr = target_addr;
    seed.width = seed.height = Side;
    seed.format = GpuCaptureColorFormat::Rgba8Unorm;
    seed.rgba.assign(size_t(Side) * Side * 4u, 0x40);
    std::string error;
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    ASSERT_TRUE(is_live_render_target(target_addr));

    ShaderResource sampled{};
    sampled.cls = ResourceClass::Texture;
    sampled.binding = 5;
    sampled.sgpr_base = 8;
    sampled.img_dim = 1;
    sampled.format = DataFormat::Unorm8;
    sampled.num_components = 4;
    sampled.width = sampled.height = Side;
    sampled.depth = 1;
    sampled.declared_mip_levels = 1;
    sampled.tile_mode = 27;
    sampled.gpu_addr = target_addr;
    sampled.compression_enabled = true;
    for (uint32_t c = 0; c < 4; ++c) sampled.swizzle[c] = 4u + c;
    sampled.metadata_addr = 0x1000;   // the footprint needs a plane address; placed below
    const uint64_t plane_bytes = gpu_capture_dcc_metadata_footprint(sampled);
    ASSERT_NE(plane_bytes, 0u);
    auto& plane = keep_alive(plane_bytes, 0xff);
    sampled.metadata_addr = reinterpret_cast<uint64_t>(plane.data());
    sampled.size = static_cast<uint32_t>(target.size());

    // Control: before any pass names the plane, the target has no reason to watch it.
    notify_guest_gpu_write(sampled.metadata_addr, 4);
    notify_guest_gpu_write(miss_addr, 4);
    drain();
    ASSERT_TRUE(is_live_render_target(target_addr))
        << "a write to a plane nobody registered must leave the target alone";

    std::vector<uint32_t> out(size_t(W) * H, 0);
    ShaderResourceTable table;
    table.resources = {output_buffer(out), sampled};
    const ComputeItem item = compile(reader_program(0xfu), table, 6, 0u);
    ASSERT_FALSE(item.spirv.empty());
    (void)prosper::frontend::execute_live_compute_items({item});   // the binding is what matters

    ASSERT_TRUE(is_live_render_target(target_addr)) << "sampling must not itself revoke the target";
    notify_guest_gpu_write(sampled.metadata_addr, 4);
    notify_guest_gpu_write(miss_addr, 4);
    drain();
    EXPECT_FALSE(is_live_render_target(target_addr))
        << "a plane the compute pass read through the T# revokes the target's pixels when "
           "rewritten";
}

namespace {

// An RGBA8 storage image over `backing`, written with every channel = 1.0 (0xff) or 0.0 by one lane
// per column.
ComputeItem rgba8_fill(std::vector<uint8_t>& backing, bool ones, uint32_t index) {
    ShaderResource output{};
    output.cls = ResourceClass::StorageImage;
    output.binding = 5;
    output.sgpr_base = 8;
    output.img_dim = 1;
    output.format = DataFormat::Unorm8;
    output.num_components = 4;
    output.width = W;
    output.height = H;
    output.depth = 1;
    output.declared_mip_levels = 1;
    output.gpu_addr = reinterpret_cast<uint64_t>(backing.data());
    output.size = static_cast<uint32_t>(size_t(W) * H * 4u);
    for (uint32_t c = 0; c < 4; ++c) output.swizzle[c] = 4u + c;
    ShaderResourceTable table;
    table.resources = {output};
    const uint32_t value = ones ? 0xf2u : 0x80u;   // inline constant 1.0 or 0
    // clang-format off: one instruction per line
    std::vector<uint32_t> code{
        0x7e080300u,                                       // v_mov_b32 v4, v0 (x)
        0x7e000200u | value, 0x7e020200u | value,          // v0 = v1 = value
        0x7e040200u | value, 0x7e060200u | value,          // v2 = v3 = value
    };
    for (uint32_t y = 0; y < H; ++y)
        code.insert(code.end(), {
            0x7e0a02ffu, y,                                // v_mov_b32 v5, y
            0xf0200f08u, 0x00020004u,                      // image_store v[0:3], v[4:5], s[8:15]
        });
    code.push_back(0xbf810000u);
    // clang-format on
    return compile(code, table, index, native_storage_format_support_bit(DataFormat::Unorm8, 4));
}

}   // namespace

TEST(ComputeLinearRowPitch, RendererTouchedAddressKeepsOneLayout) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): set before the renderer starts any thread
    ::setenv("PROSPER_GPU_REPLAY_RTT_SEEDS", "1", 1);
    prosper::frontend::register_live_renderer("", false);
    const auto expect_rows = [](const std::vector<uint8_t>& backing, uint8_t texel,
                                const char* when) {
        for (uint32_t y = 0; y < H; ++y) {
            for (size_t b = 0; b < size_t(W) * 4u; ++b)
                ASSERT_EQ(backing[size_t(y) * Pitch + b], texel) << when << " row " << y;
            for (size_t b = size_t(W) * 4u; y + 1u < H && b < Pitch; ++b)
                ASSERT_EQ(backing[size_t(y) * Pitch + b], 0x11u) << when << " padding, row " << y;
        }
    };

    // (1) Plain memory: the first result is handed to the renderer, which then owns the address.
    auto& plain = keep_alive(size_t(Pitch) * H, 0x11);
    const uint64_t plain_address = reinterpret_cast<uint64_t>(plain.data());
    const ComputeItem first = rgba8_fill(plain, true, 16);
    ASSERT_FALSE(first.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({first}));
    expect_rows(plain, 0xffu, "first dispatch");
    ASSERT_TRUE(is_live_render_target(plain_address))
        << "control: the result must reach the renderer, or the second dispatch tests nothing";
    const ComputeItem second = rgba8_fill(plain, false, 17);
    ASSERT_FALSE(second.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({second}));
    expect_rows(plain, 0x00u, "second dispatch, renderer-owned");

    // (2) An address the renderer owned before any compute pass.
    auto& seeded = keep_alive(size_t(Pitch) * H, 0x11);
    const uint64_t seeded_address = reinterpret_cast<uint64_t>(seeded.data());
    GpuCaptureRttSeed seed;
    seed.guest_addr = seeded_address;
    seed.width = W;
    seed.height = H;
    seed.format = GpuCaptureColorFormat::Rgba8Unorm;
    seed.rgba.assign(size_t(W) * H * 4u, 0x40);
    std::string error;
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    ASSERT_TRUE(is_live_render_target(seeded_address));
    const ComputeItem third = rgba8_fill(seeded, true, 18);
    ASSERT_FALSE(third.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({third}));
    expect_rows(seeded, 0xffu, "renderer-seeded target");
}

TEST(ComputeLinearRowPitch, StatedPitchOverridesTheRule) {
    // A registered 1,024-byte pitch, wider than the 512-byte alignment: both sides must use it.
    constexpr uint32_t Stated = 1024;
    auto& backing = stated_pitch(keep_alive(size_t(Stated) * H, Poison), Stated);
    constexpr uint32_t Base = 0x70000000u;
    std::vector<uint32_t> out;
    (void)produce_then_consume(backing, Base, out, 20);
    EXPECT_EQ(out, expected_values(Base));
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t value = 0;
            std::memcpy(&value, backing.data() + size_t(y) * Stated + size_t(x) * Bpt, Bpt);
            ASSERT_EQ(value, texel_value(Base, x, y)) << "guest texel (" << x << ", " << y << ")";
        }
}

namespace {

// A full-screen triangle (the live renderer's own fixture shape) and a W x H RGBA8 colour target.
// clang-format off: one instruction per line
const uint32_t kFullscreenVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u,
    0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u,
    0xF80008CFu, 0x04030201u, 0xBF810000u,
};
// clang-format on

ShaderResource draw_storage_image(uint8_t* data, size_t bytes, uint32_t sgpr, uint32_t binding) {
    ShaderResource r{};
    r.cls = ResourceClass::StorageImage;
    r.binding = binding;
    r.sgpr_base = sgpr;
    r.img_dim = 1;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.width = W;
    r.height = H;
    r.depth = 1;
    r.declared_mip_levels = 1;
    r.gpu_addr = reinterpret_cast<uint64_t>(data);
    r.size = static_cast<uint32_t>(bytes);
    for (uint32_t c = 0; c < 4; ++c) r.swizzle[c] = 4u + c;
    return r;
}

// Render one draw whose fragment program is `ps` over the `resources`; the colour target is a
// separate scratch allocation. Returns false if a stage fails to compile.
// Guest memory for a draw's storage image: the renderer reads only REGISTERED guest mappings, so
// these come from sceKernelMapNamedFlexibleMemory rather than the host heap. A span over the
// mapping, kept for the life of the process like keep_alive's vectors.
struct GuestBytes {
    uint8_t* data = nullptr;
    size_t bytes = 0;
    uint8_t& operator[](size_t i) const { return data[i]; }
};
GuestBytes guest_bytes(size_t bytes, uint8_t fill) {
    static uint64_t base = 0, used = 0;
    constexpr uint64_t Arena = 1u << 20;
    if (!base) {
        prosper::register_builtin_hle();
        auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        if (!map || map(reinterpret_cast<uint64_t>(&base), Arena, 3, 0,
                        reinterpret_cast<uint64_t>("linear-row-pitch"), 0) != 0)
            return {};
    }
    const uint64_t aligned = (bytes + 0xfffu) & ~uint64_t{0xfffu};
    if (used + aligned > Arena) return {};
    GuestBytes span{reinterpret_cast<uint8_t*>(base + used), bytes};
    used += aligned;
    std::memset(span.data, fill, bytes);
    return span;
}

bool render_storage_draw(const std::vector<uint32_t>& ps, std::vector<ShaderResource> resources) {
    auto table = std::make_shared<ShaderResourceTable>();
    table->resources = std::move(resources);
    DrawItem draw;
    draw.vs = recompile_vertex(kFullscreenVs, std::size(kFullscreenVs));
    const PixelSystemInputMapping positions{0x300u, 0x300u};   // v0, v1 = pixel position
    draw.fs = recompile_fragment(ps.data(), ps.size(), table.get(), &positions);
    if (draw.vs.empty() || draw.fs.empty()) return false;
    auto& colour = keep_alive(size_t{64} * 1024u, 0);
    draw.prt = table;
    draw.vertex_count = 3;
    draw.ps.topology = 3;
    draw.ps.color_write_mask = 15;
    draw.ps.color0_format = VK_FORMAT_R8G8B8A8_UNORM;
    draw.color0_base = reinterpret_cast<uint64_t>(colour.data());
    draw.color0_width = W;
    draw.color0_height = H;
    (void)render_submit_items({draw}, W, H);
    return true;
}

// clang-format off: one instruction per line, with its disassembly
// Per pixel: v2 = x, v3 = y (from the pixel position), v4 = base + y*W + x.
std::vector<uint32_t> pixel_index_prologue(uint32_t base) {
    return {
        0x7e040f00u,                 // v_cvt_u32_f32 v2, v0
        0x7e060f01u,                 // v_cvt_u32_f32 v3, v1
        0x160806ffu, W,              // v_mul_u32_u24 v4, W, v3
        0x4a080504u,                 // v_add_nc_u32 v4, v4, v2
        0x4a0808ffu, base,           // v_add_nc_u32 v4, base, v4
    };
}
const uint32_t kExportAndEnd[] = {
    0xf800000fu, 0x03020100u,        // exp mrt0 v0, v1, v2, v3
    0xbf810000u,                     // s_endpgm
};
// clang-format on

}   // namespace

TEST(ComputeLinearRowPitch, GraphicsStorageWriteComputeSample) {
    prosper::frontend::register_live_renderer("", false);
    const GuestBytes image = guest_bytes(size_t(Pitch) * H, Poison);
    ASSERT_TRUE(image.data);
    constexpr uint32_t Base = 0x80000000u;
    std::vector<uint32_t> ps = pixel_index_prologue(Base);
    ps.insert(ps.end(), {0xf0200108u, 0x00020402u});   // image_store v4, v[2:3], s[8:15] 2D
    ps.insert(ps.end(), std::begin(kExportAndEnd), std::end(kExportAndEnd));
    ASSERT_TRUE(render_storage_draw(ps, {draw_storage_image(image.data, image.bytes, 8, 4)}));
    expect_padded_guest_rows(image.data, Base, Poison);

    std::vector<uint32_t> out;
    ASSERT_TRUE(
        sample_into(linear_image(image.data, image.bytes, ResourceClass::Texture), out, 30));
    EXPECT_EQ(out, expected_values(Base)) << "compute reads the rows the draw wrote";
}

TEST(ComputeLinearRowPitch, ComputeStorageWriteGraphicsStorageRead) {
    prosper::frontend::register_live_renderer("", false);
    const GuestBytes source = guest_bytes(size_t(Pitch) * H, Poison);
    const GuestBytes copy = guest_bytes(size_t(Pitch) * H, Poison);
    ASSERT_TRUE(source.data && copy.data);
    constexpr uint32_t Base = 0x90000000u;
    ShaderResourceTable writer_table;
    writer_table.resources = {linear_image(source.data, source.bytes, ResourceClass::StorageImage)};
    const ComputeItem writer = compile(writer_program(Base), writer_table, 32,
                                       native_storage_format_support_bit(DataFormat::Uint32, 1));
    ASSERT_FALSE(writer.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({writer}));
    expect_padded_guest_rows(source.data, Base, Poison);

    // The draw loads each texel of `source` through one storage image and stores it into `copy`
    // through another: both the seed and the writeback must use the padded rows.
    std::vector<uint32_t> ps = pixel_index_prologue(0);
    // clang-format off
    ps.insert(ps.end(), {
        0xf0000108u, 0x00020402u,    // image_load v4, v[2:3], s[8:15] 2D
        0xbf8c3f70u,                 // s_waitcnt vmcnt(0)
        0xf0200108u, 0x00040402u,    // image_store v4, v[2:3], s[16:23] 2D
    });
    // clang-format on
    ps.insert(ps.end(), std::begin(kExportAndEnd), std::end(kExportAndEnd));
    ASSERT_TRUE(render_storage_draw(ps, {draw_storage_image(source.data, source.bytes, 8, 4),
                                         draw_storage_image(copy.data, copy.bytes, 16, 5)}));
    expect_padded_guest_rows(copy.data, Base, Poison);
}

TEST(ComputeLinearRowPitch, GraphicsStorageAtomicKeepsThePitch) {
    // Image atomics declare a typed R32_UINT storage image, which takes the renderer's native
    // R32_UINT path rather than the formatless one. `a` holds an old pattern in the padded rows;
    // the draw swaps a new pattern in and swaps the returned old value into `b`.
    prosper::frontend::register_live_renderer("", false);
    const GuestBytes a = guest_bytes(size_t(Pitch) * H, Poison);
    const GuestBytes b = guest_bytes(size_t(Pitch) * H, Poison);
    ASSERT_TRUE(a.data && b.data);
    constexpr uint32_t Old = 0xa0000000u, New = 0xb0000000u;
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const uint32_t value = texel_value(Old, x, y);
            std::memcpy(a.data + size_t(y) * Pitch + size_t(x) * Bpt, &value, Bpt);
        }
    std::vector<uint32_t> ps = pixel_index_prologue(New);
    // clang-format off
    ps.insert(ps.end(), {
        0xf03c2108u, 0x00020402u,    // image_atomic_swap v4, v[2:3], s[8:15] glc 2D: v4 = old
        0xbf8c3f70u,                 // s_waitcnt vmcnt(0)
        0xf03c2108u, 0x00040402u,    // image_atomic_swap v4, v[2:3], s[16:23] glc 2D
    });
    // clang-format on
    ps.insert(ps.end(), std::begin(kExportAndEnd), std::end(kExportAndEnd));
    ASSERT_TRUE(render_storage_draw(ps, {draw_storage_image(a.data, a.bytes, 8, 4),
                                         draw_storage_image(b.data, b.bytes, 16, 5)}));
    expect_padded_guest_rows(a.data, New, Poison);   // the writeback
    expect_padded_guest_rows(b.data, Old, Poison);   // the seed of `a`, then `b`'s writeback
}
