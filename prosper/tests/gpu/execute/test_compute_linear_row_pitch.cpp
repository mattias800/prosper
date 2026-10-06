// test_compute_linear_row_pitch -- live compute dispatches over a padded linear 2D image and over a
// fast-cleared renderer target, through `execute_live_compute_items`.
//
// `test_linear_image_pitch` and `test_sampled_dcc_fast_clear` pin the helpers; these cases pin the
// call sites in live_compute.cpp that use them, so reverting a call site turns a case red.
//
// Every padded image here STATES its pitch through register_guest_linear_texture_layout, the way
// AvPlayer registers its planes: the pitch is never inferred from the 256-byte alignment (#4606).
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
//   UnstatedPitchStaysTight             the 256-byte alignment is inferred at a live call site: an
//                                       unregistered 400-byte-row image is written back padded
//   RendererOwnedTargetStaysTight       a renderer-owned target is written back at a stated pitch,
//                                       although the renderer's own pixels and readers are tight
//   SampledRendererTargetRegistersPlane a compute pass that samples a renderer target through a
//                                       DCC-compressed T# does not tell the renderer about the
//                                       control plane, so a later plane write leaves stale pixels
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
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

// The guest STATES this allocation's row pitch, as AvPlayer does for its planes. Only a stated
// pitch is honoured (#4606); an unregistered allocation keeps tight rows.
std::vector<uint8_t>& stated_pitch(std::vector<uint8_t>& backing, uint32_t pitch = Pitch) {
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
    item.code_addr = 0x30490000ull + index * 0x100u;
    item.dispatch_index = index;
    item.command_order = index * 10u;
    item.launch.threads_x = item.launch.local_x = W;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    return item;
}

ShaderResource linear_image(std::vector<uint8_t>& backing, ResourceClass cls) {
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
    r.gpu_addr = reinterpret_cast<uint64_t>(backing.data());
    r.size = static_cast<uint32_t>(backing.size());
    for (uint32_t c = 0; c < 4; ++c) r.swizzle[c] = 4u + c;
    return r;
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
    auto& backing = stated_pitch(keep_alive(size_t(Pitch) * H, Poison));
    write_padded(backing, 0x10000000u);
    std::vector<uint32_t> out;
    ASSERT_TRUE(sample_into(linear_image(backing, ResourceClass::Texture), out, 1));
    EXPECT_EQ(out, expected_values(0x10000000u))
        << "every row of a 400-byte-wide linear image starts 512 bytes after the previous one";
}

TEST(ComputeLinearRowPitch, SampledSpanCoversThePaddedLastRow) {
    auto& backing = stated_pitch(keep_alive(PaddedSpan, Poison));
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
void expect_padded_guest_rows(const std::vector<uint8_t>& backing, uint32_t base, uint8_t padding) {
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t value = 0;
            std::memcpy(&value, backing.data() + size_t(y) * Pitch + size_t(x) * Bpt, Bpt);
            ASSERT_EQ(value, texel_value(base, x, y)) << "guest texel (" << x << ", " << y << ")";
        }
        if (y + 1u == H) break;   // the last row's padding is outside the image's span
        for (size_t b = size_t(W) * Bpt; b < Pitch; ++b)
            ASSERT_EQ(backing[size_t(y) * Pitch + b], padding) << "row " << y << " padding byte " << b;
    }
}

}  // namespace

TEST(ComputeLinearRowPitch, ComputeProducerSeedsComputeConsumer) {
    auto& backing = stated_pitch(keep_alive(size_t(Pitch) * H, Poison));
    constexpr uint32_t Base = 0x30000000u;
    std::vector<uint32_t> out;
    EXPECT_EQ(produce_then_consume(backing, Base, out, 4), 1u)
        << "the consumer must borrow the producer's retained native image, not re-read guest bytes";
    EXPECT_EQ(out, expected_values(Base));
    expect_padded_guest_rows(backing, Base, Poison);
}

TEST(ComputeLinearRowPitch, StorageSeedReadsPaddedRows) {
    // A storage binding that is only read: its seed must gather the rows from the padded pitch.
    auto& backing = stated_pitch(keep_alive(size_t(Pitch) * H, Poison));
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
    if (std::getenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB") ||
        std::getenv("PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB"))
        GTEST_SKIP() << "needs the default cache thresholds; see the _default_thresholds registration";
    auto& backing = stated_pitch(keep_alive(size_t(Pitch) * H, Poison));
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

TEST(ComputeLinearRowPitch, UnstatedPitchStaysTight) {
    // No registration: the same producer/consumer chain must keep tight rows end to end.
    auto& backing = keep_alive(size_t(Pitch) * H, Poison);
    constexpr uint32_t Base = 0x60000000u;
    std::vector<uint32_t> out;
    (void)produce_then_consume(backing, Base, out, 12);
    EXPECT_EQ(out, expected_values(Base));
    for (uint32_t i = 0; i < W * H; ++i) {
        uint32_t value = 0;
        std::memcpy(&value, backing.data() + size_t(i) * Bpt, Bpt);
        ASSERT_EQ(value, texel_value(Base, i % W, i / W)) << "tight guest texel " << i;
    }
    for (size_t b = size_t(W) * H * Bpt; b < backing.size(); ++b)
        ASSERT_EQ(backing[b], Poison) << "byte " << b << " lies past a tight image";
}

TEST(ComputeLinearRowPitch, RendererOwnedTargetStaysTight) {
    // A renderer-owned RGBA8 target whose address ALSO carries a stated 512-byte pitch: the renderer
    // owns those pixels and reads them tight, so the compute writeback must stay tight.
    ::setenv("PROSPER_GPU_REPLAY_RTT_SEEDS", "1", 1);
    prosper::frontend::register_live_renderer("", false);
    auto& target = stated_pitch(keep_alive(size_t(Pitch) * H, 0x11));
    const uint64_t address = reinterpret_cast<uint64_t>(target.data());
    GpuCaptureRttSeed seed;
    seed.guest_addr = address;
    seed.width = W;
    seed.height = H;
    seed.format = GpuCaptureColorFormat::Rgba8Unorm;
    seed.rgba.assign(size_t(W) * H * 4u, 0x40);
    std::string error;
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    ASSERT_TRUE(is_live_render_target(address));

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
    output.gpu_addr = address;
    output.size = static_cast<uint32_t>(size_t(W) * H * 4u);
    for (uint32_t c = 0; c < 4; ++c) output.swizzle[c] = 4u + c;
    ShaderResourceTable table;
    table.resources = {output};
    // clang-format off: one instruction per line
    std::vector<uint32_t> code{
        0x7e080300u,                           // v_mov_b32 v4, v0 (x)
        0x7e0002f2u, 0x7e0202f2u,              // v0 = v1 = 1.0
        0x7e0402f2u, 0x7e0602f2u,              // v2 = v3 = 1.0
    };
    for (uint32_t y = 0; y < H; ++y)
        code.insert(code.end(), {
            0x7e0a02ffu, y,                    // v_mov_b32 v5, y
            0xf0200f08u, 0x00020004u,          // image_store v[0:3], v[4:5], s[8:15] 2D
        });
    code.push_back(0xbf810000u);
    // clang-format on
    const ComputeItem item = compile(code, table, 14,
                                     native_storage_format_support_bit(DataFormat::Unorm8, 4));
    ASSERT_FALSE(item.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    const size_t tight = size_t(W) * H * 4u;
    for (size_t b = 0; b < tight; ++b) ASSERT_EQ(target[b], 0xffu) << "tight byte " << b;
    for (size_t b = tight; b < target.size(); ++b)
        ASSERT_EQ(target[b], 0x11u) << "byte " << b << " lies past the tight target";
}
