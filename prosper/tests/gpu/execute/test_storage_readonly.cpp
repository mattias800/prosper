// Read-only storage preserves guest encodings and needs no image result readback.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/host_read_barrier.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/texture/tile.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t width = 64, height = 16, texels = width * height;
int failures = 0;
bool control = false;
void check(bool value, const char* what) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
ShaderResource buffer(std::vector<uint32_t>& output) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = 2; r.sgpr_base = 0; r.stride = 4;
    r.format = DataFormat::Uint32; r.num_components = 1;
    r.size = output.size() * 4;
    r.gpu_addr = reinterpret_cast<uint64_t>(output.data());
    return r;
}
ComputeItem compile(const ShaderResourceTable& table, bool native, int write_sgpr = -1,
                    bool read = true) {
    std::vector<uint32_t> code{0x7e080300u, 0x7e0002ffu, 0xbf800000u}; // X=v4, v0=-1
    for (uint32_t y = 0; y < height; ++y) {
        code.push_back(0x7e0a0280u + y); // Y=v5
        if (read) code.insert(code.end(), {
            0xf0000108u, 0x00020804u, 0xbf8c3f70u, // image_load v8, [v4:v5], s[8:15]
            0xe0702000u | (y * width * 4u), 0x80000804u, // independent SSBO output
        });
        if (write_sgpr >= 0)
            code.insert(code.end(), {0xf0200108u,
                (static_cast<uint32_t>(write_sgpr) / 4u << 16) | 4u});
    }
    code.push_back(0xbf810000u);
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = width; config.local_y = config.local_z = 1;
    config.native_storage_format_support = native
        ? native_storage_format_support_bit(DataFormat::Unorm8, 1) : 0;
    ComputeItem item;
    item.spirv = recompile_compute(code.data(), code.size(), &table, config);
    item.resources = std::make_shared<ShaderResourceTable>(table);
    item.user_sgprs = config.user_sgprs;
    item.code_addr = 0x34078001u;
    item.launch.threads_x = item.launch.local_x = width;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    check(!item.spirv.empty(), "actual IMAGE_LOAD/STORE shader compiles");
    return item;
}
void copy_write_objects(std::vector<uint32_t>& words) {
    // Keep the directly read object, but write through a real OpCopyObject. Reflection must not
    // infer read-only from the direct read while silently missing the copied object's stores.
    std::unordered_map<uint32_t, uint32_t> image_types;
    for (size_t at = 5; at < words.size();) {
        const uint32_t count = words[at] >> 16, op = words[at] & 0xffffu;
        if (op == 61u && count >= 4) image_types[words[at + 2]] = words[at + 1];
        if (op == 99u) {
            const uint32_t object = words[at + 1];
            check(image_types.count(object), "write object has a known type for fixture transform");
            if (!image_types.count(object)) return;
            const uint32_t copy = words[3]++;
            words[at + 1] = copy;
            words.insert(words.begin() + at, {(4u << 16) | 83u, image_types[object], copy, object});
            at += 4;
        }
        at += count;
    }
}
struct Fixture {
    bool wide, native;
    uint32_t bytes, minimum;
    std::vector<uint8_t> guest, sibling;
    std::vector<uint32_t> observed;
    ShaderResource image;
    explicit Fixture(DataFormat format)
        : wide(format == DataFormat::Snorm16), native(format == DataFormat::Unorm8),
          bytes(wide ? 2u : 1u), minimum(native ? 0u : wide ? 0x8000u : 0x80u),
          guest(texels * bytes + 64u, 0xa5), sibling(guest.size(), 0x65),
          observed(texels, 0xccccccccu) {
        image.cls = ResourceClass::StorageImage;
        image.binding = 5; image.sgpr_base = 8; image.img_dim = 1;
        image.format = format; image.num_components = 1;
        image.width = width; image.height = height; image.depth = 1;
        image.size = texels * bytes; image.linear_row_pitch_bytes = width * bytes;
        image.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
    }
    void seed(uint32_t phase) {
        for (uint32_t i = 0; i < texels; ++i) {
            // Redundant SNORM minima alternate with distinct positive/negative neighbors.
            const uint32_t value = native ? (i * 37u + phase * 53u) & 255u
                : (i + phase) % 3u == 0 ? minimum
                : (i + phase) % 3u == 1 ? minimum + 1u
                : (i * 17u + phase * 29u) & (minimum - 1u);
            guest[i * bytes] = static_cast<uint8_t>(value);
            if (wide) guest[i * bytes + 1] = static_cast<uint8_t>(value >> 8);
        }
    }
    void verify_read(const std::vector<uint8_t>& source) {
        bool match = true;
        for (uint32_t i = 0; i < texels; ++i) {
            uint32_t bits = source[i * bytes];
            if (wide) bits |= uint32_t(source[i * bytes + 1]) << 8;
            const float expected = native ? float(bits) / 255.0f
                : std::max(-1.0f, float(wide ? int32_t(static_cast<int16_t>(bits))
                                           : int32_t(static_cast<int8_t>(bits))) /
                                      float(wide ? 32767 : 127));
            const float actual = std::bit_cast<float>(observed[i]);
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.000001f) {
                std::fprintf(stderr, "texel=%u actual=%g expected=%g\n", i, actual, expected);
                match = false; break;
            }
        }
        check(match, "independent GPU SSBO contains every current storage-image value");
    }
    void verify_stored(const std::vector<uint8_t>& target) {
        const uint32_t value = native ? 0u : minimum + 1u;
        bool match = true;
        for (uint32_t i = 0; i < texels; ++i)
            if (target[i * bytes] != static_cast<uint8_t>(value) ||
                (wide && target[i * bytes + 1] != static_cast<uint8_t>(value >> 8))) match = false;
        check(match, "actual image stores reach exact guest encodings");
    }
};
void read_only(DataFormat format) {
    Fixture f(format);
    ShaderResourceTable table; table.resources = {buffer(f.observed), f.image};
    const auto reader = compile(table, f.native);
    const auto report = validate_spirv_descriptor_interface(reader.spirv, &table, 0,
                                                           SpirvShaderStage::Compute, false);
    const auto* d = find_spirv_descriptor_binding(report, 0, 5);
    check(report.ok() && report.storage_image_writes_complete && d &&
          d->kind == SpirvDescriptorKind::StorageImage && d->readable && !d->writable &&
          (!f.native || d->storage_float),
          "reader has complete image-write proof and a real read-only storage binding");
    if (failures) return;
    uint32_t input_notifications = 0, output_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t bytes, const char*) {
        if (address < f.image.gpu_addr + f.image.size && f.image.gpu_addr < address + bytes)
            ++input_notifications;
        const uint64_t output = reinterpret_cast<uint64_t>(f.observed.data());
        if (address < output + f.observed.size() * 4u && output < address + bytes)
            ++output_notifications;
    });
    for (uint32_t phase : {0u, 0u, 1u, 1u}) {
        input_notifications = output_notifications = 0;
        f.seed(phase); const auto before = f.guest;
        std::fill(f.observed.begin(), f.observed.end(), 0xccccccccu);
        const auto barriers = backend_host_read_barrier_count().load();
        check(prosper::frontend::execute_live_compute_items({reader}), "read-only storage executes");
        const auto delta = backend_host_read_barrier_count().load() - barriers;
        check(control ? delta >= 2 : delta == 1,
              "only the independent writable SSBO requires a host-read barrier");
        f.verify_read(before);
        check(f.guest == before, "all read-only guest encodings and backing tail remain byte-exact");
        check(control ? input_notifications > 0 : input_notifications == 0,
              "read-only storage does not publish a guest write");
        check(output_notifications > 0, "observer detects independent SSBO output as a positive control");
    }
    set_guest_gpu_write_observer({});
    if (!f.native || failures) return;
    // A producer's old result A must not survive as a comparison baseline after a read-only
    // upload of new guest input B. A later writer producing A must actually repair guest B.
    std::vector<std::unique_ptr<Fixture>> retained;
    for (bool host_baseline : {false, true}) {
        retained.push_back(std::make_unique<Fixture>(DataFormat::Unorm8));
        auto& g = *retained.back(); // both identities stay alive, preventing allocator address reuse
        ShaderResourceTable separate; separate.resources = {buffer(g.observed), g.image};
        const auto writer = compile(separate, true, 8, false);
        const auto reader_b = compile(separate, true);
        const auto snapshots = prosper::frontend::live_compute_image_result_snapshot_bytes();
        if (host_baseline) prosper::frontend::live_compute_force_next_image_result_host_fallback_for_test();
        check(prosper::frontend::execute_live_compute_items({writer}), "native producer establishes A");
        check(prosper::frontend::live_compute_image_result_snapshot_bytes() - snapshots ==
                  (host_baseline ? texels : 0u),
              "distinct writer establishes the requested host or GPU result baseline");
        g.verify_stored(g.guest);
        g.seed(2); const auto before = g.guest;
        check(prosper::frontend::execute_live_compute_items({reader_b}), "native read-only upload observes B");
        g.verify_read(before); check(g.guest == before, "reader preserves changed guest B");
        check(prosper::frontend::execute_live_compute_items({writer}), "native producer writes A again");
        g.verify_stored(g.guest);
    }
}
void writers(bool lower_binding, bool independent, bool copied) {
    Fixture f(copied ? DataFormat::Unorm8 : DataFormat::Snorm8);
    ShaderResourceTable table; table.resources = {buffer(f.observed), f.image};
    if (!copied) {
        auto alias = f.image; alias.sgpr_base = 16; alias.binding = lower_binding ? 4 : 7;
        if (independent) alias.gpu_addr = reinterpret_cast<uint64_t>(f.sibling.data());
        table.resources.push_back(alias);
    }
    auto item = compile(table, f.native, copied ? 8 : 16);
    if (copied) copy_write_objects(item.spirv);
    const auto report = validate_spirv_descriptor_interface(item.spirv, &table, 0,
                                                           SpirvShaderStage::Compute, false);
    check(report.ok() && report.storage_image_writes_complete != copied,
          "unknown image-object write vetoes omission; direct mixed access remains proven");
    if (failures) return;
    for (uint32_t phase : {0u, 1u}) {
        f.seed(phase); const auto before = f.guest;
        const auto barriers = backend_host_read_barrier_count().load();
        check(prosper::frontend::execute_live_compute_items({item}), "writable control executes");
        check(backend_host_read_barrier_count().load() - barriers >= 2,
              "a real writable image retains its result host-read dependency");
        f.verify_read(before);
        f.verify_stored(independent ? f.sibling : f.guest);
        if (independent) check(f.guest == before, "separate read-only image remains byte-exact");
        check(std::all_of(f.guest.end() - 64, f.guest.end(), [](uint8_t v) { return v == 0xa5; }),
              "writable control preserves the guest backing tail");
    }
}
void dcc_read_only() {
    Fixture f(DataFormat::Unorm8);
    auto image = f.image;
    image.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
    const size_t tiled_bytes = tiled_surface_bytes(width, height, image.tile_mode, 0, 1);
    check(tiled_bytes >= texels, "DCC fixture uses a complete tiled base footprint");
    if (tiled_bytes < texels) return;
    std::vector<uint8_t> tiled(tiled_bytes + 64u, 0x97);
    std::vector<uint8_t> metadata(65536, 0xff);
    image.gpu_addr = reinterpret_cast<uint64_t>(tiled.data());
    image.size = tiled_bytes;
    image.compression_enabled = image.write_compress_enabled = image.meta_pipe_aligned = true;
    image.metadata_addr = reinterpret_cast<uint64_t>(metadata.data());
    const size_t metadata_bytes = gpu_capture_dcc_metadata_footprint(image);
    check(metadata_bytes && metadata_bytes <= metadata.size(), "real DCC metadata footprint is bounded");
    if (!metadata_bytes || metadata_bytes > metadata.size()) return;
    image.dcc_metadata_size = metadata_bytes;
    image.dcc_metadata_host_data = metadata.data();
    image.dcc_metadata_host_data_size = metadata_bytes;
    ShaderResourceTable table; table.resources = {buffer(f.observed), image};
    const auto reader = compile(table, true);
    uint32_t input_notifications = 0, metadata_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t bytes, const char*) {
        if (address < image.gpu_addr + tiled_bytes && image.gpu_addr < address + bytes)
            ++input_notifications;
        if (address < image.metadata_addr + metadata_bytes && image.metadata_addr < address + bytes)
            ++metadata_notifications;
    });
    for (uint32_t phase : {0u, 0u, 1u}) {
        f.seed(phase);
        tile_surface(tiled.data(), f.guest.data(), width, height, image.tile_mode, 0, 1);
        const auto original = tiled;
        const auto metadata_before = metadata;
        input_notifications = metadata_notifications = 0;
        const auto barriers = backend_host_read_barrier_count().load();
        check(prosper::frontend::execute_live_compute_items({reader}), "uncompressed DCC input executes");
        f.verify_read(f.guest);
        if (!control) {
            check(tiled == original, "read-only DCC input preserves all tiled pixels and padding");
            check(input_notifications == 0 && metadata_notifications == 0,
                  "read-only input publishes neither pixels nor DCC metadata");
            check(backend_host_read_barrier_count().load() - barriers == 1,
                  "read-only DCC input adds no image/retile host-read dependencies");
        }
        check(metadata == metadata_before, "all-uncompressed DCC metadata and its tail stay byte-exact");
    }
    set_guest_gpu_write_observer({});
}
} // namespace
int main(int argc, char** argv) {
    control = argc == 2 && !std::strcmp(argv[1], "--control");
    if (argc > 2 || (argc == 2 && !control) ||
        control != (std::getenv("PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP") != nullptr)) return 2;
    read_only(DataFormat::Snorm8);
    read_only(DataFormat::Snorm16);
    read_only(DataFormat::Unorm8);
    writers(false, false, false);
    writers(true, false, false);
    writers(false, true, false);
    writers(false, false, true);
    dcc_read_only();
    return failures ? 1 : 0;
}
