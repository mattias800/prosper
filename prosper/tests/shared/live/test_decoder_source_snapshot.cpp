// Persistent snapshots must describe the exact encoded bytes used by the real decoder.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "host/image/exec_image.hpp"
#include "shared/live/live_renderer.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
int main(int argc, char** argv) {
    const bool control = argc == 2 && std::strcmp(argv[1], "--control") == 0;
    const bool watch = argc == 2 && std::strcmp(argv[1], "--watch") == 0;
    prosper::register_builtin_hle();
    constexpr uint32_t W = 64, H = 4;
    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    const uint32_t ps_rdna[] = {
        // POS_X/Y_FLOAT occupy v0/v1. Scale pixel centers to normalized UVs so every
        // source texel is observed, rather than broadcasting one constant texture sample.
        0x100000ffu, std::bit_cast<uint32_t>(1.0f / W),
        0x100202ffu, std::bit_cast<uint32_t>(1.0f / H),
        0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResource source{};
    source.cls = ResourceClass::Texture;
    source.format = DataFormat::Unorm8;
    source.num_components = 4; source.binding = 4; source.sgpr_base = 8;
    source.img_dim = 1; source.depth = 1;
    source.width = source.height = 2; source.size = 16;
    source.linear_row_pitch_bytes = 8;
    source.mag_filter = source.min_filter = 0;
    source.gpu_addr = 0x71000000;
    std::vector<uint8_t> color(16, 0);
    for (size_t i = 0; i < color.size(); i += 4) { color[i] = 255; color[i+3] = 255; }
    source.host_data = color.data(); source.host_data_size = color.size();
    auto table = std::make_shared<ShaderResourceTable>();
    table->resources.push_back(source);
    DrawItem producer;
    producer.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
    const PixelSystemInputMapping positions{0x300u, 0x300u};
    producer.fs = recompile_fragment(ps_rdna, std::size(ps_rdna), table.get(), &positions);
    producer.prt = table; producer.vertex_count = 3;
    producer.ps.topology = 3; producer.ps.color_write_mask = 15;
    producer.color0_base = 0x72000000;
    producer.color0_width = W; producer.color0_height = H;
    check(!producer.vs.empty() && !producer.fs.empty(), "fixture shaders compile");
    prosper::frontend::register_live_renderer(".", false);
    auto render = [&](const DrawItem& draw) { return render_submit_items({draw}, W, H); };
#ifdef __linux__
    prosper::install_trap_handler();
#endif
    auto alloc = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
    auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0, physical = 0;
    check(alloc && release && map && unmap &&
          alloc(0, 0x200000000ull, 0x10000, 0x10000, 0,
                reinterpret_cast<uint64_t>(&physical)) == 0 &&
          map(reinterpret_cast<uint64_t>(&guest), 0x10000, 2, 0, physical, 0x10000) == 0 && guest,
          "fixture maps real guest memory");
    if (!guest) return 1;
    using prosper::frontend::reset_texture_decode_scope_stats;
    using prosper::frontend::texture_decode_scope_stats;
    for (auto format : {DataFormat::Float32, DataFormat::Float16,
                        DataFormat::Unorm16, DataFormat::Bc1}) {
        std::printf("format=%u\n", static_cast<unsigned>(format));
        auto& r = table->resources[0];
        r.format = format;
        r.num_components = format == DataFormat::Unorm16 ? 1 : 4;
        r.width = W; r.height = H; r.depth = 1; r.img_dim = 1;
        r.host_data = nullptr; r.host_data_size = 0;
        const size_t bpt = format == DataFormat::Float32 ? 16 :
                           format == DataFormat::Float16 ? 8 : 2;
        const size_t bytes = format == DataFormat::Bc1 ? (W / 4) * (H / 4) * 8 : W * H * bpt;
        r.size = bytes;
        r.linear_row_pitch_bytes = format == DataFormat::Bc1 ? W / 4 * 8 : W * bpt;
        // Recompile when changing the format so the sampler follows the real resource contract.
        producer.fs = recompile_fragment(ps_rdna, std::size(ps_rdna), table.get(), &positions);
        std::vector<uint8_t> full_expected[2];
        unsigned source_iteration = 0;
        for (bool short_source : {false, true, false}) {
            const size_t prefix = short_source ? bytes / 2 : bytes;
            r.gpu_addr = guest + 0x10000 - prefix;
            for (unsigned generation = 0; generation < 2; ++generation) {
                std::vector<uint8_t> encoded(bytes, 0);
                if (format == DataFormat::Bc1) {
                    const uint16_t value = generation ? 0x07e0 : 0xf800;
                    for (size_t at = 0; at < bytes; at += 8)
                        std::memcpy(encoded.data() + at, &value, 2);
                } else {
                    for (size_t at = 0; at < bytes; at += bpt) {
                        if (format == DataFormat::Float32) {
                            const float values[4] = {generation ? 0.0f : 1.0f,
                                                     generation ? 1.0f : 0.0f, 0, 1};
                            std::memcpy(encoded.data() + at, values, sizeof(values));
                        } else if (format == DataFormat::Float16) {
                            const uint16_t values[4] = {static_cast<uint16_t>(generation ? 0 : 0x3c00),
                                static_cast<uint16_t>(generation ? 0x3c00 : 0), 0, 0x3c00};
                            std::memcpy(encoded.data() + at, values, sizeof(values));
                        } else {
                            const uint16_t value = generation ? 0x5555 : 0xffff;
                            std::memcpy(encoded.data() + at, &value, sizeof(value));
                        }
                    }
                }
                // Warm the scratch/cache with nonzero bytes, then shorten the next source.
                // The watch arm lowers the admission threshold for full float sources. Mutate it
                // without a journal notification to prove that its page watch really invalidates.
#ifdef __linux__
                const bool cpu_only = watch && bytes >= 1024 && source_iteration == 0 && generation == 1;
#else
                const bool cpu_only = false;
#endif
                const auto watch_before = prosper::host::guest_write_watch_stats();
                std::memcpy(reinterpret_cast<void*>(r.gpu_addr), encoded.data(), prefix);
                if (cpu_only) {
                    check(prosper::host::guest_write_watch_stats().faults > watch_before.faults,
                          "CPU-only mutation faults an armed source watch");
                } else notify_guest_gpu_write(r.gpu_addr, prefix);
                std::fill(encoded.begin() + prefix, encoded.end(), 0);
                auto oracle_table = std::make_shared<ShaderResourceTable>(*table);
                auto& oracle_source = oracle_table->resources[0];
                oracle_source.gpu_addr = 0x78000000;
                oracle_source.host_data = encoded.data();
                oracle_source.host_data_size = bytes;
                DrawItem oracle = producer; oracle.prt = oracle_table; oracle.color0_base = 0x79000000;
                const auto expected = render(oracle);
                check(expected.size() == W * H * 4, "host-backed oracle renders a complete image");
                if (short_source)
                    check(expected != full_expected[generation], "short backing changes visible oracle pixels");
                else full_expected[generation] = expected;
                if (!short_source) {
                    bool exact = true;
                    for (size_t at = 0; at < expected.size(); at += 4) {
                        exact &= expected[at] == (generation ? (format == DataFormat::Unorm16 ? 85 : 0) : 255);
                        exact &= expected[at + 1] == (generation && format != DataFormat::Unorm16 ? 255 : 0);
                        exact &= expected[at + 2] == 0 && expected[at + 3] == 255;
                    }
                    check(exact, "pixel-center shader observes the known encoded color at every texel");
                }
                reset_texture_decode_scope_stats();
                const auto actual = render(producer);
                const auto stats = texture_decode_scope_stats();
                if (cpu_only)
                    check(stats.decodes == 1 &&
                          prosper::host::guest_write_watch_stats().dirty > watch_before.dirty,
                          "target queries its dirty watch and decodes the CPU-only mutation");
                check(actual == expected, "fresh and mutated full/short sources match the zero-tail oracle");
                check(stats.decoder_snapshot_candidate_bytes == bytes &&
                      stats.decoder_snapshot_reused_bytes == (control ? 0 : prefix) &&
                      stats.late_snapshot_read_bytes == (control ? prefix : 0),
                      "exact decoder prefix replaces one snapshot read, with an eager control");
                reset_texture_decode_scope_stats();
                for (unsigned stable = 0; stable < 4; ++stable)
                    check(render(producer) == actual, "unchanged source preserves cached pixels");
                const auto hit = texture_decode_scope_stats();
                check(hit.decodes == 0 && hit.late_snapshot_read_bytes == 0 &&
                      hit.decoder_snapshot_candidate_bytes == 0,
                      "unchanged source reuses its persistent decoded image");
            }
            ++source_iteration;
        }
    }
    unmap(guest, 0x10000, 0, 0, 0, 0);
    release(physical, 0x10000, 0, 0, 0, 0);
    std::printf("decoder source snapshot: %d failures\n", failures);
    return failures ? 1 : 0;
}
