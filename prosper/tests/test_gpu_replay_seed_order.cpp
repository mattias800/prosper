#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "test_scratch.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace prosper::gpu;

namespace {

void unset_test_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int child_exit_code(int status) {
#ifdef _WIN32
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

bool read_file(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    bytes.assign(std::istreambuf_iterator<char>(stream), {});
    return stream.good() || stream.eof();
}

uint32_t read_le32(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

bool center_pixel_is_red(const std::filesystem::path& path, uint32_t width, uint32_t height) {
    std::vector<uint8_t> bmp;
    if (!read_file(path, bmp) || bmp.size() < 54 || bmp[0] != 'B' || bmp[1] != 'M' ||
        read_le32(bmp, 18) != width || read_le32(bmp, 22) != height)
        return false;
    const uint32_t offset = read_le32(bmp, 10);
    const uint32_t stride = (width * 3u + 3u) & ~3u;
    const size_t pixel = static_cast<size_t>(offset) +
        static_cast<size_t>(height - 1u - height / 2u) * stride +
        static_cast<size_t>(width / 2u) * 3u;
    if (pixel + 3u > bmp.size()) return false;
    const uint8_t blue = bmp[pixel];
    const uint8_t green = bmp[pixel + 1];
    const uint8_t red = bmp[pixel + 2];
    return red > 0xc0 && green < 0x40 && blue < 0x40;
}

} // namespace

int main(int argc, char** argv) {
    std::printf("== test_gpu_replay_seed_order ==\n");
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <gpu_replay>\n", argv[0]);
        return 1;
    }

    constexpr uint32_t width = 64, height = 64;
    constexpr uint64_t texture_addr = 0x100000;
    constexpr uint64_t target_addr = 0x200000;
    constexpr uint64_t vs_addr = 0x300000;
    constexpr uint64_t fs_addr = 0x310000;
    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u,
        0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u,
        0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    const uint32_t fs_rdna[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u,
        0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };

    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.format = DataFormat::Unorm8;
    texture.num_components = 4;
    texture.binding = 4;
    texture.img_dim = 1;
    texture.width = texture.height = 2;
    texture.size = 16;
    texture.sgpr_base = 8;
    texture.gpu_addr = texture_addr;
    texture.linear_row_pitch_bytes = 8;
    texture.mag_filter = texture.min_filter = 0;
    ShaderResourceTable resources;
    resources.resources.push_back(texture);

    DrawItem draw;
    draw.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
    draw.fs = recompile_fragment(fs_rdna, std::size(fs_rdna), &resources);
    draw.vs_guest_addr = vs_addr;
    draw.fs_guest_addr = fs_addr;
    draw.prt = std::make_shared<ShaderResourceTable>(resources);
    draw.vertex_count = draw.raw_draw_count = 3;
    draw.ps.topology = 3;
    draw.ps.color_write_mask = 0xf;
    draw.color0_base = target_addr;
    draw.color0_width = width;
    draw.color0_height = height;
    draw.color_targets[0] = {target_addr, width, height};
    if (draw.vs.empty() || draw.fs.empty()) {
        std::fprintf(stderr, "fixture shaders did not compile\n");
        return 1;
    }

    const std::array<uint8_t, 16> stale_texture{};
    auto read_span = [&](uint64_t base, const uint8_t* source, size_t source_size,
                         uint64_t addr, uint8_t* destination, size_t bytes) -> size_t {
        if (addr < base || addr - base >= source_size) return 0;
        const size_t offset = static_cast<size_t>(addr - base);
        const size_t count = std::min(bytes, source_size - offset);
        std::memcpy(destination, source + offset, count);
        return count;
    };
    auto reader = [&](uint64_t addr, uint8_t* destination, size_t bytes) -> size_t {
        size_t read = read_span(texture_addr, stale_texture.data(), stale_texture.size(),
                                addr, destination, bytes);
        if (read) return read;
        read = read_span(vs_addr, reinterpret_cast<const uint8_t*>(vs_rdna), sizeof(vs_rdna),
                         addr, destination, bytes);
        if (read) return read;
        return read_span(fs_addr, reinterpret_cast<const uint8_t*>(fs_rdna), sizeof(fs_rdna),
                         addr, destination, bytes);
    };

    GpuCaptureMetadata metadata;
    metadata.width = width;
    metadata.height = height;
    metadata.submit_index = 1;
    metadata.revision = "seed-order-test";
    GpuCaptureFile capture;
    std::string error;
    if (!capture_draw_items({draw}, metadata, reader, capture, error)) {
        std::fprintf(stderr, "cannot create fixture: %s\n", error.c_str());
        return 1;
    }

    GpuCaptureRttSeed rtt;
    rtt.guest_addr = texture_addr;
    rtt.width = rtt.height = 2;
    rtt.rgba = {
        255, 0, 0, 255, 255, 0, 0, 255,
        255, 0, 0, 255, 255, 0, 0, 255,
    };
    capture.rtt_seeds.push_back(std::move(rtt));
    GpuCaptureDsSeed ds;
    ds.depth_read_base = ds.depth_write_base = 0x400000;
    ds.width = ds.height = 2;
    ds.depth_valid = true;
    ds.depth.resize(2u * 2u * sizeof(float));
    capture.ds_seeds.push_back(std::move(ds));

    const auto capsule_path = prosper_test::test_scratch_path("seeded-raw.prgcap");
    const auto output_path = prosper_test::test_scratch_path("seeded-raw.bmp");
    const auto log_path = prosper_test::test_scratch_path("seeded-raw.log");
    if (!write_gpu_capture(capsule_path.string(), capture, error)) {
        std::fprintf(stderr, "cannot write fixture: %s\n", error.c_str());
        return 1;
    }

    // The regression must not inherit the workaround that made the broken ordering appear green.
    unset_test_env("PROSPER_GPU_REPLAY_RTT_SEEDS");
    unset_test_env("PROSPER_GPU_REPLAY_DS_SEEDS");
    const std::string command = quote(argv[1]) + " --recompile-raw " + quote(capsule_path) +
        " " + quote(output_path) + " > " + quote(log_path) + " 2>&1";
    const int exit_code = child_exit_code(std::system(command.c_str()));
    std::vector<uint8_t> log_bytes;
    read_file(log_path, log_bytes);
    const std::string log(log_bytes.begin(), log_bytes.end());
    const bool substituted = log.find("[recompile-raw] substituted vs=1 fs=1") != std::string::npos;
    const bool seeded = log.find("RTT-seeds=1 DS-seeds=1") != std::string::npos;
    const bool executed = log.find("[gpureplay] output=64x64") != std::string::npos;
    const bool restore_failed = log.find("cannot restore") != std::string::npos;
    const bool red = center_pixel_is_red(output_path, width, height);
    if (exit_code != 0 || !substituted || !seeded || !executed || restore_failed || !red) {
        std::fprintf(stderr,
                     "[FAIL] seeded --recompile-raw must restore RTT/DS and reach renderer output "
                     "(exit=%d substituted=%d seeded=%d executed=%d restore-failed=%d red=%d)\n%s",
                     exit_code, substituted, seeded, executed, restore_failed, red, log.c_str());
        return 1;
    }
    std::printf("  [ok] seeded --recompile-raw restores RTT/DS and reaches renderer output\n");
    return 0;
}
