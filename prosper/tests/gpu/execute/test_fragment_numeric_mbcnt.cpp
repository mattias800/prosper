// One covered pixel at quad position 3, with three real helper invocations. Finite MBCNT masks
// must count physical source bits at positions 0..2, not discard them as non-guest/helper lanes.
// A separate quad-broadcast witness proves this is an edge/helper test, not a fullscreen control.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <system_error>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t W = 8;
constexpr uint32_t H = 8;
struct Checks {
    int failures = 0;
    void check(bool ok, const char* name) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
        if (!ok) ++failures;
    }
};

std::vector<uint32_t> one_pixel_triangle() {
    SpirvCompute b;
    b.begin_vertex();
    const uint32_t vertex = b.vertex_invocation_id();
    const auto choose = [&](float a, float c, float d) {
        return b.sel(b.ucmp(Op_IEqual, vertex, b.uconst(0)), b.uconst(fbits(a)),
            b.sel(b.ucmp(Op_IEqual, vertex, b.uconst(1)), b.uconst(fbits(c)),
                  b.uconst(fbits(d))));
    };
    // Window-space vertices (1.5,1.1), (1.9,1.9), (1.1,1.9); only (1.5,1.5) is covered.
    b.export_position(choose(-0.625f, -0.525f, -0.725f),
                      choose(-0.725f, -0.525f, -0.525f),
                      b.uconst(0), b.uconst(fbits(1.0f)));
    return b.finish();
}

std::vector<uint32_t> helper_witness() {
    SpirvCompute b;
    b.begin_fragment();
    const uint32_t lane = b.subgroup_local_id();
    SpirvCompute::put(b.caps, Op_Capability, {Cap_GroupNonUniformQuad});
    const uint32_t helper = b.sel(b.helper_invocation(), b.uconst(1), b.uconst(0));
    uint32_t count = b.uconst(0);
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t other = b.id();
        // OpGroupNonUniformQuadBroadcast = 365. This witness is test-only and its caller
        // explicitly checks QUAD support; it does not claim production QUAD feature admission.
        SpirvCompute::put(b.code, 365, {b.t_u32, other, b.uconst(Scope_Subgroup), helper, b.uconst(i)});
        count = b.ibin(Op_IAdd, count, other);
    }
    const auto encode = [&](uint32_t value) {
        return b.fbin(Op_FMul, b.cvt_u2f(value), b.uconst(fbits(1.0f / 255.0f)));
    };
    b.export_color(0, encode(lane), encode(count), b.uconst(0), b.uconst(fbits(1.0f)));
    return b.finish();
}

std::vector<uint8_t> render(const std::vector<uint32_t>& vs,
                            const std::vector<uint32_t>& fs) {
    prosper::test::BackendDraw draw;
    draw.vs = vs;
    draw.fs = fs;
    draw.vcount = 3;
    draw.allow_partial_wave_fragment = true;
    for (uint32_t set = 0; set < 2; ++set) {
        prosper::test::FrameResource cb; cb.binding = 2; cb.set = set; draw.R.push_back(cb);
        prosper::test::FrameResource vb; vb.binding = 3; vb.set = set; draw.R.push_back(vb);
    }
    return prosper::test::render_draws_rgba({draw}, W, H);
}

bool clear(const uint8_t* p) { return p[0] == 0 && p[1] == 0 && p[2] == 255; }

void dump(const std::filesystem::path& directory, const std::string& name,
          const std::vector<uint32_t>& spirv) {
    if (directory.empty()) return;
    const auto path = directory / (name + ".spv");
    std::ofstream file(path, std::ios::binary);
    const auto bytes = std::as_bytes(std::span{spirv});
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::filesystem::filesystem_error(
        "SPIR-V dump failed", path, std::make_error_code(std::errc::io_error));
}

bool verify_helper_witness(Checks& checks, const std::vector<uint32_t>& vertex,
                           const std::vector<uint32_t>& witness) {
    const auto pixels = render(vertex, witness);
    checks.check(pixels.size() == W * H * 4, "helper witness renders a target");
    if (pixels.size() != W * H * 4) return false;
    unsigned covered = 0;
    for (unsigned i = 0; i < W * H; ++i) if (!clear(&pixels[i * 4])) ++covered;
    const unsigned pixel = (W + 1) * 4;
    checks.check(covered == 1 && !clear(&pixels[pixel]), "exactly pixel (1,1) is covered");
    checks.check((pixels[pixel] & 3) == 3 && pixels[pixel + 1] == 3,
                  "covered lane is quad position 3 and all three neighbors are helpers");
    return covered == 1 && pixels[pixel + 1] == 3;
}

std::vector<uint32_t> numeric_fragment(bool wave32, bool hi, unsigned index, uint32_t mask) {
    // The first five use inline integers; remaining words use literal numeric data.
    uint32_t source = 255;
    if (index < 4) source = 128u + mask;
    else if (index == 4) source = 194;
    std::vector code = {0xd7650007u, 0x000100c1u,
                        0xd7660007u, 0x00020ec1u}; // physical lane v7
    code.reserve(32);
    code.insert(code.end(), {hi ? 0xd766000au : 0xd765000au,
                             source | (133u << 9)}); // v10,source,5
    if (source == 255) code.push_back(mask);
    // The derivative forces helper execution in this independently recompiled shader too.
    // Its source (physical lane id) is linear within each quad; blue is exported but not checked.
    code.insert(code.end(), {
        0x7e100d07u, // v_cvt_f32_u32 v8,v7
        0x7e0402fau, 0xff000008u, // v_mov_b32_dpp v2,v8 quad_perm:[0,0,0,0]
        0x7e000d07u, 0x100000ffu, 0x3b808081u, // red = lane / 255
        0x7e020d0au, 0x100202ffu, 0x3b808081u, // green = count / 255
        0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
    });
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
        wave32, {RecompileDiagnosticStage::Fragment, 0xa3996000u + index});
}

void run_fragment_case(Checks& checks, const std::vector<uint32_t>& vertex,
                       const std::filesystem::path& directory, bool wave32, bool hi,
                       unsigned index, uint32_t mask) {
    const auto fragment = numeric_fragment(wave32, hi, index, mask);
    checks.check(!fragment.empty(), "finite physical mask fragment compiles");
    if (fragment.empty()) return;
    dump(directory, "numeric_w" + std::to_string(wave32 ? 32 : 64) +
        (hi ? "_hi_" : "_lo_") + std::to_string(index), fragment);
    const auto pixels = render(vertex, fragment);
    checks.check(pixels.size() == W * H * 4, "numeric mask shader renders a target");
    if (pixels.size() != W * H * 4) return;
    const unsigned pixel = (W + 1) * 4;
    const unsigned lane = pixels[pixel];
    // Fail visibly before forming a CPU prefix if the encoded lane itself is outside the guest wave.
    const unsigned guest_width = wave32 ? 32 : 64;
    checks.check(lane < guest_width, "reported physical lane is inside the guest wave");
    if (lane >= guest_width) return;
    unsigned width = std::min(lane, 32u);
    if (hi) width = lane > 32 ? lane - 32 : 0;
    const uint32_t prefix = width == 32 ? UINT32_MAX : (1u << width) - 1;
    const unsigned expected = 5 + std::popcount(mask & prefix);
    std::printf("wave=%u hi=%d mask=%08x lane=%u got=%u expected=%u\n",
        guest_width, hi, mask, lane, pixels[pixel + 1], expected);
    checks.check(!clear(&pixels[pixel]) && (lane & 3) == 3 && pixels[pixel + 1] == expected,
                  "MBCNT counts explicit helper/inactive source bits at the primitive edge");
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path dump_directory;
    if (argc == 3 && std::string(argv[1]) == "--dump-spv-dir") {
        dump_directory = argv[2];
        std::filesystem::create_directories(dump_directory);
    } else if (argc != 1) return 2;
    Checks checks;
    if (const auto& context = prosper::test::render_vk_ctx();
        !(context.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_QUAD_BIT)) {
        std::puts("SKIP: fragment quad helper witness is unavailable");
        return 77;
    }
    const auto vertex = one_pixel_triangle();
    const auto witness = helper_witness();
    dump(dump_directory, "vertex", vertex);
    dump(dump_directory, "helper_witness", witness);
    if (!verify_helper_witness(checks, vertex, witness)) return 1;
    const std::array masks = {0u, 1u, 3u, 15u, 0xfffffffeu, 0x80000001u, 0x55555555u};
    for (bool wave32 : {false, true})
        for (bool hi : {false, true})
            for (unsigned index = 0; index < masks.size(); ++index)
                run_fragment_case(checks, vertex, dump_directory, wave32, hi, index, masks[index]);
    return checks.failures ? 1 : 0;
}
