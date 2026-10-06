// #4230: project-owned original guest programs expose the three pull words through raw EXP.
// Positive finite clip W and an interior triangle isolate linear planes from clipping/sample rules.
// This is a native producing-draw regression, not PS5 initial-lane or logical-Wave64 authority.
#pragma once
#include "gpu/execute/gpu_execute.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace prosper::test::ps_pull {
namespace g = prosper::gpu;
namespace p = prosper::agc::Pm4;
inline constexpr uint32_t width = 16, height = 16;
enum class Words { Pull, CenterAndLinear, LinearAfterReservedPull };
enum class ClipW { One, Two, Varying };
struct Case {
    const char* name;
    ClipW w;
    Words words;
    uint32_t ena, addr;
    uint32_t first_vgpr;
};
inline constexpr std::array cases{
    Case{"pull_unit_w", ClipW::One, Words::Pull, 8, 8, 0},
    Case{"pull_nonunit_w", ClipW::Two, Words::Pull, 8, 8, 0},
    Case{"pull_varying_w", ClipW::Varying, Words::Pull, 8, 8, 0},
    Case{"pull_after_reserved_sample", ClipW::Varying, Words::Pull, 8, 9, 2},
    Case{"center_and_linear", ClipW::Varying, Words::CenterAndLinear, 0x22, 0x22, 0},
    Case{"linear_after_reserved_pull", ClipW::Varying, Words::LinearAfterReservedPull, 0x20, 0x28,
         3},
};

inline std::vector<uint32_t> vertex_words(ClipW w) {
    // NDC vertices (-.75,-.75), (.75,-.75), (-.75,.75), with exact dyadic screen positions.
    std::vector<uint32_t> code{
        0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02ffu, 0x3fc00000u, 0x7e0c02ffu,
        0x3f400000u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u, 0x7e060280u,
    };
    if (w == ClipW::Varying) {
        // v4 = float(vertexIndex) + 1, hence clip W = (1,2,3); not a masked bit trick.
        code.insert(code.end(), {0x7e080d00u, 0x060808f2u});
    } else {
        code.push_back(w == ClipW::One ? 0x7e0802f2u : 0x7e0802f4u);
    }
    code.insert(code.end(), {
                                0x10020304u,
                                0x10040504u,   // x,y *= W before the original POS0 export
                                0xf80008cfu,
                                0x04030201u,
                                0x7e1402f2u,
                                0xf800020fu,
                                0x0a030305u,   // genuine PARAM0=(1.5,0,0,1)
                                0xbf810000u,
                            });
    return code;
}

inline std::vector<uint32_t> fragment_words(const Case& c) {
    std::vector<uint32_t> code;
    if (c.words != Words::Pull)
        code.push_back(
            0xc8620000u);   // real P10 attr0.x -> v24 requests the existing geometry path
    const uint32_t n = c.first_vgpr;
    code.insert(code.end(), {0x7e100300u | n, 0x7e120300u | (n + 1)});   // MOV v8/v9
    if (c.words == Words::LinearAfterReservedPull)
        code.push_back(0x7e140280u);   // v10=0, no read of disabled pull input
    else
        code.push_back(0x7e140300u | (n + 2));
    code.push_back(c.words == Words::CenterAndLinear ? (0x7e160300u | (n + 3)) : 0x7e1602f2u);
    code.insert(code.end(), {0xf800180fu, 0x0b0a0908u, 0xbf810000u});
    return code;
}

struct Program {
    alignas(256) std::array<uint32_t, 64> code{};
    g::AgcShaderUserData user{};
    std::array<g::ShaderReg, 2> registers{};
    g::AgcShaderHeader header{};
};
struct Owners {
    Program vs, ps;
};

inline bool register_program(Program& out, bool vertex, const std::vector<uint32_t>& raw) {
    if (raw.size() > out.code.size()) return false;
    std::copy(raw.begin(), raw.end(), out.code.begin());
    out.registers[0].offset = vertex ? p::SPI_SHADER_PGM_LO_ES : p::SPI_SHADER_PGM_LO_PS;
    out.registers[1].offset = vertex ? p::SPI_SHADER_PGM_HI_ES : p::SPI_SHADER_PGM_HI_PS;
    out.header.file_header = 0x34333231u;
    out.header.version = 0x18u;
    out.header.user_data = &out.user;
    out.header.sh_registers = out.registers.data();
    out.header.shader_size = static_cast<uint32_t>(raw.size() * sizeof(uint32_t));
    out.header.type = vertex ? 2u : 1u;
    out.header.num_sh_registers = 2;
    void* registered = nullptr;
    const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
    const uint64_t address = reinterpret_cast<uint64_t>(out.code.data());
    return create &&
           create(reinterpret_cast<uint64_t>(&registered), reinterpret_cast<uint64_t>(&out.header),
                  address, 0, 0, 0) == 0 &&
           registered == &out.header && out.header.user_data == &out.user &&
           out.registers[0].value == uint32_t(address >> 8) &&
           out.registers[1].value == uint32_t((address >> 40) & 255);
}

inline bool realize(const Case& c, g::DrawItem& draw,
                    std::optional<uint32_t> ps_rsrc1 = std::nullopt) {
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    constexpr uint64_t bytes = 0x10000;
    static_assert(sizeof(Owners) <= bytes);
    uint64_t address = 0;
    if (!map ||
        map(reinterpret_cast<uint64_t>(&address), bytes, 2, 0,
            reinterpret_cast<uint64_t>("ps-pull-model-programs"), 0) != 0 ||
        address < 0x100000000ull || address % alignof(Owners))
        return false;
    // AGC interprets low pointers as offsets and retains this owner for process lifetime.
    // Do not release/rewrite it after registration; each rail owns a distinct bounded mapping.
    auto& owner = *std::construct_at(reinterpret_cast<Owners*>(address));
    if (!register_program(owner.vs, true, vertex_words(c.w)) ||
        !register_program(owner.ps, false, fragment_words(c)))
        return false;
    g::GpuState state;
    for (const auto* program : {&owner.vs, &owner.ps})
        for (const auto& reg : program->registers) state.sh[reg.offset] = reg.value;
    state.uc[p::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[p::CB_TARGET_MASK] = state.cx[p::CB_SHADER_MASK] = 15;
    state.cx[p::SPI_PS_IN_CONTROL] = 0;
    state.cx[p::SPI_BARYC_CNTL] = 0;
    state.cx[p::SPI_PS_INPUT_ENA] = c.ena;
    state.cx[p::SPI_PS_INPUT_ADDR] = c.addr;
    // Only callers supplying a physical mode get that observation. Existing native pull
    // fixtures intentionally retain their absent RSRC1; do not stamp a consumer-side default.
    if (ps_rsrc1) state.sh[p::SPI_SHADER_PGM_RSRC1_PS] = *ps_rsrc1;
    state.sh[p::SPI_SHADER_PGM_RSRC2_PS] = 0;
    if (c.words != Words::Pull) state.cx[p::SPI_PS_INPUT_CNTL_0] = 0;
    g::GpuState::Draw packet;
    packet.index_count = 3;
    packet.instance_count = 1;
    packet.command_order = 1;
    state.draws.push_back(packet);
    return g::realize_draw_item(state, &state.draws[0], 3, 64, false, draw, nullptr, true);
}

// Primary AMD LLPC 40cb8d95 LowerInOut.cpp:5481 consumes <I/W,J/W,1/W> and divides its
// first two words by its third. Vulkan triangle rasterization supplies screen-affine lambda.
// Independently derive values from known vertices; never read prepared/generated plane values.
inline std::array<double, 4> expected(const Case& c, uint32_t x, uint32_t y) {
    const double i = (double(x) + .5 - 2) / 12;
    const double j = (double(y) + .5 - 2) / 12;
    const std::array<double, 3> w = c.w == ClipW::Varying ? std::array{1., 2., 3.}
                                    : c.w == ClipW::Two   ? std::array{2., 2., 2.}
                                                          : std::array{1., 1., 1.};
    const double r = (1 - i - j) / w[0] + i / w[1] + j / w[2];
    if (c.words == Words::Pull) return {i / w[1], j / w[2], r, 1};
    if (c.words == Words::CenterAndLinear) return {i / w[1] / r, j / w[2] / r, i, j};
    return {i, j, 0, 1};
}

inline bool matches(const std::array<float, 4>& actual, const std::array<double, 4>& want) {
    // Host raster arithmetic tolerance, not a bit-exact PS5 interpolation/FP oracle.
    for (size_t component = 0; component < actual.size(); ++component)
        if (!std::isfinite(actual[component]) ||
            std::abs(double(actual[component]) - want[component]) > 1e-5)
            return false;
    return true;
}
}   // namespace prosper::test::ps_pull
