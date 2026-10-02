// Real AGC registration -> DrawItem -> registered shipping backend -> private input-only pass.
// --cpu-only verifies producing ownership without creating Vulkan. GPU records are host raster
// observations, NOT initialized guest registers, PS5 Wave64 packing or post-depth visibility.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
static uint32_t checks = 0, failures = 0;
static void check(bool ok, const char* label) {
    ++checks; failures += !ok;
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", label); std::fflush(stdout);
}
static constexpr uint32_t W = 9, H = 7;
static constexpr uint32_t VS[]{0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,
    0x7e0a02f6u,0x7e0c02f2u,0x10020b01u,0x08020d01u,0x10040b02u,0x08040d02u,
    0x7e060280u,0x7e0802f2u,0xf80008cfu,0x04030201u,0xbf810000u};
static constexpr uint32_t PS[]{0xd7600014u,256u | (168u << 9),0x7e080214u,
    0xf800180fu,0x04040404u,0xbf810000u}; // READLANE40 requires genuine Wave64.
// Same fullscreen coordinates, plus PARAM0=(float(vertexIndex),0,0,1). The two
// indexed primitives both use vertices0,1,2, so their independently known values are identical.
static constexpr uint32_t VARYING_VS[]{0x7e140d00u,
    0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,
    0x7e0a02f6u,0x7e0c02f2u,0x10020b01u,0x08020d01u,0x10040b02u,0x08040d02u,
    0x7e060280u,0x7e0802f2u,0xf80008cfu,0x04030201u,
    0xf800020fu,0x0403030au,0xbf810000u};
static constexpr uint32_t SMOOTH_PS[]{0xc8000000u,0xc8010001u,
    0xd7600014u,256u | (168u << 9),0x7e080214u,
    0xf800180fu,0x04040404u,0xbf810000u};
static constexpr uint32_t PARAMETER_PS[]{0xc80e0000u,0xc8120001u,0xc8160002u,
    0xd54b0003u,0x04160103u,0xd54b0003u,0x040e0304u,
    0xd7600014u,259u | (168u << 9),0x7e060214u,
    0x7e080280u,0x7e0a0280u,0x7e0c02f2u,0xf800000fu,0x06050403u,0xbf810000u};
struct Program {
    alignas(256) std::array<uint32_t, 64> code{};
    AgcShaderUserData user{};
    std::array<ShaderReg, 2> registers{};
    AgcShaderHeader header{};
};
struct Owners {
    Program vs, ps, varying_vs, smooth_ps, parameter_ps;
    std::array<uint32_t, 6> indices{0,1,2,0,1,2};
};
static bool register_program(Program& p, bool vertex, const uint32_t* raw, size_t count) {
    std::copy(raw, raw + count, p.code.begin());
    p.registers[0].offset = vertex ? P::SPI_SHADER_PGM_LO_ES : P::SPI_SHADER_PGM_LO_PS;
    p.registers[1].offset = vertex ? P::SPI_SHADER_PGM_HI_ES : P::SPI_SHADER_PGM_HI_PS;
    p.header.file_header = 0x34333231u; p.header.version = 0x18u;
    p.header.user_data = &p.user; p.header.sh_registers = p.registers.data();
    p.header.shader_size = static_cast<uint32_t>(count * 4);
    p.header.type = vertex ? 2u : 1u; p.header.num_sh_registers = 2;
    void* registered = nullptr;
    const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
    const bool ok = create && create(reinterpret_cast<uint64_t>(&registered),
        reinterpret_cast<uint64_t>(&p.header), reinterpret_cast<uint64_t>(p.code.data()),0,0,0) == 0;
    const uint64_t address = reinterpret_cast<uint64_t>(p.code.data());
    check(ok && registered == &p.header && p.header.user_data == &p.user &&
        p.registers[0].value == uint32_t(address >> 8) &&
        p.registers[1].value == uint32_t((address >> 40) & 255),
        vertex ? "real VS registration retains exact code/header" : "real PS registration retains exact code/header");
    return ok;
}
static GpuState state_for(const Owners& o, const Program* vertex = nullptr,
                          const Program* fragment = nullptr, bool varying = false) {
    GpuState state;
    for (const Program* p : {vertex ? vertex : &o.vs, fragment ? fragment : &o.ps})
        for (const auto& reg : p->registers) state.sh[reg.offset] = reg.value;
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = state.cx[P::CB_SHADER_MASK] = 15;
    state.cx[P::SPI_PS_IN_CONTROL] = 0; // known Wave64, not inferred from effective SPIR-V
    state.cx[P::SPI_BARYC_CNTL] = 0xa5000000u; // full unused bits must not disappear
    state.cx[P::SPI_PS_INPUT_ENA] = state.cx[P::SPI_PS_INPUT_ADDR] = varying ? 2u : 0x300u;
    if (varying) state.cx[P::SPI_PS_INPUT_CNTL_0] = 0; // actual attr0 -> producer PARAM0, not metadata guess
    state.index_type = 1; state.index_type_announced = true;
    GpuState::Draw draw;
    draw.indexed = true; draw.index_addr = reinterpret_cast<uint64_t>(o.indices.data());
    draw.index_count = 6; draw.instance_count = 1; draw.command_order = 23;
    state.draws.push_back(draw); return state;
}
static void descriptor_limit_controls() {
    using prosper::test::raster_quad_descriptor_limits;
    VkPhysicalDeviceLimits limits{};
    limits.maxBoundDescriptorSets = 2;
    limits.maxPerStageDescriptorStorageBuffers = 2;
    limits.maxPerStageResources = 3;
    limits.maxDescriptorSetStorageBuffers = 3;
    const auto refused = [&](const VkPhysicalDeviceLimits& candidate, uint64_t count, const char* reason) {
        const char* got = raster_quad_descriptor_limits(candidate,count);
        return got && std::strcmp(got,reason) == 0;
    };
    check(!raster_quad_descriptor_limits(limits,2), "VS2 plus fragment1 fit asymmetric measured descriptor limits");
    check(!raster_quad_descriptor_limits(limits,0), "descriptor-free VS still counts the fragment collector");
    auto bad = limits; bad.maxPerStageDescriptorStorageBuffers = 1;
    check(refused(bad,2,"quad-collector-per-stage-storage-limit"),
        "actual VS stage storage limit, not aggregate capacity, rejects");
    bad = limits; bad.maxPerStageResources = 1;
    check(refused(bad,2,"quad-collector-per-stage-resource-limit"),
        "actual per-stage resource inventory rejects independently");
    bad = limits; bad.maxDescriptorSetStorageBuffers = 2;
    check(refused(bad,2,"quad-collector-aggregate-storage-limit"),
        "VS2 plus fragment1 aggregate across BOTH sets rejects");
    bad = limits; bad.maxPerStageDescriptorStorageBuffers = 0;
    check(refused(bad,0,"quad-collector-per-stage-storage-limit"),
        "fragment1 limit is enforced even when VS has no descriptors");
    bad = limits; bad.maxPerStageResources = 0;
    check(refused(bad,0,"quad-collector-per-stage-resource-limit"),
        "fragment1 stage resources are counted independently");
    bad = limits; bad.maxBoundDescriptorSets = 1;
    check(refused(bad,0,"quad-collector-descriptor-set-limit"),
        "actual two-set pipeline limit rejects before Vulkan creation");
    check(raster_quad_descriptor_limits(limits,UINT64_MAX) != nullptr,
        "oversized reflected inventory cannot wrap the aggregate count");
}
static prosper::test::BackendDraw backend_contract(const DrawItem& draw) {
    prosper::test::BackendDraw b;
    b.vs = draw.vs; b.gs = draw.gs; b.fs = draw.fs;
    b.vs_shared = draw.vs_shared; b.fs_shared = draw.fs_shared;
    b.raster_quads = draw.raster_quads; return b;
}
static void ownership_controls(DrawItem& draw, Owners& owner) {
    const auto sink = draw.raster_quads;
    check(sink && sink->inputs && sink->inputs->source_vs && sink->inputs->source_gs &&
        sink->inputs->source_fs && *sink->inputs->source_vs == draw.vs_words() &&
        *sink->inputs->source_fs == draw.fs_words(), "realization pins exact selected SOURCE modules");
    if (!sink || !sink->inputs || !sink->inputs->raw_code) { check(false, "realization-owned PS analysis required"); return; }
    const auto raw = sink->inputs->raw_code;
    const auto old = *raw;
    check(old.size() >= std::size(PS) && std::equal(std::begin(PS),std::end(PS),old.begin()),
        "raw PS belongs to actual producing analysis version");
    owner.ps.code[2] ^= 1;
    notify_guest_gpu_write(reinterpret_cast<uint64_t>(owner.ps.code.data()), owner.ps.header.shader_size);
    const auto changed = shader_analysis_owned_words(acquire_shader_analysis(owner.ps.code.data(), std::size(PS)));
    check(changed && *changed != old && *raw == old, "same-address new guest bytes cannot mutate retained producer");
    owner.ps.code[2] ^= 1;
    notify_guest_gpu_write(reinterpret_cast<uint64_t>(owner.ps.code.data()), owner.ps.header.shader_size);
    clear_shader_analysis_cache(); clear_shader_recompile_cache();
    check(*raw == old && *sink->inputs->source_fs == draw.fs_words(), "cache clear retains owned raw and SOURCE versions");
    auto b = backend_contract(draw);
    check(prosper::test::raster_quad_producing_modules_match(b), "real producing join accepts matching modules");
    b.fs_shared = sink->inputs->source_fs; b.fs = {0xdeadbeefu};
    check(prosper::test::raster_quad_producing_modules_match(b), "selected shared SOURCE wins over conflicting owned vector");
    b.set_fs({0xdeadbeefu});
    check(!prosper::test::raster_quad_producing_modules_match(b), "actual shader substitution invalidates producer join");
    b = backend_contract(draw); b.raster_quad_contract_modified = true;
    check(!prosper::test::raster_quad_producing_modules_match(b), "diagnostic override never borrows producing authority");
    check(prosper::test::raster_quad_pre_raster_readonly(draw.vs_words()), "actual selected VS has complete replay-purity proof");
    auto unsupported = draw.vs_words();
    unsupported.insert(unsupported.begin() + 5, {(2u << 16) | 17u, 11u}); // Int64 capability outside slice
    check(!prosper::test::raster_quad_pre_raster_readonly(unsupported), "unsupported pre-raster capabilities decline");
    auto xfb = draw.vs_words();
    xfb.insert(xfb.begin() + 5, {(2u << 16) | 17u, 53u}); // TransformFeedback
    check(!prosper::test::raster_quad_pre_raster_readonly(xfb), "XFB cannot execute twice in a collector pass");
    auto position_extra = draw.vs_words();
    bool found_position = false;
    for (size_t p = 5; p < position_extra.size();) {
        const auto n = position_extra[p] >> 16, op = position_extra[p] & 65535;
        if (!n || p + n > position_extra.size()) break;
        if (op == 72 && n == 5 && position_extra[p + 3] == 11 && position_extra[p + 4] == 0) {
            position_extra[p + 4] = 3; found_position = true; break; // CullDistance
        }
        if (op == 71 && n == 4 && position_extra[p + 2] == 11 && position_extra[p + 3] == 0) {
            position_extra[p + 3] = 3; found_position = true; break;
        }
        p += n;
    }
    check(found_position && !prosper::test::raster_quad_pre_raster_readonly(position_extra),
        "unsupported pre-raster builtin cannot be omitted by GS substitution");
    RasterQuadCollector attrs; attrs.fields.push_back({RasterQuadFieldKind::Interpolant,0,0,4});
    check(!prosper::test::raster_quad_varying_interface(draw.vs_words(), *sink->inputs, attrs),
        "declared FS varying without producing interface declines");
    const auto& launch = sink->inputs->launch;
    check(launch.ps_in_control_available && launch.ps_in_control == 0 && launch.baryc_cntl_available &&
        launch.baryc_cntl == 0xa5000000u && launch.input_ena_available && launch.input_ena == 0x300u &&
        launch.input_addr_available && launch.input_addr == 0x300u,
        "real DrawItem retains full known SPI launch bits, including known zero");
}
static RasterQuadResult observe(DrawItem draw, uint32_t budget) {
    static uint64_t identity = 0x7d410600u;
    draw.color0_base = identity; identity += 0x10000;
    draw.color0_width = W; draw.color0_height = H;
    draw.color_targets[0].mirror_named_identity(draw.color0_base, W, H);
    draw.raster_quads->max_quads = budget;
    (void)render_submit_items({draw}, W, H); // actual registered caller, not direct collector invocation
    std::lock_guard lock(draw.raster_quads->mutex); return draw.raster_quads->result;
}
static void observe_complete(const RasterQuadResult& result) {
    check(result.attempted && result.complete && result.rejection.empty() && !result.quads.empty(),
        "registered DrawItem refusal route completed real scratch collection");
    check(result.host_raster_domain_available && result.host_sample_count == 1 && result.host_sample_index == 0 &&
        result.host_layer == 0 && result.host_view_index == 0 && result.host_instance_count == 1 &&
        result.host_first_instance == 0 && !result.guest_export_eligibility_available,
        "actual host domain is explicit; guest export/depth visibility stays unavailable");
    std::set<std::array<uint32_t,3>> covered;
    bool helper = false, geometry_ok = true;
    for (const auto& q : result.quads) for (uint32_t lane = 0; lane < 4; ++lane) {
        const auto* value = q.data() + lane * result.lane_words;
        helper |= value[0] != 0;
        if (value[0]) { check(value[1] == 0, "real helper coverage remains unavailable"); continue; }
        const float fx = std::bit_cast<float>(value[5]), fy = std::bit_cast<float>(value[6]);
        geometry_ok &= fx >= 0.5f && fx < W && fy >= 0.5f && fy < H && value[4] < 2 &&
            value[1] == 1 && value[2] == 1;
        covered.insert({value[4],uint32_t(fx),uint32_t(fy)});
    }
    check(helper && geometry_ok, "odd extent retains partial helper quads and exact pre-depth covered inputs");
    check(covered.size() == W * H * 2u, "overlapping primitives each retain every real covered pixel");
    for (uint32_t primitive = 0; primitive < 2; ++primitive)
        for (uint32_t y = 0; y < H; ++y) for (uint32_t x = 0; x < W; ++x)
            check(covered.count({primitive,x,y}) == 1, "raw primitive/input sink, independent of append order");
    for (const auto& field : result.fields)
        check(!field.guest_initialization_proved, "observed varying never becomes MUST-defined guest input");
}
static void observe_capacity(const Owners& owners) {
    // Never infer the next draw's helper-scope population from a previous append count. These
    // two primitives each cover only (.5,.5) under the explicit guest scissor. In the supported
    // nonhelper-unique domain, each must publish exactly one separate helper-backed scope.
    auto state = state_for(owners);
    state.cx[P::PA_SC_GENERIC_SCISSOR_TL] = 0x80000000u; // disable window offset; x=y=0
    state.cx[P::PA_SC_GENERIC_SCISSOR_BR] = 0x00010001u; // exclusive right=bottom=1
    DrawItem draw;
    const bool realized = realize_draw_item(state,&state.draws[0],6,64,false,draw,nullptr,true);
    check(realized && draw.raster_quads && draw.ps.has_scissor &&
        draw.ps.scissor_left == 0 && draw.ps.scissor_top == 0 &&
        draw.ps.scissor_right == 1 && draw.ps.scissor_bottom == 1 &&
        draw.raster_quads->inputs && draw.raster_quads->inputs->raw_matches_producing_source &&
        draw.raster_quads->inputs->source_fs &&
        *draw.raster_quads->inputs->source_fs == draw.fs_words(),
        "capacity fixture freshly realizes owned producing SOURCE and its actual guest scissor");
    if (!realized || !draw.raster_quads) return;
    const auto exact = observe(draw,2);
    std::set<uint32_t> primitives;
    bool valid = exact.complete && exact.rejection.empty() && exact.quads.size() == 2;
    uint32_t covered = 0;
    for (const auto& quad : exact.quads) for (uint32_t lane = 0; lane < 4; ++lane) {
        const auto* value = quad.data() + lane * exact.lane_words;
        if (value[0]) { valid &= value[1] == 0; continue; }
        ++covered;
        valid &= value[1] == 1 && value[2] == 1 && value[4] < 2 &&
            value[5] == std::bit_cast<uint32_t>(0.5f) && value[6] == std::bit_cast<uint32_t>(0.5f) &&
            primitives.insert(value[4]).second;
    }
    check(valid && covered == 2 && primitives.size() == 2,
        "actual exact-capacity CAS retains the two independently known single-pixel primitive scopes");
    const auto overflow = observe(draw,1);
    check(overflow.attempted && !overflow.complete && overflow.quads.empty() &&
        overflow.rejection == "quad-collector-output-overflow",
        "actual capacity+1 overflow discards whole record transaction");
}
static void observe_fields(const RasterQuadResult& result, bool parameter) {
    // This is a host-input oracle, not a guest interpolation/register oracle. Independently:
    // clip vertices(-1,-1),(3,-1),(-1,3), w=1 map to (0,0),(2W,0),(0,2H).
    // Thus center barycentrics are I=x/(2W), J=y/(2H); PARAM0.x=I+2J.
    // Explicit flat parameter vectors are P10=(1,0,0,0), P20=(2,0,0,0), P0=(0,0,0,1).
    uint32_t parameters = 0, interpolants = 0, systems = 0;
    for (const auto& field : result.fields) {
        parameters += field.kind == RasterQuadFieldKind::Parameter;
        interpolants += field.kind == RasterQuadFieldKind::Interpolant;
        systems += field.kind == RasterQuadFieldKind::SystemInterpolation;
    }
    check(parameter ? (parameters == 3 && systems == 1 && interpolants == 0) :
        (interpolants == 1 && parameters == 0 && systems == 0),
        "actual collected field inventory is consumed, not empty position-only output");
    check(parameter ? bool(result.geometry_source && !result.geometry_source->empty()) : !result.geometry_source,
        "explicit fields use the actual regenerated GS; smooth fields do not synthesize one");
    uint32_t live_words = 0;
    bool values_match = true, domain_match = true;
    for (const auto& q : result.quads) for (uint32_t lane = 0; lane < 4; ++lane) {
        const auto* values = q.data() + lane * result.lane_words;
        if (values[0]) continue; // retain helpers, but do not invent numeric authority for their inputs
        const double i = double(std::bit_cast<float>(values[5])) / (2.0 * W);
        const double j = double(std::bit_cast<float>(values[6])) / (2.0 * H);
        uint32_t offset = kRasterQuadLaneFixedWords;
        for (const auto& field : result.fields) {
            if (field.kind == RasterQuadFieldKind::Parameter) {
                domain_match &= field.index == 0 && field.selector < 3 && field.words == 4;
                for (uint32_t component = 0; component < field.words; ++component) {
                    const float expected = component == 0 ? (field.selector == 0 ? 1.f : field.selector == 1 ? 2.f : 0.f)
                        : component == 3 && field.selector == 2 ? 1.f : 0.f;
                    values_match &= values[offset + component] == std::bit_cast<uint32_t>(expected);
                    ++live_words;
                }
            } else if (field.kind == RasterQuadFieldKind::SystemInterpolation) {
                domain_match &= field.index == 1 && field.words == 2;
                for (uint32_t component = 0; component < field.words; ++component) {
                    const double expected = component == 0 ? i : j;
                    const double observed = std::bit_cast<float>(values[offset + component]);
                    values_match &= std::isfinite(observed) && std::abs(observed - expected) < 0.00001;
                    ++live_words;
                }
            } else {
                domain_match &= field.index == 0 && field.words == 4;
                for (uint32_t component = 0; component < field.words; ++component) {
                    const double observed = std::bit_cast<float>(values[offset + component]);
                    const double expected = component == 0 ? i + 2 * j : component == 3 ? 1.0 : 0.0;
                    values_match &= std::isfinite(observed) && std::abs(observed - expected) < 0.00001;
                    ++live_words;
                }
            }
            offset += field.words;
        }
        domain_match &= offset == result.lane_words;
    }
    check(domain_match && values_match && live_words == W * H * 2u * (parameter ? 14u : 4u),
        "all actual covered raw field words match the independent vertex/barycentric oracle");
    std::printf("[raster-quad-fields] route=%s covered-raw-words=%u\n", parameter ? "parameter+system" : "smooth",live_words);
}
static void dump_sources(const std::filesystem::path& directory, const char* route,
                         const DrawItem& draw, const RasterQuadResult* result = nullptr) {
    if (directory.empty()) return;
    const auto dump = [&](const char* stage, const std::vector<uint32_t>& words) {
        if (words.empty()) return;
        std::ofstream file(directory / (std::string(route) + "_" + stage + ".spv"),std::ios::binary);
        file.write(reinterpret_cast<const char*>(words.data()),words.size() * sizeof(uint32_t));
        check(bool(file), "actual producing SOURCE dump written");
    };
    dump("vs",draw.vs_words()); dump("fs",draw.fs_words()); dump("gs_original",draw.gs);
    if (result && result->complete) {
        if (result->geometry_source) dump("gs_publisher",*result->geometry_source);
        if (result->collector_source) dump("collector",*result->collector_source);
    }
}
int main(int argc, char** argv) {
    bool cpu = false, missing_analysis = false, scissor_override = false;
    std::filesystem::path dump_directory;
    for (int arg = 1; arg < argc; ++arg) {
        if (!std::strcmp(argv[arg],"--cpu-only")) cpu = true;
        else if (!std::strcmp(argv[arg],"--analysis-unavailable")) missing_analysis = true;
        else if (!std::strcmp(argv[arg],"--scissor-override")) scissor_override = true;
        else if (!std::strcmp(argv[arg],"--dump-directory") && arg + 1 < argc) dump_directory = argv[++arg];
        else return 2;
    }
    if (missing_analysis && !cpu) return 2;
    if (!dump_directory.empty()) std::filesystem::create_directories(dump_directory);
    // CTest must pin activation before any cached environment read; do not setenv inside this test.
    if (!PROSPER_ENV_ON("PROSPER_FRAGMENT_QUAD_COLLECT")) return 2;
    check(prosper::test::raster_quad_raster_state_overridden() == scissor_override,
        scissor_override ? "present IGNORE_EMPTY_SCISSOR=0 is an override, not disabled" :
                           "default raster override contract is absent");
    descriptor_limit_controls();
    if (!cpu) {
        const auto& ctx = prosper::test::render_vk_ctx();
        if (!ctx.ok) return 1;
        if (!ctx.float_transport.explicit_nonfinite32() || !ctx.fragment_stores_atomics ||
            !ctx.geometry_shader_enabled || !(ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
            !(ctx.subgroup_operations & VK_SUBGROUP_FEATURE_QUAD_BIT) ||
            !(ctx.subgroup_operations & VK_SUBGROUP_FEATURE_BASIC_BIT)) {
            std::puts("SKIP: collector's actual enabled device contract unavailable"); return 77;
        }
        const bool native64 = ctx.subgroup_size_control &&
            (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
            ctx.min_subgroup_size <= 64 && ctx.max_subgroup_size >= 64;
        if (native64) { std::puts("SKIP: no native64 refusal route on this device"); return 77; }
        prosper::frontend::register_live_renderer(".", false);
    }
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    constexpr uint64_t bytes = 0x10000; static_assert(sizeof(Owners) <= bytes);
    uint64_t address = 0;
    check(map && map(reinterpret_cast<uint64_t>(&address),bytes,2,0,
        reinterpret_cast<uint64_t>("raster-quad-programs"),0) == 0 &&
        address >= 0x100000000ull && address % alignof(Owners) == 0, "immobile registry owners use absolute guest pointers");
    if (address < 0x100000000ull || address % alignof(Owners)) return 1;
    // Registry retains these addresses for process lifetime. No unmap/reuse while registered.
    auto& owner = *std::construct_at(reinterpret_cast<Owners*>(address));
    if (!register_program(owner.vs,true,VS,std::size(VS)) || !register_program(owner.ps,false,PS,std::size(PS))) return 1;
    auto state = state_for(owner); DrawItem draw;
    const bool made = realize_draw_item(state,&state.draws[0],6,64,false,draw,nullptr,true);
    check(made && draw.indices == std::vector<uint32_t>(owner.indices.begin(),owner.indices.end()) &&
        draw.fragment_wave_config_available && !draw.ps_wave32 &&
        fragment_spirv_required_subgroup_size(draw.fs_words()) == 64,
        "real indexed Wave64 DrawItem, not an invented backend fixture");
    if (!made || !draw.raster_quads) return 1;
    dump_sources(dump_directory,"position",draw);
    if (missing_analysis) {
        const auto& inputs = *draw.raster_quads->inputs;
        check(!inputs.raw_code && !inputs.raw_matches_producing_source,
            "analysis reuse disabled never reacquires guest VA for a collector token");
        RasterQuadCollector contract;
        check(build_raster_quad_collector(inputs,2,contract).empty() &&
            contract.rejection == "quad-collector-producing-source-unavailable",
            "actual DrawItem without owned producing analysis loudly declines");
    } else ownership_controls(draw,owner);
    if (!cpu && scissor_override) {
        // The normal backend would replace this empty scissor despite the string value "0".
        draw.ps.has_scissor = true;
        draw.ps.scissor_left = draw.ps.scissor_right = 0;
        draw.ps.scissor_top = draw.ps.scissor_bottom = 0;
        const auto refused = observe(draw,256);
        check(refused.attempted && !refused.complete && refused.quads.empty() &&
            refused.rejection == "quad-collector-raster-state-override",
            "real registered caller refuses presence-based empty-scissor override transactionally");
    } else if (!cpu) {
        const auto completed = observe(draw,256); observe_complete(completed);
        dump_sources(dump_directory,"position",draw,&completed);
        // Saturation is all-or-nothing without assuming repeated full-frame scope packing.
        observe_capacity(owner);
        auto replacement = draw.fs_words(); replacement[3] ^= 1; // valid module, different generator tag only
        draw.set_fs(std::move(replacement));
        const auto mismatch = observe(draw,256);
        check(mismatch.attempted && !mismatch.complete && mismatch.quads.empty() &&
            mismatch.rejection == "quad-collector-producing-module-mismatch", "shipping caller rejects selected SOURCE substitution");
    }
    if (!missing_analysis && !scissor_override) {
        if (!register_program(owner.varying_vs,true,VARYING_VS,std::size(VARYING_VS)) ||
            !register_program(owner.smooth_ps,false,SMOOTH_PS,std::size(SMOOTH_PS)) ||
            !register_program(owner.parameter_ps,false,PARAMETER_PS,std::size(PARAMETER_PS))) return 1;
        for (bool parameter : {false,true}) {
            auto varying_state = state_for(owner,&owner.varying_vs,
                parameter ? &owner.parameter_ps : &owner.smooth_ps,true);
            DrawItem varying_draw;
            const bool realized = realize_draw_item(varying_state,&varying_state.draws[0],6,64,false,varying_draw,nullptr,true);
            check(realized && varying_draw.raster_quads && varying_draw.has_pixel_inputs &&
                varying_draw.pixel_inputs.valid_mask == 1 &&
                varying_draw.raster_quads->inputs->interpolation.attribute_mask == 1 &&
                varying_draw.raster_quads->inputs->interpolation.requires_geometry == parameter &&
                (!varying_draw.gs.empty()) == parameter && fragment_spirv_required_subgroup_size(varying_draw.fs_words()) == 64,
                "real AGC varying realization supplies actual routing and Wave64 SOURCE");
            if (!realized || !varying_draw.raster_quads) continue;
            RasterQuadCollector varying_output;
            varying_output.fields.push_back({RasterQuadFieldKind::Interpolant,0,0,4});
            check(prosper::test::raster_quad_varying_interface(varying_draw.vs_words(),
                *varying_draw.raster_quads->inputs,varying_output),
                "actual owned varying VS has the required declaration AND whole-vector writer");
            dump_sources(dump_directory,parameter ? "parameter" : "smooth",varying_draw);
            if (!cpu) {
                const auto observed = observe(varying_draw,256);
                observe_complete(observed); observe_fields(observed,parameter);
                dump_sources(dump_directory,parameter ? "parameter" : "smooth",varying_draw,&observed);
                if (parameter && varying_draw.gs.size() >= 5) {
                    auto changed_gs = varying_draw.gs; changed_gs[3] ^= 1;
                    varying_draw.gs = std::move(changed_gs);
                    const auto refused = observe(varying_draw,256);
                    check(refused.attempted && !refused.complete && refused.quads.empty() &&
                        refused.rejection == "quad-collector-producing-module-mismatch",
                        "actual generated GS SOURCE substitution cannot borrow the retained producer");
                }
            }
        }
        auto missing_state = state_for(owner,&owner.vs,&owner.parameter_ps,true);
        DrawItem missing;
        const bool realized = realize_draw_item(missing_state,&missing_state.draws[0],6,64,false,missing,nullptr,true);
        check(realized && missing.raster_quads && !missing.gs.empty(),
            "actual explicit PS realization retains its generated GS with a declaration-only VS varying");
        if (realized && missing.raster_quads) {
            RasterQuadCollector needed; needed.fields.push_back({RasterQuadFieldKind::Interpolant,0,0,4});
            check(!prosper::test::raster_quad_varying_interface(missing.vs_words(),*missing.raster_quads->inputs,needed),
                "actual declared-but-unwritten producer varying fails before GPU creation");
            if (!cpu) {
                const auto refused = observe(missing,256);
                check(refused.attempted && !refused.complete && refused.quads.empty() &&
                    refused.rejection == "quad-collector-geometry-input-interface-unavailable",
                    "registered explicit-parameter caller transactionally rejects missing VS varying");
            }
        }
    }
    std::printf("[raster-quad-live] checks=%u failures=%u mode=%s; no guest64 admission/ordered commit claim\n",
        checks,failures,cpu ? "CPU ownership" : "GPU host observations");
    return failures ? 1 : 0;
}
