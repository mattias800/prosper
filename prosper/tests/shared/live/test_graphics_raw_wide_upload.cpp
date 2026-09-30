// Register actual synthetic AGC programs, realize real graphics tables, then consume their numeric
// bytes through the shipping resource builder/backend. Unsupported programs never execute on GPU.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
namespace Perf = prosper::diagnostics::perf;
static int failures = 0;
static bool cpu_only = false, pixel_control = false;
static void check(bool ok, const char* arm, const char* message) {
    std::printf("[%s] %s: %s\n", ok ? "ok" : "FAIL", arm, message);
    std::fflush(stdout);
    failures += !ok;
}
constexpr uint32_t W = 8, H = 8;
constexpr uint32_t Fullscreen[]{0x36020081u, 0x2c040081u, 0x7e020d01u,
    0x7e040d02u, 0x7e0a02f6u, 0x7e0c02f2u, 0x10020b01u, 0x08020d01u,
    0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u};
enum class Load { None, Numeric, MemorySelector, Descriptor, RuntimeSelector };
struct Program {
    alignas(256) std::array<uint32_t, 64> code{};
    AgcShaderSharp sharp{4u}; // independent metadata V# at user offset4, not a raw load source
    AgcShaderUserData user{};
    std::array<ShaderReg, 2> registers{};
    AgcShaderHeader header{};
    uint32_t load_pc = UINT32_MAX, scalar_pc = UINT32_MAX, consumer_pc = UINT32_MAX;
};
// The process-lifetime registry retains header/code addresses. Give every variant a distinct,
// immobile, aligned owner that remains alive until all fixture work and backend waits finish.
using ProgramOwners = std::array<Program, 20>;
static ProgramOwners* programs = nullptr;
static size_t next_program = 0;
static Program& register_program(bool vertex, Load load, bool wide8) {
    Program& p = programs->at(next_program++);
    std::vector<uint32_t> code;
    const uint32_t base = vertex ? 8u : 0u;
    if (load == Load::RuntimeSelector)
        code.push_back(0x7e000500u | ((base + 2u) << 17u)); // s offset <- real lane's v0
    if (load == Load::MemorySelector) {
        p.scalar_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {0xf4000580u | ((base + 8u) / 2u), 0xfa000000u});
    }
    if (load != Load::None) {
        const uint32_t selector = load == Load::MemorySelector ? 22u : base + 2u;
        code.push_back(0x8f148400u | selector); // s20 = selector << 4
        p.load_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {(wide8 ? 0xf40c0600u : 0xf4080600u) | (base / 2u),
                                 0x28000000u}); // s[24:27/31], entry pair, s20
        if (load == Load::Descriptor) {
            code.push_back(0x8f1b801bu); // s27 = s27 << 0: descriptor-only assembly
            p.consumer_pc = static_cast<uint32_t>(code.size());
            code.insert(code.end(), {0xe0300000u, vertex ? 0x80060700u : 0x80060000u});
        } else {
            const uint32_t last = wide8 ? 31u : 27u;
            code.push_back((vertex ? 0x7e0e0200u : 0x7e000200u) | last);
        }
        if (vertex && load == Load::MemorySelector) {
            code.insert(code.end(), {0x7e100c16u, 0x101010f0u, 0x060e1107u});
            // v7 += float(s22) * 0.5. Make the x1 upload itself observable beside the selected
            // wide bytes: a guest selector reread after realization moves this triangle offscreen.
        }
    }
    if (vertex) {
        code.insert(code.end(), std::begin(Fullscreen), std::end(Fullscreen));
        if (load != Load::None) code.push_back(0x06020f01u); // x += numeric v7
        code.insert(code.end(), {0xf80008cfu, 0x04030201u, 0xbf810000u});
    } else if (load == Load::None) {
        code.insert(code.end(), {0x7e000280u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
                                0xf800180fu, 0x03020100u, 0xbf810000u});
    } else {
        if (load == Load::MemorySelector)
            code.insert(code.end(), {0x7e020c16u, 0x100202ffu, 0x3e000000u}); // green=selector/8
        else code.insert(code.end(), {0x7e0202ffu, 0x3f000000u});
        code.insert(code.end(), {0x7e0402ffu, 0x3f400000u, 0x7e0602f2u,
                                0xf800180fu, 0x03020100u, 0xbf810000u});
    }
    std::copy(code.begin(), code.end(), p.code.begin());
    p.user.sharp_resource_offset[3] = &p.sharp;
    p.user.sharp_resource_count[3] = 1u;
    p.registers[0].offset = vertex ? P::SPI_SHADER_PGM_LO_ES : P::SPI_SHADER_PGM_LO_PS;
    p.registers[1].offset = vertex ? P::SPI_SHADER_PGM_HI_ES : P::SPI_SHADER_PGM_HI_PS;
    p.header.file_header = 0x34333231u;
    p.header.version = 0x18u;
    p.header.user_data = &p.user;
    p.header.sh_registers = p.registers.data();
    p.header.shader_size = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
    p.header.type = vertex ? 2u : 1u;
    p.header.num_sh_registers = 2u;
    void* registered = nullptr;
    const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
    check(create && create(reinterpret_cast<uint64_t>(&registered),
          reinterpret_cast<uint64_t>(&p.header), reinterpret_cast<uint64_t>(p.code.data()),
          0, 0, 0) == 0 && registered == &p.header, vertex ? "VS" : "PS",
          "register the owned AGC code/header through the real create-shader API");
    const uint64_t address = reinterpret_cast<uint64_t>(p.code.data());
    check(p.header.sh_registers == p.registers.data() &&
          p.registers[0].value == static_cast<uint32_t>(address >> 8u) &&
          p.registers[1].value == static_cast<uint32_t>((address >> 40u) & 0xffu) &&
          p.header.user_data == &p.user && p.user.sharp_resource_offset[3] == &p.sharp &&
          p.user.sharp_resource_count[3] == 1u, vertex ? "VS" : "PS",
          "registration patches the exact PGM pair and preserves nonempty metadata ownership");
    return p;
}

static void set_program(GpuState& state, const Program& p) {
    for (const auto& reg : p.registers) state.sh[reg.offset] = reg.value;
}
static void descriptor_words(uint32_t* out, uint64_t address, uint32_t size) {
    out[0] = static_cast<uint32_t>(address);
    out[1] = static_cast<uint32_t>((address >> 32u) & 0xffffu);
    out[2] = size;
    out[3] = (22u << 12u) | 0xfacu;
}
static GpuState state_for(const Program& vs, const Program& ps, uint64_t source,
                          uint64_t selector, uint64_t metadata) {
    GpuState state;
    set_program(state, vs); set_program(state, ps);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4u;
    state.cx[P::CB_TARGET_MASK] = state.cx[P::CB_SHADER_MASK] = 15u;
    state.sh[P::SPI_SHADER_PGM_RSRC2_GS] = 10u << P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT;
    state.sh[P::SPI_SHADER_PGM_RSRC2_PS] = 10u << P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT;
    // Make a runtime PS readfirstlane selector genuinely derive from FragCoord.x.
    state.cx[P::SPI_PS_INPUT_ENA] = state.cx[P::SPI_PS_INPUT_ADDR] = 1u << 8u;
    for (uint32_t base : {P::SPI_SHADER_USER_DATA_GS_0, P::SPI_SHADER_USER_DATA_PS_0}) {
        state.sh[base] = static_cast<uint32_t>(source);
        state.sh[base + 1u] = static_cast<uint32_t>(source >> 32u);
        state.sh[base + 2u] = 2u;
        uint32_t ordinary[4]; descriptor_words(ordinary, metadata, 64u);
        for (uint32_t i = 0; i < 4u; ++i) state.sh[base + 4u + i] = ordinary[i];
        state.sh[base + 8u] = static_cast<uint32_t>(selector);
        state.sh[base + 9u] = static_cast<uint32_t>(selector >> 32u);
    }
    GpuState::Draw draw;
    draw.index_count = 3u; draw.instance_count = 1u;
    state.draws.push_back(draw);
    return state;
}

static std::vector<uint8_t> render(DrawItem draw) {
    // This synthetic offscreen output has no guest color allocation. Shader state, tables,
    // compilation and uploads remain production-realized; give its color output a private extent.
    draw.color0_base = 0x7d000000u;
    draw.color0_width = W; draw.color0_height = H;
    draw.color_targets[0].mirror_named_identity(draw.color0_base, W, H);
    return render_submit_items({draw}, W, H);
}
static bool pixel(const std::vector<uint8_t>& image, uint32_t x, uint32_t y,
                   const std::array<uint8_t, 4>& expected) {
    if (image.size() != W * H * 4u) return false;
    for (uint32_t c = 0; c < 4u; ++c) {
        const int delta = static_cast<int>(image[(y * W + x) * 4u + c]) - expected[c];
        if (delta < -1 || delta > 1) return false;
    }
    return true;
}
static void observe_pixels(const DrawItem& draw, bool vertex, float value, const char* arm,
                           float green = 0.5f) {
    if (cpu_only) return;
    const auto image = render(draw);
    if (vertex) {
        bool correct = image.size() == W * H * 4u;
        for (uint32_t x : {0u, 6u, 7u}) {
            bool visible = static_cast<float>(2u * x + 1u) / W >= value;
            if (pixel_control) visible = !visible;
            correct &= image.size() == W * H * 4u &&
                image[(2u * W + x) * 4u + 1u] == (visible ? 255u : 0u);
        }
        check(correct, arm,
              "actual VS last-word translation reaches the real upload and pixels");
    } else {
        const uint8_t red = static_cast<uint8_t>((pixel_control ? 1.0f - value : value) * 255.0f);
        bool correct = true;
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x)
                correct &= pixel(image, x, y,
                    {red, static_cast<uint8_t>(green * 255.0f), 191u, 255u});
        check(correct, arm, "actual PS last-word color reaches the real upload and pixels");
    }
}

int main(int argc, char** argv) {
    cpu_only = argc == 2 && std::strcmp(argv[1], "--cpu-only") == 0;
    pixel_control = argc == 2 && std::strcmp(argv[1], "--pixel-control") == 0;
    if (argc > 2 || (argc == 2 && !cpu_only && !pixel_control)) {
        std::printf("[FAIL] usage: test_graphics_raw_wide_upload [--cpu-only|--pixel-control]\n");
        return 1;
    }
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    constexpr uint64_t Bytes = 0x10000u;
    uint64_t guest = 0;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), Bytes, 2u, 0,
          reinterpret_cast<uint64_t>("graphics-raw-wide"), 0) == 0 && guest,
          "setup", "map real guest data and selector backing");
    if (!guest || !unmap) return 1;
    // AGC treats pointer fields below 4 GiB as relative offsets. A non-PIE fixture's static
    // addresses occupy that range, so own the registry inputs in actual mapped guest storage.
    // Keep this bounded 64 KiB owner mapped for the registry's remaining process lifetime;
    // only the separate, transient numeric-data allocation is unmapped below.
    static_assert(sizeof(ProgramOwners) <= Bytes);
    uint64_t program_guest = 0;
    check(map(reinterpret_cast<uint64_t>(&program_guest), Bytes, 2u, 0,
              reinterpret_cast<uint64_t>("graphics-raw-programs"), 0) == 0 &&
          program_guest >= 0x100000000ull && program_guest % alignof(ProgramOwners) == 0,
          "setup", "own immobile AGC pointers in the absolute guest-address range");
    if (program_guest < 0x100000000ull || program_guest % alignof(ProgramOwners) != 0) return 1;
    programs = std::construct_at(reinterpret_cast<ProgramOwners*>(program_guest));
    if (!cpu_only) prosper::frontend::register_live_renderer(".", false);
    auto& plain_vs = register_program(true, Load::None, false);
    auto& plain_ps = register_program(false, Load::None, false);
    auto* data = reinterpret_cast<uint32_t*>(guest);
    auto* selector = reinterpret_cast<uint32_t*>(guest + 0x1000u);
    const uint64_t metadata = guest + 0x2000u;
    for (bool vertex : {true, false}) {
        const char* arm = vertex ? "VS" : "PS";
        for (bool wide8 : {false, true}) {
            auto& program = register_program(vertex, Load::Numeric, wide8);
            const uint32_t count = wide8 ? 8u : 4u;
            const float first = vertex ? 0.5f : 0.25f, second = vertex ? 0.0f : 0.75f;
            data[8u + count - 1u] = std::bit_cast<uint32_t>(first);
            data[12u + count - 1u] = std::bit_cast<uint32_t>(second);
            *selector = 2u;
            notify_guest_gpu_write(guest, 0x2004u);
            auto state = state_for(vertex ? program : plain_vs, vertex ? plain_ps : program,
                                   guest, reinterpret_cast<uint64_t>(selector), metadata);
            DrawItem draw;
            const bool made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, draw);
            const auto table = vertex ? draw.vrt : draw.prt;
            const auto* raw = table ? table->by_fetch_pc(program.load_pc) : nullptr;
            check(made && raw && valid_raw_register_snapshot_resource(*raw) &&
                  raw->gpu_addr == guest + 32u && raw->size == count * sizeof(uint32_t), arm,
                  wide8 ? "real frontend binds the exact x8 selected range"
                        : "real frontend binds the exact x4 selected range");
            if (!made || !raw) continue;
            const auto report = validate_spirv_descriptor_interface(
                vertex ? draw.vs_words() : draw.fs_words(), table.get(), vertex ? 0u : 1u,
                vertex ? SpirvShaderStage::Vertex : SpirvShaderStage::Fragment);
            const auto reflected = std::find_if(report.descriptors.begin(), report.descriptors.end(),
                [&](const auto& d) { return d.binding == raw->binding &&
                    d.kind == SpirvDescriptorKind::StorageBuffer && d.readable &&
                    d.required_bytes >= raw->size; });
            check(report.ok() && reflected != report.descriptors.end(), arm,
                  "actual stage module reflects the real exact-PC numeric buffer binding");
            std::array<uint8_t, 32> captured{};
            std::memcpy(captured.data(), reinterpret_cast<const void*>(raw->gpu_addr), raw->size);
            auto hosted_table = std::make_shared<ShaderResourceTable>(*table);
            auto hosted_source = std::find_if(hosted_table->resources.begin(),
                hosted_table->resources.end(), [&](const auto& r) {
                    return r.fetch_pc == program.load_pc;
                });
            hosted_source->host_data = captured.data();
            hosted_source->host_data_size = count * sizeof(uint32_t);
            hosted_source->host_data_prefix_bytes = 48u;
            DrawItem hosted = draw;
            if (vertex) hosted.vrt = hosted_table;
            else hosted.prt = hosted_table;
            check(valid_raw_register_snapshot_resource(*hosted_source), arm,
                  "a prefixed replay view keeps its exact rebased owned source valid");
            observe_pixels(draw, vertex, first, arm);
            data[8u + count - 1u] = std::bit_cast<uint32_t>(second);
            notify_guest_gpu_write(guest + 32u, count * sizeof(uint32_t));
            observe_pixels(draw, vertex, second, arm);
            observe_pixels(hosted, vertex, first, arm);
            const uint32_t ud = vertex ? P::SPI_SHADER_USER_DATA_GS_0 : P::SPI_SHADER_USER_DATA_PS_0;
            state.sh[ud + 2u] = 3u;
            DrawItem shifted;
            const bool shifted_made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, shifted);
            const auto shifted_table = vertex ? shifted.vrt : shifted.prt;
            check(shifted_made && shifted_table && shifted_table->by_fetch_pc(program.load_pc) &&
                  shifted_table->by_fetch_pc(program.load_pc)->gpu_addr == guest + 48u, arm,
                  "next real draw uses the changed per-draw selected address");
            if (shifted_made) observe_pixels(shifted, vertex, second, arm);
        }
        for (bool wide8 : {false, true}) {
            auto& descriptor = register_program(vertex, Load::Descriptor, wide8);
            auto* descriptor_words_at_source = data + 8u;
            descriptor_words(descriptor_words_at_source, guest + 0x3000u, 64u);
            auto* buffer_value = reinterpret_cast<uint32_t*>(guest + 0x3000u);
            *buffer_value = std::bit_cast<uint32_t>(vertex ? 0.5f : 0.25f);
            notify_guest_gpu_write(guest, 0x3004u);
            auto state = state_for(vertex ? descriptor : plain_vs, vertex ? plain_ps : descriptor,
                guest, reinterpret_cast<uint64_t>(selector), metadata);
            const auto inputs = build_stage_table(state,
                reinterpret_cast<uint64_t>(descriptor.code.data()), !vertex, 3u);
            const auto* consumer = inputs ? inputs->by_fetch_pc(descriptor.consumer_pc) : nullptr;
            check(inputs && consumer && consumer->gpu_addr == guest + 0x3000u &&
                  !inputs->by_fetch_pc(descriptor.load_pc), arm,
                  "real descriptor-only fold resolves its exact consumer without numeric raw backing");
            DrawItem draw;
            const bool made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, draw);
            check(made, arm, "descriptor-only x4/x8 positive reaches real graphics realization");
            if (made) observe_pixels(draw, vertex, vertex ? 0.5f : 0.25f, arm);
            *buffer_value = std::bit_cast<uint32_t>(vertex ? 0.0f : 0.75f);
            notify_guest_gpu_write(guest + 0x3000u, sizeof(*buffer_value));
            if (made) observe_pixels(draw, vertex, vertex ? 0.0f : 0.75f, arm);
        }
        auto& latched = register_program(vertex, Load::MemorySelector, false);
        data[11u] = std::bit_cast<uint32_t>(vertex ? 0.5f : 0.25f);
        data[15u] = std::bit_cast<uint32_t>(vertex ? 0.0f : 0.75f);
        *selector = 2u; notify_guest_gpu_write(guest, 0x2004u);
        auto state = state_for(vertex ? latched : plain_vs, vertex ? plain_ps : latched,
                               guest, reinterpret_cast<uint64_t>(selector), metadata);
        DrawItem draw;
        const bool made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, draw);
        const auto table = vertex ? draw.vrt : draw.prt;
        const auto* scalar = table ? table->by_fetch_pc(latched.scalar_pc) : nullptr;
        uint32_t owned = UINT32_MAX;
        if (scalar && scalar->host_data && scalar->host_data_size >= sizeof(owned))
            std::memcpy(&owned, scalar->host_data, sizeof(owned));
        *selector = 3u; notify_guest_gpu_write(reinterpret_cast<uint64_t>(selector), sizeof(*selector));
        check(made && scalar && valid_raw_offset_scalar_snapshot_resource(*scalar) && owned == 2u &&
              table->by_fetch_pc(latched.load_pc) &&
              table->by_fetch_pc(latched.load_pc)->gpu_addr == guest + 32u, arm,
              "real draw owns the x1 selector paired with its selected wide range");
        if (made) observe_pixels(draw, vertex, vertex ? 1.5f : 0.25f, arm, 0.25f);
        DrawItem next;
        const bool next_made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, next);
        const auto next_table = vertex ? next.vrt : next.prt;
        uint32_t next_owned = UINT32_MAX;
        const auto* next_scalar = next_table ? next_table->by_fetch_pc(latched.scalar_pc) : nullptr;
        if (next_scalar && next_scalar->host_data && next_scalar->host_data_size >= sizeof(next_owned))
            std::memcpy(&next_owned, next_scalar->host_data, sizeof(next_owned));
        check(next_made && next_owned == 3u && next_table->by_fetch_pc(latched.load_pc) &&
              next_table->by_fetch_pc(latched.load_pc)->gpu_addr == guest + 48u, arm,
              "next draw realizes the new memory selector instead of reusing the old range");
        if (next_made) observe_pixels(next, vertex, vertex ? 1.5f : 0.75f, arm, 0.375f);

        for (bool wide8 : {false, true}) {
            auto& refused = register_program(vertex, Load::RuntimeSelector, wide8);
            state = state_for(vertex ? refused : plain_vs, vertex ? plain_ps : refused,
                              guest, reinterpret_cast<uint64_t>(selector), metadata);
            const auto original_table = build_stage_table(state,
                reinterpret_cast<uint64_t>(refused.code.data()), !vertex, 3u);
            check(original_table && !original_table->resources.empty() &&
                  !original_table->by_fetch_pc(refused.load_pc), arm,
                  "runtime-selector negative has a legitimate nonempty stage table without raw backing");
            const auto reason = vertex ? Perf::DropReason::ShaderRecompileVertex
                                       : Perf::DropReason::ShaderRecompileFragment;
            const uint64_t before = Perf::ledger().drop_reasons[static_cast<size_t>(reason)].load();
            size_t backend_calls = 0;
            const auto no_image = execute_gpustate(state, [&](const std::vector<DrawItem>&) {
                ++backend_calls; return std::vector<uint8_t>{};
            }, 64u);
            check(backend_calls == 0u && no_image.empty() &&
                  Perf::ledger().drop_reasons[static_cast<size_t>(reason)].load() == before + 1u, arm,
                  "actual executor refuses unbacked numeric data with the exact stage drop before backend");
        }
    }
    unmap(guest, Bytes, 0, 0, 0, 0);
    std::printf("graphics raw-wide upload: %d failures (%s)\n", failures,
                cpu_only ? "CPU realization only; no device/pixel claim" : "actual graphics uploads/pixels");
    return failures != 0;
}
