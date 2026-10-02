// Register actual synthetic AGC programs, realize real graphics tables, then consume their numeric
// bytes through the shipping resource builder/backend. Unsupported programs never execute on GPU.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/live/live_renderer_internal.hpp" // production retained-layout certificate
#include "shared/live/submit_renderer/resolve_pass.hpp"
#include "shared/live/submit_renderer/callback_state.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
namespace Perf = prosper::diagnostics::perf;
static int failures = 0;
static bool cpu_only = false, pixel_control = false, nested_only = false;
static uint64_t last_output_identity = 0;
static void check(bool ok, const char* arm, const char* message) {
    std::printf("[%s] %s: %s\n", ok ? "ok" : "FAIL", arm, message);
    std::fflush(stdout);
    failures += !ok;
}
constexpr uint32_t W = 8, H = 8;
constexpr uint32_t Fullscreen[]{0x36020081u, 0x2c040081u, 0x7e020d01u,
    0x7e040d02u, 0x7e0a02f6u, 0x7e0c02f2u, 0x10020b01u, 0x08020d01u,
    0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u};
enum class Load { None, Numeric, MemorySelector, OwnedSelector, Nested, Descriptor, DescriptorMask, RuntimeSelector };
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
using ProgramOwners = std::array<Program, 32>;
static ProgramOwners* programs = nullptr;
static size_t next_program = 0;
static Program& register_program(bool vertex, Load load, bool wide8, uint32_t writer_position = 0,
                                  uint32_t nested_destination = 44u) {
    Program& p = programs->at(next_program++);
    std::vector<uint32_t> code;
    const uint32_t base = vertex ? 8u : 0u;
    if (writer_position == 1u) code.insert(code.end(), {0xe0700000u, 0x80000100u});
    if (load == Load::Nested) {
        p.scalar_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {(wide8 ? 0xf40c0a00u : 0xf4080a00u) | (base / 2u),
                                 0xfa000004u});
        p.load_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {(wide8 ? 0xf40c0014u : 0xf4080014u) |
                                 (nested_destination << 6u), 0xfa000010u});
        // Keep a real ordinary numeric observer even when the negative destination spans VCC.
        const uint32_t observed = std::min(nested_destination + (wide8 ? 7u : 3u), 105u);
        code.push_back((vertex ? 0x7e0e0200u : 0x7e000200u) | observed);
    }
    if (load == Load::RuntimeSelector)
        code.push_back(0x7e000500u | ((base + 2u) << 17u)); // s offset <- real lane's v0
    if (load == Load::MemorySelector) {
        p.scalar_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {0xf4000580u | ((base + 8u) / 2u), 0xfa000000u});
    }
    if (load == Load::OwnedSelector) {
        p.scalar_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {(wide8 ? 0xf40c0a00u : 0xf4080a00u) | ((base + 8u) / 2u),
                                 0xfa000004u}); // parent s[40:43/47], entry pair, +4
        code.push_back(0xbe800380u | ((base + 8u) << 16u));
        code.push_back(0xbe800380u | ((base + 9u) << 16u)); // pointer overwritten AFTER read
    }
    if (load != Load::None && load != Load::Nested) {
        const uint32_t selector = load == Load::MemorySelector ? 22u :
            load == Load::OwnedSelector ? (wide8 ? 47u : 43u) : base + 2u;
        code.push_back(0x8f148400u | selector); // s20 = selector << 4
        p.load_pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {(wide8 ? 0xf40c0600u : 0xf4080600u) | (base / 2u),
                                 0x28000000u}); // s[24:27/31], entry pair, s20
        if (load == Load::Descriptor || load == Load::DescriptorMask) {
            code.push_back(0x8f1b801bu); // s27 = s27 << 0: descriptor-only assembly
            p.consumer_pc = static_cast<uint32_t>(code.size());
            code.insert(code.end(), {0xe0300000u, vertex ? 0x80060700u : 0x80060000u});
            if (load == Load::DescriptorMask) {
                // The former raw descriptor's SGPR pair is now an independent emitted Bool.
                // Compare the separately resolved buffer value, then select a visible 0.5/0.
                const uint32_t value = vertex ? 7u : 0u;
                code.insert(code.end(), {0x7c0200f9u | (value << 9u), 0x06869880u,
                    0xd5010000u | value, 128u | (240u << 9u) | (24u << 18u)});
            }
        } else {
            const uint32_t last = wide8 ? 31u : 27u;
            code.push_back((vertex ? 0x7e0e0200u : 0x7e000200u) | last);
        }
        if (vertex && (load == Load::MemorySelector || load == Load::OwnedSelector)) {
            const uint32_t source = load == Load::MemorySelector ? 22u : (wide8 ? 47u : 43u);
            code.insert(code.end(), {0x7e100c00u | source, 0x101010f0u, 0x060e1107u});
            // v7 += float(s22) * 0.5. Make the selector upload itself observable beside the selected
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
        if (load == Load::MemorySelector || load == Load::OwnedSelector) {
            const uint32_t source = load == Load::MemorySelector ? 22u : (wide8 ? 47u : 43u);
            code.insert(code.end(), {0x7e020c00u | source, 0x100202ffu, 0x3e000000u}); // green=selector/8
        }
        else code.insert(code.end(), {0x7e0202ffu, 0x3f000000u});
        code.insert(code.end(), {0x7e0402ffu, 0x3f400000u, 0x7e0602f2u,
                                0xf800180fu, 0x03020100u, 0xbf810000u});
    }
    if (writer_position == 2u) code.insert(code.end() - 1, {0xe0700000u, 0x80000100u});
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
    state.cx[P::SPI_PS_IN_CONTROL] = 0u; // The real fragment compiler input is Wave64.
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

static std::vector<uint8_t> render(DrawItem draw, uint64_t retained_output = 0) {
    // This synthetic offscreen output has no guest color allocation. Shader state, tables,
    // compilation and uploads remain production-realized; give its color output a private extent.
    // Persistent targets preserve previously written pixels. Isolate each observation so
    // translated triangles are compared against a fresh output, not earlier coverage.
    static uint64_t output_identity = 0x7d000000u;
    draw.color0_base = output_identity;
    if (nested_only && !retained_output) {
        // Keep actual renderer targets in distinct, real direct allocations for their retained
        // lifetime. A fake offscreen VA would make the production raw authority query Unknown.
        const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
        const auto map_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
        uint64_t physical = 0, address = 0;
        const bool mapped = allocate && map_direct &&
            allocate(0, 0x200000000ull, 0x10000u, 0x10000u, 0,
                     reinterpret_cast<uint64_t>(&physical)) == 0 &&
            map_direct(reinterpret_cast<uint64_t>(&address), 0x10000u, 3, 0,
                       physical, 0x10000u) == 0;
        check(mapped && address, "nested-output", "retain a real distinct direct output allocation");
        if (!mapped || !address) return {};
        draw.color0_base = address;
    }
    if (retained_output) draw.color0_base = retained_output;
    output_identity += 0x1000u;
    draw.color0_width = W; draw.color0_height = H;
    draw.color_targets[0].mirror_named_identity(draw.color0_base, W, H);
    if (nested_only) {
        draw.color_targets[0].native_layout_known = true;
        draw.color_targets[0].tile_mode = 0u; // this fixture's actual direct output is linear
        draw.color_targets[0].raw_snapshot_footprint_bytes = linear_sampled_surface_bytes(W, H, 4u);
    }
    last_output_identity = draw.color0_base;
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
        uint32_t expected_mask = 0u, actual_mask = 0u;
        for (uint32_t x = 0u; x < W; ++x) {
            bool visible = static_cast<float>(2u * x + 1u) / W >= value;
            if (pixel_control) visible = !visible;
            if (visible) expected_mask |= 1u << x;
            const uint8_t actual = image.size() == W * H * 4u
                ? image[(2u * W + x) * 4u + 1u] : 0u;
            if (actual == 255u) actual_mask |= 1u << x;
            correct &= image.size() == W * H * 4u && actual == (visible ? 255u : 0u);
        }
        std::printf("[pixel-observation] VS selected=%g expected-mask=0x%02x "
                    "actual-mask=0x%02x image-bytes=%zu\n", value,
                    expected_mask, actual_mask, image.size());
        check(correct, arm,
              "actual VS last-word translation reaches the real upload and pixels");
    } else {
        const uint8_t expected_red = static_cast<uint8_t>(value * 255.0f);
        // Flip a byte bit, so the negative oracle has no fixed point (including value=0.5).
        const uint8_t red = pixel_control ? static_cast<uint8_t>(expected_red ^ 0x80u) : expected_red;
        bool correct = true;
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x)
                correct &= pixel(image, x, y,
                    {red, static_cast<uint8_t>(green * 255.0f), 191u, 255u});
        std::printf("[pixel-observation] PS selected=%g expected-red=%u "
                    "actual-first-red=%u image-bytes=%zu\n", value, static_cast<unsigned>(red),
                    image.size() >= 4u ? static_cast<unsigned>(image[0]) : 0u, image.size());
        check(correct, arm, "actual PS last-word color reaches the real upload and pixels");
    }
}

// Optional complete synthetic draws for stored/current-raw replay. Read only the two registered
// code spans and the fixture's owned guest allocation; required parents come from the realized
// table's old observation, after the guest's highest selector word has already changed.
static void emit_owned_replay_fixture(const std::filesystem::path& directory, DrawItem draw,
                                     const Program& vs, const Program& ps, bool vertex, bool wide8,
                                     uint64_t guest, size_t guest_bytes, uint64_t parent_address,
                                     uint32_t parent_pc, uint32_t child_pc, float child_value) {
    if (directory.empty()) return;
    const char* arm = vertex ? "VS" : "PS";
    const uint32_t size = wide8 ? 32u : 16u, last = wide8 ? 7u : 3u;
    draw.color0_base = 0x7e000000u + (vertex ? 0u : 0x2000u) + (wide8 ? 0x1000u : 0u);
    draw.color0_width = W; draw.color0_height = H;
    draw.color_targets[0].mirror_named_identity(draw.color0_base, W, H);
    unsigned parent_reads = 0u, child_reads = 0u;
    const auto original_table = vertex ? draw.vrt : draw.prt;
    const auto* original_child = original_table ? original_table->by_fetch_pc(child_pc) : nullptr;
    const uint64_t child_address = original_child ? original_child->gpu_addr : 0u;
    const auto copy = [](uint64_t address, uint8_t* dst, size_t want,
                         uint64_t base, const void* bytes, size_t length) {
        if (address < base || address - base >= length) return size_t{0};
        const auto offset = static_cast<size_t>(address - base);
        const auto copied = std::min(want, length - offset);
        std::memcpy(dst, static_cast<const uint8_t*>(bytes) + offset, copied);
        return copied;
    };
    const CaptureMemoryReader reader = [&](uint64_t address, uint8_t* dst, size_t want) {
        for (const Program* program : {&vs, &ps})
            if (const auto n = copy(address, dst, want,
                    reinterpret_cast<uint64_t>(program->code.data()), program->code.data(),
                    program->header.shader_size)) return n;
        const auto n = copy(address, dst, want, guest,
                            reinterpret_cast<const void*>(guest), guest_bytes);
        if (n && address < parent_address + size && parent_address < address + n)
            ++parent_reads;
        if (n && child_address && address < child_address + size && child_address < address + n)
            ++child_reads;
        return n;
    };
    GpuCaptureMetadata metadata;
    metadata.width = W; metadata.height = H;
    metadata.input_route = nested_only ? "synthetic-owned-nested-wide" : "synthetic-owned-raw-wide";
    GpuCaptureFile capture, decoded;
    GpuReplayFrame replay;
    std::string error;
    const std::string filename = std::string(vertex ? "vs-" : "ps-") +
        (wide8 ? "x8.prgcap" : "x4.prgcap");
    const auto path = directory / filename;
    const bool written = capture_draw_items({draw}, metadata, reader, capture, error) &&
        write_gpu_capture(path.string(), capture, error) &&
        read_gpu_capture(path.string(), decoded, error);
    const auto matches_raw = [&](uint32_t index, const Program& program) {
        if (index >= decoded.raw_shader_versions.size()) return false;
        const auto& raw = decoded.raw_shader_versions[index];
        return raw.has_endpgm && raw.words == std::vector<uint32_t>(program.code.begin(),
            program.code.begin() + program.header.shader_size / sizeof(uint32_t));
    };
    const bool complete = written && decoded.draws.size() == 1u &&
        decoded.metadata.width == W && decoded.metadata.height == H &&
        !decoded.draws[0].vs.empty() && !decoded.draws[0].fs.empty() &&
        decoded.draws[0].vertex_count == 3u && decoded.draws[0].instance_count == 1u &&
        decoded.draws[0].raw_draw_count == 3u && !decoded.draws[0].raw_indexed &&
        decoded.draws[0].color0_width == W && decoded.draws[0].color0_height == H &&
        draw.fragment_wave_config_available && !draw.ps_wave32 &&
        decoded.draws[0].fragment_wave_config_available &&
        decoded.draws[0].ps_wave32 == draw.ps_wave32 &&
        decoded.draws[0].vs_chain_raw_shader_index == UINT32_MAX &&
        matches_raw(decoded.draws[0].vs_raw_shader_index, vs) &&
        matches_raw(decoded.draws[0].fs_raw_shader_index, ps);
    check(complete, arm, "optional owned replay writes both raw stages, complete draw state and guest fragment width");
    const bool materialized = complete && materialize_gpu_replay(decoded, replay, error);
    const auto table = materialized && replay.items.size() == 1u
        ? (vertex ? replay.items[0].vrt : replay.items[0].prt) : nullptr;
    const auto* parent = table ? table->by_fetch_pc(parent_pc) : nullptr;
    const auto* child = table ? table->by_fetch_pc(child_pc) : nullptr;
    uint32_t parent_high = UINT32_MAX, child_high = UINT32_MAX;
    if (parent && parent->host_data && parent->host_data_size == size)
        std::memcpy(&parent_high, parent->host_data + last * 4u, 4u);
    if (child && child->host_data && child->host_data_size >= size)
        std::memcpy(&child_high, child->host_data + last * 4u, 4u);
    check(materialized && parent_reads == 0u && (!nested_only || child_reads == 0u) && parent &&
          (nested_only ? parent->owned_nested_snapshot_bytes == size :
                         parent->owned_raw_snapshot_bytes == size) && parent_high == 2u &&
          child && child->size == size && child_high == std::bit_cast<uint32_t>(child_value) &&
          replay.items[0].vs_words() == draw.vs_words() &&
          replay.items[0].fs_words() == draw.fs_words() &&
          replay.items[0].fragment_wave_config_available &&
          replay.items[0].ps_wave32 == draw.ps_wave32 &&
          decoded.draws[0].float_transport == draw.float_transport &&
          replay.items[0].float_transport == draw.float_transport, arm,
          "optional owned replay preserves old highest parent and separate highest child bytes");
    if (nested_only && materialized && parent && child) {
        const auto& item = replay.items[0];
        const auto recompile_pair = [&](const DrawItem& producing) {
            auto raw_vs = recompile_graphics_shader_cached(ShaderProgramStage::Vertex,
                vs.code.data(), vs.header.shader_size / 4u, producing.vrt.get(),
                producing.has_pixel_inputs ? &producing.pixel_inputs : nullptr, nullptr, nullptr,
                false, producing.vertex_lds_dwords, false, {}, {}, producing.float_transport);
            auto raw_fs = recompile_graphics_shader_cached(ShaderProgramStage::Fragment,
                ps.code.data(), ps.header.shader_size / 4u, producing.prt.get(),
                producing.has_pixel_inputs ? &producing.pixel_inputs : nullptr,
                producing.has_system_inputs ? &producing.system_inputs : nullptr, nullptr,
                producing.ps_wave32, 0u, false, {}, producing.ps_float_mode, producing.float_transport,
                producing.ps_float_flags, producing.ps_launch_rsrc1);
            return std::pair{std::move(raw_vs), std::move(raw_fs)};
        };
        const auto [raw_vs, raw_fs] = recompile_pair(item);
        check(decoded.format_version == 68u && !raw_vs.empty() && !raw_fs.empty() &&
              raw_vs == item.vs_words() && raw_fs == item.fs_words() &&
              owned_nested_snapshot_at(*table, parent_pc, size) &&
              owned_nested_snapshot_at(*table, child_pc, size), arm,
              "fresh owned nested capture raw/stored paired SOURCE is exact after guest A changed to B");
        // This is a CPU producing-profile control, not a device-enablement witness. The
        // executing backend must still gate the resulting words against its enabled features.
        auto explicit_item = item;
        explicit_item.float_transport = {FloatTransportProfile::ExplicitNonFinite32};
        explicit_item.ps_float_flags = {true, true, false};
        explicit_item.ps_launch_rsrc1 = {true, 0x20810000u};
        auto explicit_words = recompile_pair(explicit_item);
        explicit_item.set_vs(std::move(explicit_words.first));
        explicit_item.set_fs(std::move(explicit_words.second));
        GpuCaptureFile combined, combined_loaded;
        GpuReplayFrame combined_replay;
        std::vector<uint8_t> combined_wire;
        const bool combined_ok = !explicit_item.vs_words().empty() && !explicit_item.fs_words().empty() &&
            capture_draw_items({explicit_item}, metadata, reader, combined, error) &&
            serialize_gpu_capture(combined, combined_wire, error) &&
            deserialize_gpu_capture(combined_wire, combined_loaded, error) &&
            materialize_gpu_replay(combined_loaded, combined_replay, error) &&
            combined_replay.items.size() == 1u;
        const auto combined_table = combined_ok
            ? (vertex ? combined_replay.items[0].vrt : combined_replay.items[0].prt) : nullptr;
        const auto* combined_parent = combined_table ? owned_nested_snapshot_at(*combined_table, parent_pc, size) : nullptr;
        const auto* combined_child = combined_table ? owned_nested_snapshot_at(*combined_table, child_pc, size) : nullptr;
        check(combined_ok && combined_loaded.format_version == 68u && combined_parent && combined_child &&
              combined_loaded.draws[0].float_transport == explicit_item.float_transport &&
              combined_replay.items[0].float_transport == explicit_item.float_transport &&
              combined_loaded.draws[0].ps_float_flags == explicit_item.ps_float_flags &&
              combined_loaded.draws[0].ps_launch_rsrc1 == explicit_item.ps_launch_rsrc1 &&
              combined_replay.items[0].ps_float_flags == explicit_item.ps_float_flags &&
              combined_replay.items[0].ps_launch_rsrc1 == explicit_item.ps_launch_rsrc1 &&
              std::memcmp(combined_parent->host_data, parent->host_data, size) == 0 &&
              std::memcmp(combined_child->host_data, child->host_data, size) == 0 &&
              combined_replay.items[0].vs_words() == explicit_item.vs_words() &&
              combined_replay.items[0].fs_words() == explicit_item.fs_words() &&
              parent_reads == 0u && child_reads == 0u, arm,
              "same roundtrip retains producing profile, launch context and immutable owned parent/child SOURCE");
        if (combined_ok) {
            const auto paired = recompile_pair(combined_replay.items[0]);
            check(paired.first == explicit_item.vs_words() && paired.second == explicit_item.fs_words(), arm,
                  "combined explicit profile and exact owned bytes recompile both stored stages without replacement authority");
            auto missing_owner = combined_loaded;
            auto& resources = vertex ? missing_owner.draws[0].vrt.resources : missing_owner.draws[0].prt.resources;
            for (auto& resource : resources)
                if (resource.resource.fetch_pc == child_pc) resource.resource.owned_nested_snapshot_bytes = 0u;
            GpuReplayFrame refused_profile;
            check(!materialize_gpu_replay(missing_owner, refused_profile, error) &&
                  !float_transport_module_supported(explicit_item.fs_words().data(),
                      explicit_item.fs_words().size(), {FloatTransportProfile::Implicit}), arm,
                  "explicit profile cannot replace a child owner and ownership cannot grant executing-device features");
        }
        std::vector<uint8_t> wire;
        const size_t resource_count = decoded.draws[0].vrt.resources.size() +
                                      decoded.draws[0].prt.resources.size();
        const size_t tail_bytes = 4u * (1u + resource_count);
        if (serialize_gpu_capture(decoded, wire, error) && wire.size() > tail_bytes) {
            auto legacy = wire;
            legacy.resize(legacy.size() - tail_bytes); legacy[8] = 67u;
            GpuCaptureFile previous;
            GpuReplayFrame refused;
            check(deserialize_gpu_capture(legacy, previous, error) && previous.format_version == 67u &&
                  previous.draws[0].float_transport == item.float_transport &&
                  !materialize_gpu_replay(previous, refused, error), arm,
                  "genuine official v67 preserves producing profile and launch evidence but lacks nested runtime ownership");
            auto legacy66 = legacy;
            legacy66.resize(legacy66.size() - 16u); legacy66[8] = 66u;
            check(deserialize_gpu_capture(legacy66, previous, error) && previous.format_version == 66u &&
                  previous.draws[0].float_transport == item.float_transport &&
                  previous.draws[0].ps_float_flags == FragmentFloatFlags{} &&
                  previous.draws[0].ps_launch_rsrc1 == FragmentLaunchRsrc1{} &&
                  !materialize_gpu_replay(previous, refused, error), arm,
                  "genuine official v66 retains profile but invents no launch evidence or nested ownership");
            auto legacy65 = legacy66;
            legacy65.resize(legacy65.size() - 13u); legacy65[8] = 65u;
            check(deserialize_gpu_capture(legacy65, previous, error) && previous.format_version == 65u &&
                  previous.draws[0].float_transport == FloatTransportConfig{} &&
                  !materialize_gpu_replay(previous, refused, error), arm,
                  "genuine official v65 invents neither producing profile nor nested runtime ownership");
            auto wrong_count = wire;
            const size_t tail = wire.size() - tail_bytes;
            wrong_count[tail] = 0u; wrong_count[tail + 1u] = 0u;
            wrong_count[tail + 2u] = 0u; wrong_count[tail + 3u] = 0u;
            check(!deserialize_gpu_capture(wrong_count, previous, error), arm,
                  "v68 nested count must match every already bounded resource");
            auto truncated = wire; truncated.pop_back();
            check(!deserialize_gpu_capture(truncated, previous, error), arm,
                  "truncated v68 nested tail refuses before replay");
        } else check(false, arm, "serialize complete nested capture for independent prefix controls");
        const auto rejects = [&](const GpuCaptureFile& bad, const char* reason) {
            GpuReplayFrame refused;
            check(!materialize_gpu_replay(bad, refused, error), arm, reason);
        };
        for (uint32_t pc : {parent_pc, child_pc}) {
            const auto& captured_table = vertex ? decoded.draws[0].vrt : decoded.draws[0].prt;
            const auto found = std::find_if(captured_table.resources.begin(), captured_table.resources.end(),
                [pc](const auto& r) { return r.resource.fetch_pc == pc; });
            check(found != captured_table.resources.end(), arm, "captured parent and child retain exact PC owners");
            if (found == captured_table.resources.end()) continue;
            const size_t index = static_cast<size_t>(found - captured_table.resources.begin());
            for (uint32_t marker : {0u, wide8 ? 16u : 32u}) {
                auto bad = decoded;
                auto& resources = vertex ? bad.draws[0].vrt.resources : bad.draws[0].prt.resources;
                resources[index].resource.owned_nested_snapshot_bytes = marker;
                rejects(bad, "stored modules cannot bypass a missing or opposite-width nested owner");
            }
            if (found->blob_index < decoded.blobs.size()) {
                auto short_bytes = decoded;
                short_bytes.blobs[found->blob_index].bytes_read = size - 4u;
                rejects(short_bytes, "stored modules cannot borrow planned bytes that capture never observed");
            }
        }
        auto no_raw = decoded;
        (vertex ? no_raw.draws[0].vs_raw_shader_index : no_raw.draws[0].fs_raw_shader_index) = UINT32_MAX;
        rejects(no_raw, "stored modules require complete original raw nested provenance");
        auto duplicate = decoded;
        auto& duplicated = vertex ? duplicate.draws[0].vrt.resources : duplicate.draws[0].prt.resources;
        const auto first = std::find_if(duplicated.begin(), duplicated.end(),
            [parent_pc](const auto& r) { return r.resource.fetch_pc == parent_pc; });
        if (first != duplicated.end()) duplicated.push_back(*first);
        rejects(duplicate, "duplicate exact parent PC cannot claim snapshot authority");
        const auto& captured_table = vertex ? decoded.draws[0].vrt : decoded.draws[0].prt;
        const auto first_parent = std::find_if(captured_table.resources.begin(), captured_table.resources.end(),
            [parent_pc](const auto& r) { return r.resource.fetch_pc == parent_pc; });
        if (first_parent != captured_table.resources.end() && first_parent->blob_index < decoded.blobs.size()) {
            auto mixed = decoded;
            const uint64_t different_child = child->gpu_addr + 0x1000u - 16u;
            auto& bytes = mixed.blobs[first_parent->blob_index].bytes;
            if (bytes.size() >= sizeof(different_child)) {
                std::memcpy(bytes.data(), &different_child, sizeof(different_child));
                rejects(mixed, "stored modules reject a parent pointer mixed with another child's observation");
            }
        }
    }
    if (vertex) {
        const float translation = child_value + (nested_only ? 0.0f : 1.0f);
        uint32_t expected_mask = 0u;
        for (uint32_t x = 0u; x < W; ++x)
            if (static_cast<float>(2u * x + 1u) / W >= translation) expected_mask |= 1u << x;
        std::printf("[owned-replay-fixture] %s complete=%d materialized=%d parent-reads=%u "
                    "expected-vs-translation=%g expected-row2-green-mask=0x%02x error=%s\n",
                    path.string().c_str(), complete, materialized, parent_reads,
                    translation, expected_mask, error.c_str());
    } else {
        std::printf("[owned-replay-fixture] %s complete=%d materialized=%d parent-reads=%u "
                    "expected-ps-rgba=%u,%u,191,255 error=%s\n",
                    path.string().c_str(), complete, materialized, parent_reads,
                    static_cast<unsigned>(static_cast<uint8_t>(child_value * 255.0f)),
                    static_cast<unsigned>(static_cast<uint8_t>((nested_only ? 0.5f : 0.25f) * 255.0f)), error.c_str());
    }
}

static void run_nested_cases(const Program& plain_vs, const Program& plain_ps,
                             uint64_t guest, uint64_t physical,
                             const std::filesystem::path& replay_directory) {
    constexpr uint64_t Bytes = 0x10000u;
    ColorTargetState layout;
    layout.base = layout.allocation_base = guest;
    layout.width = 257u; layout.height = 3u;
    layout.has_extent = layout.has_attrib3 = true;
    layout.format = 0xau; layout.resource_type = 1u;
    layout.color_sw_mode = 27u;
    const auto footprint = [&] {
        return color_target_physical_bytes(layout);
    };
    check(footprint() == tiled_surface_bytes(257u, 3u, 27u) && footprint() > 257u * 3u * 4u,
          "native-layout", "production certificate includes complete tile padding rather than linear pixels");
    layout.in_mip_tail = true;
    check(!footprint(), "native-layout", "unproved packed tail cannot lend a native allocation bound");
    layout.in_mip_tail = false;
    layout.mip0_depth = 1u;
    check(!footprint(), "native-layout", "unproved array layers cannot borrow a single-plane bound");
    layout.mip0_depth = 0u;
    layout.color_sw_mode = 31u;
    check(!footprint(), "native-layout", "unknown tile equation is not treated as linear allocation proof");
    layout.color_sw_mode = 0u;
    layout.width = W; layout.height = H;
    check(footprint() == linear_sampled_surface_bytes(W, H, 4u), "native-layout",
          "known linear native fixture has a bounded row-padded allocation certificate");
    layout.log2_samples = 2u;
    check(!footprint(), "native-layout", "unsupported MSAA layout remains explicit unknown");
    layout.log2_samples = 0u;
    layout.has_attrib3 = false;
    check(!footprint(), "native-layout", "absence of native guest layout cannot authenticate a Vulkan extent");
    uint64_t output_physical = 0, output_guest = 0;
    if (cpu_only) {
        const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
        check(allocate(0, 0x200000000ull, Bytes, Bytes, 0,
                  reinterpret_cast<uint64_t>(&output_physical)) == 0 &&
              map(reinterpret_cast<uint64_t>(&output_guest), Bytes, 3u, 0u,
                  output_physical, Bytes) == 0 && output_guest,
              "current-output", "current color control owns a distinct physical allocation");
        // Exercise the real resolve producer before any Vulkan call: a missing source fails,
        // while a complete CPU fallback publishes successfully without an empty-pixel heuristic.
        const bool batched = false, timed = false;
        std::vector<DrawItem> items(1);
        items[0].color0_base = guest; items[0].color1_base = output_guest;
        items[0].color_targets[1].raw_snapshot_footprint_bytes =
            linear_sampled_surface_bytes(W, H, 4u);
        prosper::frontend::submit_renderer::RenderTiming timing;
        prosper::frontend::RttCache cache;
        size_t pass_i = items.size();
        prosper::test::BackendSubmissionBatch batch;
        std::vector<const DrawItem*> pass{&items[0]};
        prosper::frontend::submit_renderer::ResolvePassContext ctx{
            batched, items, timing, timed, cache, pass_i, batch, pass};
        const auto before = prosper::test::backend_failed_publication_generation().load();
        prosper::frontend::submit_renderer::resolve_pass(ctx);
        check(prosper::test::backend_failed_publication_generation().load() == before + 1u &&
              cache.find(output_guest) == cache.end(), "resolve-publication",
              "real missing-source resolve refuses and publishes its failed producer epoch");
        auto& source = cache[guest]; source.w = W; source.h = H;
        source.rgba = std::make_shared<const std::vector<uint8_t>>(W * H * 4u, 0x5au);
        prosper::frontend::submit_renderer::resolve_pass(ctx);
        check(prosper::test::backend_failed_publication_generation().load() == before + 1u &&
              cache.at(output_guest).rgba == source.rgba, "resolve-publication",
              "real complete CPU resolve fallback publishes without a false failure");
    }
    uint64_t authority_calls = 0, failures_in_epoch = 7; // historical failures do not poison new epochs
    bool pending = false;
    if (cpu_only) {
        set_graphics_producer_status_query([&] {
            return GraphicsProducerStatus{true, pending, failures_in_epoch +
                prosper::test::backend_failed_publication_generation().load(std::memory_order_acquire)};
        });
        set_graphics_raw_source_authority([&](const prosper::GuestMappingLease& lease,
                                              uint64_t address, uint32_t bytes) {
            ++authority_calls;
            return prosper::guest_memory_direct_allocation_relation(lease, address, bytes,
                address, bytes) == prosper::GuestMemoryTopologyRelation::Overlap;
        });
    }
    for (bool vertex : {true, false}) for (bool wide8 : {false, true}) {
        const char* arm = vertex ? "nested-VS" : "nested-PS";
        auto& program = register_program(vertex, Load::Nested, wide8);
        const uint32_t width = wide8 ? 32u : 16u, last = wide8 ? 7u : 3u;
        auto* parent = reinterpret_cast<uint32_t*>(guest + 0x1000u);
        auto* child_a = reinterpret_cast<uint32_t*>(guest + 0x2000u);
        auto* child_b = reinterpret_cast<uint32_t*>(guest + 0x3000u);
        const uint64_t a = reinterpret_cast<uint64_t>(child_a), b = reinterpret_cast<uint64_t>(child_b);
        std::fill_n(parent, 16, 0u); std::fill_n(child_a, 16, 0u); std::fill_n(child_b, 16, 0u);
        std::memcpy(parent + 1, &a, sizeof(a)); parent[1u + last] = 2u;
        child_a[4u + last] = std::bit_cast<uint32_t>(0.25f);
        child_b[4u + last] = std::bit_cast<uint32_t>(0.75f);
        auto state = state_for(vertex ? program : plain_vs, vertex ? plain_ps : program,
                               guest + 0x1000u, 0, guest + 0x4000u);
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(program.code.data(), program.header.shader_size / 4u, instructions);
        check(rdna2_owned_nested_wide_chains(instructions) ==
                  std::vector<RawNestedWideChain>{{0u, 2u, width, width, 4u, 16u}}, arm,
              "original complete program proves one exact-width no-writer hop and highest numeric child word");
        const auto publication = graphics_producer_status();
        const GraphicsRawSnapshotContext complete{publication.known && !publication.pending};
        check(complete.producers_complete, arm,
              "real backend (or CPU production fixture) reports completed producer publication");
        for (bool straddles_vcc : {false, true}) {
            const uint32_t destination = (wide8 ? 98u : 102u) + (straddles_vcc ? 2u : 0u);
            const auto& boundary = register_program(vertex, Load::Nested, wide8, 0u, destination);
            auto bounded_state = state_for(vertex ? boundary : plain_vs,
                vertex ? plain_ps : boundary, guest + 0x1000u, 0u, guest + 0x4000u);
            std::vector<Rdna2Inst> bounded_instructions;
            rdna2_walk(boundary.code.data(), boundary.header.shader_size / 4u, bounded_instructions);
            check(rdna2_raw_wide_data_loads(bounded_instructions) ==
                      std::vector<uint32_t>({0u, 2u}), arm,
                  "destination boundary control still requires the original numeric child bytes");
            authority_calls = 0u;
            DrawItem bounded_draw;
            const bool bounded_made = realize_draw_item(bounded_state, &bounded_state.draws[0],
                3u, 64u, false, bounded_draw, nullptr, true, nullptr, &complete);
            if (straddles_vcc) {
                const auto blocked = build_stage_table(bounded_state,
                    reinterpret_cast<uint64_t>(boundary.code.data()), !vertex, 3u, 0u, &complete);
                check(rdna2_owned_nested_wide_chains(bounded_instructions).empty() &&
                      !bounded_made && blocked && blocked->resources.empty() &&
                      blocked->owned_host_data.empty() && (!cpu_only || authority_calls == 0u), arm,
                      "x4/x8 straddling VCC refuses production realization before observing guest bytes");
            } else {
                const auto bounded_table = vertex ? bounded_draw.vrt : bounded_draw.prt;
                const auto* bounded_parent = bounded_table ?
                    owned_nested_snapshot_at(*bounded_table, 0u, width) : nullptr;
                const auto* bounded_child = bounded_table ?
                    owned_nested_snapshot_at(*bounded_table, 2u, width) : nullptr;
                uint32_t highest = 0u;
                if (bounded_child) std::memcpy(&highest, bounded_child->host_data + last * 4u, 4u);
                check(bounded_made && bounded_parent && bounded_child &&
                      bounded_parent->gpu_addr == guest + 0x1004u &&
                      bounded_child->gpu_addr == a + 16u &&
                      highest == std::bit_cast<uint32_t>(0.25f), arm,
                      "x4/x8 ending at s105 owns the highest ordinary numeric word through realization");
            }
        }
        DrawItem old;
        const bool made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, old,
                                            nullptr, true, nullptr, &complete);
        const auto table = vertex ? old.vrt : old.prt;
        const auto* observed_parent = table ? owned_nested_snapshot_at(*table, 0u, width) : nullptr;
        const auto* observed_child = table ? owned_nested_snapshot_at(*table, 2u, width) : nullptr;
        uint32_t observed = 0;
        if (observed_child) std::memcpy(&observed, observed_child->host_data + last * 4u, 4u);
        check(made && observed_parent && observed_child && observed == std::bit_cast<uint32_t>(0.25f) &&
              observed_parent->gpu_addr == guest + 0x1004u && observed_child->gpu_addr == a + 16u &&
              table->owned_nested_snapshot_requirements ==
                  std::vector<std::pair<uint32_t, uint32_t>>{{0u, width}, {2u, width}}, arm,
              "live realization owns effective parent and child once at their exact PCs");
        if (!made || !observed_parent || !observed_child) continue;
        observe_pixels(old, vertex, 0.25f, arm);
        std::memcpy(parent + 1, &b, sizeof(b)); child_a[4u + last] = std::bit_cast<uint32_t>(0.5f);
        notify_guest_gpu_write(guest + 0x1004u, width);
        notify_guest_gpu_write(a + 16u, width);
        observe_pixels(old, vertex, 0.25f, arm);
        DrawItem next;
        const bool next_made = realize_draw_item(state, &state.draws[0], 3u, 64u, false, next,
                                                 nullptr, true, nullptr, &complete);
        const auto next_table = vertex ? next.vrt : next.prt;
        const auto* next_child = next_table ? owned_nested_snapshot_at(*next_table, 2u, width) : nullptr;
        uint64_t old_pointer = 0;
        std::memcpy(&old_pointer, observed_parent->host_data, sizeof(old_pointer));
        check(next_made && next_child && next_child->gpu_addr == b + 16u && old_pointer == a, arm,
              "next draw observes changed pointer while the prior table retains its original parent/child");
        if (next_made) observe_pixels(next, vertex, 0.75f, arm);
        if (!cpu_only && vertex && !wide8) {
            prosper::GuestDirectAllocation retained;
            {
                prosper::GuestMappingLease lease;
                retained = prosper::guest_memory_direct_allocation(lease, last_output_identity, 1u);
                check(retained.identity && graphics_raw_source_is_guest_current(lease,
                      guest + 0x1004u, width) &&
                      !graphics_raw_source_is_guest_current(lease, last_output_identity + 0x4000u, width),
                      arm, "actual live retained authority admits distinct allocation and refuses same-allocation raw bytes");
            }
            const auto map_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
            const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
            const auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
            const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
            uint64_t alias = 0;
            check(retained.identity && map_direct(reinterpret_cast<uint64_t>(&alias), Bytes, 3u, 0u,
                      retained.physical_begin, Bytes) == 0 && alias, arm,
                  "map a different VA of the actual native retained producer allocation");
            if (alias) {
                GraphicsNestedWideReader reader(rdna2_owned_nested_wide_chains(instructions), &complete);
                check(!reader.probe(FoldProbe::Raw, 0u, alias + 0x1004u, width), arm,
                      "actual live provider refuses physical alias before retaining parent bytes");
            }
            if (alias) unmap(alias, Bytes, 0, 0, 0, 0);
            if (retained.identity) {
                check(unmap(last_output_identity, Bytes, 0, 0, 0, 0) == 0 &&
                      release(retained.physical_begin, Bytes, 0, 0, 0, 0) == 0, arm,
                      "remove the old native producer mapping while its renderer owner remains retained");
                {
                    prosper::GuestMappingLease lease;
                    check(graphics_raw_source_is_guest_current(lease, guest + 0x1004u, width), arm,
                          "unmapped old retained VA does not ban an authenticated distinct physical source");
                }
                uint64_t reuse = 0, address = 0;
                check(allocate(retained.physical_begin, retained.physical_end, Bytes, Bytes, 0u,
                          reinterpret_cast<uint64_t>(&reuse)) == 0 && reuse == retained.physical_begin &&
                      map_direct(reinterpret_cast<uint64_t>(&address), Bytes, 3u, 0u, reuse, Bytes) == 0 && address,
                      arm, "reuse old producer physical bytes under a new allocation birth");
                if (address) {
                    prosper::GuestMappingLease lease;
                    const auto current = prosper::guest_memory_direct_allocation(lease, address, width);
                    check(current.identity != retained.identity &&
                          !graphics_raw_source_is_guest_current(lease, address + 0x1004u, width), arm,
                          "actual retained origin excludes physical reuse despite new allocation identity");
                }
                if (address) unmap(address, Bytes, 0, 0, 0, 0);
                if (reuse) release(reuse, Bytes, 0, 0, 0, 0);
                uint64_t newer_physical = 0, rebound = last_output_identity;
                check(allocate(retained.physical_end, 0x200000000ull, Bytes, Bytes, 0u,
                          reinterpret_cast<uint64_t>(&newer_physical)) == 0 &&
                      map_direct(reinterpret_cast<uint64_t>(&rebound), Bytes, 3u, 0x10u,
                          newer_physical, Bytes) == 0 && rebound == last_output_identity,
                      arm, "rebind the same native target VA to a distinct physical allocation");
                if (newer_physical && rebound == last_output_identity) {
                    const auto produced = render(next, rebound);
                    check(pixel(produced, 7u, 2u, {0u, 255u, 0u, 255u}), arm,
                          "real producer reuses the retained image after its guest VA is rebound");
                    uint64_t older_reuse = 0, older_view = 0;
                    check(allocate(retained.physical_begin, retained.physical_end, Bytes, Bytes, 0u,
                              reinterpret_cast<uint64_t>(&older_reuse)) == 0 &&
                          map_direct(reinterpret_cast<uint64_t>(&older_view), Bytes, 3u, 0u,
                              older_reuse, Bytes) == 0 && older_view,
                          arm, "reuse the prior origin again after the second actual producer");
                    {
                        prosper::GuestMappingLease lease;
                        check(older_view &&
                              !graphics_raw_source_is_guest_current(lease, older_view + 0x1004u, width) &&
                              !graphics_raw_source_is_guest_current(lease, rebound + 0x1004u, width) &&
                              graphics_raw_source_is_guest_current(lease, guest + 0x1004u, width),
                              arm, "actual retained owner excludes both producer origins and still admits a distinct source");
                    }
                    if (older_view) unmap(older_view, Bytes, 0, 0, 0, 0);
                    if (older_reuse) release(older_reuse, Bytes, 0, 0, 0, 0);
                    // The second actual output remains mapped for its native owner's lifetime.
                }
            }
            // Exercise the actual registered callback, including its two pre-backend omissions.
            // Failure generation alone is not the oracle: B must be refused by the ordered
            // executor's real vertex-recompile drop site. A fresh epoch then consumes changed
            // child data and publishes pixels that the previous .75 observation cannot cover.
            auto& window = prosper::frontend::submit_renderer::CallbackState::instance();
            struct RestoreWindow {
                int& first; int& last;
                int saved_first, saved_last;
                ~RestoreWindow() { first = saved_first; last = saved_last; }
            } restore{window.g_render_first(), window.g_render_last(),
                      window.g_render_first(), window.g_render_last()};
            uint64_t window_physical = 0, window_guest = 0;
            check(allocate(0, 0x200000000ull, Bytes, Bytes, 0u,
                      reinterpret_cast<uint64_t>(&window_physical)) == 0 &&
                  map_direct(reinterpret_cast<uint64_t>(&window_guest), Bytes, 3u, 0u,
                      window_physical, Bytes) == 0 && window_guest, arm,
                  "render-window control owns a fresh real direct output allocation");
            if (window_guest) {
                auto ordered = state;
                ordered.cx[P::CB_COLOR0_BASE] = static_cast<uint32_t>(window_guest >> 8u);
                ordered.cx[P::CB_COLOR0_BASE_EXT] = static_cast<uint32_t>(window_guest >> 40u);
                ordered.cx[P::CB_COLOR0_INFO] = 0xau << P::CB_COLOR0_INFO_FORMAT_SHIFT;
                ordered.cx[P::CB_COLOR0_ATTRIB2] =
                    ((W - 1u) << P::CB_COLOR0_ATTRIB2_MIP0_WIDTH_SHIFT) | (H - 1u);
                ordered.cx[P::CB_COLOR0_ATTRIB3] =
                    1u << P::CB_COLOR0_ATTRIB3_RESOURCE_TYPE_SHIFT;
                ordered.draws[0].command_order = 10u;
                auto later = ordered.draws[0]; later.command_order = 30u;
                ordered.draws.push_back(later);
                const auto vertex_drops = [] {
                    return Perf::ledger().drop_reasons[
                        static_cast<size_t>(Perf::DropReason::ShaderRecompileVertex)].load();
                };
                for (bool last_gate : {false, true}) {
                    restore.first = last_gate ? 0 : std::numeric_limits<int>::max();
                    restore.last = last_gate ? -1 : std::numeric_limits<int>::max();
                    const auto generation = prosper::test::backend_failed_publication_generation().load();
                    (void)render_submit_items({}, W, H);
                    check(prosper::test::backend_failed_publication_generation().load() == generation,
                          arm, "empty render-window callback is not a failed producer");
                    const auto drops = vertex_drops();
                    (void)execute_ordered_and_present(ordered, W, H,
                        600u + next_program + static_cast<uint64_t>(last_gate), false);
                    check(prosper::test::backend_failed_publication_generation().load() == generation + 1u &&
                          vertex_drops() == drops + 1u, arm,
                          last_gate ? "actual last-window omission of A refuses nested B in the same epoch" :
                                      "actual warmup omission of A refuses nested B in the same epoch");
                }
                restore.first = 0; restore.last = std::numeric_limits<int>::max();
                ordered.draws.erase(ordered.draws.begin());
                child_b[4u + last] = std::bit_cast<uint32_t>(0.5f);
                notify_guest_gpu_write(b + 16u, width);
                const auto generation = prosper::test::backend_failed_publication_generation().load();
                const auto drops = vertex_drops();
                const bool published = execute_ordered_and_present(ordered, W, H,
                    700u + next_program, true);
                PresentFrameLease recovered;
                check(published && present_acquire_rendered_frame(recovered) && recovered.rgba &&
                      recovered.width == W && recovered.height == H &&
                      pixel(*recovered.rgba, 2u, 2u, {0u, 255u, 0u, 255u}) &&
                      pixel(*recovered.rgba, 0u, 2u, {0u, 0u, 0u, 255u}) &&
                      vertex_drops() == drops &&
                      prosper::test::backend_failed_publication_generation().load() == generation,
                      arm, "fresh completed submit recovers changed numeric data after historical window omissions");
                child_b[4u + last] = std::bit_cast<uint32_t>(0.75f);
                notify_guest_gpu_write(b + 16u, width);
                // Keep this real output mapped for its retained native owner's lifetime.
            }
        }
        emit_owned_replay_fixture(replay_directory, old,
            vertex ? program : plain_vs, vertex ? plain_ps : program,
            vertex, wide8, guest, Bytes, guest + 0x1004u, 0u, 2u, 0.25f);
        const auto stage = vertex ? ShaderProgramStage::Vertex : ShaderProgramStage::Fragment;
        const auto cold = recompile_graphics_shader_cached(stage, program.code.data(),
            program.header.shader_size / 4u, table.get());
        const auto warm_before = shader_recompile_cache_stats();
        const auto warm = recompile_graphics_shader_cached(stage, program.code.data(),
            program.header.shader_size / 4u, next_table.get());
        const auto warm_after = shader_recompile_cache_stats();
        check(!cold.empty() && warm == cold && warm_after.hits == warm_before.hits + 1u, arm,
              "different current byte owners reuse code without reusing another draw's data");
        for (uint32_t pc : {0u, 2u}) {
            auto malformed = *table;
            auto resource = std::find_if(malformed.resources.begin(), malformed.resources.end(),
                [pc](const auto& r) { return r.fetch_pc == pc; });
            resource->owned_nested_snapshot_bytes = wide8 ? 16u : 32u;
            check(recompile_graphics_shader_cached(stage, program.code.data(),
                      program.header.shader_size / 4u, &malformed).empty(), arm,
                  "warm cache cannot borrow admission after either parent's or child's width changes");
        }
        // Explicit context absence and producer refusal must fail before the backend, beside a
        // positive with identical code, mappings and ordinary metadata.
        DrawItem refused;
        check(!realize_draw_item(state, &state.draws[0], 3u, 64u, false, refused), arm,
              "eager/direct caller without ordered producer authority refuses nested numeric data");
        const GraphicsRawSnapshotContext incomplete{false};
        check(!realize_draw_item(state, &state.draws[0], 3u, 64u, false, refused,
                                nullptr, true, nullptr, &incomplete), arm,
              "an explicitly incomplete producer epoch cannot read or admit either snapshot");
        for (uint32_t writer_position : {1u, 2u}) {
            const auto& writer = register_program(vertex, Load::Nested, wide8, writer_position);
            auto writing = state_for(vertex ? writer : plain_vs, vertex ? plain_ps : writer,
                                     guest + 0x1000u, 0u, guest + 0x4000u);
            authority_calls = 0;
            const auto blocked = build_stage_table(writing, reinterpret_cast<uint64_t>(writer.code.data()),
                !vertex, 3u, 0u, &complete);
            check(blocked && blocked->resources.empty() && blocked->owned_host_data.empty() &&
                  (!cpu_only || authority_calls == 0u) &&
                  !realize_draw_item(writing, &writing.draws[0], 3u, 64u, false, refused,
                                    nullptr, true, nullptr, &complete), arm,
                  "a writer before or after the hop refuses before raw folding/ownership despite valid mappings");
        }
        {
            GraphicsNestedWideReader unsafe(rdna2_owned_nested_wide_chains(instructions), &complete);
            check(!unsafe.probe(FoldProbe::Base48, 0u, guest + 0x1004u, width) &&
                  !unsafe.probe(FoldProbe::Base40, 0u, guest + 0x1004u, width) &&
                  !unsafe.probe(FoldProbe::Raw, 0u, 0xdead00000000ull, width) &&
                  !unsafe.probe(FoldProbe::Raw, 0u, guest + Bytes - 4u, width), arm,
                  "canonical-address repair, untracked backing and mapping-boundary overflow never supply snapshots");
        }

        if (cpu_only) {
            // A real ordered DMA publishes B's pointer between two graphics consumers. The live
            // executor must bypass eager preparation and realize each stage after that producer.
            std::memcpy(parent + 1, &a, sizeof(a)); child_a[4u + last] = std::bit_cast<uint32_t>(0.25f);
            auto* staged = reinterpret_cast<uint32_t*>(guest + 0x5000u);
            std::memcpy(staged, parent + 1, width); std::memcpy(staged, &b, sizeof(b));
            state.draws[0].command_order = 10u;
            auto draw_b = state.draws[0]; draw_b.command_order = 30u; state.draws.push_back(draw_b);
            state.dma_copies.push_back({guest + 0x1004u, guest + 0x5000u, width, 0, 20u, 0});
            std::vector<std::shared_ptr<ShaderResourceTable>> observed_tables;
            set_submit_renderer([&](const std::vector<DrawItem>& draws, uint32_t, uint32_t) {
                for (const auto& draw : draws) observed_tables.push_back(vertex ? draw.vrt : draw.prt);
                return RenderedFrame(std::vector<uint8_t>(W * H * 4u, 0));
            });
            execute_ordered_and_present(state, W, H, 100u + next_program, false);
            const auto* first_child = observed_tables.size() == 2u
                ? owned_nested_snapshot_at(*observed_tables[0], 2u, width) : nullptr;
            const auto* second_child = observed_tables.size() == 2u
                ? owned_nested_snapshot_at(*observed_tables[1], 2u, width) : nullptr;
            check(first_child && second_child && first_child->gpu_addr == a + 16u &&
                  second_child->gpu_addr == b + 16u &&
                  first_child->host_data != second_child->host_data, arm,
                  "production ordered A -> completed DMA -> B observes each current pointer and distinct stable owners");
            state.draws.erase(state.draws.begin()); state.dma_copies[0].src = 0xdead00000000ull;
            observed_tables.clear(); authority_calls = 0;
            execute_ordered_and_present(state, W, H, 200u + next_program, false);
            check(observed_tables.empty() && authority_calls == 0u, arm,
                  "failed production DMA epoch refuses the later draw before raw authority or byte consumption");
            state.dma_copies.clear(); pending = true; observed_tables.clear(); authority_calls = 0;
            execute_ordered_and_present(state, W, H, 300u + next_program, false);
            check(observed_tables.empty() && authority_calls == 0u, arm,
                  "pending producer publication refuses before any parent or child byte request");
            pending = false;
            // Refuse a real backend resource contract BEFORE Vulkan initialization/enqueue.
            // A successful nonfinal callback also returns empty, so the production generation,
            // rather than pixel emptiness, must prevent B from observing this producer as complete.
            state.draws.insert(state.draws.begin(), draw_b);
            state.draws[0].command_order = 10u;
            observed_tables.clear(); authority_calls = 0;
            const auto generation = prosper::test::backend_failed_publication_generation().load();
            set_submit_renderer([&](const std::vector<DrawItem>& draws, uint32_t, uint32_t) {
                for (const auto& draw : draws) observed_tables.push_back(vertex ? draw.vrt : draw.prt);
                prosper::test::BackendDraw bad;
                prosper::test::FrameResource invalid;
                invalid.is_storage_image = true; invalid.storage_image_contract_valid = false;
                bad.R.push_back(invalid);
                const auto rejected = prosper::test::render_draws_rgba({bad}, W, H);
                check(rejected.empty() &&
                      prosper::test::backend_failed_publication_generation().load() > generation,
                      arm, "nonempty backend early refusal publishes failure without initializing Vulkan");
                authority_calls = 0;
                return RenderedFrame{};
            });
            execute_ordered_and_present(state, W, H, 400u + next_program, false);
            check(observed_tables.size() == 1u && authority_calls == 0u, arm,
                  "failed completed producer epoch refuses B before raw source consumption");
            state.draws.erase(state.draws.begin());
            observed_tables.clear();
            set_submit_renderer([&](const std::vector<DrawItem>& draws, uint32_t, uint32_t) {
                for (const auto& draw : draws) observed_tables.push_back(vertex ? draw.vrt : draw.prt);
                return RenderedFrame{}; // successful nonfinal empty pixel output is permissible
            });
            execute_ordered_and_present(state, W, H, 500u + next_program, false);
            check(observed_tables.size() == 1u, arm,
                  "a new producer epoch can admit disjoint inputs after a historical failure");
            const auto map_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
            const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
            uint64_t alias = 0;
            check(map_direct(reinterpret_cast<uint64_t>(&alias), Bytes, 3, 0, physical, Bytes) == 0 && alias,
                  arm, "map a VA-disjoint physical alias of the retained producer allocation");
            if (alias) {
                prosper::GuestMappingLease lease;
                check(prosper::guest_memory_direct_allocation_relation(lease, guest + 0x1004u, width,
                          alias + 0x1004u, width) == prosper::GuestMemoryTopologyRelation::Overlap &&
                      prosper::guest_memory_topology_relation(guest + 0x1004u, width,
                          alias + 0x1004u, width) == prosper::GuestMemoryTopologyRelation::Overlap &&
                      prosper::guest_memory_direct_allocation_relation(lease, guest + 0x1004u, width,
                          alias + 0x3000u, 1u) == prosper::GuestMemoryTopologyRelation::Overlap &&
                      prosper::guest_memory_topology_relation(guest + 0x1004u, width,
                          alias + 0x3000u, 1u) == prosper::GuestMemoryTopologyRelation::Disjoint, arm,
                      "allocation-level producer exclusion refuses same allocation beyond the selected linear extent");
            }
            if (alias) {
                auto output_alias = state;
                output_alias.cx[P::CB_COLOR0_BASE] = static_cast<uint32_t>((alias + 0x3000u) >> 8u);
                output_alias.cx[P::CB_COLOR0_BASE_EXT] = static_cast<uint32_t>((alias + 0x3000u) >> 40u);
                output_alias.cx[P::CB_COLOR0_INFO] = 0xau << P::CB_COLOR0_INFO_FORMAT_SHIFT;
                output_alias.cx[P::CB_COLOR0_ATTRIB2] =
                    ((W - 1u) << P::CB_COLOR0_ATTRIB2_MIP0_WIDTH_SHIFT) | (H - 1u);
                output_alias.cx[P::CB_COLOR0_ATTRIB3] =
                    1u << P::CB_COLOR0_ATTRIB3_RESOURCE_TYPE_SHIFT;
                DrawItem aliased;
                check(color_target_physical_bytes(extract_render_state(output_alias).color_targets[0]) != 0,
                      arm, "current alias refusal uses a proved complete color footprint");
                check(!realize_draw_item(output_alias, &output_alias.draws[0], 3u, 64u, false,
                      aliased, nullptr, true, nullptr, &complete), arm,
                      "production realization refuses a VA-disjoint current target in the same physical allocation");
                auto isolated = output_alias;
                isolated.cx[P::CB_COLOR0_BASE] = static_cast<uint32_t>(output_guest >> 8u);
                isolated.cx[P::CB_COLOR0_BASE_EXT] = static_cast<uint32_t>(output_guest >> 40u);
                check(output_guest && realize_draw_item(isolated, &isolated.draws[0], 3u, 64u, false,
                      aliased, nullptr, true, nullptr, &complete), arm,
                      "identical current target shape in a distinct physical allocation admits real snapshots");
            }
            // Drop the lease before changing mappings; source admission itself holds one until
            // it has retained and published both observations.
            if (alias) unmap(alias, Bytes, 0, 0, 0, 0);
        }
    }
    if (cpu_only && output_guest) {
        const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        const auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
        check(unmap(output_guest, Bytes, 0, 0, 0, 0) == 0 &&
              release(output_physical, Bytes, 0, 0, 0, 0) == 0,
              "current-output", "release the completed CPU control output allocation");
    }
    if (cpu_only) { set_submit_renderer({}); set_graphics_raw_source_authority({}); set_graphics_producer_status_query({}); }
}

int main(int argc, char** argv) {
    std::filesystem::path replay_directory;
    const bool emit_replay = argc == 3 && std::strcmp(argv[1], "--emit-owned-replay-fixtures") == 0;
    const bool emit_nested = argc == 3 && std::strcmp(argv[1], "--emit-nested-replay-fixtures") == 0;
    nested_only = emit_nested || (argc == 2 && (std::strcmp(argv[1], "--nested-only") == 0 ||
                                               std::strcmp(argv[1], "--nested-cpu-only") == 0));
    cpu_only = emit_replay || emit_nested || (argc == 2 && (std::strcmp(argv[1], "--cpu-only") == 0 ||
                                                         std::strcmp(argv[1], "--nested-cpu-only") == 0));
    pixel_control = argc == 2 && std::strcmp(argv[1], "--pixel-control") == 0;
    if ((!emit_replay && !emit_nested && argc > 2) || (argc == 2 && !cpu_only && !pixel_control && !nested_only)) {
        std::printf("[FAIL] usage: test_graphics_raw_wide_upload [--cpu-only|--pixel-control|"
                    "--nested-cpu-only|--nested-only|--emit-owned-replay-fixtures NEW_DIRECTORY|"
                    "--emit-nested-replay-fixtures NEW_DIRECTORY]\n");
        return 1;
    }
#if !defined(__linux__)
    if (nested_only) {
        std::printf("[NO_DATA] nested graphics direct-memory lazy-fault proof is Linux-only\n");
        return 77;
    }
#endif
    if (emit_replay || emit_nested) {
        replay_directory = argv[2];
        std::error_code error;
        if (!std::filesystem::create_directory(replay_directory, error)) {
            std::fprintf(stderr, "owned replay requires a new directory: %s\n", error.message().c_str());
            return 2;
        }
    }
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    constexpr uint64_t Bytes = 0x10000u;
    uint64_t guest = 0;
    uint64_t physical = 0;
    const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
    const auto map_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
    const bool mapped = nested_only ? allocate && map_direct &&
        allocate(0, 0x200000000ull, Bytes, Bytes, 0, reinterpret_cast<uint64_t>(&physical)) == 0 &&
        map_direct(reinterpret_cast<uint64_t>(&guest), Bytes, 3, 0, physical, Bytes) == 0 :
        map && map(reinterpret_cast<uint64_t>(&guest), Bytes, 2u, 0,
                   reinterpret_cast<uint64_t>("graphics-raw-wide"), 0) == 0;
    check(mapped && unmap && guest,
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
    if (!cpu_only) {
        prosper::frontend::register_live_renderer(".", false);
        if (nested_only && !prosper::test::render_vk_ctx().ok) {
            std::printf("[NO_DATA] native nested graphics requires an available Vulkan backend\n");
            return 77;
        }
    }
    auto& plain_vs = register_program(true, Load::None, false);
    auto& plain_ps = register_program(false, Load::None, false);
    if (nested_only) {
        run_nested_cases(plain_vs, plain_ps, guest, physical, replay_directory);
        unmap(guest, Bytes, 0, 0, 0, 0);
        const auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
        if (release && physical) release(physical, Bytes, 0, 0, 0, 0);
        std::printf("graphics nested-wide upload: %d failures (%s)\n", failures,
            cpu_only ? "CPU realization only" : "actual direct graphics owned numeric pixels");
        return failures != 0;
    }
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
            constexpr size_t Prefix = 48u;
            alignas(uint32_t) std::array<uint8_t, Prefix + 32u> captured{};
            std::memcpy(captured.data() + Prefix,
                        reinterpret_cast<const void*>(raw->gpu_addr), raw->size);
            auto hosted_table = std::make_shared<ShaderResourceTable>(*table);
            auto hosted_source = std::find_if(hosted_table->resources.begin(),
                hosted_table->resources.end(), [&](const auto& r) {
                    return r.fetch_pc == program.load_pc;
                });
            hosted_source->host_data = captured.data() + Prefix;
            hosted_source->host_data_size = count * sizeof(uint32_t);
            hosted_source->host_data_prefix_bytes = Prefix;
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
        auto& mask = register_program(vertex, Load::DescriptorMask, false);
        descriptor_words(data + 8u, guest + 0x3000u, 64u);
        auto* mask_value = reinterpret_cast<uint32_t*>(guest + 0x3000u);
        *mask_value = std::bit_cast<uint32_t>(0.5f);
        notify_guest_gpu_write(guest, 0x3004u);
        auto mask_state = state_for(vertex ? mask : plain_vs, vertex ? plain_ps : mask,
            guest, reinterpret_cast<uint64_t>(selector), metadata);
        const auto mask_inputs = build_stage_table(mask_state,
            reinterpret_cast<uint64_t>(mask.code.data()), !vertex, 3u);
        std::vector<Rdna2Inst> mask_instructions;
        rdna2_walk(mask.code.data(), mask.header.shader_size / sizeof(uint32_t), mask_instructions);
        const auto* mask_consumer = mask_inputs ? mask_inputs->by_fetch_pc(mask.consumer_pc) : nullptr;
        check(mask_inputs && !mask_inputs->resources.empty() && mask_consumer &&
              mask_consumer->gpu_addr == guest + 0x3000u &&
              !mask_inputs->by_fetch_pc(mask.load_pc) &&
              rdna2_raw_wide_data_loads(mask_instructions).empty(), arm,
              "real descriptor-to-fresh-mask program has exact consumer and no numeric raw source");
        DrawItem mask_draw;
        const bool mask_made = realize_draw_item(mask_state, &mask_state.draws[0], 3u, 64u, false, mask_draw);
        check(mask_made && !mask_draw.vs_words().empty() && !mask_draw.fs_words().empty(), arm,
              "real fresh mask lifetimes retain both nonempty graphics stages");
        if (mask_made) observe_pixels(mask_draw, vertex, 0.5f, arm);
        *mask_value = std::bit_cast<uint32_t>(0.0f);
        notify_guest_gpu_write(guest + 0x3000u, sizeof(*mask_value));
        if (mask_made) observe_pixels(mask_draw, vertex, 0.0f, arm);
        for (bool wide8 : {false, true}) {
            auto& owned_program = register_program(vertex, Load::OwnedSelector, wide8);
            const uint32_t count = wide8 ? 8u : 4u;
            const uint32_t last = count - 1u;
            // Export-only VS translation stays visible: child -0.5 plus saved selector 2/2.
            // A parent reread (3/2) or wrong zero child changes the expected row coverage.
            const float first = vertex ? (emit_replay ? -0.5f : 0.5f) : 0.25f;
            const float second = vertex ? 0.0f : 0.75f;
            data[8u + last] = std::bit_cast<uint32_t>(first);
            data[12u + last] = std::bit_cast<uint32_t>(second);
            std::fill(selector, selector + 9u, 0u);
            selector[1u + last] = 2u;
            notify_guest_gpu_write(guest, 0x2024u);
            auto owned_state = state_for(vertex ? owned_program : plain_vs,
                vertex ? plain_ps : owned_program, guest,
                reinterpret_cast<uint64_t>(selector), metadata);
            DrawItem owned_draw;
            const bool owned_made = realize_draw_item(owned_state, &owned_state.draws[0],
                3u, 64u, false, owned_draw);
            const auto owned_table = vertex ? owned_draw.vrt : owned_draw.prt;
            const auto* parent = owned_table ? owned_table->by_fetch_pc(owned_program.scalar_pc) : nullptr;
            const auto* child = owned_table ? owned_table->by_fetch_pc(owned_program.load_pc) : nullptr;
            uint32_t observed = UINT32_MAX;
            if (parent && parent->host_data && parent->host_data_size == count * 4u)
                std::memcpy(&observed, parent->host_data + last * 4u, 4u);
            check(owned_made && parent && child && observed == 2u &&
                  parent->owned_raw_snapshot_bytes == count * 4u &&
                  parent->gpu_addr == reinterpret_cast<uint64_t>(selector + 1u) &&
                  owned_table->owned_host_data.size() == 1u &&
                  std::count_if(owned_table->resources.begin(), owned_table->resources.end(),
                      [&](const auto& r) { return r.fetch_pc == owned_program.scalar_pc; }) == 1 &&
                  child->gpu_addr == guest + 32u && child->size == count * 4u, arm,
                  "real stage owns complete x4/x8 parent highest word with exactly one authoritative binding");
            if (!owned_made || !parent || !child) continue;
            const auto interface = validate_spirv_descriptor_interface(
                vertex ? owned_draw.vs_words() : owned_draw.fs_words(), owned_table.get(),
                vertex ? 0u : 1u, vertex ? SpirvShaderStage::Vertex : SpirvShaderStage::Fragment);
            const auto reflected = [&](const ShaderResource* source) {
                return std::any_of(interface.descriptors.begin(), interface.descriptors.end(),
                    [&](const auto& d) { return d.binding == source->binding &&
                        d.kind == SpirvDescriptorKind::StorageBuffer && d.readable &&
                        d.required_bytes >= source->size; });
            };
            check(interface.ok() && reflected(parent) && reflected(child), arm,
                  "both real graphics bindings reflect complete highest parent and child words");
            observe_pixels(owned_draw, vertex, vertex ? first + 1.0f : first, arm, 0.25f);
            selector[1u + last] = 3u;
            notify_guest_gpu_write(reinterpret_cast<uint64_t>(selector + 1u), count * 4u);
            emit_owned_replay_fixture(replay_directory, owned_draw,
                vertex ? owned_program : plain_vs, vertex ? plain_ps : owned_program,
                vertex, wide8, guest, Bytes, reinterpret_cast<uint64_t>(selector + 1u),
                owned_program.scalar_pc, owned_program.load_pc, first);
            observe_pixels(owned_draw, vertex, vertex ? first + 1.0f : first, arm, 0.25f);
            data[8u + last] = std::bit_cast<uint32_t>(second);
            notify_guest_gpu_write(guest + 32u, count * 4u);
            observe_pixels(owned_draw, vertex, vertex ? second + 1.0f : second, arm, 0.25f);
            DrawItem next_owned;
            const bool next_made = realize_draw_item(owned_state, &owned_state.draws[0],
                3u, 64u, false, next_owned);
            const auto next_table = vertex ? next_owned.vrt : next_owned.prt;
            check(next_made && next_table && next_table->by_fetch_pc(owned_program.load_pc) &&
                  next_table->by_fetch_pc(owned_program.load_pc)->gpu_addr == guest + 48u, arm,
                  "next real draw uses changed highest parent word to select the new child bytes");
            if (next_made) observe_pixels(next_owned, vertex,
                vertex ? second + 1.5f : second, arm, 0.375f);
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
