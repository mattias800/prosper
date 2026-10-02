// #4017: numeric 32 and 64 must never implicitly become recompile_fragment's bool Wave32 mode.
// The CLI companion drives both production regeneration routes with the fixture written here.
#include "../tools/gpu_replay/fragment_wave_size.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace prosper;

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++failures; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static_assert(!std::is_convertible_v<tools::ReplayFragmentWaveSize, bool>);
static_assert(!std::is_convertible_v<tools::ReplayFragmentWaveSize, uint32_t>);
static_assert(!std::is_convertible_v<tools::ReplayFragmentWaveSelection, bool>);

namespace {

// Project-owned gfx1030 synthetic ISA. The vertex words are the existing fullscreen-triangle
// fixture from test_recompiled_shaders. No guest/captured game shader bytes are used.
constexpr uint32_t vertex_words[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u, 0x7e0c02f2u,
    0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u,
    0xf80008cfu, 0x04030201u, 0xbf810000u,
};
constexpr uint32_t fragment_words[] = {
    0xd7660000u, 0x00010081u, // pc0: v_mbcnt_hi_u32_b32 v0, 1, 0
    0x7e000d00u,              // pc2: v_cvt_f32_u32 v0, v0 (tap here)
    0x7e0202f2u,              // pc3: v_mov_b32 v1, 1.0
    0x7e040280u,              // pc4: v_mov_b32 v2, 0
    0x7e0602f2u,              // pc5: v_mov_b32 v3, 1.0
    0xf800180fu, 0x03020100u, // pc6: exp mrt0 v0, v1, v2, v3 done vm
    0xbf810000u,
};
constexpr uint32_t stored_fragment_words[] = {
    0xd7660004u, 0x00010081u, // dead v4 high-half MBCNT: stored width is not launch provenance
    0x7e000280u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
    0xf800180fu, 0x03020100u, 0xbf810000u,
};

struct Instruction {
    uint32_t opcode = 0;
    std::vector<uint32_t> operands;
};

// A deliberately small structural reader, not the translator's internal builder/proof helpers.
// Follow the actual Location=0 Output store's red component backwards through typed SSA. An
// unrelated/dead lane mask or BitCount elsewhere in the module cannot satisfy this oracle.
struct Module {
    std::map<uint32_t, Instruction> definitions;
    std::map<uint32_t, uint32_t> builtins, locations;
    std::vector<Instruction> stores;

    bool read(const std::vector<uint32_t>& words) {
        if (words.size() < 5 || words[0] != 0x07230203u) return false;
        for (size_t pc = 5; pc < words.size();) {
            const uint32_t count = words[pc] >> 16, op = words[pc] & 0xffffu;
            if (!count || count > words.size() - pc) return false;
            Instruction inst{op, {words.begin() + pc + 1, words.begin() + pc + count}};
            const auto& a = inst.operands;
            if (op == 71 && a.size() == 3) { // OpDecorate
                if (a[1] == 11) builtins[a[0]] = a[2];
                if (a[1] == 30) locations[a[0]] = a[2];
            }
            if (op == 62) stores.push_back(inst);
            const bool type = op == 21 || op == 22 || op == 23 || op == 32;
            const bool result = op == 41 || op == 42 || op == 43 || op == 59 || op == 61 ||
                op == 80 || op == 112 || op == 124 || op == 128 || op == 130 || op == 169 ||
                op == 170 || op == 174 || op == 196 || op == 199 || op == 205;
            if (type && !a.empty()) definitions[a[0]] = inst;
            else if (result && a.size() >= 2) definitions[a[1]] = inst;
            pc += count;
        }
        return true;
    }

    const Instruction* get(uint32_t id) const {
        const auto it = definitions.find(id);
        return it == definitions.end() ? nullptr : &it->second;
    }

    bool unsigned32(uint32_t type) const {
        const auto* d = get(type);
        return d && d->opcode == 21 && d->operands.size() == 3 &&
               d->operands[1] == 32 && d->operands[2] == 0;
    }

    bool constant(uint32_t id, uint32_t value) const {
        const auto* d = get(id);
        return d && d->opcode == 43 && d->operands.size() == 3 &&
               unsigned32(d->operands[0]) && d->operands[2] == value;
    }

    bool lane_load(uint32_t id) const {
        const auto* d = get(id);
        if (!d || d->opcode != 61 || d->operands.size() != 3 ||
            !unsigned32(d->operands[0])) return false;
        const auto builtin = builtins.find(d->operands[2]);
        return builtin != builtins.end() && builtin->second == 41; // SubgroupLocalInvocationId
    }

    void ancestors(uint32_t id, std::set<uint32_t>& ids) const {
        if (!ids.insert(id).second) return;
        const auto* d = get(id);
        if (!d) return;
        switch (d->opcode) {
            case 80: case 112: case 124: case 128: case 130: case 169:
            case 170: case 174: case 196: case 199: case 205:
                for (size_t i = 2; i < d->operands.size(); ++i)
                    ancestors(d->operands[i], ids);
                break;
            default: break; // constants/types/pointers are not data operands
        }
    }

    bool live_high_half_mask(uint32_t width) const {
        uint32_t red = 0;
        size_t mrt0_stores = 0;
        for (const auto& store : stores) {
            if (store.operands.size() != 2) continue;
            const auto loc = locations.find(store.operands[0]);
            const auto* variable = get(store.operands[0]);
            if (loc == locations.end() || loc->second != 0 || !variable ||
                variable->opcode != 59 || variable->operands.size() < 3 ||
                variable->operands[2] != 3) continue; // Output, not an input at Location 0
            const auto* value = get(store.operands[1]);
            if (!value || value->opcode != 80 || value->operands.size() != 6) return false;
            red = value->operands[2];
            ++mrt0_stores;
        }
        if (mrt0_stores != 1) return false;
        std::set<uint32_t> live;
        ancestors(red, live);
        uint32_t lane_mask = 0;
        bool bitcount = false, convert = false;
        for (uint32_t id : live) {
            const auto* d = get(id);
            if (!d) continue;
            const auto& a = d->operands;
            if (d->opcode == 205 && a.size() == 3 && unsigned32(a[0])) bitcount = true;
            if (d->opcode == 112 && a.size() == 3) convert = true;
            if (d->opcode == 199 && a.size() == 4 && unsigned32(a[0]) &&
                ((lane_load(a[2]) && constant(a[3], width - 1)) ||
                 (lane_load(a[3]) && constant(a[2], width - 1)))) lane_mask = id;
        }
        bool high_half = false;
        for (uint32_t id : live) {
            const auto* d = get(id);
            if (d && d->opcode == 174 && d->operands.size() == 4 &&
                d->operands[2] == lane_mask && constant(d->operands[3], 32)) high_half = true;
        }
        return lane_mask && high_half && bitcount && convert;
    }
};

void clear_tap() {
#ifdef _WIN32
    _putenv_s("PROSPER_FS_TAP", "");
#else
    unsetenv("PROSPER_FS_TAP");
#endif
}

gpu::GpuCaptureRawShaderVersion raw_shader(const uint32_t* words, size_t count) {
    gpu::GpuCaptureRawShaderVersion result;
    result.words.assign(words, words + count);
    result.has_endpgm = true;
    result.content_hash = gpu::gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(words), count * sizeof(uint32_t));
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 3 || std::strcmp(argv[1], "--write-fixture") != 0)) {
        std::fprintf(stderr, "usage: %s [--write-fixture DIRECTORY]\n", argv[0]);
        return 2;
    }
    std::printf("== test_replay_fragment_width ==\n");
    tools::ReplayFragmentWaveSelection selection{tools::ReplayFragmentWaveSize::Wave32, true};
    CHECK(tools::parse_replay_fragment_wave_size(nullptr, selection) &&
              selection.size == tools::ReplayFragmentWaveSize::Wave64 &&
              !selection.wave32() && !selection.explicit_override,
          "absent override resets selection to legacy-default Wave64, not captured ABI");
    const char* positive32[] = {"32", "0x20", "0X20", "040"};
    const char* positive64[] = {"64", "0x40", "0X40", "0100"};
    for (const char* text : positive32) {
        CHECK(tools::parse_replay_fragment_wave_size(text, selection) && selection.wave32() &&
                  selection.size == tools::ReplayFragmentWaveSize::Wave32 && selection.explicit_override,
              "complete base-0 Wave32 override selects Boolean true");
    }
    for (const char* text : positive64) {
        CHECK(tools::parse_replay_fragment_wave_size(text, selection) && !selection.wave32() &&
                  selection.size == tools::ReplayFragmentWaveSize::Wave64 && selection.explicit_override,
              "complete base-0 Wave64 override selects Boolean false");
    }
    const char* invalid[] = {"", "0", "16", "128", "032", "064", "08", "0x", "0xgg",
        "+32", "-32", " 32", "32 ", "\t64", "64\n", "32junk", "64:32", "32.0",
        "4294967328", "18446744073709551616", "0x10000000000000020"};
    for (const char* text : invalid) {
        for (const auto size : {tools::ReplayFragmentWaveSize::Wave32,
                                tools::ReplayFragmentWaveSize::Wave64}) {
            selection = {size, size == tools::ReplayFragmentWaveSize::Wave32};
            const auto before = selection;
            CHECK(!tools::parse_replay_fragment_wave_size(text, selection) &&
                      selection.size == before.size && selection.explicit_override == before.explicit_override,
                  "malformed width rejects without mutating either prior selection");
        }
    }

    for (const char* text : {static_cast<const char*>(nullptr), "32", "64"}) {
        CHECK(tools::parse_replay_fragment_wave_size(text, selection), "resolver request parses");
        const auto requested = selection;
        for (const auto captured : {std::pair{false, false}, std::pair{false, true},
                                    std::pair{true, false}, std::pair{true, true}}) {
            const auto resolved = tools::resolve_replay_fragment_wave_size(
                requested, captured.first, captured.second);
            const bool expected32 = text ? std::strcmp(text, "32") == 0
                                         : captured.first && captured.second;
            CHECK(resolved.wave32() == expected32 && resolved.explicit_override == (text != nullptr),
                  "override beats captured ABI; captured ABI beats unknown legacy Wave64 default");
            CHECK(selection.size == requested.size &&
                      selection.explicit_override == requested.explicit_override,
                  "per-draw resolution does not mutate the process-wide request");
        }
    }

    const tools::ReplayFragmentWaveSelection stale_non_override{
        tools::ReplayFragmentWaveSize::Wave32, false};
    for (bool unavailable_width : {false, true}) {
        const auto resolved = tools::resolve_replay_fragment_wave_size(
            stale_non_override, false, unavailable_width);
        CHECK(!resolved.wave32() && !resolved.explicit_override,
              "unavailable ABI resets stale non-override selection to canonical legacy Wave64");
    }

    // This tiny stream cannot match the unchanged 3142-dword legacy capture exception.
    clear_tap();
    std::vector<gpu::Rdna2Inst> decoded;
    CHECK(gpu::rdna2_walk(fragment_words, std::size(fragment_words), decoded) ==
              std::size(fragment_words) && decoded.size() == 7 &&
              decoded[0].fmt == gpu::Rdna2Format::VOP3 && decoded[0].opcode == 0x366 &&
              decoded[0].dst.kind == gpu::OperandKind::VGPR && decoded[0].dst.value == 0 &&
              decoded[0].src[0].kind == gpu::OperandKind::InlineInt && decoded[0].src[0].value == 1 &&
              decoded[1].pc == 2 && decoded[1].opcode == 6,
          "fixture decodes real high-half numeric MBCNT followed by tapped u32-to-float conversion");
    for (const char* text : {static_cast<const char*>(nullptr), "32", "64"}) {
        CHECK(tools::parse_replay_fragment_wave_size(text, selection), "semantic control width parses");
        for (const auto captured : {std::pair{false, false}, std::pair{true, false},
                                    std::pair{true, true}}) {
            const uint32_t width = text ? (std::strcmp(text, "32") == 0 ? 32u : 64u)
                                        : (captured.first && captured.second ? 32u : 64u);
            const auto resolved = tools::resolve_replay_fragment_wave_size(
                selection, captured.first, captured.second);
            const auto spirv = gpu::recompile_fragment(fragment_words, std::size(fragment_words),
                nullptr, nullptr, UINT32_MAX, nullptr, resolved.wave32());
            CHECK(!spirv.empty() && gpu::fragment_spirv_required_subgroup_size(spirv) == width,
                  "resolved captured/requested width reaches the real fragment compiler");
            Module module;
            CHECK(module.read(spirv) && module.live_high_half_mask(width),
                  "actual MRT0 red depends on typed live width mask, high-half compare and BitCount");
        }
    }

    if (argc == 3) {
        const std::filesystem::path directory(argv[2]);
        gpu::GpuCaptureFile capture;
        capture.metadata.width = capture.metadata.height = 1;
        capture.raw_shader_versions.push_back(raw_shader(vertex_words, std::size(vertex_words)));
        capture.raw_shader_versions.push_back(raw_shader(fragment_words, std::size(fragment_words)));
        gpu::GpuCapturedDraw draw;
        draw.vs = gpu::recompile_vertex(vertex_words, std::size(vertex_words));
        // Stored width deliberately opposes the captured/default choice while its MRT0 value
        // remains distinct. Inferring launch width from a stored marker, or silently keeping the
        // stored shader, must fail the CLI's live SSA/output oracle independently of notices.
        const auto stored32 = gpu::recompile_fragment(stored_fragment_words,
            std::size(stored_fragment_words), nullptr, nullptr, UINT32_MAX, nullptr, true);
        const auto stored64 = gpu::recompile_fragment(stored_fragment_words,
            std::size(stored_fragment_words), nullptr, nullptr, UINT32_MAX, nullptr, false);
        CHECK(!draw.vs.empty() && !stored32.empty() && !stored64.empty() &&
                  gpu::fragment_spirv_required_subgroup_size(stored32) == 32 &&
                  gpu::fragment_spirv_required_subgroup_size(stored64) == 64,
              "stored synthetic modules retain opposing width markers but a distinct MRT0 value");
        draw.fs = stored32;
        draw.draw_index = 7; // not compact item 0: pin semantic draw selection as well
        draw.command_order = 1;
        draw.vertex_count = draw.raw_draw_count = 3;
        draw.color0_width = draw.color0_height = 1;
        // Neither synthetic FS consumes interpolated/system VGPR inputs. Preserve the explicit
        // no-programmed-input state; marking zero mappings present is invalid capture ABI.
        draw.has_pixel_inputs = draw.has_system_inputs = false;
        draw.vs_raw_shader_index = 0;
        draw.fs_raw_shader_index = 1;
        capture.draws.push_back(draw);
        capture.operations.push_back({gpu::SubmitOperationKind::Draw, 7, 1, true});
        auto write = [&](const char* name, const gpu::GpuCaptureFile& fixture) {
            std::string error;
            CHECK(gpu::write_gpu_capture((directory / name).string(), fixture, error),
                  "fixture mode writes width states with production capture serialization");
            if (!error.empty()) std::fprintf(stderr, "fixture %s: %s\n", name, error.c_str());
        };
        write("fragment-width.prgcap", capture); // canonical unavailable current format
        capture.draws[0].fragment_wave_config_available = true;
        capture.draws[0].ps_wave32 = true;
        capture.draws[0].fs = stored64;
        write("fragment-width-captured32.prgcap", capture);
        auto no_raw = capture;
        no_raw.draws[0].vs_raw_shader_index = no_raw.draws[0].fs_raw_shader_index = UINT32_MAX;
        no_raw.raw_shader_versions.clear();
        write("fragment-width-no-raw.prgcap", no_raw);
        capture.draws[0].ps_wave32 = false;
        capture.draws[0].fs = stored32;
        write("fragment-width-captured64.prgcap", capture);
        auto second = capture.draws[0];
        second.draw_index = 11;
        second.command_order = 2;
        capture.draws[0].ps_wave32 = true;
        capture.draws[0].fs = stored64;
        capture.draws.push_back(second);
        capture.operations.push_back({gpu::SubmitOperationKind::Draw, 11, 2, true});
        write("fragment-width-mixed.prgcap", capture);

        // Realized-draw selection must not broaden the unrelated v61 failed-stage retry contract.
        // The override is dormant there, even when it opposes the captured failure ABI or is bad.
        gpu::GpuCaptureFile failed;
        failed.metadata.width = failed.metadata.height = 1;
        failed.failure_diagnostics_available = true;
        failed.raw_shader_versions.push_back(raw_shader(fragment_words, std::size(fragment_words)));
        failed.operations.push_back({gpu::SubmitOperationKind::Draw, 7, 1, false});
        gpu::GpuCapturedOperationFailure failure;
        failure.source_index = 7;
        failure.command_order = 1;
        failure.reason = gpu::RealizationFailureReason::ShaderRecompile;
        failure.vertex_retry_config_available = true; // prerequisite for the v61 fragment tail
        failure.fragment_retry_config_available = true;
        gpu::GpuCapturedStageDiagnostic stage;
        stage.stage = gpu::ShaderProgramStage::Fragment;
        stage.program_addr = 0x7000u;
        stage.raw_shader_index = 0;
        failure.stages.push_back(stage);
        failed.failure_diagnostics.push_back(failure);
        failed.failure_diagnostics[0].ps_wave32 = true;
        write("fragment-width-failed32.prgcap", failed);
        failed.failure_diagnostics[0].ps_wave32 = false;
        write("fragment-width-failed64.prgcap", failed);
    }
    std::printf("== %s (%d failures) ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
