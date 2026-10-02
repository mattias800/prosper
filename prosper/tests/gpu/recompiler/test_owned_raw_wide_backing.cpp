// Real fold, module cache and capture materializer controls; no device or GPU execution.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/state/render_state.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* label) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", label);
    failures += !ok;
}

int main(int argc, char** argv) {
    std::filesystem::path replay_fixture_directory;
    if (argc == 3 && std::string(argv[1]) == "--emit-replay-fixtures") {
        replay_fixture_directory = argv[2];
        std::error_code directory_error;
        std::filesystem::create_directories(replay_fixture_directory, directory_error);
        if (directory_error) {
            std::fprintf(stderr, "cannot create replay fixture directory: %s\n",
                         directory_error.message().c_str());
            return 2;
        }
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--emit-replay-fixtures DIRECTORY]\n", argv[0]);
        return 2;
    }
    for (bool wide8 : {false, true}) {
        const uint32_t size = wide8 ? 32u : 16u;
        const uint32_t last = wide8 ? 7u : 3u;
        alignas(16) std::array<uint32_t, 16> parent{};
        alignas(16) std::array<uint32_t, 128> child{};
        for (size_t i = 0; i < child.size(); ++i) child[i] = 100u + static_cast<uint32_t>(i);
        parent[1u + last] = 2u;
        const auto parent_address = reinterpret_cast<uint64_t>(parent.data());
        const auto child_address = reinterpret_cast<uint64_t>(child.data());
        std::array<uint32_t, 4> user{static_cast<uint32_t>(parent_address),
            static_cast<uint32_t>(parent_address >> 32u), static_cast<uint32_t>(child_address),
            static_cast<uint32_t>(child_address >> 32u)};
        const std::vector<uint32_t> code{
            wide8 ? 0xf40c0600u : 0xf4080600u, 0xfa000004u,
            0xbe800380u, 0xbe810380u,
            0x8f6b8400u | (24u + last), 0x876bff6bu, 0x1f0u,
            wide8 ? 0xf40c0201u : 0xf4080201u, 0xd6000000u,
            wide8 ? 0x7e00020fu : 0x7e00020bu,
            0xf80008cfu, 0u, 0xbf810000u};
        ShaderResourceTable table;
        add_compute_buffer_resources(table, code.data(), code.size(), user.data(), user.size());
        assign_convention_bindings(table, 2u);
        const auto* source = table.by_fetch_pc(0u);
        const auto* selected = table.by_fetch_pc(7u);
        check(source && selected && table.owned_host_data.size() == 1u &&
              table.owned_raw_snapshot_requirements ==
                  std::vector<std::pair<uint32_t, uint32_t>>{{0u, size}} &&
              std::count_if(table.resources.begin(), table.resources.end(),
                  [](const auto& r) { return r.fetch_pc == 0u; }) == 1,
              "actual realization publishes one authoritative owned parent and required exact width");
        if (!source || !selected) continue;
        uint32_t parent_high = UINT32_MAX;
        if (source->host_data && source->host_data_size == size)
            std::memcpy(&parent_high, source->host_data + last * 4u, 4u);
        check(parent_high == 2u && source->gpu_addr == parent_address + 4u &&
              selected->gpu_addr == child_address + 32u && selected->size == size,
              "highest parent word and child highest-data range derive from the same observed fold");
        // Both paths join ON the authenticated parent. Control before that completed
        // definition cannot invalidate an already resolved child selector dependency.
        auto joined_parent = code;
        joined_parent.insert(joined_parent.begin(),
            {0xbf068080u, 0xbf840001u, 0xbe940380u});
        std::vector<Rdna2Inst> joined_ins;
        rdna2_walk(joined_parent.data(), joined_parent.size(), joined_ins);
        std::vector<uint32_t> joined_sources;
        const auto joined_children = rdna2_proven_raw_register_wide_data_loads(
            joined_ins, &joined_sources);
        ShaderResourceTable joined_table;
        add_compute_buffer_resources(joined_table, joined_parent.data(), joined_parent.size(),
                                     user.data(), user.size());
        assign_convention_bindings(joined_table, 2u);
        const auto* joined_source = joined_table.by_fetch_pc(3u);
        const auto* joined_child = joined_table.by_fetch_pc(10u);
        uint32_t joined_high = UINT32_MAX;
        if (joined_source && joined_source->host_data && joined_source->host_data_size == size)
            std::memcpy(&joined_high, joined_source->host_data + last * 4u, 4u);
        check(joined_children == std::vector<uint32_t>{10u} &&
              joined_sources == std::vector<uint32_t>{3u} && joined_high == 2u &&
              joined_child && joined_child->gpu_addr == child_address + 32u &&
              joined_table.owned_raw_snapshot_requirements ==
                  std::vector<std::pair<uint32_t, uint32_t>>{{3u, size}} &&
              !recompile_vertex(joined_parent.data(), joined_parent.size(), &joined_table).empty(),
              "control joining on a complete owned parent preserves highest-word selector and actual child module");
        auto bypass_parent = joined_parent;
        bypass_parent[1] = 0xbf840003u; // target pc5, AFTER the parent at pc3
        std::vector<Rdna2Inst> bypass_ins;
        rdna2_walk(bypass_parent.data(), bypass_parent.size(), bypass_ins);
        check(rdna2_proven_raw_register_wide_data_loads(bypass_ins).empty(),
              "branch bypassing the owned parent cannot resolve a later selector");
        auto changed_parent_pointer = joined_parent;
        changed_parent_pointer.insert(changed_parent_pointer.begin() + 3, 0xbe800380u);
        std::vector<Rdna2Inst> changed_parent_ins;
        rdna2_walk(changed_parent_pointer.data(), changed_parent_pointer.size(), changed_parent_ins);
        check(rdna2_proven_raw_register_wide_data_loads(changed_parent_ins).empty(),
              "parent pointer overwrite before its read cannot resolve a later selector");
        auto intervening_control = joined_parent;
        intervening_control.insert(intervening_control.begin() + 7, 0xbf840000u);
        std::vector<Rdna2Inst> intervening_ins;
        rdna2_walk(intervening_control.data(), intervening_control.size(), intervening_ins);
        check(rdna2_proven_raw_register_wide_data_loads(intervening_ins).empty(),
              "intervening conditional control remains outside the owned selector slice");

        parent[1u + last] = 3u;
        uint32_t latched_high = UINT32_MAX;
        if (source->host_data && source->host_data_size == size)
            std::memcpy(&latched_high, source->host_data + last * 4u, 4u);
        check(latched_high == 2u && selected->gpu_addr == child_address + 32u,
              "post-realization highest-parent mutation cannot recompute the selected child range");
        uint64_t valid_identity = 0u;
        const auto valid = recompile_graphics_shader_cached(ShaderProgramStage::Vertex,
            code.data(), code.size(), &table, nullptr, nullptr, &valid_identity);
        check(!valid.empty() && !recompile_vertex(code.data(), code.size(), &table).empty(),
              "complete owned parent and register child compile cold and cached direct vertex modules");
        const auto refuse = [&](ShaderResourceTable malformed, const char* label) {
            uint64_t rejected_identity = 0u, recovered_identity = 0u;
            const bool cold = recompile_vertex(code.data(), code.size(), &malformed).empty();
            const bool warm = recompile_graphics_shader_cached(ShaderProgramStage::Vertex,
                code.data(), code.size(), &malformed, nullptr, nullptr, &rejected_identity).empty();
            const auto recovered = recompile_graphics_shader_cached(ShaderProgramStage::Vertex,
                code.data(), code.size(), &table, nullptr, nullptr, &recovered_identity);
            check(cold && warm && !recovered.empty() && recovered_identity == valid_identity, label);
        };
        auto absent = table;
        absent.resources.erase(absent.resources.begin());
        refuse(absent, "absent parent refuses cold and warm before valid recovery");
        auto short_parent = table;
        short_parent.resources.front().host_data_size = size - 4u;
        refuse(short_parent, "missing highest parent word refuses cold and warm before recovery");
        auto wrong_width = table;
        wrong_width.resources.front().size = wide8 ? 16u : 32u;
        wrong_width.resources.front().host_data_size = wrong_width.resources.front().size;
        wrong_width.resources.front().owned_raw_snapshot_bytes = wrong_width.resources.front().size;
        refuse(wrong_width, "opposite complete x4 or x8 width cannot substitute decoded source width");
        auto poisoned = table;
        poisoned.resources.front().format = DataFormat::Float32;
        poisoned.resources.push_back(*source);
        refuse(poisoned, "later valid duplicate cannot hide first poisoned exact-PC source");

        DrawItem draw;
        draw.vs = valid;
        draw.vs_guest_addr = reinterpret_cast<uint64_t>(code.data());
        draw.vrt = std::make_shared<ShaderResourceTable>(table);
        std::string error;
        uint64_t planned = 0u;
        auto ownerless = draw;
        ownerless.vrt = std::make_shared<ShaderResourceTable>(table);
        ownerless.vrt->owned_host_data.clear();
        check(!preflight_gpu_capture_draw_resources(ownerless, 1u << 20u, planned, error),
              "capture interval planning refuses a required hosted parent without its owner");
        const auto copy_range = [](uint64_t requested, uint8_t* dst, size_t want,
                                   uint64_t base, const void* data, size_t length) {
            if (requested < base || requested >= base + length) return size_t{0};
            const size_t offset = static_cast<size_t>(requested - base);
            const size_t copied = std::min(want, length - offset);
            std::memcpy(dst, static_cast<const uint8_t*>(data) + offset, copied);
            return copied;
        };
        // Both inputs deliberately name the same guest interval. Planning owns the old parent
        // separately, while the child interval can supply newer guest bytes. Removing the owner
        // after planning must fail at table copying rather than choose that unrelated observation.
        alignas(16) std::array<uint32_t, 128> overlap{};
        overlap[8u + last] = 2u;
        const auto overlap_address = reinterpret_cast<uint64_t>(overlap.data());
        const auto overlap_parent_address = overlap_address + 28u;
        const std::array<uint32_t, 4> overlap_user{
            static_cast<uint32_t>(overlap_parent_address),
            static_cast<uint32_t>(overlap_parent_address >> 32u),
            static_cast<uint32_t>(overlap_address),
            static_cast<uint32_t>(overlap_address >> 32u)};
        ShaderResourceTable overlap_table;
        add_compute_buffer_resources(overlap_table, code.data(), code.size(),
                                     overlap_user.data(), overlap_user.size());
        assign_convention_bindings(overlap_table, 2u);
        DrawItem overlap_draw = draw;
        overlap_draw.vrt = std::make_shared<ShaderResourceTable>(overlap_table);
        overlap_draw.vs = recompile_vertex(code.data(), code.size(), overlap_draw.vrt.get());
        const auto* overlap_source = overlap_table.by_fetch_pc(0u);
        const auto* overlap_child = overlap_table.by_fetch_pc(7u);
        check(!overlap_draw.vs.empty() && overlap_source && overlap_child &&
              overlap_source->gpu_addr == overlap_address + 32u &&
              overlap_child->gpu_addr == overlap_source->gpu_addr &&
              overlap_source->size == overlap_child->size &&
              preflight_gpu_capture_draw_resources(overlap_draw, 1u << 20u, planned, error),
              "overlapping owned parent and guest child are valid before capture planning");
        overlap[8u + last] = 3u;
        bool lose_owner = false;
        unsigned owner_losses = 0u;
        const CaptureMemoryReader overlap_reader = [&](uint64_t address, uint8_t* dst, size_t want) {
            if (address >= overlap_draw.vs_guest_addr &&
                address < overlap_draw.vs_guest_addr + code.size() * 4u) {
                if (lose_owner && !overlap_draw.vrt->owned_host_data.empty()) {
                    overlap_draw.vrt->owned_host_data.clear();
                    ++owner_losses;
                }
                return copy_range(address, dst, want, overlap_draw.vs_guest_addr,
                                  code.data(), code.size() * 4u);
            }
            return copy_range(address, dst, want, overlap_address, overlap.data(), sizeof(overlap));
        };
        GpuCaptureFile overlap_capture;
        GpuReplayFrame overlap_replay;
        const bool overlap_captured = capture_draw_items({overlap_draw}, {}, overlap_reader,
                                                         overlap_capture, error);
        const bool overlap_materialized = overlap_captured &&
            materialize_gpu_replay(overlap_capture, overlap_replay, error);
        uint32_t overlap_parent_high = UINT32_MAX, overlap_child_high = UINT32_MAX;
        if (overlap_materialized && overlap_replay.items.size() == 1u && overlap_replay.items[0].vrt) {
            if (const auto* r = overlap_replay.items[0].vrt->by_fetch_pc(0u); r && r->host_data)
                std::memcpy(&overlap_parent_high, r->host_data + last * 4u, 4u);
            if (const auto* r = overlap_replay.items[0].vrt->by_fetch_pc(7u); r && r->host_data)
                std::memcpy(&overlap_child_high, r->host_data + last * 4u, 4u);
        }
        check(overlap_materialized && overlap_parent_high == 2u && overlap_child_high == 3u,
              "capture keeps distinct observed parent and current child bytes at the same guest address");
        lose_owner = true;
        GpuCaptureFile refused_owner_capture;
        const bool late_owner_capture = capture_draw_items({overlap_draw}, {}, overlap_reader,
                                                           refused_owner_capture, error);
        check(owner_losses == 1u && !late_owner_capture &&
              error == "code-required wide scalar source has no complete owned capture backing",
              "table copying refuses owner loss after successful planning instead of rereading overlapping guest bytes");
        unsigned parent_reads = 0u;
        const CaptureMemoryReader reader = [&](uint64_t address, uint8_t* dst, size_t want) {
            if (address >= parent_address && address < parent_address + sizeof(parent)) ++parent_reads;
            if (const auto n = copy_range(address, dst, want, draw.vs_guest_addr,
                                          code.data(), code.size() * 4u)) return n;
            return copy_range(address, dst, want, child_address, child.data(), sizeof(child));
        };
        GpuCaptureFile capture, decoded;
        std::vector<uint8_t> encoded;
        GpuReplayFrame replay;
        const bool captured = capture_draw_items({draw}, {}, reader, capture, error);
        const bool roundtrip = captured && serialize_gpu_capture(capture, encoded, error) &&
            deserialize_gpu_capture(encoded, decoded, error) && materialize_gpu_replay(decoded, replay, error);
        if (!roundtrip) std::printf("capture error: %s\n", error.c_str());
        uint32_t restored_high = UINT32_MAX, restored_child_high = UINT32_MAX;
        if (roundtrip && replay.items.size() == 1u && replay.items[0].vrt) {
            if (const auto* r = replay.items[0].vrt->by_fetch_pc(0u); r && r->host_data)
                std::memcpy(&restored_high, r->host_data + last * 4u, 4u);
            if (const auto* r = replay.items[0].vrt->by_fetch_pc(7u); r && r->host_data)
                std::memcpy(&restored_child_high, r->host_data + last * 4u, 4u);
        }
        check(roundtrip && parent_reads == 0u && restored_high == 2u &&
              restored_child_high == child[8u + last],
              "current capture owns highest selector and separate highest child word without guest reread");
        if (!roundtrip) continue;
        // The same owned parent/selected child feeds a real PS zero relation. Child words are
        // nonzero subnormal bit patterns; the incoming mode fixtures own its category oracle.
        auto fragment_code = code;
        fragment_code.resize(fragment_code.size() - 3u);
        fragment_code.insert(fragment_code.end(), {
            0x7c040080u, 0xd5010001u, 128u | (242u << 9u) | (106u << 18u),
            0xf800180fu, 0x01010101u, 0xbf810000u});
        std::vector<uint64_t> mode_identities;
        std::vector<std::vector<uint32_t>> mode_modules;
        for (const auto requested : {FragmentFloatMode{true, 0u}, FragmentFloatMode{true, 16u},
                                     FragmentFloatMode{true, 0xabu}, FragmentFloatMode{}}) {
            GpuState launch;
            if (requested.available)
                launch.sh[prosper::agc::Pm4::SPI_SHADER_PGM_RSRC1_PS] =
                    uint32_t(requested.value) << prosper::agc::Pm4::SPI_SHADER_PGM_RSRC1_PS_FLOAT_MODE_SHIFT;
            const auto mode = extract_render_state(launch).ps_float_mode;
            uint64_t identity = 0u, warm_identity = 0u, refused_identity = 99u, recovered_identity = 0u;
            const auto direct = recompile_fragment(fragment_code.data(), fragment_code.size(), &table,
                nullptr, UINT32_MAX, nullptr, false, {RecompileDiagnosticStage::Fragment, 0u}, mode);
            const auto module = recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
                fragment_code.data(), fragment_code.size(), &table, nullptr, nullptr, &identity,
                false, 0u, false, {}, mode);
            const auto warm = recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
                fragment_code.data(), fragment_code.size(), &table, nullptr, nullptr, &warm_identity,
                false, 0u, false, {}, mode);
            check(mode == requested && !direct.empty() && module && module == warm &&
                  identity && identity == warm_identity && *module == direct,
                  "producing register mode and owned PS input share exact cold/warm compiler authority");
            const auto refused = recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
                fragment_code.data(), fragment_code.size(), &short_parent, nullptr, nullptr, &refused_identity,
                false, 0u, false, {}, mode);
            const auto recovered = recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
                fragment_code.data(), fragment_code.size(), &table, nullptr, nullptr, &recovered_identity,
                false, 0u, false, {}, mode);
            check(refused && refused->empty() && refused_identity != identity &&
                  recovered == module && recovered_identity == identity &&
                  recompile_fragment(fragment_code.data(), fragment_code.size(), &short_parent,
                      nullptr, UINT32_MAX, nullptr, false, {RecompileDiagnosticStage::Fragment, 0u}, mode).empty(),
                  "known or unknown mode cannot admit short owned PS backing after a warm valid entry");
            mode_identities.push_back(identity);
            mode_modules.push_back(direct);
            DrawItem combined_draw = draw;
            combined_draw.fs = direct;
            combined_draw.fs_guest_addr = reinterpret_cast<uint64_t>(fragment_code.data());
            combined_draw.prt = std::make_shared<ShaderResourceTable>(table);
            combined_draw.ps_float_mode = mode;
            combined_draw.fragment_wave_config_available = true;
            combined_draw.ps_wave32 = false;
            const CaptureMemoryReader combined_reader = [&](uint64_t address, uint8_t* dst, size_t want) {
                if (const auto n = copy_range(address, dst, want, combined_draw.fs_guest_addr,
                                              fragment_code.data(), fragment_code.size() * 4u)) return n;
                return reader(address, dst, want);
            };
            GpuCaptureFile combined_capture, combined_decoded;
            GpuReplayFrame combined_replay;
            std::vector<uint8_t> combined_bytes;
            const bool combined_ok = capture_draw_items({combined_draw}, {}, combined_reader, combined_capture, error) &&
                serialize_gpu_capture(combined_capture, combined_bytes, error) &&
                deserialize_gpu_capture(combined_bytes, combined_decoded, error) &&
                materialize_gpu_replay(combined_decoded, combined_replay, error);
            const auto* restored_parent = combined_ok && combined_replay.items.size() == 1u && combined_replay.items[0].prt
                ? combined_replay.items[0].prt->by_fetch_pc(0u) : nullptr;
            const auto* restored_child = combined_ok && combined_replay.items.size() == 1u && combined_replay.items[0].prt
                ? combined_replay.items[0].prt->by_fetch_pc(7u) : nullptr;
            check(combined_ok && combined_decoded.format_version == 67u &&
                  combined_decoded.draws.front().ps_float_mode == mode && combined_replay.items[0].ps_float_mode == mode &&
                  combined_replay.items[0].fragment_wave_config_available && !combined_replay.items[0].ps_wave32 &&
                  restored_parent && restored_parent->host_data && restored_parent->owned_raw_snapshot_bytes == size &&
                  restored_parent->host_data_size == size && std::memcmp(restored_parent->host_data, source->host_data, size) == 0 &&
                  restored_child && restored_child->host_data && restored_child->host_data_size >= size &&
                  std::memcmp(restored_child->host_data, child.data() + 8u, size) == 0,
                  "current capture retains every owned parent/child byte, exact width and full producing mode");
            auto combined_ownerless = combined_draw;
            combined_ownerless.prt = std::make_shared<ShaderResourceTable>(table);
            combined_ownerless.prt->owned_host_data.clear();
            check(!preflight_gpu_capture_draw_resources(combined_ownerless, 1u << 20u, planned, error),
                  "producing mode never replaces the missing owned PS capture owner");
        }
        std::sort(mode_identities.begin(), mode_identities.end());
        check(mode_identities.size() == 4u && std::adjacent_find(mode_identities.begin(), mode_identities.end()) == mode_identities.end() &&
              mode_modules[0] != mode_modules[1] && mode_modules[3] != mode_modules[0],
              "full mode availability/bits partition owned PS cache and preserve flush/preserve/unknown relation paths");
        const auto owned_tail_size = 4u + 4u * decoded.draws.front().vrt.resources.size();
        const auto transport_tail_size = 12u + decoded.draws.size();
        const auto flags_tail_size = 8u + 8u * decoded.draws.size();
        auto v65_bytes = encoded;
        v65_bytes.resize(v65_bytes.size() - flags_tail_size - transport_tail_size); v65_bytes[8] = 65u;
        GpuCaptureFile official65;
        check(deserialize_gpu_capture(v65_bytes, official65, error) && official65.format_version == 65u &&
              official65.draws.front().vrt.resources.front().resource.owned_raw_snapshot_bytes == size &&
              official65.draws.front().float_transport == FloatTransportConfig{},
              "genuine v65 retains the owned obligation but not unavailable transport authority");
        auto v64_bytes = v65_bytes;
        v64_bytes.resize(v64_bytes.size() - owned_tail_size);
        v64_bytes[8] = 64u;
        GpuCaptureFile official64;
        check(deserialize_gpu_capture(v64_bytes, official64, error) &&
              official64.format_version == 64u && official64.draws.size() == 1u &&
              !official64.draws.front().ps_float_mode.available &&
              official64.draws.front().ps_float_mode.value == 0u &&
              materialize_gpu_replay(official64, replay, error),
              "genuine official v64 retains unknown mode and derives owned source authority from raw code");
        auto v63_bytes = v64_bytes;
        v63_bytes.resize(v63_bytes.size() - 8u - 2u * decoded.draws.size());
        v63_bytes[8] = 63u;
        v63_bytes[9] = v63_bytes[10] = v63_bytes[11] = 0u;
        GpuCaptureFile official63;
        check(deserialize_gpu_capture(v63_bytes, official63, error) &&
              official63.format_version == 63u && official63.draws.size() == 1u &&
              !official63.draws.front().fragment_wave_config_available &&
              !official63.draws.front().ps_wave32 &&
              std::all_of(official63.draws.front().vrt.resources.begin(),
                          official63.draws.front().vrt.resources.end(), [](const auto& resource) {
                  return resource.resource.owned_raw_snapshot_bytes == 0u;
              }), "official v63 retains width availability without inventing an owned obligation");
        auto v62_bytes = v63_bytes;
        const auto v63_tail_size = 4u + decoded.draws.size();
        if (v62_bytes.size() >= v63_tail_size) {
            v62_bytes.resize(v62_bytes.size() - v63_tail_size);
            v62_bytes[8] = 62u;
            v62_bytes[9] = v62_bytes[10] = v62_bytes[11] = 0u;
        }
        auto bad_count = v65_bytes;
        if (bad_count.size() >= owned_tail_size) bad_count[bad_count.size() - owned_tail_size] ^= 1u;
        GpuCaptureFile invalid_format;
        check(!deserialize_gpu_capture(bad_count, invalid_format, error),
              "v65 reader refuses mismatched owned-obligation resource count");
        auto cut_width = v65_bytes;
        cut_width.pop_back();
        check(!deserialize_gpu_capture(cut_width, invalid_format, error),
              "v65 reader refuses a truncated owned-obligation width field");
        GpuCaptureFile v62;
        GpuReplayFrame legacy_replay;
        check(deserialize_gpu_capture(v62_bytes, v62, error) && v62.format_version == 62u &&
              std::all_of(v62.draws.front().vrt.resources.begin(), v62.draws.front().vrt.resources.end(),
                  [](const auto& r) { return r.resource.owned_raw_snapshot_bytes == 0u; }) &&
              materialize_gpu_replay(v62, legacy_replay, error),
              "v62 parsing preserves complete payloads and derives new source authority from code");
        auto tail_code = code;
        tail_code.insert(tail_code.end() - 3, {
            0xbeb21f00u, 0x803232ffu, 52u, 0x82333380u,
            0xbeb40394u, 0xbeb503ffu, 0x10005004u, 0xbeea0384u,
            0xf4280e19u, 0xd4000000u, 0x7e020238u});
        tail_code.insert(tail_code.end(), {7u, 11u, 13u, 17u, 19u});
        DrawItem tail_draw = draw;
        tail_draw.vs_guest_addr = reinterpret_cast<uint64_t>(tail_code.data());
        tail_draw.vs = recompile_vertex(tail_code.data(), tail_code.size(), &table);
        GpuCaptureFile tail_capture, tail_decoded;
        GpuReplayFrame tail_replay;
        std::vector<uint8_t> tail_encoded;
        const CaptureMemoryReader tail_reader = [&](uint64_t requested, uint8_t* dst, size_t want) {
            if (const auto n = copy_range(requested, dst, want, tail_draw.vs_guest_addr,
                                         tail_code.data(), tail_code.size() * 4u)) return n;
            return reader(requested, dst, want);
        };
        const bool tail_ok = !tail_draw.vs.empty() &&
            rdna2_recompile_code_span(tail_code.data(), tail_code.size()) == tail_code.size() &&
            capture_draw_items({tail_draw}, {}, tail_reader, tail_capture, error) &&
            serialize_gpu_capture(tail_capture, tail_encoded, error) &&
            deserialize_gpu_capture(tail_encoded, tail_decoded, error) &&
            materialize_gpu_replay(tail_decoded, tail_replay, error);
        if (!tail_ok) std::printf("owned inline-tail capture error: %s\n", error.c_str());
        check(tail_ok && tail_decoded.raw_shader_versions.size() == 1u &&
              tail_decoded.raw_shader_versions.front().words == tail_code,
              "owned source replay admits its proven post-END inline-data tail without decoding data as ISA");

        const auto source_index = static_cast<size_t>(source - table.resources.data());
        const auto emit_replay_fixture = [&](const GpuCaptureFile& fixture, const char* name) {
            if (replay_fixture_directory.empty()) return;
            const auto path = replay_fixture_directory /
                (std::string(wide8 ? "x8-" : "x4-") + name + ".prgcap");
            GpuCaptureFile parsed;
            const bool written = write_gpu_capture(path.string(), fixture, error);
            const bool read = written && read_gpu_capture(path.string(), parsed, error);
            check(read, "replay CLI input writes and parses before materializer refusal");
            std::printf("[replay-fixture] %s: parsed=%d error=%s\n",
                        path.string().c_str(), read, error.c_str());
        };
        emit_replay_fixture(decoded, "valid");
        const auto rejects_stored = [&](GpuCaptureFile malformed, const char* expected, const char* label) {
            GpuReplayFrame refused;
            const bool rejected = !materialize_gpu_replay(malformed, refused, error);
            std::printf("[replay-refusal] %s: rejected=%d reason=%s\n", label, rejected, error.c_str());
            check(rejected && error == expected, label);
        };
        auto missing_payload = decoded;
        missing_payload.draws.front().vrt.resources[source_index].blob_index = UINT32_MAX;
        emit_replay_fixture(missing_payload, "absent-parent");
        rejects_stored(missing_payload, "owned wide replay has missing, malformed or incomplete exact-PC backing", "stored-module replay refuses absent parent payload before execution");
        auto unmarked_missing = missing_payload;
        unmarked_missing.draws.front().vrt.resources[source_index].resource.owned_raw_snapshot_bytes = 0u;
        emit_replay_fixture(unmarked_missing, "unmarked-absent-parent");
        rejects_stored(unmarked_missing, "owned wide replay has missing, malformed or incomplete exact-PC backing", "removing obligation cannot hide raw-code-required absent backing");
        auto noncanonical = unmarked_missing;
        auto& raw = noncanonical.raw_shader_versions.front();
        raw.words.push_back(0xdeadbeefu); // unrelated post-END data outside the owning compile span
        raw.content_hash = gpu_capture_hash(reinterpret_cast<const uint8_t*>(raw.words.data()),
                                            raw.words.size() * 4u);
        emit_replay_fixture(noncanonical, "unmarked-noncanonical");
        rejects_stored(noncanonical, "owned wide replay lacks complete raw shader or resource table",
            "marker-free noncanonical raw span cannot erase its executable-prefix backing requirement");
        auto short_read = decoded;
        const auto blob = short_read.draws.front().vrt.resources[source_index].blob_index;
        short_read.blobs[blob].bytes_read = size - 4u;
        emit_replay_fixture(short_read, "short-read-parent");
        rejects_stored(short_read, "owned wide replay source exceeds the actually observed capture bytes", "padded blob extent cannot conceal an actually short captured parent read");
        auto replay_wrong_width = decoded;
        auto& wrong_source = replay_wrong_width.draws.front().vrt.resources[source_index];
        const uint32_t opposite_width = wide8 ? 16u : 32u;
        wrong_source.resource.size = opposite_width;
        wrong_source.resource.owned_raw_snapshot_bytes = opposite_width;
        wrong_source.captured_size = opposite_width;
        auto& wrong_blob = replay_wrong_width.blobs[wrong_source.blob_index];
        wrong_blob.bytes.resize(std::max<size_t>(wrong_blob.bytes.size(), opposite_width));
        wrong_blob.bytes_read = wrong_blob.bytes.size();
        wrong_blob.content_hash = gpu_capture_hash(wrong_blob.bytes.data(), wrong_blob.bytes.size());
        emit_replay_fixture(replay_wrong_width, "opposite-width-parent");
        rejects_stored(replay_wrong_width,
            "owned wide replay has missing, malformed or incomplete exact-PC backing",
            "stored replay refuses a complete opposite width at the decoded parent PC");
        auto replay_poisoned = decoded;
        auto duplicate_source = replay_poisoned.draws.front().vrt.resources[source_index];
        duplicate_source.resource.binding = 127u; // keep bindings distinct; collide only in fetch PC
        replay_poisoned.draws.front().vrt.resources[source_index].resource.format = DataFormat::Float32;
        replay_poisoned.draws.front().vrt.resources[source_index].resource.owned_raw_snapshot_bytes = 0u;
        replay_poisoned.draws.front().vrt.resources.push_back(duplicate_source);
        emit_replay_fixture(replay_poisoned, "poisoned-first-duplicate-parent");
        rejects_stored(replay_poisoned,
            "owned wide replay has missing, malformed or incomplete exact-PC backing",
            "stored replay refuses an ordinary poisoned first source followed by a valid same-PC source");
        auto no_table = decoded;
        no_table.draws.front().vrt = {};
        emit_replay_fixture(no_table, "absent-table");
        rejects_stored(no_table, "owned wide replay lacks complete raw shader or resource table", "raw-code requirement survives omission of the whole stored replay table");
        auto no_code = decoded;
        no_code.draws.front().vs_raw_shader_index = UINT32_MAX;
        no_code.raw_shader_versions.clear(); // retain structural validity; no unreferenced program
        emit_replay_fixture(no_code, "absent-code");
        rejects_stored(no_code, "owned wide replay lacks exact raw shader provenance", "marked stored replay refuses missing raw shader provenance");
        auto chain = decoded;
        const std::vector<uint32_t> prolog{0xbfa00003u, 0xbe802006u};
        chain.raw_shader_versions.push_back({gpu_capture_hash(
            reinterpret_cast<const uint8_t*>(prolog.data()), prolog.size() * 4u), false, prolog});
        chain.draws.front().vs_chain_raw_shader_index = chain.draws.front().vs_raw_shader_index;
        chain.draws.front().vs_raw_shader_index = static_cast<uint32_t>(chain.raw_shader_versions.size() - 1u);
        for (auto& r : chain.draws.front().vrt.resources) r.resource.owned_raw_snapshot_bytes = 0u;
        emit_replay_fixture(chain, "unmarked-chain");
        rejects_stored(chain, "owned raw wide replay inputs require a direct vertex stage", "marker-free stored chain derives refusal from the linked main raw code");
    }
    return failures ? 1 : 0;
}
