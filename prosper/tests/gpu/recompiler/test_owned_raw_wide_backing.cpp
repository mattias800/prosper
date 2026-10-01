// Real fold, module cache and capture materializer controls; no device or GPU execution.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* label) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", label);
    failures += !ok;
}

int main() {
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
              "v63 capture owns highest selector and separate highest child word without guest reread");
        if (!roundtrip) continue;
        auto v62_bytes = encoded;
        const auto v63_tail_size = 4u + 4u * decoded.draws.front().vrt.resources.size();
        if (v62_bytes.size() >= v63_tail_size) {
            v62_bytes.resize(v62_bytes.size() - v63_tail_size);
            v62_bytes[8] = 62u;
            v62_bytes[9] = v62_bytes[10] = v62_bytes[11] = 0u;
        }
        GpuCaptureFile v62;
        GpuReplayFrame legacy_replay;
        check(deserialize_gpu_capture(v62_bytes, v62, error) && v62.format_version == 62u &&
              std::all_of(v62.draws.front().vrt.resources.begin(), v62.draws.front().vrt.resources.end(),
                  [](const auto& r) { return r.resource.owned_raw_snapshot_bytes == 0u; }) &&
              materialize_gpu_replay(v62, legacy_replay, error),
              "v62 parsing preserves complete payloads and derives new source authority from code");
        const auto source_index = static_cast<size_t>(source - table.resources.data());
        const auto rejects_stored = [&](GpuCaptureFile malformed, const char* label) {
            GpuReplayFrame refused;
            check(!materialize_gpu_replay(malformed, refused, error), label);
        };
        auto missing_payload = decoded;
        missing_payload.draws.front().vrt.resources[source_index].blob_index = UINT32_MAX;
        rejects_stored(missing_payload, "stored-module replay refuses absent parent payload before execution");
        auto unmarked_missing = missing_payload;
        unmarked_missing.draws.front().vrt.resources[source_index].resource.owned_raw_snapshot_bytes = 0u;
        rejects_stored(unmarked_missing, "removing obligation cannot hide raw-code-required absent backing");
        auto short_read = decoded;
        const auto blob = short_read.draws.front().vrt.resources[source_index].blob_index;
        short_read.blobs[blob].bytes_read = size - 4u;
        rejects_stored(short_read, "padded blob extent cannot conceal an actually short captured parent read");
        auto no_table = decoded;
        no_table.draws.front().vrt = {};
        rejects_stored(no_table, "raw-code requirement survives omission of the whole stored replay table");
        auto no_code = decoded;
        no_code.draws.front().vs_raw_shader_index = UINT32_MAX;
        rejects_stored(no_code, "marked stored replay refuses missing raw shader provenance");
        auto chain = decoded;
        chain.raw_shader_versions.push_back({0u, false, {0xbfa00003u, 0xbe802006u}});
        chain.draws.front().vs_chain_raw_shader_index = chain.draws.front().vs_raw_shader_index;
        chain.draws.front().vs_raw_shader_index = static_cast<uint32_t>(chain.raw_shader_versions.size() - 1u);
        for (auto& r : chain.draws.front().vrt.resources) r.resource.owned_raw_snapshot_bytes = 0u;
        rejects_stored(chain, "marker-free stored chain derives refusal from the linked main raw code");
    }
    return failures ? 1 : 0;
}
