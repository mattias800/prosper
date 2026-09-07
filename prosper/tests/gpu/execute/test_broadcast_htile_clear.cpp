// A semantic zero fill must discard rendered depth even when guest HTILE was already zero.
// The adjacent negative arm copies each destination value back unchanged through the Vulkan path.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "shared/live/live_compute.hpp"
#include "fixtures/render_runner.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

// Hand-assembled bounded broadcast/copy, built from ISA fields rather than captured shader data.
static std::array<uint32_t, 18> broadcast_program() {
    auto sopp = [](uint32_t op, uint32_t imm) { return 0xbf800000u | (op << 16) | imm; };
    auto smem = [](uint32_t op, uint32_t dst, uint32_t base) {
        return 0xf4000000u | (op << 18) | (dst << 6) | (base / 2);
    };
    auto mubuf = [](uint32_t op) { return 0xe0000000u | (op << 18) | (1u << 13); };
    auto args = [](uint32_t data, uint32_t addr, uint32_t descriptor) {
        return (128u << 24) | ((descriptor / 4) << 16) | (data << 8) | addr;
    };
    return {sopp(0x20, 1),                                  // prefetch
        0xd4000000u | (0x346u << 16), 12u | (134u << 9) | (256u << 18), // i=(s12<<6)+v0
        smem(8, 106, 8), 125u << 25, sopp(12, 0xc07f),       // count=CB[0]
        0x7c000000u | (0xd4u << 17) | 106u, sopp(8, 9),      // active=count>i; skip to end
        smem(8, 106, 8), (125u << 25) | 4u, sopp(12, 0xc07f),// mask=CB[1]
        (0x1bu << 25) | (1u << 17) | 106u,                  // src_index=mask&i
        mubuf(0), args(1, 1, 0), sopp(12, 0x3f70),          // value=src[src_index]
        mubuf(4), args(1, 0, 4), sopp(1, 0)};               // dst[i]=value
}

static void descriptor(std::vector<uint32_t>& sgprs, unsigned slot,
                       uint64_t address, uint32_t stride, uint32_t records, uint32_t format) {
    sgprs[slot] = static_cast<uint32_t>(address);
    sgprs[slot + 1] = static_cast<uint32_t>(address >> 32) | (stride << 16);
    sgprs[slot + 2] = records;
    sgprs[slot + 3] = (format << 12) | 4u;
}

int main() {
    const auto code = broadcast_program();
    check(classify_compute_cpu_fast_path(code.data(), code.size()) ==
              ComputeCpuFastPath::BroadcastBufferU32, "complete bounded broadcast is recognized");
    check(classify_compute_cpu_fast_path(code.data(), code.size() - 1) == ComputeCpuFastPath::None,
          "truncated broadcast is refused");
    for (const auto [word, bit] : {std::pair{15u, 0u}, {15u, 17u}, {16u, 0u}, {7u, 0u}}) {
        auto altered = code; altered[word] ^= 1u << bit;
        check(classify_compute_cpu_fast_path(altered.data(), altered.size()) == ComputeCpuFastPath::None,
              "changed store offset, address controls, address register or branch is refused");
    }

    constexpr uint64_t cb_address = 0x20000000u, target_address = 0x21000000u;
    std::array<uint32_t, 4> constants{128, 0, 0, 0};
    std::array<uint32_t, 128> metadata{};
    ShaderResourceTable table;
    auto resource = [&](uint32_t binding, uint32_t pc, uint32_t sgpr, uint64_t addr,
                        uint32_t size, uint32_t stride, DataFormat fmt, uint32_t nc, void* host) {
        ShaderResource r;
        r.binding = binding; r.fetch_pc = pc; r.sgpr_base = sgpr; r.gpu_addr = addr;
        r.size = size; r.stride = stride; r.format = fmt; r.num_components = nc;
        r.host_data = static_cast<uint8_t*>(host); r.host_data_size = size;
        table.resources.push_back(r);
    };
    resource(3, 15, 4, target_address, sizeof(metadata), 4, DataFormat::Uint32, 1, metadata.data());
    resource(4, 12, 0, cb_address + 8, 4, 4, DataFormat::Uint32, 1, constants.data() + 2);
    resource(5, 3, UINT32_MAX, cb_address, 16, 16, DataFormat::Float32, 4, constants.data());
    resource(6, 8, UINT32_MAX, cb_address, 16, 16, DataFormat::Float32, 4, constants.data());
    ComputeItem item;
    item.user_sgprs.resize(12);
    descriptor(item.user_sgprs, 0, cb_address + 8, 4, 1, 20); // 32_UINT
    descriptor(item.user_sgprs, 4, target_address, 4, metadata.size(), 20);
    descriptor(item.user_sgprs, 8, cb_address, 16, 1, 77); // scalar loads ignore numeric format
    item.launch.local_x = 64; item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = 2; item.launch.groups_y = item.launch.groups_z = 1;
    item.launch.threads_x = 128; item.launch.threads_y = item.launch.threads_z = 1;
    item.recompile_config_available = true;
    item.recompile_config.user_sgprs = item.user_sgprs;
    item.recompile_config.local_x = 64;
    item.recompile_config.local_y = item.recompile_config.local_z = 1;
    item.recompile_config.tgid_x_en = true;
    item.recompile_config.tgid_y_en = item.recompile_config.tgid_z_en = false;
    item.recompile_config.tidig_comp_cnt = 0;
    item.recompile_config.threads_x = item.launch.threads_x;
    item.recompile_config.threads_y = item.launch.threads_y;
    item.recompile_config.threads_z = item.launch.threads_z;
    item.cpu_fast_path = classify_compute_cpu_fast_path(code.data(), code.size());
    item.resources = std::make_shared<ShaderResourceTable>(table);
    item.spirv = recompile_compute(code.data(), code.size(), item.resources.get(), item.recompile_config);
    check(!item.spirv.empty(), "broadcast shader independently recompiles for Vulkan control");
    if (item.spirv.empty()) return 1;

    auto run = [&](ComputeItem candidate, uint32_t expected_fills) {
        const uint64_t before = prosper::frontend::live_compute_cpu_fill_dispatches();
        const bool ok = prosper::frontend::execute_live_compute_items({candidate});
        check(prosper::frontend::live_compute_cpu_fill_dispatches() == before + expected_fills,
              "CPU admission counter matches expected execution path");
        return ok;
    };
    // Use the live renderer's device before compute initializes, matching the shared-device route.
    // Creating an independent compute device first also changes layer teardown ordering at exit.
    check(prosper::test::render_vk_ctx().ok, "shared renderer device initializes");
    if (!prosper::test::render_vk_ctx().ok) return 1;
    constants[2] = 0x12345678;
    check(run(item, 1) && std::all_of(metadata.begin(), metadata.end(),
          [](uint32_t v) { return v == 0x12345678; }), "broadcast carries nonzero runtime source value");
    constants[0] = 65; metadata.fill(0xa5a5a5a5);
    check(run(item, 1) && metadata[64] == constants[2] && metadata[65] == 0xa5a5a5a5,
          "shader count bounds writes independently of the full workgroup extent");
    constants[0] = 128;
    metadata.fill(0xa5a5a5a5);
    ComputeItem partial = item;
    partial.recompile_config.exact_thread_extent = true;
    partial.launch.threads_x = partial.recompile_config.threads_x = 65;
    partial.recompile_config.threads_y = partial.recompile_config.threads_z = 1;
    check(run(partial, 1) && metadata[64] == constants[2] && metadata[65] == 0xa5a5a5a5,
          "exact partial workgroup leaves padded lanes untouched");

    // Admission-only probes use a valid terminating fallback instead of sending deliberately
    // altered descriptors to a shader that accesses them. A declined shortcut must neither touch
    // bytes nor increment its counter; the fallback succeeds without any external memory access.
    const uint32_t no_op_code[] = {0xbf810000u};
    const auto no_op_spirv = recompile_compute(no_op_code, std::size(no_op_code), nullptr,
                                              item.recompile_config);
    check(!no_op_spirv.empty(), "admission probes have a valid terminating fallback");
    if (no_op_spirv.empty()) return 1;
    auto reject = [&](ComputeItem candidate) {
        candidate.spirv = no_op_spirv;
        candidate.terminator_only_program_validated = true;
        metadata.fill(0xa5a5a5a5);
        check(run(candidate, 0) && std::all_of(metadata.begin(), metadata.end(),
              [](uint32_t v) { return v == 0xa5a5a5a5; }),
              "unsafe shortcut declines without writing destination");
    };
    for (auto [word, bit] : {std::pair{1u, 30u}, {3u, 23u}, {3u, 0u}, {5u, 31u}, {7u, 21u}}) {
        auto bad = item; bad.user_sgprs[word] ^= 1u << bit; reject(bad);
    }
    auto inexact = partial; inexact.recompile_config.exact_thread_extent = false; reject(inexact);
    auto short_source = item;
    short_source.resources = std::make_shared<ShaderResourceTable>(table);
    short_source.resources->resources[1].host_data_size = 3; reject(short_source);
    auto host_alias = item;
    host_alias.resources = std::make_shared<ShaderResourceTable>(table);
    host_alias.resources->resources[1].host_data = reinterpret_cast<uint8_t*>(metadata.data());
    reject(host_alias);

    // Render real nonzero depth, then test its consumer before and after the semantic metadata clear.
    #include "../tools/boot_trace/refvs.inc"
    std::vector<uint32_t> vs(kRefVs, kRefVs + std::size(kRefVs));
    auto with_depth = [&](uint32_t bits) {
        auto words = vs;
        for (size_t i = 5; i < words.size(); i += words[i] >> 16)
            if ((words[i] & 0xffffu) == 43 && (words[i] >> 16) == 4 &&
                words[i + 1] == 6 && words[i + 2] == 0x25) words[i + 3] = bits;
        return words;
    };
    const uint32_t red_code[] = {0x7e0002f2, 0x7e020280, 0x7e040280, 0x7e0602f2,
                                 0xf800180f, 0x03020100, 0xbf810000};
    const uint32_t green_code[] = {0x7e000280, 0x7e0202f2, 0x7e040280, 0x7e0602f2,
                                   0xf800180f, 0x03020100, 0xbf810000};
    ResolvedPipelineState writer;
    writer.topology = 3; writer.color_write_mask = 15;
    writer.depth_test_enable = writer.depth_write_enable = true;
    writer.depth_compare_op = 7; writer.has_depth_clear = true; writer.depth_clear_value = 0;
    writer.depth_read_base = writer.depth_write_base = 0x22000000;
    writer.htile_data_base = target_address;
    ResolvedPipelineState reader = writer;
    reader.depth_write_enable = false; reader.depth_compare_op = 6; // GREATER_OR_EQUAL, reverse-Z
    prosper::test::BackendDraw w, r;
    w.vs = with_depth(0x3f400000); w.fs = recompile_fragment(red_code, std::size(red_code), nullptr);
    w.ps = &writer; w.vcount = 3;
    r.vs = with_depth(0x3f000000); r.fs = recompile_fragment(green_code, std::size(green_code), nullptr);
    r.ps = &reader; r.vcount = 3;
    auto green_visible = [&]() {
        auto pixels = prosper::test::render_draws_rgba({r}, 64, 64, nullptr, nullptr, true);
        return pixels.size() == 64 * 64 * 4 && pixels[(32 * 64 + 32) * 4 + 1] > 192;
    };
    auto produced = prosper::test::render_draws_rgba({w}, 64, 64, nullptr, nullptr, true);
    check(produced.size() == 64 * 64 * 4 && !green_visible(),
          "rendered nearer depth rejects the green consumer before a clear");
    uint32_t preserving = 0, overwrites = 0;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t bytes, const char* origin) {
        if (std::strcmp(origin, "gpu-preserving") == 0) ++preserving; else ++overwrites;
        const char* previous = guest_gpu_write_origin();
        set_guest_gpu_write_origin(origin);
        prosper::test::invalidate_persistent_ds_guest_write(addr, bytes);
        set_guest_gpu_write_origin(previous);
    });
    constants[2] = 0; metadata.fill(0);
    // Source aliases destination and i&~0 selects the same element: a real Vulkan identity copy.
    auto identity = item;
    constants[1] = UINT32_MAX;
    identity.resources = std::make_shared<ShaderResourceTable>(table);
    auto& identity_source = identity.resources->resources[1];
    identity_source.gpu_addr = target_address; identity_source.size = sizeof(metadata);
    identity_source.host_data = reinterpret_cast<uint8_t*>(metadata.data());
    identity_source.host_data_size = sizeof(metadata);
    descriptor(identity.user_sgprs, 0, target_address, 4, metadata.size(), 20);
    identity.recompile_config.user_sgprs = identity.user_sgprs;
    identity.spirv = recompile_compute(code.data(), code.size(), identity.resources.get(), identity.recompile_config);
    check(!identity.spirv.empty() && run(identity, 0) && preserving > 0 && !green_visible(),
          "Vulkan identity metadata update preserves current-present rendered depth");
    constants[1] = 0;
    // Exercise the same clear through the public capture roundtrip: replay owns the snapshots
    // and must rederive the semantic proof from retained ISA, rather than losing it or trusting
    // a serialized shortcut. The fake guest addresses are served only by this bounded reader.
    item.code_addr = 0x23000000u;
    const CaptureMemoryReader memory = [&](uint64_t address, uint8_t* out, size_t bytes) {
        const void* source = nullptr;
        size_t available = 0;
        auto region = [&](uint64_t base, const void* data, size_t size) {
            if (address >= base && address - base < size) {
                const size_t offset = static_cast<size_t>(address - base);
                source = static_cast<const uint8_t*>(data) + offset;
                available = size - offset;
            }
        };
        region(cb_address, constants.data(), sizeof(constants));
        region(target_address, metadata.data(), sizeof(metadata));
        region(item.code_addr, code.data(), sizeof(code));
        const size_t copied = std::min(bytes, available);
        if (copied) std::memcpy(out, source, copied);
        return copied;
    };
    GpuCaptureMetadata capture_metadata;
    capture_metadata.width = capture_metadata.height = 64;
    GpuCaptureFile captured, loaded;
    GpuReplayFrame replay;
    std::vector<uint8_t> serialized;
    std::string error;
    const bool roundtrip = capture_submit_items({}, {item}, {}, capture_metadata, memory,
                                                captured, error) &&
        serialize_gpu_capture(captured, serialized, error) &&
        deserialize_gpu_capture(serialized, loaded, error) &&
        materialize_gpu_replay(loaded, replay, error);
    if (!roundtrip) std::fprintf(stderr, "capture roundtrip: %s\n", error.c_str());
    check(roundtrip && replay.computes.size() == 1,
          "bounded broadcast capture serializes and materializes");
    if (roundtrip && replay.computes.size() == 1) {
        const auto& restored = replay.computes[0];
        check(restored.cpu_fast_path == ComputeCpuFastPath::BroadcastBufferU32,
              "replay rederives the broadcast proof from retained raw ISA");
        const auto* destination = restored.resources ? restored.resources->by_fetch_pc(15) : nullptr;
        check(destination && destination->host_data &&
              destination->host_data != reinterpret_cast<uint8_t*>(metadata.data()),
              "replay destination uses owned captured backing");
        // Poison live bytes after capture: the replay must consume its independent snapshots.
        metadata.fill(0xa5a5a5a5);
        check(run(restored, 1) && overwrites > 0 && green_visible(),
              "replayed zero-to-zero broadcast invalidates depth and restores the green consumer");
        check(std::all_of(metadata.begin(), metadata.end(),
                         [](uint32_t v) { return v == 0xa5a5a5a5; }),
              "replayed broadcast does not mutate the live source allocation");
        check(destination && destination->host_data &&
              destination->host_data_size >= sizeof(metadata) &&
              std::all_of(destination->host_data, destination->host_data + sizeof(metadata),
                          [](uint8_t v) { return v == 0; }),
              "replayed clear writes the captured destination bytes");
        auto refused_proof = [&](GpuCaptureFile candidate) {
            GpuReplayFrame without_proof;
            const bool materialized = materialize_gpu_replay(candidate, without_proof, error);
            const bool refused = materialized &&
                  without_proof.computes.size() == 1 &&
                  without_proof.computes[0].cpu_fast_path == ComputeCpuFastPath::None;
            if (!refused)
                std::fprintf(stderr, "refused proof: materialized=%u computes=%zu fast-path=%d error=%s\n",
                             materialized, without_proof.computes.size(),
                             without_proof.computes.empty() ? -1 :
                                 static_cast<int>(without_proof.computes[0].cpu_fast_path), error.c_str());
            check(refused,
                  "replay does not infer a broadcast shortcut without complete matching ISA and config");
        };
        auto missing = loaded;
        missing.computes[0].raw_shader_index = UINT32_MAX;
        missing.raw_shader_versions.clear(); // No orphaned version: reach the materializer.
        refused_proof(std::move(missing));
        auto altered = loaded;
        auto& altered_raw = altered.raw_shader_versions[altered.computes[0].raw_shader_index];
        altered_raw.words[15] ^= 1u;
        // Keep the container valid so changed ISA, rather than a stale hash, refuses the shortcut.
        altered_raw.content_hash = gpu_capture_hash(
            reinterpret_cast<const uint8_t*>(altered_raw.words.data()),
            altered_raw.words.size() * sizeof(uint32_t));
        refused_proof(std::move(altered));
        auto no_config = loaded;
        no_config.computes[0].recompile_config_available = false;
        GpuReplayFrame without_config;
        check(!materialize_gpu_replay(no_config, without_config, error) &&
              error == "compute raw shader has no recompile state" && without_config.computes.empty(),
              "capture validation refuses retained raw ISA without the required compute config");
    }
    set_guest_gpu_write_observer({});
    return failures ? 1 : 0;
}
