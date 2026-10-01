#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
#define CHECK(c, text) do { if (!(c)) { std::printf("[FAIL] %s\n", text); ++failures; } \
    else std::printf("[ok] %s\n", text); } while (0)

static std::vector<uint32_t> proof(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(code.data(), code.size(), decoded);
    return rdna2_proven_raw_register_wide_data_loads(decoded);
}

int main(int argc, char** argv) {
    alignas(16) std::array<uint32_t, 64> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint32_t>(100u + i);
    const auto address = reinterpret_cast<uint64_t>(bytes.data());
    std::array<uint32_t, 3> user = {
        static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u), 2u};
    const std::vector<uint32_t> x4 = {
        0x8f148402u,                 // s_lshl_b32 s20,s2,4
        0xf4080200u, 0x28000010u,   // x4 s[8:11],entry s[0:1],s20 offset:16
        0x7e000c09u,                 // v_cvt_f32_u32 v0,s9
        0xbf810000u,
    };
    auto table_for = [&](const auto& code) {
        ShaderResourceTable table;
        add_compute_buffer_resources(table, code.data(), code.size(), user.data(), user.size());
        assign_convention_bindings(table, 2);
        return table;
    };
    auto table = table_for(x4);
    const auto* resource = table.by_fetch_pc(1);
    CHECK(proof(x4) == std::vector<uint32_t>{1u}, "SALU offset load has the entry-pointer lifetime proof");
    CHECK(resource && valid_raw_register_snapshot_resource(*resource) &&
          resource->gpu_addr == address + 48u && resource->size == 16u,
          "fold binds exactly base plus current offset plus immediate, with no prefix upload");
    if (!resource) return failures + 1;
    auto spv = recompile_valu(x4.data(), x4.size(), 1, 0, &table);
    CHECK(!spv.empty(), "numeric x4 recompiles from exact rebased backing");
    auto x8 = x4;
    x8[1] = 0xf40c0200u;
    x8[3] = 0x7e000c0fu; // last of all eight current words
    auto x8_table = table_for(x8);
    CHECK(proof(x8) == std::vector<uint32_t>{1u} && x8_table.by_fetch_pc(1) &&
          x8_table.by_fetch_pc(1)->size == 32u &&
          !recompile_valu(x8.data(), x8.size(), 1, 0, &x8_table).empty(),
          "x8 is backed by a complete 32-byte exact-PC range");
    user[2] = 3u;
    auto shifted = table_for(x4);
    CHECK(shifted.by_fetch_pc(1) && shifted.by_fetch_pc(1)->gpu_addr == address + 64u,
          "changed entry offset selects different current source bytes");
    user[2] = 2u;

    auto unmarked = table;
    unmarked.resources[0].raw_register_snapshot = false;
    CHECK(recompile_valu(x4.data(), x4.size(), 1, 0, &unmarked).empty(),
          "ordinary exact-PC buffer does not authorize rebased register-offset data");
    auto short_table = table;
    short_table.resources[0].size = 12u;
    CHECK(recompile_valu(x4.data(), x4.size(), 1, 0, &short_table).empty(),
          "short backing stays refused");
    auto short_host = table;
    short_host.resources[0].host_data = reinterpret_cast<uint8_t*>(bytes.data());
    short_host.resources[0].host_data_size = 12u;
    CHECK(recompile_valu(x4.data(), x4.size(), 1, 0, &short_host).empty(),
          "short captured host backing stays refused");
    auto wrong_pc = table;
    wrong_pc.resources[0].fetch_pc = 0u;
    CHECK(recompile_valu(x4.data(), x4.size(), 1, 0, &wrong_pc).empty(),
          "another load's snapshot cannot authorize this PC");

    auto firstlane = x4;
    firstlane.insert(firstlane.begin(), 0x7e040500u); // v_readfirstlane_b32 s2,v0
    CHECK(proof(firstlane).empty() && !table_for(firstlane).by_fetch_pc(2),
          "offset dependency from readfirstlane remains unbacked");
    auto vcc = x4;
    vcc[0] = 0x8f6b8402u; // s_lshl_b32 vcc_hi,s2,4
    vcc[2] = 0xd6000010u;
    auto vcc_table = table_for(vcc);
    ComputeShaderConfig config;
    config.user_sgprs.assign(user.begin(), user.end());
    config.wave_size = 32u;
    CHECK(proof(vcc) == std::vector<uint32_t>{1u} &&
          !recompile_compute(vcc.data(), vcc.size(), &vcc_table, config).empty(),
          "Wave32 VCC_HI scalar offset has real current-byte backing");
    config.wave_size = 64u;
    CHECK(!recompile_compute(vcc.data(), vcc.size(), &vcc_table, config).empty(),
          "Wave64 explicitly scalar-defined VCC offset has the same real backing");
    auto compare = vcc;
    compare.insert(compare.begin() + 1, 0x7da80484u); // implicit VCC writer after scalar definition
    CHECK(proof(compare).empty() && !table_for(compare).by_fetch_pc(2),
          "implicit VALU VCC clobber invalidates the offset even if the compact fold omits it");
    auto carry = vcc;
    carry.insert(carry.begin() + 1, 0x50000300u); // implicit VOP2 carry writer
    CHECK(proof(carry).empty(), "implicit VOP2 carry cannot preserve a scalar VCC offset");
    auto store = x4;
    store.insert(store.begin() + 1, {0xe0700000u, 0x80000000u});
    CHECK(proof(store).empty() && !table_for(store).by_fetch_pc(3),
          "a possible pre-load guest store forbids dispatch-time backing");
    auto pointer_write = x4;
    pointer_write.insert(pointer_write.begin(), 0xbe800380u); // s_mov_b32 s0,0
    CHECK(proof(pointer_write).empty(), "entry pointer clobber refuses the snapshot");
    auto branch = x4;
    branch.insert(branch.begin(), 0xbf840000u); // conditional branch to following instruction
    CHECK(proof(branch).empty(), "pre-load branch is outside the initial scalar-offset subset");

    clear_shader_recompile_cache();
    config.wave_size = 64u;
    uint64_t marked_identity = 0, unmarked_identity = 0, invalid_identity = 0;
    CHECK(!recompile_compute_shader_cached(x4.data(), x4.size(), &table, config, &marked_identity).empty(),
          "cached marked source compiles");
    CHECK(recompile_compute_shader_cached(x4.data(), x4.size(), &unmarked, config, &unmarked_identity).empty() &&
          marked_identity != unmarked_identity, "compiled cache distinguishes marked and unmarked resources");
    auto misaligned = table;
    ++misaligned.resources[0].gpu_addr;
    CHECK(recompile_compute_shader_cached(x4.data(), x4.size(), &misaligned, config, &invalid_identity).empty() &&
          invalid_identity != marked_identity,
          "warm cache cannot borrow admission across a malformed source alignment");
    CHECK(recompile_compute_shader_cached(x4.data(), x4.size(), &short_host, config).empty(),
          "warm cache cannot borrow admission for truncated captured host bytes");
    uint64_t shifted_identity = 0;
    CHECK(!recompile_compute_shader_cached(x4.data(), x4.size(), &shifted, config, &shifted_identity).empty() &&
          shifted_identity == marked_identity, "changed source address reuses the rebased-load module");

    GpuCaptureFile capture;
    GpuCapturedCompute compute;
    compute.spirv = spv;
    GpuCapturedResource captured;
    captured.resource = *resource;
    captured.captured_size = resource->size;
    captured.blob_index = 0u;
    GpuCaptureBlob blob;
    blob.guest_addr = resource->gpu_addr;
    blob.bytes.resize(resource->size);
    std::memcpy(blob.bytes.data(), reinterpret_cast<const void*>(resource->gpu_addr),
                blob.bytes.size());
    blob.bytes_read = blob.bytes.size();
    blob.content_hash = gpu_capture_hash(blob.bytes);
    capture.blobs.push_back(blob);
    compute.resources.resources.push_back(captured);
    compute.resources.present = true;
    capture.computes.push_back(compute);
    std::vector<uint8_t> encoded;
    GpuCaptureFile decoded;
    std::string error;
    const bool roundtrip = serialize_gpu_capture(capture, encoded, error) &&
                           deserialize_gpu_capture(encoded, decoded, error);
    if (!roundtrip) std::printf("capture codec error: %s\n", error.c_str());
    CHECK(roundtrip &&
          decoded.computes[0].resources.resources[0].resource.raw_register_snapshot,
          "rebased scalar snapshot admission survives the capture codec");
    GpuReplayFrame replay;
    CHECK(roundtrip && materialize_gpu_replay(decoded, replay, error) &&
          replay.computes[0].resources &&
          valid_raw_register_snapshot_resource(replay.computes[0].resources->resources[0]) &&
          replay.computes[0].resources->resources[0].host_data &&
          std::memcmp(replay.computes[0].resources->resources[0].host_data,
                      blob.bytes.data(), blob.bytes.size()) == 0,
          "replay preserves the rebased range and its owned current bytes");
    // This compute-only fixture has no draw or failure records: v64 appends two zero counts.
    constexpr size_t mode_tail_bytes = 8u;
    constexpr size_t width_tail_bytes = 4u;
    constexpr size_t backing_tail_bytes = 5u;
    CHECK(encoded.size() >= mode_tail_bytes + width_tail_bytes + backing_tail_bytes &&
          encoded[8] == 64u,
          "legacy controls require the current versioned capture tail");
    if (encoded.size() >= mode_tail_bytes + width_tail_bytes + backing_tail_bytes) {
        auto legacy = encoded;
        // Remove v64 mode counts, v63 draw-width count, then v62 count/resource marker.
        legacy.resize(legacy.size() - mode_tail_bytes - width_tail_bytes - backing_tail_bytes);
        legacy[8] = 61u;
        CHECK(deserialize_gpu_capture(legacy, decoded, error) &&
              !decoded.computes[0].resources.resources[0].resource.raw_register_snapshot,
              "v61 capture leaves the new backing admission unavailable");
        auto malformed = encoded;
        malformed[malformed.size() - mode_tail_bytes - width_tail_bytes - 1u] = 2u;
        CHECK(!deserialize_gpu_capture(malformed, decoded, error) &&
              error == "invalid raw register snapshot state",
              "codec refuses an invented marker encoding");
    }
    capture.computes[0].resources.resources[0].resource.size = 12u;
    CHECK(!serialize_gpu_capture(capture, encoded, error), "codec refuses a malformed snapshot span");
    if (argc == 3 && std::strcmp(argv[1], "--write-spv") == 0 && !spv.empty()) {
        std::ofstream output(argv[2], std::ios::binary);
        output.write(reinterpret_cast<const char*>(spv.data()), spv.size() * sizeof(uint32_t));
        CHECK(output.good(), "representative register-offset module written for spirv-val");
    }
    std::printf("raw register-wide failures: %d\n", failures);
    return failures != 0;
}
