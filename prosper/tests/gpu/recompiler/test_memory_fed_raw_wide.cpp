#include "gpu/execute/gpu_execute.hpp"
#include <gtest/gtest.h>
#include "gpu/capture/fold_capture.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
#define CHECK(c, text) EXPECT_TRUE(c) << (text)

static std::vector<uint32_t> proof(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(code.data(), code.size(), decoded);
    return rdna2_proven_raw_register_wide_data_loads(decoded);
}

TEST(MemoryFedRawWide, Contract) {
    alignas(16) std::array<uint32_t, 128> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = 100u + static_cast<uint32_t>(i);
    uint32_t selector = 2u;
    const auto address = reinterpret_cast<uint64_t>(bytes.data());
    const auto selector_address = reinterpret_cast<uint64_t>(&selector);
    std::array<uint32_t, 4> user{static_cast<uint32_t>(address),
        static_cast<uint32_t>(address >> 32u), static_cast<uint32_t>(selector_address),
        static_cast<uint32_t>(selector_address >> 32u)};
    // The x1 source pair is overwritten AFTER its read; its result in s4 remains live.
    const std::vector<uint32_t> code = {
        0xf4000101u, 0xfa000000u,       // x1 s4,entry s[2:3],0
        0xbe820380u, 0xbe830380u,       // s[2:3] = 0 after the read
        0x8f6b8404u,                   // s_lshl_b32 vcc_hi,s4,4
        0x876bff6bu, 0x000001f0u,      // s_and_b32 vcc_hi,vcc_hi,0x1f0
        0xf4080200u, 0xd6000000u,       // x4 s[8:11],entry s[0:1],vcc_hi
        0x7e000c0bu,                   // last loaded word -> v0 float
        0xbf810000u,
    };
    auto table_for = [&](const auto& program) {
        ShaderResourceTable table;
        add_compute_buffer_resources(table, program.data(), program.size(), user.data(), user.size());
        assign_convention_bindings(table, 2u);
        return table;
    };
    auto table = table_for(code);
    const auto* selected = table.by_fetch_pc(7u);
    const auto* scalar = table.by_fetch_pc(0u);
    CHECK(proof(code) == std::vector<uint32_t>{7u},
          "latched immediate x1 scalar admits the later wide offset after pointer overwrite");
    CHECK(table.by_fetch_pc(0u) && selected && valid_raw_register_snapshot_resource(*selected) &&
          selected->gpu_addr == address + 32u && selected->size == 16u,
          "both the x1 read point and exact selected x4 bytes have numeric backing");
    uint32_t latched = UINT32_MAX;
    if (scalar && scalar->host_data && scalar->host_data_size == sizeof(latched))
        std::memcpy(&latched, scalar->host_data, sizeof(latched));
    CHECK(latched == 2u && table.owned_host_data.size() == 1u,
          "the authenticated x1 source owns the exact four bytes observed during realization");
    selector = 3u;
    uint32_t still_latched = UINT32_MAX;
    if (scalar && scalar->host_data)
        std::memcpy(&still_latched, scalar->host_data, sizeof(still_latched));
    CHECK(still_latched == 2u && selected && selected->gpu_addr == address + 32u,
          "post-realization guest mutation preserves the selector and selected range together");
    const auto copied = table;
    CHECK(copied.by_fetch_pc(0u) && copied.by_fetch_pc(0u)->host_data == scalar->host_data,
          "table copies retain shared ownership of the latched scalar source");
    selector = 2u;
    CHECK(!recompile_valu(code.data(), code.size(), 1u, 0u, &table).empty(),
          "Wave64 test shell recompiles the memory-fed scalar VCC_HI path");
    auto descriptor_only = code;
    descriptor_only.erase(descriptor_only.begin() + 9);
    descriptor_only.insert(descriptor_only.end() - 1,
                           {0x8f0b810bu, 0xe0042000u, 0x80020000u});
    std::vector<Rdna2Inst> descriptor_ins;
    rdna2_walk(descriptor_only.data(), descriptor_only.size(), descriptor_ins);
    CHECK(rdna2_raw_wide_data_loads(descriptor_ins).empty() && proof(descriptor_only).empty(),
          "a memory-fed wide descriptor patch needs no numeric snapshot without a data observer");
    auto descriptor_with_data = descriptor_only;
    descriptor_with_data.insert(descriptor_with_data.end() - 3, 0x7e00020bu);
    CHECK(proof(descriptor_with_data) == std::vector<uint32_t>{7u},
          "adding an ordinary data observer to that descriptor shape requires real wide backing");
    auto allocation = code;
    allocation.insert(allocation.begin(), {0xbfa00003u, 0xbf900009u});
    const auto allocation_table = table_for(allocation);
    CHECK(proof(allocation) == std::vector<uint32_t>{9u} &&
          allocation_table.by_fetch_pc(9u) &&
          !recompile_valu(allocation.data(), allocation.size(), 1u, 0u,
                          &allocation_table).empty(),
          "instruction prefetch and GS allocation preserve the scalar read-point proof");
    auto other_message = allocation;
    other_message[1] = 0xbf900006u;
    CHECK(proof(other_message).empty(), "unproven send-message effects stay outside the subset");
    ComputeShaderConfig config;
    config.user_sgprs.assign(user.begin(), user.end());
    for (uint32_t wave : {32u, 64u}) {
        config.wave_size = wave;
        CHECK(!recompile_compute(code.data(), code.size(), &table, config).empty(),
              "live compute shell accepts authenticated scalar VCC at either wave size");
    }
    selector = 3u;
    auto changed = table_for(code);
    CHECK(changed.by_fetch_pc(7u) && changed.by_fetch_pc(7u)->gpu_addr == address + 48u,
          "changed memory selector chooses new current bytes at the same fetch PC");
    uint64_t first_identity = 0, changed_identity = 0;
    CHECK(!recompile_compute_shader_cached(code.data(), code.size(), &table, config,
                                          &first_identity).empty() &&
          !recompile_compute_shader_cached(code.data(), code.size(), &changed, config,
                                          &changed_identity).empty() &&
          first_identity == changed_identity,
          "changed selector bytes reuse code while retaining distinct exact-PC resource backing");
    // Each malformed table is checked cold by the emitter and after the valid module is cached.
    // Restoring the original resource must recover that same module; guest addresses and scalar
    // values are not code identity, but a missing/truncated hosted word is a different admission.
    const auto reject_scalar = [&](ShaderResourceTable malformed, const char* label) {
        const bool cold_rejected = recompile_compute(code.data(), code.size(),
                                                     &malformed, config).empty();
        uint64_t invalid_identity = 0u, restored_identity = 0u;
        const bool warm_rejected = recompile_compute_shader_cached(
            code.data(), code.size(), &malformed, config, &invalid_identity).empty();
        const bool restored_valid = !recompile_compute_shader_cached(
            code.data(), code.size(), &table, config, &restored_identity).empty() &&
            restored_identity == first_identity;
        std::printf("[admission-state] %s: cold_rejected=%d warm_rejected=%d restored_valid=%d\n",
                    label, static_cast<int>(cold_rejected), static_cast<int>(warm_rejected),
                    static_cast<int>(restored_valid));
        CHECK(cold_rejected && warm_rejected && restored_valid, label);
    };
    auto truncated_scalar = table;
    truncated_scalar.resources.front().host_data_size = 3u;
    reject_scalar(truncated_scalar,
                  "cached valid -> truncated source -> valid rejects the short hosted word");
    auto borrowed_scalar = table;
    borrowed_scalar.resources.front().host_data = nullptr;
    borrowed_scalar.resources.front().host_data_size = 0u;
    reject_scalar(borrowed_scalar,
                  "cached valid -> guest-only source -> valid cannot reread an unowned selector");
    auto wrong_size_scalar = table;
    wrong_size_scalar.resources.front().size = 3u;
    reject_scalar(wrong_size_scalar,
                  "cached valid -> short source range -> valid rejects incomplete backing");
    auto unaligned_scalar = table;
    ++unaligned_scalar.resources.front().gpu_addr;
    reject_scalar(unaligned_scalar,
                  "cached valid -> unaligned source -> valid rejects malformed source alignment");
    auto overflow_scalar = table;
    overflow_scalar.resources.front().gpu_addr = UINT64_MAX - 3u;
    reject_scalar(overflow_scalar,
                  "cached valid -> overflowing source -> valid rejects malformed source range");
    auto prefixed_scalar = table;
    prefixed_scalar.resources.front().host_data_prefix_bytes = 4u;
    reject_scalar(prefixed_scalar,
                  "cached valid -> prefixed source -> valid rejects a different snapshot origin");
    auto wrong_pc_scalar = table;
    wrong_pc_scalar.resources.front().fetch_pc = 1u;
    reject_scalar(wrong_pc_scalar,
                  "cached valid -> wrong-PC source -> valid requires the exact source fetch PC");
    auto poisoned_scalar = table;
    poisoned_scalar.resources.erase(poisoned_scalar.resources.begin());
    ShaderResource legacy;
    legacy.cls = ResourceClass::ConstantBuffer;
    legacy.format = DataFormat::Uint32;
    legacy.num_components = 1u;
    legacy.binding = 2u;
    legacy.size = sizeof(bytes);
    legacy.gpu_addr = address;
    poisoned_scalar.resources.push_back(legacy);
    reject_scalar(poisoned_scalar,
                  "absent exact-PC source cannot borrow an unrelated legacy binding-2 buffer");
    auto unrelated_code = code;
    unrelated_code.insert(unrelated_code.begin(), {0xf4000183u, 0xfa000000u});
    auto unrelated_table = table;
    for (auto& resource : unrelated_table.resources) resource.fetch_pc += 2u;
    CHECK(proof(unrelated_code) == std::vector<uint32_t>{9u} &&
          recompile_compute(unrelated_code.data(), unrelated_code.size(),
                            &unrelated_table, config).empty(),
          "an unrelated raw x1 cannot borrow the proof-owned scalar at legacy binding 2");
    for (auto& resource : unrelated_table.resources) ++resource.binding;
    unrelated_table.resources.push_back(legacy);
    CHECK(!recompile_compute(unrelated_code.data(), unrelated_code.size(),
                             &unrelated_table, config).empty(),
          "an ordinary raw x1 still accepts its existing legacy binding-2 convention");
    DrawItem first_draw, changed_draw;
    first_draw.vs = changed_draw.vs = {0x07230203u, 1u, 2u};
    first_draw.vrt = std::make_shared<ShaderResourceTable>(table);
    changed_draw.vrt = std::make_shared<ShaderResourceTable>(changed);
    uint64_t planned = 0;
    std::string error;
    CHECK(preflight_gpu_capture_draw_resources(first_draw, 1u << 20u, planned, error) &&
          planned == 20u,
          "capture budget includes the owned four-byte source and selected sixteen-byte range");
    unsigned selector_reads = 0;
    const CaptureMemoryReader reader = [&](uint64_t addr, uint8_t* dst, size_t size) {
        if (addr == selector_address) ++selector_reads;
        if (addr < address || addr > address + sizeof(bytes) ||
            size > address + sizeof(bytes) - addr)
            return size_t{0};
        std::memcpy(dst, reinterpret_cast<const void*>(addr), size);
        return size;
    };
    GpuCaptureFile capture, restored;
    std::vector<uint8_t> encoded;
    GpuReplayFrame replay;
    const bool restored_ok = capture_draw_items({first_draw, changed_draw}, {}, reader,
                                                capture, error) &&
        serialize_gpu_capture(capture, encoded, error) &&
        deserialize_gpu_capture(encoded, restored, error) &&
        materialize_gpu_replay(restored, replay, error);
    uint32_t replay_first = UINT32_MAX, replay_changed = UINT32_MAX;
    if (restored_ok && replay.items.size() == 2u && replay.items[0].vrt && replay.items[1].vrt) {
        const auto* first = replay.items[0].vrt->by_fetch_pc(0u);
        const auto* second = replay.items[1].vrt->by_fetch_pc(0u);
        if (first && first->host_data) std::memcpy(&replay_first, first->host_data, 4u);
        if (second && second->host_data) std::memcpy(&replay_changed, second->host_data, 4u);
    }
    if (!restored_ok) std::printf("capture error: %s\n", error.c_str());
    CHECK(restored_ok && selector_reads == 0u && replay_first == 2u && replay_changed == 3u,
          "capture owns separate draw observations at one guest address without re-reading it");
    uint64_t replay_identity = 0u;
    CHECK(restored_ok && replay.items.size() == 2u && replay.items[0].vrt &&
          replay.items[1].vrt &&
          !recompile_compute_shader_cached(code.data(), code.size(), replay.items[0].vrt.get(),
                                           config, &replay_identity).empty() &&
          replay_identity == first_identity &&
          !recompile_compute_shader_cached(code.data(), code.size(), replay.items[1].vrt.get(),
                                           config, &replay_identity).empty() &&
          replay_identity == first_identity,
          "materialized owned replay observations pass the same exact-source cache admission");
    FoldInputs fold;
    fold.code = code;
    fold.user.assign(user.begin(), user.end());
    fold.srt_requested = true;
    const auto transcript = capture_fold(fold);
    const auto restored_fold = decode_fold_capture(encode_fold_capture(transcript));
    const auto replayed_fold = replay_fold(restored_fold);
    CHECK(replayed_fold.uses == transcript.uses &&
          std::any_of(replayed_fold.uses.begin(), replayed_fold.uses.end(), [](const SrtUse& use) {
              return use.kind == 5 && use.use_pc == 0u && use.v4[2] == 3u;
          }), "fold capture retains the latched source word and its exact read point");
    selector = 2u;
    auto early_pointer = code;
    early_pointer.insert(early_pointer.begin(), 0xbe820380u);
    CHECK(proof(early_pointer).empty(), "x1 source pointer overwritten BEFORE its read stays unproven");
    auto register_read = code;
    register_read[1] = 0x08000000u; // x1 itself has register SOFFSET
    CHECK(proof(register_read).empty(), "a register-offset source read is outside the immediate x1 subset");
    auto pair_read = code;
    pair_read[0] = 0xf4040101u;
    // #4578: UE4's vertex-factory index is an s_load_dwordx2 pair; its snapshot owns both words.
    CHECK(!proof(pair_read).empty(), "an x2 source read is a latched source too");
    auto unaligned_read = code;
    unaligned_read[1] = 0xfa000002u;
    CHECK(proof(unaligned_read).empty(), "unaligned source immediate stays unproven");
    auto nonzero_read = code;
    nonzero_read[1] = 0xfa000004u;
    // #3135 widened the unit to aligned non-negative immediates (Kena's indexed NGG prolog reads its
    // selector at +4); test_raw_offset_immediate_source pins the snapshot and the emitted read.
    CHECK(proof(nonzero_read) == std::vector<uint32_t>{7u},
          "an aligned nonzero source immediate is a latched source too");
    auto negative_read = code;
    negative_read[1] = 0xfa1ffffcu;
    CHECK(proof(negative_read).empty(), "negative source immediate stays outside this subset");
    auto branch = code;
    branch.insert(branch.begin() + 2, 0xbf840000u);
    CHECK(proof(branch).empty(), "control transfer between x1 and wide load stays unproven");
    auto store = code;
    store.insert(store.begin() + 2, {0xe0700000u, 0x80000000u});
    CHECK(proof(store).empty(), "a possible guest store between the two reads invalidates the snapshot");
    auto compare = code;
    compare.insert(compare.begin() + 7, 0x7da80484u);
    CHECK(proof(compare).empty() && !table_for(compare).by_fetch_pc(8u),
          "implicit VALU VCC overwrite invalidates memory-fed VCC too");
    auto dynamic = code;
    dynamic.insert(dynamic.begin() + 2, 0x7e080500u); // v_readfirstlane_b32 s4,v0
    CHECK(proof(dynamic).empty(), "a wave-derived replacement of the fetched scalar stays unproven");
    // The register-offset wide load reads the fold's exact per-PC snapshot and never re-reads its
    // base register, and control is forward-only, so its entry pair need only survive UNTIL the
    // load (UE4's NGG vertex programs reuse s[34:35] after it). An overwrite before it still
    // refuses.
    auto wide_pointer = code;
    wide_pointer.insert(wide_pointer.begin() + 9, 0xbe800380u);   // s_mov_b32 s0, 0 after the x4
    const auto wide_pointer_table = table_for(wide_pointer);
    CHECK(proof(wide_pointer) == std::vector<uint32_t>{7u} && wide_pointer_table.by_fetch_pc(7u) &&
              wide_pointer_table.by_fetch_pc(7u)->gpu_addr == address + 32u,
          "a wide source pointer overwritten after the load keeps the proof and its snapshot");
    auto wide_pointer_early = code;
    wide_pointer_early.insert(wide_pointer_early.begin() + 7, 0xbe800380u);   // before the x4
    CHECK(proof(wide_pointer_early).empty(),
          "a wide source pointer overwritten before the load stays unproven");
    // A branch BEFORE the latched source is harmless unless it lands between source and load.
    auto join_before = code;
    join_before.insert(join_before.begin(), 0xbf840000u);   // s_cbranch_scc0 -> the x1 source
    const auto join_before_table = table_for(join_before);
    CHECK(proof(join_before) == std::vector<uint32_t>{8u} && join_before_table.by_fetch_pc(1u) &&
              join_before_table.by_fetch_pc(8u) &&
              join_before_table.by_fetch_pc(8u)->gpu_addr == address + 32u,
          "a branch landing at the source joins ahead of it");
    auto lands_on_load = code;
    lands_on_load.insert(lands_on_load.begin(), 0xbf840007u);   // s_cbranch_scc0 -> the x4 (pc8)
    CHECK(proof(lands_on_load).empty(), "a branch landing on the load itself stays unproven");
    auto lands_between = code;
    lands_between.insert(lands_between.begin(), 0xbf840004u);   // s_cbranch_scc0 -> s_lshl (pc5)
    CHECK(proof(lands_between).empty(), "a branch landing between source and load stays unproven");
    auto skipped_writer = code;
    skipped_writer.insert(skipped_writer.begin(), {0xbf840001u, 0xbe820380u});   // may skip s2 = 0
    CHECK(proof(skipped_writer).empty(),
          "a source-pointer write on either path before the read stays unproven");
    auto no_scalar = table;
    no_scalar.resources.erase(no_scalar.resources.begin());
    CHECK(recompile_valu(code.data(), code.size(), 1u, 0u, &no_scalar).empty(),
          "wide snapshot alone cannot fabricate the earlier scalar value");
    auto invalid_user = user;
    invalid_user[2] = 0xfffffffcu;
    invalid_user[3] = 0xffffffffu;
    ShaderResourceTable invalid_source;
    add_compute_buffer_resources(invalid_source, code.data(), code.size(),
                                 invalid_user.data(), invalid_user.size());
    CHECK(!invalid_source.by_fetch_pc(0u) && !invalid_source.by_fetch_pc(7u),
          "an overflowing x1 source range cannot publish either backing");
    invalid_user = user;
    invalid_user[0] = 0xfffffffcu;
    invalid_user[1] = 0xffffffffu;
    ShaderResourceTable invalid_wide;
    add_compute_buffer_resources(invalid_wide, code.data(), code.size(),
                                 invalid_user.data(), invalid_user.size());
    CHECK(!invalid_wide.by_fetch_pc(7u), "an overflowing selected wide range stays unbacked");
}

// #4578: UE4's vertex-factory fetch loads its index PAIR with s_load_dwordx2. The table owns both
// observed words, and a frame capture keeps each draw's own pair rather than re-reading the guest.
TEST(MemoryFedRawWide, X2SourceOwnsBothWordsThroughCapture) {
    alignas(16) std::array<uint32_t, 128> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = 100u + static_cast<uint32_t>(i);
    alignas(8) std::array<uint32_t, 2> pair{2u, 0x77u};
    const auto address = reinterpret_cast<uint64_t>(bytes.data());
    const auto pair_address = reinterpret_cast<uint64_t>(pair.data());
    std::array<uint32_t, 4> user{
        static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u),
        static_cast<uint32_t>(pair_address), static_cast<uint32_t>(pair_address >> 32u)};
    const std::vector<uint32_t> code = {
        0xf4040101u, 0xfa000000u,   // x2 s[4:5],entry s[2:3],0
        0xbe820380u, 0xbe830380u,   // s[2:3] = 0 after the read
        0x8f6b8404u,   // s_lshl_b32 vcc_hi,s4,4
        0x876bff6bu, 0x000001f0u,   // s_and_b32 vcc_hi,vcc_hi,0x1f0
        0xf4080200u, 0xd6000000u,   // x4 s[8:11],entry s[0:1],vcc_hi
        0x7e000c0bu,   // last loaded word -> v0 float
        0xbf810000u,
    };
    auto table_for = [&] {
        ShaderResourceTable table;
        add_compute_buffer_resources(table, code.data(), code.size(), user.data(), user.size());
        assign_convention_bindings(table, 2u);
        return table;
    };
    CHECK(proof(code) == std::vector<uint32_t>{7u}, "an x2 source admits the later wide offset");
    const auto table = table_for();
    const auto* scalar = table.by_fetch_pc(0u);
    std::array<uint32_t, 2> owned{UINT32_MAX, UINT32_MAX};
    if (scalar && scalar->host_data && scalar->host_data_size == sizeof(owned))
        std::memcpy(owned.data(), scalar->host_data, sizeof(owned));
    CHECK(scalar && valid_raw_offset_scalar_snapshot_resource(*scalar) && scalar->size == 8u &&
              owned[0] == 2u && owned[1] == 0x77u,
          "the x2 source owns both observed words");
    CHECK(table.by_fetch_pc(7u) && table.by_fetch_pc(7u)->gpu_addr == address + 32u,
          "the low word selects the wide range");
    ComputeShaderConfig config;
    config.user_sgprs.assign(user.begin(), user.end());
    config.wave_size = 64u;
    CHECK(!recompile_compute(code.data(), code.size(), &table, config).empty(),
          "the x2-sourced program recompiles");
    auto short_backing = table;
    for (auto& r : short_backing.resources)
        if (r.fetch_pc == 0u) r.host_data_size = 4u;
    CHECK(recompile_compute(code.data(), code.size(), &short_backing, config).empty(),
          "an x2 source with only one owned word is refused");

    pair[0] = 3u;
    pair[1] = 0x88u;
    const auto changed = table_for();
    DrawItem first_draw, changed_draw;
    first_draw.vs = changed_draw.vs = {0x07230203u, 1u, 2u};
    first_draw.vrt = std::make_shared<ShaderResourceTable>(table);
    changed_draw.vrt = std::make_shared<ShaderResourceTable>(changed);
    uint64_t planned = 0;
    std::string error;
    CHECK(preflight_gpu_capture_draw_resources(first_draw, 1u << 20u, planned, error) &&
              planned == 24u,
          "capture budget includes the owned eight-byte source and selected sixteen-byte range");
    unsigned pair_reads = 0;
    const CaptureMemoryReader reader = [&](uint64_t addr, uint8_t* dst, size_t size) {
        if (addr >= pair_address && addr < pair_address + sizeof(pair)) ++pair_reads;
        if (addr < address || addr > address + sizeof(bytes) ||
            size > address + sizeof(bytes) - addr)
            return size_t{0};
        std::memcpy(dst, reinterpret_cast<const void*>(addr), size);
        return size;
    };
    GpuCaptureFile capture, restored;
    std::vector<uint8_t> encoded;
    GpuReplayFrame replay;
    const bool restored_ok =
        capture_draw_items({first_draw, changed_draw}, {}, reader, capture, error) &&
        serialize_gpu_capture(capture, encoded, error) &&
        deserialize_gpu_capture(encoded, restored, error) &&
        materialize_gpu_replay(restored, replay, error);
    std::array<uint32_t, 2> replay_first{}, replay_changed{};
    if (restored_ok && replay.items.size() == 2u && replay.items[0].vrt && replay.items[1].vrt) {
        const auto* first = replay.items[0].vrt->by_fetch_pc(0u);
        const auto* second = replay.items[1].vrt->by_fetch_pc(0u);
        if (first && first->host_data && first->host_data_size == 8u)
            std::memcpy(replay_first.data(), first->host_data, 8u);
        if (second && second->host_data && second->host_data_size == 8u)
            std::memcpy(replay_changed.data(), second->host_data, 8u);
    }
    if (!restored_ok) std::printf("capture error: %s\n", error.c_str());
    CHECK(restored_ok && pair_reads == 0u && replay_first == (std::array<uint32_t, 2>{2u, 0x77u}) &&
              replay_changed == (std::array<uint32_t, 2>{3u, 0x88u}),
          "capture owns each draw's own x2 pair without re-reading the guest");
}

// The x2 destination pair must stay below VCC: s[104:105] is a latched source, s[105:106] is not.
TEST(MemoryFedRawWide, X2SourceDestinationStaysBelowVcc) {
    std::vector<uint32_t> code = {
        0xf4041a01u, 0xfa000000u,   // x2 s[104:105],entry s[2:3],0
        0x8f6b8468u,   // s_lshl_b32 vcc_hi,s104,4
        0x876bff6bu, 0x000001f0u,   // s_and_b32 vcc_hi,vcc_hi,0x1f0
        0xf4080200u, 0xd6000000u,   // x4 s[8:11],entry s[0:1],vcc_hi
        0x7e000c0bu, 0xbf810000u,
    };
    CHECK(proof(code) == std::vector<uint32_t>{5u}, "an x2 pair ending at s105 is a source");
    code[0] = 0xf4041a41u;   // x2 s[105:106]: its high word is VCC_LO
    code[2] = 0x8f6b8469u;   // s_lshl_b32 vcc_hi,s105,4
    CHECK(proof(code).empty(), "an x2 pair reaching into VCC is not");
}
