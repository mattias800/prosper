// An x2 pointer followed by numeric VCC data must use one owned readpoint and never stale masks.
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/resources/fold_reader.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <cstdint>
#include <utility>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;
namespace {
std::vector<uint32_t> program() {
    return {0xf4041a80u, 0xfa000004u, // x2 VCC <- entry s0:1 + 4
            0xf4041ab5u, 0xfa000038u, // x2 VCC <- former pointer + 0x38
            0x93046b6au,              // s_mul_i32 s4, VCC_LO, VCC_HI
            0x906a8404u,              // s_lshr_b32 VCC_LO, s4, 4 (partial overwrite)
            0x7e000c6au,              // v_cvt_f32_u32 v0, VCC_LO
            0x7d820204u,              // v_cmp_lt_u32 VCC, s4, v1 (fresh mask)
            0xbf810000u};
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    return result;
}
std::vector<RawNestedWideChain> chains(const std::vector<uint32_t>& code) {
    return rdna2_owned_raw_x2_chains(decode(code));
}
ShaderResource source(uint32_t pc, uint32_t binding, uint64_t address, uint8_t* data) {
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.fetch_pc = pc;
    r.binding = binding;
    r.gpu_addr = address;
    r.size = 8;
    r.host_data = data;
    r.host_data_size = 8;
    r.owned_nested_snapshot_bytes = 8;
    return r;
}
} // namespace
TEST(OwnedRawX2, DefinitionLifetimes) {
    EXPECT_EQ(chains(program()), (std::vector<RawNestedWideChain>{{0, 2, 8, 8, 4, 0x38}}));
    EXPECT_EQ(raw_snapshot_write_plan(decode(program()), chains(program())).required_user_words,
              2u);
    auto clobbered = program();
    clobbered.insert(clobbered.begin() + 2, 0xbeeb0380u);
    EXPECT_TRUE(chains(clobbered).empty()) << "overwriting only pointer HI invalidates SBASE";
    auto entry = program();
    entry.insert(entry.begin(), 0xbe800380u);
    EXPECT_TRUE(chains(entry).empty()) << "the base must still be genuine entry data at its read";
    auto one_word = program();
    one_word[4] = 0x9304816au;
    EXPECT_TRUE(chains(one_word).empty()) << "MUL with one original word cannot prove the pair";
    auto implicit = program();
    implicit.insert(implicit.begin() + 7, 0x02000080u);
    EXPECT_TRUE(chains(implicit).empty()) << "cndmask cannot consume a Bool from before the load";
    auto bypass = program();
    bypass.insert(bypass.begin(), 0xbf880002u);
    EXPECT_TRUE(chains(bypass).empty()) << "a path bypassing the parent cannot invent a definition";
    auto loop = program();
    loop.insert(loop.end() - 1, 0xbf82fff7u);
    EXPECT_TRUE(chains(loop).empty()) << "re-entering an owned read needs a fresh observation";
    auto mask_loop = program();
    mask_loop.insert(mask_loop.end() - 1, {0xbf800000u, 0xbf82fffeu});
    EXPECT_EQ(chains(mask_loop).size(), 1u)
        << "a downstream mask-phase loop does not reread the load";
    mask_loop[9] = 0xbf82fffau; // back into numeric MUL pc4, after its mask was overwritten
    EXPECT_TRUE(chains(mask_loop).empty());
    auto ordinary = program();
    ordinary[0] = 0xf4040100u;
    ordinary[2] = 0xf4040102u;
    ordinary[4] = 0x93060504u;
    ordinary[5] = 0x90048406u;
    ordinary[6] = 0x7e000c04u;
    EXPECT_TRUE(chains(ordinary).empty()) << "other numeric pairs need their own proven lifetime";
    auto incomplete = program();
    incomplete.resize(4);
    EXPECT_TRUE(chains(incomplete).empty())
        << "an unobserved numeric use cannot be inferred at EOF";
    auto unknown = decode(program());
    unknown[2].fmt = Rdna2Format::Unknown;
    EXPECT_TRUE(rdna2_owned_raw_x2_chains(unknown).empty());
}
TEST(OwnedRawX2, OnlyAnUnnamedWriterOrTransferVoidsTheChains) {
    // Past the fresh compare, where the numeric lifetime has ended and nothing reads the pair.
    const auto with = [](uint32_t word) {
        auto code = program();
        code.insert(code.end() - 1, word);
        return code;
    };
    // s_orn2/s_nand/s_nor_saveexec_b64 s[40:41], -1 (SOP1 0x28..0x2a): ordinary mask transfers.
    // They were refused here under the name "relative SGPR write".
    for (uint32_t opcode : {0x28u, 0x29u, 0x2au}) {
        const auto code = with(0xbea800c1u | (opcode << 8u));
        ASSERT_EQ(decode(code)[code.size() - 4].opcode, opcode);
        EXPECT_EQ(chains(code).size(), 1u) << "SOP1 0x" << std::hex << opcode;
    }
    // The instructions that range was meant to be, and the other escapes: each may write a
    // register its encoding does not name, or run code that was never decoded.
    struct Escape {
        const char* name;
        uint32_t word;
    };
    const Escape escapes[] = {
        {"s_movreld_b32 s60, s40", 0xbebc3028u},
        {"s_movreld_b64 s[60:61], s[40:41]", 0xbebc3128u},
        {"s_movrelsd_2_b32 s60, s40", 0xbebc4928u},
        {"s_setpc_b64 s[60:61]", 0xbe80203cu},
        {"s_call_b64 s[60:61], +0", 0xbb3c0000u},
    };
    for (const Escape& escape : escapes) {
        const Rdna2Inst in = rdna2_decode_one(&escape.word, 1);
        ASSERT_TRUE(rdna2_may_write_unnamed_register_or_leave_cfg(in)) << escape.name;
        EXPECT_TRUE(chains(with(escape.word)).empty()) << escape.name;
    }
    // s_movrels_b32 s60, s40 writes the register it names and reads s40..s105, never VCC: the
    // chains stand, as they did before.
    EXPECT_EQ(chains(with(0xbebc2e28u)).size(), 1u);
}
TEST(OwnedRawX2, BothControlPathsMustReplaceMaskBeforeConsumer) {
    auto code = program();
    code.insert(code.begin() + 5, 0xbf880002u); // branch pc5 -> fresh compare pc8
    code.insert(code.end() - 1, 0x02000080u);  // consumer pc9
    EXPECT_EQ(chains(code).size(), 1u);
    code[5] = 0xbf880003u; // same CFG, but one path reaches pc9 without a new predicate
    EXPECT_TRUE(chains(code).empty());
}
TEST(OwnedRawX2, PartialWritesRequireOwnedReachingDefinitions) {
    auto code = program();
    code[5] = 0xbeea0364u; // s_mov_b32 VCC_LO, s100 (no owned definition)
    EXPECT_TRUE(chains(code).empty());
    code = program();
    code.insert(code.begin() + 5, 0xbeeb0364u); // unowned VCC_HI overwrite
    EXPECT_TRUE(chains(code).empty());
    code = program();
    code.insert(code.begin() + 5, 0xbe840364u);   // lose the product in s4 before its VCC_LO write
    EXPECT_TRUE(chains(code).empty());
    code = program();
    code.insert(code.begin() + 5, {0xf4040102u, 0xfa000004u});   // unrelated SMEM x2 -> s4:s5
    EXPECT_TRUE(chains(code).empty());
    code = program();
    code.insert(code.begin() + 5, 0xbe840385u);   // real inline 5 replaces the product
    EXPECT_EQ(chains(code).size(), 1u);
    code = program();
    code.insert(code.begin() + 5, {0xbf880001u, 0xbe840364u});
    EXPECT_TRUE(chains(code).empty()) << "s4 is unknown on one incoming path to the partial write";
    code[6] = 0xbe840385u;
    EXPECT_EQ(chains(code).size(), 1u) << "both incoming definitions contain genuine numeric data";
}
TEST(OwnedRawX2, EmissionRequiresBothExactOwners) {
    const auto code = program();
    std::array<uint32_t, 2> pointer{0x200000u, 0x20u}, numeric{60u, 68u};
    ShaderResourceTable table;
    table.resources = {source(0, 2, 0x2000100004ull, reinterpret_cast<uint8_t*>(pointer.data())),
                       source(2, 3, 0x2000200038ull, reinterpret_cast<uint8_t*>(numeric.data()))};
    ASSERT_FALSE(recompile_valu(code.data(), code.size(), 2, 0, &table).empty());
    auto malformed = table;
    malformed.resources[1].host_data_size = 4;
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 2, 0, &malformed).empty());
    malformed = table;
    malformed.resources[0].owned_nested_snapshot_bytes = 0;
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 2, 0, &malformed).empty());
    malformed = table;
    malformed.resources.push_back(malformed.resources[1]);
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 2, 0, &malformed).empty());
    malformed = table;
    malformed.resources.clear();
    ShaderResource unrelated = table.resources[0];
    unrelated.fetch_pc = UINT32_MAX;
    malformed.resources.push_back(unrelated);
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 2, 0, &malformed).empty())
        << "binding2 must not supply a substitute for either exact owner";
    auto mask = code;
    mask.insert(mask.begin() + 7, 0x02000080u);
    EXPECT_TRUE(recompile_valu(mask.data(), mask.size(), 2, 0, &table).empty());
}

class OwnedRawX2Mapped : public ::testing::Test {
protected:
    static constexpr uint64_t page = 0x10000u;
    uint64_t physical = 0, parent = 0, child = 0, output = 0, alias = 0;
    HleFn unmap = nullptr, release = nullptr;
    void SetUp() override {
        register_builtin_hle();
        const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
        unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
        release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
        ASSERT_TRUE(allocate && map && unmap && release);
        ASSERT_EQ(
            allocate(0, 0x200000000ull, 3 * page, page, 0, reinterpret_cast<uint64_t>(&physical)),
            0);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&parent), page, 3, 0, physical, page), 0);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&child), page, 3, 0, physical + page, page), 0);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&output), page, 3, 0, physical + 2 * page, page),
                  0);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&alias), page, 3, 0, physical + page, page), 0);
        {
            GuestMappingLease lease;
            if (!guest_memory_direct_range_fault_safe(lease, parent, page))
                GTEST_SKIP() << "this platform has no live direct-range fault proof";
        }
        std::memcpy(reinterpret_cast<void*>(parent + 4), &child, 8);
        const uint32_t words[2] = {60, 68};
        std::memcpy(reinterpret_cast<void*>(child + 0x38), words, 8);
        set_graphics_raw_source_authority(
            [](const GuestMappingLease&, uint64_t, uint32_t) { return true; });
    }
    void TearDown() override {
        set_graphics_raw_source_authority({});
        if (unmap)
            for (auto p : {parent, child, output, alias})
                if (p) unmap(p, page, 0, 0, 0, 0);
        if (release && physical) release(physical, 3 * page, 0, 0, 0, 0);
    }
    bool observe(GraphicsNestedWideReader& reader) {
        return reader.probe(FoldProbe::Raw, 0, parent + 4, 8) &&
               reader.word(0, parent + 4) == static_cast<uint32_t>(child) &&
               reader.probe(FoldProbe::Raw, 2, child + 0x38, 8);
    }
};
TEST_F(OwnedRawX2Mapped, SameOwnerSurvivesGuestChangesAfterObservation) {
    GuestMappingLease lease;
    GraphicsRawSnapshotContext context{true, {}};
    const auto decoded = decode(program());
    GraphicsNestedWideReader reader(rdna2_owned_raw_x2_chains(decoded), &context, &lease);
    ASSERT_TRUE(observe(reader));
    const uint32_t changed[2] = {111, 222};
    std::memcpy(reinterpret_cast<void*>(child + 0x38), changed, 8);
    std::memcpy(reinterpret_cast<void*>(parent + 4), &output, 8);
    EXPECT_EQ(reader.word(2, child + 0x38), 60u);
    EXPECT_EQ(reader.word(2, child + 0x3c), 68u);
    ShaderResourceTable table;
    ASSERT_TRUE(reader.publish_compute_x2(table, raw_snapshot_write_plan(decoded), {}));
    const auto* owned = owned_nested_snapshot_at(table, 2, 8);
    ASSERT_NE(owned, nullptr);
    uint32_t retained[2];
    std::memcpy(retained, owned->host_data, 8);
    EXPECT_EQ(retained[0], 60u);
    EXPECT_EQ(retained[1], 68u);
}
TEST_F(OwnedRawX2Mapped, MissingEpochCurrentBytesAndPointerEquationRefuse) {
    GuestMappingLease lease;
    const auto proof = chains(program());
    GraphicsRawSnapshotContext context{false, {}};
    GraphicsNestedWideReader incomplete(proof, &context, &lease);
    EXPECT_FALSE(incomplete.probe(FoldProbe::Raw, 0, parent + 4, 8));
    context.producers_complete = true;
    set_graphics_raw_source_authority({});
    GraphicsNestedWideReader no_authority(proof, &context, &lease);
    EXPECT_FALSE(no_authority.probe(FoldProbe::Raw, 0, parent + 4, 8));
    set_graphics_raw_source_authority(
        [](const GuestMappingLease&, uint64_t, uint32_t) { return false; });
    GraphicsNestedWideReader retained_image(proof, &context, &lease);
    EXPECT_FALSE(retained_image.probe(FoldProbe::Raw, 0, parent + 4, 8));
    set_graphics_raw_source_authority(
        [](const GuestMappingLease&, uint64_t, uint32_t) { return true; });
    GraphicsNestedWideReader wrong_child(proof, &context, &lease);
    ASSERT_TRUE(wrong_child.probe(FoldProbe::Raw, 0, parent + 4, 8));
    EXPECT_FALSE(wrong_child.probe(FoldProbe::Raw, 2, output + 0x38, 8));
    EXPECT_FALSE(wrong_child.probe(FoldProbe::Base48, 2, child + 0x38, 8));
    ShaderResourceTable table;
    EXPECT_FALSE(
        wrong_child.publish_compute_x2(table, raw_snapshot_write_plan(decode(program())), {}));
    EXPECT_TRUE(table.resources.empty());
}
TEST_F(OwnedRawX2Mapped, PhysicalStoreAliasAndDescriptorChangesRefuse) {
    auto code = program();
    code.insert(code.end() - 1, {0xf0200f08u, 0x00060004u});
    const auto decoded = decode(code);
    const auto store_pc = decoded[decoded.size() - 2].pc;
    ASSERT_EQ(decoded[decoded.size() - 2].opcode, 8u);
    SrtUse use;
    use.kind = 0;
    use.use_pc = store_pc;
    use.descriptor_source_addr = parent + 0x100;
    std::memcpy(reinterpret_cast<void*>(use.descriptor_source_addr), use.t8.data(), 32);
    ShaderResource target;
    target.cls = ResourceClass::StorageImage;
    target.gpu_addr = output;
    target.fetch_pc = store_pc;
    target.width = target.height = 64;
    target.depth = target.sample_count = target.declared_mip_levels = 1;
    target.img_dim = 1;
    target.tile_mode = 27;
    target.format = DataFormat::Uint8;
    target.num_components = 1;
    target.size = 64 * 64;
    GuestMappingLease lease;
    GraphicsRawSnapshotContext context{true, {}};
    GraphicsNestedWideReader reader(rdna2_owned_raw_x2_chains(decoded), &context, &lease);
    ASSERT_TRUE(observe(reader));
    ShaderResourceTable okay;
    okay.resources.push_back(target);
    ASSERT_TRUE(reader.publish_compute_x2(okay, raw_snapshot_write_plan(decoded), {use}));
    ShaderResourceTable bad;
    target.gpu_addr = alias;
    bad.resources.push_back(target);
    EXPECT_FALSE(reader.publish_compute_x2(bad, raw_snapshot_write_plan(decoded), {use}))
        << "different VAs backed by the same physical page must remain an alias";
    EXPECT_EQ(bad.resources.size(), 1u);
    EXPECT_TRUE(bad.owned_host_data.empty());
    bad.resources[0].gpu_addr = output;
    bad.resources[0].metadata_addr = output;
    EXPECT_FALSE(reader.publish_compute_x2(bad, raw_snapshot_write_plan(decoded), {use}));
    bad.resources[0].metadata_addr = 0;
    *reinterpret_cast<uint32_t*>(use.descriptor_source_addr) = 1;
    EXPECT_FALSE(reader.publish_compute_x2(bad, raw_snapshot_write_plan(decoded), {use}));
}

TEST(OwnedRawX2, CacheAdmissionIsDistinctFromReusableCode) {
    const auto code = program();
    std::array<uint32_t, 2> pointer{0x200000, 0x20}, words{60, 68};
    ShaderResourceTable table;
    table.resources = {source(0, 2, 0x2000100004ull, reinterpret_cast<uint8_t*>(pointer.data())),
                       source(2, 3, 0x2000200038ull, reinterpret_cast<uint8_t*>(words.data()))};
    ComputeShaderConfig config;
    config.local_x = 64;
    config.user_sgprs = {1, 2};
    clear_shader_recompile_cache();
    uint64_t valid = 0, changed = 0, refused = 0;
    const auto first =
        recompile_compute_shader_cached(code.data(), code.size(), &table, config, &valid);
    ASSERT_FALSE(first.empty());
    words = {7, 11};
    table.resources[1].gpu_addr += 0x100;
    EXPECT_EQ(recompile_compute_shader_cached(code.data(), code.size(), &table, config, &changed),
              first);
    EXPECT_EQ(changed, valid) << "module identity does not cache observed numeric values";
    auto missing = table;
    missing.resources[1].owned_nested_snapshot_bytes = 0;
    EXPECT_TRUE(
        recompile_compute_shader_cached(code.data(), code.size(), &missing, config, &refused)
            .empty());
    EXPECT_NE(refused, valid) << "warm compiled code cannot supply absent dispatch authority";
    EXPECT_EQ(recompile_compute_shader_cached(code.data(), code.size(), &table, config), first);
}
TEST(OwnedRawX2, CaptureRoundtripRebuildsBothOriginalObligations) {
    const auto code = program();
    GpuCaptureFile capture;
    GpuCapturedCompute compute;
    compute.raw_shader_index = 0;
    compute.resources.present = true;
    compute.recompile_config_available = true;
    compute.recompile_config.user_sgprs = {1, 2};
    compute.recompile_config.local_x = compute.launch.local_x;
    GpuCaptureRawShaderVersion raw;
    raw.words = code;
    raw.has_endpgm = true;
    raw.content_hash =
        gpu_capture_hash(reinterpret_cast<const uint8_t*>(code.data()), code.size() * 4);
    capture.raw_shader_versions.push_back(raw);
    const uint64_t pointer = 0x2000200000ull;
    for (uint32_t k = 0; k < 2; ++k) {
        GpuCaptureBlob blob;
        blob.guest_addr = k ? pointer + 0x38 : 0x2000100004ull;
        blob.bytes.resize(8);
        blob.bytes_read = 8;
        if (k) {
            const uint32_t words[2] = {60, 68};
            std::memcpy(blob.bytes.data(), words, 8);
        } else
            std::memcpy(blob.bytes.data(), &pointer, 8);
        blob.content_hash = gpu_capture_hash(blob.bytes);
        GpuCapturedResource resource;
        resource.resource = source(k * 2, k + 2, blob.guest_addr, nullptr);
        resource.resource.host_data_size = 0;
        resource.captured_size = 8;
        resource.blob_index = k;
        compute.resources.resources.push_back(resource);
        capture.blobs.push_back(blob);
    }
    capture.computes.push_back(compute);
    std::vector<uint8_t> bytes;
    std::string error;
    GpuCaptureFile restored;
    GpuReplayFrame replay;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, restored, error)) << error;
    ASSERT_TRUE(materialize_gpu_replay(restored, replay, error)) << error;
    ASSERT_EQ(replay.computes.size(), 1u);
    const auto& table = *replay.computes[0].resources;
    EXPECT_EQ(table.owned_nested_snapshot_requirements,
              (std::vector<std::pair<uint32_t, uint32_t>>{{0, 8}, {2, 8}}));
    ASSERT_NE(owned_nested_snapshot_at(table, 2, 8), nullptr);
    uint32_t words[2];
    std::memcpy(words, owned_nested_snapshot_at(table, 2, 8)->host_data, 8);
    EXPECT_EQ(words[0], 60u);
    EXPECT_EQ(words[1], 68u);
    auto bad = restored;
    bad.computes[0].resources.resources[0].resource.owned_nested_snapshot_bytes = 0;
    EXPECT_FALSE(materialize_gpu_replay(bad, replay, error));
    bad = restored;
    bad.blobs[1].bytes_read = 4;
    EXPECT_FALSE(materialize_gpu_replay(bad, replay, error));
    bad = restored;
    bad.raw_shader_versions[0].words[4] = 0x9304816au;
    EXPECT_FALSE(materialize_gpu_replay(bad, replay, error));
    bad = restored;
    bad.computes[0].resources.resources[1].resource.gpu_addr += 4;
    EXPECT_FALSE(materialize_gpu_replay(bad, replay, error));
}
