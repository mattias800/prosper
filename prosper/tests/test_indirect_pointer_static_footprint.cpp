#include "gpu/rdna2_decode.hpp"
#include "gpu/gpu_capture.hpp"
#include "gpu/gpu_dependency_graph.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/rdna2_to_spirv.hpp"
#include "gpu/shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

uint8_t nibble(char value) {
    return value >= '0' && value <= '9' ? static_cast<uint8_t>(value - '0')
                                        : static_cast<uint8_t>(value - 'a' + 10);
}

std::vector<uint32_t> program() {
    const std::filesystem::path path = std::filesystem::path(__FILE__).parent_path() /
        "data/rdna2_indirect_pointer_static_footprint.hex";
    std::ifstream input(path);
    std::string hex((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    CHECK(input.good() || input.eof(), "static-footprint program fixture is readable");
    CHECK(hex.size() == 386u * 8u, "static-footprint program fixture has 386 dwords");
    std::vector<uint32_t> words(hex.size() / 8u);
    auto* bytes = reinterpret_cast<uint8_t*>(words.data());
    for (size_t index = 0; index < words.size() * sizeof(uint32_t); ++index)
        bytes[index] = static_cast<uint8_t>(
            nibble(hex[index * 2u]) << 4u | nibble(hex[index * 2u + 1u]));
    return words;
}

void store_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void store_u64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

struct Fixture {
    static constexpr uint32_t kStride = 96u;
    static constexpr uint32_t kSelectorOffset = 88u;

    std::vector<uint8_t> source;
    ShaderResourceTable table;
    ComputeShaderConfig config;

    explicit Fixture(uint32_t threads = 8u)
        : source(static_cast<size_t>(threads) * kStride) {
        for (uint32_t index = 0; index < threads; ++index) {
            uint64_t pointer = 0x0000000200001000ull +
                static_cast<uint64_t>(index) * 0x1000u;
            uint32_t selector = 1u;
            if (index == 0u) {
                // The exact U64 CMPX guard keeps this active even though its low word is zero.
                pointer = 0x0000000100000000ull;
                selector = 0u;
            } else if (index == 1u) {
                // A nonzero pointer can be statically inactive by selector and remains a witness.
                pointer = 0x0000000123456000ull;
                selector = 1u;
            } else if (index == 2u) {
                pointer = 0x0000000200002000ull;
                selector = 2u;
            } else if (index == 3u) {
                pointer = 0u;
                selector = 0u;
            }
            store_u64(source, static_cast<size_t>(index) * kStride, pointer);
            store_u32(source, static_cast<size_t>(index) * kStride + kSelectorOffset, selector);
        }

        ShaderResource pointer_source;
        pointer_source.cls = ResourceClass::ConstantBuffer;
        pointer_source.format = DataFormat::Uint32;
        pointer_source.num_components = 1u;
        pointer_source.gpu_addr = reinterpret_cast<uint64_t>(source.data());
        pointer_source.size = static_cast<uint32_t>(source.size());
        pointer_source.stride = kStride;
        pointer_source.srt_offset = 0xa0u;
        pointer_source.fetch_pc = 6u;
        pointer_source.host_data = source.data();
        pointer_source.host_data_size = source.size();
        table.resources.push_back(pointer_source);

        ShaderResource selector_source = pointer_source;
        selector_source.srt_offset = UINT32_MAX;
        selector_source.fetch_pc = 16u;
        table.resources.push_back(selector_source);

        config.user_sgprs.resize(5u);
        config.local_x = 64u;
        config.local_y = config.local_z = 1u;
        config.exact_thread_extent = true;
        config.threads_x = threads;
        config.threads_y = config.threads_z = 1u;
        config.wave_size = 64u;
        config.tgid_x_en = true;
        config.tidig_comp_cnt = 0u;
    }
};

struct LiveFixture {
    static constexpr uint32_t kStride = 96u;
    static constexpr uint32_t kSelectorOffset = 88u;
    static constexpr uint32_t kPointeeBytes = 52u;

    uint32_t threads;
    std::vector<std::array<uint8_t, kPointeeBytes>> pointees;
    std::vector<uint8_t> source;
    std::vector<uint8_t> scalar;
    std::vector<uint8_t> output;
    ShaderResourceTable table;
    ComputeShaderConfig config;

    explicit LiveFixture(uint32_t thread_count = 13u)
        : threads(thread_count), pointees(thread_count),
          source(static_cast<size_t>(thread_count) * kStride), scalar(116u),
          output(static_cast<size_t>(thread_count) * 128u) {
        for (uint32_t index = 0; index < threads; ++index) {
            for (uint32_t byte = 0; byte < kPointeeBytes; ++byte)
                pointees[index][byte] = static_cast<uint8_t>(index * 37u + byte);
            const uint32_t selector = index % 3u;
            const uint64_t pointer = index == 3u ? 0u
                : reinterpret_cast<uint64_t>(pointees[index].data());
            store_u64(source, static_cast<size_t>(index) * kStride, pointer);
            store_u32(source, static_cast<size_t>(index) * kStride + kSelectorOffset,
                      selector);
        }

        for (uint32_t pc : {6u, 8u, 10u, 12u, 14u, 16u}) {
            ShaderResource resource;
            resource.cls = ResourceClass::ConstantBuffer;
            resource.format = DataFormat::Uint8;
            resource.num_components = 1u;
            resource.gpu_addr = reinterpret_cast<uint64_t>(source.data());
            resource.size = static_cast<uint32_t>(source.size());
            resource.stride = kStride;
            resource.srt_offset = pc == 6u ? 0xa0u : UINT32_MAX;
            resource.fetch_pc = pc;
            resource.host_data = source.data();
            resource.host_data_size = source.size();
            table.resources.push_back(resource);
        }

        ShaderResource scalar_resource;
        scalar_resource.cls = ResourceClass::ConstantBuffer;
        scalar_resource.format = DataFormat::Uint32;
        scalar_resource.num_components = 1u;
        scalar_resource.gpu_addr = reinterpret_cast<uint64_t>(scalar.data());
        scalar_resource.size = static_cast<uint32_t>(scalar.size());
        scalar_resource.fetch_pc = 41u;
        scalar_resource.host_data = scalar.data();
        scalar_resource.host_data_size = scalar.size();
        table.resources.push_back(scalar_resource);

        for (uint32_t pc : {358u, 360u, 362u, 364u, 366u,
                            368u, 370u, 372u, 383u}) {
            ShaderResource resource;
            resource.cls = ResourceClass::ConstantBuffer;
            resource.format = DataFormat::Float32;
            resource.num_components = 1u;
            resource.gpu_addr = reinterpret_cast<uint64_t>(output.data());
            resource.size = static_cast<uint32_t>(output.size());
            resource.stride = 128u;
            resource.fetch_pc = pc;
            resource.host_data = output.data();
            resource.host_data_size = output.size();
            table.resources.push_back(resource);
        }

        config.user_sgprs.resize(5u);
        config.local_x = 64u;
        config.local_y = config.local_z = 1u;
        config.exact_thread_extent = true;
        config.threads_x = threads;
        config.threads_y = config.threads_z = 1u;
        config.wave_size = 64u;
        config.tgid_x_en = true;
        config.tidig_comp_cnt = 0u;
    }
};

bool analyze(const std::vector<uint32_t>& words, const Fixture& fixture,
             IndirectPointerRelocationProof* output = nullptr) {
    IndirectPointerRelocationProof proof;
    const bool accepted = analyze_rdna2_static_pointer_footprint(
        words.data(), words.size(), fixture.config, fixture.table, proof);
    if (output) *output = std::move(proof);
    return accepted;
}

const Rdna2Inst* instruction_at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    for (const Rdna2Inst& instruction : instructions)
        if (instruction.pc == pc) return &instruction;
    return nullptr;
}

} // namespace

int main() {
    const std::vector<uint32_t> exact = program();
    Fixture fixture;
    IndirectPointerRelocationProof proof;
    CHECK(analyze(exact, fixture, &proof),
          "exact structural producer/guard/consumer shape is admitted");
    CHECK(proof.schema_version == kIndirectPointerProofSchema &&
              proof.bound_kind == IndirectPointerBoundKind::StaticFootprint &&
              proof.guard_kind == IndirectPointerGuardKind::Full64NonZero,
          "proof names the generic StaticFootprint and complete-U64 guard semantics");
    CHECK(proof.source_fetch_pc == 6u && proof.source_stride == 96u &&
              proof.pointer_byte_offset == 0u &&
              proof.footprint_selector_byte_offset == 88u &&
              proof.selector_footprint_bytes == std::vector<uint32_t>({52u, 0u, 28u}) &&
              proof.max_footprint_bytes == 52u,
          "proof retains the exact record layout and selector-specific bounds");
    CHECK(proof.accesses.size() == 3u && proof.accesses[0].pc == 50u &&
              proof.accesses[1].pc == 274u && proof.accesses[2].pc == 276u &&
              proof.accesses[0].immediate_byte_offset == 24u &&
              proof.accesses[1].immediate_byte_offset == 24u &&
              proof.accesses[2].immediate_byte_offset == 48u,
          "proof enumerates all and only the three exact GLOBAL consumers");
    CHECK(proof.records.size() == fixture.config.threads_x &&
              proof.records[0].guest_address == 0x0000000100000000ull &&
              proof.records[0].byte_count == 52u,
          "low-word-zero/high-word-nonzero U64 pointer remains active");
    CHECK(proof.records[1].guest_address == 0x0000000123456000ull &&
              proof.records[1].byte_count == 0u,
          "selector-one nonzero pointer is retained as an inactive witness");
    CHECK(proof.records[2].byte_count == 28u &&
              proof.records[3].guest_address == 0u && proof.records[3].byte_count == 0u,
          "selector-two and exact-null records receive their exact bounds");
    CHECK(proof.fingerprint != 0u && proof.witness_words.size() == 4u &&
              proof.witness_words[0] == kIndirectPointerProofSchema &&
              proof.witness_words[3] ==
                  (kIndirectPointerStaticFootprintTag ^ proof.witness_words[0] ^
                   proof.witness_words[1] ^ proof.witness_words[2]),
          "proof fingerprint has a self-checking four-word carrier witness");

    std::vector<Rdna2Inst> decoded;
    rdna2_walk(exact.data(), exact.size(), decoded);
    const Rdna2Inst* pc50 = instruction_at(decoded, 50u);
    CHECK(pc50 && rdna2_indirect_pointer_access(proof, *pc50) == &proof.accesses[0],
          "access lookup requires and returns the exact pc50 packet proof");
    if (pc50) {
        Rdna2Inst changed_packet = *pc50;
        changed_packet.words[1] ^= 1u;
        CHECK(!rdna2_indirect_pointer_access(proof, changed_packet),
              "PC alone cannot borrow relocation authority after a packet mutation");
    }

    for (uint32_t threads : {13u, 152u}) {
        Fixture extent(threads);
        IndirectPointerRelocationProof extent_proof;
        CHECK(analyze(exact, extent, &extent_proof) &&
                  extent_proof.records.size() == threads &&
                  extent_proof.records.back().source_byte_offset == (threads - 1u) * 96u,
              "exact thread extent derives one complete 96-byte record per invocation");
    }

    struct Mutation { uint32_t pc; uint32_t word; const char* name; };
    const std::array<Mutation, 27> mutations{{
        {1u, 0u, "global record-index producer"},
        {3u, 1u, "source descriptor load"},
        {5u, 0u, "descriptor producer lifetime"},
        {6u, 0u, "pointer record load"},
        {8u, 1u, "record-index producer lifetime"},
        {16u, 0u, "selector record load"},
        {29u, 1u, "selector-two mask producer"},
        {38u, 0u, "pre-filter EXEC save"},
        {39u, 0u, "selector-one exclusion"},
        {43u, 0u, "filtered-empty exit"},
        {44u, 0u, "mode-two saveexec split"},
        {45u, 0u, "mode-two empty branch"},
        {47u, 0u, "mode-two EXEC save"},
        {48u, 0u, "mode-two U64 null guard"},
        {49u, 0u, "mode-two null branch"},
        {50u, 0u, "mode-two GLOBAL +24"},
        {78u, 0u, "mode-two EXEC restore"},
        {267u, 0u, "complementary mask"},
        {268u, 0u, "complement-empty branch"},
        {271u, 0u, "mode-zero EXEC save"},
        {272u, 0u, "mode-zero U64 null guard"},
        {273u, 0u, "mode-zero null branch"},
        {274u, 0u, "mode-zero GLOBAL +24"},
        {276u, 0u, "mode-zero GLOBAL +48"},
        {298u, 0u, "mode-zero EXEC restore"},
        {351u, 0u, "pre-arm EXEC restore"},
        {374u, 0u, "original EXEC complement"},
    }};
    for (const Mutation& mutation : mutations) {
        std::vector<uint32_t> changed = exact;
        changed[mutation.pc + mutation.word] ^= 1u;
        CHECK(!analyze(changed, fixture), mutation.name);
    }

    std::vector<uint32_t> descriptor_clobber = exact;
    descriptor_clobber[5u] = 0xbe880400u; // s_mov_b64 s[8:9], s[0:1]
    CHECK(!analyze(descriptor_clobber, fixture),
          "same-site pc5 scalar clobber cannot sever descriptor provenance");

    std::vector<uint32_t> prefix_branch = exact;
    prefix_branch[0u] = 0xbf820013u; // s_branch pc0 -> pc20, skipping all producers
    CHECK(!analyze(prefix_branch, fixture),
          "a prefix branch cannot bypass the complete pointer producer chain");

    std::vector<uint32_t> index_clobber = exact;
    index_clobber[9u] = (index_clobber[9u] & ~0xff00u) | (43u << 8u);
    CHECK(!analyze(index_clobber, fixture),
          "intervening load cannot redefine v43 before the selector fetch");

    std::vector<uint32_t> selector_clobber = exact;
    selector_clobber[20u] = 0x7e040280u; // v_mov_b32 v2, 0
    CHECK(!analyze(selector_clobber, fixture),
          "selector v2 must reach both mask consumers without redefinition");

    std::vector<uint32_t> pointer_tfe_clobber = exact;
    pointer_tfe_clobber[31u] = 0xe0382000u;
    pointer_tfe_clobber[32u] = 0x8082142bu; // load v20..v23; TFE status overwrites v24
    CHECK(!analyze(pointer_tfe_clobber, fixture),
          "a TFE status destination cannot hide a pointer-register redefinition");

    std::vector<uint32_t> inner_exec_clobber = exact;
    inner_exec_clobber[120u] = 0xbefe0406u; // s_mov_b64 exec, s[6:7]
    CHECK(!analyze(inner_exec_clobber, fixture),
          "unlisted inner EXEC restore cannot poison the complementary mode mask");

    std::vector<uint32_t> inner_saved_mask_clobber = exact;
    inner_saved_mask_clobber[120u] = 0xbe820406u; // s_mov_b64 s[2:3], s[6:7]
    CHECK(!analyze(inner_saved_mask_clobber, fixture),
          "inner scalar write cannot clobber the saved mask consumed by pc267");

    std::vector<uint32_t> overlapping_saved_mask_clobber = exact;
    overlapping_saved_mask_clobber[120u] = 0x87810100u; // s_and_b64 s[1:2], s0, s1
    CHECK(!analyze(overlapping_saved_mask_clobber, fixture),
          "a B64 write beginning before s[2:3] cannot hide its high-word clobber");

    std::vector<uint32_t> original_mask_clobber = exact;
    original_mask_clobber[120u] = 0xbe800406u; // s_mov_b64 s[0:1], s[6:7]
    CHECK(!analyze(original_mask_clobber, fixture),
          "the original pc38 EXEC mask must survive until its pc374 complement");

    std::vector<uint32_t> mode_mask_clobber = exact;
    mode_mask_clobber[40u] = 0xbe860400u; // s_mov_b64 s[6:7], s[0:1]
    CHECK(!analyze(mode_mask_clobber, fixture),
          "selector-two mask must reach pc44 without a scalar redefinition");

    std::vector<uint32_t> first_guard_mask_clobber = exact;
    first_guard_mask_clobber[53u] = 0xbe860400u;
    CHECK(!analyze(first_guard_mask_clobber, fixture),
          "mode-two guard mask must survive until its pc78 restore");

    std::vector<uint32_t> second_guard_mask_clobber = exact;
    second_guard_mask_clobber[278u] = 0xbe860400u;
    CHECK(!analyze(second_guard_mask_clobber, fixture),
          "mode-zero guard mask must survive until its pc298 restore");

    std::vector<uint32_t> inner_trap = exact;
    inner_trap[120u] = 0xbf920000u; // s_trap 0
    CHECK(!analyze(inner_trap, fixture),
          "unlisted inner control transfer cannot truncate the proven consumer domain");

    std::vector<uint32_t> trailing_indirect_branch = exact;
    trailing_indirect_branch[376u] = 0xbe842000u; // s_setpc_b64 s[4:5]
    CHECK(!analyze(trailing_indirect_branch, fixture),
          "an indirect transfer anywhere in the program invalidates whole-program authority");

    std::vector<uint32_t> subvector_loop = exact;
    subvector_loop[376u] = 0xbd840000u; // s_subvector_loop_begin s4, 0
    CHECK(!analyze(subvector_loop, fixture),
          "a subvector-loop control marker invalidates whole-program authority");

    std::vector<uint32_t> unrelated = exact;
    unrelated[20u] ^= 1u; // waitcnt immediate: neither proof producer nor consumer
    IndirectPointerRelocationProof unrelated_proof;
    CHECK(analyze(unrelated, fixture, &unrelated_proof) &&
              unrelated_proof.fingerprint == proof.fingerprint,
          "analyzer is structural rather than complete-program byte identity");

    Fixture bad_selector;
    store_u32(bad_selector.source, Fixture::kSelectorOffset, 3u);
    CHECK(!analyze(exact, bad_selector),
          "selector outside the complete 0..2 runtime domain rejects");

    Fixture bad_size;
    bad_size.table.resources[0].size -= Fixture::kStride;
    CHECK(!analyze(exact, bad_size),
          "source size must equal exact-thread-count times record stride");

    Fixture bad_alias;
    bad_alias.table.resources[1].gpu_addr += Fixture::kStride;
    CHECK(!analyze(exact, bad_alias),
          "pointer and selector loads must resolve to one logical record source");

    Fixture split_snapshot;
    std::vector<uint8_t> selector_shadow = split_snapshot.source;
    store_u32(selector_shadow, 2u * Fixture::kStride + Fixture::kSelectorOffset, 0u);
    split_snapshot.table.resources[1].host_data = selector_shadow.data();
    split_snapshot.table.resources[1].host_data_size = selector_shadow.size();
    CHECK(!analyze(exact, split_snapshot),
          "pointer and selector loads require byte-identical complete source snapshots");

    Fixture bad_launch;
    bad_launch.config.user_sgprs.pop_back();
    CHECK(!analyze(exact, bad_launch),
          "record-index proof requires TGID X immediately after five user SGPRs");
    bad_launch = Fixture{};
    bad_launch.config.tgid_x_en = false;
    CHECK(!analyze(exact, bad_launch),
          "record-index proof requires the TGID-X system input");
    bad_launch = Fixture{};
    bad_launch.config.exact_thread_extent = false;
    CHECK(!analyze(exact, bad_launch),
          "partial final workgroup requires an exact-thread host guard");

    LiveFixture live;
    const std::vector<uint8_t> original_source = live.source;
    CHECK(discover_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, live.table),
          "live dispatch materializes its generic relocation carrier");
    const ShaderResource* relocated = live.table.by_fetch_pc(6u);
    CHECK(relocated && is_indirect_pointer_relocation_resource(*relocated) &&
              relocated->host_data != live.source.data() &&
              relocated->host_data_size > relocated->size &&
              live.table.owned_host_data.size() == 1u,
          "pc6 owns one distinct proof-bearing carrier while keeping its logical size");
    CHECK(live.source == original_source && relocated &&
              std::memcmp(relocated->host_data, original_source.data(),
                          original_source.size()) == 0,
          "discovery preserves both guest source bytes and the carrier's byte-exact prefix");
    IndirectPointerRelocationProof live_proof;
    IndirectBufferRelocationInfo live_info;
    CHECK(validate_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, live.table,
              &live_proof, &live_info) &&
              live_proof.records.size() == live.threads &&
              live_info.records == live_proof.records && !live_info.segments.empty(),
          "final validation re-establishes shader, source, record, and payload authority");

    ShaderResourceTable copied = live.table;
    live.table.owned_host_data.clear();
    CHECK(validate_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, copied),
          "resource-table copies retain carrier lifetime and validation authority");
    assign_convention_bindings(copied, 2u);
    const std::vector<uint32_t> spirv = recompile_compute(
        exact.data(), exact.size(), &copied, live.config,
        {RecompileDiagnosticStage::Compute, 0u});
    CHECK(!spirv.empty(),
          "production emitter lowers every proof-authorized indirect consumer");
    const DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, &copied, 0u, SpirvShaderStage::Compute, false);
    CHECK(report.ok(), "relocated module and runtime descriptor table agree");
    const ShaderResource* compiled_source = copied.by_fetch_pc(6u);
    const SpirvDescriptorBinding* compiled_descriptor = compiled_source
        ? find_spirv_descriptor_binding(report, 0u, compiled_source->binding) : nullptr;
    const StorageBufferMaterializationPlan compiled_plan =
        compiled_source && compiled_descriptor
            ? plan_storage_buffer_materialization(*compiled_descriptor, *compiled_source)
            : StorageBufferMaterializationPlan{};
    CHECK(compiled_descriptor && compiled_descriptor->readable &&
              !compiled_descriptor->writable && compiled_plan.valid &&
              compiled_plan.binding_bytes == compiled_source->host_data_size,
          "Vulkan materialization binds the complete read-only relocation carrier");

    ShaderResourceTable replay = copied;
    replay.owned_host_data.clear();
    ShaderResource* replay_source = nullptr;
    for (ShaderResource& resource : replay.resources)
        if (resource.fetch_pc == 6u) replay_source = &resource;
    if (replay_source) replay_source->indirect_pointer_relocation = {};
    CHECK(discover_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, replay) &&
              validate_rdna2_indirect_pointer_relocations(
                  exact.data(), exact.size(), live.config, replay),
          "replay-owned serialized carriers reconstruct derived proof markers");

    const uint32_t saved_selector = [&] {
        uint32_t value = 0;
        std::memcpy(&value, live.source.data() + LiveFixture::kSelectorOffset,
                    sizeof(value));
        return value;
    }();
    store_u32(live.source, LiveFixture::kSelectorOffset, 2u);
    CHECK(!validate_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, copied),
          "changed live selector invalidates a previously built carrier");
    store_u32(live.source, LiveFixture::kSelectorOffset, saved_selector);

    live.pointees[0][24u] ^= 1u;
    CHECK(!validate_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, copied),
          "changed active pointee invalidates the exact payload snapshot");
    live.pointees[0][24u] ^= 1u;
    live.pointees[1][0u] ^= 1u;
    CHECK(validate_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), live.config, copied),
          "inactive nonzero pointer remains only a source witness, not a copied dependency");

    ComputeItem captured_compute;
    captured_compute.spirv = spirv;
    captured_compute.user_sgprs = live.config.user_sgprs;
    captured_compute.resources = std::make_shared<ShaderResourceTable>(copied);
    captured_compute.launch.threads_x = live.config.threads_x;
    captured_compute.launch.threads_y = captured_compute.launch.threads_z = 1u;
    captured_compute.launch.local_x = live.config.local_x;
    captured_compute.launch.local_y = captured_compute.launch.local_z = 1u;
    captured_compute.launch.groups_x =
        (live.config.threads_x + live.config.local_x - 1u) / live.config.local_x;
    captured_compute.launch.groups_y = captured_compute.launch.groups_z = 1u;
    captured_compute.code_addr = reinterpret_cast<uint64_t>(exact.data());
    captured_compute.dispatch_index = 0u;
    captured_compute.command_order = 1u;
    captured_compute.recompile_config = live.config;
    captured_compute.recompile_config_available = true;
    const auto capture_reader = [&](uint64_t address, uint8_t* destination,
                                    size_t bytes) -> size_t {
        const auto read = [&](uint64_t base, const uint8_t* source,
                              size_t available) -> size_t {
            if (address < base || address >= base + available) return 0u;
            const size_t offset = static_cast<size_t>(address - base);
            const size_t count = std::min(bytes, available - offset);
            std::memcpy(destination, source + offset, count);
            return count;
        };
        if (const size_t count = read(
                reinterpret_cast<uint64_t>(exact.data()),
                reinterpret_cast<const uint8_t*>(exact.data()),
                exact.size() * sizeof(uint32_t)))
            return count;
        if (const size_t count = read(reinterpret_cast<uint64_t>(live.source.data()),
                                      live.source.data(), live.source.size()))
            return count;
        if (const size_t count = read(reinterpret_cast<uint64_t>(live.scalar.data()),
                                      live.scalar.data(), live.scalar.size()))
            return count;
        return read(reinterpret_cast<uint64_t>(live.output.data()),
                    live.output.data(), live.output.size());
    };
    GpuCaptureFile captured;
    GpuCaptureMetadata metadata;
    std::string capture_error;
    CHECK(capture_submit_items(
              {}, {captured_compute},
              {{SubmitOperationKind::Dispatch, 0u, 1u}}, metadata,
              capture_reader, captured, capture_error) &&
              captured.format_version == 55u && captured.computes.size() == 1u &&
              captured.computes[0].resources.resources[0].internal_bytes.size() ==
                  compiled_source->host_data_size,
          "capture stores the dispatch-owned relocation snapshot as v53 internal bytes");
    std::vector<uint8_t> capture_bytes;
    GpuCaptureFile loaded_capture;
    GpuReplayFrame replay_frame;
    const bool replay_materialized =
        serialize_gpu_capture(captured, capture_bytes, capture_error) &&
        deserialize_gpu_capture(capture_bytes, loaded_capture, capture_error) &&
        materialize_gpu_replay(loaded_capture, replay_frame, capture_error);
    const ShaderResource* replay_relocated =
        replay_materialized && replay_frame.computes.size() == 1u &&
                replay_frame.computes[0].resources
            ? replay_frame.computes[0].resources->by_fetch_pc(6u) : nullptr;
    CHECK(replay_relocated && is_indirect_pointer_relocation_resource(*replay_relocated),
          "capture round-trip re-derives generic relocation authority from raw state");

    GpuDependencyGraph dependency_graph;
    const bool graph_built = replay_materialized &&
        build_gpu_dependency_graph(replay_frame, dependency_graph, capture_error);
    const uint64_t first_pointee =
        reinterpret_cast<uint64_t>(live.pointees[0].data());
    CHECK(graph_built && std::any_of(
              dependency_graph.external_leaves.begin(),
              dependency_graph.external_leaves.end(),
              [&](const GpuDependencyLeaf& leaf) {
                  return leaf.access.addr == first_pointee &&
                         leaf.access.size == LiveFixture::kPointeeBytes;
              }),
          "dependency closure retains the exact guest pointee footprint as an external read");
    CHECK(graph_built && replay_relocated && std::none_of(
              dependency_graph.external_leaves.begin(),
              dependency_graph.external_leaves.end(),
              [&](const GpuDependencyLeaf& leaf) {
                  return leaf.access.addr == replay_relocated->gpu_addr &&
                         leaf.access.size == replay_relocated->host_data_size;
              }),
          "dependency closure never treats appended carrier bytes as adjacent guest memory");

    GpuCaptureFile malformed_capture = loaded_capture;
    malformed_capture.computes[0].resources.resources[0].internal_bytes[
        live.source.size()] ^= 1u;
    CHECK(!materialize_gpu_replay(malformed_capture, replay_frame, capture_error),
          "replay rejects a relocation carrier whose v2 header tag was corrupted");

    return failures ? 1 : 0;
}
