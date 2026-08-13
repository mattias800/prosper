#include "gpu/gpu_capture.hpp"
#include "gpu/rdna2_decode.hpp"
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

std::vector<uint32_t> program(const char* fixture, size_t expected_dwords) {
    const std::filesystem::path path = std::filesystem::path(__FILE__).parent_path() /
        "data" / fixture;
    std::ifstream input(path);
    std::string hex((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    CHECK(input.good() || input.eof(), "descriptor-range program fixture is readable");
    CHECK(hex.size() == expected_dwords * 8u,
          "descriptor-range program fixture has its retained dword count");
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

struct Shape {
    const char* fixture;
    uint32_t dwords;
    uint32_t user_sgprs;
    uint32_t source_pc;
    uint32_t source_vgpr;
    uint32_t first_access_pc;
    uint32_t last_access_pc;
    uint32_t wait_pc;
    uint32_t full_descriptor_pc;
};

constexpr Shape kShapes[]{
    {"rdna2_indirect_pointer_descriptor_range_642.hex", 642u, 5u,
     64u, 27u, 87u, 528u, 93u, 588u},
    {"rdna2_indirect_pointer_descriptor_range_662.hex", 662u, 6u,
     66u, 18u, 89u, 560u, 95u, 608u},
};

struct Fixture {
    static constexpr uint32_t kOuterBytes = 2992u;
    static constexpr uint32_t kMainBytes = 1310720u;
    static constexpr uint32_t kPointerBytes = 2097152u;
    static constexpr uint32_t kMainStride = 80u;
    static constexpr uint32_t kPointerStride = 16u;
    static constexpr uint32_t kFirstPointeeBytes = 64u;
    static constexpr uint32_t kSecondPointeeBytes = 96u;

    const Shape& shape;
    std::vector<uint8_t> outer = std::vector<uint8_t>(kOuterBytes);
    std::vector<uint8_t> main = std::vector<uint8_t>(kMainBytes);
    std::vector<uint8_t> pointers = std::vector<uint8_t>(kPointerBytes);
    // Keep backing storage beyond the initially declared ranges so boundary tests can grow a V#
    // without ever making a host-side out-of-object read part of the test oracle.
    std::array<uint8_t, 512> first_pointee{};
    std::array<uint8_t, 512> second_pointee{};
    ShaderResourceTable table;
    ComputeShaderConfig config;

    explicit Fixture(const Shape& selected) : shape(selected) {
        for (size_t byte = 0; byte < first_pointee.size(); ++byte)
            first_pointee[byte] = static_cast<uint8_t>(byte * 3u + 1u);
        for (size_t byte = 0; byte < second_pointee.size(); ++byte)
            second_pointee[byte] = static_cast<uint8_t>(byte * 5u + 7u);
        store_u32(outer, 20u, 5u);
        store_u32(outer, 24u, 3u);
        main_record(5u, 0u, 32u, 7u);
        main_record(6u, second_first_thread(), 32u, 11u);
        // The 662 sibling advances its logical group id by 64 while dispatching 32 physical lanes:
        // [32,64) is never emitted. Both shapes leave this null V# record unreachable.
        main_record(7u, shape.dwords == 662u ? 32u : 128u, 32u, 13u);
        descriptor(7u, reinterpret_cast<uint64_t>(first_pointee.data()),
                   kFirstPointeeBytes / 16u, 16u);
        descriptor(11u, reinterpret_cast<uint64_t>(second_pointee.data()),
                   kSecondPointeeBytes / 16u, 16u);

        add_resource(9u, outer, 16u);
        add_resource(28u, main, kMainStride);
        add_resource(51u, main, kMainStride);
        // The live front half emits exact per-fetch aliases even when several consumers share one
        // main-table V#. Preserve that provenance in the integration fixture so compilation reaches
        // the relocated GLOBAL consumers instead of failing earlier at an unrelated raw MUBUF.
        if (shape.dwords == 642u) {
            for (uint32_t fetch_pc : std::array<uint32_t, 9>{
                    53u, 55u, 118u, 248u, 306u, 317u, 420u, 549u, 551u})
                add_resource(fetch_pc, main, kMainStride);
        } else {
            for (uint32_t fetch_pc : std::array<uint32_t, 12>{
                    53u, 55u, 57u, 120u, 122u, 253u, 309u, 311u,
                    533u, 535u, 537u, 539u})
                add_resource(fetch_pc, main, kMainStride);
        }
        add_resource(shape.source_pc, pointers, kPointerStride);

        config.user_sgprs.resize(shape.user_sgprs);
        config.local_x = 32u;
        config.local_y = config.local_z = 1u;
        config.exact_thread_extent = true;
        config.threads_x = 64u;
        config.threads_y = config.threads_z = 1u;
        config.wave_size = 32u;
        config.tgid_x_en = true;
        config.tidig_comp_cnt = 0u;
    }

    uint32_t second_first_thread() const {
        return shape.dwords == 662u ? 64u : 32u;
    }

    void main_record(uint32_t index, uint32_t first, uint32_t count,
                     uint32_t pointer_index) {
        const size_t offset = static_cast<size_t>(index) * kMainStride;
        store_u32(main, offset + 36u, count);
        store_u32(main, offset + 40u, first);
        store_u32(main, offset + 48u, pointer_index);
    }

    void main_u32(uint32_t index, uint32_t field_offset, uint32_t value) {
        store_u32(main, static_cast<size_t>(index) * kMainStride + field_offset, value);
    }

    void main_byte(uint32_t index, uint32_t field_offset, uint8_t value) {
        main[static_cast<size_t>(index) * kMainStride + field_offset] = value;
    }

    void descriptor(uint32_t index, uint64_t address, uint32_t bytes,
                    uint32_t stride = 0u) {
        const size_t offset = static_cast<size_t>(index) * kPointerStride;
        store_u32(pointers, offset, static_cast<uint32_t>(address));
        store_u32(pointers, offset + 4u,
                  static_cast<uint32_t>((address >> 32u) & 0xffffu) |
                      ((stride & 0x3fffu) << 16u));
        store_u32(pointers, offset + 8u, bytes);
        store_u32(pointers, offset + 12u, 0x00004facu);
    }

    void add_resource(uint32_t pc, std::vector<uint8_t>& bytes, uint32_t stride) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1u;
        resource.gpu_addr = reinterpret_cast<uint64_t>(bytes.data());
        resource.size = static_cast<uint32_t>(bytes.size());
        resource.stride = stride;
        resource.fetch_pc = pc;
        resource.host_data = bytes.data();
        resource.host_data_size = bytes.size();
        table.resources.push_back(resource);
    }

    void add_empty_resource(uint32_t pc) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Unknown;
        resource.num_components = 0u;
        resource.fetch_pc = pc;
        table.resources.push_back(resource);
    }
};

bool analyze(const std::vector<uint32_t>& words, const Fixture& fixture,
             IndirectPointerRelocationProof* output = nullptr) {
    IndirectPointerRelocationProof proof;
    const bool accepted = analyze_rdna2_descriptor_pointer_range(
        words.data(), words.size(), fixture.config, fixture.table, proof);
    if (output) *output = std::move(proof);
    return accepted;
}

const Rdna2Inst* instruction_at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    for (const Rdna2Inst& instruction : instructions)
        if (instruction.pc == pc) return &instruction;
    return nullptr;
}

GpuCapturedResource* captured_resource_at(GpuCaptureFile& capture, uint32_t fetch_pc) {
    if (capture.computes.size() != 1u) return nullptr;
    GpuCapturedResource* result = nullptr;
    for (GpuCapturedResource& resource : capture.computes[0].resources.resources) {
        if (resource.resource.fetch_pc != fetch_pc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

bool spirv_declares_capability(const std::vector<uint32_t>& spirv, uint32_t capability) {
    if (spirv.size() < 5u) return false;
    for (size_t cursor = 5u; cursor < spirv.size();) {
        const uint32_t word_count = spirv[cursor] >> 16u;
        const uint32_t opcode = spirv[cursor] & 0xffffu;
        if (!word_count || cursor + word_count > spirv.size()) return false;
        if (opcode == 17u && word_count == 2u && spirv[cursor + 1u] == capability)
            return true;
        cursor += word_count;
    }
    return false;
}

void test_emitter_integration(const Shape& shape, const std::vector<uint32_t>& exact,
                              const Fixture& fixture) {
    ShaderResourceTable compile_table = fixture.table;
    assign_convention_bindings(compile_table, 2u);
    const std::vector<uint32_t> spirv = recompile_compute(
        exact.data(), exact.size(), &compile_table, fixture.config,
        {RecompileDiagnosticStage::Compute, 0u});
    CHECK(!spirv.empty(),
          "production emitter lowers each exact DescriptorRange sibling");
    if (spirv.empty()) return;

    const DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, &compile_table, 0u, SpirvShaderStage::Compute, false);
    CHECK(report.ok(),
          "DescriptorRange SPIR-V and runtime descriptor interface agree");
    const ShaderResource* source = compile_table.by_fetch_pc(shape.source_pc);
    const SpirvDescriptorBinding* descriptor = source
        ? find_spirv_descriptor_binding(report, 0u, source->binding) : nullptr;
    const StorageBufferMaterializationPlan plan = source && descriptor
        ? plan_storage_buffer_materialization(*descriptor, *source)
        : StorageBufferMaterializationPlan{};
    CHECK(descriptor && descriptor->readable && !descriptor->writable && plan.valid &&
              plan.logical_bytes == source->host_data_size &&
              plan.binding_bytes == source->host_data_size &&
              plan.binding_bytes > source->size,
          "DescriptorRange emitter materializes the complete read-only v3 carrier");
    // SPIR-V Capability Int64 is enumerant 11. Address selection is deliberately implemented with
    // bounded u32 pairs so this path remains usable on devices without shaderInt64.
    CHECK(!spirv_declares_capability(spirv, 11u),
          "DescriptorRange lowering does not require ShaderInt64");
}

bool rebase_descriptor_carrier_for_foreign_replay(
        GpuCaptureFile& capture, uint32_t source_pc) {
    GpuCapturedResource* captured = captured_resource_at(capture, source_pc);
    if (!captured) return false;
    ShaderResource& source = captured->resource;
    IndirectBufferRelocationInfo info;
    if (!inspect_indirect_buffer_relocation(
            source, captured->internal_bytes.data(), captured->internal_bytes.size(),
            kIndirectPointerDescriptorRangeLayout, info) ||
        info.records.size() != info.segments.size())
        return false;

    const size_t records_base = static_cast<size_t>(source.size) +
        kIndirectBufferRelocationHeaderBytes;
    const size_t segments_base = records_base +
        info.records.size() * kIndirectBufferRelocationRecordBytes;
    constexpr uint64_t kForeignPayloadBase = UINT64_C(0x0000123400000000);
    for (size_t index = 0; index < info.records.size(); ++index) {
        const IndirectBufferRelocationRecord& record = info.records[index];
        const IndirectBufferRelocationSegment& segment = info.segments[index];
        if (record.guest_address != segment.guest_address ||
            record.byte_count != segment.byte_count)
            return false;
        const uint64_t address = kForeignPayloadBase + index * UINT64_C(0x10000);
        store_u32(captured->internal_bytes, record.source_byte_offset,
                  static_cast<uint32_t>(address));
        uint32_t word1 = 0;
        std::memcpy(&word1,
                    captured->internal_bytes.data() + record.source_byte_offset + 4u,
                    sizeof(word1));
        word1 = (word1 & 0xffff0000u) |
            static_cast<uint32_t>((address >> 32u) & 0xffffu);
        store_u32(captured->internal_bytes, record.source_byte_offset + 4u, word1);
        store_u64(captured->internal_bytes,
                  records_base + index * kIndirectBufferRelocationRecordBytes + 8u,
                  address);
        store_u64(captured->internal_bytes,
                  segments_base + index * kIndirectBufferRelocationSegmentBytes,
                  address);
    }
    // In a different replay process neither this logical source address nor the guest addresses
    // encoded above are host pointers. The carrier is the only permitted source of their bytes.
    source.gpu_addr = UINT64_C(0x0000123500000000);
    return inspect_indirect_buffer_relocation(
        source, captured->internal_bytes.data(), captured->internal_bytes.size(),
        kIndirectPointerDescriptorRangeLayout, info);
}

void test_capture_roundtrip(const Shape& shape, const std::vector<uint32_t>& exact,
                            Fixture& fixture,
                            const IndirectBufferRelocationInfo& live_info) {
    ShaderResourceTable capture_table = fixture.table;
    assign_convention_bindings(capture_table, 2u);
    const std::vector<uint32_t> spirv = recompile_compute(
        exact.data(), exact.size(), &capture_table, fixture.config,
        {RecompileDiagnosticStage::Compute, 0u});
    CHECK(!spirv.empty(),
          "production emitter lowers the DescriptorRange shader before capture");
    ComputeItem compute;
    compute.spirv = spirv;
    compute.resources = std::make_shared<ShaderResourceTable>(capture_table);
    compute.launch.threads_x = fixture.config.threads_x;
    compute.launch.threads_y = compute.launch.threads_z = 1u;
    compute.launch.local_x = fixture.config.local_x;
    compute.launch.local_y = compute.launch.local_z = 1u;
    compute.launch.groups_x =
        (fixture.config.threads_x + fixture.config.local_x - 1u) /
        fixture.config.local_x;
    compute.launch.groups_y = compute.launch.groups_z = 1u;
    compute.code_addr = reinterpret_cast<uint64_t>(exact.data());
    compute.dispatch_index = 0u;
    compute.command_order = 1u;
    compute.recompile_config = fixture.config;
    compute.recompile_config_available = true;

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
        if (const size_t count = read(
                reinterpret_cast<uint64_t>(fixture.outer.data()),
                fixture.outer.data(), fixture.outer.size()))
            return count;
        return read(reinterpret_cast<uint64_t>(fixture.main.data()),
                    fixture.main.data(), fixture.main.size());
    };

    GpuCaptureFile captured;
    GpuCaptureMetadata metadata;
    std::string error;
    CHECK(capture_submit_items(
              {}, {compute}, {{SubmitOperationKind::Dispatch, 0u, 1u}},
              metadata, capture_reader, captured, error) &&
              captured.format_version == 53u,
          "v53 capture stores a DescriptorRange dispatch");
    GpuCapturedResource* captured_source = captured_resource_at(captured, shape.source_pc);
    CHECK(captured_source &&
              captured_source->internal_bytes.size() ==
                  capture_table.by_fetch_pc(shape.source_pc)->host_data_size &&
              captured_source->blob_index == UINT32_MAX,
          "capture owns the complete v3 carrier instead of its guest pointee intervals");

    std::vector<uint8_t> serialized;
    GpuCaptureFile loaded;
    GpuReplayFrame replay;
    const bool loaded_and_rebased = serialize_gpu_capture(captured, serialized, error) &&
        deserialize_gpu_capture(serialized, loaded, error) &&
        rebase_descriptor_carrier_for_foreign_replay(loaded, shape.source_pc);
    const bool replayed = loaded_and_rebased &&
        materialize_gpu_replay(loaded, replay, error);
    const ShaderResource* replay_source = replayed && replay.computes.size() == 1u &&
            replay.computes[0].resources
        ? replay.computes[0].resources->by_fetch_pc(shape.source_pc) : nullptr;
    CHECK(replay_source && is_indirect_pointer_relocation_resource(*replay_source),
          "v53 serialize/deserialize/replay re-derives DescriptorRange authority");

    GpuCaptureFile q_mutation = loaded;
    GpuCapturedResource* q_source = captured_resource_at(q_mutation, shape.source_pc);
    if (q_source && !live_info.records.empty() && !live_info.segments.empty()) {
        const IndirectBufferRelocationRecord& record = live_info.records.front();
        const auto segment = std::find_if(
            live_info.segments.begin(), live_info.segments.end(),
            [&](const IndirectBufferRelocationSegment& candidate) {
                return candidate.guest_address <= record.guest_address &&
                    record.guest_address < candidate.guest_address + candidate.byte_count;
            });
        CHECK(segment != live_info.segments.end(),
              "first q-producing access has an exact packed segment");
        if (segment != live_info.segments.end()) {
            const size_t q2_byte = static_cast<size_t>(segment->packed_byte_offset) +
                static_cast<size_t>(record.guest_address - segment->guest_address) + 2u;
            CHECK(q2_byte < q_source->internal_bytes.size(),
                  "q2 mutation remains inside the packed payload");
            if (q2_byte < q_source->internal_bytes.size())
                q_source->internal_bytes[q2_byte] = 255u;
        }
    }
    CHECK(!materialize_gpu_replay(q_mutation, replay, error),
          "same-site q2 payload mutation cannot retain stale residual authority");

    GpuCaptureFile malformed = loaded;
    GpuCapturedResource* malformed_source = captured_resource_at(malformed, shape.source_pc);
    if (malformed_source &&
        malformed_source->resource.size < malformed_source->internal_bytes.size())
        malformed_source->internal_bytes[malformed_source->resource.size] ^= 1u;
    CHECK(!materialize_gpu_replay(malformed, replay, error),
          "replay rejects a malformed DescriptorRange v3 carrier");

    // The marker itself is intentionally not serialized: replay reconstructs it. Exercise missing
    // carrier rejection on the in-memory capture where the marker still identifies the required
    // derived state.
    GpuCaptureFile missing = captured;
    GpuCapturedResource* missing_source = captured_resource_at(missing, shape.source_pc);
    if (missing_source) missing_source->internal_bytes.clear();
    CHECK(!materialize_gpu_replay(missing, replay, error),
          "replay rejects a DescriptorRange dispatch with its carrier removed");
}

void test_shape(const Shape& shape) {
    const std::vector<uint32_t> exact = program(shape.fixture, shape.dwords);
    Fixture fixture(shape);
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(exact.data(), exact.size(), decoded);
    for (const Rdna2Inst& instruction : decoded) {
        if (instruction.fmt != Rdna2Format::MUBUF ||
            fixture.table.by_fetch_pc(instruction.pc))
            continue;
        fixture.add_empty_resource(instruction.pc);
    }
    IndirectPointerRelocationProof proof;
    CHECK(analyze(exact, fixture, &proof),
          "exact Wave32 descriptor-range sibling is admitted");
    CHECK(proof.schema_version == kIndirectPointerProofSchema &&
              proof.bound_kind == IndirectPointerBoundKind::DescriptorRange &&
              proof.guard_kind == IndirectPointerGuardKind::None &&
              proof.source_fetch_pc == shape.source_pc &&
              proof.source_result_vgpr == shape.source_vgpr &&
              proof.source_stride == Fixture::kPointerStride &&
              proof.pointer_byte_offset == 0u &&
              proof.source_record_index_vgpr == 2u &&
              proof.source_address_kind ==
                  IndirectBufferRelocationRecord::SourceAddressKind::BufferDescriptorBase48,
          "proof retains exact V# producer and dynamic source-record identity");
    CHECK(proof.records.size() == 2u && proof.record_count == 2u &&
              proof.records[0].source_byte_offset == 7u * Fixture::kPointerStride &&
              proof.records[1].source_byte_offset == 11u * Fixture::kPointerStride &&
              proof.records[0].guest_address ==
                  reinterpret_cast<uint64_t>(fixture.first_pointee.data()) &&
              proof.records[0].byte_count == Fixture::kFirstPointeeBytes &&
              proof.records[1].guest_address ==
                  reinterpret_cast<uint64_t>(fixture.second_pointee.data()) &&
              proof.records[1].byte_count == Fixture::kSecondPointeeBytes &&
              proof.max_footprint_bytes == Fixture::kSecondPointeeBytes,
          "reachable main records enumerate, sort, and bound only exact V# table entries");
    const size_t expected_accesses = shape.dwords == 642u ? 19u : 20u;
    CHECK(proof.accesses.size() == expected_accesses &&
              proof.accesses.front().pc == shape.first_access_pc &&
              proof.accesses.back().pc == shape.last_access_pc &&
              std::all_of(proof.accesses.begin(), proof.accesses.end(),
                  [](const IndirectPointerAccessProof& access) {
                      return access.immediate_byte_offset == 0u &&
                             access.component_bytes == 4u && access.components == 1u;
                  }),
          "proof enumerates every exact unmodified GLOBAL_LOAD_DWORD consumer");
    CHECK(proof.fingerprint != 0u && proof.witness_words.size() == 4u &&
              proof.witness_words[3] ==
                  (kIndirectPointerDescriptorRangeTag ^ proof.witness_words[0] ^
                   proof.witness_words[1] ^ proof.witness_words[2]),
          "v3 descriptor-range proof has a self-checking carrier witness");

    CHECK(discover_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), fixture.config, fixture.table),
          "generic discovery materializes a DescriptorRange v3 carrier");
    const ShaderResource* relocated = fixture.table.by_fetch_pc(shape.source_pc);
    CHECK(relocated && is_indirect_pointer_relocation_resource(*relocated) &&
              relocated->indirect_pointer_relocation.carrier_version ==
                  kIndirectPointerDescriptorRangeLayout.version &&
              relocated->indirect_pointer_relocation.record_count == 2u &&
              relocated->host_data_size > relocated->size &&
              fixture.table.owned_host_data.size() == 1u,
          "DescriptorRange marker retains v3 carrier layout and ownership");
    IndirectPointerRelocationProof validated_proof;
    IndirectBufferRelocationInfo validated_info;
    CHECK(validate_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), fixture.config, fixture.table,
              &validated_proof, &validated_info) &&
              validated_proof.bound_kind == IndirectPointerBoundKind::DescriptorRange &&
              validated_info.records == proof.records &&
              !validated_info.segments.empty(),
          "generic validation re-establishes the v3 shader/data proof");

    test_emitter_integration(shape, exact, fixture);

    if (shape.dwords == 642u)
        test_capture_roundtrip(shape, exact, fixture, validated_info);

    if (relocated) {
        std::vector<uint8_t> serialized(relocated->host_data,
                                       relocated->host_data + relocated->host_data_size);
        ShaderResource syntax = *relocated;
        syntax.indirect_pointer_relocation = {};
        CHECK(is_indirect_pointer_relocation_serialized(
                  syntax, serialized.data(), serialized.size()),
              "syntax classifier recognizes a marker-free v3 carrier");

        const uint32_t source_offset = 7u * Fixture::kPointerStride;
        fixture.pointers[source_offset] ^= 1u;
        CHECK(!validate_rdna2_indirect_pointer_relocations(
                  exact.data(), exact.size(), fixture.config, fixture.table),
              "changed live descriptor Base48 invalidates the v3 carrier");
        fixture.pointers[source_offset] ^= 1u;

        fixture.first_pointee[0] ^= 1u;
        CHECK(!validate_rdna2_indirect_pointer_relocations(
                  exact.data(), exact.size(), fixture.config, fixture.table),
              "changed live DescriptorRange payload invalidates the v3 carrier");
        fixture.first_pointee[0] ^= 1u;

        const size_t records_base = Fixture::kPointerBytes +
            kIndirectBufferRelocationHeaderBytes;
        serialized.at(records_base + 20u) = static_cast<uint8_t>(
            IndirectBufferRelocationRecord::SourceAddressKind::RawU64);
        CHECK(!is_indirect_pointer_relocation_serialized(
                  syntax, serialized.data(), serialized.size()),
              "same-record v3 source-kind mutation fails syntax validation");
    }

    const Rdna2Inst* source = instruction_at(decoded, shape.source_pc);
    CHECK(source && rdna2_indirect_pointer_source(proof, *source),
          "source helper recognizes the exact V# producer packet");
    if (source) {
        Rdna2Inst changed = *source;
        changed.words[1] ^= 1u;
        CHECK(!rdna2_indirect_pointer_source(proof, changed),
              "source PC alone cannot grant record identity after packet mutation");
    }

    for (const IndirectPointerAccessProof& access : proof.accesses) {
        std::vector<uint32_t> changed = exact;
        changed[access.pc] ^= 1u;
        CHECK(!analyze(changed, fixture),
              "same-site GLOBAL consumer mutation invalidates DescriptorRange authority");
    }
    struct Mutation { uint32_t pc; uint32_t word; };
    std::vector<Mutation> mutations{{
        {1u, 0u}, {5u, 0u}, {9u, 0u}, {19u, 0u}, {25u, 0u},
        {28u, 1u}, {36u, 0u}, {42u, 0u},
        {32u, 0u}, {34u, 0u}, {51u, 1u}, {53u, 1u},
        {shape.dwords == 642u ? 59u : 61u, 0u},
        {shape.source_pc, 0u}, {shape.source_pc, 1u},
        {shape.source_pc + 17u, 0u}, {shape.source_pc + 19u, 1u},
        {shape.source_pc + 21u, 0u}, {shape.source_pc + 25u, 1u},
        {shape.full_descriptor_pc, 1u},
    }};
    if (shape.dwords == 642u) {
        mutations.insert(mutations.end(), {
            {67u, 0u}, {70u, 1u}, {76u, 0u}, {95u, 0u}, {107u, 1u},
            {116u, 0u}, {117u, 0u}, {137u, 1u}, {245u, 0u}, {246u, 0u},
            {511u, 1u}, {513u, 0u},
        });
    } else {
        mutations.insert(mutations.end(), {
            {69u, 0u}, {72u, 1u}, {78u, 0u}, {177u, 1u},
            {489u, 1u}, {491u, 0u},
        });
    }
    for (const Mutation& mutation : mutations) {
        std::vector<uint32_t> changed = exact;
        changed[mutation.pc + mutation.word] ^= 1u;
        CHECK(!analyze(changed, fixture),
              "same-site producer/control/lineage mutation invalidates the proof");
    }
    std::vector<uint32_t> index_clobber = exact;
    index_clobber[shape.dwords == 642u ? 59u : 61u] = 0x7e040280u;
    CHECK(!analyze(index_clobber, fixture),
          "same-length v2 clobber between pc51 and the V# fetch loses pointer identity");
    std::vector<uint32_t> pointer_descriptor_clobber = exact;
    pointer_descriptor_clobber[shape.dwords == 642u ? 59u : 61u] = 0xbe880380u;
    CHECK(!analyze(pointer_descriptor_clobber, fixture),
          "same-length s8 clobber between pointer-descriptor load and V# fetch rejects");
    std::vector<uint32_t> main_offset_clobber = exact;
    main_offset_clobber[shape.dwords == 642u ? 59u : 61u] =
        shape.dwords == 642u ? 0x7e3e0280u : 0x7e3c0280u;
    CHECK(!analyze(main_offset_clobber, fixture),
          "same-length selected-main offset clobber before later modeled fields rejects");
    if (shape.dwords == 642u) {
        std::vector<uint32_t> main68_producer = exact;
        main68_producer[118u] ^= 1u;
        CHECK(!analyze(main68_producer, fixture),
              "same-site main+68 producer mutation loses residual/gate authority");
    } else {
        std::vector<uint32_t> saved_exec_clobber = exact;
        saved_exec_clobber[414u] = 0xbe860380u;
        CHECK(!analyze(saved_exec_clobber, fixture),
              "same-length s6 clobber before the pc526 restore loses final-region authority");
    }
    std::vector<uint32_t> unexpected_exec = exact;
    unexpected_exec[45u] = 0xbeeb3c00u;
    CHECK(!analyze(unexpected_exec, fixture),
          "same-length saveexec at the pc45 wait site loses exact EXEC authority");

    std::vector<uint32_t> unrelated = exact;
    unrelated[shape.wait_pc] ^= 1u;
    IndirectPointerRelocationProof unrelated_proof;
    CHECK(analyze(unrelated, fixture, &unrelated_proof) &&
              unrelated_proof.fingerprint == proof.fingerprint,
          "analyzer is structural rather than complete-program byte identity");

    std::vector<uint32_t> too_short(exact.begin(), exact.end() - 1u);
    CHECK(!analyze(too_short, fixture),
          "a registered span shorter than the exact executable prefix is rejected");
    std::vector<uint32_t> early_end = exact;
    early_end[shape.wait_pc] = 0xbf810000u;
    CHECK(!analyze(early_end, fixture),
          "an alternate END before the exact prefix boundary is executable, not trailing data");

    std::vector<uint32_t> registered_span = exact;
    const size_t registered_dwords = shape.dwords == 642u ? 752u : 764u;
    registered_span.resize(registered_dwords);
    for (size_t pc = exact.size(); pc < registered_span.size(); ++pc)
        registered_span[pc] = (pc & 1u) ? 0xbf820000u : 0xdeadbeefu;
    Fixture registered_fixture(shape);
    IndirectPointerRelocationProof registered_proof;
    CHECK(analyze(registered_span, registered_fixture, &registered_proof) &&
              registered_proof.fingerprint == proof.fingerprint,
          "unreachable AGC trailing storage preserves the executable-prefix proof");
    CHECK(discover_rdna2_indirect_pointer_relocations(
              registered_span.data(), registered_span.size(),
              registered_fixture.config, registered_fixture.table) &&
              validate_rdna2_indirect_pointer_relocations(
                  registered_span.data(), registered_span.size(),
                  registered_fixture.config, registered_fixture.table),
          "discovery and final validation retain the full registered dword span");

    Fixture duplicate(shape);
    duplicate.main_record(6u, duplicate.second_first_thread(), 32u, 7u);
    IndirectPointerRelocationProof duplicate_proof;
    CHECK(analyze(exact, duplicate, &duplicate_proof) &&
              duplicate_proof.records.size() == 1u,
          "multiple reachable main records deduplicate one V# source record");

    Fixture reversed_fields(shape);
    reversed_fields.main_u32(5u, 36u, 0u);
    reversed_fields.main_u32(5u, 40u, 32u);
    reversed_fields.main_u32(6u, 36u, 32u);
    reversed_fields.main_u32(6u, 40u, 96u);
    CHECK(!analyze(exact, reversed_fields),
          "+36=count and +40=first cannot be reinterpreted in the old reversed order");

    Fixture overlap(shape);
    overlap.main_record(5u, 0u, shape.dwords == 662u ? 80u : 40u, 7u);
    CHECK(!analyze(exact, overlap),
          "overlapping lane ranges with different pointer records are not ambiguous authority");

    Fixture partial_final_wave(shape);
    partial_final_wave.main_record(
        6u, partial_final_wave.second_first_thread(), 30u, 11u);
    CHECK(analyze(exact, partial_final_wave),
          "unmatched padded lanes in a partial final wave exit before pointer consumers");

    Fixture exact_boundary(shape);
    exact_boundary.main_u32(5u, 68u, 48u);
    exact_boundary.main_u32(6u, 68u, 48u);
    exact_boundary.descriptor(7u,
        reinterpret_cast<uint64_t>(exact_boundary.first_pointee.data()), 64u);
    exact_boundary.descriptor(11u,
        reinterpret_cast<uint64_t>(exact_boundary.second_pointee.data()), 64u);
    CHECK(analyze(exact, exact_boundary),
          "last M68 consumer ending exactly at the declared V# length is contained");
    exact_boundary.descriptor(7u,
        reinterpret_cast<uint64_t>(exact_boundary.first_pointee.data()), 63u);
    CHECK(!analyze(exact, exact_boundary),
          "the same M68 consumer ending one byte past the V# length is rejected");

    Fixture wrapped_offset(shape);
    wrapped_offset.main_u32(5u, 68u, UINT32_MAX);
    CHECK(!analyze(exact, wrapped_offset),
          "u32 residual addition wrap cannot escape an exact containment obligation");

    if (shape.dwords == 642u) {
        Fixture q2_branch(shape);
        q2_branch.first_pointee[2] = 254u;
        q2_branch.second_pointee[2] = 254u;
        CHECK(analyze(exact, q2_branch),
              "q2=254 leaves the pc117 guarded residual consumers inactive");
        q2_branch.first_pointee[2] = 255u;
        CHECK(!analyze(exact, q2_branch),
              "q2=255 activates the same pc117 residual sites and enforces containment");

        Fixture main68_inactive(shape);
        main68_inactive.first_pointee[2] = 255u;
        main68_inactive.second_pointee[2] = 255u;
        main68_inactive.main_u32(5u, 68u, 0u);
        main68_inactive.main_u32(6u, 68u, 0u);
        main68_inactive.descriptor(7u,
            reinterpret_cast<uint64_t>(main68_inactive.first_pointee.data()), 268u);
        main68_inactive.descriptor(11u,
            reinterpret_cast<uint64_t>(main68_inactive.second_pointee.data()), 268u);
        CHECK(analyze(exact, main68_inactive),
              "main+68 zero leaves the pc245/246 residual subgroup inactive");

        Fixture main68_boundary(shape);
        main68_boundary.first_pointee[2] = 255u;
        main68_boundary.second_pointee[2] = 255u;
        main68_boundary.main_u32(5u, 68u, 252u);
        main68_boundary.main_u32(6u, 68u, 252u);
        main68_boundary.descriptor(7u,
            reinterpret_cast<uint64_t>(main68_boundary.first_pointee.data()), 268u);
        main68_boundary.descriptor(11u,
            reinterpret_cast<uint64_t>(main68_boundary.second_pointee.data()), 268u);
        CHECK(analyze(exact, main68_boundary),
              "main+68 residual ending exactly at L is contained");
        main68_boundary.descriptor(7u,
            reinterpret_cast<uint64_t>(main68_boundary.first_pointee.data()), 267u);
        CHECK(!analyze(exact, main68_boundary),
              "the same main+68 residual ending one byte past L is rejected");
    }

    Fixture outer_oob(shape);
    store_u32(outer_oob.outer, 20u, 16383u);
    store_u32(outer_oob.outer, 24u, 2u);
    CHECK(!analyze(exact, outer_oob), "outer start/count cannot exceed the main table");

    Fixture pointer_oob(shape);
    pointer_oob.main_record(5u, 0u, 32u, Fixture::kPointerBytes / Fixture::kPointerStride);
    CHECK(!analyze(exact, pointer_oob), "reachable pointer-table index must be in bounds");

    Fixture no_reachable(shape);
    no_reachable.main_record(5u, 1000u, 1u, 7u);
    no_reachable.main_record(6u, 1001u, 1u, 11u);
    CHECK(!analyze(exact, no_reachable), "an empty reachable descriptor set is not authority");

    Fixture bad_range(shape);
    bad_range.main_record(5u, UINT32_MAX - 1u, 4u, 7u);
    CHECK(!analyze(exact, bad_range), "main-record thread range cannot wrap");

    Fixture split_snapshot(shape);
    std::vector<uint8_t> shadow = split_snapshot.main;
    shadow[5u * Fixture::kMainStride + 48u] ^= 1u;
    split_snapshot.table.resources[2].host_data = shadow.data();
    split_snapshot.table.resources[2].host_data_size = shadow.size();
    CHECK(!analyze(exact, split_snapshot),
          "range and pointer-index fetches require byte-identical main-table snapshots");

    Fixture oversized_pointer_snapshot(shape);
    std::vector<uint8_t> oversized_pointer_bytes = oversized_pointer_snapshot.pointers;
    oversized_pointer_bytes.resize(oversized_pointer_bytes.size() + 64u, 0x5au);
    ShaderResource* oversized_pointer = nullptr;
    for (ShaderResource& resource : oversized_pointer_snapshot.table.resources)
        if (resource.fetch_pc == shape.source_pc) oversized_pointer = &resource;
    if (oversized_pointer) {
        oversized_pointer->host_data = oversized_pointer_bytes.data();
        oversized_pointer->host_data_size = oversized_pointer_bytes.size();
    }
    CHECK(analyze(exact, oversized_pointer_snapshot),
          "ordinary host backing may extend beyond the logical V# without becoming a carrier");

    Fixture malformed_v3_snapshot(shape);
    std::vector<uint8_t> malformed_v3_bytes = malformed_v3_snapshot.pointers;
    malformed_v3_bytes.resize(
        malformed_v3_bytes.size() + kIndirectBufferRelocationHeaderBytes, 0u);
    store_u32(malformed_v3_bytes, Fixture::kPointerBytes,
              kIndirectPointerDescriptorRangeLayout.tag);
    store_u32(malformed_v3_bytes, Fixture::kPointerBytes + sizeof(uint32_t),
              kIndirectPointerDescriptorRangeLayout.version);
    ShaderResource* malformed_v3 = nullptr;
    for (ShaderResource& resource : malformed_v3_snapshot.table.resources)
        if (resource.fetch_pc == shape.source_pc) malformed_v3 = &resource;
    if (malformed_v3) {
        malformed_v3->host_data = malformed_v3_bytes.data();
        malformed_v3->host_data_size = malformed_v3_bytes.size();
    }
    CHECK(!analyze(exact, malformed_v3_snapshot),
          "advertised but malformed v3 backing cannot fall back to ordinary live authority");

    Fixture null_descriptor(shape);
    null_descriptor.descriptor(7u, 0u, 64u);
    CHECK(!analyze(exact, null_descriptor), "reachable null V# base is rejected");

    Fixture negative_descriptor(shape);
    const size_t descriptor_word1 = 7u * Fixture::kPointerStride + 4u;
    store_u32(negative_descriptor.pointers, descriptor_word1,
              static_cast<uint32_t>((reinterpret_cast<uint64_t>(
                  negative_descriptor.first_pointee.data()) >> 32u) & 0xffffu) | 0x8000u);
    CHECK(!analyze(exact, negative_descriptor), "bit-47-set V# base is rejected");

    Fixture zero_range(shape);
    zero_range.descriptor(7u,
        reinterpret_cast<uint64_t>(zero_range.first_pointee.data()), 0u);
    CHECK(!analyze(exact, zero_range), "reachable V# must declare a nonzero range");

    Fixture huge_range(shape);
    huge_range.descriptor(7u,
        reinterpret_cast<uint64_t>(huge_range.first_pointee.data()), UINT32_MAX, 0x3fffu);
    CHECK(!analyze(exact, huge_range), "V# record-count times stride cannot overflow the cap");

    Fixture bad_launch(shape);
    bad_launch.config.wave_size = 64u;
    CHECK(!analyze(exact, bad_launch), "DescriptorRange requires the exact Wave32 launch");
    Fixture group_count_launch(shape);
    group_count_launch.config.exact_thread_extent = false;
    IndirectPointerRelocationProof group_count_proof;
    CHECK(analyze(exact, group_count_launch, &group_count_proof) &&
              group_count_proof.records.size() == 2u,
          "full workgroup-count launch retains the finite DescriptorRange domain");
    // Production resource discovery retains explicit null V# aliases for the inactive raw-buffer
    // arms and per-fetch aliases for the unrelated live-record table. Keep the group-count emitter
    // fixture faithful to that table so compilation tests the relocated GLOBAL consumers rather
    // than stopping at an absent ordinary MUBUF binding.
    std::vector<uint8_t> live_records(6291456u);
    if (shape.dwords == 642u) {
        for (uint32_t fetch_pc : std::array<uint32_t, 18>{
                 271u, 273u, 277u, 310u, 312u, 314u, 324u, 326u, 328u,
                 330u, 354u, 356u, 358u, 365u, 367u, 369u, 402u, 404u})
            group_count_launch.add_resource(fetch_pc, live_records, 48u);
        for (uint32_t fetch_pc : std::array<uint32_t, 11>{
                 140u, 142u, 144u, 146u, 148u, 150u,
                 591u, 593u, 595u, 597u, 637u})
            group_count_launch.add_empty_resource(fetch_pc);
    } else {
        for (uint32_t fetch_pc : std::array<uint32_t, 15>{
                 276u, 278u, 282u, 326u, 328u, 330u, 375u, 377u,
                 379u, 406u, 408u, 410u, 412u, 436u, 438u})
            group_count_launch.add_resource(fetch_pc, live_records, 48u);
        group_count_launch.add_resource(527u, group_count_launch.outer, 16u);
        for (uint32_t fetch_pc : std::array<uint32_t, 11>{
                 133u, 135u, 137u, 153u, 155u, 157u,
                 611u, 613u, 615u, 617u, 657u})
            group_count_launch.add_empty_resource(fetch_pc);
    }
    CHECK(discover_rdna2_indirect_pointer_relocations(
              exact.data(), exact.size(), group_count_launch.config,
              group_count_launch.table) &&
              validate_rdna2_indirect_pointer_relocations(
                  exact.data(), exact.size(), group_count_launch.config,
                  group_count_launch.table),
          "full workgroup-count launch materializes and validates its v3 carrier");
    test_emitter_integration(shape, exact, group_count_launch);

    Fixture inexact_partial_launch(shape);
    inexact_partial_launch.config.exact_thread_extent = false;
    inexact_partial_launch.config.threads_x = 63u;
    CHECK(!analyze(exact, inexact_partial_launch),
          "non-aligned inexact extent cannot masquerade as a full workgroup-count launch");
    Fixture huge_launch(shape);
    huge_launch.config.threads_x = 65537u;
    CHECK(!analyze(exact, huge_launch),
          "DescriptorRange rejects capture-controlled extents above the finite proof cap");
}

} // namespace

int main() {
    for (const Shape& shape : kShapes) test_shape(shape);
    return failures ? 1 : 0;
}
