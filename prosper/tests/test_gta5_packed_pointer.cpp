#include "gpu/gpu_execute.hpp"
#include "gpu/gpu_capture.hpp"
#include "gpu/gpu_dependency_graph.hpp"
#include "gpu/rdna2_gta5_packed_pointer.hpp"
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

std::vector<uint32_t> program() {
    const std::filesystem::path path =
        std::filesystem::path(__FILE__).parent_path() / "data/gta5_413cf9d00.hex";
    std::ifstream input(path);
    std::string hex((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    CHECK(input.good() || input.eof(), "exact program fixture is readable");
    CHECK(hex.size() == 362u * 8u, "exact program fixture has 362 dwords");
    std::vector<uint32_t> words(hex.size() / 8u);
    auto* bytes = reinterpret_cast<uint8_t*>(words.data());
    for (size_t index = 0; index < words.size() * sizeof(uint32_t); ++index)
        bytes[index] = static_cast<uint8_t>(
            nibble(hex[index * 2u]) << 4u | nibble(hex[index * 2u + 1u]));
    return words;
}

uint32_t load_u32(const uint8_t* bytes, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

size_t count_spirv_opcode(const std::vector<uint32_t>& words, uint16_t opcode) {
    size_t count = 0;
    for (size_t index = 5u; index < words.size();) {
        const uint16_t word_count = static_cast<uint16_t>(words[index] >> 16u);
        if (!word_count || index + word_count > words.size()) return 0u;
        count += static_cast<uint16_t>(words[index]) == opcode;
        index += word_count;
    }
    return count;
}

struct Fixture {
    std::vector<uint8_t> source;
    std::array<std::array<uint8_t, kGta5PackedPointerSlotBytes>, 3> pointees{};
    std::vector<uint8_t> other = std::vector<uint8_t>(16'384u);
    ShaderResourceTable table;
    ComputeShaderConfig config;

    explicit Fixture(uint32_t threads = 151u)
        : source(static_cast<size_t>(threads) * kGta5PackedPointerSourceStride) {
        for (size_t slot = 0; slot < pointees.size(); ++slot)
            for (size_t index = 0; index < pointees[slot].size(); ++index)
                pointees[slot][index] = static_cast<uint8_t>(slot * 73u + index);
        const size_t pointer_count = (static_cast<size_t>(threads) + 63u) / 64u;
        for (size_t index = 0; index < pointer_count; ++index) {
            const uint64_t pointer = reinterpret_cast<uint64_t>(pointees[index].data());
            std::memcpy(source.data() + index * 64u * kGta5PackedPointerSourceStride,
                        &pointer, sizeof(pointer));
        }

        for (uint32_t pc : {26u, 28u, 30u, 32u, 34u, 36u}) {
            ShaderResource resource;
            resource.cls = ResourceClass::ConstantBuffer;
            resource.format = DataFormat::Uint32;
            resource.num_components = 1u;
            resource.gpu_addr = reinterpret_cast<uint64_t>(source.data());
            resource.size = source.size();
            resource.stride = kGta5PackedPointerSourceStride;
            resource.fetch_pc = pc;
            resource.host_data = source.data();
            resource.host_data_size = source.size();
            table.resources.push_back(resource);
        }
        auto add = [&](uint32_t pc, uint32_t size, uint32_t stride) {
            ShaderResource resource;
            resource.cls = ResourceClass::ConstantBuffer;
            resource.format = DataFormat::Uint32;
            resource.num_components = 1u;
            resource.gpu_addr = reinterpret_cast<uint64_t>(other.data());
            resource.size = size;
            resource.stride = stride;
            resource.fetch_pc = pc;
            resource.host_data = other.data();
            resource.host_data_size = other.size();
            if (pc == kGta5PackedPointerAtomicSourcePc)
                resource.scalar_raw_pointer_word_hi =
                    static_cast<uint32_t>(resource.gpu_addr >> 32u);
            table.resources.push_back(resource);
        };
        add(297u, 4228u, 28u); add(303u, 4228u, 28u);
        add(305u, 604u, 4u);
        for (uint32_t pc : {321u, 329u, 331u, 333u, 336u, 338u}) add(pc, 24u, 4u);
        add(353u, 28u, 0u);

        config.user_sgprs.resize(14u);
        config.local_x = 64u;
        config.local_y = config.local_z = 1u;
        config.exact_thread_extent = true;
        config.threads_x = threads;
        config.threads_y = config.threads_z = 1u;
        config.wave_size = 64u;
        config.tgid_x_en = true;
        config.tidig_comp_cnt = 0u;
        config.storage_buffer_int64_atomics = true;
    }
};

} // namespace

int main() {
    const std::vector<uint32_t> exact = program();
    Fixture valid;
    const std::vector<uint8_t> original_source = valid.source;
    CHECK(rdna2_gta5_packed_pointer_shader(exact.data(), exact.size()),
          "exact routed program identity");
    CHECK(rdna2_gta5_packed_pointer_launch(exact.data(), exact.size(), valid.config),
          "exact routed launch ABI");
    CHECK(discover_rdna2_gta5_packed_pointer(
              exact.data(), exact.size(), valid.config, valid.table),
          "three readable pointer records acquire packed authority");
    const ShaderResource* packed = valid.table.by_fetch_pc(kGta5PackedPointerSourcePc);
    CHECK(packed && is_gta5_packed_pointer_resource(*packed),
          "pc26 becomes the sole packed source resource");
    CHECK(packed && packed->host_data_size ==
              valid.source.size() + kGta5PackedPointerHeaderBytes +
                  3u * kGta5PackedPointerSlotBytes &&
              packed->indirect_buffer_slot_count == 3u,
          "shadow contains one header and three 368-byte slots");
    CHECK(valid.table.owned_host_data.size() == 1u &&
              packed && valid.table.owned_host_data[0]->data() == packed->host_data,
          "resource table owns the packed backing");
    CHECK(valid.table.by_fetch_pc(28u)->host_data == valid.source.data() &&
              valid.table.by_fetch_pc(36u)->host_data == valid.source.data(),
          "pc28..36 retain the original unmodified table");
    CHECK(valid.source == original_source,
          "discovery never rewrites guest source bytes");
    if (packed) {
        for (size_t record = 0; record < 3; ++record) {
            const size_t source_offset = record * 64u * kGta5PackedPointerSourceStride;
            const uint32_t packed_offset = load_u32(packed->host_data, source_offset);
            CHECK(load_u32(packed->host_data, source_offset + 4u) == kGta5PackedPointerTag,
                  "each eligible pc26 qword carries the exact tag");
            CHECK(std::memcmp(packed->host_data + packed_offset,
                              valid.pointees[record].data(), kGta5PackedPointerSlotBytes) == 0,
                  "each tagged record resolves to its exact pointee snapshot");
        }
    }
    CHECK(rdna2_gta5_packed_pointer_dispatch(
              exact.data(), exact.size(), valid.config, valid.table),
          "final compiler boundary revalidates the live source and pointees");

    ShaderResourceTable copied = valid.table;
    valid.table.owned_host_data.clear();
    CHECK(copied.by_fetch_pc(26u) && copied.by_fetch_pc(26u)->host_data &&
              rdna2_gta5_packed_pointer_dispatch(
                  exact.data(), exact.size(), valid.config, copied),
          "table copy retains packed backing ownership");

    assign_convention_bindings(copied, 2u);
    const std::vector<uint32_t> spirv = recompile_compute(
        exact.data(), exact.size(), &copied, valid.config,
        {RecompileDiagnosticStage::Compute, 0x413cf9d00u});
    CHECK(!spirv.empty(), "production emitter lowers all exact packed-pointer consumers");
    CHECK(count_spirv_opcode(spirv, 241u /* OpAtomicOr */) == 1u,
          "production emitter contains exactly one real 64-bit atomic OR for pc355");
    CHECK(validate_spirv_descriptor_interface(
              spirv, &copied, 0u, SpirvShaderStage::Compute, false).ok(),
          "packed module and runtime descriptor table agree");
    const DescriptorValidationReport packed_report = validate_spirv_descriptor_interface(
        spirv, &copied, 0u, SpirvShaderStage::Compute, false);
    const SpirvDescriptorBinding* packed_descriptor = packed
        ? find_spirv_descriptor_binding(packed_report, 0u, copied.by_fetch_pc(26u)->binding)
        : nullptr;
    const StorageBufferMaterializationPlan packed_plan = packed_descriptor
        ? plan_storage_buffer_materialization(*packed_descriptor, *copied.by_fetch_pc(26u))
        : StorageBufferMaterializationPlan{};
    CHECK(packed_descriptor && packed_descriptor->readable && !packed_descriptor->writable &&
              packed_plan.valid && packed_plan.logical_bytes == valid.source.size() &&
              packed_plan.binding_bytes == packed->host_data_size,
          "Vulkan materialization binds the complete read-only packed shadow");
    const ShaderResource* atomic_source = copied.by_fetch_pc(kGta5PackedPointerAtomicSourcePc);
    const SpirvDescriptorBinding* atomic_descriptor = atomic_source
        ? find_spirv_descriptor_binding(
              packed_report, 0u, atomic_source->binding) : nullptr;
    CHECK(atomic_source && atomic_source->size == kGta5PackedPointerAtomicBindingBytes &&
              atomic_descriptor && atomic_descriptor->readable && atomic_descriptor->writable &&
              atomic_descriptor->atomic_access &&
              atomic_descriptor->required_bytes == kGta5PackedPointerAtomicBindingBytes,
          "pc353 backing expands to byte 31 and reflects pc355 as a writable atomic binding");

    for (uint32_t threads : {8u, 13u, 152u}) {
        Fixture variable_launch(threads);
        CHECK(discover_rdna2_gta5_packed_pointer(
                  exact.data(), exact.size(), variable_launch.config, variable_launch.table),
              "observed variable exact-thread launch acquires packed authority");
        const ShaderResource* variable_packed =
            variable_launch.table.by_fetch_pc(kGta5PackedPointerSourcePc);
        const uint32_t expected_pointers = (threads + 63u) / 64u;
        CHECK(variable_packed &&
                  variable_packed->indirect_buffer_slot_count == expected_pointers &&
                  variable_packed->host_data_size ==
                      variable_launch.source.size() + kGta5PackedPointerHeaderBytes +
                          expected_pointers * kGta5PackedPointerSlotBytes,
              "variable launch snapshots only its reachable lane-zero pointer records");
        assign_convention_bindings(variable_launch.table, 2u);
        CHECK(!recompile_compute(exact.data(), exact.size(), &variable_launch.table,
                                 variable_launch.config).empty(),
              "variable exact-thread launch compiles through the production emitter");
    }

    // Exercise the production scalar-fold/resource-builder boundary: decode_buffer_descriptor
    // deliberately masks Base48, so the exact raw pointer's upper word must survive separately.
    Fixture frontend_metadata_bit;
    std::vector<uint8_t> scalar_table(256u);
    const uint64_t source_address =
        reinterpret_cast<uint64_t>(frontend_metadata_bit.source.data());
    const uint64_t scalar_table_address = reinterpret_cast<uint64_t>(scalar_table.data());
    const uint64_t dynamic_pointer =
        reinterpret_cast<uint64_t>(frontend_metadata_bit.other.data()) |
        (static_cast<uint64_t>(kGta5PackedPointerRawWordHiFirstMetadataBit) << 32u);
    std::memcpy(scalar_table.data() + 0x88u, &dynamic_pointer, sizeof(dynamic_pointer));
    std::vector<uint32_t> frontend_user(14u);
    frontend_user[0] = static_cast<uint32_t>(scalar_table_address);
    frontend_user[1] = static_cast<uint32_t>(scalar_table_address >> 32u);
    frontend_user[6] = static_cast<uint32_t>(source_address);
    frontend_user[7] = static_cast<uint32_t>(source_address >> 32u) & 0xffffu;
    frontend_user[7] |= kGta5PackedPointerSourceStride << 16u;
    frontend_user[8] = frontend_metadata_bit.config.threads_x;
    frontend_user[9] = 0x00016204u;
    ShaderResourceTable frontend_table;
    (void)add_compute_buffer_resources(
        frontend_table, exact.data(), exact.size(), frontend_user.data(), frontend_user.size(),
        frontend_metadata_bit.config.local_x, frontend_metadata_bit.config.threads_x, 14u, nullptr);
    const ShaderResource* frontend_pc353 =
        frontend_table.by_fetch_pc(kGta5PackedPointerAtomicSourcePc);
    CHECK(frontend_pc353 &&
              (frontend_pc353->scalar_raw_pointer_word_hi &
               kGta5PackedPointerRawWordHiMetadataMask),
          "production resource builder retains pc353 raw-pointer metadata above Base48");
    if (frontend_pc353) {
        ShaderResource* fixture_pc353 = const_cast<ShaderResource*>(
            frontend_metadata_bit.table.by_fetch_pc(kGta5PackedPointerAtomicSourcePc));
        *fixture_pc353 = *frontend_pc353;
        fixture_pc353->host_data = frontend_metadata_bit.other.data();
        fixture_pc353->host_data_size = frontend_metadata_bit.other.size();
        CHECK(!discover_rdna2_gta5_packed_pointer(
                  exact.data(), exact.size(), frontend_metadata_bit.config,
                  frontend_metadata_bit.table),
              "frontend-preserved raw-pointer metadata keeps pc355 fail-visible");
    }

    Fixture dedup;
    const uint64_t same = reinterpret_cast<uint64_t>(dedup.pointees[0].data());
    std::memcpy(dedup.source.data() + 64u * kGta5PackedPointerSourceStride, &same, sizeof(same));
    CHECK(discover_rdna2_gta5_packed_pointer(
              exact.data(), exact.size(), dedup.config, dedup.table) &&
              dedup.table.by_fetch_pc(26u)->indirect_buffer_slot_count == 2u,
          "equal guest pointers share one immutable packed slot");

    Fixture null_pointer;
    std::memset(null_pointer.source.data(), 0, sizeof(uint64_t));
    CHECK(!discover_rdna2_gta5_packed_pointer(
               exact.data(), exact.size(), null_pointer.config, null_pointer.table),
          "null dereferenced record remains fail-visible");

    Fixture launch_mutation;
    launch_mutation.config.exact_thread_extent = false;
    CHECK(!discover_rdna2_gta5_packed_pointer(
               exact.data(), exact.size(), launch_mutation.config, launch_mutation.table),
          "non-exact launch cannot acquire packed authority");

    Fixture missing_atomic_feature;
    missing_atomic_feature.config.storage_buffer_int64_atomics = false;
    CHECK(!discover_rdna2_gta5_packed_pointer(
               exact.data(), exact.size(), missing_atomic_feature.config,
               missing_atomic_feature.table) &&
              missing_atomic_feature.table.by_fetch_pc(kGta5PackedPointerAtomicSourcePc)->size ==
                  kGta5PackedPointerAtomicLoadBytes &&
              missing_atomic_feature.table.owned_host_data.empty(),
          "missing Int64 atomics keeps the exact contract fail-visible and rolls back acquisition");

    Fixture pc26_mutation;
    std::vector<uint32_t> changed_pc26 = exact;
    changed_pc26[26] ^= 1u;
    CHECK(!discover_rdna2_gta5_packed_pointer(
               changed_pc26.data(), changed_pc26.size(),
               pc26_mutation.config, pc26_mutation.table),
          "same pc26 pointer-load packet mutation rejects");

    Fixture pc70_mutation;
    std::vector<uint32_t> changed_pc70 = exact;
    changed_pc70[70] ^= 4u;
    CHECK(!discover_rdna2_gta5_packed_pointer(
               changed_pc70.data(), changed_pc70.size(),
               pc70_mutation.config, pc70_mutation.table),
          "same pc70 GLOBAL-load packet mutation rejects");

    Fixture pc355_mutation;
    std::vector<uint32_t> changed_pc355 = exact;
    changed_pc355[355] ^= 1u;
    CHECK(!discover_rdna2_gta5_packed_pointer(
               changed_pc355.data(), changed_pc355.size(),
               pc355_mutation.config, pc355_mutation.table) &&
              recompile_compute(changed_pc355.data(), changed_pc355.size(),
                                &pc355_mutation.table, pc355_mutation.config).empty(),
          "same pc355 atomic packet mutation remains fail-visible");

    Fixture pc347_mutation;
    std::vector<uint32_t> changed_pc347 = exact;
    changed_pc347[347] ^= 1u;
    CHECK(!discover_rdna2_gta5_packed_pointer(
               changed_pc347.data(), changed_pc347.size(),
               pc347_mutation.config, pc347_mutation.table) &&
              recompile_compute(changed_pc347.data(), changed_pc347.size(),
                                &pc347_mutation.table, pc347_mutation.config).empty(),
          "same pc347 descriptor-stride producer mutation cannot authorize pc355");

    Fixture pc353_shape_mutation;
    auto pc353 = std::find_if(
        pc353_shape_mutation.table.resources.begin(),
        pc353_shape_mutation.table.resources.end(),
        [](const ShaderResource& resource) {
            return resource.fetch_pc == kGta5PackedPointerAtomicSourcePc;
        });
    CHECK(pc353 != pc353_shape_mutation.table.resources.end(),
          "fixture retains the pc353 descriptor witness");
    if (pc353 != pc353_shape_mutation.table.resources.end()) {
        const auto pc26_is_restored = [&] {
            const ShaderResource* source =
                pc353_shape_mutation.table.by_fetch_pc(kGta5PackedPointerSourcePc);
            return source && !is_gta5_packed_pointer_marker_candidate(*source) &&
                source->host_data == pc353_shape_mutation.source.data() &&
                source->host_data_size == pc353_shape_mutation.source.size() &&
                pc353_shape_mutation.table.owned_host_data.empty();
        };
        pc353->size = 32u;
        CHECK(!discover_rdna2_gta5_packed_pointer(
                   exact.data(), exact.size(), pc353_shape_mutation.config,
                   pc353_shape_mutation.table) && pc26_is_restored(),
              "caller-expanded pc353 size cannot authorize the pc355 atomic contract");
        pc353->size = kGta5PackedPointerAtomicLoadBytes;
        pc353->stride = 4u;
        CHECK(!discover_rdna2_gta5_packed_pointer(
                   exact.data(), exact.size(), pc353_shape_mutation.config,
                   pc353_shape_mutation.table) && pc26_is_restored(),
              "pc353 stride mutation cannot authorize the pc355 atomic contract");
        pc353->stride = 0u;
        pc353->scalar_raw_pointer_word_hi |=
            kGta5PackedPointerRawWordHiFirstMetadataBit;
        CHECK(!discover_rdna2_gta5_packed_pointer(
                   exact.data(), exact.size(), pc353_shape_mutation.config,
                   pc353_shape_mutation.table) &&
                  pc26_is_restored() &&
                  recompile_compute(exact.data(), exact.size(),
                                    &pc353_shape_mutation.table,
                                    pc353_shape_mutation.config).empty(),
              "pc353 raw-pointer metadata mutation keeps the pc355 atomic fail-visible");
    }

    Fixture stale_source;
    CHECK(discover_rdna2_gta5_packed_pointer(
              exact.data(), exact.size(), stale_source.config, stale_source.table),
          "stale-source fixture starts valid");
    assign_convention_bindings(stale_source.table, 2u);
    CHECK(!recompile_compute_shader_cached(
               exact.data(), exact.size(), &stale_source.table, stale_source.config).empty(),
          "cold shader-cache lookup accepts the current source and pointee snapshots");
    const uint64_t replacement = reinterpret_cast<uint64_t>(stale_source.pointees[1].data());
    std::memcpy(stale_source.source.data(), &replacement, sizeof(replacement));
    CHECK(!rdna2_gta5_packed_pointer_dispatch(
               exact.data(), exact.size(), stale_source.config, stale_source.table),
          "post-discovery pc26 source-pointer change rejects at the final boundary");
    CHECK(recompile_compute_shader_cached(
              exact.data(), exact.size(), &stale_source.table, stale_source.config).empty(),
          "warm shader-cache lookup revalidates pc26 before returning the cached module");

    Fixture stale_pointee;
    CHECK(discover_rdna2_gta5_packed_pointer(
              exact.data(), exact.size(), stale_pointee.config, stale_pointee.table),
          "stale-pointee fixture starts valid");
    stale_pointee.pointees[2][367] ^= 1u;
    CHECK(!rdna2_gta5_packed_pointer_dispatch(
               exact.data(), exact.size(), stale_pointee.config, stale_pointee.table),
          "post-discovery pointee-footprint change rejects at the final boundary");

    // Capture two dispatches with the same bindings but different pointee snapshots. Replay must
    // preserve them as distinct owned instances; sharing by binding is correct for GDS, not for
    // command-ordered packed-pointer state.
    Fixture capture_second;
    capture_second.pointees[0][0] ^= 0x5au;
    CHECK(discover_rdna2_gta5_packed_pointer(
              exact.data(), exact.size(), capture_second.config, capture_second.table),
          "second capture dispatch acquires its own packed snapshot");
    assign_convention_bindings(capture_second.table, 2u);
    auto make_compute = [&](const ShaderResourceTable& table, uint64_t index) {
        ComputeItem item;
        item.spirv = spirv;
        item.user_sgprs = valid.config.user_sgprs;
        item.resources = std::make_shared<ShaderResourceTable>(table);
        item.launch.threads_x = valid.config.threads_x;
        item.launch.threads_y = item.launch.threads_z = 1u;
        item.launch.local_x = 64u;
        item.launch.local_y = item.launch.local_z = 1u;
        item.launch.groups_x = (valid.config.threads_x + valid.config.local_x - 1u) /
            valid.config.local_x;
        item.launch.groups_y = item.launch.groups_z = 1u;
        item.code_addr = reinterpret_cast<uint64_t>(exact.data());
        item.dispatch_index = index;
        item.command_order = index;
        item.recompile_config = valid.config;
        item.recompile_config_available = true;
        return item;
    };
    const std::vector<ComputeItem> capture_computes{
        make_compute(copied, 1u), make_compute(capture_second.table, 2u)};
    const std::vector<SubmitOperation> capture_operations{
        {SubmitOperationKind::Dispatch, 0u, 1u},
        {SubmitOperationKind::Dispatch, 1u, 2u},
    };
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
                exact.size() * sizeof(uint32_t))) return count;
        if (const size_t count = read(reinterpret_cast<uint64_t>(valid.source.data()),
                                      valid.source.data(), valid.source.size())) return count;
        if (const size_t count = read(reinterpret_cast<uint64_t>(valid.other.data()),
                                      valid.other.data(), valid.other.size())) return count;
        if (const size_t count = read(
                reinterpret_cast<uint64_t>(capture_second.source.data()),
                capture_second.source.data(), capture_second.source.size())) return count;
        return read(reinterpret_cast<uint64_t>(capture_second.other.data()),
                    capture_second.other.data(), capture_second.other.size());
    };
    GpuCaptureFile captured;
    GpuCaptureMetadata metadata;
    std::string capture_error;
    CHECK(capture_submit_items({}, capture_computes, capture_operations, metadata,
                               capture_reader, captured, capture_error) &&
              captured.computes.size() == 2u &&
              captured.computes[0].resources.resources[0].internal_bytes.size() ==
                  copied.by_fetch_pc(26u)->host_data_size &&
              captured.computes[1].resources.resources[0].internal_bytes.size() ==
                  capture_second.table.by_fetch_pc(26u)->host_data_size &&
              captured.computes[0].resources.resources[0].internal_bytes !=
                  captured.computes[1].resources.resources[0].internal_bytes,
          "capture stores each dispatch-owned packed shadow as internal bytes");
    std::vector<uint8_t> capture_bytes;
    GpuCaptureFile loaded_capture;
    GpuReplayFrame replay;
    const bool replay_materialized =
        serialize_gpu_capture(captured, capture_bytes, capture_error) &&
        deserialize_gpu_capture(capture_bytes, loaded_capture, capture_error) &&
        materialize_gpu_replay(loaded_capture, replay, capture_error);
    const ShaderResource* replay_first = replay_materialized && replay.computes.size() == 2u &&
            replay.computes[0].resources
        ? replay.computes[0].resources->by_fetch_pc(26u) : nullptr;
    const ShaderResource* replay_second = replay_materialized && replay.computes.size() == 2u &&
            replay.computes[1].resources
        ? replay.computes[1].resources->by_fetch_pc(26u) : nullptr;
    CHECK(replay_first && replay_second &&
              is_gta5_packed_pointer_resource(*replay_first) &&
              is_gta5_packed_pointer_resource(*replay_second) &&
              replay_first->host_data != replay_second->host_data,
          "capture round-trip revalidates two distinct replay-owned packed snapshots");
    GpuDependencyGraph packed_graph;
    const bool packed_graph_built = replay_materialized &&
        build_gpu_dependency_graph(replay, packed_graph, capture_error);
    const auto has_external_range = [&](uint64_t address, uint64_t bytes, uint32_t binding) {
        return std::any_of(
            packed_graph.external_leaves.begin(), packed_graph.external_leaves.end(),
            [&](const GpuDependencyLeaf& leaf) {
                return leaf.access.addr == address && leaf.access.size == bytes &&
                       leaf.access.binding == binding;
            });
    };
    CHECK(packed_graph_built && replay_first &&
              has_external_range(
                  replay_first->gpu_addr, replay_first->size, replay_first->binding) &&
              !has_external_range(
                  replay_first->gpu_addr, replay_first->host_data_size, replay_first->binding),
          "dependency closure keeps the packed source at its logical guest-table span");
    const ShaderResource* replay_atomic = replay_materialized && replay.computes.size() == 2u &&
            replay.computes[0].resources
        ? replay.computes[0].resources->by_fetch_pc(kGta5PackedPointerAtomicSourcePc) : nullptr;
    CHECK(packed_graph_built && replay_atomic &&
              has_external_range(replay_atomic->gpu_addr,
                                 kGta5PackedPointerAtomicBindingBytes,
                                 replay_atomic->binding),
          "dependency closure includes pc355's complete writable qword footprint");
    GpuCaptureFile malformed_capture = loaded_capture;
    malformed_capture.computes[0].resources.resources[0].internal_bytes[
        valid.source.size()] ^= 1u;
    CHECK(!materialize_gpu_replay(malformed_capture, replay, capture_error),
          "replay rejects a packed shadow whose exact header tag was corrupted");

    GpuCaptureFile stale_raw_capture = loaded_capture;
    auto& stale_raw = stale_raw_capture.raw_shader_versions[
        stale_raw_capture.computes[0].raw_shader_index];
    stale_raw.words[70] ^= 4u;
    stale_raw.content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(stale_raw.words.data()),
        stale_raw.words.size() * sizeof(uint32_t));
    CHECK(!materialize_gpu_replay(stale_raw_capture, replay, capture_error) &&
              replay.computes.empty(),
          "replay rejects packed state after the exact pc70 consumer mutates");

    GpuCaptureFile stale_atomic_word_capture = loaded_capture;
    auto& packed_bytes =
        stale_atomic_word_capture.computes[0].resources.resources[0].internal_bytes;
    const size_t atomic_word_offset = valid.source.size() + 40u;
    uint32_t stale_atomic_word = load_u32(packed_bytes.data(), atomic_word_offset) |
        kGta5PackedPointerRawWordHiFirstMetadataBit;
    std::memcpy(packed_bytes.data() + atomic_word_offset,
                &stale_atomic_word, sizeof(stale_atomic_word));
    const uint32_t stale_atomic_word_check = stale_atomic_word ^ kGta5PackedPointerTag;
    std::memcpy(packed_bytes.data() + atomic_word_offset + 4u,
                &stale_atomic_word_check, sizeof(stale_atomic_word_check));
    CHECK(!materialize_gpu_replay(stale_atomic_word_capture, replay, capture_error) &&
              replay.computes.empty(),
          "replay rejects a structurally valid shadow with pc353 raw-pointer metadata");

    GpuCaptureFile stale_descriptor_capture = loaded_capture;
    auto replay_pc353 = std::find_if(
        stale_descriptor_capture.computes[0].resources.resources.begin(),
        stale_descriptor_capture.computes[0].resources.resources.end(),
        [](const GpuCapturedResource& resource) {
            return resource.resource.fetch_pc == kGta5PackedPointerAtomicSourcePc;
        });
    CHECK(replay_pc353 != stale_descriptor_capture.computes[0].resources.resources.end(),
          "capture retains the pc353 descriptor witness");
    if (replay_pc353 != stale_descriptor_capture.computes[0].resources.resources.end()) {
        replay_pc353->resource.stride = 4u;
        CHECK(!materialize_gpu_replay(stale_descriptor_capture, replay, capture_error),
              "replay rejects packed state when pc353 no longer proves the writable atomic");
    }

    GpuCaptureFile stale_launch_capture = loaded_capture;
    ++stale_launch_capture.computes[0].recompile_config.threads_x;
    CHECK(!materialize_gpu_replay(stale_launch_capture, replay, capture_error) &&
              replay.computes.empty(),
          "replay rejects packed state whose exact launch no longer matches the source table");

    GpuCaptureFile stale_grid_capture = loaded_capture;
    ++stale_grid_capture.computes[0].launch.groups_x;
    CHECK(!materialize_gpu_replay(stale_grid_capture, replay, capture_error) &&
              replay.computes.empty(),
          "replay rejects packed state whose captured workgroup grid changed");

    GpuCaptureFile missing_config_capture = loaded_capture;
    missing_config_capture.computes[0].recompile_config_available = false;
    CHECK(!materialize_gpu_replay(missing_config_capture, replay, capture_error),
          "replay rejects packed state without captured launch authority");

    GpuCaptureFile missing_raw_capture = loaded_capture;
    missing_raw_capture.computes[0].raw_shader_index = UINT32_MAX;
    CHECK(!materialize_gpu_replay(missing_raw_capture, replay, capture_error),
          "replay rejects packed state without captured program authority");

    GpuCaptureFile draw_scope_capture = loaded_capture;
    GpuCapturedDraw draw_with_packed_state;
    draw_with_packed_state.vrt.present = true;
    draw_with_packed_state.vrt.resources.push_back(
        draw_scope_capture.computes[0].resources.resources[0]);
    draw_scope_capture.draws.push_back(std::move(draw_with_packed_state));
    std::vector<uint8_t> invalid_draw_bytes;
    CHECK(!serialize_gpu_capture(draw_scope_capture, invalid_draw_bytes, capture_error) &&
              !materialize_gpu_replay(draw_scope_capture, replay, capture_error),
          "draw resource tables cannot consume compute-only packed state");

    std::vector<uint8_t> reserialized_bytes;
    GpuCaptureFile reserialized_capture;
    CHECK(serialize_gpu_capture(loaded_capture, reserialized_bytes, capture_error) &&
              deserialize_gpu_capture(
                  reserialized_bytes, reserialized_capture, capture_error) &&
              materialize_gpu_replay(reserialized_capture, replay, capture_error),
          "zero-marker loaded capture reserializes and remints packed authority");

    if (failures) {
        std::fprintf(stderr, "%d GTA packed-pointer assertion(s) failed\n", failures);
        return 1;
    }
    std::puts("GTA V packed-pointer contract tests passed");
    return 0;
}
