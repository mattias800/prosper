// Runtime-selected storage-buffer arrays through the production compute backend. The pure plan
// assertions pin the one source of truth used by pool/layout/write sizing; the Vulkan execution arm
// proves each concrete table entry reaches its descriptor slot instead of being collapsed to one.
#include "../frontends/shared/live_compute.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace prosper::gpu;

static int failures = 0;
#define CHECK(condition, message)                                                   \
    do {                                                                            \
        if (!(condition)) {                                                         \
            std::printf("  [FAIL] %s\n", message);                                \
            ++failures;                                                             \
        } else {                                                                    \
            std::printf("  [ok]   %s\n", message);                                \
        }                                                                           \
    } while (0)

static ShaderBufferTableEntry table_entry(std::array<uint32_t, 4>& words) {
    ShaderBufferTableEntry entry;
    entry.gpu_addr = reinterpret_cast<uint64_t>(words.data());
    entry.size = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    entry.stride = sizeof(uint32_t);
    entry.host_data = reinterpret_cast<uint8_t*>(words.data());
    entry.host_data_size = entry.size;
    entry.vsharp = {
        static_cast<uint32_t>(entry.gpu_addr),
        static_cast<uint32_t>(entry.gpu_addr >> 32u) | (entry.stride << 16u),
        entry.size / entry.stride,
        (20u << 12u) | 0xfacu,
    };
    return entry;
}

int main() {
    std::printf("== test_live_compute_descriptor_array ==\n");

    std::array<uint32_t, 4> entry0{0x10203040u, 0, 0, 0};
    std::array<uint32_t, 4> entry1{0x55667788u, 0, 0, 0};
    std::array<uint32_t, 4> entry2{0xa1b2c3d4u, 0, 0, 0};
    uint32_t result = 0;

    ShaderResource input;
    input.cls = ResourceClass::ConstantBuffer;
    input.format = DataFormat::Uint32;
    input.num_components = 1;
    input.binding = 2;
    input.gpu_addr = reinterpret_cast<uint64_t>(entry0.data());
    input.size = static_cast<uint32_t>(entry0.size() * sizeof(uint32_t));
    input.stride = sizeof(uint32_t);
    input.sgpr_base = 0;
    input.table_index_count = 4;
    input.table_entry_stride = 16;
    input.table_index_sgpr = 8;
    input.table_selector_mode = BufferTableSelectorMode::UserSgprIndex;
    input.table_entries = {table_entry(entry0), table_entry(entry1), table_entry(entry2), {}};

    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.gpu_addr = reinterpret_cast<uint64_t>(&result);
    output.size = sizeof(result);
    output.sgpr_base = 4;
    output.host_data = reinterpret_cast<uint8_t*>(&result);
    output.host_data_size = sizeof(result);

    ShaderResourceTable resources;
    resources.resources = {input, output};

    // load v1 = input-array[s8][0], then publish it through the ordinary scalar output binding.
    const uint32_t code[] = {
        0xe0300000u, 0x80000100u, // buffer_load_dword v1, off, s[0:3]
        0xbf8c3f70u,              // s_waitcnt vmcnt(0)
        0xe0700000u, 0x80010100u, // buffer_store_dword v1, off, s[4:7]
        0xbf810000u,
    };
    ComputeShaderConfig config;
    config.user_sgprs.resize(9);
    config.local_x = config.local_y = config.local_z = 1;
    const std::vector<uint32_t> spirv =
        recompile_compute(code, std::size(code), &resources, config);
    CHECK(!spirv.empty(), "indexed-buffer compute fixture recompiles");
    if (spirv.empty()) return 1;

    DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, &resources, 0, SpirvShaderStage::Compute, false);
    CHECK(report.ok(), "array input and scalar output validate against the generated module");
    std::sort(report.descriptors.begin(), report.descriptors.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.binding < rhs.binding; });

    const prosper::frontend::LiveComputeBufferDescriptorPlan plan =
        prosper::frontend::plan_live_compute_buffer_descriptors(
            report.descriptors, &resources, true);
    CHECK(plan.valid && plan.bindings.size() == 2 &&
              plan.bindings[0].first_descriptor == 0 &&
              plan.bindings[0].descriptor_count == 4 &&
              plan.bindings[1].first_descriptor == 4 &&
              plan.bindings[1].descriptor_count == 1 &&
              plan.total_descriptor_count == 5,
          "one plan gives the exact array/scalar runs and total pool descriptor count");

    CHECK(!prosper::frontend::plan_live_compute_buffer_descriptors(
               report.descriptors, &resources, false).valid,
          "the same array rejects when descriptor indexing is unavailable");
    std::vector<SpirvDescriptorBinding> runtime_sized = report.descriptors;
    runtime_sized[0].descriptor_count = 0;
    CHECK(!prosper::frontend::plan_live_compute_buffer_descriptors(
               runtime_sized, &resources, true).valid,
          "a runtime-sized shader array rejects against the fixed-array backend contract");
    std::vector<SpirvDescriptorBinding> writable = report.descriptors;
    writable[0].writable = true;
    CHECK(!prosper::frontend::plan_live_compute_buffer_descriptors(
               writable, &resources, true).valid,
          "the same array rejects when its reflected access becomes writable");
    ShaderResourceTable incomplete = resources;
    incomplete.resources[0].table_entries.pop_back();
    CHECK(!prosper::frontend::plan_live_compute_buffer_descriptors(
               report.descriptors, &incomplete, true).valid,
          "the same plan rejects a table payload shorter than its declared arity");

    auto rejected_contract = [&](const ShaderResourceTable& mutated) {
        return !validate_spirv_descriptor_interface(
                    spirv, &mutated, 0, SpirvShaderStage::Compute, false).ok() &&
               !prosper::frontend::plan_live_compute_buffer_descriptors(
                    report.descriptors, &mutated, true).valid;
    };
    ShaderResourceTable wrong_stride = resources;
    ShaderBufferTableEntry& wrong_stride_entry =
        wrong_stride.resources[0].table_entries[1];
    wrong_stride_entry.stride = 8;
    wrong_stride_entry.vsharp[1] =
        (wrong_stride_entry.vsharp[1] & 0x0000ffffu) | (8u << 16u);
    wrong_stride_entry.vsharp[2] = wrong_stride_entry.size / 8u;
    CHECK(rejected_contract(wrong_stride),
          "one entry with a different guest stride rejects in validation and the backend plan");

    ShaderResourceTable wrong_format = resources;
    wrong_format.resources[0].table_entries[1].vsharp[3] = 22u << 12u;
    CHECK(rejected_contract(wrong_format),
          "one entry with a different guest format rejects in validation and the backend plan");

    ShaderResourceTable unsupported_control = resources;
    unsupported_control.resources[0].table_entries[1].vsharp[3] |= 1u << 19u;
    CHECK(rejected_contract(unsupported_control),
          "one entry with an unrepresented V# control rejects in validation and the backend plan");

    ShaderResourceTable unsupported_dst_sel = resources;
    unsupported_dst_sel.resources[0].table_entries[1].vsharp[3] ^= 1u;
    CHECK(rejected_contract(unsupported_dst_sel),
          "same-entry mutation: a descriptor swizzle the flattened backend cannot preserve rejects");

    prosper::gpu::ComputeItem item;
    item.spirv = spirv;
    item.user_sgprs = config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.threads_x = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_x = item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x510000u;
    item.recompile_config = config;
    item.recompile_config_available = true;

    const uint32_t expected[] = {entry0[0], entry1[0], entry2[0], 0u, 0u};
    bool every_entry_selected = true;
    for (uint32_t index = 0; index < std::size(expected); ++index) {
        result = 0xdeadbeefu;
        item.user_sgprs[8] = index;
        every_entry_selected &=
            prosper::frontend::execute_live_compute_items({item}) && result == expected[index];
    }
    CHECK(every_entry_selected,
          "the production backend binds every entry and returns zero for null or out-of-range slots");

    // Scalar control through the same backend: no table payload, one reflected descriptor, and the
    // historical one-descriptor plan remains unchanged.
    SpirvDescriptorBinding scalar_descriptor{};
    scalar_descriptor.binding = output.binding;
    scalar_descriptor.kind = SpirvDescriptorKind::StorageBuffer;
    scalar_descriptor.descriptor_count = 1;
    ShaderResourceTable scalar_resources;
    scalar_resources.resources.push_back(output);
    const auto scalar_plan = prosper::frontend::plan_live_compute_buffer_descriptors(
        {scalar_descriptor}, &scalar_resources, false);
    CHECK(scalar_plan.valid && scalar_plan.total_descriptor_count == 1 &&
              scalar_plan.bindings.size() == 1 &&
              scalar_plan.bindings[0].first_descriptor == 0 &&
              scalar_plan.bindings[0].descriptor_count == 1,
          "ordinary scalar storage buffers retain the one-descriptor backend contract");

    std::printf(failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
