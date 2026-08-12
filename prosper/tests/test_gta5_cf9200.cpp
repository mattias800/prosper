#include "gpu/gpu_execute.hpp"
#include "gpu/rdna2_decode.hpp"
#include "gpu/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/rdna2_to_spirv.hpp"
#include "gpu/shader_resources.hpp"
#include "gta5_cf9200_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>

using namespace prosper::gpu;

namespace {

int failures = 0;
int checks = 0;
#define CHECK(condition, message) do { \
    ++checks; \
    if (!(condition)) { std::printf("FAIL: %s\n", message); ++failures; } \
} while (0)

constexpr uint64_t kRootAddress = 0x900000u;

ComputeShaderConfig exact_config() {
    ComputeShaderConfig config;
    config.user_sgprs = {
        static_cast<uint32_t>(kRootAddress),
        static_cast<uint32_t>(kRootAddress >> 32u),
    };
    config.local_x = config.local_y = config.local_z = 1u;
    config.threads_x = config.threads_y = config.threads_z = 1u;
    config.wave_size = 64u;
    return config;
}

ShaderResource exact_root(std::array<uint32_t, 56>& bytes) {
    ShaderResource resource;
    resource.cls = ResourceClass::ConstantBuffer;
    resource.format = DataFormat::Float32;
    resource.num_components = 1u;
    resource.gpu_addr = kRootAddress;
    resource.size = kGtaCf9200RootBytes;
    resource.stride = kGtaCf9200RootBytes;
    resource.fetch_pc = kGtaCf9200RootPc;
    resource.host_data = reinterpret_cast<uint8_t*>(bytes.data());
    resource.host_data_size = sizeof(bytes);
    return resource;
}

size_t marker_count(const ShaderResourceTable& table) {
    return static_cast<size_t>(std::count_if(
        table.resources.begin(), table.resources.end(),
        is_gta5_cf9200_no_backing_marker_candidate));
}

} // namespace

int main() {
    const auto& exact = prosper::test::kGta5Cf9200Program;
    ComputeShaderConfig config = exact_config();
    auto root_bytes = prosper::test::gta5_cf9200_source_and_output_null_root();
    ShaderResourceTable table;
    table.resources.push_back(exact_root(root_bytes));

    CHECK(rdna2_gta5_cf9200_shader(exact.data(), exact.size()),
          "complete production program has exact identity");
    CHECK(rdna2_gta5_cf9200_launch(exact.data(), exact.size(), config),
          "production 1x1x1 launch has exact identity");
    CHECK(discover_rdna2_gta5_cf9200_no_backing(
              exact.data(), exact.size(), config, table) && marker_count(table) == 15u,
          "source sentinel plus null output manufactures all fifteen exact-site markers");
    CHECK(rdna2_gta5_cf9200_no_backing_dispatch(
              exact.data(), exact.size(), config, table),
          "complete marker set revalidates against the retained root witness");

    assign_convention_bindings(table, 2u);
    const std::vector<uint32_t> spirv = recompile_compute(
        exact.data(), exact.size(), &table, config);
    const DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, &table, 0u, SpirvShaderStage::Compute, false);
    CHECK(!spirv.empty() && report.ok() && report.descriptors.size() == 1u,
          "translation emits the real root store while no-backing accesses declare no descriptors");

    const ShaderResource* load_marker = table.by_fetch_pc(5u);
    const ShaderResource* output_marker = table.by_fetch_pc(102u);
    CHECK(load_marker && output_marker &&
              is_proven_gta5_cf9200_no_backing(*load_marker) &&
              is_proven_gta5_cf9200_no_backing(*output_marker),
          "pc5 and pc102 carry independently recognizable root-backed markers");

    std::array<uint32_t, 135> wrong_load = exact;
    wrong_load[5] ^= 1u; // mutate the exact pc5 BUFFER_LOAD_DWORD packet fixed by this contract
    CHECK(!rdna2_gta5_cf9200_shader(wrong_load.data(), wrong_load.size()) &&
              recompile_compute(wrong_load.data(), wrong_load.size(), &table, config).empty(),
          "same-site pc5 packet mutation revokes program identity and zero-load authority");

    std::array<uint32_t, 135> wrong_output_store = exact;
    wrong_output_store[102] ^= 1u; // mutate exact pc102 BUFFER_STORE_DWORDX4 packet
    CHECK(!rdna2_gta5_cf9200_shader(
              wrong_output_store.data(), wrong_output_store.size()) &&
              recompile_compute(wrong_output_store.data(), wrong_output_store.size(),
                                &table, config).empty(),
          "same-site pc102 packet mutation revokes null-output store authority");

    ShaderResourceTable missing_marker = table;
    std::erase_if(missing_marker.resources,
                  [](const ShaderResource& resource) { return resource.fetch_pc == 29u; });
    CHECK(!rdna2_gta5_cf9200_no_backing_dispatch(
              exact.data(), exact.size(), config, missing_marker),
          "an incomplete marker set is rejected");

    ShaderResourceTable duplicate_marker = table;
    duplicate_marker.resources.push_back(*table.by_fetch_pc(5u));
    CHECK(!rdna2_gta5_cf9200_no_backing_dispatch(
              exact.data(), exact.size(), config, duplicate_marker),
          "a duplicate marker cannot widen the exact-site contract");

    auto changed_root_bytes = root_bytes;
    changed_root_bytes[50] = 0u;
    ShaderResourceTable changed_root = table;
    for (ShaderResource& resource : changed_root.resources) {
        resource.host_data = reinterpret_cast<uint8_t*>(changed_root_bytes.data());
        resource.host_data_size = sizeof(changed_root_bytes);
    }
    CHECK(!rdna2_gta5_cf9200_no_backing_dispatch(
              exact.data(), exact.size(), config, changed_root),
          "changing the source sentinel revokes the source marker set");

    changed_root_bytes = root_bytes;
    changed_root_bytes[20] = 0x1000u;
    changed_root_bytes[21] = 0x20u;
    for (ShaderResource& resource : changed_root.resources)
        resource.host_data = reinterpret_cast<uint8_t*>(changed_root_bytes.data());
    CHECK(!rdna2_gta5_cf9200_no_backing_dispatch(
              exact.data(), exact.size(), config, changed_root),
          "changing the optional output pointer revokes all eight output markers");

    ComputeShaderConfig wrong_launch = config;
    wrong_launch.local_x = 64u;
    CHECK(!rdna2_gta5_cf9200_no_backing_dispatch(
              exact.data(), exact.size(), wrong_launch, table),
          "a different launch ABI cannot reuse the contract");

    Rdna2Inst pc5 = rdna2_decode_one(exact.data() + 5u, exact.size() - 5u);
    pc5.pc = 5u;
    Rdna2Inst pc102 = rdna2_decode_one(exact.data() + 102u, exact.size() - 102u);
    pc102.pc = 102u;
    CHECK(rdna2_gta5_cf9200_no_backing_site(pc5) ==
              GtaCf9200NoBackingAccess::LoadZero &&
              rdna2_gta5_cf9200_no_backing_site(pc102) ==
              GtaCf9200NoBackingAccess::DropStore,
          "site classifier distinguishes the zero load from dropped stores");

    if (failures) {
        std::printf("== FAIL: %d == (%d assertions executed)\n", failures, checks);
        return 1;
    }
    std::printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
