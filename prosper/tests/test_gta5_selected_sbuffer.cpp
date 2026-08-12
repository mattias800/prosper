#include "gpu/gpu_execute.hpp"
#include "gpu/rdna2_gta5_compute_contracts.hpp"
#include "gpu/rdna2_to_spirv.hpp"
#include "gpu/shader_resources.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

// Exact 276-dword consumed prefix of routed GTA V 0x413ce6000. Hex keeps the fixture compact while
// still making packet mutations address the production word PCs directly.
constexpr char kProgramHex[] =
"0300a0bf0b0046d70b0c01040a826a8f000408f4700000fa00080cf4400000fa7fc08cbf1112168127ff158800000400260394beff0397be0462010023ff198800000c00220398be11039abeff039bbe046201001103a6beff03a7be046201101203a2beff03a3be0462010012821184806a07bf02039cbe03039dbe04039ebe05039fbe961da5be941da1be1a0084bf082030e00b06088088970092ff02067e000080ffff02087e000080ffff020a7e000080ff703f8cbf00203ce006000780703f8cbf0001107cf900180200060686f9001a0201060686f9001c02020606866a37eabe020088bf0c203ce0060307806a04febe760082bf000104f4980000fa120386beff0387be046201007fc08cbf931d85be7e0482be002034e00b050180703f8cbf870a0036830a0e2c8700aa7d3e0088bf8200002606004fd580fe15040c24243008004fd580fe1504181830008013ea81000108f4b80000fa8200003403006dd76a881d04ff03eabeffff7f7fff02187e0000807fff021a7e0000807f040046d700030104ff021c7e0000807f7fc08cbf00303ce003000180ff0c0836ff00000000303ce003050180ff100836ff00000000303ce003080180703f8cbf0f0054d508011604f91e067c6a8486060437eabe060088bf0c0051d5080116040d0051d509031a040e0051d50a051e046a04febeff02067e000080ffff02087e000080ffff020a7e000080ff0437eabe050088bf040054d509031a04050054d50a051e040f03067e6a04febe027efe8a260088bf8102107e8010aa7d230088bf0705d47e7e0492be6a0ea47d1d0088bf8002107e78006ab8000108f4a80000fa7fc08cbf020228f4080000d47fc08cbf002038e006000280102034e00604028088970c9289970d92713f8cbf0001107cf918180200060686f9181a0201060686f9181c0202060686f91a060203060686703f8cbff91a080204060686f91a0a02050606861204febedbff82bf0204febe00006dd711822d04703f8cbff918027c03800606041b027cf91c027c0582060688970492002030e00006058089970592006aea886a02ea88f90800020c060686f90802020d060686f90804020e060686f90a060203060686f90a100204060686f90a120205060686703f8cbf0a0048d506076d02410088bf9e0c182c98181616107078e00a000980207074e00a0809800000fdbb8102087e0860c8e00a040680703f8cbf8f0808368008aa7d330088bf002034e00a040680c1020c7ec1020e7e8018164c831616340b0046d70b032d04703f8cbf81081c30810a1830040049d504010502050049d5050105020f0046d70e0781020d0046d70c0781020e0047d7101c0e020c0047d710180e02851e1e38851a1a3804004ad5041d3e0405004ad505193604006078e00a04098028f038e00a04098038f034e00a0b0980713f8cbf0401001e010b021e020d041e07070620703f8cbf08171020091912208014aa7d060088bf002030e00a060580703f8cbf0a0048d506076d02beff82bf000081bf";

uint8_t nibble(char c) {
    return c >= '0' && c <= '9' ? static_cast<uint8_t>(c - '0')
                                : static_cast<uint8_t>(c - 'a' + 10);
}

std::vector<uint32_t> program() {
    static_assert(sizeof(kProgramHex) - 1u == 276u * 8u);
    std::vector<uint32_t> words(276u);
    auto* bytes = reinterpret_cast<uint8_t*>(words.data());
    for (size_t index = 0; index < words.size() * sizeof(uint32_t); ++index)
        bytes[index] = static_cast<uint8_t>(
            nibble(kProgramHex[index * 2u]) << 4u | nibble(kProgramHex[index * 2u + 1u]));
    return words;
}

struct Fixture {
    std::vector<uint32_t> source = std::vector<uint32_t>(2064u * 2u);
    std::array<uint8_t, 600> outer{};
    std::vector<uint8_t> target = std::vector<uint8_t>(13360u);
    ShaderResourceTable table;
    ComputeShaderConfig config;

    Fixture() {
        for (uint32_t record = 0; record < 2064u; ++record)
            source[record * 2u] = (record + 4u) << 3u;

        const uint64_t target_address = reinterpret_cast<uint64_t>(target.data());
        const std::array<uint32_t, 4> selected{
            static_cast<uint32_t>(target_address),
            static_cast<uint32_t>((target_address >> 32u) & 0xffffu) | (20u << 16u),
            668u,
            0x0004a3acu,
        };
        std::memcpy(outer.data() + 488u, selected.data(), sizeof(selected));

        ShaderResource source_resource;
        source_resource.cls = ResourceClass::ConstantBuffer;
        source_resource.format = DataFormat::Uint32;
        source_resource.num_components = 1u;
        source_resource.gpu_addr = reinterpret_cast<uint64_t>(source.data());
        source_resource.size = 16512u;
        source_resource.stride = 8u;
        source_resource.fetch_pc = 70u;
        source_resource.host_data = reinterpret_cast<uint8_t*>(source.data());
        source_resource.host_data_size = source.size() * sizeof(uint32_t);
        table.resources.push_back(source_resource);

        ShaderResource outer_resource;
        outer_resource.cls = ResourceClass::ConstantBuffer;
        outer_resource.format = DataFormat::Uint32;
        outer_resource.num_components = 1u;
        outer_resource.gpu_addr = reinterpret_cast<uint64_t>(outer.data());
        outer_resource.size = 600u;
        outer_resource.stride = 120u;
        outer_resource.fetch_pc = 153u;
        outer_resource.host_data = outer.data();
        outer_resource.host_data_size = outer.size();
        table.resources.push_back(outer_resource);

        config.user_sgprs.resize(11u);
        config.local_x = 64u;
        config.local_y = config.local_z = 1u;
        config.exact_thread_extent = true;
        config.threads_x = 2064u;
        config.threads_y = config.threads_z = 1u;
        config.wave_size = 64u;
        config.tgid_x_en = true;
        config.tidig_comp_cnt = 0u;
    }
};

size_t marker_count(const ShaderResourceTable& table) {
    size_t count = 0;
    for (const ShaderResource& resource : table.resources)
        count += is_gta5_selected_sbuffer_marker_candidate(resource);
    return count;
}

} // namespace

int main() {
    const std::vector<uint32_t> exact = program();
    Fixture valid;
    CHECK(rdna2_gta5_selected_sbuffer_shader(exact.data(), exact.size()),
          "exact routed program identity");
    CHECK(discover_rdna2_gta5_selected_sbuffer(
              exact.data(), exact.size(), valid.config, valid.table),
          "complete selector domain specializes");
    CHECK(marker_count(valid.table) == 1u &&
              valid.table.by_fetch_pc(153u) &&
              is_gta5_selected_sbuffer_descriptor(*valid.table.by_fetch_pc(153u)),
          "one marker belongs to the outer pc153 resource");
    CHECK(valid.table.by_fetch_pc(156u) && valid.table.by_fetch_pc(158u) &&
              !is_gta5_selected_sbuffer_marker_candidate(*valid.table.by_fetch_pc(156u)) &&
              !is_gta5_selected_sbuffer_marker_candidate(*valid.table.by_fetch_pc(158u)),
          "pc156/158 are separate ordinary resources");
    CHECK(rdna2_gta5_selected_sbuffer_dispatch(
              exact.data(), exact.size(), valid.config, valid.table),
          "final compiler-boundary proof accepts current witnesses");

    // Mirror the remaining ordinary/empty exact-PC resources from the routed table so the test
    // crosses the actual pc153 emitter and both pc156/158 raw-load gates, not only discovery.
    auto add_buffer = [&](uint32_t pc, uint32_t size, uint32_t stride) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = size ? DataFormat::Uint32 : DataFormat::Unknown;
        resource.num_components = size ? 1u : 0u;
        resource.gpu_addr = size ? reinterpret_cast<uint64_t>(valid.target.data()) : 0u;
        resource.size = size;
        resource.stride = stride;
        resource.fetch_pc = pc;
        valid.table.resources.push_back(resource);
    };
    add_buffer(36u, 33024u, 16u);
    add_buffer(46u, 0u, 0u);
    add_buffer(58u, 0u, 0u);
    for (uint32_t pc : {101u, 105u, 109u}) add_buffer(pc, 132096u, 64u);
    add_buffer(189u, 16508u, 4u);
    for (uint32_t pc : {212u, 214u, 253u, 255u, 257u})
        add_buffer(pc, 132032u, 64u);
    for (uint32_t pc : {218u, 224u}) add_buffer(pc, 24756u, 12u);
    add_buffer(269u, 16508u, 4u);
    assign_convention_bindings(valid.table, 2u);
    const std::vector<uint32_t> translated = recompile_compute(
        exact.data(), exact.size(), &valid.table, valid.config,
        {RecompileDiagnosticStage::Compute, 0x413ce6000u});
    CHECK(!translated.empty(),
          "production emitter crosses pc153 and both selected-buffer consumers");
    CHECK(validate_spirv_descriptor_interface(
              translated, &valid.table, 0u, SpirvShaderStage::Compute, false).ok(),
          "record-4 module matches the routed descriptor contract");

    Fixture zero_chain;
    zero_chain.table.resources.clear();
    for (const uint32_t pc : {70u, 153u}) {
        ShaderResource zero;
        zero.cls = ResourceClass::ConstantBuffer;
        zero.format = DataFormat::Unknown;
        zero.num_components = 0u;
        zero.fetch_pc = pc;
        zero_chain.table.resources.push_back(zero);
    }
    CHECK(discover_rdna2_gta5_selected_sbuffer(
              exact.data(), exact.size(), zero_chain.config, zero_chain.table) &&
              marker_count(zero_chain.table) == 1u &&
              zero_chain.table.by_fetch_pc(156u) && zero_chain.table.by_fetch_pc(158u) &&
              is_zero_record_raw_buffer(*zero_chain.table.by_fetch_pc(156u)) &&
              is_zero_record_raw_buffer(*zero_chain.table.by_fetch_pc(158u)),
          "exact zero source/outer descriptors propagate through both raw consumers");
    CHECK(rdna2_gta5_selected_sbuffer_dispatch(
              exact.data(), exact.size(), zero_chain.config, zero_chain.table),
          "final boundary accepts the complete zero-descriptor chain");
    for (const ShaderResource& resource : valid.table.resources) {
        if (resource.fetch_pc == 70u || resource.fetch_pc == 153u ||
            resource.fetch_pc == 156u || resource.fetch_pc == 158u)
            continue;
        zero_chain.table.resources.push_back(resource);
    }
    assign_convention_bindings(zero_chain.table, 2u);
    const std::vector<uint32_t> zero_translated = recompile_compute(
        exact.data(), exact.size(), &zero_chain.table, zero_chain.config,
        {RecompileDiagnosticStage::Compute, 0x413ce6000u});
    CHECK(!zero_translated.empty(),
          "production emitter propagates the zero descriptor through pc153/156/158");
    CHECK(validate_spirv_descriptor_interface(
              zero_translated, &zero_chain.table, 0u, SpirvShaderStage::Compute, false).ok(),
          "zero-chain module matches the routed descriptor contract");

    Fixture all_oob;
    for (uint32_t record = 0; record < 2064u; ++record)
        all_oob.source[record * 2u] = (record + 5u) << 3u;
    const std::array<uint32_t, 4> dead_record{
        0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
    };
    std::memcpy(all_oob.outer.data() + 488u, dead_record.data(), sizeof(dead_record));
    CHECK(discover_rdna2_gta5_selected_sbuffer(
              exact.data(), exact.size(), all_oob.config, all_oob.table) &&
              all_oob.table.by_fetch_pc(153u) &&
              all_oob.table.by_fetch_pc(153u)->selected_sbuffer_soffset ==
                  kGtaSelectedSbufferAllOobSoffset &&
              all_oob.table.by_fetch_pc(156u) && all_oob.table.by_fetch_pc(158u) &&
              is_zero_record_raw_buffer(*all_oob.table.by_fetch_pc(156u)) &&
              is_zero_record_raw_buffer(*all_oob.table.by_fetch_pc(158u)),
          "all-OOB selector domain ignores dead record-4 bytes and zeroes consumers");
    CHECK(rdna2_gta5_selected_sbuffer_dispatch(
              exact.data(), exact.size(), all_oob.config, all_oob.table),
          "final boundary rescans and accepts the all-OOB selector domain");
    all_oob.source[0] = 4u << 3u;
    CHECK(!rdna2_gta5_selected_sbuffer_dispatch(
               exact.data(), exact.size(), all_oob.config, all_oob.table),
          "same pc70 source site: final boundary rejects stale all-OOB authority");
    all_oob.source[0] = 5u << 3u;
    for (const ShaderResource& resource : valid.table.resources) {
        if (resource.fetch_pc == 70u || resource.fetch_pc == 153u ||
            resource.fetch_pc == 156u || resource.fetch_pc == 158u)
            continue;
        all_oob.table.resources.push_back(resource);
    }
    assign_convention_bindings(all_oob.table, 2u);
    const std::vector<uint32_t> all_oob_translated = recompile_compute(
        exact.data(), exact.size(), &all_oob.table, all_oob.config,
        {RecompileDiagnosticStage::Compute, 0x413ce6000u});
    CHECK(!all_oob_translated.empty(),
          "production emitter zeroes pc153/156/158 for an all-OOB domain");
    CHECK(validate_spirv_descriptor_interface(
              all_oob_translated, &all_oob.table, 0u,
              SpirvShaderStage::Compute, false).ok(),
          "all-OOB module matches the routed descriptor contract");

    Fixture mixed_zero;
    mixed_zero.table.resources.front().format = DataFormat::Unknown;
    mixed_zero.table.resources.front().num_components = 0u;
    mixed_zero.table.resources.front().gpu_addr = 0u;
    mixed_zero.table.resources.front().size = 0u;
    mixed_zero.table.resources.front().stride = 0u;
    mixed_zero.table.resources.front().host_data = nullptr;
    mixed_zero.table.resources.front().host_data_size = 0u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               exact.data(), exact.size(), mixed_zero.config, mixed_zero.table),
          "one zero descriptor cannot authorize a mixed source/outer chain");

    Fixture oob_only;
    oob_only.source[0] = 5u << 3u;
    CHECK(discover_rdna2_gta5_selected_sbuffer(
              exact.data(), exact.size(), oob_only.config, oob_only.table),
          "selector 5 is a valid wholly-OOB partition");

    Fixture partial;
    partial.source[0] = 3u << 3u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               exact.data(), exact.size(), partial.config, partial.table) &&
              marker_count(partial.table) == 0u,
          "same pc70 source site: selector 3 rejects as an unmodelled in-bounds record");

    Fixture wrapped;
    // (0x88888888 >> 3) * 120 == 0xfffffff8; pc153's +8 wraps back to byte zero.
    wrapped.source[0] = 0x88888888u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               exact.data(), exact.size(), wrapped.config, wrapped.table),
          "same pc70 source site: uint32 address wrap into the table rejects");

    Fixture component_wrapped;
    // (0x11111110 >> 3) * 120 == 0xfffffff0. The immediate +8 remains high-OOB,
    // but the third dword of the 16-byte pc153 read would wrap to byte zero.
    component_wrapped.source[0] = 0x11111110u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               exact.data(), exact.size(), component_wrapped.config,
               component_wrapped.table),
          "same pc70 source site: component address wrap into the table rejects");

    Fixture pc153_mutation;
    std::vector<uint32_t> changed_pc153 = exact;
    changed_pc153[153] ^= 1u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               changed_pc153.data(), changed_pc153.size(),
               pc153_mutation.config, pc153_mutation.table),
          "same pc153 production packet mutation rejects");

    Fixture pc156_mutation;
    std::vector<uint32_t> changed_pc156 = exact;
    changed_pc156[156] ^= 1u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               changed_pc156.data(), changed_pc156.size(),
               pc156_mutation.config, pc156_mutation.table),
          "same pc156 downstream-gate packet mutation rejects");

    ShaderResource* marker = nullptr;
    for (ShaderResource& resource : valid.table.resources)
        if (is_gta5_selected_sbuffer_marker_candidate(resource)) marker = &resource;
    CHECK(marker != nullptr, "marker remains available for stale-proof mutation");
    if (marker) marker->selected_sbuffer_words[0] ^= 4u;
    CHECK(!rdna2_gta5_selected_sbuffer_dispatch(
               exact.data(), exact.size(), valid.config, valid.table),
          "final boundary rejects stale selected descriptor words");

    Fixture launch_mutation;
    launch_mutation.config.threads_x = 2000u;
    CHECK(!discover_rdna2_gta5_selected_sbuffer(
               exact.data(), exact.size(), launch_mutation.config, launch_mutation.table),
          "non-observed launch cannot acquire the marker");

    if (failures) {
        std::fprintf(stderr, "%d selected-SBUFFER contract assertion(s) failed\n", failures);
        return 1;
    }
    std::puts("GTA V selected-SBUFFER contract tests passed");
    return 0;
}
