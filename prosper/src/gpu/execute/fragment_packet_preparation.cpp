#include "gpu/execute/fragment_packet_preparation.hpp"
#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/recompiler/raster_quad_collector.hpp"
#include <exception>

namespace prosper::gpu {
FragmentPacketResources own_fragment_packet_resources(const ShaderResourceTable* table) {
    FragmentPacketResources result;
    try {
        // Reuse the producing compiler-input ownership transaction's bounds and pointer removal.
        // This local is not a compiler/cache record: no changing draw state enters that cache.
        FragmentCompileCase owner;
        own_fragment_compile_case_resources(owner, table);
        result.host_backing_owned.reserve(owner.backing.size());
        for (const auto& backing : owner.backing) {
            const auto index = backing.host.blob;
            result.host_backing_owned.push_back(index < owner.blobs.size() &&
                !owner.blobs[index].opaque && bool(owner.blobs[index].bytes));
        }
        result.table = std::make_shared<const ShaderResourceTable>(std::move(owner.resources));
        result.observed = true; // null producing table is observed descriptor-free state
        if (!owner.complete) result.rejection = "packet-resource-live-carrier-unproved";
    } catch (const std::exception&) {
        // No partial metadata or borrowed bytes survive a failed bounded ownership transaction.
        result = {};
        result.rejection = "packet-resource-ownership-transaction-failed";
    }
    return result;
}

std::shared_ptr<const FragmentPacketPreparation> prepare_fragment_packet_inputs(
        std::shared_ptr<const RasterQuadInputs> inputs, bool producing_modules_match) {
    auto result = std::make_shared<FragmentPacketPreparation>();
    result->inputs = std::move(inputs);
    const auto gap = [&](const char* reason) { result->unmet.emplace_back(reason); };
    if (!result->inputs) {
        gap("packet-producing-inputs-unavailable");
        return result;
    }
    const auto& in = *result->inputs;
    if (!producing_modules_match || !in.raw_matches_producing_source || !in.raw_code || in.raw_code->empty() ||
        !in.source_fs || in.source_fs->empty()) gap("packet-producing-source-unavailable");
    if (!in.entry.observed || !in.entry.canonical()) gap("packet-entry-register-observation-unavailable");
    if (!in.entry.rsrc2_available) gap("packet-entry-ps-rsrc2-unavailable");
    if (!in.launch.canonical() || !in.launch.ps_in_control_available || !in.launch.baryc_cntl_available ||
        !in.launch.input_ena_available || !in.launch.input_addr_available)
        gap("packet-entry-raster-launch-unavailable");
    if (!in.float_mode.canonical() || !in.float_flags.canonical() || !in.launch_rsrc1.canonical() ||
        !in.float_mode.available || !in.float_flags.available || !in.launch_rsrc1.available)
        gap("packet-entry-float-launch-unavailable");
    if (!in.float_transport.canonical() || !in.float_transport.explicit_nonfinite32())
        gap("packet-entry-float-transport-unavailable");
    if (!in.ps_resources.observed || !in.ps_resources.table)
        gap("packet-resource-identity-unavailable");
    if (!in.ps_resources.rejection.empty()) result->unmet.push_back(in.ps_resources.rejection);
    if (in.ps_resources.table) {
        const auto& resources = in.ps_resources.table->resources;
        if (!resources.empty()) gap("packet-resource-guest-fetch-and-read-point-unproved");
        for (const auto& resource : resources)
            if (resource.cls == ResourceClass::Texture || resource.cls == ResourceClass::StorageImage ||
                resource.cls == ResourceClass::Sampler) {
                gap("packet-resource-image-and-sampler-capabilities-unproved");
                break;
            }
        bool content_unproved = in.ps_resources.host_backing_owned.size() != resources.size();
        for (size_t i = 0; i < resources.size() && !content_unproved; ++i)
            content_unproved = resources[i].cls != ResourceClass::Sampler &&
                (resources[i].size || resources[i].cls == ResourceClass::Texture ||
                 resources[i].cls == ResourceClass::StorageImage || !resources[i].table_entries.empty()) &&
                !in.ps_resources.host_backing_owned[i];
        if (content_unproved) gap("packet-resource-content-authority-unproved");
    }
    // The availability mask is evidence, not a required/launch count. Even a fully present
    // physical window does not prove the guest SGPR ABI or justify absent-word zero filling.
    gap("packet-entry-userdata-required-set-and-mapping-unproved");
    gap("packet-entry-vgpr-and-mask-abi-unproved");
    gap("packet-guest-helper-and-coverage-unproved");
    gap("packet-logical64-composition-unproved");
    gap("packet-ordered-export-commit-unimplemented");
    return result;
}
} // namespace prosper::gpu
