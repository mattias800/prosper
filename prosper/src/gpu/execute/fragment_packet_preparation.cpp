#include "gpu/execute/fragment_packet_preparation.hpp"
#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/recompiler/raster_quad_collector.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <exception>

namespace prosper::gpu {
namespace P = prosper::agc::Pm4;
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
    const bool source_available = producing_modules_match && in.raw_matches_producing_source &&
        in.raw_code && !in.raw_code->empty() && in.source_fs && !in.source_fs->empty();
    const bool entry_available = in.entry.observed && in.entry.canonical();
    if (!source_available) gap("packet-producing-source-unavailable");
    if (!entry_available) gap("packet-entry-register-observation-unavailable");
    if (!in.entry.rsrc2_available) gap("packet-entry-ps-rsrc2-unavailable");
    if (source_available && entry_available && in.entry.rsrc2_available) {
        // CONFIDENCE: HIGH. AMD RDNA2 ISA 70648 section 3.12.2 defines the consecutive PS user
        // prefix, followed by system words. PAL's SPI_SHADER_PGM_RSRC2_PS fields independently
        // verify USER_SGPR[4:0] at bit1 and USER_SGPR_MSB at bit27. Do not copy physical UD[N]
        // into system sN, infer an M0/parameter association, or clamp invalid six-bit counts.
        const uint32_t count =
            ((in.entry.rsrc2 >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) &
                P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MASK) |
            (((in.entry.rsrc2 >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT) &
                P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_MASK) << 5);
        if (count > in.entry.user_data.size()) {
            gap("packet-entry-user-sgpr-count-out-of-range");
        } else {
            result->user_sgpr_count_available = true;
            result->user_sgpr_count = count;
            for (uint32_t reg = 0; reg < count; ++reg) {
                const uint32_t bit = uint32_t(1) << reg;
                if (in.entry.user_data_available & bit)
                    result->initial_user_sgprs.emplace_back(reg, in.entry.user_data[reg]);
                else
                    result->missing_user_sgprs |= bit;
            }
            if (result->missing_user_sgprs) gap("packet-entry-user-sgpr-words-unavailable");
        }
    }
    if (!in.launch.canonical() || !in.launch.ps_in_control_available || !in.launch.baryc_cntl_available ||
        !in.launch.input_ena_available || !in.launch.input_addr_available)
        gap("packet-entry-raster-launch-unavailable");
    if (!in.float_mode.canonical() || !in.float_flags.canonical() || !in.launch_rsrc1.canonical() ||
        !in.float_mode.available || !in.float_flags.available || !in.launch_rsrc1.available)
        gap("packet-entry-float-launch-unavailable");
    if (!in.float_transport.canonical() || !in.float_transport.explicit_nonfinite32())
        gap("packet-entry-float-transport-unavailable");
    bool scalar_demand = false;
    if (source_available && in.vgpr_requirements &&
        in.vgpr_requirements->source_words == in.raw_code.get() &&
        in.vgpr_requirements->scalar_reads.source_words == in.raw_code.get()) {
        const auto& reads = in.vgpr_requirements->scalar_reads;
        result->scalar_read_requirements =
            std::shared_ptr<const FragmentPacketScalarReadRequirements>(in.vgpr_requirements,
                                                                        &reads);
        scalar_demand = reads.has_smem;
        if (!reads.rejection.empty()) {
            result->unmet.push_back(reads.rejection);
        } else if (scalar_demand) {
            bool complete = result->user_sgpr_count_available;
            for (uint32_t index = 0; index < reads.sites.size() && complete; ++index) {
                FragmentPacketPreparation::ScalarDescriptorObservation observed;
                observed.site_index = index;
                for (uint32_t word = 0; word < 4; ++word) {
                    const uint32_t reg = reads.sites[index].entry_words[word];
                    if (reg >= result->user_sgpr_count ||
                        !(in.entry.user_data_available & (uint32_t(1) << reg))) {
                        complete = false;
                        result->unmet.push_back(
                            "packet-scalar-descriptor-user-word-unavailable:pc=" +
                            std::to_string(reads.sites[index].pc));
                        break;
                    }
                    observed.descriptor[word] = in.entry.user_data[reg];
                }
                if (complete) result->scalar_descriptors.push_back(observed);
            }
            if (!complete) {
                result->scalar_descriptors.clear();
                if (!result->user_sgpr_count_available)
                    gap("packet-scalar-descriptor-user-prefix-unavailable");
            }
        }
        if (scalar_demand) gap("packet-scalar-live-read-lease-unimplemented");
    } else {
        gap("packet-scalar-program-requirements-unavailable");
    }
    if (!in.ps_resources.observed || !in.ps_resources.table)
        gap("packet-resource-identity-unavailable");
    if (!in.ps_resources.rejection.empty()) result->unmet.push_back(in.ps_resources.rejection);
    if (in.ps_resources.table) {
        const auto& resources = in.ps_resources.table->resources;
        if (scalar_demand || !resources.empty())
            gap("packet-resource-guest-fetch-and-read-point-unproved");
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
    // User-prefix authority does not supply per-wave primitive/LDS/system values or M0. A true
    // user s16 when N>=17 still says nothing about parameter-cache ownership or interpolation.
    gap("packet-entry-system-sgprs-and-m0-unproved");
    if (!source_available || !in.vgpr_requirements ||
        in.vgpr_requirements->source_words != in.raw_code.get()) {
        gap("packet-vgpr-program-requirements-unavailable");
        gap("packet-entry-vgpr-and-mask-abi-unproved");
    } else if (!in.vgpr_requirements->rejection.empty()) {
        result->unmet.push_back(in.vgpr_requirements->rejection);
    } else {
        result->vgpr_requirements = in.vgpr_requirements;
        if (in.vgpr_requirements->possible_entry.any()) gap("packet-entry-vgpr-values-unproved");
        // A structural writer is not a definition on EXEC-off lanes. This includes the existing
        // raw EXP observation and selected inactive peers; don't demand ALL allocated scratch.
        if (!in.vgpr_requirements->reads.empty()) gap("packet-vgpr-runtime-read-validity-unproved");
        const auto& masks = in.vgpr_requirements->masks;
        if (masks.source_words != in.raw_code.get()) {
            gap("packet-mask-program-requirements-unavailable");
        } else if (!masks.rejection.empty()) {
            result->unmet.push_back(masks.rejection);
        } else {
            // Alias the exact producing ShaderCodeAnalysis owner, never reparse per draw or
            // infer state from coverage/host helpers. Unused initial words need no launch value.
            result->mask_requirements =
                std::shared_ptr<const FragmentPacketMaskRequirements>(in.vgpr_requirements, &masks);
            if (masks.demanded & kPacketInitialExec) gap("packet-entry-exec-value-unproved");
            if (masks.demanded & kPacketInitialVcc) gap("packet-entry-vcc-value-unproved");
            if (masks.demanded & kPacketInitialScc) gap("packet-entry-scc-value-unproved");
            if (masks.demanded) gap("packet-entry-mask-abi-unproved");
        }
    }
    gap("packet-guest-helper-and-coverage-unproved");
    gap("packet-logical64-composition-unproved");
    gap("packet-ordered-export-commit-unimplemented");
    return result;
}
} // namespace prosper::gpu
