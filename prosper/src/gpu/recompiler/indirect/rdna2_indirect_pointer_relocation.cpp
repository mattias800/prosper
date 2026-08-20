#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"

#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

bool marker_directory_offset(const ShaderResource& source, uint32_t record_count,
                             uint32_t& offset) {
    const uint64_t value = static_cast<uint64_t>(source.size) +
        kIndirectBufferRelocationHeaderBytes +
        static_cast<uint64_t>(record_count) * kIndirectBufferRelocationRecordBytes;
    if (value > UINT32_MAX) return false;
    offset = static_cast<uint32_t>(value);
    return true;
}

bool proof_witnesses_match(const IndirectPointerRelocationProof& proof,
                           const IndirectBufferRelocationLayout& layout,
                           const IndirectBufferRelocationInfo& info) {
    return proof.witness_words == info.witness_words &&
           proof.witness_words.size() == layout.witness_word_count;
}

bool serialized_witnesses_valid(const IndirectBufferRelocationInfo& info,
                                const IndirectBufferRelocationLayout& layout,
                                uint64_t* fingerprint = nullptr) {
    if (info.witness_words.size() != 4u ||
        info.witness_words[0] != kIndirectPointerProofSchema ||
        info.witness_words[3] !=
            (layout.tag ^ info.witness_words[0] ^
             info.witness_words[1] ^ info.witness_words[2]))
        return false;
    if (fingerprint)
        *fingerprint = static_cast<uint64_t>(info.witness_words[1]) |
            (static_cast<uint64_t>(info.witness_words[2]) << 32u);
    return true;
}

const IndirectBufferRelocationLayout* layout_for_proof(
        const IndirectPointerRelocationProof& proof) {
    switch (proof.bound_kind) {
    case IndirectPointerBoundKind::StaticFootprint:
        return proof.source_address_kind ==
                IndirectBufferRelocationRecord::SourceAddressKind::RawU64
            ? &kIndirectPointerStaticFootprintLayout : nullptr;
    case IndirectPointerBoundKind::DescriptorRange:
        return proof.source_address_kind ==
                IndirectBufferRelocationRecord::SourceAddressKind::BufferDescriptorBase48
            ? &kIndirectPointerDescriptorRangeLayout : nullptr;
    case IndirectPointerBoundKind::None:
        return nullptr;
    }
    return nullptr;
}

const IndirectBufferRelocationLayout* layout_for_serialized_resource(
        const ShaderResource& resource, IndirectBufferRelocationInfo* output = nullptr) {
    const auto& marker = resource.indirect_pointer_relocation;
    for (const IndirectBufferRelocationLayout* layout : {
             &kIndirectPointerStaticFootprintLayout,
             &kIndirectPointerDescriptorRangeLayout}) {
        if (marker.carrier_version && marker.carrier_version != layout->version) continue;
        IndirectBufferRelocationInfo info;
        if (!inspect_indirect_buffer_relocation(
                resource, resource.host_data, resource.host_data_size, *layout, info) ||
            !serialized_witnesses_valid(info, *layout))
            continue;
        if (output) *output = std::move(info);
        return layout;
    }
    return nullptr;
}

bool analyze_indirect_pointer_relocations(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        const ShaderResourceTable& resources,
        IndirectPointerRelocationProof& proof) {
    if (analyze_rdna2_static_pointer_footprint(
            code, dwords, config, resources, proof))
        return true;
    return analyze_rdna2_descriptor_pointer_range(
        code, dwords, config, resources, proof);
}

ShaderResource* unique_resource_at(ShaderResourceTable& resources, uint32_t fetch_pc) {
    ShaderResource* result = nullptr;
    for (ShaderResource& resource : resources.resources) {
        if (resource.fetch_pc != fetch_pc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

const ShaderResource* unique_resource_at(const ShaderResourceTable& resources,
                                         uint32_t fetch_pc) {
    return unique_resource_at(const_cast<ShaderResourceTable&>(resources), fetch_pc);
}

const uint8_t* complete_source_bytes(const ShaderResource& source) {
    if (source.host_data && source.host_data_size >= source.size) return source.host_data;
    return source.size <= UINT32_MAX && guest_readable(source.gpu_addr, source.size)
        ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(source.gpu_addr))
        : nullptr;
}

void clear_marker(ShaderResource& resource) {
    resource.indirect_pointer_relocation = {};
}

bool install_marker(ShaderResource& source,
                    const IndirectPointerRelocationProof& proof,
                    const IndirectBufferRelocationLayout& layout,
                    const IndirectBufferRelocationInfo& info) {
    if (!source.host_data || source.host_data_size > UINT32_MAX ||
        proof.record_count != proof.records.size() ||
        info.records.size() != proof.records.size() ||
        info.segments.size() > UINT32_MAX ||
        !proof_witnesses_match(proof, layout, info))
        return false;
    uint32_t directory_offset = 0;
    if (!marker_directory_offset(source, proof.record_count, directory_offset)) return false;
    source.indirect_pointer_relocation = {
        layout.version,
        proof.schema_version,
        static_cast<uint32_t>(source.host_data_size),
        proof.record_count,
        static_cast<uint32_t>(info.segments.size()),
        directory_offset,
        proof.fingerprint,
    };
    return true;
}

} // namespace

bool is_indirect_pointer_relocation_marker_candidate(const ShaderResource& resource) {
    const auto& marker = resource.indirect_pointer_relocation;
    return marker.carrier_version != 0u || marker.proof_schema != 0u ||
           marker.binding_bytes != 0u || marker.record_count != 0u ||
           marker.segment_count != 0u ||
           marker.segment_directory_byte_offset != 0u ||
           marker.proof_fingerprint != 0u;
}

bool is_indirect_pointer_relocation_serialized(
        const ShaderResource& resource, const uint8_t* bytes, size_t byte_count) {
    ShaderResource serialized = resource;
    serialized.host_data = const_cast<uint8_t*>(bytes);
    serialized.host_data_size = byte_count;
    return layout_for_serialized_resource(serialized) != nullptr;
}

bool is_indirect_pointer_relocation_resource(const ShaderResource& resource) {
    const auto& marker = resource.indirect_pointer_relocation;
    if (marker.proof_schema != kIndirectPointerProofSchema ||
        marker.binding_bytes != resource.host_data_size ||
        marker.record_count == 0u || !resource.host_data ||
        resource.host_data_size > UINT32_MAX)
        return false;
    IndirectBufferRelocationInfo info;
    uint64_t fingerprint = 0;
    uint32_t directory_offset = 0;
    const IndirectBufferRelocationLayout* layout =
        layout_for_serialized_resource(resource, &info);
    return layout && marker.carrier_version == layout->version &&
           serialized_witnesses_valid(info, *layout, &fingerprint) &&
           marker.record_count == info.records.size() &&
           marker.segment_count == info.segments.size() &&
           marker_directory_offset(resource, marker.record_count, directory_offset) &&
           marker.segment_directory_byte_offset == directory_offset &&
           marker.proof_fingerprint == fingerprint;
}

bool validate_rdna2_indirect_pointer_relocations(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        const ShaderResourceTable& resources,
        IndirectPointerRelocationProof* validated_proof,
        IndirectBufferRelocationInfo* validated_info) {
    if (validated_proof) *validated_proof = {};
    if (validated_info) *validated_info = {};

    IndirectPointerRelocationProof proof;
    if (!analyze_indirect_pointer_relocations(
            code, dwords, config, resources, proof))
        return false;
    const IndirectBufferRelocationLayout* layout = layout_for_proof(proof);
    if (!layout) return false;
    const ShaderResource* source = unique_resource_at(resources, proof.source_fetch_pc);
    if (!source || !is_indirect_pointer_relocation_resource(*source)) return false;
    size_t markers = 0;
    for (const ShaderResource& resource : resources.resources) {
        if (!is_indirect_pointer_relocation_marker_candidate(resource)) continue;
        if (&resource != source || !is_indirect_pointer_relocation_resource(resource))
            return false;
        ++markers;
    }
    if (markers != 1u) return false;

    IndirectBufferRelocationInfo info;
    if (!parse_indirect_buffer_relocation(
            *source, source->host_data, source->host_data_size,
            *layout, proof.records, info) ||
        !proof_witnesses_match(proof, *layout, info) ||
        source->indirect_pointer_relocation.proof_fingerprint != proof.fingerprint ||
        !current_indirect_buffer_relocation_matches(
            resources, *source, *layout, proof.records))
        return false;

    if (validated_proof) *validated_proof = std::move(proof);
    if (validated_info) *validated_info = std::move(info);
    return true;
}

bool discover_rdna2_indirect_pointer_relocations(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        ShaderResourceTable& resources) {
    for (ShaderResource& resource : resources.resources) clear_marker(resource);

    IndirectPointerRelocationProof proof;
    if (!analyze_indirect_pointer_relocations(
            code, dwords, config, resources, proof))
        return false;
    const IndirectBufferRelocationLayout* layout = layout_for_proof(proof);
    if (!layout) return false;
    ShaderResource* source = unique_resource_at(resources, proof.source_fetch_pc);
    if (!source) return false;

    IndirectBufferRelocationInfo info;
    if (parse_indirect_buffer_relocation(
            *source, source->host_data, source->host_data_size,
            *layout, proof.records, info)) {
        if (!proof_witnesses_match(proof, *layout, info) ||
            !install_marker(*source, proof, *layout, info))
            return false;
        if (validate_rdna2_indirect_pointer_relocations(
                code, dwords, config, resources))
            return true;
        clear_marker(*source);
        return false;
    }

    const uint8_t* original = complete_source_bytes(*source);
    if (!original) return false;
    std::shared_ptr<std::vector<uint8_t>> owner;
    if (!build_indirect_buffer_relocation(
            *source, original, *layout,
            proof.records, proof.witness_words, owner, info) || !owner)
        return false;

    uint8_t* prior_host_data = source->host_data;
    const uint64_t prior_host_data_size = source->host_data_size;
    source->host_data = owner->data();
    source->host_data_size = owner->size();
    resources.owned_host_data.push_back(owner);
    if (install_marker(*source, proof, *layout, info) &&
        validate_rdna2_indirect_pointer_relocations(
            code, dwords, config, resources))
        return true;

    clear_marker(*source);
    resources.owned_host_data.pop_back();
    source->host_data = prior_host_data;
    source->host_data_size = prior_host_data_size;
    return false;
}

} // namespace prosper::gpu
