#include "compressed_source_authority.hpp"

#include <algorithm>

#include "tile.hpp"

namespace prosper::gpu {
namespace {

// DCC's "nothing in this block is compressed" code is 0xff per byte. Taken from the predicate this
// replaces (the former `dcc_cache_safe` scan in the live compute backend), not newly invented — what
// changed is that it now selects the SOURCE rather than only cache eligibility, that it is applied only
// when the metadata is known to BE DCC, and that its span is derived rather than supplied.
bool dcc_plane_proves_plain(const uint8_t* plane, size_t span) {
    return std::all_of(plane, plane + span, [](uint8_t value) { return value == 0xffu; });
}

uint32_t texel_bytes(DataFormat format, uint32_t num_components) {
    const uint32_t components = num_components ? num_components : 1u;
    const uint32_t bytes = data_format_bytes(format) * components;
    if (bytes) return bytes;
    // Packed formats report no per-component width but occupy one dword.
    if (format == DataFormat::Float10_11_11 || format == DataFormat::Unorm2_10_10_10) return 4u;
    return 0u;
}

// The exact span of this shape's metadata plane, or 0 when no exact footprint is derivable.
//
// Deriving it here is the whole point: an adapter that reached for the colour-DCC helper on a depth
// resource is exactly the original defect, and it cannot express that mistake through this interface.
size_t exact_metadata_span(const MetadataPlaneShape& shape) {
    switch (shape.kind) {
        case CompressionMetadataKind::Htile:
            // Only the 2D_MSAA Z_X shape has a published exact span in this tree. Single-sample HTILE
            // — GTA V's main depth — deliberately has none, so its metadata cannot authorize its base.
            if (shape.img_dim == 6u && shape.format == DataFormat::Float32 &&
                shape.num_components == 1u && shape.sample_count > 1u)
                return gfx10_htile_msaa_metadata_bytes(shape.width, shape.height, shape.tile_mode,
                                                       shape.sample_count, shape.meta_pipe_aligned);
            return 0;
        case CompressionMetadataKind::Dcc: {
            // The DCC helper's published contract is SINGLE-SAMPLE base-level SW_64KB_R_X. It takes no
            // sample count, so it cannot reject a multisample shape -- this caller must, or a 4xAA
            // colour surface receives the single-sample span and an all-0xff prefix over that wrong
            // window authorizes the base. That is the same self-consistent wrong footprint this whole
            // resolver exists to make inexpressible.
            if (shape.sample_count != 1u) return 0;
            if (shape.img_dim == 6u) return 0;   // 2D_MSAA is not a single-sample colour shape
            const uint32_t bytes_per_texel = texel_bytes(shape.format, shape.num_components);
            if (!bytes_per_texel) return 0;
            const uint32_t layers =
                shape.img_dim == 3u ? 6u : (shape.img_dim == 2u ? std::max(shape.depth, 1u) : 1u);
            return gfx10_dcc_metadata_bytes(shape.width, shape.height, layers, shape.tile_mode,
                                            bytes_per_texel, shape.meta_pipe_aligned);
        }
        case CompressionMetadataKind::Unknown:
            return 0;
    }
    return 0;
}

}  // namespace

MetadataProof resolve_metadata_proof(const MetadataPlaneShape& shape, const uint8_t* plane,
                                    size_t readable_bytes) {
    if (shape.kind == CompressionMetadataKind::Unknown) return MetadataProof::UnknownKind;

    const size_t span = exact_metadata_span(shape);
    if (!span) return MetadataProof::NoExactFootprint;
    // `readable_bytes` bounds what may be touched; it never defines the span. A plane the caller can
    // only partly read cannot be proven uniform over the whole thing.
    if (!plane || readable_bytes < span) return MetadataProof::Absent;

    switch (shape.kind) {
        case CompressionMetadataKind::Dcc:
            return dcc_plane_proves_plain(plane, span) ? MetadataProof::ProvesPlain
                                                       : MetadataProof::NotDecompressed;
        case CompressionMetadataKind::Htile:
            return gfx10_htile_metadata_is_decompressed(plane, span, span, nullptr)
                       ? MetadataProof::ProvesPlain
                       : MetadataProof::NotDecompressed;
        case CompressionMetadataKind::Unknown:
            break;
    }
    return MetadataProof::UnknownKind;
}

bool producer_writeback_covers_read(const ProducerWritebackRecord& record,
                                   const ConsumerReadRequest& read) {
    if (!record.present) return false;

    // EXACT physical identity. Anything else and the same bytes are a different image -- a 3D versus
    // arrayed view, a shifted mip, a different mip-tail placement, or a different owned backing.
    if (!(record.layout == read.layout)) return false;

    // The layout must actually establish how wide a texel is. Two unknown formats both report 0, and
    // comparing 0 == 0 would accept them as compatible -- zero means "cannot establish", not "matches".
    if (!texel_bytes(read.layout.format, read.layout.num_components)) return false;

    // Same version of the contents.
    if (record.content_version != read.content_version) return false;

    // Overflow-checked containment. `offset + size` wraps on a hostile or buggy input, and a wrapped
    // end compares as CONTAINED -- a record covering [0,64) would admit a read at UINT64_MAX-3 of 8
    // bytes, whose end computes as 4.
    if (!record.byte_size || !read.byte_size) return false;
    if (record.byte_size > UINT64_MAX - record.byte_offset) return false;
    if (read.byte_size > UINT64_MAX - read.byte_offset) return false;
    if (read.byte_offset < record.byte_offset) return false;
    if (read.byte_offset + read.byte_size > record.byte_offset + record.byte_size) return false;

    // Ordering: the consumer must come after the producer on the same submit timeline.
    if (record.submit_no != read.submit_no) return false;
    return record.command_order < read.command_order;
}

SampledSourceDecision sampled_source_decision(const SampledSourceFacts& facts) {
    // A selected import selects THE IMAGE. Checked first because it makes the base allocation's
    // encoding irrelevant: those bytes are not the source.
    if (facts.import_state == ImportState::Selected)
        return {SampledSourceRepresentation::ImportedImage, SampledSourceReason::RendererImageImport};

    // Not a compressed source: the base carries ordinary texels by construction.
    if (!facts.compression_enabled)
        return {SampledSourceRepresentation::GuestBase, SampledSourceReason::UncompressedDescriptor};

    // An exact materialization produces its own pixels, named by the materializer that made them.
    if (facts.materialized == MaterializedSource::ExactDcc)
        return {SampledSourceRepresentation::MaterializedPixels,
                SampledSourceReason::ExactDccMaterialization};
    if (facts.materialized == MaterializedSource::ExactHtile)
        return {SampledSourceRepresentation::MaterializedPixels,
                SampledSourceReason::ExactHtileMaterialization};

    // prosper's own writeback put plain texels in the base — verified against the consumer's read
    // rather than asserted, so a stale record, a different layout over the same address, or a consumer
    // that precedes the producer cannot license raw bytes.
    if (producer_writeback_covers_read(facts.producer, facts.read))
        return {SampledSourceRepresentation::GuestBase,
                SampledSourceReason::OrderedProducerWriteback};

    // Metadata may prove the base holds ordinary texels, through the proof belonging to its own kind
    // over a span derived here from the resource shape.
    const MetadataProof proof = resolve_metadata_proof(facts.metadata_shape, facts.metadata_plane,
                                                       facts.metadata_readable_bytes);
    if (proof == MetadataProof::ProvesPlain) {
        return {SampledSourceRepresentation::GuestBase,
                facts.metadata_shape.kind == CompressionMetadataKind::Htile
                    ? SampledSourceReason::HtileUncompressedBase
                    : SampledSourceReason::DccUncompressedBase};
    }

    // Only now may a bypassed import determine the outcome. It is not evidence AGAINST any of the
    // independent proofs above — an earlier draft declined here first, which let a bypassed import veto
    // an uncompressed base and every separately proven compressed source.
    if (facts.import_state == ImportState::BypassedCompatible)
        return {SampledSourceRepresentation::None, SampledSourceReason::DeclinedImportBypassed};

    switch (proof) {
        case MetadataProof::UnknownKind:
            return {SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnknownMetadataKind};
        case MetadataProof::NoExactFootprint:
            return {SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedNoExactMetadataFootprint};
        case MetadataProof::Absent:
        case MetadataProof::NotDecompressed:
            return {SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnsupportedMetadataState};
        case MetadataProof::ProvesPlain:
            break;
    }
    return {SampledSourceRepresentation::None, SampledSourceReason::DeclinedNoAuthority};
}

const char* sampled_source_representation_name(SampledSourceRepresentation representation) {
    switch (representation) {
        case SampledSourceRepresentation::None:               return "none";
        case SampledSourceRepresentation::ImportedImage:      return "imported-image";
        case SampledSourceRepresentation::MaterializedPixels: return "materialized-pixels";
        case SampledSourceRepresentation::GuestBase:          return "guest-base";
    }
    return "none";
}

const char* sampled_source_reason_name(SampledSourceReason reason) {
    switch (reason) {
        case SampledSourceReason::UncompressedDescriptor:   return "uncompressed-descriptor";
        case SampledSourceReason::RendererImageImport:       return "renderer-image-import";
        case SampledSourceReason::ExactDccMaterialization:   return "exact-dcc-materialization";
        case SampledSourceReason::ExactHtileMaterialization: return "exact-htile-materialization";
        case SampledSourceReason::DccUncompressedBase:       return "dcc-uncompressed-base";
        case SampledSourceReason::HtileUncompressedBase:     return "htile-uncompressed-base";
        case SampledSourceReason::OrderedProducerWriteback:  return "ordered-producer-writeback";
        case SampledSourceReason::DeclinedUnknownMetadataKind:
            return "declined-unknown-metadata-kind";
        case SampledSourceReason::DeclinedNoExactMetadataFootprint:
            return "declined-no-exact-metadata-footprint";
        case SampledSourceReason::DeclinedUnsupportedMetadataState:
            return "declined-unsupported-metadata-state";
        case SampledSourceReason::DeclinedImportBypassed:    return "declined-import-bypassed";
        case SampledSourceReason::DeclinedNoAuthority:       return "declined-no-authority";
    }
    return "declined-no-authority";
}

const char* metadata_proof_name(MetadataProof proof) {
    switch (proof) {
        case MetadataProof::Absent:           return "absent";
        case MetadataProof::UnknownKind:      return "unknown-kind";
        case MetadataProof::NoExactFootprint: return "no-exact-footprint";
        case MetadataProof::NotDecompressed:  return "not-decompressed";
        case MetadataProof::ProvesPlain:      return "proves-plain";
    }
    return "absent";
}

}  // namespace prosper::gpu
