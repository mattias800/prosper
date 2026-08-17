#include "compressed_source_authority.hpp"

#include <algorithm>

#include "tile.hpp"

namespace prosper::gpu {
namespace {

// DCC's "nothing in this block is compressed" code is 0xff per byte. Taken from the predicate this
// replaces (the former `dcc_cache_safe` scan in the live compute backend), not newly invented — what
// changed is that it now selects the SOURCE rather than only cache eligibility, and that it is applied
// only when the metadata is known to BE DCC.
bool dcc_metadata_proves_plain(const uint8_t* metadata, size_t metadata_bytes,
                               size_t expected_bytes) {
    if (!metadata || !expected_bytes || metadata_bytes != expected_bytes) return false;
    return std::all_of(metadata, metadata + metadata_bytes,
                       [](uint8_t value) { return value == 0xffu; });
}

}  // namespace

SampledSourceDecision sampled_source_decision(const SampledSourceFacts& facts) {
    // An import that the caller will consume selects THE IMAGE. Checked before compression, because it
    // makes the base allocation's encoding irrelevant: those bytes are not the source.
    if (facts.import_selected)
        return {SampledSourceRepresentation::ImportedImage, SampledSourceReason::RendererImageImport};

    // An import exists and the caller has chosen NOT to consume it. This is a decline, not a fallback:
    // the recovery switch that bypasses an imported image is expressing that the image should not be
    // used, which is the opposite of permission to read guest bytes in its place. The old guard
    // exempted exactly this case because it only asked whether an import existed.
    if (facts.import_available)
        return {SampledSourceRepresentation::None, SampledSourceReason::DeclinedImportBypassed};

    // Not a compressed source: the base carries ordinary texels by construction.
    if (!facts.compression_enabled)
        return {SampledSourceRepresentation::GuestBase,
                SampledSourceReason::UncompressedDescriptor};

    // An exactly supported compressed decode produces its own pixels; the compressed bytes are
    // interpreted rather than read as though they were plain.
    if (facts.exact_materialization)
        return {SampledSourceRepresentation::MaterializedPixels,
                SampledSourceReason::ExactDccMaterialization};

    // prosper's own ordered writeback put plain texels in the base. The guest's metadata may still read
    // "compressed" — that is the documented shape of the ordered storage-writeback round trip, where
    // the base IS authoritative because we wrote it. Requires exact provenance; see the field comment.
    if (facts.ordered_producer_writeback)
        return {SampledSourceRepresentation::GuestBase,
                SampledSourceReason::OrderedProducerWriteback};

    // Metadata may prove the base holds ordinary texels — but ONLY through the proof belonging to its
    // own kind. Applying DCC's all-0xff scan to an HTILE plane is how a DCC-sized window over depth
    // metadata came to look like permission, so the kinds do not share a path and Unknown has none.
    switch (facts.kind) {
        case CompressionMetadataKind::Dcc:
            if (dcc_metadata_proves_plain(facts.metadata, facts.metadata_bytes,
                                          facts.expected_metadata_bytes))
                return {SampledSourceRepresentation::GuestBase,
                        SampledSourceReason::DccUncompressedBase};
            return {SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnsupportedMetadataState};
        case CompressionMetadataKind::Htile:
            // PAL's decompressed initial values, checked uniformly across the exact expected span.
            if (gfx10_htile_metadata_is_decompressed(facts.metadata, facts.metadata_bytes,
                                                     facts.expected_metadata_bytes, nullptr))
                return {SampledSourceRepresentation::GuestBase,
                        SampledSourceReason::HtileUncompressedBase};
            return {SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnsupportedMetadataState};
        case CompressionMetadataKind::Unknown:
            // Deliberately no proof. An aspect-unknown plane cannot be shown decompressed, because
            // which bit pattern means "decompressed" depends on the kind.
            return {SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnknownMetadataKind};
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
        case SampledSourceReason::UncompressedDescriptor:           return "uncompressed-descriptor";
        case SampledSourceReason::RendererImageImport:              return "renderer-image-import";
        case SampledSourceReason::ExactDccMaterialization:          return "exact-dcc-materialization";
        case SampledSourceReason::DccUncompressedBase:              return "dcc-uncompressed-base";
        case SampledSourceReason::HtileUncompressedBase:            return "htile-uncompressed-base";
        case SampledSourceReason::OrderedProducerWriteback:         return "ordered-producer-writeback";
        case SampledSourceReason::DeclinedUnknownMetadataKind:      return "declined-unknown-metadata-kind";
        case SampledSourceReason::DeclinedUnsupportedMetadataState: return "declined-unsupported-metadata-state";
        case SampledSourceReason::DeclinedImportBypassed:           return "declined-import-bypassed";
        case SampledSourceReason::DeclinedNoAuthority:              return "declined-no-authority";
    }
    return "declined-no-authority";
}

}  // namespace prosper::gpu
