// WHICH representation of a sampled source is authoritative — and, as a consequence, whether the
// guest base allocation may be read as ordinary texels.
//
// One pure decision over explicit facts, shared by both backends, because getting it wrong in either
// is a silent-wrong-pixels defect. The compute backend asks before guest preparation; the graphics
// backend asks before decode/detile. Previously neither asked explicitly: compute checked only that
// the base was *readable*, and graphics merely warned. A mapped base allocation whose texels actually
// live in DCC or HTILE decodes to plausible garbage, which is strictly worse than declining — it
// arrives as wrong pixels instead of a visible skip.
//
// THE DISTINCTION THAT MATTERS, and an earlier draft of this file got it wrong: an imported image is
// authority to consume THAT IMAGE. It is not permission to read the guest base. Those are different
// selected sources, so the decision names the source rather than answering a single "may I read"
// boolean — which would have licensed exactly the wrong thing for an imported binding.
//
// This is deliberately NOT part of the zero-mip proof. Compression describes how bytes are ENCODED;
// that proof is about which LOD is ADDRESSED. Conflating them turned "we cannot decode these bytes"
// into "this whole program is unsupported", dropping every other thing the program did. Compile on the
// semantic proof; select the source here, at bind time.
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

// Which kind of compression metadata a descriptor's `metadata_addr` points at.
//
// T# WORD6[21] means "metadata texture fetch is enabled". It does NOT say which kind: AMD PAL's GFX10
// SRD builder sets it for depth/stencil and colour alike, putting `GetHtile256BAddr()` in
// `meta_data_address` for a depth parent and the DCC address there otherwise — the SAME field. So the
// bit is not an aspect tag, and the T# format alone is not enough either (a Float32x1 view is not
// necessarily depth). The kind is established by correlating the metadata address with retained
// depth/HTILE state, which is why it is an INPUT here rather than something this function derives.
enum class CompressionMetadataKind : uint8_t { Unknown, Dcc, Htile };

// The source the backend is authorized to consume.
enum class SampledSourceRepresentation : uint8_t {
    None,               // declined; consume nothing
    ImportedImage,      // a retained renderer-owned image — NOT permission to read the base
    MaterializedPixels, // pixels produced by an exactly supported compressed decode
    GuestBase,          // the guest base allocation, read as ordinary texels
};

// Why that source was selected. Stable and typed so tests assert the decision rather than log text,
// and so a decline is attributable instead of sharing an anonymous skip with unrelated reasons.
enum class SampledSourceReason : uint8_t {
    UncompressedDescriptor,             // no metadata compression declared
    RendererImageImport,                // the import is the representation actually selected
    ExactDccMaterialization,            // uniform DCC fast clear, materialized exactly
    DccUncompressedBase,                // DCC metadata proves the base holds ordinary texels
    HtileUncompressedBase,              // HTILE metadata proves the same, by ITS OWN proof
    OrderedProducerWriteback,           // prosper's own ordered writeback established these bytes
    DeclinedUnknownMetadataKind,        // aspect unknown: no proof exists to apply
    DeclinedUnsupportedMetadataState,   // kind known, state not provably decompressed
    DeclinedImportBypassed,             // an import exists but the caller will not consume it
    DeclinedNoAuthority,                // compressed, and nothing establishes any source
};

struct SampledSourceDecision {
    SampledSourceRepresentation representation = SampledSourceRepresentation::None;
    SampledSourceReason reason = SampledSourceReason::DeclinedNoAuthority;

    // Only one representation licenses interpreting the guest allocation as plain texels.
    bool may_read_guest_base() const {
        return representation == SampledSourceRepresentation::GuestBase;
    }
};

struct SampledSourceFacts {
    bool compression_enabled = false;

    // The metadata plane, and how many bytes this surface's kind says it should be. A window that is
    // not exactly the expected size proves nothing: a short read can be uniform by accident, and a
    // wrong-kind footprint is precisely the case where the caller's sizing is not to be trusted.
    CompressionMetadataKind kind = CompressionMetadataKind::Unknown;
    const uint8_t* metadata = nullptr;
    size_t metadata_bytes = 0;
    size_t expected_metadata_bytes = 0;

    // An import exists for this binding at all.
    bool import_available = false;
    // ...and the caller will actually consume it. A recovery switch that has chosen to bypass the
    // imported image leaves this false, and that is a DECLINE rather than a fallback to guest bytes.
    bool import_selected = false;

    bool exact_materialization = false;

    // prosper's own ordered writeback established the base bytes. This must be EXACT provenance —
    // same base identity, range and version, in order — not "prosper wrote this address at some
    // earlier time". The ordered-submit journal is the intended input.
    bool ordered_producer_writeback = false;
};

SampledSourceDecision sampled_source_decision(const SampledSourceFacts& facts);

const char* sampled_source_representation_name(SampledSourceRepresentation representation);
const char* sampled_source_reason_name(SampledSourceReason reason);

}  // namespace prosper::gpu
