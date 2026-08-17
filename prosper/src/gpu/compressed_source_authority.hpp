// WHICH representation of a sampled source is authoritative — and, as a consequence, whether the
// guest base allocation may be read as ordinary texels.
//
// One pure decision over facts the CALLER CANNOT SELF-CERTIFY, shared by both backends, because
// getting it wrong in either is a silent-wrong-pixels defect. The compute backend asks before guest
// preparation; the graphics backend asks before decode/detile. Previously neither asked explicitly:
// compute checked only that the base was *readable*, and graphics merely warned. A mapped base
// allocation whose texels live in DCC or HTILE decodes to plausible garbage, which is strictly worse
// than declining — it arrives as wrong pixels instead of a visible skip.
//
// WHY THE INPUT TYPES LOOK PARANOID. An earlier draft took `metadata_bytes` and
// `expected_metadata_bytes` and proved only that the two were equal — both supplied by the adapter. The
// original defect survived that unchanged: call the colour-DCC footprint helper for an HTILE resource,
// read exactly that wrong-sized window, pass the same number twice, and a uniform prefix authorizes the
// base. A shared decision exists so the adapters are NOT trusted; every authority below is therefore
// either derived here from the resource shape, or minted by the verifier that can actually establish
// it. A plain `bool` authority is a way for one careless adapter assignment to license raw bytes.
//
// This is deliberately NOT part of the zero-mip proof. Compression describes how bytes are ENCODED;
// that proof is about which LOD is ADDRESSED. Conflating them turned "we cannot decode these bytes"
// into "this whole program is unsupported", dropping every other thing the program did.
#pragma once

#include <cstddef>
#include <cstdint>

#include "shader_resources.hpp"   // DataFormat

namespace prosper::gpu {

// Which kind of compression metadata a descriptor's `metadata_addr` points at.
//
// T# WORD6[21] means "metadata texture fetch is enabled". It does NOT say which kind: AMD PAL's GFX10
// SRD builder sets it for depth/stencil and colour alike, putting `GetHtile256BAddr()` in
// `meta_data_address` for a depth parent and the DCC address there otherwise — the SAME field. So the
// bit is not an aspect tag, and the T# format alone is not enough either (a Float32x1 view is not
// necessarily depth). The kind is established by correlating the metadata address with retained
// depth/HTILE state, which is why it is an input rather than something derived from the descriptor.
enum class CompressionMetadataKind : uint8_t { Unknown, Dcc, Htile };

// The physical shape a metadata plane belongs to. The resolver derives the plane's exact span from
// this, so no caller can assert a span of its own choosing.
struct MetadataPlaneShape {
    CompressionMetadataKind kind = CompressionMetadataKind::Unknown;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t img_dim = 0;
    uint32_t sample_count = 1;
    uint32_t tile_mode = 0;
    DataFormat format = DataFormat::Float32;
    uint32_t num_components = 1;
    bool meta_pipe_aligned = false;
};

// What the metadata plane establishes. Only `ProvesPlain` can contribute authority, and only the
// resolver below can produce it.
enum class MetadataProof : uint8_t {
    Absent,             // no plane supplied, or fewer bytes readable than the exact span needs
    UnknownKind,        // aspect unknown: which pattern means "decompressed" is undefined
    NoExactFootprint,   // kind known, but no exact span is derivable for THIS shape
    NotDecompressed,    // exact span read; the state is not provably decompressed
    ProvesPlain,        // exact span read; the kind's OWN proof is satisfied
};

// Derive the exact plane span from `shape` and apply the proof belonging to `shape.kind`.
//
// `readable_bytes` says how much of the plane the caller can actually read — it is a bound, never the
// span. If the derived span exceeds it, the answer is `Absent`, not a proof over a short window.
//
// `NoExactFootprint` is a real and load-bearing answer, not an omission: single-sample HTILE has no
// exact footprint in this tree today, and GTA V's main depth is exactly that shape. Such a surface may
// still be imported or exactly materialized — but its metadata must never authorize its base.
MetadataProof resolve_metadata_proof(const MetadataPlaneShape& shape, const uint8_t* plane,
                                     size_t readable_bytes);

// Whether an import exists for this binding and whether it will be consumed.
//
// A tri-state rather than two bools: the pair permitted `selected && !available`, and "available"
// conflated an exactly compatible import with a renderer lookup that was later rejected for
// device/format/extent. Only an exactly compatible import can meaningfully be *bypassed*.
enum class ImportState : uint8_t {
    Unavailable,          // no compatible import for this binding
    Selected,             // an exactly compatible import that the backend WILL consume
    BypassedCompatible,   // an exactly compatible import the caller has chosen not to consume
};

// Pixels produced by an exact compressed materialization, named by the materializer that produced
// them. Untyped, this was a boolean that reported `ExactDccMaterialization` for an HTILE plane and made
// "Unknown authorizes nothing" false through a fact claiming DCC without proving DCC. Graphics already
// has exact HTILE materialization, so this was never going to stay DCC-only.
enum class MaterializedSource : uint8_t { None, ExactDcc, ExactHtile };

// The COMPLETE physical identity of an image over a byte range — every field that changes how those
// bytes are interpreted.
//
// It mirrors the existing exact storage cache key (`ComputeImageCacheKey` in the live compute backend)
// field for field rather than keeping a convenient subset, because a partial identity is what makes
// "same bytes" mean "same image" when it does not: equal extent, tile mode and pitch still leave a 3D
// versus arrayed view, a shifted mip, a mip-tail placement, or a different replay-owned backing
// interpreting the same range differently. Adapters derive this from that key; comparison here is
// EXACT, and stays exact until some broader alias is independently proved safe.
struct SampledSourceLayout {
    uint64_t base_addr = 0;
    uintptr_t host_data = 0;          // owned-backing identity: two captures may share a guest address
    uint32_t guest_bytes = 0;
    uint32_t resource_bytes = 0;
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t img_dim = 0;
    uint32_t tile_mode = 0;
    uint32_t linear_row_pitch = 0;
    uint32_t layer_stride = 0, layer_mip_offset = 0;
    uint32_t mip_tail_offset = 0, mip_tail_bytes = 0, mip_tail_x = 0, mip_tail_y = 0;
    bool in_mip_tail = false;
    DataFormat format = DataFormat::Float32;
    uint32_t num_components = 1;

    bool operator==(const SampledSourceLayout& other) const = default;
};

// Exact producer provenance for "prosper's own writeback established these bytes".
//
// The most dangerous positive authority there is: one mistaken assignment licenses raw bytes while the
// metadata says they are encoded. So it is verified in this seam from a record, not asserted by a
// caller — and identity is more than submit ordering.
struct ProducerWritebackRecord {
    bool present = false;
    SampledSourceLayout layout;
    uint64_t content_version = 0;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
    // Ordering: the consumer must come after the producer in the same submit timeline.
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
};

// What the consumer is about to read, in the same terms, so the seam can compare rather than trust.
struct ConsumerReadRequest {
    SampledSourceLayout layout;
    uint64_t content_version = 0;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
};

// True only when the record covers this exact read, in order, over an identical physical layout whose
// texel width is actually establishable. Range arithmetic is overflow-checked: an unchecked
// offset + size wraps, and a wrapped end compares as CONTAINED, which authorizes an arbitrary read.
bool producer_writeback_covers_read(const ProducerWritebackRecord& record,
                                   const ConsumerReadRequest& read);

// The source the backend is authorized to consume.
enum class SampledSourceRepresentation : uint8_t {
    None,               // declined; consume nothing
    ImportedImage,      // a retained renderer-owned image — NOT permission to read the base
    MaterializedPixels, // pixels produced by an exact compressed materialization
    GuestBase,          // the guest base allocation, read as ordinary texels
};

// Why that source was selected. Typed so tests assert the decision rather than log text, and so a
// decline is attributable instead of sharing an anonymous skip with unrelated reasons.
enum class SampledSourceReason : uint8_t {
    UncompressedDescriptor,
    RendererImageImport,
    ExactDccMaterialization,
    ExactHtileMaterialization,
    DccUncompressedBase,
    HtileUncompressedBase,
    OrderedProducerWriteback,
    DeclinedUnknownMetadataKind,
    DeclinedNoExactMetadataFootprint,
    DeclinedUnsupportedMetadataState,
    DeclinedImportBypassed,
    DeclinedNoAuthority,
};

struct SampledSourceDecision {
    SampledSourceRepresentation representation = SampledSourceRepresentation::None;
    SampledSourceReason reason = SampledSourceReason::DeclinedNoAuthority;

    bool may_read_guest_base() const {
        return representation == SampledSourceRepresentation::GuestBase;
    }
};

struct SampledSourceFacts {
    bool compression_enabled = false;

    ImportState import_state = ImportState::Unavailable;
    MaterializedSource materialized = MaterializedSource::None;

    // The metadata plane's shape and readable extent. The proof is resolved HERE, from the shape.
    MetadataPlaneShape metadata_shape;
    const uint8_t* metadata_plane = nullptr;
    size_t metadata_readable_bytes = 0;

    ProducerWritebackRecord producer;
    ConsumerReadRequest read;
};

SampledSourceDecision sampled_source_decision(const SampledSourceFacts& facts);

const char* sampled_source_representation_name(SampledSourceRepresentation representation);
const char* sampled_source_reason_name(SampledSourceReason reason);
const char* metadata_proof_name(MetadataProof proof);

}  // namespace prosper::gpu
