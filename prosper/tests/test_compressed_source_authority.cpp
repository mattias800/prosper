// May a compressed sampled source's base allocation be read as ordinary texels?
//
// Every arm here exists because the predicate this replaces got that question wrong in a way that
// produced plausible pixels rather than a visible failure. The two that matter most:
//
//   * the metadata proof must belong to the metadata's OWN KIND. A DCC all-0xff scan applied to an
//     HTILE plane is not a weaker proof, it is a different question -- and it authorized reading GTA V's
//     main depth because `gpu_capture_dcc_metadata_footprint()` hands an img_dim==1 Float32x1 shape
//     colour-DCC sizing.
//   * an imported image only counts when it is the representation ACTUALLY SELECTED. An import a
//     recovery switch has chosen to bypass is the opposite of permission to read guest bytes.
#include "gpu/compressed_source_authority.hpp"

#include "gpu/tile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// PAL's decompressed HTILE initial values (gfx9MaskRam.cpp): depth-only and depth+stencil.
constexpr uint32_t kHtileDepthOnlyDecompressed = 0xfffc000fu;
constexpr uint32_t kHtileDepthStencilDecompressed = 0xfffff3ffu;

std::vector<uint8_t> uniform_dwords(uint32_t value, size_t dwords) {
    std::vector<uint8_t> bytes(dwords * sizeof(uint32_t));
    for (size_t i = 0; i < dwords; ++i)
        std::memcpy(bytes.data() + i * sizeof(uint32_t), &value, sizeof(value));
    return bytes;
}

// GTA V's main depth: 4K, single-sample, Float32x1, img_dim==1. The shape with NO exact HTILE
// footprint in this tree -- which is precisely the shape whose metadata must never authorize its base.
MetadataPlaneShape gta_main_depth_shape() {
    MetadataPlaneShape s;
    s.kind = CompressionMetadataKind::Htile;
    s.width = 3840; s.height = 2160; s.depth = 1;
    s.img_dim = 1; s.sample_count = 1; s.tile_mode = 24;
    s.format = DataFormat::Float32; s.num_components = 1;
    return s;
}

// The one HTILE shape with a published exact span: 2D_MSAA Z_X, 4xAA, pipe-aligned, mode 24. The
// footprint helper is deliberately fail-closed outside that envelope, so a shape just outside it
// (2xAA, mode 27) yields NoExactFootprint -- which is how the first draft of this fixture was wrong.
MetadataPlaneShape msaa_depth_shape() {
    MetadataPlaneShape s;
    s.kind = CompressionMetadataKind::Htile;
    s.width = 1920; s.height = 1080; s.depth = 1;
    s.img_dim = 6; s.sample_count = 4; s.tile_mode = 24;
    s.format = DataFormat::Float32; s.num_components = 1;
    s.meta_pipe_aligned = true;
    return s;
}

MetadataPlaneShape colour_dcc_shape() {
    MetadataPlaneShape s;
    s.kind = CompressionMetadataKind::Dcc;
    s.width = 256; s.height = 128; s.depth = 1;
    s.img_dim = 1; s.sample_count = 1; s.tile_mode = 27;
    s.format = DataFormat::Unorm8; s.num_components = 4;
    return s;
}

SampledSourceFacts compressed(const MetadataPlaneShape& shape, const std::vector<uint8_t>& plane) {
    SampledSourceFacts f;
    f.compression_enabled = true;
    f.metadata_shape = shape;
    f.metadata_plane = plane.data();
    f.metadata_readable_bytes = plane.size();
    return f;
}

bool is(const SampledSourceDecision& d, SampledSourceRepresentation r, SampledSourceReason why) {
    return d.representation == r && d.reason == why;
}

}  // namespace

int main() {
    printf("== test_compressed_source_authority ==\n");

    // THE CONCRETE WRONG ROUTE the old code took, and the arm that proves the interface forbids it.
    // GTA V's main depth is single-sample HTILE, for which this tree has no exact footprint. An adapter
    // previously reached for the COLOUR-DCC footprint helper, read exactly that wrong-sized window, and
    // a uniform prefix authorized the base. Here the span is derived from the shape, so the wrong-sized
    // window is not expressible -- and the plane is filled with VALID uniform HTILE dwords so this
    // cannot pass merely because the bytes were wrong.
    {
        const auto valid_htile = uniform_dwords(kHtileDepthOnlyDecompressed, 4096);
        const auto shape = gta_main_depth_shape();
        CHECK(resolve_metadata_proof(shape, valid_htile.data(), valid_htile.size()) ==
                  MetadataProof::NoExactFootprint,
              "single-sample HTILE has no exact footprint, so its plane proves nothing");
        const auto d = sampled_source_decision(compressed(shape, valid_htile));
        CHECK(is(d, SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedNoExactMetadataFootprint) &&
              !d.may_read_guest_base(),
              "the GTA-shaped depth declines on footprint even with valid uniform HTILE dwords");
    }

    // The same shape may still be imported or exactly materialized -- the footprint gap forbids the
    // BASE, not the surface.
    {
        const auto valid_htile = uniform_dwords(kHtileDepthOnlyDecompressed, 4096);
        auto imported = compressed(gta_main_depth_shape(), valid_htile);
        imported.import_state = ImportState::Selected;
        CHECK(is(sampled_source_decision(imported), SampledSourceRepresentation::ImportedImage,
                 SampledSourceReason::RendererImageImport),
              "a footprint gap does not prevent consuming an imported image");

        auto materialized = compressed(gta_main_depth_shape(), valid_htile);
        materialized.materialized = MaterializedSource::ExactHtile;
        CHECK(is(sampled_source_decision(materialized),
                 SampledSourceRepresentation::MaterializedPixels,
                 SampledSourceReason::ExactHtileMaterialization),
              "exact HTILE materialization is reported as HTILE, not as DCC");
    }

    // An imported image is authority to consume THAT IMAGE, never permission to read the base.
    {
        auto f = compressed(colour_dcc_shape(), uniform_dwords(0x40404040u, 64));
        f.import_state = ImportState::Selected;
        const auto d = sampled_source_decision(f);
        CHECK(d.representation == SampledSourceRepresentation::ImportedImage &&
              !d.may_read_guest_base(),
              "a selected import is NOT permission to read the guest base");
    }

    // A BYPASSED import must not veto an independent proof. This was a global decline before all other
    // authorities, so a bypassed import rejected even an uncompressed base.
    {
        SampledSourceFacts uncompressed;
        uncompressed.compression_enabled = false;
        uncompressed.import_state = ImportState::BypassedCompatible;
        CHECK(is(sampled_source_decision(uncompressed), SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::UncompressedDescriptor),
              "a bypassed import does not veto an uncompressed base");

        const auto plain_dcc = uniform_dwords(0xffffffffu, 4096);
        auto proven = compressed(colour_dcc_shape(), plain_dcc);
        proven.import_state = ImportState::BypassedCompatible;
        CHECK(is(sampled_source_decision(proven), SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::DccUncompressedBase),
              "a bypassed import does not veto a proven-plain DCC base");

        auto unproven = compressed(colour_dcc_shape(), uniform_dwords(0x40404040u, 4096));
        unproven.import_state = ImportState::BypassedCompatible;
        CHECK(is(sampled_source_decision(unproven), SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedImportBypassed),
              "with no independent proof, a bypassed import names the decline");
    }

    // Kind-specific proofs over a DERIVED span, and the cross-kind arms.
    {
        const auto msaa = msaa_depth_shape();
        const size_t span = gfx10_htile_msaa_metadata_bytes(
            msaa.width, msaa.height, msaa.tile_mode, msaa.sample_count, msaa.meta_pipe_aligned);
        CHECK(span > 0, "the MSAA HTILE shape has a derivable exact span");
        // And the resolver agrees the span exists: with no plane it is Absent (span known, unreadable)
        // rather than NoExactFootprint (span unknown). Distinguishing those is the whole point.
        CHECK(resolve_metadata_proof(msaa, nullptr, 0) == MetadataProof::Absent,
              "a known span with no readable plane is absent, not a missing footprint");

        std::vector<uint8_t> htile_plain(span);
        for (size_t o = 0; o + 4 <= span; o += 4)
            std::memcpy(htile_plain.data() + o, &kHtileDepthStencilDecompressed, 4);
        CHECK(is(sampled_source_decision(compressed(msaa, htile_plain)),
                 SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::HtileUncompressedBase),
              "MSAA HTILE proves plain on PAL's decompressed value over the derived span");

        std::vector<uint8_t> dcc_pattern_over_htile(span, 0xff);
        CHECK(!sampled_source_decision(compressed(msaa, dcc_pattern_over_htile)).may_read_guest_base(),
              "DCC's all-0xff does NOT prove an HTILE plane decompressed");

        const auto dcc = colour_dcc_shape();
        std::vector<uint8_t> htile_pattern_over_dcc(8192);
        for (size_t o = 0; o + 4 <= htile_pattern_over_dcc.size(); o += 4)
            std::memcpy(htile_pattern_over_dcc.data() + o, &kHtileDepthOnlyDecompressed, 4);
        CHECK(!sampled_source_decision(compressed(dcc, htile_pattern_over_dcc)).may_read_guest_base(),
              "HTILE's decompressed value does NOT prove a DCC plane plain");
    }

    // A plane the caller can only partly read cannot be proven uniform over the whole span.
    {
        const auto dcc = colour_dcc_shape();
        const auto plain = uniform_dwords(0xffffffffu, 4096);
        auto f = compressed(dcc, plain);
        f.metadata_readable_bytes = 8;   // far less than the derived span
        CHECK(!sampled_source_decision(f).may_read_guest_base(),
              "a plane shorter than the derived span proves nothing");
        CHECK(resolve_metadata_proof(dcc, plain.data(), 8) == MetadataProof::Absent,
              "a short readable window resolves as absent, not as a proof");
    }

    // Unknown kind authorizes nothing, whatever the bytes are.
    {
        MetadataPlaneShape unknown = colour_dcc_shape();
        unknown.kind = CompressionMetadataKind::Unknown;
        bool all_declined = true;
        for (uint32_t pattern : {0xffffffffu, kHtileDepthOnlyDecompressed,
                                 kHtileDepthStencilDecompressed}) {
            const auto plane = uniform_dwords(pattern, 4096);
            if (!is(sampled_source_decision(compressed(unknown, plane)),
                    SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnknownMetadataKind))
                all_declined = false;
        }
        CHECK(all_declined, "an aspect-unknown plane declines as unknown-kind, whatever it contains");

        // ...and materialization must not launder an unknown kind into a DCC claim.
        auto claimed = compressed(unknown, uniform_dwords(0x40404040u, 64));
        claimed.materialized = MaterializedSource::ExactHtile;
        CHECK(sampled_source_decision(claimed).reason ==
                  SampledSourceReason::ExactHtileMaterialization,
              "materialization reports the materializer that ran, not a DCC default");
    }

    // Producer authority is VERIFIED against the consumer's read, not asserted.
    {
        const auto compressed_plane = uniform_dwords(0x40404040u, 4096);
        ProducerWritebackRecord rec;
        rec.present = true;
        rec.base_addr = 0x2000000000ull; rec.byte_offset = 0; rec.byte_size = 65536;
        rec.content_version = 7;
        rec.width = 256; rec.height = 64; rec.depth = 1;
        rec.tile_mode = 27; rec.row_pitch = 1024;
        rec.format = DataFormat::Unorm8; rec.num_components = 4;
        rec.submit_no = 11; rec.command_order = 3;

        ConsumerReadRequest read;
        read.base_addr = rec.base_addr; read.byte_offset = 0; read.byte_size = 65536;
        read.content_version = 7;
        read.width = 256; read.height = 64; read.depth = 1;
        read.tile_mode = 27; read.row_pitch = 1024;
        read.format = DataFormat::Unorm8; read.num_components = 4;
        read.submit_no = 11; read.command_order = 9;

        auto with = [&](ProducerWritebackRecord r, ConsumerReadRequest c) {
            auto f = compressed(colour_dcc_shape(), compressed_plane);
            f.producer = r; f.read = c;
            return sampled_source_decision(f);
        };

        CHECK(is(with(rec, read), SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::OrderedProducerWriteback),
              "an exactly matching ordered producer writeback makes the base authoritative");

        auto stale = read; stale.content_version = 8;
        CHECK(!with(rec, stale).may_read_guest_base(),
              "a different content version withdraws producer authority");
        auto other_layout = read; other_layout.tile_mode = 24;
        CHECK(!with(rec, other_layout).may_read_guest_base(),
              "a different tile mode over the same bytes is a different image");
        auto other_pitch = read; other_pitch.row_pitch = 2048;
        CHECK(!with(rec, other_pitch).may_read_guest_base(),
              "a different row pitch withdraws producer authority");
        auto wider_texel = read; wider_texel.format = DataFormat::Float32;
        CHECK(!with(rec, wider_texel).may_read_guest_base(),
              "a different texel width withdraws producer authority");
        auto beyond = read; beyond.byte_size = 65536 * 2;
        CHECK(!with(rec, beyond).may_read_guest_base(),
              "a read past the written range is not covered");
        auto before = read; before.command_order = 1;
        CHECK(!with(rec, before).may_read_guest_base(),
              "a consumer that precedes the producer is not covered");
        auto other_submit = read; other_submit.submit_no = 12;
        CHECK(!with(rec, other_submit).may_read_guest_base(),
              "a different submit timeline is not covered");
        ProducerWritebackRecord absent = rec; absent.present = false;
        CHECK(!with(absent, read).may_read_guest_base(),
              "no producer record means no producer authority");
    }

    // Compressed with nothing behind it fails closed.
    {
        SampledSourceFacts f;
        f.compression_enabled = true;
        CHECK(!sampled_source_decision(f).may_read_guest_base(),
              "a compressed source with no authority forbids the base");
    }

    // Names are part of the contract: a decline must be attributable in a log.
    {
        CHECK(std::string(sampled_source_reason_name(
                  SampledSourceReason::DeclinedNoExactMetadataFootprint)) ==
                  "declined-no-exact-metadata-footprint" &&
              std::string(metadata_proof_name(MetadataProof::NoExactFootprint)) ==
                  "no-exact-footprint",
              "representations, reasons and proofs report distinct stable names");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
