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

// The fixture OWNS its metadata plane.
//
// The previous helper took a `const std::vector<uint8_t>&` and stored `plane.data()` in the returned
// facts. Every call site that passed a temporary -- `compressed(shape, uniform_dwords(...))` -- left a
// dangling pointer that the decision then scanned, and the arms "passed" only because the freed storage
// happened to be unchanged. Owning the bytes makes that mistake inexpressible rather than merely fixed.
struct Compressed {
    std::vector<uint8_t> plane;
    SampledSourceFacts facts;

    Compressed(const MetadataPlaneShape& shape, std::vector<uint8_t> bytes)
        : plane(std::move(bytes)) {
        facts.compression_enabled = true;
        facts.metadata_shape = shape;
        facts.metadata_plane = plane.data();
        facts.metadata_readable_bytes = plane.size();
    }
    // Non-copyable: a copy would leave `facts.metadata_plane` pointing into the original's vector.
    Compressed(const Compressed&) = delete;
    Compressed& operator=(const Compressed&) = delete;
};

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
        Compressed gta(shape, valid_htile);
        const auto d = sampled_source_decision(gta.facts);
        CHECK(is(d, SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedNoExactMetadataFootprint) &&
              !d.may_read_guest_base(),
              "the GTA-shaped depth declines on footprint even with valid uniform HTILE dwords");
    }

    // The same shape may still be imported or exactly materialized -- the footprint gap forbids the
    // BASE, not the surface.
    {
        const auto valid_htile = uniform_dwords(kHtileDepthOnlyDecompressed, 4096);
        Compressed imported(gta_main_depth_shape(), valid_htile);
        imported.facts.import_state = ImportState::Selected;
        CHECK(is(sampled_source_decision(imported.facts), SampledSourceRepresentation::ImportedImage,
                 SampledSourceReason::RendererImageImport),
              "a footprint gap does not prevent consuming an imported image");

        Compressed materialized(gta_main_depth_shape(), valid_htile);
        materialized.facts.materialized = MaterializedSource::ExactHtile;
        CHECK(is(sampled_source_decision(materialized.facts),
                 SampledSourceRepresentation::MaterializedPixels,
                 SampledSourceReason::ExactHtileMaterialization),
              "exact HTILE materialization is reported as HTILE, not as DCC");
    }

    // An imported image is authority to consume THAT IMAGE, never permission to read the base.
    {
        Compressed f(colour_dcc_shape(), uniform_dwords(0x40404040u, 64));
        f.facts.import_state = ImportState::Selected;
        const auto d = sampled_source_decision(f.facts);
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

        Compressed proven(colour_dcc_shape(), uniform_dwords(0xffffffffu, 4096));
        proven.facts.import_state = ImportState::BypassedCompatible;
        CHECK(is(sampled_source_decision(proven.facts), SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::DccUncompressedBase),
              "a bypassed import does not veto a proven-plain DCC base");

        Compressed unproven(colour_dcc_shape(), uniform_dwords(0x40404040u, 4096));
        unproven.facts.import_state = ImportState::BypassedCompatible;
        CHECK(is(sampled_source_decision(unproven.facts), SampledSourceRepresentation::None,
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
        Compressed msaa_ok(msaa, htile_plain);
        CHECK(is(sampled_source_decision(msaa_ok.facts),
                 SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::HtileUncompressedBase),
              "MSAA HTILE proves plain on PAL's decompressed value over the derived span");

        Compressed dcc_over_htile(msaa, std::vector<uint8_t>(span, 0xff));
        CHECK(!sampled_source_decision(dcc_over_htile.facts).may_read_guest_base(),
              "DCC's all-0xff does NOT prove an HTILE plane decompressed");

        const auto dcc = colour_dcc_shape();
        std::vector<uint8_t> htile_pattern_over_dcc(8192);
        for (size_t o = 0; o + 4 <= htile_pattern_over_dcc.size(); o += 4)
            std::memcpy(htile_pattern_over_dcc.data() + o, &kHtileDepthOnlyDecompressed, 4);
        Compressed htile_over_dcc(dcc, htile_pattern_over_dcc);
        CHECK(!sampled_source_decision(htile_over_dcc.facts).may_read_guest_base(),
              "HTILE's decompressed value does NOT prove a DCC plane plain");
    }

    // A plane the caller can only partly read cannot be proven uniform over the whole span.
    {
        const auto dcc = colour_dcc_shape();
        const auto plain = uniform_dwords(0xffffffffu, 4096);
        Compressed f(dcc, plain);
        f.facts.metadata_readable_bytes = 8;   // far less than the derived span
        CHECK(!sampled_source_decision(f.facts).may_read_guest_base(),
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
            Compressed u(unknown, uniform_dwords(pattern, 4096));
            if (!is(sampled_source_decision(u.facts),
                    SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnknownMetadataKind))
                all_declined = false;
        }
        CHECK(all_declined, "an aspect-unknown plane declines as unknown-kind, whatever it contains");

        // ...and materialization must not launder an unknown kind into a DCC claim.
        Compressed claimed(unknown, uniform_dwords(0x40404040u, 64));
        claimed.facts.materialized = MaterializedSource::ExactHtile;
        CHECK(sampled_source_decision(claimed.facts).reason ==
                  SampledSourceReason::ExactHtileMaterialization,
              "materialization reports the materializer that ran, not a DCC default");
    }

    // Producer authority is VERIFIED against the consumer's read, not asserted -- over the COMPLETE
    // physical identity, with overflow-checked ranges.
    {
        SampledSourceLayout layout;
        layout.base_addr = 0x2000000000ull;
        layout.host_data = 0;
        layout.guest_bytes = 65536; layout.resource_bytes = 65536;
        layout.width = 256; layout.height = 64; layout.depth = 1;
        layout.img_dim = 1; layout.tile_mode = 27; layout.linear_row_pitch = 1024;
        layout.format = DataFormat::Unorm8; layout.num_components = 4;

        ProducerWritebackRecord rec;
        rec.present = true;
        rec.layout = layout;
        rec.content_version = 7;
        rec.byte_offset = 0; rec.byte_size = 65536;
        rec.submit_no = 11; rec.command_order = 3;

        ConsumerReadRequest read;
        read.layout = layout;
        read.content_version = 7;
        read.byte_offset = 0; read.byte_size = 65536;
        read.submit_no = 11; read.command_order = 9;

        Compressed fixture(colour_dcc_shape(), uniform_dwords(0x40404040u, 4096));
        auto decide = [&](const ProducerWritebackRecord& r, const ConsumerReadRequest& c) {
            SampledSourceFacts f = fixture.facts;   // plane owned by `fixture`, outlives every call
            f.producer = r; f.read = c;
            return sampled_source_decision(f);
        };

        CHECK(is(decide(rec, read), SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::OrderedProducerWriteback),
              "an exactly matching ordered producer writeback makes the base authoritative");

        // OVERFLOW. A record covering [0,64) must not admit a read whose offset+size WRAPS: the wrapped
        // end computes as 4 and compares as contained, which would authorize an arbitrary range.
        {
            ProducerWritebackRecord small = rec;
            small.byte_offset = 0; small.byte_size = 64;
            ConsumerReadRequest wrapping = read;
            wrapping.byte_offset = UINT64_MAX - 3; wrapping.byte_size = 8;
            CHECK(!decide(small, wrapping).may_read_guest_base(),
                  "a read whose offset+size overflows is not contained by any record");
            ProducerWritebackRecord wrapping_record = rec;
            wrapping_record.byte_offset = UINT64_MAX - 3; wrapping_record.byte_size = 8;
            CHECK(!decide(wrapping_record, read).may_read_guest_base(),
                  "a record whose own range overflows covers nothing");
        }

        // JOINTLY OVERSIZED RANGES. The two ranges agree with each other perfectly, so containment is
        // satisfied and nothing wraps -- but both exceed the extent the layout claims, so they describe
        // bytes outside the resource. The arm keeps record and read IDENTICAL: enlarging only the
        // producer would leave the consumer failing containment instead, and the arm would then pass
        // for a reason unrelated to the bound it is meant to test.
        {
            SampledSourceLayout small = layout;
            small.guest_bytes = 64; small.resource_bytes = 64;      // claims only 64 bytes

            ProducerWritebackRecord over = rec;
            ConsumerReadRequest over_read = read;
            over.layout = small;       over_read.layout = small;
            over.byte_offset = 0;      over.byte_size = 128;        // both exceed the claimed extent
            over_read.byte_offset = 0; over_read.byte_size = 128;
            CHECK(!decide(over, over_read).may_read_guest_base(),
                  "two mutually consistent ranges that exceed the layout extent authorize nothing");

            ProducerWritebackRecord past = rec;
            ConsumerReadRequest past_read = read;
            past.layout = small;       past_read.layout = small;
            past.byte_offset = 64;      past.byte_size = 1;         // starts at the extent boundary
            past_read.byte_offset = 64; past_read.byte_size = 1;
            CHECK(!decide(past, past_read).may_read_guest_base(),
                  "a matching pair starting at the extent boundary authorizes nothing");

            // ...and the in-bounds full-extent pair over that SAME small layout still works, so the two
            // arms above are about the bound rather than about the small layout itself.
            ProducerWritebackRecord exact = rec;
            ConsumerReadRequest exact_read = read;
            exact.layout = small;       exact_read.layout = small;
            exact.byte_offset = 0;      exact.byte_size = 64;
            exact_read.byte_offset = 0; exact_read.byte_size = 64;
            CHECK(decide(exact, exact_read).may_read_guest_base(),
                  "an exact full-extent pair over the same layout still authorizes the base");
        }

        // TWO UNKNOWN TEXEL WIDTHS must not compare equal. Zero means "cannot establish", not "matches".
        {
            SampledSourceLayout unknown_width = layout;
            unknown_width.format = static_cast<DataFormat>(0xfeedu);
            unknown_width.num_components = 0;
            ProducerWritebackRecord r = rec; r.layout = unknown_width;
            ConsumerReadRequest c = read;   c.layout = unknown_width;
            CHECK(!decide(r, c).may_read_guest_base(),
                  "two unestablishable texel widths are not a compatible layout");
        }

        // One field of the COMPLETE identity at a time. Each must withdraw authority on its own.
        struct Arm { const char* what; SampledSourceLayout layout; };
        std::vector<Arm> arms;
        auto arm = [&](const char* what, auto mutate) {
            SampledSourceLayout m = layout; mutate(m); arms.push_back({what, m});
        };
        arm("base address",   [](SampledSourceLayout& l){ l.base_addr += 0x1000; });
        arm("owned backing",  [](SampledSourceLayout& l){ l.host_data = 0xdead; });
        arm("guest bytes",    [](SampledSourceLayout& l){ l.guest_bytes /= 2; });
        arm("resource bytes", [](SampledSourceLayout& l){ l.resource_bytes /= 2; });
        arm("width",          [](SampledSourceLayout& l){ l.width /= 2; });
        arm("height",         [](SampledSourceLayout& l){ l.height /= 2; });
        arm("depth",          [](SampledSourceLayout& l){ l.depth += 1; });
        arm("img_dim",        [](SampledSourceLayout& l){ l.img_dim = 2; });
        arm("tile mode",      [](SampledSourceLayout& l){ l.tile_mode = 24; });
        arm("row pitch",      [](SampledSourceLayout& l){ l.linear_row_pitch *= 2; });
        arm("layer stride",   [](SampledSourceLayout& l){ l.layer_stride = 4096; });
        arm("layer mip off",  [](SampledSourceLayout& l){ l.layer_mip_offset = 256; });
        arm("mip tail off",   [](SampledSourceLayout& l){ l.mip_tail_offset = 512; });
        arm("mip tail bytes", [](SampledSourceLayout& l){ l.mip_tail_bytes = 512; });
        arm("mip tail x",     [](SampledSourceLayout& l){ l.mip_tail_x = 8; });
        arm("mip tail y",     [](SampledSourceLayout& l){ l.mip_tail_y = 8; });
        arm("in mip tail",    [](SampledSourceLayout& l){ l.in_mip_tail = true; });
        arm("format",         [](SampledSourceLayout& l){ l.format = DataFormat::Float32; });
        arm("components",     [](SampledSourceLayout& l){ l.num_components = 2; });

        bool every_field_matters = true;
        for (const auto& a : arms) {
            ConsumerReadRequest c = read; c.layout = a.layout;
            if (decide(rec, c).may_read_guest_base()) {
                printf("  [FAIL] producer authority survived a changed %s\n", a.what);
                every_field_matters = false;
            }
        }
        CHECK(every_field_matters,
              "every field of the physical identity withdraws producer authority on its own");

        ConsumerReadRequest stale = read; stale.content_version = 8;
        CHECK(!decide(rec, stale).may_read_guest_base(),
              "a different content version withdraws producer authority");
        ConsumerReadRequest beyond = read; beyond.byte_size = 65536 * 2;
        CHECK(!decide(rec, beyond).may_read_guest_base(),
              "a read past the written range is not covered");
        ConsumerReadRequest before = read; before.command_order = 1;
        CHECK(!decide(rec, before).may_read_guest_base(),
              "a consumer that precedes the producer is not covered");
        ConsumerReadRequest other_submit = read; other_submit.submit_no = 12;
        CHECK(!decide(rec, other_submit).may_read_guest_base(),
              "a different submit timeline is not covered");
        ProducerWritebackRecord absent = rec; absent.present = false;
        CHECK(!decide(absent, read).may_read_guest_base(),
              "no producer record means no producer authority");
    }

    // A MULTISAMPLE DCC shape has no derivable span: the helper's contract is single-sample, and it
    // takes no sample count, so it cannot reject one itself. Only the sample count differs from the
    // known-good shape above.
    {
        MetadataPlaneShape msaa_colour = colour_dcc_shape();
        msaa_colour.sample_count = 4;
        const auto plain = uniform_dwords(0xffffffffu, 8192);
        CHECK(resolve_metadata_proof(msaa_colour, plain.data(), plain.size()) ==
                  MetadataProof::NoExactFootprint,
              "a multisample DCC shape has no derivable span");
        Compressed f(msaa_colour, plain);
        CHECK(is(sampled_source_decision(f.facts), SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedNoExactMetadataFootprint),
              "a multisample DCC shape declines on footprint, not on an all-0xff prefix");

        // ...and the single-sample sibling still proves plain, so this is the sample count alone.
        Compressed ok(colour_dcc_shape(), uniform_dwords(0xffffffffu, 8192));
        CHECK(sampled_source_decision(ok.facts).may_read_guest_base(),
              "the same shape at one sample still proves plain");
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
