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

SampledSourceFacts compressed_with(CompressionMetadataKind kind,
                                   const std::vector<uint8_t>& metadata) {
    SampledSourceFacts f;
    f.compression_enabled = true;
    f.kind = kind;
    f.metadata = metadata.data();
    f.metadata_bytes = metadata.size();
    f.expected_metadata_bytes = metadata.size();
    return f;
}

bool is(const SampledSourceDecision& d, SampledSourceRepresentation r, SampledSourceReason why) {
    return d.representation == r && d.reason == why;
}

}  // namespace

int main() {
    printf("== test_compressed_source_authority ==\n");

    // THE DISTINCTION AN EARLIER DRAFT GOT WRONG. An imported image is authority to consume THAT
    // IMAGE; it is not permission to read the guest base. A single "may I read the base" boolean
    // answered true here and would have licensed reading a stale/encoded allocation for an imported
    // binding. The representation must be named, and it must not be GuestBase.
    {
        const auto compressed_metadata = uniform_dwords(0x40404040u, 64);
        auto imported = compressed_with(CompressionMetadataKind::Htile, compressed_metadata);
        imported.import_available = true;
        imported.import_selected = true;
        const auto d = sampled_source_decision(imported);
        CHECK(is(d, SampledSourceRepresentation::ImportedImage,
                 SampledSourceReason::RendererImageImport),
              "a selected import selects the IMAGE, whatever the metadata says");
        CHECK(!d.may_read_guest_base(),
              "a selected import is NOT permission to read the guest base");
    }

    // An import the caller will not consume is a DECLINE, not a fallback to guest bytes. This is the
    // PROSPER_NO_IMPORTED_IMAGE_GUEST_BYPASS shape: the old guard exempted the case precisely because
    // it only asked whether an import existed.
    {
        const auto compressed_metadata = uniform_dwords(0x40404040u, 64);
        auto bypassed = compressed_with(CompressionMetadataKind::Htile, compressed_metadata);
        bypassed.import_available = true;
        bypassed.import_selected = false;
        const auto d = sampled_source_decision(bypassed);
        CHECK(is(d, SampledSourceRepresentation::None, SampledSourceReason::DeclinedImportBypassed),
              "an import the caller bypasses declines, naming the bypass");
        CHECK(!d.may_read_guest_base(),
              "a bypassed import does not fall back to reading guest bytes");
    }

    // An uncompressed source reads the base by construction, and says so as its own reason.
    {
        SampledSourceFacts f;
        f.compression_enabled = false;
        const auto d = sampled_source_decision(f);
        CHECK(is(d, SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::UncompressedDescriptor) && d.may_read_guest_base(),
              "an uncompressed descriptor selects the guest base, with nothing to prove");
    }

    // FAIL CLOSED for a compressed source with nothing behind it. The production code lacked this: it
    // checked only that the base was READABLE, and readable is not authority.
    {
        SampledSourceFacts f;
        f.compression_enabled = true;   // kind Unknown, no metadata, no import, no producer
        const auto d = sampled_source_decision(f);
        CHECK(d.representation == SampledSourceRepresentation::None && !d.may_read_guest_base(),
              "a compressed source with no authority declines and forbids the base");
    }

    // An UNKNOWN kind authorizes nothing, however uniform its bytes happen to be -- tried with BOTH
    // kind-specific patterns, so this cannot pass merely because the bytes were wrong.
    {
        bool all_declined = true;
        for (uint32_t pattern : {0xffffffffu, kHtileDepthOnlyDecompressed,
                                 kHtileDepthStencilDecompressed}) {
            const auto metadata = uniform_dwords(pattern, 64);
            const auto d = sampled_source_decision(
                compressed_with(CompressionMetadataKind::Unknown, metadata));
            if (!is(d, SampledSourceRepresentation::None,
                    SampledSourceReason::DeclinedUnknownMetadataKind))
                all_declined = false;
        }
        CHECK(all_declined,
              "an aspect-unknown plane declines as unknown-kind, whatever it contains");
    }

    // DCC: all-0xff selects the base. One compressed block withdraws it, with the state reason.
    {
        const auto plain = uniform_dwords(0xffffffffu, 64);
        CHECK(is(sampled_source_decision(compressed_with(CompressionMetadataKind::Dcc, plain)),
                 SampledSourceRepresentation::GuestBase, SampledSourceReason::DccUncompressedBase),
              "all-0xff DCC metadata selects the guest base");

        auto one_block = plain;
        one_block[17] = 0x40;
        CHECK(is(sampled_source_decision(compressed_with(CompressionMetadataKind::Dcc, one_block)),
                 SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedUnsupportedMetadataState),
              "one compressed DCC block declines on metadata state");
    }

    // HTILE: only PAL's decompressed values select the base -- and the CROSS-KIND arms are the point.
    // Without them, sharing one proof between kinds would pass.
    {
        bool both = true;
        for (uint32_t v : {kHtileDepthOnlyDecompressed, kHtileDepthStencilDecompressed}) {
            const auto metadata = uniform_dwords(v, 64);
            if (!is(sampled_source_decision(compressed_with(CompressionMetadataKind::Htile, metadata)),
                    SampledSourceRepresentation::GuestBase,
                    SampledSourceReason::HtileUncompressedBase))
                both = false;
        }
        CHECK(both, "HTILE selects the base only on PAL's decompressed initial values");

        const auto dcc_pattern = uniform_dwords(0xffffffffu, 64);
        CHECK(is(sampled_source_decision(compressed_with(CompressionMetadataKind::Htile, dcc_pattern)),
                 SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedUnsupportedMetadataState),
              "DCC's all-0xff does NOT prove an HTILE plane decompressed");

        const auto htile_pattern = uniform_dwords(kHtileDepthOnlyDecompressed, 64);
        CHECK(is(sampled_source_decision(compressed_with(CompressionMetadataKind::Dcc, htile_pattern)),
                 SampledSourceRepresentation::None,
                 SampledSourceReason::DeclinedUnsupportedMetadataState),
              "HTILE's decompressed value does NOT prove a DCC plane plain");
    }

    // A window that is not exactly the expected span proves nothing. The GTA V shape got DCC sizing for
    // an HTILE plane, so a mismatched length is precisely where the caller's sizing is not trustworthy.
    {
        const auto metadata = uniform_dwords(0xffffffffu, 64);
        auto short_window = compressed_with(CompressionMetadataKind::Dcc, metadata);
        short_window.expected_metadata_bytes = metadata.size() * 2;
        CHECK(!sampled_source_decision(short_window).may_read_guest_base(),
              "a metadata window shorter than expected proves nothing");

        auto absent = compressed_with(CompressionMetadataKind::Dcc, metadata);
        absent.metadata = nullptr;
        CHECK(!sampled_source_decision(absent).may_read_guest_base(),
              "absent metadata proves nothing");
    }

    // The two remaining positive sources, each on metadata that proves nothing, so each arm can only
    // pass through the authority it names.
    {
        const auto compressed_metadata = uniform_dwords(0x40404040u, 64);

        auto materialized = compressed_with(CompressionMetadataKind::Dcc, compressed_metadata);
        materialized.exact_materialization = true;
        const auto md = sampled_source_decision(materialized);
        CHECK(is(md, SampledSourceRepresentation::MaterializedPixels,
                 SampledSourceReason::ExactDccMaterialization) && !md.may_read_guest_base(),
              "an exact materialization selects its own pixels, not the base");

        auto produced = compressed_with(CompressionMetadataKind::Htile, compressed_metadata);
        produced.ordered_producer_writeback = true;
        const auto pd = sampled_source_decision(produced);
        CHECK(is(pd, SampledSourceRepresentation::GuestBase,
                 SampledSourceReason::OrderedProducerWriteback) && pd.may_read_guest_base(),
              "an ordered producer writeback makes the base authoritative despite stale metadata");
    }

    // Names are part of the contract: a decline must be attributable in a log.
    {
        CHECK(std::string(sampled_source_reason_name(
                  SampledSourceReason::DeclinedUnknownMetadataKind)) ==
                  "declined-unknown-metadata-kind" &&
              std::string(sampled_source_representation_name(
                  SampledSourceRepresentation::ImportedImage)) == "imported-image",
              "representations and reasons report distinct stable names");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
