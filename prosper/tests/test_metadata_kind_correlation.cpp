// Which kind of compression metadata is at an address — established by POSITIVE correlation only.
//
// These arms exist because the version they replace answered by elimination, inside a renderer lambda
// that no test reached. Two consequences, and both are pinned here:
//   * "not Float32x1, therefore DCC" classified an uncorrelated Uint32x1 raw alias as colour DCC, after
//     which its all-0xff plane authorized reading the base — the exact interpretation the
//     source-authority policy exists to withhold until the kind is KNOWN.
//   * a metadata-address match alone identified HTILE without proving the plane belonged to THIS
//     resource, so a stale retained entry whose address had been recycled answered for an unrelated one.
//
// The rule is tested here rather than through the renderer's registration, because mutating that lambda
// to answer DCC unconditionally left the forwarding test green — plumbing coverage, not rule coverage.
#include "gpu/metadata_kind_correlation.hpp"

#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// GTA V's main depth and its HTILE, as the renderer retains them.
constexpr uint64_t kDepthRead = 0x2052ac0000ull;
constexpr uint64_t kStencilRead = 0x2054aa0000ull;
constexpr uint64_t kHtile = 0x2055310000ull;
// An ordinary colour target and its DCC plane.
constexpr uint64_t kColorAddr = 0x2063380000ull;
constexpr uint64_t kColorDcc = 0x2064000000ull;

std::vector<RetainedDepthCorrelation> retained_depth() {
    return {{kDepthRead, kDepthRead, kStencilRead, kStencilRead, kHtile}};
}
std::vector<RetainedColorCorrelation> retained_color() {
    return {{kColorAddr, kColorDcc}};
}

MetadataKindRequest ask(uint64_t metadata_addr, uint64_t resource_addr,
                        DataFormat format = DataFormat::Float32, uint32_t components = 1,
                        uint32_t img_dim = 1) {
    MetadataKindRequest r;
    r.metadata_addr = metadata_addr;
    r.resource_addr = resource_addr;
    r.format = format;
    r.num_components = components;
    r.img_dim = img_dim;
    return r;
}

CompressionMetadataKind kind(const MetadataKindRequest& r) {
    return correlate_compression_metadata_kind(r, retained_depth(), retained_color());
}

}  // namespace

int main() {
    printf("== test_metadata_kind_correlation ==\n");

    // POSITIVE HTILE: the retained surface names this plane AND this resource is one of its aspects.
    CHECK(kind(ask(kHtile, kDepthRead)) == CompressionMetadataKind::Htile,
          "a retained depth surface's HTILE plane correlates for its depth base");
    CHECK(kind(ask(kHtile, kStencilRead)) == CompressionMetadataKind::Htile,
          "...and for its stencil base, which shares the same HTILE plane");

    // POSITIVE DCC: this resource registered this plane.
    CHECK(kind(ask(kColorDcc, kColorAddr, DataFormat::Unorm8, 4)) == CompressionMetadataKind::Dcc,
          "a retained colour surface's registered DCC plane correlates for that resource");

    // RESOURCE IDENTITY IS REQUIRED, both ways. A correct metadata address paired with a resource the
    // retained entry does not describe is the recycled-address case, and must not answer.
    CHECK(kind(ask(kHtile, 0x20aaaa0000ull)) == CompressionMetadataKind::Unknown,
          "an HTILE plane does not identify a resource the retained surface never names");
    CHECK(kind(ask(kColorDcc, 0x20bbbb0000ull)) == CompressionMetadataKind::Unknown,
          "a DCC plane does not identify a resource that did not register it");

    // METADATA IDENTITY IS REQUIRED. The right resource with the wrong plane proves nothing.
    CHECK(kind(ask(0x20cccc0000ull, kDepthRead)) == CompressionMetadataKind::Unknown,
          "a depth resource with an unrelated metadata address is Unknown");
    CHECK(kind(ask(0x20cccc0000ull, kColorAddr, DataFormat::Unorm8, 4)) ==
              CompressionMetadataKind::Unknown,
          "a colour resource with an unrelated metadata address is Unknown");

    // CROSS-PAIRING must not correlate: depth's plane against the colour resource and vice versa.
    CHECK(kind(ask(kHtile, kColorAddr, DataFormat::Unorm8, 4)) == CompressionMetadataKind::Unknown,
          "depth's HTILE plane does not become DCC by being asked about a colour resource");
    CHECK(kind(ask(kColorDcc, kDepthRead)) == CompressionMetadataKind::Unknown,
          "a colour DCC plane does not become HTILE by being asked about a depth resource");

    // NO ELIMINATION. This is the arm the previous implementation failed: an uncorrelated resource must
    // be Unknown WHATEVER its format, and Uint32x1 is the case that was silently classified DCC.
    {
        const DataFormat formats[] = {DataFormat::Float32, DataFormat::Uint32, DataFormat::Unorm8,
                                      DataFormat::Float16, DataFormat::Uint8};
        const uint32_t component_counts[] = {1u, 2u, 4u};
        bool all_unknown = true;
        for (DataFormat f : formats)
            for (uint32_t c : component_counts)
                if (kind(ask(0x20dddd0000ull, 0x20eeee0000ull, f, c)) !=
                    CompressionMetadataKind::Unknown)
                    all_unknown = false;
        CHECK(all_unknown,
              "an uncorrelated resource is Unknown for EVERY format/component shape, never DCC by "
              "elimination");
    }

    // Degenerate inputs correlate with nothing.
    CHECK(kind(ask(0, kDepthRead)) == CompressionMetadataKind::Unknown,
          "no metadata address correlates with nothing");
    CHECK(kind(ask(kHtile, 0)) == CompressionMetadataKind::Unknown,
          "no resource address correlates with nothing");
    CHECK(correlate_compression_metadata_kind(ask(kHtile, kDepthRead), {}, {}) ==
              CompressionMetadataKind::Unknown,
          "with no retained state at all, nothing correlates");

    // A retained entry with a zero HTILE base must not match a zero metadata address by accident.
    {
        std::vector<RetainedDepthCorrelation> no_htile = {{kDepthRead, kDepthRead, 0, 0, 0}};
        CHECK(correlate_compression_metadata_kind(ask(kHtile, kDepthRead), no_htile, {}) ==
                  CompressionMetadataKind::Unknown,
              "a retained surface with no HTILE plane correlates nothing");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
