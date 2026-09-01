// #3204: the renderer-target format compatibility table, extracted from execute_item.
//
// It was a 22-line `||` chain over LiveTargetPixelFormat -- the exact shape live_target_format.hpp
// exists to eliminate. That file's own header records what the shape costs: Dead Cells lost an HDR
// lighting target (#773) and Syberia lost its whole 3D menu scene, both because a partial mapping
// silently reported one format as another. An `||` chain answers "not compatible" for any enum
// member nobody taught it about, with no build error and no failing test.
//
// As an exhaustive switch with no `default:`, adding a member is a compile error. This test pins the
// TABLE; the compiler pins the exhaustiveness.
#include "shared/live/live_target_format.hpp"
#include <cstdio>

using prosper::gpu::LiveTargetPixelFormat;
using prosper::gpu::DataFormat;
using prosper::frontend::live_target_format_matches_declaration;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); ++fails; } \
                         else printf("  [ok]   %s\n", m); } while (0)

int main() {
    printf("== live target format match (#3204) ==\n");

    // Every member's positive case: the declaration the target actually accepts.
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::Rgba8Unorm, DataFormat::Unorm8, 4), "Rgba8Unorm <- Unorm8 x4");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::Rgba16Float, DataFormat::Float16, 4), "Rgba16Float <- Float16 x4");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::Rg16Float, DataFormat::Float16, 2), "Rg16Float <- Float16 x2");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R16Float, DataFormat::Float16, 1), "R16Float <- Float16 x1");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R11G11B10Float, DataFormat::Float10_11_11, 3), "R11G11B10Float <- Float10_11_11 x3");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R8Unorm, DataFormat::Unorm8, 1), "R8Unorm <- Unorm8 x1");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::Rg8Unorm, DataFormat::Unorm8, 2), "Rg8Unorm <- Unorm8 x2");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R32Float, DataFormat::Float32, 1), "R32Float <- Float32 x1");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::Rgba32Float, DataFormat::Float32, 4), "Rgba32Float <- Float32 x4");

    // R32Uint is the one target accepting TWO declarations: a guest may store float bits through a
    // uint view. Preserved from the chain this replaced -- dropping it would silently push those
    // bindings onto the guest-backing path.
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R32Uint, DataFormat::Uint32, 1), "R32Uint <- Uint32 x1");
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R32Uint, DataFormat::Float32, 1), "R32Uint <- Float32 x1 (float bits through a uint view)");

    // Component count is part of the match, not decoration: the same DataFormat at the wrong width
    // is a different surface.
    CHECK(!live_target_format_matches_declaration(LiveTargetPixelFormat::Rgba8Unorm, DataFormat::Unorm8, 1), "Rgba8Unorm rejects Unorm8 x1");
    CHECK(!live_target_format_matches_declaration(LiveTargetPixelFormat::R8Unorm, DataFormat::Unorm8, 4), "R8Unorm rejects Unorm8 x4");
    CHECK(!live_target_format_matches_declaration(LiveTargetPixelFormat::Rg16Float, DataFormat::Float16, 4), "Rg16Float rejects Float16 x4");

    // Cross-format rejection, so the predicate is not "any float matches any float target".
    CHECK(!live_target_format_matches_declaration(LiveTargetPixelFormat::Rgba16Float, DataFormat::Float32, 4), "Rgba16Float rejects Float32 x4");
    CHECK(!live_target_format_matches_declaration(LiveTargetPixelFormat::Rgba32Float, DataFormat::Float16, 4), "Rgba32Float rejects Float16 x4");
    CHECK(!live_target_format_matches_declaration(LiveTargetPixelFormat::R32Float, DataFormat::Uint32, 1), "R32Float rejects Uint32 x1 (only R32Uint takes both)");

    // Zero components is normalised to one, matching the `nc` the call site computed.
    CHECK(live_target_format_matches_declaration(LiveTargetPixelFormat::R8Unorm, DataFormat::Unorm8, 0), "zero components reads as one");

    printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
