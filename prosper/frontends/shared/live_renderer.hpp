// live_renderer.hpp — register prosper's live Vulkan submit renderer.
//
// Extracted from boot_trace so boot_trace and prosper-app share ONE renderer instead of duplicating
// the ~200-line DrawItem→Vulkan compositor. On every guest GPU submit with draws, it composites the
// draws (recompiled VS/PS + detiled textures + resources) into an RGBA frame and hands it to the
// present layer (present_write_frame → present_readback). All the boot-time diagnostics
// (PROSPER_RENDER_*, PROSPER_DUMP_*, PROSPER_TESTTEX, …) are preserved and remain env-gated.
#pragma once
#include <string>

namespace prosper::frontend {

// Register the live renderer. `frame_dir` is where periodic BMP screenshots are written; pass
// `dump_bmps = false` (the frontend app) to suppress them — a windowed app presents to screen and
// doesn't want the periodic disk writes the headless runner uses for verification.
void register_live_renderer(const std::string& frame_dir = ".", bool dump_bmps = true);

} // namespace prosper::frontend
