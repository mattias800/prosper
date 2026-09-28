#pragma once
// The submit callback's own diagnostics, carved out of it (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types these contexts name

namespace prosper::frontend::submit_renderer {

// ---- PROSPER_SUBMITLOG / PROSPER_SUBMITLOG_DIM (#3892) ------------------------------------------
//
// Submit-index reports for aiming PROSPER_RENDER_FIRST, moved out of the submit callback verbatim.
// The submit callback's state that log_submit_index reads and writes, one reference per object.
struct SubmitLogContext {
    const std::vector<prosper::gpu::DrawItem> & items;
    const prosper::gpu::LiveRenderPhase& phase;
    int& g_this_submit;
};
void log_submit_index(SubmitLogContext& ctx);

// ---- PROSPER_RENDER_TARGET_DIM / PROSPER_RENDER_RESOURCE_DIM (#3892) ----------------------------
//
// Sets force_target when a diagnostic extent selector names this submit, so the render window
// gates let it through. Moved out of the submit callback verbatim.
// The submit callback's state that diagnostic_target_forces_render reads and writes, one reference per object.
struct DiagnosticTargetContext {
    const std::vector<prosper::gpu::DrawItem> & items;
    bool& force_target;
};
void diagnostic_target_forces_render(DiagnosticTargetContext& ctx);

// ---- PROSPER_SHADER_DUMP (#3892) ----------------------------------------------------------------
//
// Writes the first draw item's recompiled SPIR-V. Moved out of the submit callback verbatim,
// together with the PROSPER_SHADER_DUMP test that gates it.
// The submit callback's state that dump_first_item_spirv reads and writes, one reference per object.
struct ShaderDumpContext {
    const std::vector<prosper::gpu::DrawItem> & items;
};
void dump_first_item_spirv(ShaderDumpContext& ctx);

// ---- Presented-frame report and dumps (#3892) ---------------------------------------------------
//
// The per-frame failure report and the BMP / nonzero-byte dumps of the presented frame
// (PROSPER_DUMP_CONTENT, PROSPER_FRAME_DUMP_FIRST/EVERY, PROSPER_PRESENT_NZLOG). Moved out of the
// submit callback verbatim.
// The submit callback's state that dump_presented_frame reads and writes, one reference per object.
struct PresentedFrameDumpContext {
    const std::string & frame_dir;
    const bool& dump_bmps;
    uint32_t& w;
    uint32_t& h;
    bool& published_gpu;
    const std::vector<uint8_t> & px;
    int& n;
};
void dump_presented_frame(PresentedFrameDumpContext& ctx);

} // namespace prosper::frontend::submit_renderer
