#pragma once
// Per-callback setup work of the submit callback, carved out of it (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types these contexts name

namespace prosper::frontend::submit_renderer {

// ---- Diagnostic shader overrides (#3892) ----------------------------------------------------------
//
// The REFVS / TESTPS / FS_SPV / NOPS / SKIP_DRAW setup the submit callback ran before assembling
// backend draws, moved here verbatim. It returns the same locals as one value, which
// BackendDrawContext then refers to.
// The diagnostic shader/state overrides build_backend_draws applies to every draw, loaded once per
// callback by load_shader_overrides. Each member is the callback local of the same name it replaces.
// skip_draws_env names load_shader_overrides' process-lifetime static; program_skip and link_scan
// name the process-wide selectors.
struct ShaderOverrides {
    const bool refvs;
    std::vector<uint32_t> refvs_spv;
    std::vector<uint32_t> ps_override;
    bool ps_override_is_file;
    bool ps_override_is_test;
    int fs_match_mode;
    std::vector<uint32_t> fs_match;
    const char* fs_guest_addr_text;
    uint64_t fs_guest_addr;
    const bool fs_guest_addr_valid;
    const char* fs_target_addr_text;
    uint64_t fs_target_addr;
    const bool fs_target_addr_valid;
    const char* fs_target_dim_text;
    uint32_t fs_target_width;
    uint32_t fs_target_height;
    const bool fs_target_dim_valid;
    int testps_match_mode;
    std::vector<uint32_t> testps_match;
    const bool nops;
    const char*& skip_draws_env;
    prosper::gpu::DrawProgramSkipSelector& program_skip;
    const bool program_skip_armed;
    const bool program_census;
    prosper::gpu::DrawLinkScanSelector& link_scan;
    const bool link_scan_armed;
};
ShaderOverrides load_shader_overrides();

// ---- Dirty DCC fast-clear materialization (#3892) -----------------------------------------------
//
// Before a graphics span, turn each sampled retained target whose DCC metadata an ordered compute
// span dirtied into its uniform clear colour (PROSPER_DCCLOG reports each decode). Moved out of the
// submit callback verbatim.
// The submit callback's state that materialize_dirty_dcc_clears reads and writes, one reference per object.
struct DccClearContext {
    RttCache& g_rtt;
    const std::vector<prosper::gpu::DrawItem> & items;
    RenderTiming& pending_timing;
    const bool& timing_enabled;
};
void materialize_dirty_dcc_clears(DccClearContext& ctx);

} // namespace prosper::frontend::submit_renderer
