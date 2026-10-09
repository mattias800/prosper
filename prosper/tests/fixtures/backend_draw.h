// Same-TU shipping draw contract, included at its original declaration site inside
// prosper::test after FrameResource/FrameBufferResource and core GPU types. Standard
// library headers are supplied by render_runner.h; no namespace, storage, or policy
// boundary changes, and no independent backend/device is created here.
#pragma once

// One draw for the multi-draw backend: recompiled VS+PS SPIR-V, its resolved fixed-function state, its
// set-tagged resources, vertex count, and instance count. render_draws_rgba preserves every draw in
// order, normally in one render pass; an attached-depth write/sample transition is the narrow case
// that requires an ordered pass boundary. render_triangle_rgba is a thin single-draw wrapper (below).
struct BackendDraw {
    std::shared_ptr<prosper::gpu::RasterQuadCollection> raster_quads;
    std::shared_ptr<const prosper::gpu::RasterQuadInputs> fragment_draw_inputs;
    std::shared_ptr<const prosper::gpu::OriginalGraphicsDrawEffects> original_graphics_effects;
    std::shared_ptr<const prosper::gpu::GraphicsOwnedWaveDraw> owned_waves;
    // A merged ES+GS NGG draw (#3135 P4): the backend expands it into its pass-through run draws
    // and dispatches its subgroup shell first (ngg_subgroup_gpu.h). Its set-0 resources are the
    // shell's guest inputs; vs/gs/vcount/instance_count/vertex_offset are not used. No live
    // producer sets it yet (P5).
    std::shared_ptr<const prosper::gpu::NggSubgroupDraw> ngg_subgroup;
    bool raster_quad_contract_modified = false;
    std::vector<uint32_t> vs, gs, fs;
    // For a mesh draw, `vs` carries a MeshEXT module instead of a vertex module. The group counts
    // are draw-time state, not pipeline identity. The backend refuses this path unless the optional
    // device feature and command entry point were both acquired.
    bool mesh_draw = false;
    std::array<uint32_t, 3> mesh_groups{1, 1, 1};
    prosper::gpu::SharedShaderWords vs_shared, fs_shared;
    uint64_t vs_identity = 0, fs_identity = 0;
    uint64_t fs_guest_addr = 0;   // diagnostic provenance only; zero for direct/override callers
    // Explicit guest-semantic lowering. Strict replay/direct callers never transform captured
    // words. Live frontends opt into per-vote certificates, independent of the game being run.
    prosper::gpu::FragmentWavePolicy fragment_wave_policy =
        prosper::gpu::FragmentWavePolicy::Strict;
    // Legacy diagnostic/test-only contracts. The live renderer never sets either flag: a reason
    // bit or an output-width proof alone is NOT authority for the new semantic transformation.
    bool allow_native_fragment_vote_width = false;
    bool allow_partial_wave_fragment = false;
    // Stable semantic draw ID from DrawItem::draw_index. Diagnostics must not use this backend
    // vector's pass-local offset: target/compute splitting can make that offset differ per pass.
    uint64_t draw_index = UINT64_MAX;
    uint64_t source_submit = 0;   // live architectural submit, zero for replay/direct callers
    // Global PM4 ordinal. Unlike draw_index this is comparable with interleaved compute operations
    // and therefore identifies which retained attachment layer is newer than a compute image.
    uint64_t command_order = 0;
    const prosper::gpu::ResolvedPipelineState* ps =
        nullptr;   // null -> triangle-list, write RGBA, no depth
    std::vector<FrameResource> R;   // textures plus compatibility/test buffers
    std::vector<FrameBufferResource> B;   // compact production storage buffers
    // Original frontend binding order. High bit selects B; the remaining bits index R or B. Empty
    // means every resource is in R, preserving the replay/test construction contract.
    std::vector<uint32_t> resource_order;
    uint32_t vcount = 3;
    uint32_t instance_count = 1;
    uint32_t first_instance = 0;   // vkCmdDraw firstInstance (an NGG replay's selected layer)
    // A layered depth replay (#3135): the layer this item's ngg_subgroup draws (DrawItem's).
    uint32_t ngg_layer_select = 0;
    int32_t vertex_offset = 0;
    // Indexed draw: 32-bit index data (the executor widens guest 16-bit indices). The live frontend
    // lends the DrawItem's already-owned words for this synchronous backend call; replay and direct
    // tests keep using the owned vector. The backend copies either form into its host-visible Vulkan
    // upload before returning, so the borrowed span never crosses submission or GPU completion.
    // Consumers must use index_words(): a borrowed value wins over `indices`, matching the shader
    // accessors above and making a deliberately conflicting owned value a useful regression control.
    std::vector<uint32_t> indices;
    std::span<const uint32_t> borrowed_indices{};
    bool has_borrowed_indices = false;

    // Same precedence — and same trap — as DrawItem (#1434): a shared value WINS, so assigning
    // `vs`/`fs` on a draw that already carries one silently keeps the ORIGINAL shader. Substitute
    // through set_vs()/set_fs(), which also clear the cache identity so the persistent pipeline
    // cache compares the new words instead of hitting the stale entry. `gs` has no shared form; if
    // one is ever added it needs the same accessor/setter pair.
    const std::vector<uint32_t>& vs_words() const { return vs_shared ? *vs_shared : vs; }
    const std::vector<uint32_t>& gs_words() const { return gs; }
    const std::vector<uint32_t>& fs_words() const { return fs_shared ? *fs_shared : fs; }
    std::span<const uint32_t> index_words() const {
        return has_borrowed_indices ? borrowed_indices : std::span<const uint32_t>(indices);
    }
    size_t index_count() const { return index_words().size(); }
    void borrow_indices(const std::vector<uint32_t>& words) {
        borrowed_indices = words;
        has_borrowed_indices = true;
    }
    void borrow_indices(std::vector<uint32_t>&&) = delete;

    void set_vs(std::vector<uint32_t> words) {
        vs = std::move(words);
        vs_shared.reset();
        vs_identity = 0;
    }
    void set_fs(std::vector<uint32_t> words) {
        fs = std::move(words);
        fs_shared.reset();
        fs_identity = 0;
    }
};

// Why `draw` may change a colour attachment, or null when it cannot: an ordinary attachment draw
// whose every slot's write mask is zero. A draw with no resolved state writes RGBA, and a draw that
// produces colour by another route (raster quads, owned waves) is never claimed. A decoded
// fast-clear VALUE (has_clear_color, ColorTarget::has_clear) is not a write: it is only the pass's
// load-op clear, taken from the caller's argument (slots 0 and 1) or from the first decoded value
// of the whole logical call (slots 2+), so every segment of a split starts from the same one.
inline const char* backend_draw_colour_writer(const BackendDraw& draw) {
    if (!draw.ps) return "no-state";
    if (draw.raster_quads) return "raster-quads";
    if (draw.owned_waves) return "owned-waves";
    if (draw.fragment_draw_inputs) return "fragment-inputs";
    const prosper::gpu::ResolvedPipelineState& ps = *draw.ps;
    if (ps.color_write_mask) return "mask-slot0";
    if (ps.color1_write_mask) return "mask-slot1";
    for (const auto& target : ps.color_targets)
        if (target.write_mask) return "mask-slot";
    return nullptr;
}
inline bool backend_draw_leaves_colour(const BackendDraw& draw) {
    return backend_draw_colour_writer(draw) == nullptr;
}
// A call none of whose draws changes colour: every physical segment of it starts from the same
// colour state the first one did and ends with it unchanged, so a split needs no colour carried.
inline bool backend_draws_leave_colour(std::span<const BackendDraw> draws) {
    return !draws.empty() && std::all_of(draws.begin(), draws.end(), backend_draw_leaves_colour);
}
