// load_shader_overrides, materialize_dirty_dcc_clears -- see callback_prelude.hpp. Moved verbatim out of
// live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/callback_prelude.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"
#include "shared/live/submit_renderer/mrt_slots.hpp"   // color_binding

namespace prosper::frontend::submit_renderer {

ShaderOverrides load_shader_overrides() {
    // Diagnostic shader/state overrides (computed once, applied to EVERY draw item):
    //   REFVS  -> a known-good fullscreen-triangle VS (isolates the game's real VS).
    //   TESTPS -> a solid-magenta PS (isolates VS geometry from PS shading). The optional
    //             TESTPS_MATCH file restricts it to one exact recompiled guest PS.
    //   FS_SPV -> a caller-supplied PS SPIR-V (e.g. a UV visualizer).
    //   NOPS   -> bypass the resolved pipeline state (default state).
    #include "refvs.inc"
    const bool refvs = PROSPER_ENV_VALUE("PROSPER_RENDER_REFVS");
    std::vector<uint32_t> refvs_spv(kRefVs, kRefVs + sizeof(kRefVs) / 4);
    std::vector<uint32_t> ps_override;
    bool ps_override_is_file = false;   // true only for a valid PROSPER_FS_SPV *file* override
    bool ps_override_is_test = false;
    if (PROSPER_ENV_ON("PROSPER_RENDER_TESTPS")) {
        static const uint32_t kMagentaPs[] = {   // v0=1.0(R) v1=0.0(G) v2=1.0(B) v3=1.0(A); exp mrt0; endpgm
            0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
        ps_override = prosper::gpu::recompile_fragment(kMagentaPs, sizeof(kMagentaPs) / 4, nullptr);
        ps_override_is_test = true;
    }
    // Validated SPIR-V file load: require a complete read of a word-aligned file >= 20 bytes
    // (5 words: the minimum SPIR-V header). Validate size BEFORE allocating so a failed ftell
    // (-1 -> huge size_t) cannot trigger a wild allocation, and reject non-word-aligned files
    // rather than silently dropping trailing bytes. Returns false (out untouched) on any failure.
    auto load_spv_file = [](const char* path, std::vector<uint32_t>& out) -> bool {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        bool ok = false;
        if (fseek(f, 0, SEEK_END) == 0) {
            long sz = ftell(f);
            if (sz >= 20 && (sz % 4) == 0 && fseek(f, 0, SEEK_SET) == 0) {
                std::vector<uint32_t> m(static_cast<size_t>(sz) / 4);
                if (fread(m.data(), 4, m.size(), f) == m.size()) { out = std::move(m); ok = true; }
            }
        }
        fclose(f);
        return ok;
    };
    if (const char* fsp = PROSPER_ENV_VALUE("PROSPER_FS_SPV")) {
        std::vector<uint32_t> m;
        if (load_spv_file(fsp, m)) { ps_override = std::move(m); ps_override_is_file = true; }
        else fprintf(stderr, "[fs-spv] PROSPER_FS_SPV='%s' invalid/unreadable -> no file override\n", fsp);
    }
    // PROSPER_FS_SPV_MATCH=<file>: restrict the PROSPER_FS_SPV *file* override to draws whose
    // recompiled fragment SPIR-V EXACTLY equals this file (a per-draw A/B substitution that does
    // not touch draws with a different descriptor contract). FAILS CLOSED: if requested but the
    // file is missing/short/unaligned/unreadable, NO file override is applied — never a silent
    // global fallback (that would recreate the exact hazard this gate exists to prevent). Does
    // NOT gate PROSPER_RENDER_TESTPS, which stays global by design.
    // mode: 0 = not requested (legacy global file override), 1 = loaded+valid (exact match only),
    //       2 = requested but invalid (file override disabled).
    int fs_match_mode = 0;
    std::vector<uint32_t> fs_match;
    if (const char* mp = PROSPER_ENV_VALUE("PROSPER_FS_SPV_MATCH")) {
        fs_match_mode = load_spv_file(mp, fs_match) ? 1 : 2;
        if (fs_match_mode == 2)
            fprintf(stderr, "[fs-match] PROSPER_FS_SPV_MATCH='%s' invalid/unreadable -> applying NO "
                    "fragment file override (fail closed)\n", mp);
    }
    // Optional second selector for a live exact-program experiment. A matching SPIR-V
    // module can be reused by unrelated draws, so a shader-only A/B is not necessarily
    // an exact draw substitution. Invalid input fails closed.
    const char* fs_guest_addr_text = PROSPER_ENV_VALUE("PROSPER_FS_SPV_GUEST_ADDR");
    uint64_t fs_guest_addr = 0;
    const bool fs_guest_addr_valid =
        parse_diagnostic_address(fs_guest_addr_text, fs_guest_addr);
    if (!fs_guest_addr_valid) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set())
            std::fprintf(stderr,
                         "[fs-match] PROSPER_FS_SPV_GUEST_ADDR invalid -> no file override\n");
    }
    const char* fs_target_addr_text = PROSPER_ENV_VALUE("PROSPER_FS_SPV_TARGET_ADDR");
    uint64_t fs_target_addr = 0;
    const bool fs_target_addr_valid =
        parse_diagnostic_address(fs_target_addr_text, fs_target_addr);
    if (!fs_target_addr_valid) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set())
            std::fprintf(stderr,
                         "[fs-match] PROSPER_FS_SPV_TARGET_ADDR invalid -> no file override\n");
    }
    const char* fs_target_dim_text = PROSPER_ENV_VALUE("PROSPER_FS_SPV_TARGET_DIM");
    uint32_t fs_target_width = 0, fs_target_height = 0;
    const bool fs_target_dim_valid =
        parse_diagnostic_extent(fs_target_dim_text, fs_target_width, fs_target_height);
    if (!fs_target_dim_valid) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set())
            std::fprintf(stderr,
                         "[fs-match] PROSPER_FS_SPV_TARGET_DIM invalid -> no file override\n");
    }
    // PROSPER_RENDER_TESTPS_MATCH=<file> is the geometry half of a per-shader A/B test: replace
    // only that exact guest PS with the known solid output while retaining its real VS, indices,
    // viewport, depth and raster state. As with FS_SPV_MATCH, a bad path fails closed.
    int testps_match_mode = 0;
    std::vector<uint32_t> testps_match;
    if (const char* mp = PROSPER_ENV_VALUE("PROSPER_RENDER_TESTPS_MATCH")) {
        testps_match_mode = load_spv_file(mp, testps_match) ? 1 : 2;
        if (testps_match_mode == 2)
            fprintf(stderr, "[testps-match] PROSPER_RENDER_TESTPS_MATCH='%s' invalid/unreadable -> "
                    "applying NO test fragment override (fail closed)\n", mp);
    }
    const bool nops = PROSPER_ENV_VALUE("PROSPER_RENDER_NOPS");
    // Assemble backend draws for a subset of the submit's items — one BackendDraw per realized
    // DrawItem with its own resources + fixed-function state (or the diagnostic overrides above).
    // build_R reads the CURRENT g_rtt, so calling this AFTER an earlier target-group has been
    // rendered+stored lets the later group sample that group's pixels (a HIT, not empty memory).
    // PROSPER_SKIP_DRAW="N[,N...]" (diagnostic): drop these semantic draw_index values from
    // every pass — isolate whether a specific draw (e.g. a suspected opaque UI backdrop that
    // hides the composited world) is what corrupts the frame, without touching any state.
    static const char* skip_draws_env = getenv("PROSPER_SKIP_DRAW");
    // PROSPER_SKIP_DRAW_PROGRAM / PROSPER_DRAW_PROGRAM_CENSUS: decline draws by shader
    // PROGRAM identity, and enumerate the programs a title draws with. Both are process
    // singletons configured once from the environment (see draw_program_skip.hpp for the
    // contract and the four limits a reader of a skipped run cannot see in the output).
    // Hoisted here for the same reason descriptor_validate_mode is: the accessor is one
    // function-local-static test, but calling it per draw on a 2,100-draw submit is a
    // measurable cost for a variable nobody set.
    auto& program_skip = prosper::gpu::draw_program_skip_selector();
    const bool program_skip_armed = program_skip.armed();
    const bool program_census = prosper::gpu::draw_program_census_enabled();
    // PROSPER_DRAW_LINKSCAN: the graphics counterpart of PROSPER_COMPUTE_PARENTSCAN.
    // Hoisted for the same reason as the two above -- disarmed it is one bool.
    auto& link_scan = prosper::gpu::draw_link_scan_selector();
    const bool link_scan_armed = prosper::gpu::draw_link_scan_enabled();
    return ShaderOverrides{
        .refvs = refvs,
        .refvs_spv = std::move(refvs_spv),
        .ps_override = std::move(ps_override),
        .ps_override_is_file = ps_override_is_file,
        .ps_override_is_test = ps_override_is_test,
        .fs_match_mode = fs_match_mode,
        .fs_match = std::move(fs_match),
        .fs_guest_addr_text = fs_guest_addr_text,
        .fs_guest_addr = fs_guest_addr,
        .fs_guest_addr_valid = fs_guest_addr_valid,
        .fs_target_addr_text = fs_target_addr_text,
        .fs_target_addr = fs_target_addr,
        .fs_target_addr_valid = fs_target_addr_valid,
        .fs_target_dim_text = fs_target_dim_text,
        .fs_target_width = fs_target_width,
        .fs_target_height = fs_target_height,
        .fs_target_dim_valid = fs_target_dim_valid,
        .testps_match_mode = testps_match_mode,
        .testps_match = std::move(testps_match),
        .nops = nops,
        .skip_draws_env = skip_draws_env,
        .program_skip = program_skip,
        .program_skip_armed = program_skip_armed,
        .program_census = program_census,
        .link_scan = link_scan,
        .link_scan_armed = link_scan_armed};
}

void materialize_dirty_dcc_clears(DccClearContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& g_rtt = ctx.g_rtt;
    auto& items = ctx.items;
    auto& pending_timing = ctx.pending_timing;
    auto& timing_enabled = ctx.timing_enabled;
    // A uniform DCC clear is self-contained in metadata. If the ordered compute span dirtied
    // a retained target's metadata, materialize that clear now so the following graphics pass
    // loads the clear value rather than stale pixels (or an arbitrary fallback clear).
    const auto dcc_materialize_start = timing_enabled
        ? RenderClock::now() : RenderClock::time_point{};
    uint64_t dcc_materialize_surfaces = 0;
    uint64_t dcc_materialize_bytes = 0;
    // One uniform clear, decoded from `metadata_bytes` at `metadata_addr` (or from `host_data`
    // when the descriptor carries the bytes itself) as a descriptor with these components reads it.
    // `rendered` says which of the two loops below asked, and is only reported.
    const auto materialize = [&](RttCache::iterator found, uint64_t metadata_addr,
                                 uint64_t metadata_bytes, const uint8_t* host_data,
                                 uint64_t host_data_bytes, uint32_t num_components,
                                 bool alpha_is_on_msb, bool rendered) {
        if (!metadata_bytes || metadata_bytes > SIZE_MAX) return;
        std::vector<uint8_t> metadata(static_cast<size_t>(metadata_bytes));
        size_t copied = 0;
        if (host_data) {
            copied = static_cast<size_t>(std::min<uint64_t>(metadata.size(), host_data_bytes));
            std::memcpy(metadata.data(), host_data, copied);
        } else {
            copied = safe_copy(metadata.data(), metadata_addr, metadata.size());
        }
        const uint64_t texels = static_cast<uint64_t>(found->second.w) * found->second.h;
        const VkFormat format = prosper::test::backend_color_format(found->second.format);
        const uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(format);
        if (copied != metadata.size() || !bpp || texels > SIZE_MAX / bpp) return;
        uint8_t clear_rgba[4]{};
        if (!prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                clear_rgba, 1, metadata.data(), metadata.size(), num_components, alpha_is_on_msb))
            return;
        if (format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_R16G16B16A16_SFLOAT &&
            format != VK_FORMAT_B10G11R11_UFLOAT_PACK32)
            return;
        found->second.rgba.reset();
        found->second.has_uniform_color = true;
        for (uint32_t channel = 0; channel < 4; ++channel)
            found->second.uniform_color[channel] = clear_rgba[channel] ? 1.0f : 0.0f;
        // PROSPER_DCCLOG=1 -- diagnostic only, no behaviour change. A surface
        // materialised from a DCC fast-clear code becomes a UNIFORM colour for the
        // whole target, so if this decode is wrong the entire frame is one wrong
        // colour with no content -- which is exactly Little Nightmares III's
        // uniform-yellow presents (#2014). Deduped per (address, decoded colour, path) so a
        // run costs a handful of lines. `via=` names the path that took the clear: a span that
        // samples the target, or (since #4621) one that renders to it.
        // NOLINTNEXTLINE(concurrency-mt-unsafe): the cached environment read this block always made
        if (const char* dcclog = PROSPER_ENV_VALUE("PROSPER_DCCLOG")) {
            if (dcclog[0] == '1' && dcclog[1] == '\0') {
                static std::mutex dcc_mutex;
                static std::set<std::tuple<uint64_t, uint32_t, bool>> dcc_seen;
                const uint32_t packed = (uint32_t)clear_rgba[0] | ((uint32_t)clear_rgba[1] << 8) |
                                        ((uint32_t)clear_rgba[2] << 16) |
                                        ((uint32_t)clear_rgba[3] << 24);
                bool first = false;
                {
                    std::lock_guard<std::mutex> lock(dcc_mutex);
                    first = dcc_seen.emplace((uint64_t)found->first, packed, rendered).second;
                }
                if (first)
                    fprintf(
                        stderr,
                        "[dcclog] addr=0x%llx %ux%u fmt=%d ncomp=%u "
                        "alpha_msb=%d clear_rgba=(%u,%u,%u,%u) -> uniform=(%.0f,%.0f,%.0f,%.0f) "
                        "via=%s\n",
                        (unsigned long long)found->first, found->second.w, found->second.h,
                        (int)format, num_components, (int)alpha_is_on_msb, clear_rgba[0],
                        clear_rgba[1], clear_rgba[2], clear_rgba[3], found->second.uniform_color[0],
                        found->second.uniform_color[1], found->second.uniform_color[2],
                        found->second.uniform_color[3], rendered ? "rendered" : "sampled");
            }
        }
        found->second.dcc_metadata_dirty = false;
        ++dcc_materialize_surfaces;
        dcc_materialize_bytes += sizeof(found->second.uniform_color);
    };
    // Nearly every span has no dirty target at all, and then neither loop below has anything to
    // find: walk the draws only when one exists.
    static const std::vector<prosper::gpu::DrawItem> no_draws;
    const bool any_dirty = std::any_of(g_rtt.begin(), g_rtt.end(), [](const auto& entry) {
        return entry.second.dcc_metadata_dirty;
    });
    const auto& candidates = any_dirty ? items : no_draws;
    for (const auto& item : candidates) {
        const prosper::gpu::ShaderResourceTable* tables[] = {
            item.vrt.get(), item.prt.get(),
        };
        for (const auto* table : tables) {
            if (!table) continue;
            for (const auto& resource : table->resources) {
                auto found = g_rtt.find(resource.gpu_addr);
                if (found == g_rtt.end() || !found->second.dcc_metadata_dirty ||
                    !resource.compression_enabled ||
                    resource.metadata_addr != found->second.dcc_metadata_addr ||
                    resource.width != found->second.w ||
                    resource.height != found->second.h)
                    continue;
                materialize(found, resource.metadata_addr,
                            prosper::gpu::gpu_capture_dcc_metadata_footprint(resource),
                            resource.dcc_metadata_host_data, resource.dcc_metadata_host_data_size,
                            resource.num_components, resource.alpha_is_on_msb, /*rendered=*/false);
            }
        }
    }
    // A pass that RENDERS to such a target loads it as surely as one that samples it: a blend
    // reads the destination, and whatever the draws do not cover stays as the clear left it.
    // Nothing in such a span need sample the target, so the clear is decoded as the last
    // descriptor to sample it would have (note_rtt_dcc_descriptor). MOUSE: P.I. For Hire fills
    // its decal buffer's metadata with 0x40 -- the code for (0,0,0,1) -- each frame and blends
    // decals over it with the destination alpha as the running transmittance; the pass had been
    // starting from its own clear colour, (0,0,0,0), instead (#4556).
    //
    // Every colour slot: an engine's G-buffer attachments sit above slot 1 (#4624; Stray, The
    // Forgotten City and Bendy and the Dark Revival all clear theirs this way).
    //
    // Narrower than the sampled case on purpose, because here no descriptor is in hand to check:
    //   - the draw's guest extent must be the one the recorded descriptor described, so facts
    //     left over from another surface at this address are not applied to it;
    //   - the retained extent must be that extent, or it reduced by PROSPER_RENDER_SCALE;
    //   - no volume target: for those the dirty flag also marks a partial colour-plane write.
    for (const auto& item : candidates) {
        for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
            if (!prosper::frontend::mrt_write_mask(item, slot)) continue;
            const auto binding = color_binding(item, slot);
            auto found = binding.base ? g_rtt.find(binding.base) : g_rtt.end();
            if (found == g_rtt.end() || !found->second.dcc_metadata_dirty) continue;
            const RttSurf& surface = found->second;
            if (surface.volume_depth || surface.volume_guest_bytes ||
                binding.width != surface.dcc_width || binding.height != surface.dcc_height ||
                !((binding.width == surface.w && binding.height == surface.h) ||
                  prosper::frontend::rtt_scaled_extent_compatible(
                      binding.width, binding.height, surface.w, surface.h, ctx.render_scale)))
                continue;
            materialize(found, surface.dcc_metadata_addr, surface.dcc_metadata_bytes, nullptr, 0,
                        surface.dcc_num_components, surface.dcc_alpha_is_on_msb, /*rendered=*/true);
        }
    }
    if (timing_enabled) {
        pending_timing.dcc_materialize_ms +=
            std::chrono::duration<double, std::milli>(
                RenderClock::now() - dcc_materialize_start).count();
        pending_timing.dcc_materialize_surfaces += dcc_materialize_surfaces;
        pending_timing.dcc_materialize_bytes += dcc_materialize_bytes;
    }
}

} // namespace prosper::frontend::submit_renderer
