// The submit callback's diagnostics -- see callback_diagnostics.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/callback_diagnostics.hpp"

namespace prosper::frontend::submit_renderer {

void log_submit_index(SubmitLogContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& items = ctx.items;
    auto& phase = ctx.phase;
    auto& g_this_submit = ctx.g_this_submit;
    // PROSPER_SUBMITLOG: print the GPU-submit index periodically (at native speed, before the slow
    // render) so it can be correlated with guest-side log lines (e.g. a MsgDialog wait) to find the
    // exact submit at which a scene appears — for aiming PROSPER_RENDER_FIRST at it.
    if (phase.first_span && PROSPER_ENV_ON("PROSPER_SUBMITLOG") && (g_this_submit % 1000 == 0))
        fprintf(stderr, "[submit] index=%d (%zu draw items)\n", g_this_submit, items.size());
    if (const char* sd = PROSPER_ENV_VALUE("PROSPER_SUBMITLOG_DIM")) {
        uint32_t sw = 0, sh = 0;
        if (sscanf(sd, "%ux%u", &sw, &sh) == 2)
            for (const auto& it : items)
                if (it.color0_width == sw && it.color0_height == sh) {
                    fprintf(stderr, "[submit] index=%d target=0x%llx extent=%ux%u (%zu draw items)\n",
                            g_this_submit, (unsigned long long)it.color0_base, sw, sh, items.size());
                    break;
                }
    }
}

void diagnostic_target_forces_render(DiagnosticTargetContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& items = ctx.items;
    auto& force_target = ctx.force_target;
    if (const char* td = PROSPER_ENV_VALUE("PROSPER_RENDER_TARGET_DIM")) {
        uint32_t tw = 0, th = 0;
        if (sscanf(td, "%ux%u", &tw, &th) == 2)
            for (const auto& it : items)
                if (it.color0_width == tw && it.color0_height == th) { force_target = true; break; }
    }
    if (const char* rd = PROSPER_ENV_VALUE("PROSPER_RENDER_RESOURCE_DIM")) {
        uint32_t rw = 0, rh = 0;
        if (sscanf(rd, "%ux%u", &rw, &rh) == 2)
            for (const auto& it : items) {
                auto has_dim = [&](const prosper::gpu::ShaderResourceTable* table) {
                    if (!table) return false;
                    for (const auto& r : table->resources)
                        if (r.width == rw && r.height == rh) return true;
                    return false;
                };
                if (has_dim(it.vrt.get()) || has_dim(it.prt.get())) { force_target = true; break; }
            }
    }
}

void dump_first_item_spirv(ShaderDumpContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& items = ctx.items;
    if (PROSPER_ENV_ON("PROSPER_SHADER_DUMP") && !items.empty()) {
        std::string d = PROSPER_ENV_VALUE("PROSPER_SHADER_DUMP");
        const auto& dump_vs = items[0].vs_words();
        const auto& dump_fs = items[0].fs_words();
        if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(dump_vs.data(), 4, dump_vs.size(), f); fclose(f); }
        if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(dump_fs.data(), 4, dump_fs.size(), f); fclose(f); }
        fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", dump_vs.size(), dump_fs.size()); fflush(stderr);
    }
}

void dump_presented_frame(PresentedFrameDumpContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& frame_dir = ctx.frame_dir;
    auto& dump_bmps = ctx.dump_bmps;
    auto& w = ctx.w;
    auto& h = ctx.h;
    auto& published_gpu = ctx.published_gpu;
    auto& px = ctx.px;
    auto& n = ctx.n;
    // PROSPER_DUMP_CONTENT=<min-nonzero-bytes>: dump ONLY frames whose framebuffer has at least
    // that many nonzero bytes — catches the intermittent content submits the periodic dump misses.
    size_t content_thr = 0; if (const char* c = PROSPER_ENV_VALUE("PROSPER_DUMP_CONTENT")) content_thr = (size_t)atol(c);
    // Sparse long-route captures can override the default first-60/every-10 cadence. This is
    // particularly useful for 4K titles, where capturing itself would otherwise add gigabytes
    // of readback I/O before the scene under investigation is reached. Zero disables a phase.
    static const int dump_first = [] { const char* e = getenv("PROSPER_FRAME_DUMP_FIRST");
                                        return e ? (int)atol(e) : 60; }();
    static const int dump_every = [] { const char* e = getenv("PROSPER_FRAME_DUMP_EVERY");
                                        return e ? (int)atol(e) : 10; }();
    // PROSPER_PRESENT_NZLOG=N: log the presented frame's nonzero-byte count every N frames WITHOUT
    // writing any image. A memory-safe content proxy for long progression runs — dumping BMPs to a
    // tmpfs frame dir exhausts RAM, this does not. 0/unset disables.
    static const int nzlog_every = [] { const char* e = getenv("PROSPER_PRESENT_NZLOG");
                                        return e ? (int)atol(e) : 0; }();
    size_t px_nz = 0;
    if (dump_bmps || nzlog_every) for (uint8_t b : px) px_nz += (b != 0);
    if (px.empty() && !published_gpu) {
        fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
    } else if (dump_bmps && ((content_thr && px_nz >= content_thr) ||
               (!content_thr && ((dump_first > 0 && n < dump_first) ||
                                 (dump_every > 0 && n % dump_every == 0))))) {
        char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", frame_dir.c_str(), n);
        prosper::test::dump_bmp(fn, px, w, h);
        fprintf(stderr, "[render] frame %d rendered (%ux%u) nz=%zu -> %s\n", n, w, h, px_nz, fn);
    } else if (nzlog_every && !px.empty() && (n % nzlog_every == 0)) {
        fprintf(stderr, "[render-nz] frame %d (%ux%u) nz=%zu\n", n, w, h, px_nz);
    }
}

} // namespace prosper::frontend::submit_renderer
