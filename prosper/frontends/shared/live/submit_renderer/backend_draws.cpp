// build_backend_draws, clear_for -- see backend_draws.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/backend_draws.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"

namespace prosper::frontend::submit_renderer {

std::vector<prosper::test::BackendDraw> build_backend_draws(BackendDrawContext& ctx,
                                                            const std::vector<const prosper::gpu::DrawItem*>& group,
                                                            prosper::test::BackendSubmissionBatch* producer_batch) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& native_fragment_vote_width = ctx.native_fragment_vote_width;
    auto& partial_wave_fragment = ctx.partial_wave_fragment;
    auto& phase = ctx.phase;
    auto& rtt_log = ctx.rtt_log;
    auto& pending_timing = ctx.pending_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& use_direct_index_views = ctx.use_direct_index_views;
    auto& descriptor_validate_mode = ctx.descriptor_validate_mode;
    auto& draw_resource_ctx = ctx.draw_resource_ctx;
    auto& refvs = ctx.overrides.refvs;
    auto& refvs_spv = ctx.overrides.refvs_spv;
    auto& ps_override = ctx.overrides.ps_override;
    auto& ps_override_is_file = ctx.overrides.ps_override_is_file;
    auto& ps_override_is_test = ctx.overrides.ps_override_is_test;
    auto& fs_match_mode = ctx.overrides.fs_match_mode;
    auto& fs_match = ctx.overrides.fs_match;
    auto& fs_guest_addr_text = ctx.overrides.fs_guest_addr_text;
    auto& fs_guest_addr = ctx.overrides.fs_guest_addr;
    auto& fs_guest_addr_valid = ctx.overrides.fs_guest_addr_valid;
    auto& fs_target_addr_text = ctx.overrides.fs_target_addr_text;
    auto& fs_target_addr = ctx.overrides.fs_target_addr;
    auto& fs_target_addr_valid = ctx.overrides.fs_target_addr_valid;
    auto& fs_target_dim_text = ctx.overrides.fs_target_dim_text;
    auto& fs_target_width = ctx.overrides.fs_target_width;
    auto& fs_target_height = ctx.overrides.fs_target_height;
    auto& fs_target_dim_valid = ctx.overrides.fs_target_dim_valid;
    auto& testps_match_mode = ctx.overrides.testps_match_mode;
    auto& testps_match = ctx.overrides.testps_match;
    auto& nops = ctx.overrides.nops;
    auto& program_skip = ctx.overrides.program_skip;
    auto& program_skip_armed = ctx.overrides.program_skip_armed;
    auto& program_census = ctx.overrides.program_census;
    auto& link_scan = ctx.overrides.link_scan;
    auto& link_scan_armed = ctx.overrides.link_scan_armed;
    // The callback's process-lifetime skip_draws_env is named, uncaptured, by the captureless
    // draw_is_skipped lambda (a variable with static storage duration needs no capture). A
    // function-scope static reference binds once to that same object, so the lambda keeps
    // reaching it exactly as it did inside the callback.
    static const char *& skip_draws_env = ctx.overrides.skip_draws_env;
    auto build_R = [&](const prosper::gpu::DrawItem& draw,
                       const prosper::gpu::ShaderResourceTable* vrt,
                       const prosper::gpu::ShaderResourceTable* prt,
                       prosper::test::BackendSubmissionBatch* producer_batch) {
        return build_draw_frame_resources(draw_resource_ctx, draw, vrt, prt, producer_batch);
    };
    // Poison mode keeps the draw running while making invalid bindings visually/numerically
    // unmistakable: magenta/cyan texels for images and NaN-like dwords for buffers. Missing,
    // duplicate, wrong-type, and undersized bindings are replaced from the reflected manifest.
    // Read once per submit, not once per call. poison_R is invoked TWICE per draw (VS and
    // PS), so at 2,179 draws in a submit its getenv was 4,358 calls -- and on Windows
    // getenv takes a process-wide lock and walks the environment block, which is why #2214
    // measured +43% from removing exactly this shape from the per-resource path.
    //
    // Hoisted rather than wrapped in PROSPER_ENV_VALUE, and the difference is not stylistic:
    // test_gpu_capture_render.cpp and test_shader_resources.cpp ARM this variable at runtime
    // between phases, and a process-lifetime cache would not observe the second write. Those
    // tests would not fail -- they would go VACUOUS and keep printing [ok] against a stale
    // value, which is the #2214 defect that `cached_env_arming_logic` now gates. A per-submit
    // hoist keeps every runtime write observable from the next submit on.
    auto poison_R = [descriptor_validate_mode](
                       std::vector<prosper::test::FrameResource>& resources,
                       const std::vector<uint32_t>& spirv,
                       const prosper::gpu::ShaderResourceTable* table,
                       uint32_t set, prosper::gpu::SpirvShaderStage stage) {
        const char* mode = descriptor_validate_mode;
        if (!mode || strcmp(mode, "poison")) return;
        auto report = prosper::gpu::validate_spirv_descriptor_interface(spirv, table, set, stage, false);
        static const uint8_t poison_tex[16] = {
            255, 0, 255, 255,   0, 255, 255, 255,
            0, 255, 255, 255,   255, 0, 255, 255,
        };
        static const uint32_t poison_storage_uint[16] = {
            0x3f800000u, 0u, 0x3f800000u, 0x3f800000u,
            0u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0x3f800000u, 0u, 0x3f800000u, 0x3f800000u,
        };
        for (const auto& d : report.descriptors) {
            bool invalid = false;
            for (const auto& issue : report.issues)
                if (issue.error && issue.binding == d.binding) { invalid = true; break; }
            if (!invalid) continue;
            auto first = std::find_if(resources.begin(), resources.end(), [&](const auto& r) {
                return r.set == set && r.binding == d.binding;
            });
            size_t count = 0;
            for (const auto& r : resources) if (r.set == set && r.binding == d.binding) ++count;
            const bool wants_buffer = d.kind == prosper::gpu::SpirvDescriptorKind::StorageBuffer;
            const bool wants_storage_image =
                d.kind == prosper::gpu::SpirvDescriptorKind::StorageImage;
            const uint64_t available = first == resources.end() ? 0 :
                (first->is_texture() ? (uint64_t)first->tw * first->th * first->td * 4
                                     : first->buffer_word_count() * 4);
            const bool wrong_type = first != resources.end() &&
                (wants_buffer ? first->is_texture()
                              : (!first->is_texture() ||
                                 first->is_storage_image != wants_storage_image));
            const bool undersized = wants_buffer && available < std::max<uint64_t>(d.required_bytes, 4);
            resources.erase(std::remove_if(resources.begin(), resources.end(), [&](const auto& r) {
                return r.set == set && r.binding == d.binding;
            }), resources.end());
            prosper::test::FrameResource replacement;
            replacement.set = set; replacement.binding = d.binding;
            if (wants_buffer) {
                size_t words = static_cast<size_t>((std::max<uint64_t>(d.required_bytes, 16) + 3) / 4);
                // Poison replacements stay deliberately small: this is a diagnostic
                // substitute for an invalid binding, not a real upload, so it does not
                // follow build_R's ceiling (which is now the declared range, #1427).
                words = std::min<size_t>(words, 1u << 18);
                replacement.dwords.assign(words, 0x7FC0CDCDu);
            } else {
                replacement.storage_image_numeric_class = d.image_numeric_class;
                replacement.storage_image_contract_valid =
                    d.image_numeric_class ==
                        prosper::gpu::SpirvImageNumericClass::Float ||
                    d.image_numeric_class ==
                        prosper::gpu::SpirvImageNumericClass::Uint;
                if (wants_storage_image &&
                    d.image_numeric_class ==
                        prosper::gpu::SpirvImageNumericClass::Uint) {
                    replacement.tex_rgba = reinterpret_cast<const uint8_t*>(
                        poison_storage_uint);
                    replacement.texture_format = VK_FORMAT_R32G32B32A32_UINT;
                } else {
                    replacement.tex_rgba = poison_tex;
                }
                replacement.tw = 2; replacement.th = 2;
                replacement.is_storage_image = wants_storage_image;
                replacement.mag_filter = replacement.min_filter = 0;
            }
            fprintf(stderr, "[descriptor] poison set=%u binding=%u type=%s reason=%s%s%s%s\n",
                    set, d.binding, prosper::gpu::spirv_descriptor_kind_name(d.kind),
                    count == 0 ? "missing" : "",
                    count > 1 ? "duplicate" : "",
                    wrong_type ? "wrong-type" : "",
                    undersized ? "undersized" : "");
            resources.push_back(std::move(replacement));
        }
    };
    auto draw_is_skipped = [](uint64_t idx) -> bool {
        if (!skip_draws_env) return false;
        for (const char* s = skip_draws_env; *s;) {
            char* end = nullptr; unsigned long long v = strtoull(s, &end, 0);
            if (end != s && v == idx) return true;
            s = (end && *end) ? end + 1 : end;
            if (!s || !*s) break;
        }
        return false;
    };
    std::vector<prosper::test::BackendDraw> bds;
    // Once per call, not once per draw -- see the note on descriptor_validate_mode above
    // for why this is a hoist and not a PROSPER_ENV_* cache (test_eop_write.cpp arms
    // PROSPER_GFXLOG at runtime, and a cached read would make its arms vacuous).
    const bool gfxlog = getenv("PROSPER_GFXLOG") != nullptr;
    for (const auto* itp : group) {
        const auto& it = *itp;
        if (draw_is_skipped(it.draw_index)) continue;
        prosper::test::BackendDraw bd;
        bd.source_submit = phase.source_submit;
        if (refvs) {
            bd.vs = refvs_spv;
        } else if (it.vs_shared) {
            bd.vs_shared = it.vs_shared;
        } else {
            bd.vs = it.vs;
        }
        bd.gs = refvs ? std::vector<uint32_t>{} : it.gs;
        // File and synthetic overrides have independent exact-match gates.
        bool fs_ov = !ps_override.empty();
        if (fs_ov && ps_override_is_file) {
            if (fs_match_mode == 1)      fs_ov = (it.fs_words() == fs_match);   // valid match -> exact only
            else if (fs_match_mode == 2) fs_ov = false;                // requested-but-invalid -> off
            // mode 0 -> legacy global file override (unchanged)
            if (!fs_guest_addr_valid || !fs_target_addr_valid || !fs_target_dim_valid ||
                (fs_guest_addr_text && it.fs_guest_addr != fs_guest_addr) ||
                (fs_target_addr_text && it.color0_base != fs_target_addr) ||
                (fs_target_dim_text &&
                 (it.color0_width != fs_target_width ||
                  it.color0_height != fs_target_height)))
                fs_ov = false;
        }
        if (fs_ov && ps_override_is_test) {
            if (testps_match_mode == 1)      fs_ov = (it.fs_words() == testps_match);
            else if (testps_match_mode == 2) fs_ov = false;
            // mode 0 -> legacy global TESTPS override (unchanged)
        }
        if (fs_ov && ps_override_is_file && fs_match_mode == 1)
            fprintf(stderr, "[fs-match] file override applied to draw#%llu\n",
                    (unsigned long long)it.draw_index);
        if (fs_ov && ps_override_is_test && testps_match_mode == 1)
            fprintf(stderr, "[testps-match] synthetic override applied to draw#%llu\n",
                    (unsigned long long)it.draw_index);
        if (fs_ov) {
            bd.fs = ps_override;
        } else if (it.fs_shared) {
            bd.fs_shared = it.fs_shared;
        } else {
            bd.fs = it.fs;
        }
        bd.vs_identity = refvs ? 0 : it.vs_identity;
        bd.fs_identity = fs_ov ? 0 : it.fs_identity;
        bd.fs_guest_addr = fs_ov ? 0 : it.fs_guest_addr;
        bd.allow_native_fragment_vote_width =
            !fs_ov && native_fragment_vote_width;
        bd.allow_partial_wave_fragment = !fs_ov && partial_wave_fragment;
        bd.draw_index = it.draw_index;
        bd.command_order = it.command_order;
        bd.vcount = refvs ? 3u : it.vertex_count;
        bd.instance_count = it.instance_count;
        bd.vertex_offset = refvs ? 0 : it.vertex_offset;
        bd.ps     = nops ? nullptr : &it.ps;
        // Five clock reads bounding four spans, only when timing is armed -- ~0.9% of
        // this bucket at 2,100 draws a submit. It inflates what it measures while
        // armed, as every timer here does; read the shares, not the totals.
        const auto bt0 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        auto built_resources = build_R(it, it.vrt.get(), it.prt.get(), producer_batch);
        bd.R = std::move(built_resources.full);
        bd.B = std::move(built_resources.buffers);
        bd.resource_order = std::move(built_resources.order);
        const auto bt1 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        // && of the positives rather than || of the negations: identical short-circuit
        // and identical outcome, but it puts the accounting below on a path the
        // `continue` cannot skip.
        // descriptor_validate_mode is the per-submit hoist made for poison_R above, and
        // it serves this pair for the same reason: without it each call re-reads getenv,
        // and with the variable unset that read IS the whole call. It measured
        // 5.77 ms/submit in the build_resources partition -- larger than the validation
        // it was declining to do.
        const bool contract_ok =
            prosper::gpu::validate_runtime_descriptor_contract(
                "VS/backend", bd.vs_words(), it.vrt.get(), 0,
                prosper::gpu::SpirvShaderStage::Vertex, descriptor_validate_mode) &&
            prosper::gpu::validate_runtime_descriptor_contract(
                "PS/backend", bd.fs_words(), it.prt.get(), 1,
                prosper::gpu::SpirvShaderStage::Fragment, descriptor_validate_mode);
        const auto bt2 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        if (timing_enabled) {
            pending_timing.build_r_ms +=
                std::chrono::duration<double, std::milli>(bt1 - bt0).count();
            pending_timing.build_validate_ms +=
                std::chrono::duration<double, std::milli>(bt2 - bt1).count();
        }
        // A rejected draw has already spent build_R and both validations, and they are
        // counted above. Dropping it without accounting would move that time into the
        // residual, where it reads as unattributed work -- the exact defect this
        // partition exists to make visible.
        if (!built_resources.complete || !contract_ok) {
            // #3891: a dropped draw is a correctness alarm, not only a timing bucket,
            // and it carries the site that dropped it.
            prosper::diagnostics::perf::drop_draw(built_resources.complete
                ? DropReason::ContractMismatch : built_resources.drop_reason);
            if (timing_enabled) ++pending_timing.build_rejected;
            continue;
        }
        poison_R(bd.R, bd.vs_words(), it.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex);
        poison_R(bd.R, bd.fs_words(), it.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment);
        const auto bt3 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        // Indexed draw: hand the executor-fetched index data to the backend (vkCmdDrawIndexed).
        // Skipped under REFVS — the reference VS is a 3-vertex non-indexed fullscreen triangle.
        if (!refvs) {
            if (use_direct_index_views) bd.borrow_indices(it.indices);
            else bd.indices = it.indices;
        }
        if (timing_enabled) {
            const auto bt4 = RenderClock::now();
            pending_timing.build_poison_ms +=
                std::chrono::duration<double, std::milli>(bt3 - bt2).count();
            pending_timing.build_indices_ms +=
                std::chrono::duration<double, std::milli>(bt4 - bt3).count();
            // Counted AFTER the reject, so `build_draws` is accepted draws — and the
            // leaves do not all share it. `build_R` and `validate` ran for
            // draws + rejected; `poison` and `indices` ran for draws alone. Both
            // numbers are printed so the reader can pick the right denominator, and
            // the report line says which is which — a per-draw figure divided by the
            // wrong one over-states silently, and with rejected=0 nobody would notice
            // the rule until the first run where it is not.
            ++pending_timing.build_draws;
        }
        if (gfxlog) fprintf(stderr,
            "[render] item %zu: %zu resources vcount=%u instances=%u nidx=%zu topo=%u mask=0x%x blend=%d\n",
            bds.size(), bd.R.size() + bd.B.size(), bd.vcount, bd.instance_count,
            bd.index_count(), it.ps.topology,
            it.ps.color_write_mask, (int)it.ps.blend_enable);
        // RTTLOG per-draw detail (render-window-only, unlike the GFXLOG firehose): enough
        // state to diagnose a pass whose inputs HIT the RTT cache yet outputs nothing —
        // blend factors, write mask, viewport, and each PS-sampled texture address (#319).
        if (rtt_log) {
            fprintf(stderr, "[rtt]   draw#%llu vs=0x%llx fs=0x%llx tgt=0x%llx "
                    "vcount=%u nidx=%zu topo=%u mask=0x%x "
                    "blend=%d(src=%u dst=%u) vp=%d(%.2f,%.2f %.2fx%.2f) z=%d zw=%d ps=",
                    (unsigned long long)it.draw_index,
                    (unsigned long long)it.vs_guest_addr,
                    (unsigned long long)it.fs_guest_addr,
                    (unsigned long long)it.color0_base, bd.vcount, bd.index_count(),
                    it.ps.topology, it.ps.color_write_mask, (int)it.ps.blend_enable,
                    it.ps.src_color_blend_factor, it.ps.dst_color_blend_factor,
                    (int)it.ps.has_viewport,
                    it.ps.viewport_x, it.ps.viewport_y, it.ps.viewport_w, it.ps.viewport_h,
                    (int)it.ps.depth_test_enable, (int)it.ps.depth_write_enable);
            if (it.prt) for (const auto& r : it.prt->resources)
                if (r.cls == RC::Texture)
                    fprintf(stderr, " tex@%u=0x%llx(%ux%u f%u)", r.binding,
                            (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
            fprintf(stderr, "\n");
        }
        // PROSPER_DRAW_PROGRAM_CENSUS: which programs does this title actually draw
        // with? One line per distinct (vs, vs-chain, ps) triple, then powers of two --
        // bounded by the program count, not the draw count. This is the census that
        // supplies PROSPER_SKIP_DRAW_PROGRAM's input.
        if (program_census) {
            const auto sighting = prosper::gpu::draw_program_census().observe(
                it.vs_guest_addr, it.vs_chain_guest_addr, it.fs_guest_addr);
            if (sighting.print)
                fprintf(stderr, "[draw-program] %s vs=0x%llx chain=0x%llx ps=0x%llx "
                        "draws=%llu distinct=%llu vcount=%u nidx=%zu topo=%u tgt=0x%llx\n",
                        sighting.first ? "NEW " : "    ",
                        (unsigned long long)it.vs_guest_addr,
                        (unsigned long long)it.vs_chain_guest_addr,
                        (unsigned long long)it.fs_guest_addr,
                        (unsigned long long)sighting.ordinal,
                        (unsigned long long)sighting.distinct,
                        bd.vcount, bd.index_count(), it.ps.topology,
                        (unsigned long long)it.color0_base);
        }
        // PROSPER_DRAW_LINKSCAN: census the LINKED LISTS this draw's scalar buffers
        // contain, taken from the exact bytes prosper is about to upload -- not from
        // guest memory, because the question is what the SHADER sees. Ordered with the
        // other observers and before the skip, for the trap-166 reason below.
        if (link_scan_armed &&
            link_scan.matches(it.vs_guest_addr, it.vs_chain_guest_addr,
                              it.fs_guest_addr)) {
            const auto& lcfg = prosper::gpu::draw_link_scan_settings();
            const uint32_t* heads_words = nullptr; size_t heads_count = 0;
            const uint32_t* rec_words = nullptr;   size_t rec_count = 0;
            size_t buffers_seen = 0;
            std::vector<const prosper::test::FrameBufferResource*> link_resources;
            link_resources.reserve(bd.R.size() + bd.B.size());
            if (bd.resource_order.empty()) {
                for (const auto& fr : bd.R)
                    if (!fr.is_texture() && !fr.is_storage_image)
                        link_resources.push_back(&fr);
                for (const auto& fr : bd.B) link_resources.push_back(&fr);
            } else {
                for (uint32_t token : bd.resource_order) {
                    const bool compact = (token & kCompactBufferResourceBit) != 0;
                    const uint32_t index = token & ~kCompactBufferResourceBit;
                    if (compact) link_resources.push_back(&bd.B[index]);
                    else if (!bd.R[index].is_texture() && !bd.R[index].is_storage_image)
                        link_resources.push_back(&bd.R[index]);
                }
            }
            for (const auto* fr_ptr : link_resources) {
                const auto& fr = *fr_ptr;
                const uint32_t* words = fr.buffer_words_data();
                const size_t nwords = fr.buffer_word_count();
                if (!words || nwords == 0) continue;
                ++buffers_seen;
                // The DECLARED descriptor beside the UPLOADED bytes. A short upload is
                // the mis-resolved-descriptor case and is invisible in the census alone:
                // the shader reads zeros past the end and a zero link is not an exit.
                uint64_t res_addr = 0; uint32_t res_declared = 0; int res_cls = -1;
                const prosper::gpu::ShaderResourceTable* table =
                    fr.set == 0 ? it.vrt.get() : it.prt.get();
                if (table)
                    for (const auto& r : table->resources)
                        if (r.binding == fr.binding) {
                            res_addr = r.gpu_addr; res_declared = r.size;
                            res_cls = static_cast<int>(r.cls);
                            break;
                        }
                if (fr.set == 1 && fr.binding == lcfg.heads_binding) {
                    heads_words = words; heads_count = nwords;
                }
                if (fr.set == 1 && fr.binding == lcfg.records_binding) {
                    rec_words = words; rec_count = nwords;
                }
                bool exhausted = false;
                const uint32_t scan_ordinal = link_scan.should_scan(
                    it.fs_guest_addr, fr.set, fr.binding,
                    lcfg.max_scans_per_buffer, &exhausted);
                if (!scan_ordinal) {
                    if (exhausted)
                        fprintf(stderr,
                                "[linkscan] capped ps=0x%llx set=%u binding=%u "
                                "addr=0x%llx after %u scan(s) -- later draws of this "
                                "buffer are NOT scanned\n",
                                (unsigned long long)it.fs_guest_addr, fr.set,
                                fr.binding, (unsigned long long)res_addr,
                                lcfg.max_scans_per_buffer);
                    continue;
                }
                const std::span<const uint32_t> span(words, nwords);
                const auto hist = prosper::gpu::histogram_words(
                    span, lcfg.encoding.terminator);
                const auto self = prosper::gpu::census_self_walk(span, lcfg.encoding);
                fprintf(stderr,
                        "[linkscan] ps=0x%llx draw#%llu scan=%u set=%u binding=%u "
                        "cls=%d addr=0x%llx declared=%u uploaded_dw=%zu | "
                        "hist zero=%u term=%u other=%u first_other=[%u]=0x%08x "
                        "other_range=0x%08x..0x%08x | self-walk stride=%u next=+%u "
                        "term=0x%08x records=%u starts=%u terminating=%u cyclic=%u "
                        "cycle-nodes=%u oob-starts=%u longest=%u\n",
                        (unsigned long long)it.fs_guest_addr,
                        (unsigned long long)it.draw_index, scan_ordinal, fr.set,
                        fr.binding, res_cls, (unsigned long long)res_addr, res_declared,
                        nwords, hist.zero, hist.terminator, hist.other,
                        hist.first_other_index, hist.first_other_value, hist.min_other,
                        hist.max_other, lcfg.encoding.record_stride_dwords,
                        lcfg.encoding.next_dword_offset, lcfg.encoding.terminator,
                        self.records, self.starts, self.terminating, self.cyclic,
                        self.cycle_nodes, self.oob_starts, self.longest);
                for (uint32_t k = 0; k < self.sample_count; ++k)
                    fprintf(stderr,
                            "[linkscan]   cycle-member[%u] link=%u -> next=%u\n", k,
                            self.sample_link[k], self.sample_next[k]);
                // WHO last wrote these bytes. An all-zero buffer has two completely
                // different causes -- a producer that never ran, and a producer that
                // ran and wrote zeros -- and the census alone cannot tell them apart.
                // The recorder summary is printed with the answer, unconditionally,
                // because a bounded and per-kind-gated history makes "no writer"
                // a statement about the INSTRUMENT unless the reader can see which
                // recorders fired (writer_provenance.hpp's scope list, #2111).
                if (res_addr && res_declared) {
                    if (!prosper::gpu::writer_provenance_enabled()) {
                        static bool provenance_note = false;
                        if (!provenance_note) {
                            provenance_note = true;
                            fprintf(stderr,
                                    "[linkscan]   writer provenance is OFF -- set "
                                    "PROSPER_WRITER_PROVENANCE=1 to learn whether an "
                                    "all-zero buffer was never written or was written "
                                    "with zeros\n");
                        }
                    } else {
                        const auto writer = prosper::gpu::last_guest_write_overlap(
                            res_addr, res_declared);
                        if (writer)
                            fprintf(stderr,
                                    "[linkscan]   last writer binding=%u kind=%s "
                                    "addr=0x%llx size=%llu submit=%llu item=%llu "
                                    "order=%llu identity=0x%llx | recorders %s\n",
                                    fr.binding,
                                    prosper::gpu::guest_writer_kind_name(writer->kind),
                                    (unsigned long long)writer->addr,
                                    (unsigned long long)writer->size,
                                    (unsigned long long)writer->submit,
                                    (unsigned long long)writer->item,
                                    (unsigned long long)writer->order,
                                    (unsigned long long)writer->identity,
                                    prosper::gpu::guest_write_recorder_summary());
                        else
                            fprintf(stderr,
                                    "[linkscan]   last writer binding=%u NONE -- no "
                                    "recorded write overlaps 0x%llx+%u | history=%zu "
                                    "recorders %s\n",
                                    fr.binding, (unsigned long long)res_addr,
                                    res_declared,
                                    prosper::gpu::guest_write_history_size(),
                                    prosper::gpu::guest_write_recorder_summary());
                    }
                }
            }
            if (heads_words && rec_words) {
                const auto heads = prosper::gpu::census_head_walk(
                    std::span<const uint32_t>(rec_words, rec_count),
                    std::span<const uint32_t>(heads_words, heads_count),
                    lcfg.encoding);
                fprintf(stderr,
                        "[linkscan] ps=0x%llx draw#%llu HEAD-WALK heads_binding=%u "
                        "(%zu dw) records_binding=%u (%zu dw) starts=%u "
                        "terminating=%u cyclic=%u cycle-nodes=%u oob-starts=%u "
                        "longest=%u\n",
                        (unsigned long long)it.fs_guest_addr,
                        (unsigned long long)it.draw_index, lcfg.heads_binding,
                        heads_count, lcfg.records_binding, rec_count, heads.starts,
                        heads.terminating, heads.cyclic, heads.cycle_nodes,
                        heads.oob_starts, heads.longest);
            }
            // Silence would otherwise be indistinguishable between "the program never
            // drew" and "it drew with no buffers at all". Say which, once.
            if (buffers_seen == 0) {
                static std::set<uint64_t> no_buffer_reported;
                if (no_buffer_reported.insert(it.fs_guest_addr).second)
                    fprintf(stderr,
                            "[linkscan] ps=0x%llx draw#%llu matched but has NO storage "
                            "buffer resources -- nothing to census\n",
                            (unsigned long long)it.fs_guest_addr,
                            (unsigned long long)it.draw_index);
            }
        }
        // PROSPER_SKIP_DRAW_PROGRAM: withhold this draw from the GPU because one of its
        // programs is named. LAST in the per-draw build on purpose -- instrument trap
        // 166: a diagnostic skip ordered before the dump blinds every other instrument
        // to the thing under suspicion, and the program worth declining is precisely the
        // one that hangs the GPU. So the draw is fully realized, fully validated and
        // fully logged by [render]/[rtt]/[draw-program] first; only the Vulkan draw call
        // is withheld. Its CPU-side cost is therefore real, and the timing partition
        // above has already counted it in build_draws, which is arithmetically right --
        // it did spend build_R, validation, poison and indices.
        if (program_skip_armed) {
            const auto decision = program_skip.evaluate(
                it.vs_guest_addr, it.vs_chain_guest_addr, it.fs_guest_addr);
            if (decision.skip) {
                if (decision.print)
                    fprintf(stderr,
                            "[draw-decline] program=0x%llx stage=%s "
                            "reason=skipped-by-selector count=%llu draw#%llu order=%llu "
                            "vs=0x%llx chain=0x%llx ps=0x%llx vcount=%u nidx=%zu "
                            "tgt=0x%llx\n",
                            (unsigned long long)decision.address,
                            prosper::gpu::draw_program_stage_name(decision.stage),
                            (unsigned long long)decision.ordinal,
                            (unsigned long long)it.draw_index,
                            (unsigned long long)it.command_order,
                            (unsigned long long)it.vs_guest_addr,
                            (unsigned long long)it.vs_chain_guest_addr,
                            (unsigned long long)it.fs_guest_addr,
                            bd.vcount, bd.index_count(),
                            (unsigned long long)it.color0_base);
                continue;
            }
        }
        bds.push_back(std::move(bd));
    }
    return bds;
}

// Clear color for a group: the game's decoded fast-clear taken from the group's first item's
// resolved pipeline state, or its default opaque black when no fast-clear was programmed.
// Passing this to render_draws_rgba replaces the old hardcoded debug blue on the live path
// (#309) — PROSPER_CLEAR_DEBUG still forces blue for spotting unrendered areas.
const float* clear_for(const std::vector<const prosper::gpu::DrawItem*>& g) {
    return g.empty() ? nullptr : g.front()->ps.clear_color;
}

} // namespace prosper::frontend::submit_renderer
