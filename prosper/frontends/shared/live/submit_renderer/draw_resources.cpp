// build_draw_frame_resources -- see draw_resources.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/draw_resources.hpp"
#include "shared/live/submit_renderer/image_resources.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"

namespace prosper::frontend::submit_renderer {


namespace {

// ---- One buffer resource of one draw (#3892) ----------------------------------------------------
//
// The per-resource loop's buffer branch: materialize one V#/storage/constant buffer into its
// FrameBufferResource (a direct guest view, a copy, or the all-zero fallback). Moved out of
// build_draw_frame_resources verbatim, except that its two `continue` statements -- the
// materialization rejects, which leave the resource out of the draw -- became `return
// ResourceOutcome::Skip`. The caller turns Skip back into `continue`, so the loop tail (census,
// timing, emplace) is skipped exactly as before.
enum class ResourceOutcome { Keep, Skip };

// The per-resource loop's state that materialize_buffer_resource reads and writes, one reference per object.
struct BufferResourceContext {
    const prosper::gpu::DrawItem & draw;
    int & g_this_submit;
    RenderTiming & pending_timing;
    const bool & timing_enabled;
    const bool & render_timing_detail;
    const bool & use_direct_buffer_views;
    const bool & use_tracked_buffer_membership_cache;
    uint32_t& set;
    const prosper::gpu::ShaderResource & r;
    const prosper::gpu::SpirvDescriptorBinding *& reflected_binding;
    const RenderClock::time_point& resource_timing_start;
    bool& resource_buffer_view;
    double& resource_buffer_probe_ms;
    double& resource_buffer_copy_ms;
    prosper::test::FrameBufferResource & buffer_resource;
};

template <class CopyResource, class DirectResource>
ResourceOutcome materialize_buffer_resource(BufferResourceContext& ctx,
                                            CopyResource& copy_resource,
                                            DirectResource& direct_resource) {
    // Every name the moved body used from build_draw_frame_resources, bound once to the same object.
    auto& draw = ctx.draw;
    auto& g_this_submit = ctx.g_this_submit;
    auto& pending_timing = ctx.pending_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& render_timing_detail = ctx.render_timing_detail;
    auto& use_direct_buffer_views = ctx.use_direct_buffer_views;
    auto& use_tracked_buffer_membership_cache = ctx.use_tracked_buffer_membership_cache;
    auto& set = ctx.set;
    auto& r = ctx.r;
    auto& reflected_binding = ctx.reflected_binding;
    auto& resource_timing_start = ctx.resource_timing_start;
    auto& resource_buffer_view = ctx.resource_buffer_view;
    auto& resource_buffer_probe_ms = ctx.resource_buffer_probe_ms;
    auto& resource_buffer_copy_ms = ctx.resource_buffer_copy_ms;
    auto& buffer_resource = ctx.buffer_resource;
    auto& fr = buffer_resource;
    fr.buffer_identity = r.gpu_addr;
    const prosper::gpu::StorageBufferMaterializationPlan materialization =
        prosper::gpu::plan_storage_buffer_materialization(
            *reflected_binding, r);
    if (!materialization.valid) {
        fprintf(stderr,
                "[buffer-materialization-reject] set=%u binding=%u addr=%llx "
                "declared=%u\n",
                set, r.binding, (unsigned long long)r.gpu_addr, r.size);
        // Name the sub-condition, once per (set,binding,address). The line above
        // collapses every reason into one string, and an investigation cannot act
        // on that: "the descriptor carries a partial zero-pad marker set" and
        // "the resource is the wrong class" are different pieces of work. Same
        // shape and the same justification as [mimg-mip] in rdna2_emit_alu.cpp.
        //
        // Deduped rather than gated, because the volume is the whole problem:
        // Dragon Quest VII emits 549,623 of the line above in one routed run,
        // ALL of them for a single address (0x20013f1bc0, set 0 binding 10) with
        // 257 different declared sizes -- so the useful signal is one line
        // describing that binding, not half a million repetitions of it (#1486).
        {
            static std::mutex why_mutex;
            static std::set<std::tuple<uint32_t, uint32_t, uint64_t>> why_seen;
            bool first_why = false;
            {
                std::lock_guard<std::mutex> lock(why_mutex);
                first_why = why_seen.emplace(set, r.binding,
                                             (uint64_t)r.gpu_addr).second;
            }
            if (first_why) {
                const auto& d = *reflected_binding;
                const bool has_logical = d.zero_pad_logical_bytes != 0;
                const bool has_binding_pad = d.zero_pad_binding_bytes != 0;
                const bool has_semantic =
                    d.zero_pad_semantic !=
                    prosper::gpu::StorageBufferTailSemantic::None;
                fprintf(stderr,
                        "[buffer-why] set=%u binding=%u addr=%llx declared=%u "
                        "markers=%d/%d/%d kind=%d readable=%d writable=%d "
                        "atomic=%d dynamic=%d required=%llu "
                        "pad_logical=%u pad_binding=%u semantic=%d "
                        "res_cls=%d fmt=%d ncomp=%u stride=%u size=%u "
                        "scalar_dwords=%u\n",
                        set, r.binding, (unsigned long long)r.gpu_addr, r.size,
                        (int)has_logical, (int)has_binding_pad,
                        (int)has_semantic,
                        (int)d.kind, (int)d.readable, (int)d.writable,
                        (int)d.atomic_access, (int)d.dynamic_access,
                        (unsigned long long)d.required_bytes,
                        d.zero_pad_logical_bytes, d.zero_pad_binding_bytes,
                        (int)d.zero_pad_semantic,
                        (int)r.cls, (int)r.format, r.num_components,
                        r.stride, r.size, r.scalar_buffer_dword_count);
            }
        }
        return ResourceOutcome::Skip;
    }
    // #1427: the guest's declared V#/V-buffer size is the real requirement — a
    // vertex fetch indexes anywhere inside it. The old 1 MiB clamp silently
    // truncated larger buffers, so every element past the cap read ZEROS: those
    // vertices all transformed to the same clip point (the MVP translation
    // column) and the primitive died as degenerate, with no reject and no log.
    // On Blue Prince's entrance hall that erased 44 of 248 scene draws —
    // the tile floor, the far table, most of the room — and read as a shading
    // defect for weeks. Upload the declared range under a ceiling that exists
    // only to bound a corrupt descriptor (a 64 MiB read also costs ~16K
    // guest_readable page probes, so it must stay bounded), and make any
    // truncation that does happen FAIL-VISIBLE. PROSPER_MAX_BUFFER_UPLOAD_MB
    // lowers the ceiling for a same-build A/B of this exact defect.
    const uint32_t requested_bytes = materialization.binding_bytes
        ? static_cast<uint32_t>(materialization.binding_bytes) : 256u;
    uint32_t nb = materialization.zero_padded_tail
        ? static_cast<uint32_t>(materialization.binding_bytes)
        : prosper::frontend::buffer_upload_bytes(requested_bytes);
    if (nb < (requested_bytes & ~3u)) {
        static std::set<uint64_t> truncated_reported;
        if (truncated_reported.size() < 32 &&
            truncated_reported.insert(r.gpu_addr).second)
            fprintf(stderr,
                    "[buffer-truncated] set=%u binding=%u addr=%llx declared=%u "
                    "uploaded=%u — fetches past the uploaded range read zeros and "
                    "collapse geometry (#1427)\n",
                    set, r.binding, (unsigned long long)r.gpu_addr,
                    requested_bytes, nb);
    }
    // A definitely unmapped source cannot contribute a byte. Preserve the
    // renderer's established all-zero fallback without allocating/probing the
    // descriptor's potentially corrupt declared size; robust buffer access makes
    // accesses beyond this minimum zero as well. Static reflection tells us how
    // much in-bounds storage the shader can definitely address.
    const BufferSourceGateResult source_gate = classify_buffer_source(
        r.host_data != nullptr, r.gpu_addr,
        use_tracked_buffer_membership_cache,
        prosper_renderer_guest_address_tracked,
        prosper_reserved_range_state);
    const bool unavailable_guest_buffer = source_gate.unavailable;
    if (timing_enabled)
        pending_timing.buffer_source_gate.record(source_gate);
    if (materialization.zero_padded_tail) {
        uint8_t logical[2] = {};
        const uint8_t* logical_source = nullptr;
        if (r.host_data && r.host_data_size >= sizeof(logical)) {
            logical_source = r.host_data;
        } else if (!r.host_data && !unavailable_guest_buffer &&
                   copy_resource(logical, r.gpu_addr, sizeof(logical)) ==
                       sizeof(logical)) {
            logical_source = logical;
        }
        fr.dwords.assign(1, 0);
        if (!logical_source ||
            !prosper::gpu::materialize_storage_buffer_bytes(
                materialization, logical_source, sizeof(logical),
                reinterpret_cast<uint8_t*>(fr.dwords.data()),
                sizeof(uint32_t))) {
            fprintf(stderr,
                    "[buffer-materialization-reject] set=%u binding=%u "
                    "two-byte source unavailable\n",
                    set, r.binding);
            return ResourceOutcome::Skip;
        }
    } else if (unavailable_guest_buffer) {
        const uint64_t minimum_bytes = std::min<uint64_t>(
            std::max<uint64_t>(reflected_binding->required_bytes, 256u),
            kMaxBufferUploadBytes);
        fr.dwords.assign(static_cast<size_t>((minimum_bytes + 3u) / 4u), 0);
    } else if (use_direct_buffer_views && nb >= 4) {
        const auto probe_start = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        if (const uint8_t* source = direct_resource(r.gpu_addr, nb)) {
            fr.dwords_view = reinterpret_cast<const uint32_t*>(source);
            fr.dwords_view_count = nb / sizeof(uint32_t);
            // Hosted/capture views can advertise the same guest identity while
            // supplying different bytes. Only this actual guest mapping may
            // authorize write-watch validation in the retained-upload backend.
            if (!r.host_data && reinterpret_cast<uintptr_t>(source) == r.gpu_addr)
                fr.direct_guest_buffer_addr = r.gpu_addr;
            resource_buffer_view = true;
        }
        if (timing_enabled)
            resource_buffer_probe_ms =
                std::chrono::duration<double, std::milli>(
                    RenderClock::now() - probe_start).count();
    }
    const auto copy_start = timing_enabled
        ? RenderClock::now() : RenderClock::time_point{};
    if (!materialization.zero_padded_tail &&
        !unavailable_guest_buffer && !fr.dwords_view_count &&
        use_direct_buffer_views) {
        if (nb >= 4) {
            fr.dwords.assign(nb / sizeof(uint32_t), 0);
            if (!copy_resource(reinterpret_cast<uint8_t*>(fr.dwords.data()),
                               r.gpu_addr, nb))
                fr.dwords.clear();
        }
        if (fr.dwords.empty()) fr.dwords.assign(64, 0);
    } else if (!materialization.zero_padded_tail &&
               !unavailable_guest_buffer && !use_direct_buffer_views) {
        if (nb >= 4) {
            std::vector<uint8_t> tmp(nb, 0);
            if (copy_resource(tmp.data(), r.gpu_addr, nb) > 0)
                fr.dwords.assign(
                    reinterpret_cast<const uint32_t*>(tmp.data()),
                    reinterpret_cast<const uint32_t*>(tmp.data() + nb));
        }
        if (fr.dwords.empty()) fr.dwords.assign(64, 0);
    }
    if (timing_enabled)
        resource_buffer_copy_ms = std::chrono::duration<double, std::milli>(
            RenderClock::now() - copy_start).count();
    if (render_timing_detail) {
        const uint64_t detail_min_submit =
            PROSPER_ENV_VALUE("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT")
                ? strtoull(getenv(
                      "PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT"), nullptr, 0)
                : 0;
        const double elapsed = std::chrono::duration<double, std::milli>(
            RenderClock::now() - resource_timing_start).count();
        static uint64_t detail_buffer_lines = 0;
        if (static_cast<uint64_t>(g_this_submit) >= detail_min_submit &&
            elapsed >= 0.5 && detail_buffer_lines++ < 250) {
            fprintf(stderr,
                    "[render-timing] buffer draw=%llu set=%u binding=%u "
                    "addr=0x%llx declared=%u uploaded=%u class=%u direct=%d "
                    "probe=%.2f copy=%.2f total=%.2f ms\n",
                    (unsigned long long)draw.draw_index, set, r.binding,
                    (unsigned long long)r.gpu_addr, requested_bytes, nb,
                    static_cast<unsigned>(r.cls),
                    static_cast<int>(resource_buffer_view),
                    resource_buffer_probe_ms, resource_buffer_copy_ms, elapsed);
        }
    }
    // PROSPER_CBLOG: log each constant buffer's first 4 dwords as floats, once per
    // address. If a scene draw's color/tint CB is (0,0,0,0), the PS outputs black
    // regardless of the (correctly-decoded) texture — the #300 black-scene suspect.
    if (PROSPER_ENV_ON("PROSPER_CBLOG") && r.cls == RC::ConstantBuffer) {
        static std::set<uint64_t> cbseen;
        if (cbseen.insert(r.gpu_addr).second) {
            const uint32_t* words = fr.buffer_words_data();
            size_t n = fr.buffer_word_count();
            const float* fp = reinterpret_cast<const float*>(words);
            fprintf(stderr, "[cb] bind=%u addr=0x%llx size=%u dw=%08x %08x %08x %08x  f=%.3f %.3f %.3f %.3f\n",
                    r.binding, (unsigned long long)r.gpu_addr, (unsigned)r.size,
                    n>0?words[0]:0, n>1?words[1]:0, n>2?words[2]:0, n>3?words[3]:0,
                    n>0?fp[0]:0.f, n>1?fp[1]:0.f, n>2?fp[2]:0.f, n>3?fp[3]:0.f);
        }
    }
    return ResourceOutcome::Keep;
}

} // namespace

BuiltFrameResources build_draw_frame_resources(DrawResourceContext& ctx,
                                               const prosper::gpu::DrawItem& draw,
                                               const prosper::gpu::ShaderResourceTable* vrt,
                                               const prosper::gpu::ShaderResourceTable* prt,
                                               prosper::test::BackendSubmissionBatch* producer_batch) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& g_rtt = ctx.g_rtt;
    auto& g_pass_log_submit = ctx.g_pass_log_submit;
    auto& invalidate_ds = ctx.invalidate_ds;
    auto& frame_no = ctx.frame_no;
    auto& rtt_on = ctx.rtt_on;
    auto& live_gpu_targets = ctx.live_gpu_targets;
    auto& write_watch_promotion_budget = ctx.write_watch_promotion_budget;
    auto& pinned_renderer_mip_targets = ctx.pinned_renderer_mip_targets;
    auto& g_this_submit = ctx.g_this_submit;
    auto& rtt_log = ctx.rtt_log;
    auto& pending_timing = ctx.pending_timing;
    auto& validation_census = ctx.validation_census;
    auto& timing_enabled = ctx.timing_enabled;
    auto& render_timing_detail = ctx.render_timing_detail;
    auto& texstore = ctx.texstore;
    auto& texstore_used = ctx.texstore_used;
    auto& decode_span_ordinal = ctx.decode_span_ordinal;
    auto& use_direct_buffer_views = ctx.use_direct_buffer_views;
    auto& use_tracked_buffer_membership_cache = ctx.use_tracked_buffer_membership_cache;
    auto& persistent_decoded_textures = ctx.persistent_decoded_textures;
    auto& persistent_decoded_texture_bytes = ctx.persistent_decoded_texture_bytes;
    auto& decode_generation = ctx.decode_generation;
    auto& retired_submit_pixels = ctx.retired_submit_pixels;
    auto& retired_submit_bytes = ctx.retired_submit_bytes;
    auto& persistent_texture_id = ctx.persistent_texture_id;
    auto& persistent_validation_scratch = ctx.persistent_validation_scratch;
    auto& persistent_decode_limit = ctx.persistent_decode_limit;
    auto& resource_hash_w = ctx.resource_hash_w;
    auto& resource_hash_h = ctx.resource_hash_h;
    auto& rtt_injection_cache = ctx.rtt_injection_cache;
    auto& reflect_memo = ctx.reflect_memo;
    auto& compact_buffer_resources = ctx.compact_buffer_resources;
    auto& reserve_frame_resources = ctx.reserve_frame_resources;
    auto& depth_array_census = ctx.depth_array_census;
    auto& depth_array_snapshots = ctx.depth_array_snapshots;
    auto& depth_array_snapshot_bytes = ctx.depth_array_snapshot_bytes;
    auto& depth_array_snapshot_admitted = ctx.depth_array_snapshot_admitted;
    auto& reuse_depth_arrays_across_draws = ctx.reuse_depth_arrays_across_draws;
    auto& compact_depth_array_snapshots = ctx.compact_depth_array_snapshots;
    auto& gpu_depth_array_snapshots = ctx.gpu_depth_array_snapshots;
    auto& disable_guest_depth_layers = ctx.disable_guest_depth_layers;
    auto& depth_array_guest_scans = ctx.depth_array_guest_scans;
    auto& depth_array_guest_scan_epoch = ctx.depth_array_guest_scan_epoch;
    auto& gpu_depth_cube_snapshots = ctx.gpu_depth_cube_snapshots;
    auto& depth_cube_gpu_snapshots = ctx.depth_cube_gpu_snapshots;
    // These two are the callback's own `static thread_local` objects, and the storage-image
    // writeback lambdas below name them WITHOUT capturing them (a variable with static storage
    // duration cannot be captured). A function-scope `static thread_local` reference keeps that
    // true: it binds once per thread to the calling thread's object, which is the same object
    // the callback passes in on every call, so every use -- deferred writebacks included --
    // reaches exactly the instance it reached when this body was a lambda inside the callback.
    static thread_local std::vector<bool>& texstore_pinned = ctx.texstore_pinned;
    static thread_local std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& decoded_textures = ctx.decoded_textures;
    auto consume_renderer_mip_chain = [&](const RendererMipChainLayout& chain,
                                          uint32_t width, uint32_t height,
                                          VkFormat format) {
        return consume_image_renderer_mip_chain(ctx, chain, width, height, format);
    };
    auto acquire_texstore_slot = [&]() -> size_t {
        return acquire_image_texstore_slot(ctx);
    };
    const auto clear_depth_array_snapshots = [&] {
        clear_image_depth_array_snapshots(ctx);
    };
    BuiltFrameResources built;
    const size_t candidate_resources =
        (vrt ? vrt->resources.size() : 0) + (prt ? prt->resources.size() : 0);
    const size_t reserve_hint = reserve_frame_resources
        ? std::min<size_t>(candidate_resources, 16) : 0;
    const auto reserve_if_empty = [reserve_hint](auto& resources) {
        if (reserve_hint > 1 && resources.empty())
            resources.reserve(reserve_hint);
    };
    // Once this callback has admitted a snapshot, observe queued guest writes at every
    // later draw, including draws with no array bindings. A callback-local latch keeps
    // this boundary identical when the per-draw control clears its memo; failure/clear
    // must not turn off subsequent notification processing.
    if (depth_array_snapshot_admitted) drain_guest_gpu_writes(g_rtt, invalidate_ds);
    if (!reuse_depth_arrays_across_draws) clear_depth_array_snapshots();
    auto add = [&](const prosper::gpu::ShaderResourceTable* t, uint32_t set,
                   const std::vector<uint32_t>& spirv,
                   prosper::gpu::SpirvShaderStage stage,
                   uint64_t shader_identity){
      if (!t) return;
      // #2256. This call is memoized inside shader_resources.cpp, but the memo's HIT path
      // is O(module size) three times over -- an FNV-1a walk to build the key, a full
      // vector equality compare to confirm it, and a copy of the report -- under a
      // process-global lock, twice per draw. Time it as its own leaf of build_R rather
      // than guessing: build_R's residual is the largest unexplained bucket in the
      // renderer (+6.16 of 8.02 ms/submit on RADV, #2250) and this is the biggest fixed
      // per-draw cost inside it.
      //
      // `words` is the falsifiable magnitude: it is exactly what the hash and the memcmp
      // each traverse, so a reader can check the ms against it instead of taking it on
      // faith. `distinct` is the memo's CEILING and is deliberately UNCAPPED -- the last
      // attempt on this path (#2239) died because its justifying probe hit a 4,096-entry
      // cap on precisely this count, and a truncated ceiling reads as a small one.
      const prosper::gpu::DescriptorValidationReport* memoized = nullptr;
      if (shader_identity) {
          const auto found = reflect_memo.find(shader_identity);
          // set/stage are checked rather than assumed. They are fixed per module on this
          // path today (VS -> set 0, PS -> set 1), so a mismatch cannot occur -- but the
          // key is the identity alone, and a silent wrong answer here would be a wrong
          // descriptor contract rather than a slow one. A mismatch falls through to the
          // real call and is counted.
          if (found != reflect_memo.end() &&
              found->second.set == set && found->second.stage == stage)
              memoized = &found->second.report;
          else if (found != reflect_memo.end()) {
              ++pending_timing.build_reflect_memo_mismatch;
              // Loud and unconditional on first occurrence, NOT gated on timing: this
              // path is unreachable by the argument above, and if the argument is wrong
              // the consequence is a wrong descriptor contract rather than a slow frame.
              // A counter only a timing run prints would hide exactly the case that
              // matters. The fallback is correct either way -- it re-reflects -- so this
              // reports rather than fails.
              static thread_local bool warned = false;
              if (!warned) {
                  warned = true;
                  fprintf(stderr,
                          "[render] *** shader identity %llu was reflected for set=%u "
                          "stage=%u and is now asked for set=%u stage=%u -- identity is "
                          "not stage-unique after all (#2256). Re-reflecting; the "
                          "descriptor contract is unaffected%s",
                          (unsigned long long)shader_identity, found->second.set,
                          (unsigned)found->second.stage, set, (unsigned)stage, "\n");
              }
          }
      }
      if (memoized && timing_enabled) ++pending_timing.build_reflect_memo_hits;
      const auto rt0 = (timing_enabled && !memoized)
          ? RenderClock::now() : RenderClock::time_point{};
      prosper::gpu::DescriptorValidationReport computed;
      if (!memoized)
          computed = prosper::gpu::validate_spirv_descriptor_interface(
              spirv, t, set, stage, false);
      if (!memoized && timing_enabled) {
          pending_timing.build_reflect_ms +=
              std::chrono::duration<double, std::milli>(
                  RenderClock::now() - rt0).count();
          ++pending_timing.build_reflect_calls;
          pending_timing.build_reflect_words += spirv.size();
          // Identity 0 means the module came from an external/replay path, so no memo
          // keyed on identity could serve it. Counted separately: folding those into
          // `distinct` would inflate the ceiling, folding them into the hit population
          // would inflate the saving. They are neither.
          if (shader_identity)
              pending_timing.build_reflect_identities.insert(shader_identity);
          else
              ++pending_timing.build_reflect_unidentified;
      }
      if (!memoized && shader_identity) {
          if (reflect_memo.size() >= kReflectMemoMaxEntries) {
              reflect_memo.clear();
              if (timing_enabled) ++pending_timing.build_reflect_memo_clears;
          }
          ReflectMemoEntry entry;
          entry.report = computed;
          entry.set = set;
          entry.stage = stage;
          memoized = &reflect_memo.emplace(shader_identity, std::move(entry))
                         .first->second.report;
      }
      // Bound to the memo entry when there is one and to the local otherwise. Both outlive
      // the per-resource loop below, and the loop does not touch the map.
      const prosper::gpu::DescriptorValidationReport& reflected =
          memoized ? *memoized : computed;
      for (auto& r : t->resources) {
          const prosper::gpu::SpirvDescriptorBinding* reflected_binding =
              prosper::gpu::find_spirv_descriptor_binding(
                  reflected, set, r.binding);
          // The front half deliberately retains every descriptor candidate it can prove
          // while folding guest code. The final recompiler can use only a subset (for
          // example, pc-specific scalar loads supersede the original broad V#). Extra
          // runtime bindings are harmless to validation, but materializing one absurd
          // unused declaration can allocate and zero the 64 MiB safety ceiling per draw.
          // The generated SPIR-V is authoritative about which bindings the backend needs.
          if (!reflected_binding) continue;
          const auto resource_timing_start = timing_enabled
              ? RenderClock::now() : RenderClock::time_point{};
          // #3891 ledger: count every texture reference, time one in 32.
          prosper::diagnostics::perf::TextureReferenceSample perf_texref_sample(
              r.cls == RC::Texture);
          // PROSPER_TEXREF_CENSUS: null unless the census is armed for this reference.
          prosper::frontend::TextureReferenceCensus* const texref_census =
              prosper::frontend::TextureReferenceCensus::enabled() &&
                      (r.cls == RC::Texture || r.cls == RC::StorageImage)
                  ? &prosper::frontend::texture_reference_census() : nullptr;
          uint64_t texref_census_key = 0;
          if (texref_census) texref_census->begin();
          bool resource_rtt_hit = false;
          bool resource_compute_image_hit = false;
          bool resource_compute_image_candidate = false;
          bool resource_compute_depth_hybrid = false;
          uint64_t resource_compute_producer_order = 0;
          uint32_t resource_compute_depth_overlay_mask = 0;
          bool resource_local_reuse = false;
          bool resource_persistent_hit = false;
          bool resource_persistent_submit_reuse = false;
          bool resource_persistent_miss = false;
          bool resource_persistent_invalidation = false;
          bool resource_buffer_view = false;
          double resource_buffer_probe_ms = 0.0;
          double resource_buffer_copy_ms = 0.0;
          double resource_texture_validation_ms = 0.0;
          size_t resource_texture_validated_bytes = 0;
          size_t resource_texture_source_bytes = 0;
          int resource_texture_submit_query = -1;
          int resource_texture_watch_query = -1;
          uint32_t resource_texture_watch_stability = 0;
          bool resource_texture_exact_validation = false;
          bool resource_texture_watch_active = false;
          bool resource_texture_watch_disabled = false;
          bool resource_texture_watch_only = false;
          bool resource_has_live_rtt = false;
          bool resource_has_ds_live = false;
          bool resource_persistent_candidate = false;
          size_t resource_persistent_source_size = 0;
          const bool image_resource = r.cls == RC::Texture || r.cls == RC::StorageImage;
          std::optional<prosper::test::FrameResource> full_resource;
          prosper::test::FrameBufferResource compact_resource;
          if (image_resource || !compact_buffer_resources) full_resource.emplace();
          prosper::test::FrameBufferResource& buffer_resource = full_resource
              ? static_cast<prosper::test::FrameBufferResource&>(*full_resource)
              : compact_resource;
          buffer_resource.binding = r.binding;
          buffer_resource.set = set;
          const bool normalized_sampling =
              reflected_binding->kind ==
                  prosper::gpu::SpirvDescriptorKind::CombinedImageSampler &&
              reflected_binding->normalized_sampling &&
              !reflected_binding->texel_access;
          const bool writable_storage_image =
              reflected_binding->kind ==
                  prosper::gpu::SpirvDescriptorKind::StorageImage &&
              reflected_binding->writable;
          // Captured replays preserve the logical guest address for RT identity but provide
          // owned bytes here. Production resources leave host_data null and use guest memory.
          auto copy_resource = [&](uint8_t* dst, uint64_t addr, size_t n) -> size_t {
              return copy_shader_resource(r, dst, addr, n);
          };
          // The same bytes `copy_resource` would deliver, read IN PLACE, or nullptr when it
          // would deliver fewer than `n` of them.
          //
          // Guest memory is 1:1 mapped, so a full-surface staging copy buys nothing except a
          // second address for the same bytes -- and on a 4K HDR intermediate that second
          // address is 64+ MiB of freshly mmap'd, page-faulted, memset and then munmap'd
          // anonymous memory per reference. Stray's title screen paid it twice a frame
          // (#3149's `__memmove_avx512` at 14.5% self, and its kernel page-management
          // frames, are this).
          //
          // Returning nullptr on a SHORT range is the load-bearing half: the decode branches
          // below rely on `copy_resource`'s partial return to fall back to a linear read or
          // to a zero-filled tail, so a sparse or over-declared surface must keep taking the
          // staged path rather than being handed a pointer into a hole.
          // PROSPER_NO_DIRECT_TEXTURE_SOURCE restores the staged copy on BOTH source kinds.
          // A lever that disabled only the guest-memory half would still be named after the
          // whole optimisation, and an A/B under it would attribute the remaining half's
          // cost to whatever else moved -- the instrument-trap shape this tree records.
          static const bool no_direct_texture_source =
              PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_TEXTURE_SOURCE") != nullptr;
          auto direct_resource_source = [&](uint64_t addr, size_t n) -> const uint8_t* {
              return direct_shader_resource_source(r, no_direct_texture_source, addr, n);
          };
          // The tiled bytes a detiler will read, without staging a copy of them when the
          // range is fully mapped. `lease` must outlive the returned pointer; it stays
          // empty on the direct path and holds the staging buffer otherwise, so an early
          // exit from a decode branch returns it to the pool either way.
          //
          // `got` reports what the STAGED path would have delivered, exactly as
          // `copy_resource` did, so every downstream `got < ...` fallback keeps its
          // meaning. On the direct path `got == n` by construction: the pointer is only
          // handed out when the whole range is readable.
          auto stage_tiled_source =
              [&](prosper::frontend::DecodeScratchPool::Lease& lease, uint64_t addr,
                  size_t n, size_t& got) -> const uint8_t* {
              return stage_tiled_shader_resource(r, no_direct_texture_source, lease, addr, n, got);
          };
          // Avoid allocating and copying storage buffers that are already present in stable,
          // readable unified guest memory. The callback and backend upload are synchronous;
          // compute/DMA boundaries split graphics spans before this point. The submit-scoped
          // readability guard proves the complete range; an invalid tail retains the copied
          // fallback.
          auto direct_resource = [&](uint64_t addr, size_t n) -> const uint8_t* {
              if (!n || addr < 0x1000 || addr > UINT64_MAX - n ||
                  (addr & (alignof(uint32_t) - 1))) return nullptr;
              if (r.host_data) {
                  if (addr < r.gpu_addr) return nullptr;
                  const uint64_t off = addr - r.gpu_addr;
                  if (off > r.host_data_size || n > r.host_data_size - off) return nullptr;
                  const uint8_t* source = r.host_data + off;
                  return (reinterpret_cast<uintptr_t>(source) &
                          (alignof(uint32_t) - 1)) ? nullptr : source;
              }
              // Reuse the executor's submit-scoped range cache; on Windows the same guard
              // also materializes sparse direct-memory pages when necessary.
              if (n > UINT32_MAX || safe_span(addr, n) != n ||
                  !prosper::gpu::guest_readable(addr, static_cast<uint32_t>(n)))
                  return nullptr;
              return reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr));
          };
          auto copy_dcc_metadata = [&](uint8_t* dst, size_t n) -> size_t {
              return copy_shader_dcc_metadata(r, dst, n);
          };
          if (image_resource) {
          ImageBindingContext image_binding{
              .t = t,
              .set = set,
              .shader_identity = shader_identity,
              .r = r,
              .reflected_binding = reflected_binding,
              .texref_census = texref_census,
              .texref_census_key = texref_census_key,
              .resource_rtt_hit = resource_rtt_hit,
              .resource_compute_image_hit = resource_compute_image_hit,
              .resource_compute_image_candidate = resource_compute_image_candidate,
              .resource_compute_depth_hybrid = resource_compute_depth_hybrid,
              .resource_compute_producer_order = resource_compute_producer_order,
              .resource_compute_depth_overlay_mask = resource_compute_depth_overlay_mask,
              .resource_local_reuse = resource_local_reuse,
              .resource_persistent_hit = resource_persistent_hit,
              .resource_persistent_submit_reuse = resource_persistent_submit_reuse,
              .resource_persistent_miss = resource_persistent_miss,
              .resource_persistent_invalidation = resource_persistent_invalidation,
              .resource_texture_validation_ms = resource_texture_validation_ms,
              .resource_texture_validated_bytes = resource_texture_validated_bytes,
              .resource_texture_source_bytes = resource_texture_source_bytes,
              .resource_texture_submit_query = resource_texture_submit_query,
              .resource_texture_watch_query = resource_texture_watch_query,
              .resource_texture_watch_stability = resource_texture_watch_stability,
              .resource_texture_exact_validation = resource_texture_exact_validation,
              .resource_texture_watch_active = resource_texture_watch_active,
              .resource_texture_watch_disabled = resource_texture_watch_disabled,
              .resource_texture_watch_only = resource_texture_watch_only,
              .resource_has_live_rtt = resource_has_live_rtt,
              .resource_has_ds_live = resource_has_ds_live,
              .resource_persistent_candidate = resource_persistent_candidate,
              .resource_persistent_source_size = resource_persistent_source_size,
              .full_resource = full_resource,
              .normalized_sampling = normalized_sampling,
              .writable_storage_image = writable_storage_image,
              .no_direct_texture_source = no_direct_texture_source,
          };
          const auto image_status = materialize_image_resource(ctx, image_binding, draw, producer_batch);
          // reject is a private result store. Image-local destruction now precedes this store;
          // enclosing resource teardown and the next binding still follow it.
          if (image_status.disposition == ImageDisposition::Reject)
              built.reject(image_status.reason);
          if (image_status.disposition != ImageDisposition::Keep)
              continue;
          } else {
              BufferResourceContext materialize_buffer_resource_ctx{
                  .draw = draw,
                  .g_this_submit = g_this_submit,
                  .pending_timing = pending_timing,
                  .timing_enabled = timing_enabled,
                  .render_timing_detail = render_timing_detail,
                  .use_direct_buffer_views = use_direct_buffer_views,
                  .use_tracked_buffer_membership_cache = use_tracked_buffer_membership_cache,
                  .set = set,
                  .r = r,
                  .reflected_binding = reflected_binding,
                  .resource_timing_start = resource_timing_start,
                  .resource_buffer_view = resource_buffer_view,
                  .resource_buffer_probe_ms = resource_buffer_probe_ms,
                  .resource_buffer_copy_ms = resource_buffer_copy_ms,
                  .buffer_resource = buffer_resource};
              if (materialize_buffer_resource(materialize_buffer_resource_ctx, copy_resource,
                                              direct_resource) == ResourceOutcome::Skip)
                  continue;
          }
          if (texref_census && full_resource) {
              using Census = prosper::frontend::TextureReferenceCensus;
              const auto& image = *full_resource;
              const int klass = resource_rtt_hit ? Census::kRtt
                  : (resource_compute_image_hit || resource_compute_depth_hybrid) ? Census::kCompute
                  : resource_persistent_submit_reuse ? Census::kPersistSubmit
                  : resource_persistent_hit ? Census::kPersistHit
                  : resource_persistent_miss ? Census::kPersistMiss
                  : resource_persistent_invalidation ? Census::kPersistInvalid
                  : resource_local_reuse ? Census::kLocal
                  : resource_has_live_rtt ? Census::kOtherRttAuthority
                  : !resource_persistent_candidate ? Census::kOtherNotCandidate
                  : Census::kOther;
              // The resolved identity only: what the backend would bind. The class is
              // deliberately excluded -- a persistent hit followed by a submit-local reuse
              // of the same pixels is the same answer reached two ways.
              uint64_t o = 1469598103934665603ull;
              auto mix = [&o](uint64_t v) { o ^= v; o *= 1099511628211ull; o ^= v >> 32; };
              mix(reinterpret_cast<uintptr_t>(image.tex_rgba));
              mix(reinterpret_cast<uintptr_t>(image.gpu_detile.get()));
              mix(image.tex_byte_size); mix(image.has_uniform_color);
              for (float f : image.uniform_color) { uint32_t b; std::memcpy(&b, &f, 4); mix(b); }
              mix(image.tw); mix(image.th); mix(image.td); mix(image.sample_count);
              mix(image.declared_mip_levels); mix(image.img_dim); mix(image.guest_array);
              mix(static_cast<uint64_t>(image.texture_format)); mix(image.is_storage_image);
              mix(image.storage_image_contract_valid);
              mix(static_cast<bool>(image.storage_image_writeback));
              mix(image.mag_filter); mix(image.min_filter); mix(image.mip_filter);
              for (uint32_t a : image.addr_uvw) mix(a);
              for (uint32_t w : image.swizzle) mix(w);
              mix(static_cast<uint64_t>(image.render_target_guest_format));
              mix(image.persistent_texture_id); mix(image.persistent_texture_version);
              mix(image.persistent_render_target_id);
              mix(image.persistent_render_target_mip_count);
              for (uint64_t id : image.persistent_render_target_mip_ids) mix(id);
              mix(image.persistent_depth_target_id);
              mix(reinterpret_cast<uintptr_t>(image.borrowed_compute_image));
              mix(image.borrowed_compute_vertical_stack_layers);
              texref_census->finish(g_this_submit, texref_census_key, o, klass);
          }
          perf_texref_sample.finish();
          if (timing_enabled) {
              const double elapsed = std::chrono::duration<double, std::milli>(
                  RenderClock::now() - resource_timing_start).count();
              if (image_resource) {
                  const auto& image = *full_resource;
                  // Classified in the SAME order the path decides them, and exclusively:
                  // a reference satisfied by an earlier class never reaches a later one,
                  // so every reference lands in exactly one bucket or in `other`. The
                  // residual is therefore two independent numbers -- a COUNT of references
                  // nothing above claimed, and a signed remainder of ms -- and they
                  // disagree only if the classification is wrong.
                  if (resource_rtt_hit)                      { pending_timing.tex_rtt_ms += elapsed;           ++pending_timing.tex_rtt_n; }
                  else if (resource_compute_image_hit ||
                           resource_compute_depth_hybrid)    { pending_timing.tex_compute_ms += elapsed;       ++pending_timing.tex_compute_n; }
                  else if (resource_persistent_submit_reuse) { pending_timing.tex_persist_reuse_ms += elapsed; ++pending_timing.tex_persist_reuse_n; }
                  else if (resource_persistent_hit)          { pending_timing.tex_persist_hit_ms += elapsed;   ++pending_timing.tex_persist_hit_n; }
                  else if (resource_persistent_miss)         { pending_timing.tex_persist_miss_ms += elapsed;  ++pending_timing.tex_persist_miss_n; }
                  else if (resource_persistent_invalidation) { pending_timing.tex_persist_invalid_ms += elapsed; ++pending_timing.tex_persist_invalid_n; }
                  else if (resource_local_reuse)             { pending_timing.tex_local_ms += elapsed;         ++pending_timing.tex_local_n; }
                  else {
                      ++pending_timing.tex_other_n;
                      if (elapsed > pending_timing.tex_other_slowest_ms) {
                          pending_timing.tex_other_slowest_ms = elapsed;
                          pending_timing.tex_other_addr = r.gpu_addr;
                          pending_timing.tex_other_source_bytes =
                              resource_persistent_source_size;
                          pending_timing.tex_other_width = r.width;
                          pending_timing.tex_other_height = r.height;
                          pending_timing.tex_other_depth = r.depth;
                          pending_timing.tex_other_format =
                              static_cast<uint32_t>(r.format);
                          pending_timing.tex_other_components = r.num_components;
                          pending_timing.tex_other_tile_mode = r.tile_mode;
                          pending_timing.tex_other_img_dim = r.img_dim;
                          pending_timing.tex_other_class =
                              static_cast<uint32_t>(r.cls);
                          pending_timing.tex_other_compute_candidate =
                              resource_compute_image_candidate;
                          pending_timing.tex_other_persistent_candidate =
                              resource_persistent_candidate;
                          pending_timing.tex_other_compressed =
                              r.compression_enabled;
                          pending_timing.tex_other_depth_compare = r.depth_compare;
                          pending_timing.tex_other_host_backed = r.host_data != nullptr;
                      }
                  }
                  pending_timing.textures++;
                  pending_timing.texture_bytes += static_cast<uint64_t>(image.tw) *
                      image.th * image.td * image.sample_count *
                      prosper::test::backend_color_bytes_per_pixel(image.texture_format);
                  pending_timing.texture_ms += elapsed;
                  const uint64_t detail_min_submit = PROSPER_ENV_VALUE("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT")
                      ? strtoull(PROSPER_ENV_VALUE("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT"), nullptr, 0) : 0;
                  // render_timing_detail is the SAME hoist #2214 introduced, already used
                  // at the sibling site a few lines up; this one was missed. It is inside
                  // `if (timing_enabled)`, so it costs a normal run nothing -- but it made
                  // an F8 capture pay a getenv per resource, inflating the enclosing totals
                  // the capture then reports. An instrument that charges for being read.
                  if (render_timing_detail &&
                      static_cast<uint64_t>(g_this_submit) >= detail_min_submit) {
                      static uint64_t detail_lines = 0;
                      if ((elapsed >= 0.5 || resource_persistent_invalidation ||
                           resource_compute_image_hit ||
                           resource_compute_depth_hybrid) &&
                          detail_lines++ < 250) {
                          const char* cache_state = resource_rtt_hit ? "rtt" :
                              (resource_compute_depth_hybrid ? "compute-depth-hybrid" :
                              (resource_compute_image_hit ? "compute-image" :
                              (resource_local_reuse ? "local" :
                              (resource_persistent_submit_reuse ? "persistent-submit" :
                              (resource_persistent_hit ? "persistent-hit" :
                              (resource_persistent_invalidation ? "persistent-invalid" :
                              (resource_persistent_miss ? "persistent-miss" : "uncached")))))));
                          auto submit_query_name = [](int query) {
                              switch (query) {
                                  case static_cast<int>(
                                      prosper::gpu::GuestGpuWriteQuery::Unchanged):
                                      return "unchanged";
                                  case static_cast<int>(
                                      prosper::gpu::GuestGpuWriteQuery::Overlap):
                                      return "overlap";
                                  case static_cast<int>(
                                      prosper::gpu::GuestGpuWriteQuery::Unknown):
                                      return "unknown";
                                  default: return "none";
                              }
                          };
                          auto watch_query_name = [](int query) {
                              switch (query) {
                                  case static_cast<int>(
                                      prosper::host::GuestWriteWatchQuery::Unchanged):
                                      return "unchanged";
                                  case static_cast<int>(
                                      prosper::host::GuestWriteWatchQuery::Dirty):
                                      return "dirty";
                                  case static_cast<int>(
                                      prosper::host::GuestWriteWatchQuery::Unknown):
                                      return "unknown";
                                  default: return "none";
                              }
                          };
                          fprintf(stderr,
                                  "[render-timing] texture addr=0x%llx %ux%ux%u out=%ux%ux%u "
                                  "dim=%u fmt=%u comps=%u tile=%u class=%u storage=%d "
                                  "host=%d live=%d depth-live=%d candidate=%d source=%zu "
                                  "compressed=%d depth-compare=%d compute-candidate=%d "
                                  "cache=%s id=%llu validate=%s %.2fms/%zuB/%zuB "
                                  "submit=%s watch=%s active=%d disabled=%d only=%d stable=%u "
                                  "total=%.2f ms\n",
                                  (unsigned long long)r.gpu_addr, r.width, r.height,
                                  r.depth, image.tw, image.th, image.td, r.img_dim,
                                  (unsigned)r.format, r.num_components,
                                  r.tile_mode, static_cast<unsigned>(r.cls),
                                  static_cast<int>(image.is_storage_image),
                                  static_cast<int>(r.host_data != nullptr),
                                  static_cast<int>(resource_has_live_rtt),
                                  static_cast<int>(resource_has_ds_live),
                                  static_cast<int>(resource_persistent_candidate),
                                  resource_persistent_source_size,
                                  static_cast<int>(r.compression_enabled),
                                  static_cast<int>(r.depth_compare),
                                  static_cast<int>(resource_compute_image_candidate),
                                  cache_state,
                                  (unsigned long long)image.persistent_texture_id,
                                  resource_texture_exact_validation ? "exact" : "skip",
                                  resource_texture_validation_ms,
                                  resource_texture_validated_bytes,
                                  resource_texture_source_bytes,
                                  submit_query_name(resource_texture_submit_query),
                                  watch_query_name(resource_texture_watch_query),
                                  static_cast<int>(resource_texture_watch_active),
                                  static_cast<int>(resource_texture_watch_disabled),
                                  static_cast<int>(resource_texture_watch_only),
                                  resource_texture_watch_stability, elapsed);
                      }
                  }
              } else {
                  const size_t buffer_bytes =
                      buffer_resource.buffer_word_count() * sizeof(uint32_t);
                  pending_timing.buffers++;
                  pending_timing.buffer_views += resource_buffer_view;
                  pending_timing.buffer_bytes += buffer_bytes;
                  if (!resource_buffer_view)
                      pending_timing.buffer_materialized_bytes += buffer_bytes;
                  pending_timing.buffer_ms += elapsed;
              }
          }
          if (full_resource) {
              reserve_if_empty(built.full);
              const uint32_t index = static_cast<uint32_t>(built.full.size());
              built.full.push_back(std::move(*full_resource));
              if (compact_buffer_resources) {
                  reserve_if_empty(built.order);
                  built.order.push_back(index);
              }
          } else {
              reserve_if_empty(built.buffers);
              reserve_if_empty(built.order);
              const uint32_t index = static_cast<uint32_t>(built.buffers.size());
              built.buffers.push_back(std::move(compact_resource));
              built.order.push_back(kCompactBufferResourceBit | index);
          }
          if (timing_enabled && !image_resource) {
              if (compact_buffer_resources) ++pending_timing.compact_buffer_resources;
              else ++pending_timing.full_buffer_resources;
          }
      }
    };
    add(vrt, 0, draw.vs_words(), prosper::gpu::SpirvShaderStage::Vertex,
        draw.vs_identity);
    add(prt, 1, draw.fs_words(), prosper::gpu::SpirvShaderStage::Fragment,
        draw.fs_identity);
    // VS resources -> descriptor set 0, PS -> set 1
    return built;
}

} // namespace prosper::frontend::submit_renderer
