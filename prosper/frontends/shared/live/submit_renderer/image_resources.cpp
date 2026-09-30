// build_draw_frame_resources -- see draw_resources.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/draw_resources.hpp"
#include "shared/live/submit_renderer/image_resources.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"

namespace prosper::frontend::submit_renderer {

size_t consume_image_renderer_mip_chain(DrawResourceContext& ctx, const RendererMipChainLayout& chain, uint32_t width, uint32_t height, VkFormat format) {
    auto& pinned_renderer_mip_targets = ctx.pinned_renderer_mip_targets;
        size_t matched = 0;
        for (uint32_t level = 0; level < chain.level_count; ++level) {
            const uint64_t id = chain.level_ids[level];
            const uint32_t level_width = std::max(width >> level, 1u);
            const uint32_t level_height = std::max(height >> level, 1u);
            for (PinnedRendererMipTarget& target : pinned_renderer_mip_targets) {
                if (target.id != id || target.width != level_width ||
                    target.height != level_height || target.format != format)
                    continue;
                if (!target.consumed) ++matched;
                target.consumed = true;
                break;
            }
        }
        return matched;
}

size_t acquire_image_texstore_slot(DrawResourceContext& ctx) {
    auto& texstore = ctx.texstore;
    auto& texstore_pinned = ctx.texstore_pinned;
    auto& texstore_used = ctx.texstore_used;
        while (texstore_used < texstore.size() && texstore_pinned[texstore_used])
            ++texstore_used;
        if (texstore_used == texstore.size()) {
            texstore.emplace_back();
            texstore_pinned.push_back(false);
        }
        return texstore_used++;
}

void clear_image_depth_array_snapshots(DrawResourceContext& ctx) {
    auto& depth_array_snapshots = ctx.depth_array_snapshots;
    auto& depth_array_snapshot_bytes = ctx.depth_array_snapshot_bytes;
        depth_array_snapshots.clear();
        depth_array_snapshot_bytes = 0;
}

size_t copy_shader_resource(const prosper::gpu::ShaderResource& r, uint8_t* dst, uint64_t addr, size_t n) {
              if (!r.host_data) return safe_copy(dst, addr, n);
              if (addr < r.gpu_addr) return 0;
              uint64_t off = addr - r.gpu_addr;
              if (off >= r.host_data_size) return 0;
              size_t take = static_cast<size_t>(std::min<uint64_t>(n, r.host_data_size - off));
              std::memcpy(dst, r.host_data + off, take);
              return take;
}

const uint8_t* direct_shader_resource_source(const prosper::gpu::ShaderResource& r, const bool& no_direct_texture_source, uint64_t addr, size_t n) {
              if (!n || no_direct_texture_source) return nullptr;
              if (r.host_data) {
                  // Mirrors copy_resource's host_data arm: the same bounds, and `take == n`.
                  if (addr < r.gpu_addr) return nullptr;
                  const uint64_t off = addr - r.gpu_addr;
                  if (off >= r.host_data_size || n > r.host_data_size - off) return nullptr;
                  return r.host_data + off;
              }
              if (addr < 0x1000 || addr > UINT64_MAX - n) return nullptr;
              return safe_span(addr, n) == n
                  ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr))
                  : nullptr;
}

const uint8_t* stage_tiled_shader_resource(const prosper::gpu::ShaderResource& r, const bool& no_direct_texture_source, prosper::frontend::DecodeScratchPool::Lease& lease, uint64_t addr, size_t n, size_t& got) {
    const auto direct_resource_source = [&](uint64_t addr, size_t n) {
        return direct_shader_resource_source(r, no_direct_texture_source, addr, n);
    };
    const auto copy_resource = [&](uint8_t* dst, uint64_t addr, size_t n) {
        return copy_shader_resource(r, dst, addr, n);
    };
              if (const uint8_t* direct = direct_resource_source(addr, n)) {
                  got = n;
                  return direct;
              }
              lease = prosper::frontend::decode_scratch_pool().take(n);
              got = copy_resource(lease.data(), addr, n);
              lease.zero_tail(got);   // restores what `std::vector<uint8_t>(n, 0)` gave
              return lease.data();
}

size_t copy_shader_dcc_metadata(const prosper::gpu::ShaderResource& r, uint8_t* dst, size_t n) {
              if (!r.dcc_metadata_host_data)
                  return safe_copy(dst, r.metadata_addr, n);
              const size_t take = static_cast<size_t>(std::min<uint64_t>(
                  n, r.dcc_metadata_host_data_size));
              std::memcpy(dst, r.dcc_metadata_host_data, take);
              return take;
}

ImageResourceStatus materialize_image_resource(DrawResourceContext& ctx, ImageBindingContext& binding, const prosper::gpu::DrawItem& draw, prosper::test::BackendSubmissionBatch* producer_batch) {
    auto& g_rtt = ctx.g_rtt;
    auto& g_pass_log_submit = ctx.g_pass_log_submit;
    auto& frame_no = ctx.frame_no;
    auto& rtt_on = ctx.rtt_on;
    auto& live_gpu_targets = ctx.live_gpu_targets;
    auto& write_watch_promotion_budget = ctx.write_watch_promotion_budget;
    auto& g_this_submit = ctx.g_this_submit;
    auto& rtt_log = ctx.rtt_log;
    auto& pending_timing = ctx.pending_timing;
    auto& validation_census = ctx.validation_census;
    auto& timing_enabled = ctx.timing_enabled;
    auto& texstore = ctx.texstore;
    auto& decode_span_ordinal = ctx.decode_span_ordinal;
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
    auto& depth_array_census = ctx.depth_array_census;
    auto& depth_array_snapshots = ctx.depth_array_snapshots;
    auto& depth_array_snapshot_bytes = ctx.depth_array_snapshot_bytes;
    auto& depth_array_snapshot_admitted = ctx.depth_array_snapshot_admitted;
    auto& compact_depth_array_snapshots = ctx.compact_depth_array_snapshots;
    auto& gpu_depth_array_snapshots = ctx.gpu_depth_array_snapshots;
    auto& disable_guest_depth_layers = ctx.disable_guest_depth_layers;
    auto& depth_array_guest_scans = ctx.depth_array_guest_scans;
    auto& depth_array_guest_scan_epoch = ctx.depth_array_guest_scan_epoch;
    auto& gpu_depth_cube_snapshots = ctx.gpu_depth_cube_snapshots;
    auto& depth_cube_gpu_snapshots = ctx.depth_cube_gpu_snapshots;
    static thread_local std::vector<bool> & texstore_pinned = ctx.texstore_pinned;
    static thread_local std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash> & decoded_textures = ctx.decoded_textures;
    auto& t = binding.t;
    auto& set = binding.set;
    auto& shader_identity = binding.shader_identity;
    auto& r = binding.r;
    auto& reflected_binding = binding.reflected_binding;
    auto& texref_census = binding.texref_census;
    auto& texref_census_key = binding.texref_census_key;
    auto& resource_rtt_hit = binding.resource_rtt_hit;
    auto& resource_compute_image_hit = binding.resource_compute_image_hit;
    auto& resource_compute_image_candidate = binding.resource_compute_image_candidate;
    auto& resource_compute_depth_hybrid = binding.resource_compute_depth_hybrid;
    auto& resource_compute_producer_order = binding.resource_compute_producer_order;
    auto& resource_compute_depth_overlay_mask = binding.resource_compute_depth_overlay_mask;
    auto& resource_local_reuse = binding.resource_local_reuse;
    auto& resource_persistent_hit = binding.resource_persistent_hit;
    auto& resource_persistent_submit_reuse = binding.resource_persistent_submit_reuse;
    auto& resource_persistent_miss = binding.resource_persistent_miss;
    auto& resource_persistent_invalidation = binding.resource_persistent_invalidation;
    auto& resource_texture_validation_ms = binding.resource_texture_validation_ms;
    auto& resource_texture_validated_bytes = binding.resource_texture_validated_bytes;
    auto& resource_texture_source_bytes = binding.resource_texture_source_bytes;
    auto& resource_texture_submit_query = binding.resource_texture_submit_query;
    auto& resource_texture_watch_query = binding.resource_texture_watch_query;
    auto& resource_texture_watch_stability = binding.resource_texture_watch_stability;
    auto& resource_texture_exact_validation = binding.resource_texture_exact_validation;
    auto& resource_texture_watch_active = binding.resource_texture_watch_active;
    auto& resource_texture_watch_disabled = binding.resource_texture_watch_disabled;
    auto& resource_texture_watch_only = binding.resource_texture_watch_only;
    auto& resource_has_live_rtt = binding.resource_has_live_rtt;
    auto& resource_has_ds_live = binding.resource_has_ds_live;
    auto& resource_persistent_candidate = binding.resource_persistent_candidate;
    auto& resource_persistent_source_size = binding.resource_persistent_source_size;
    auto& full_resource = binding.full_resource;
    auto& normalized_sampling = binding.normalized_sampling;
    auto& writable_storage_image = binding.writable_storage_image;
    auto& no_direct_texture_source = binding.no_direct_texture_source;
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
          auto copy_resource = [&](uint8_t* dst, uint64_t addr, size_t n) -> size_t {
              return copy_shader_resource(r, dst, addr, n);
          };
          auto stage_tiled_source =
              [&](prosper::frontend::DecodeScratchPool::Lease& lease, uint64_t addr,
                  size_t n, size_t& got) -> const uint8_t* {
              return stage_tiled_shader_resource(r, no_direct_texture_source, lease, addr, n, got);
          };
          auto copy_dcc_metadata = [&](uint8_t* dst, size_t n) -> size_t {
              return copy_shader_dcc_metadata(r, dst, n);
          };
              auto& fr = *full_resource;
              fr.is_storage_image = r.cls == RC::StorageImage;
              fr.storage_image_numeric_class = reflected_binding->image_numeric_class;
              fr.storage_image_contract_valid = !fr.is_storage_image ||
                  reflected_binding->image_numeric_class !=
                      prosper::gpu::SpirvImageNumericClass::Unknown;
              uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
              const bool float32_layered_texture = r.cls == RC::Texture &&
                  r.img_dim == 5u && r.depth > 1u &&
                  r.format == prosper::gpu::DataFormat::Float32;
              // A layered T# may be consumed by an ordinary DIM=2D instruction.
              // That shader selects the base slice and may use its current color RTT;
              // only an arrayed declaration requires every layer to be materialized.
              // Unknown or incompatible reflection is not proof of a base-slice view.
              if (float32_layered_texture &&
                  (reflected_binding->kind !=
                       prosper::gpu::SpirvDescriptorKind::CombinedImageSampler ||
                   reflected_binding->image_dim != 1u ||
                   reflected_binding->image_multisampled ||
                   reflected_binding->image_depth || !reflected_binding->sampled_float)) {
                  std::fprintf(stderr,
                      "[render-array-reject] binding=%u unsupported Float32 reflected image shape\n",
                      r.binding);
                  return {ImageDisposition::Reject, DropReason::ArrayFloat32Shape};
              }
              const bool float32_array = float32_layered_texture &&
                  reflected_binding->image_arrayed;
              // AvPlayer exposes NV12 as an R8 luma plane followed by an RG8 UV plane.
              // Which resource IS that chroma plane — and why a candidate was rejected —
              // is decided by avplayer_plane_policy.hpp, which carries the reasoning and
              // has its own unit test. A false verdict here is silent: the plane falls
              // into the legacy narrow coverage broadcast, the shader's V becomes its U,
              // and the movie keeps correct luma, detail and geometry while every colour
              // collapses onto one green<->magenta axis. (#2005)
              const AvpChromaVerdict avplayer_chroma_verdict =
                  classify_avplayer_chroma_plane(r, tw, th, t->resources);
              const bool avplayer_chroma_layout = avplayer_chroma_verdict.match;
              // Deliberately NOT filtered by img_dim: a plane the classifier rejects on
              // exactly that field has to appear in this log, or the log cannot report
              // the rejection that matters most.
              if (avp_chroma_log() && r.cls == RC::Texture &&
                  r.format == prosper::gpu::DataFormat::Unorm8 && r.num_components <= 2)
                  log_avp_chroma_candidate(r, tw, th, avplayer_chroma_verdict);
              // PROSPER_AVPCHROMA_DUMP=<dir> — write each candidate plane's exact guest
              // source bytes once, at the resolved row pitch. Converting the two dumped
              // planes on the CPU separates a decode/staging defect (planes wrong) from a
              // sampling/recognition defect (planes right, picture wrong), which is the
              // only way to tell a chroma cast's two very different causes apart.
              if (const char* avp_dump_dir = PROSPER_ENV_VALUE("PROSPER_AVPCHROMA_DUMP")) {
                  if (r.cls == RC::Texture &&
                      r.format == prosper::gpu::DataFormat::Unorm8 &&
                      r.num_components >= 1 && r.num_components <= 2 &&
                      (r.img_dim == 1u || r.img_dim == 5u) &&
                      r.tile_mode == 0u && tw && th) {
                      static std::mutex dump_mx;
                      static std::map<uint64_t, uint64_t> sightings;
                      static uint64_t dump_count = 0;
                      // PROSPER_AVPCHROMA_DUMP_EVERY samples one in N sightings of each
                      // plane allocation (default: only the first). A movie opens on a
                      // fade from black, so the first sighting is a useless all-dark
                      // plane; sampling a series lands somewhere mid-shot without having
                      // to guess a frame index. Capped so a long run cannot fill a disk.
                      const char* every_env = PROSPER_ENV_VALUE("PROSPER_AVPCHROMA_DUMP_EVERY");
                      const uint64_t every = every_env
                          ? std::max<uint64_t>(1, strtoull(every_env, nullptr, 0)) : 0;
                      uint64_t sighting = 0;
                      bool first = false;
                      {
                          std::lock_guard<std::mutex> lock(dump_mx);
                          sighting = ++sightings[r.gpu_addr ^
                                                 (uint64_t{r.num_components} << 56)];
                          first = every ? (sighting % every == 0 && dump_count < 40)
                                        : sighting == 1;
                          if (first) ++dump_count;
                      }
                      if (first) {
                          const uint32_t bpt = r.num_components;
                          const uint32_t tight = tw * bpt;
                          uint32_t pitch = r.linear_row_pitch_bytes;
                          if (!pitch)
                              pitch = prosper::gpu::guest_linear_texture_row_pitch(
                                  r.gpu_addr, tight);
                          if (!pitch)
                              pitch = static_cast<uint32_t>(
                                  prosper::gpu::linear_sampled_row_pitch(tw, bpt));
                          std::vector<uint8_t> plane(
                              static_cast<size_t>(pitch) * th, 0);
                          const size_t got = copy_resource(plane.data(), r.gpu_addr,
                                                           plane.size());
                          char name[512];
                          snprintf(name, sizeof name,
                                   "%s/avpplane_%ux%u_c%u_p%u_%llx_s%06llu.bin",
                                   avp_dump_dir, tw, th, bpt, pitch,
                                   (unsigned long long)r.gpu_addr,
                                   (unsigned long long)sighting);
                          if (FILE* f = fopen(name, "wb")) {
                              fwrite(plane.data(), 1, plane.size(), f);
                              fclose(f);
                              fprintf(stderr,
                                      "[avpchroma] dumped %zu/%zu bytes -> %s\n",
                                      got, plane.size(), name);
                              fflush(stderr);
                          }
                      }
                  }
              }
              const uint64_t sampled_source_addr = texture_decode_source_address(
                  r.gpu_addr, r.img_dim, r.in_mip_tail,
                  r.layer_mip_offset_bytes);
              const bool sampled_2d_view = r.img_dim == 1u || r.img_dim == 5u;
              const size_t msaa_tiled_source_span = r.img_dim == 6u
                  ? prosper::gpu::tiled_msaa_surface_bytes(
                        tw, th, r.tile_mode, 4u, r.sample_count)
                  : 0u;
              TextureDecodeKey decode_key{
                  r.gpu_addr, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(r.host_data)),
                  r.host_data_size, r.size, msaa_tiled_source_span,
                  static_cast<uint32_t>(r.cls),
                  static_cast<uint32_t>(r.format), r.num_components, tw, th, r.depth,
                  r.sample_count,
                  r.tile_mode, r.linear_row_pitch_bytes, r.img_dim,
                  r.mip_tail_bytes, r.mip_tail_x, r.mip_tail_y,
                  r.layer_stride_bytes, r.layer_mip_offset_bytes,
                  r.compression_enabled ? r.max_uncompressed_block_size : 0u,
                  r.compression_enabled ? r.max_compressed_block_size : 0u,
                  r.compression_enabled
                      ? ((r.meta_pipe_aligned ? 1u : 0u) |
                         (r.write_compress_enabled ? 2u : 0u) | 4u |
                         (r.alpha_is_on_msb ? 8u : 0u) |
                         (r.color_transform ? 16u : 0u))
                      : 0u,
                  r.compression_enabled ? r.metadata_addr : 0u,
                  r.compression_enabled
                      ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                            r.dcc_metadata_host_data))
                      : 0u,
                  r.compression_enabled ? r.dcc_metadata_host_data_size : 0u,
                  r.in_mip_tail,
                  avplayer_chroma_layout,
                  static_cast<uint32_t>(reflected_binding->image_numeric_class),
                  reflected_binding->storage_image_format,
                  reflected_binding->image_dim,
                  reflected_binding->image_arrayed,
                  reflected_binding->image_multisampled,
                  reflected_binding->writable,
              };
              if (texref_census) {
                  size_t h = TextureDecodeKeyHash{}(decode_key);
                  auto mix = [&h](uint64_t v) {
                      h ^= static_cast<size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
                  };
                  mix(r.mag_filter); mix(r.min_filter); mix(r.mip_filter);
                  mix(r.addr_uvw[0]); mix(r.addr_uvw[1]); mix(r.addr_uvw[2]);
                  mix(r.border_color_type); mix(r.max_aniso_ratio);
                  uint32_t lod_bits[3];
                  std::memcpy(&lod_bits[0], &r.min_lod, 4);
                  std::memcpy(&lod_bits[1], &r.max_lod, 4);
                  std::memcpy(&lod_bits[2], &r.lod_bias, 4);
                  mix(lod_bits[0]); mix(lod_bits[1]); mix(lod_bits[2]);
                  for (int k = 0; k < 4; ++k) mix(r.swizzle[k]);
                  mix(r.declared_mip_levels); mix(r.srgb); mix(r.depth_compare);
                  mix(static_cast<uint32_t>(reflected_binding->kind));
                  mix(reflected_binding->normalized_sampling);
                  mix(reflected_binding->texel_access);
                  mix(reflected_binding->texel_fetch);
                  mix(reflected_binding->image_depth);
                  mix(reflected_binding->sampled_float);
                  texref_census_key = h;
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kSetupKey);
              }
              if (r.img_dim == 6u) {
                  // GFX10 TYPE=2D_MSAA interleaves the sample coordinate into the tiled
                  // address. It is neither an ordinary 2D texture nor a native Vulkan
                  // multisample upload: IMAGE_LOAD names one sample explicitly, so the
                  // recompiler exposes the samples as four layers of a single-sample 2D
                  // array. Materialize exactly that representation here. Keep the first
                  // live contract deliberately as narrow as the observed Asterix surface;
                  // unsupported sample counts/layouts/formats must remain fail-visible.
                  const bool reflected_msaa_fetch =
                      reflected_binding->kind ==
                          prosper::gpu::SpirvDescriptorKind::CombinedImageSampler &&
                      reflected_binding->image_dim == 1u &&
                      reflected_binding->image_arrayed &&
                      !reflected_binding->image_multisampled &&
                      reflected_binding->texel_access &&
                      !reflected_binding->normalized_sampling;
                  const bool exact_msaa_shape = sampled_msaa_fetch_shape_supported(
                      r, fr.is_storage_image, reflected_msaa_fetch);

                  prosper::gpu::Gfx10HtileMsaaSource htile_source =
                      r.compression_enabled
                          ? prosper::gpu::Gfx10HtileMsaaSource::Unsupported
                          : prosper::gpu::Gfx10HtileMsaaSource::UncompressedBase;
                  uint32_t decompressed_htile_value = 0;
                  std::vector<uint8_t> htile_metadata;
                  size_t htile_metadata_got = 0;
                  if (exact_msaa_shape && r.compression_enabled) {
                      const uint64_t metadata_size =
                          prosper::gpu::gpu_capture_dcc_metadata_footprint(r);
                      htile_metadata.resize(
                          static_cast<size_t>(metadata_size), 0);
                      htile_metadata_got = metadata_size
                          ? copy_dcc_metadata(htile_metadata.data(),
                                              htile_metadata.size())
                          : 0u;
                      // Mode-24 MSAA metadata is HTILE, not color DCC. The two exact PAL
                      // initialization dwords disable depth compression for depth-only and
                      // depth+stencil surfaces respectively. The metadata itself must prove
                      // one uniform state across the complete AddrLib-sized plane; every
                      // compressed, nonuniform, ambiguous, or short state remains rejected.
                      htile_source = prosper::gpu::gfx10_htile_msaa_source(
                          htile_metadata.data(), htile_metadata_got,
                          tw, th, r.tile_mode, 4u, r.sample_count,
                          r.meta_pipe_aligned);
                      if (htile_source ==
                          prosper::gpu::Gfx10HtileMsaaSource::UncompressedBase) {
                          prosper::gpu::gfx10_htile_metadata_is_decompressed(
                              htile_metadata.data(), htile_metadata_got,
                              htile_metadata.size(), &decompressed_htile_value);
                      }
                  }

                  const bool uncompressed_base = htile_source ==
                      prosper::gpu::Gfx10HtileMsaaSource::UncompressedBase;
                  const bool depth_zero_fast_clear = htile_source ==
                      prosper::gpu::Gfx10HtileMsaaSource::DepthZeroFastClear;
                  const bool materializable_msaa = exact_msaa_shape &&
                      (uncompressed_base || depth_zero_fast_clear);
                  const size_t tiled_bytes = exact_msaa_shape && uncompressed_base
                      ? msaa_tiled_source_span : 0u;
                  const uint64_t linear_bytes64 =
                      static_cast<uint64_t>(tw) * th * r.sample_count * 4u;
                  if (!materializable_msaa || !linear_bytes64 ||
                      linear_bytes64 > SIZE_MAX) {
                      static uint32_t rejected = 0;
                      const uint32_t ordinal = ++rejected;
                      if (ordinal <= 32u) {
                          uint32_t htile_first = 0;
                          uint32_t htile_first_different = 0;
                          size_t htile_dwords_different = 0;
                          if (htile_metadata_got == htile_metadata.size() &&
                              htile_metadata.size() >= sizeof(uint32_t) &&
                              (htile_metadata.size() % sizeof(uint32_t)) == 0u) {
                              std::memcpy(&htile_first, htile_metadata.data(),
                                          sizeof(htile_first));
                              for (size_t offset = sizeof(uint32_t);
                                   offset < htile_metadata.size();
                                   offset += sizeof(uint32_t)) {
                                  uint32_t value = 0;
                                  std::memcpy(&value, htile_metadata.data() + offset,
                                              sizeof(value));
                                  if (value != htile_first) {
                                      if (!htile_dwords_different)
                                          htile_first_different = value;
                                      ++htile_dwords_different;
                                  }
                              }
                          }
                          fprintf(stderr,
                                  "[render-msaa-reject] ordinal=%u set=%u binding=%u "
                                  "addr=0x%llx "
                                  "%ux%u samples=%u fmt=%u/%u tile=%u compression=%d "
                                  "shape=%d reflected=%d base-uncompressed=%d "
                                  "depth-zero-fast-clear=%d "
                                  "htile=%zu/%zu first=0x%08x first-different=0x%08x "
                                  "dwords-different=%zu\n",
                                  ordinal, set, r.binding,
                                  (unsigned long long)r.gpu_addr,
                                  tw, th, r.sample_count, (unsigned)r.format,
                                  r.num_components, r.tile_mode,
                                  static_cast<int>(r.compression_enabled),
                                  static_cast<int>(exact_msaa_shape),
                                  static_cast<int>(reflected_msaa_fetch),
                                  static_cast<int>(uncompressed_base),
                                  static_cast<int>(depth_zero_fast_clear),
                                  htile_metadata_got, htile_metadata.size(), htile_first,
                                  htile_first_different, htile_dwords_different);
                      }
                      return {ImageDisposition::Skip};
                  }

                  // Captured backing and DCC metadata cannot use the one-range journal
                  // proof across callback spans. They still reuse safely inside the span
                  // that decoded them. Plain guest backing may cross spans only while the
                  // ordered write journal proves the complete padded tiled allocation was
                  // untouched.
                  const uint64_t cross_span_source_size =
                      !r.host_data && !r.compression_enabled ? tiled_bytes : 0u;
                  auto reused = decoded_textures.find(decode_key);
                  if (reused != decoded_textures.end()) {
                      const bool same_span =
                          reused->second.span == decode_span_ordinal;
                      const prosper::gpu::GuestGpuWriteQuery journal_query = same_span
                          ? prosper::gpu::GuestGpuWriteQuery::Unknown
                          : prosper::gpu::guest_gpu_writes_since(
                                reused->second.snapshot,
                                reused->second.source_addr,
                                reused->second.source_size);
                      if (!submit_local_texture_decode_reusable(
                              reused->second.span, decode_span_ordinal,
                              reused->second.source_addr, reused->second.source_size,
                              sampled_source_addr, cross_span_source_size,
                              journal_query) || !reused->second.pixels_owner) {
                          decoded_textures.erase(reused);
                          reused = decoded_textures.end();
                          ++g_texture_decode_scope.invalidations;
                      }
                  }

                  if (reused != decoded_textures.end()) {
                      fr.tex_rgba_owner = reused->second.pixels_owner;
                      fr.tex_rgba = reused->second.pixels;
                      resource_local_reuse = true;
                      if (reused->second.span == decode_span_ordinal) {
                          ++g_texture_decode_scope.same_span_reuses;
                      } else {
                          ++g_texture_decode_scope.cross_span_reuses;
                          reused->second.snapshot =
                              prosper::gpu::guest_gpu_write_snapshot();
                      }
                      if (timing_enabled) pending_timing.texture_reuses++;
                  } else {
                      std::vector<uint8_t> tiled;
                      size_t copied = 0;
                      if (uncompressed_base) {
                          tiled.resize(tiled_bytes, 0);
                          copied = copy_resource(
                              tiled.data(), sampled_source_addr, tiled.size());
                      }
                      std::vector<uint8_t> linear(
                          static_cast<size_t>(linear_bytes64), 0);
                      prosper::gpu::Gfx10HtileMsaaSource realized_source =
                          prosper::gpu::Gfx10HtileMsaaSource::Unsupported;
                      const bool materialized = r.compression_enabled
                          ? prosper::gpu::materialize_gfx10_htile_msaa_surface(
                                linear.data(), linear.size(),
                                tiled.empty() ? nullptr : tiled.data(), copied,
                                htile_metadata.data(), htile_metadata_got,
                                tw, th, r.tile_mode, 4u, r.sample_count,
                                r.meta_pipe_aligned, &realized_source)
                          : (copied == tiled.size() &&
                             prosper::gpu::detile_msaa_surface(
                                 linear.data(), tiled.data(), copied, tw, th,
                                 r.tile_mode, 4u, r.sample_count));
                      if (!materialized ||
                          (r.compression_enabled && realized_source != htile_source)) {
                          static uint32_t short_sources = 0;
                          if (short_sources++ < 32u)
                              fprintf(stderr,
                                      "[render-msaa-reject] set=%u binding=%u addr=0x%llx "
                                      "source=%zu/%zu bytes\n",
                                      set, r.binding, (unsigned long long)r.gpu_addr,
                                      copied, tiled.size());
                          return {ImageDisposition::Skip};
                      }

                      fr.tex_rgba_owner =
                          std::make_shared<const std::vector<uint8_t>>(std::move(linear));
                      fr.tex_rgba = fr.tex_rgba_owner->data();
                      ++g_texture_decode_scope.decodes;
                      DecodedTexture decoded;
                      decoded.pixels = fr.tex_rgba;
                      decoded.output_height = th;
                      decoded.span = decode_span_ordinal;
                      decoded.snapshot = prosper::gpu::guest_gpu_write_snapshot();
                      decoded.source_addr = sampled_source_addr;
                      decoded.source_size = cross_span_source_size;
                      decoded.pixels_owner = fr.tex_rgba_owner;
                      decoded.texture_format = VK_FORMAT_R32_SFLOAT;
                      decoded_textures.emplace(decode_key, std::move(decoded));
                  }
                  fr.tex_byte_size = fr.tex_rgba_owner->size();
                  fr.tw = tw;
                  fr.th = th;
                  fr.td = 1u;
                  fr.img_dim = r.img_dim;
                  fr.sample_count = r.sample_count;
                  fr.declared_mip_levels = 1u;
                  fr.texture_format = VK_FORMAT_R32_SFLOAT;
                  resource_texture_source_bytes = depth_zero_fast_clear
                      ? htile_metadata.size() : tiled_bytes;
                  if (decompressed_htile_value && getenv("PROSPER_GFXLOG")) {
                      static uint32_t logged = 0;
                      if (logged++ < 16u)
                          fprintf(stderr,
                                  "[render-msaa] addr=0x%llx %ux%u samples=%u "
                                  "decompressed-htile=0x%08x source=%zu bytes\n",
                                  (unsigned long long)r.gpu_addr, tw, th,
                                  r.sample_count, decompressed_htile_value, tiled_bytes);
                  }
                  if (depth_zero_fast_clear && getenv("PROSPER_GFXLOG")) {
                      static uint32_t logged_zero_clear = 0;
                      const uint32_t ordinal = ++logged_zero_clear;
                      if (ordinal <= 16u)
                          fprintf(stderr,
                                  "[render-msaa] ordinal=%u addr=0x%llx %ux%u "
                                  "samples=%u depth-zero-fast-clear metadata=%zu bytes\n",
                                  ordinal, (unsigned long long)r.gpu_addr, tw, th,
                                  r.sample_count, htile_metadata.size());
                  }

                  fr.mag_filter = r.mag_filter;
                  fr.min_filter = r.min_filter;
                  fr.mip_filter = r.mip_filter;
                  fr.addr_uvw[0] = r.addr_uvw[0];
                  fr.addr_uvw[1] = r.addr_uvw[1];
                  fr.addr_uvw[2] = r.addr_uvw[2];
                  fr.border_color_type = r.border_color_type;
                  fr.min_lod = r.min_lod;
                  fr.max_lod = r.max_lod;
                  fr.lod_bias = r.lod_bias;
                  fr.max_aniso_ratio = r.max_aniso_ratio;
                  for (int k = 0; k < 4; ++k) fr.swizzle[k] = r.swizzle[k];
              } else {
              if (g_ever_volume_target && sampled_source_addr) {
                  const uint64_t sampled_bytes = std::max<uint64_t>(
                      1u, prosper::gpu::gpu_capture_resource_footprint(r));
                  const bool unpublished_interior_alias = any_volume_target(g_rtt,
                      [&](uint64_t volume_base, const RttSurf& surface) {
                          return volume_base != sampled_source_addr &&
                              unpublished_volume_may_overlap(
                                  volume_base, surface.volume_guest_bytes,
                                  sampled_source_addr, sampled_bytes);
                      });
                  if (unpublished_interior_alias) {
                      report_volume_sample_drop("interior-alias", sampled_source_addr, sampled_bytes);
                      return {ImageDisposition::Reject, DropReason::VolumeInteriorAlias};
                  }
              }
              auto live_rtt = rtt_on ? g_rtt.find(sampled_source_addr) : g_rtt.end();
              static const uint32_t render_scale = [] {
                  const char* e = PROSPER_ENV_VALUE("PROSPER_RENDER_SCALE");
                  const long v = e ? std::strtol(e, nullptr, 10) : 1;
                  return v > 0 ? static_cast<uint32_t>(v) : 1u;
              }();
              const bool sampled_retained_volume =
                  live_rtt != g_rtt.end() && live_rtt->second.volume_guest_bytes != 0u;
              if (live_rtt != g_rtt.end() &&
                  !prosper::frontend::rtt_sampled_extent_compatible(
                      tw, th, live_rtt->second.w, live_rtt->second.h, render_scale,
                      normalized_sampling)) {
                  live_rtt = g_rtt.end();
              }
              if (sampled_retained_volume && live_rtt == g_rtt.end()) {
                  // A shape mismatch cannot turn a renderer-only volume into valid
                  // guest bytes. Keep the retained producer authoritative even when
                  // this descriptor cannot consume its image.
                  report_volume_sample_drop("shape-mismatch", sampled_source_addr,
                                            0);
                  return {ImageDisposition::Reject, DropReason::VolumeShapeMismatch};
              }
              // A single retained color image proves only one layer. Never reinterpret its
              // CPU snapshot as the complete Float32 array or replace renderer authority
              // with stale guest bytes. Multi-layer color ownership needs a separate proof.
              if (float32_array && live_rtt != g_rtt.end()) {
                  static thread_local uint32_t array_reject_logged = 0;
                  if (log_array_rejection(array_reject_logged, "single-color-rtt"))
                      std::fprintf(stderr,
                          "[render-array-reject] binding=%u Float32 array aliases a single color RTT addr=0x%llx\n",
                          r.binding, (unsigned long long)r.gpu_addr);
                  return {ImageDisposition::Reject, DropReason::ArraySingleColorRtt};
              }
              // Deferred RTT readback (#1284): a GPU-resident target consumed in a way the
              // GPU bind below cannot serve (e.g. format mismatch or storage image) materializes
              // its CPU copy here on demand. Exact 2D attachment feedback is served by the
              // backend's prior-version GPU snapshot rather than by a synchronous readback.
              // The producer
              // ran in an earlier batch — same-batch CPU consumers force the eager readback
              // at defer time — so the queue-ordered copy reads current pixels.
              if (live_rtt != g_rtt.end() && live_gpu_targets && r.img_dim != 2u &&
                  !r.in_mip_tail && live_rtt->second.gpu_valid) {
                  RttSurf& surface = live_rtt->second;
                  const bool sampled_extent_compatible =
                      prosper::frontend::rtt_sampled_extent_compatible(
                          tw, th, surface.w, surface.h, render_scale,
                          normalized_sampling);
                  // Must use the SAME all-slot rule as the gate it feeds. On colour-0
                  // alone an MRT2+ feedback collision set direct_serves=true, which
                  // suppressed the lazy CPU materialisation; the corrected gate below then
                  // refused the direct image because the collision is real, and the
                  // resource fell through to stale guest bytes with no snapshot to use.
                  const bool direct_serves = prosper::frontend::mrt_direct_serves(
                      draw, sampled_source_addr, tw, th,
                      fr.is_storage_image, r.img_dim, r.depth, r.sample_count,
                      sampled_extent_compatible,
                      prosper::test::find_persistent_color_target(
                          sampled_source_addr, surface.w, surface.h,
                          surface.format) != nullptr,
                      mrt_format_defined,
                      /*feedback_copy_supported=*/true);
                  const VkFormat surface_format =
                      prosper::test::backend_color_format(surface.format);
                  const uint32_t surface_bpp =
                      prosper::test::backend_color_bytes_per_pixel(surface_format);
                  const uint64_t surface_texels =
                      static_cast<uint64_t>(surface.w) * surface.h;
                  if (!direct_serves && surface.w && surface.h && surface_bpp &&
                      surface_texels <= UINT64_MAX / surface_bpp &&
                      (!surface.rgba ||
                       surface.rgba->size() != surface_texels * surface_bpp)) {
                      std::vector<uint8_t> materialized;
                      std::string error;
                      if (prosper::test::readback_persistent_color_target(
                              sampled_source_addr, surface.w, surface.h, surface_format,
                              materialized, error) &&
                          materialized.size() == surface_texels * surface_bpp) {
                          surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                              std::move(materialized));
                      } else {
                          static std::atomic<int> warned{0};
                          if (warned.fetch_add(1) < 24)
                              fprintf(stderr,
                                      "[rtt] lazy sampled target readback failed: "
                                      "base=0x%llx extent=%ux%u error=%s\n",
                                      (unsigned long long)sampled_source_addr, surface.w,
                                      surface.h, error.c_str());
                      }
                  }
              }
              static const bool retain_cpu_rtt_snapshots =
                  PROSPER_ENV_VALUE("PROSPER_NO_RTT_SNAPSHOT_BORROW") == nullptr;
              // Pixel-mutating/inspection diagnostics intentionally retain their owned
              // scratch copy. Normal consumers retain exact immutable snapshots directly
              // and share each scaled materialization within this submit callback.
              static const bool cpu_rtt_copy_diagnostics =
                  PROSPER_ENV_VALUE("PROSPER_DUMP_SAMPLED_RTT") || PROSPER_ENV_VALUE("PROSPER_DUMP_RAWTEX") ||
                  getenv("PROSPER_GFXLOG") || PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") ||
                  PROSPER_ENV_VALUE("PROSPER_PALETTELOG") || PROSPER_ENV_VALUE("PROSPER_TESTTEX") ||
                  PROSPER_ENV_VALUE("PROSPER_TESTLUT") || PROSPER_ENV_VALUE("PROSPER_TESTLUT32") ||
                  PROSPER_ENV_VALUE("PROSPER_DUMP_TEX") || PROSPER_ENV_VALUE("PROSPER_DUMP_ATLAS") ||
                  PROSPER_ENV_VALUE("PROSPER_KILL_RING");
              const bool uniform_cpu_diagnostic_path =
                  live_rtt != g_rtt.end() &&
                  prosper::frontend::live_rtt_uniform_uses_cpu_diagnostic_path(
                      live_rtt->second.has_uniform_color,
                      cpu_rtt_copy_diagnostics);
              if (uniform_cpu_diagnostic_path)
                  materialize_uniform_rtt(live_rtt->second);
              const bool has_cpu_live_rtt = !fr.is_storage_image && sampled_2d_view &&
                  !r.in_mip_tail &&
                  live_rtt != g_rtt.end() &&
                  live_rtt->second.w && live_rtt->second.h && live_rtt->second.rgba &&
                  prosper::frontend::live_rtt_cpu_snapshot_matches(
                      live_rtt->second.w, live_rtt->second.h,
                      prosper::test::backend_color_bytes_per_pixel(
                          prosper::test::backend_color_format(live_rtt->second.format)),
                      live_rtt->second.rgba->size());
              // Match the established CPU injection gate below. Cube descriptors and
              // mismatched 2D views can also retain identical materialized bytes; 3D
              // volumes, storage images, and mip tails remain on their specialized paths.
              const bool retain_cpu_live_rtt = retain_cpu_rtt_snapshots &&
                  !cpu_rtt_copy_diagnostics && !fr.is_storage_image &&
                  r.img_dim != 2u && !r.in_mip_tail && live_rtt != g_rtt.end() &&
                  live_rtt->second.w && live_rtt->second.h && live_rtt->second.rgba &&
                  prosper::frontend::live_rtt_cpu_snapshot_matches(
                      live_rtt->second.w, live_rtt->second.h,
                      prosper::test::backend_color_bytes_per_pixel(
                          live_rtt->second.format),
                      live_rtt->second.rgba->size());
              // A retained render target was created for color-attachment + sampled usage,
              // not storage usage. Storage images therefore take the decoded/upload path.
              const bool sampled_volume_rtt = r.img_dim == 2u &&
                  r.sample_count == 1u && r.declared_mip_levels == 1u &&
                  !r.in_mip_tail && live_rtt != g_rtt.end() &&
                  live_rtt->second.volume_depth == r.depth && r.depth != 0u &&
                  !draw_binds_color_target(draw, sampled_source_addr, tw, th);
              const bool has_gpu_live_rtt = !fr.is_storage_image && live_gpu_targets &&
                  ((prosper::frontend::rtt_single_layer_sample_shape(
                        r.img_dim, r.depth, r.sample_count) &&
                    live_rtt != g_rtt.end() &&
                    live_rtt->second.volume_depth == 0u) ||
                   sampled_volume_rtt) &&
                  live_rtt != g_rtt.end() && live_rtt->second.gpu_valid &&
                  prosper::frontend::rtt_sampled_extent_compatible(
                      tw, th, live_rtt->second.w, live_rtt->second.h, render_scale,
                      normalized_sampling) &&
                  prosper::test::find_persistent_color_target(
                      sampled_source_addr, live_rtt->second.w, live_rtt->second.h,
                      live_rtt->second.format, true,
                      sampled_volume_rtt ? r.depth : 0u) != nullptr;
              // A renderer-produced volume has no sound guest-byte substitute while only
              // some slices are complete or the exact retained allocation is unavailable.
              const bool has_uniform_live_rtt = prosper::frontend::mrt_uniform_live_serves(
                  draw, sampled_source_addr, tw, th,
                  /*preconditions=*/!fr.is_storage_image &&
                      !uniform_cpu_diagnostic_path && sampled_2d_view &&
                      !r.in_mip_tail && live_rtt != g_rtt.end() &&
                      live_rtt->second.has_uniform_color && live_rtt->second.w &&
                      live_rtt->second.h &&
                      prosper::frontend::rtt_sampled_extent_compatible(
                          tw, th, live_rtt->second.w, live_rtt->second.h, render_scale,
                          normalized_sampling),
                  mrt_format_defined);
              const bool has_live_rtt =
                  has_cpu_live_rtt || has_gpu_live_rtt || has_uniform_live_rtt;
              if (live_rtt != g_rtt.end() &&
                  prosper::frontend::live_rtt_unpublished_volume_blocks_sample(
                      live_rtt->second.volume_guest_bytes,
                      r.img_dim == 2u || r.depth > 1u,
                      sampled_volume_rtt && has_gpu_live_rtt,
                      has_live_rtt)) {
                  report_volume_sample_drop("no-renderer-image", sampled_source_addr,
                                            live_rtt->second.volume_guest_bytes);
                  return {ImageDisposition::Reject, DropReason::VolumeNoRendererImage};
              }
              // A dim-5 base-slice view may need the CPU injection path rather than a direct
              // Vulkan bind, but the selected renderer target is still authoritative even if
              // an on-demand readback cannot currently materialize it. Never validate/cache a
              // guest decode for that identity: renderer-only writes cannot dirty guest pages.
              const bool has_live_rtt_authority = has_live_rtt ||
                  (!fr.is_storage_image && r.img_dim == 5u && !r.in_mip_tail &&
                   live_rtt != g_rtt.end());
              resource_has_live_rtt = has_live_rtt_authority;
              // The guest-decode cache honours that authority only while the entry holds
              // pixels of its own; an identity-only shell is served from guest bytes either
              // way (live_rtt_base_slice_blocks_decode_cache). PROSPER_NO_RTT_SHELL_DECODE_CACHE=1
              // restores the per-submit re-decode for A/B.
              static const bool rtt_shell_decode_cache =
                  !PROSPER_ENV_ON("PROSPER_NO_RTT_SHELL_DECODE_CACHE");
              const bool rtt_blocks_decode_cache = has_live_rtt ||
                  (has_live_rtt_authority &&
                   (!rtt_shell_decode_cache ||
                    prosper::frontend::live_rtt_base_slice_blocks_decode_cache(
                        live_rtt->second.gpu_valid, live_rtt->second.rgba != nullptr,
                        live_rtt->second.has_uniform_color)));
              if (texref_census)
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kRttProbe);
              // Resolve renderer-owned depth before considering guest-byte texture
              // decoding. A sampled depth attachment has no authoritative color payload
              // in guest memory: the retained Vulkan image is the source of truth. The old
              // ordering first copied DCC metadata and probed the persistent CPU decode
              // cache for every such binding, then discarded that work when this bridge
              // won below. Deferred workloads can bind hundreds of depth views per submit,
              // so select the exact GPU path up front.
              const prosper::test::PersistentDsSampled sampled_ds =
                  !has_live_rtt && !fr.is_storage_image && r.img_dim == 1u &&
                          r.cls == RC::Texture
                      ? prosper::test::find_persistent_ds_sampled(
                            r.gpu_addr, tw, th, render_scale, normalized_sampling)
                      : prosper::test::PersistentDsSampled{};
              const bool has_ds_live = sampled_ds.image != nullptr;
              resource_has_ds_live = has_ds_live;
              // A binding rejected by the gate above never reaches the lookup, so it is
              // absent from every statistic the lookup keeps -- it reads as "we hold no
              // such surface" when we may hold it and be perfectly valid. Name the
              // failing sub-condition, once per (address, reason).
              if (PROSPER_ENV_ON("PROSPER_DSBRIDGE_LOG") && !has_ds_live && !float32_array) {
                  uint32_t held_w = 0, held_h = 0;
                  if (!(!has_live_rtt && !fr.is_storage_image && r.img_dim == 1u &&
                        r.cls == RC::Texture) &&
                      prosper::test::is_retained_ds_plane(r.gpu_addr, &held_w, &held_h)) {
                      static std::mutex gate_mutex;
                      static std::set<std::pair<uint64_t, int>> reported;
                      const int reason = has_live_rtt ? 0
                          : fr.is_storage_image ? 1
                          : r.img_dim != 1u ? 2 : 3;
                      bool first = false;
                      {
                          std::lock_guard lock(gate_mutex);
                          first = reported.emplace(r.gpu_addr, reason).second;
                      }
                      if (first)
                          fprintf(stderr,
                                  "[dsbridge] GATED addr=0x%llx T#=%ux%ux%u dim=%u "
                                  "cls=%u fmt=%u (retained DS %ux%u) never reached the "
                                  "lookup: %s\n",
                                  (unsigned long long)r.gpu_addr, tw, th, r.depth,
                                  r.img_dim, (unsigned)r.cls, (unsigned)r.format,
                                  held_w, held_h,
                                  reason == 0 ? "has_live_rtt (colour cache claims it)"
                                  : reason == 1 ? "is_storage_image"
                                  : reason == 2 ? "img_dim != 1 (not a plain 2D view)"
                                                : "cls != Texture");
                  }
              }
              const bool is_cube = r.img_dim == 3u;   // CUBE: six faces stacked vertically (#273)
              const bool is_volume = r.img_dim == 2u;
              // #325: a guest 2D_ARRAY. Cubes stack their six faces into height because
              // six is small; an array cannot -- this title's level atlas is 256 slices,
              // and 512x131072 exceeds every device's maxImageDimension2D. So arrays ride
              // the backend's real layer channel (`sample_count`) instead, which
              // image creation and the view already follow.
              // prosper decodes BC to RGBA8 on the CPU, so a compressed array inflates 4x
              // and is then held in BOTH host RAM (the persistent cache) and VRAM. This
              // title's world atlas is 512x512x256 Bc7 = 256 MiB decoded, and another
              // resource reaches 464 MiB. Above a budget, decode the base slice only --
              // the pre-#325 behaviour -- rather than failing: an allocation failure here
              // is silent, because vkCreateImage's result is discarded and the null handle
              // is used anyway (#3045).
              const uint64_t array_budget_bytes = array_decode_budget_bytes();
              const uint32_t array_output_bpp = float32_array ? 16u : 4u;
              const uint64_t array_texels = static_cast<uint64_t>(tw) * th;
              const uint64_t array_footprint =
                  array_texels > UINT64_MAX / array_output_bpp / std::max(r.depth, 1u)
                      ? UINT64_MAX
                      : array_texels * array_output_bpp * std::max(r.depth, 1u);
              if (float32_array && (!r.width || !r.height || r.depth > 2048u ||
                  (r.num_components != 1u && r.num_components != 2u && r.num_components != 4u) ||
                  array_footprint > array_budget_bytes)) {
                  static thread_local uint32_t array_reject_logged = 0;
                  if (log_array_rejection(array_reject_logged, "shape-budget"))
                      std::fprintf(stderr,
                          "[render-array-reject] binding=%u Float32 %ux%ux%u components=%u "
                          "compression=%u decoded-bytes=%llu budget=%llu addr=0x%llx\n",
                          r.binding, r.width, r.height, r.depth, r.num_components,
                          unsigned(r.compression_enabled),
                          (unsigned long long)array_footprint,
                          (unsigned long long)array_budget_bytes, (unsigned long long)r.gpu_addr);
                  return {ImageDisposition::Reject, DropReason::ArrayShapeBudget};
              }
              std::shared_ptr<const std::vector<uint8_t>> retained_depth_array;
              std::shared_ptr<prosper::test::PersistentDsDepthArrayGpuImage>
                  retained_depth_array_gpu;
              VkFormat retained_depth_array_format = VK_FORMAT_R32G32B32A32_SFLOAT;
              if (float32_array && !resource_compute_image_hit) {
                  const auto& array_ctx = prosper::test::render_vk_ctx();
                  VkFormatProperties array_properties{};
                  if (array_ctx.ok)
                      vkGetPhysicalDeviceFormatProperties(array_ctx.phys,
                          VK_FORMAT_R32G32B32A32_SFLOAT, &array_properties);
                  const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                      ((r.min_filter || r.mag_filter)
                          ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0u);
                  if (!array_ctx.ok ||
                      (array_properties.optimalTilingFeatures & needed) != needed) {
                      static thread_local uint32_t array_reject_logged = 0;
                      if (log_array_rejection(array_reject_logged, "format-support"))
                          std::fprintf(stderr,
                              "[render-array-reject] binding=%u RGBA32F sampling unsupported addr=0x%llx\n",
                              r.binding, (unsigned long long)r.gpu_addr);
                      return {ImageDisposition::Reject, DropReason::ArrayFormatSupport};
                  }
                  const bool depth_array_shape = r.num_components == 1u &&
                      !r.in_mip_tail && r.layer_mip_offset_bytes == 0u;
                  const bool border_addressing = std::any_of(
                      std::begin(r.addr_uvw), std::end(r.addr_uvw),
                      [](uint32_t mode) { return mode == 6u || mode == 7u; });
                  if (depth_array_shape && compact_depth_array_snapshots && !border_addressing &&
                      reflected_binding && !reflected_binding->texel_fetch) {
                      // Interior missing components become (R,0,0,1) before view swizzle.
                      // Border replacement applies only to PRESENT format components, so
                      // R32 and RGBA32 borders can differ: keep the expanded border route.
                      // Robust out-of-range OpImageFetch has the same alpha distinction,
                      // independently of sampler state; all fetch users retain RGBA32.
                      VkFormatProperties compact_properties{};
                      vkGetPhysicalDeviceFormatProperties(array_ctx.phys,
                          VK_FORMAT_R32_SFLOAT, &compact_properties);
                      if ((compact_properties.optimalTilingFeatures & needed) == needed)
                          retained_depth_array_format = VK_FORMAT_R32_SFLOAT;
                  }
                  bool retained_exact = false;
                  bool retained_subview = false;
                  bool retained_noncanonical = false;
                  using TexCensus = prosper::frontend::TextureReferenceCensus;
                  const uint64_t census_scan_start = texref_census ? TexCensus::aux_begin() : 0;
                  {
                      const prosper::test::BackendPersistentResourceGuard guard;
                      const auto& ds_cache = prosper::test::persistent_ds_cache();
                      std::vector<std::pair<uint32_t, uint64_t>> write_aliases;
                      for (const auto& [key, image] : ds_cache) {
                          // Invalidation addresses layer strides by the read base when
                          // both bases exist. A consumer of a distinct write alias does
                          // not establish a stride for that canonical identity. Only a
                          // matching, selected layer can be an array source candidate.
                          if (key.dw == r.gpu_addr && key.dr && key.dr != r.gpu_addr &&
                              key.w == tw && key.h == th && key.slice < r.depth)
                              write_aliases.emplace_back(key.slice, image.last_depth_write);
                          for (const uint64_t base : {key.dr, key.dw}) {
                              if (!base) continue;
                              if (base == r.gpu_addr) retained_exact = true;
                              // The descriptor is already rebased to its first selected
                              // slice. Its explicit stride can prove overlap with an
                              // actual retained layer, but does not grant a complete view.
                              if (base >= r.gpu_addr || !r.layer_stride_bytes ||
                                  key.w != tw || key.h != th) continue;
                              const uint64_t offset = r.gpu_addr - base;
                              if (offset % r.layer_stride_bytes) continue;
                              const uint64_t first = offset / r.layer_stride_bytes;
                              if (first <= key.slice && key.slice - first < r.depth)
                                  retained_subview = true;
                          }
                      }
                      // The readback selects the newest key for each layer. An older
                      // write-base alias cannot veto a newer canonical read-base plane;
                      // a selected or tied alias cannot prove the stride identity.
                      // Do this extra scan only when an alias exists (normally none).
                      for (const auto& [slice, alias_generation] : write_aliases) {
                          bool superseded = false;
                          for (const auto& [key, image] : ds_cache) {
                              if (key.dr == r.gpu_addr && key.w == tw && key.h == th &&
                                  key.slice == slice &&
                                  image.last_depth_write > alias_generation) {
                                  superseded = true;
                                  break;
                              }
                          }
                          if (!superseded) {
                              retained_noncanonical = true;
                              break;
                          }
                      }
                  }
                  if (texref_census)
                      texref_census->aux_end(TexCensus::kAuxDsScan, census_scan_start);
                  if (retained_noncanonical) {
                      static thread_local uint32_t array_reject_logged = 0;
                      if (log_array_rejection(array_reject_logged, "noncanonical-depth"))
                          std::fprintf(stderr,
                              "[render-array-reject] binding=%u unproven retained depth write-base alias addr=0x%llx\n",
                              r.binding, (unsigned long long)r.gpu_addr);
                      return {ImageDisposition::Reject, DropReason::ArrayNoncanonicalDepth};
                  }
                  if (retained_subview || (retained_exact && !depth_array_shape)) {
                      static thread_local uint32_t array_reject_logged = 0;
                      if (log_array_rejection(array_reject_logged, "depth-view"))
                          std::fprintf(stderr,
                              "[render-array-reject] binding=%u unsupported retained depth view addr=0x%llx\n",
                              r.binding, (unsigned long long)r.gpu_addr);
                      return {ImageDisposition::Reject, DropReason::ArrayDepthView};
                  }
                  if (depth_array_shape) {
                      // The exact, non-rebased consumer proves the depth-plane stride.
                      // Without it a guest write to layer N > 0 is tested against layer
                      // zero's bytes and leaves the retained image incorrectly valid.
                      if (retained_exact) {
                          const prosper::test::BackendPersistentResourceGuard guard;
                          const uint64_t known_stride = prosper::test::ds_layer_stride_for(
                              r.gpu_addr, tw, th);
                          if (r.layer_stride_bytes <
                                  static_cast<uint64_t>(tw) * th * sizeof(float) ||
                              (known_stride && known_stride != r.layer_stride_bytes)) {
                              static thread_local uint32_t array_reject_logged = 0;
                              if (log_array_rejection(array_reject_logged, "depth-stride"))
                                  std::fprintf(stderr,
                                      "[render-array-reject] binding=%u unproven retained depth stride addr=0x%llx\n",
                                      r.binding, (unsigned long long)r.gpu_addr);
                              return {ImageDisposition::Reject, DropReason::ArrayDepthStride};
                          }
                          prosper::test::note_ds_layer_stride_locked(
                              r.gpu_addr, tw, th, r.layer_stride_bytes);
                      }
                      // The GPU route records its copy into this same ordered batch, so
                      // the (speculatively published) source identity is already the one
                      // its copy will observe; only the CPU snapshot needs completion.
                      const bool gpu_route = gpu_depth_array_snapshots && producer_batch;
                      if (!gpu_route && retained_exact && !depth_array_snapshots.empty() &&
                          producer_batch && producer_batch->pending()) {
                          // Complete earlier work before testing the source identity. An
                          // unrelated color pass need not discard an unchanged DS snapshot;
                          // a depth producer changes its generation and misses below.
                          bool completed = false;
                          const uint64_t census_flush_start =
                              texref_census ? TexCensus::aux_begin() : 0;
                          {
                              const prosper::test::BackendPersistentResourceGuard guard;
                              const auto& ctx = prosper::test::render_vk_ctx();
                              if (ctx.ok && !prosper::test::backend_has_unproven_submission()) {
                                  const auto result = producer_batch->submit_and_wait(
                                      ctx.dev, ctx.queue, false);
                                  completed = result.submit_result == VK_SUCCESS &&
                                              result.wait_result == VK_SUCCESS;
                              }
                          }
                          if (texref_census)
                              texref_census->aux_end(TexCensus::kAuxDsFlush, census_flush_start);
                          if (!completed) {
                              clear_depth_array_snapshots();
                              static thread_local uint32_t array_reject_logged = 0;
                              if (log_array_rejection(array_reject_logged, "producer-completion"))
                                  std::fprintf(stderr,
                                      "[render-array-reject] binding=%u retained depth producer did not complete addr=0x%llx\n",
                                      r.binding, (unsigned long long)r.gpu_addr);
                              return {ImageDisposition::Reject, DropReason::ArrayProducerCompletion};
                          }
                          if (depth_array_census.enabled) ++depth_array_census.producer_flushes;
                      }
                      // Layers no retained identity names are read from guest memory
                      // (#3893). HTILE-compressed arrays keep the old all-retained rule:
                      // their guest depth bytes are not the whole sampled value.
                      const uint64_t scan_epoch =
                          guest_gpu_write_drain_epoch().load(std::memory_order_relaxed);
                      if (scan_epoch != depth_array_guest_scan_epoch) {
                          depth_array_guest_scans.entries.clear();
                          depth_array_guest_scan_epoch = scan_epoch;
                      }
                      // A guest-sourced layer whose bytes overlap a live COLOR target
                      // has renderer-owned pixels its guest bytes do not show. The
                      // footprint is an upper bound -- extent padded to whole 256x256
                      // texels (covers any 64 KiB swizzle block), 16 bytes per texel for
                      // a format without a per-texel size, or the proven volume span --
                      // so an uncertain overlap keeps the old refusal.
                      const std::function<bool(uint64_t, size_t)> overlaps_color_target =
                          [&](uint64_t addr, size_t bytes) {
                              for (const auto& [color_base, surf] : g_rtt) {
                                  uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(
                                      surf.format);
                                  if (!bpp) bpp = 16u;
                                  // An unresolved MSAA target stores every sample
                                  // in guest memory (#3906).
                                  const uint64_t padded =
                                      ((static_cast<uint64_t>(surf.w) + 255u) & ~255ull) *
                                      ((static_cast<uint64_t>(surf.h) + 255u) & ~255ull) * bpp *
                                      std::max(surf.samples, 1u);
                                  const uint64_t span =
                                      std::max<uint64_t>(padded, surf.volume_guest_bytes);
                                  if (color_base < addr + bytes &&
                                      addr < color_base + std::max<uint64_t>(span, 1))
                                      return true;
                              }
                              return false;
                          };
                      prosper::test::DepthArrayGuestSource guest_source_storage{
                          r.tile_mode, r.layer_stride_bytes, r.linear_row_pitch_bytes,
                          r.host_data, static_cast<size_t>(r.host_data_size),
                          &depth_array_guest_scans, &overlaps_color_target};
                      const prosper::test::DepthArrayGuestSource* guest_source =
                          retained_exact && !r.compression_enabled && r.layer_stride_bytes &&
                              !disable_guest_depth_layers
                              ? &guest_source_storage : nullptr;
                      const auto source_identity = [&] {
                          const prosper::test::BackendPersistentResourceGuard guard;
                          std::vector<DepthPlaneIdentity> planes;
                          if (prosper::test::backend_has_unproven_submission()) return planes;
                          for (const auto& [key, image] : prosper::test::persistent_ds_cache()) {
                              if ((key.dr != r.gpu_addr && key.dw != r.gpu_addr) ||
                                  key.w != tw || key.h != th || key.slice >= r.depth)
                                  continue;
                              planes.emplace_back(key.dr, key.dw, key.sr, key.sw, key.htile,
                                  key.w, key.h, key.fmt, key.slice, image.image,
                                  image.last_depth_write, image.depth_valid,
                                  image.layout_initialized);
                          }
                          // A guest layer's CONTENT is part of the identity: the memo must
                          // not serve a snapshot taken before the guest rewrote it.
                          if (guest_source && !planes.empty()) {
                              std::vector<prosper::test::DepthArrayGuestLayer> guest;
                              std::string guest_error;
                              if (!prosper::test::collect_depth_array_guest_layers(
                                      r.gpu_addr, tw, th, 0u, r.depth, *guest_source,
                                      guest, guest_error))
                                  return std::vector<DepthPlaneIdentity>{};
                              for (const auto& layer : guest)
                                  planes.emplace_back(layer.addr, layer.fingerprint, 0, 0,
                                      0, tw, th, layer.word, layer.layer, VK_NULL_HANDLE,
                                      0, layer.uniform, false);
                          }
                          return planes;
                      };
                      const uint64_t census_identity_start =
                          texref_census ? TexCensus::aux_begin() : 0;
                      auto planes = source_identity();
                      if (texref_census)
                          texref_census->aux_end(TexCensus::kAuxDsIdentity, census_identity_start);
                      if (!planes.empty()) {
                          for (const auto& cached : depth_array_snapshots) {
                              if (cached.base == r.gpu_addr && cached.stride == r.layer_stride_bytes &&
                                  cached.width == tw && cached.height == th && cached.layers == r.depth &&
                                  cached.format == retained_depth_array_format && cached.planes == planes &&
                                  // A GPU snapshot serves only consumers recorded into
                                  // the batch that carries its copy (see servable_to).
                                  // build_bds' unbatched diagnostic re-renders
                                  // (PROSPER_TARGET_STEP_HASH_DIM, PROSPER_DUMP_DRAWSTEPS)
                                  // currently also disable batching altogether, so this
                                  // guards an invariant rather than a live path.
                                  (!cached.gpu || cached.gpu->servable_to(
                                      gpu_route ? producer_batch : nullptr))) {
                                  retained_depth_array = cached.pixels;
                                  retained_depth_array_gpu = cached.gpu;
                                  if (depth_array_census.enabled) ++depth_array_census.reuses;
                                  break;
                              }
                          }
                      }
                      std::vector<float> depth;
                      std::string error;
                      const bool snapshot_reused = retained_depth_array || retained_depth_array_gpu;
                      const uint64_t census_read_start =
                          texref_census && !snapshot_reused ? TexCensus::aux_begin() : 0;
                      auto status = snapshot_reused
                          ? prosper::test::PersistentDsDepthArrayStatus::Ready
                          : prosper::test::PersistentDsDepthArrayStatus::NoIdentity;
                      bool gpu_fallback = !gpu_route;
                      if (!snapshot_reused && gpu_route) {
                          using GpuResult = prosper::test::DepthArrayGpuResult;
                          const GpuResult result =
                              prosper::test::copy_persistent_ds_depth_array_gpu(
                                  r.gpu_addr, tw, th, 0u, r.depth,
                                  retained_depth_array_format, *producer_batch,
                                  retained_depth_array_gpu, error, guest_source);
                          gpu_fallback = result == GpuResult::Fallback;
                          status = result == GpuResult::Ready
                              ? prosper::test::PersistentDsDepthArrayStatus::Ready
                              : result == GpuResult::Unavailable
                                  ? prosper::test::PersistentDsDepthArrayStatus::Unavailable
                                  : prosper::test::PersistentDsDepthArrayStatus::NoIdentity;
                      }
                      if (!snapshot_reused && gpu_fallback)
                          status = prosper::test::read_persistent_ds_depth_array(
                              r.gpu_addr, tw, th, 0u, r.depth, depth, error, producer_batch,
                              guest_source);
                      if (texref_census)
                          texref_census->aux_end(TexCensus::kAuxDsRead, census_read_start);
                      if (status == prosper::test::PersistentDsDepthArrayStatus::Unavailable) {
                          static thread_local uint32_t array_reject_logged = 0;
                          if (log_array_rejection(array_reject_logged, "depth-unavailable")) {
                              // Name every retained identity at this base, whatever its
                              // shape, so a refusal says which layers are missing (#3893).
                              std::string layers;
                              {
                                  const prosper::test::BackendPersistentResourceGuard guard;
                                  layers = prosper::test::describe_persistent_ds_depth_array_layers(
                                      r.gpu_addr, tw, th, r.depth, r.layer_stride_bytes);
                              }
                              std::fprintf(stderr,
                                  "[render-array-reject] binding=%u retained depth: %s addr=0x%llx "
                                  "compressed=%u stride=%llu %s\n",
                                  r.binding, error.c_str(), (unsigned long long)r.gpu_addr,
                                  unsigned(r.compression_enabled),
                                  (unsigned long long)r.layer_stride_bytes, layers.c_str());
                          }
                          return {ImageDisposition::Reject, DropReason::ArrayDepthUnavailable};
                      }
                      if (status == prosper::test::PersistentDsDepthArrayStatus::Ready &&
                          retained_depth_array_gpu && !snapshot_reused) {
                          if (depth_array_census.enabled) {
                              ++depth_array_census.reads;
                              depth_array_census.payload_bytes +=
                                  static_cast<uint64_t>(tw) * th * r.depth * sizeof(float) *
                                  (retained_depth_array_format == VK_FORMAT_R32_SFLOAT ? 1u : 4u);
                          }
                          // Same admission rule as the CPU snapshot below, charging the
                          // snapshot's real device allocation against the byte budget.
                          const size_t gpu_bytes = static_cast<size_t>(
                              retained_depth_array_gpu->slot->bytes);
                          if (!planes.empty() && planes == source_identity() &&
                              depth_array_snapshots.size() < kDepthArraySnapshotEntries &&
                              gpu_bytes <= kDepthArraySnapshotBytes - depth_array_snapshot_bytes) {
                              depth_array_snapshot_bytes += gpu_bytes;
                              depth_array_snapshots.push_back({r.gpu_addr, r.layer_stride_bytes,
                                  tw, th, r.depth, retained_depth_array_format,
                                  std::move(planes), nullptr, retained_depth_array_gpu});
                              depth_array_snapshot_admitted = true;
                          }
                      }
                      if (status == prosper::test::PersistentDsDepthArrayStatus::Ready &&
                          !retained_depth_array && !retained_depth_array_gpu) {
                          const bool compact = retained_depth_array_format == VK_FORMAT_R32_SFLOAT;
                          auto pixels = std::make_shared<std::vector<uint8_t>>(
                              depth.size() * (compact ? 1u : 4u) * sizeof(float));
                          if (compact) {
                              std::memcpy(pixels->data(), depth.data(), depth.size() * sizeof(float));
                          } else {
                              constexpr uint32_t defaults[4] = {0, 0, 0, 0x3f800000u};
                              for (size_t i = 0; i < depth.size(); ++i) {
                                  auto* pixel = pixels->data() + i * sizeof(defaults);
                                  std::memcpy(pixel, defaults, sizeof(defaults));
                                  std::memcpy(pixel, &depth[i], sizeof(float));
                              }
                          }
                          retained_depth_array = std::move(pixels);
                          if (depth_array_census.enabled) {
                              ++depth_array_census.reads;
                              depth_array_census.payload_bytes += retained_depth_array->size();
                          }
                          // A concurrent writer may have changed the selection while the
                          // synchronous read was in progress. Such a result can serve this
                          // binding but must not be cached under a later source identity.
                          if (!planes.empty() && planes == source_identity() &&
                              depth_array_snapshots.size() < kDepthArraySnapshotEntries &&
                              retained_depth_array->size() <=
                                  kDepthArraySnapshotBytes - depth_array_snapshot_bytes) {
                              depth_array_snapshot_bytes += retained_depth_array->size();
                              depth_array_snapshots.push_back({r.gpu_addr, r.layer_stride_bytes,
                                  tw, th, r.depth, retained_depth_array_format,
                                  std::move(planes), retained_depth_array, nullptr});
                              depth_array_snapshot_admitted = true;
                          }
                      }
                      if (retained_depth_array || retained_depth_array_gpu) {
                          resource_has_ds_live = true;
                          if (PROSPER_ENV_ON("PROSPER_DSBRIDGE_LOG")) {
                              static unsigned reports = 0;
                              if (reports++ < 16u)
                                  std::fprintf(stderr,
                                      "[dsbridge] array addr=0x%llx %ux%ux%u exact-f32%s guest-layers=%zu "
                                      "binding=%u draw-color0=0x%llx %ux%u\n",
                                      (unsigned long long)r.gpu_addr, tw, th, r.depth,
                                      retained_depth_array_gpu ? " gpu" : "",
                                      depth_array_guest_scans.entries.size(), r.binding,
                                      (unsigned long long)draw.color0_base,
                                      draw.color0_width, draw.color0_height);
                          }
                      }
                  }
                  if (r.compression_enabled && !retained_depth_array && !retained_depth_array_gpu) {
                      static thread_local uint32_t array_reject_logged = 0;
                      if (log_array_rejection(array_reject_logged, "compressed-no-depth"))
                          std::fprintf(stderr,
                              "[render-array-reject] binding=%u compressed Float32 array has no retained depth addr=0x%llx\n",
                              r.binding, (unsigned long long)r.gpu_addr);
                      return {ImageDisposition::Reject, DropReason::ArrayCompressedNoDepth};
                  }
              }
              // The SHAPE question -- is this a layered array? -- is what the
              // recompiler also answers, so it must not depend on anything the recompiler
              // cannot see. The budget below decides only how many layers we DECODE.
              const bool guest_array =
                  prosper::gpu::guest_texture_is_uploaded_array(r.img_dim, r.depth,
                                                                r.format) &&
                  r.cls == RC::Texture &&
                  (!float32_layered_texture || float32_array);
              fr.guest_array = guest_array;
              const bool is_array = guest_array &&
                  array_footprint <= array_budget_bytes;
              if (guest_array && array_footprint > array_budget_bytes) {
                  static uint32_t over_budget_reports = 0;
                  if (over_budget_reports++ < 16u)
                      fprintf(stderr,
                              "[array-budget] binding=%u %ux%u x%u layers = %llu MiB "
                              "decoded exceeds %llu MiB; sampling the base slice only\n",
                              r.binding, tw, th, r.depth,
                              (unsigned long long)(array_footprint >> 20),
                              (unsigned long long)(array_budget_bytes >> 20));
              }
              const uint32_t decoded_layers =
                  is_array ? (r.depth ? r.depth : 1u) : 1u;
              // GTA V retains each shadow-cube face as a renderer-owned D32 image, then
              // samples the cube through a UNORM16 descriptor. Guest memory is not the
              // authority for these pixels. Select the complete six-face generation now
              // and include it in the decode identity; unchanged cubes can then reuse the
              // already quantized vertical stack without six synchronous depth readbacks.
              prosper::test::PersistentDsCubeSelection retained_depth_cube;
              bool retained_depth_cube_cache_candidate = false;
              if (is_cube && r.depth == 6u &&
                  r.format == prosper::gpu::DataFormat::Unorm16 &&
                  r.num_components == 1u && r.cls == RC::Texture && !r.host_data &&
                  !has_live_rtt && !has_ds_live && !fr.is_storage_image &&
                  prosper::test::is_retained_ds_plane(r.gpu_addr)) {
                  if (r.layer_stride_bytes)
                      prosper::test::note_ds_layer_stride(
                          r.gpu_addr, tw, th, r.layer_stride_bytes);
                  retained_depth_cube =
                      prosper::test::select_persistent_ds_cube_depth(
                          r.gpu_addr, tw, th);
                  retained_depth_cube_cache_candidate =
                      retained_depth_cube.present_mask != 0 &&
                      retained_depth_cube.overlay_version != 0;
                  if (retained_depth_cube_cache_candidate)
                      decode_key.renderer_overlay_version =
                          (static_cast<uint64_t>(retained_depth_cube.present_mask) << 56) |
                          (retained_depth_cube.overlay_version & 0x00ffffffffffffffull);
              }
              if (texref_census)
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kDepthProbe);
              const uint32_t persistent_pitch = PROSPER_ENV_VALUE("PROSPER_PITCH")
                  ? static_cast<uint32_t>(atoi(getenv("PROSPER_PITCH"))) : 0;
              const uint32_t persistent_bc_block_bytes =
                  prosper::gpu::bc_block_bytes(r.format);
              const bool persistent_unorm8_texture =
                  r.format == prosper::gpu::DataFormat::Unorm8 &&
                  r.num_components >= 1 && r.num_components <= 4;
              // UNORM16 inputs follow the same deterministic guest-byte decode contract.
              // Retaining the already-decoded pixels does not change the existing channel
              // expansion/swizzle semantics; it only avoids repeating them for immutable data.
              const bool persistent_unorm16_texture =
                  r.format == prosper::gpu::DataFormat::Unorm16 &&
                  r.num_components >= 1 && r.num_components <= 2;
              // Float16/HDR sampled textures decode to RGBA8 via a dedicated branch below but were
              // excluded from the persistent decode cache, so HDR art was re-detiled every frame
              // (#1177, the largest `notsampled` bucket). The cache machinery is format-agnostic —
              // it stores the decoded RGBA8 and validates against the raw source bytes — so caching
              // fp16 is correctness-preserving as long as persistent_source_size counts the fp16
              // source (2 B/component) exactly, which is what the decode reads.
              // Must match the fp16 DECODE branch's `f16` predicate exactly (bpt in {2,4,8} =
              // 1/2/4 components); RGB16F (3 comp, bpt 6) takes a different decode path that reads
              // a different byte count, so it must NOT be cached here or the exact-byte source
              // validation would compare the wrong region.
              const bool persistent_fp16_texture =
                  r.format == prosper::gpu::DataFormat::Float16 &&
                  (r.num_components == 1 || r.num_components == 2 || r.num_components == 4);
              // Float32 sampled textures narrow to RGBA16F below. Like the existing fp16 path,
              // the decoded result is fully determined by the exact source bytes and descriptor
              // shape, so immutable versions can use the same validated cross-submit cache.
              // This matters for titles that repeatedly bind large Float32 post-process inputs:
              // otherwise every callback re-detiles and scalar-narrows the complete surface.
              const bool persistent_fp32_texture =
                  r.format == prosper::gpu::DataFormat::Float32 &&
                  (r.num_components == 1 || r.num_components == 2 || r.num_components == 4);
              // Exact single-channel integer compute outputs may be consumed immediately
              // by graphics. Their guest bytes and native Vulkan texels have the same
              // width, so the existing submit journal can authorize a device-local image
              // lease instead of a writeback -> detile -> upload round trip.
              const bool persistent_uint8_texture =
                  r.format == prosper::gpu::DataFormat::Uint8 &&
                  r.num_components == 1;
              const bool persistent_uint32_texture =
                  r.format == prosper::gpu::DataFormat::Uint32 &&
                  r.num_components == 1;
              // Packed 32-bit sampled formats are also pure decode inputs.  Keeping their
              // decoded pixels is especially important for full-resolution HDR intermediates:
              // unpacking R11G11B10F scalar-by-scalar on every callback otherwise dominates an
              // entire frame.  Their encoded source is exactly one dword per texel regardless
              // of the logical component count.
              const bool persistent_packed32_texture =
                  (r.format == prosper::gpu::DataFormat::Float10_11_11 &&
                   r.num_components == 3) ||
                  (r.format == prosper::gpu::DataFormat::Unorm2_10_10_10 &&
                   r.num_components == 4);
              // Real source bytes per texel. It also determines the visible row size used to
              // resolve a guest-backed linear image's pitch below: ordinary sampled images use
              // GFX10's 256-byte alignment, while exact HLE-producer provenance may stay tight.
              uint32_t sampled_source_bpt =
                  prosper::gpu::data_format_bytes(r.format) *
                  (r.num_components ? r.num_components : 1u);
              const bool sampled_source_f16 =
                  r.format == prosper::gpu::DataFormat::Float16 &&
                  (sampled_source_bpt == 2 || sampled_source_bpt == 4 || sampled_source_bpt == 8);
              const bool sampled_source_f32 =
                  r.cls == RC::Texture && r.img_dim != 3u && !r.compression_enabled &&
                  r.format == prosper::gpu::DataFormat::Float32 &&
                  (sampled_source_bpt == 4 || sampled_source_bpt == 8 ||
                   sampled_source_bpt == 16);
              if (sampled_source_bpt == 0 ||
                  (sampled_source_bpt > 4 && !sampled_source_f16 && !sampled_source_f32))
                  sampled_source_bpt = 4;
              const bool persistent_format_supported =
                  persistent_unorm8_texture || persistent_fp16_texture ||
                  persistent_unorm16_texture || persistent_fp32_texture ||
                  persistent_uint8_texture || persistent_uint32_texture ||
                  persistent_packed32_texture ||
                  (persistent_bc_block_bytes != 0 && !is_volume);
              // GTA V's five shadow cubes are ordinary tiled Z16 allocations. The fallback
              // decoder below reads all six faces from one exact descriptor footprint, so
              // unlike the broad non-BC cube class that regressed Plucky Squire this one
              // narrow shape can use exact-byte validation safely.
              const bool exact_unorm16_cube = is_cube && r.depth == 6u &&
                  persistent_unorm16_texture && r.num_components == 1u &&
                  !r.compression_enabled;
              const bool persistent_sampled_texture = texture_decode_cache_candidate(
                  rtt_blocks_decode_cache,
                  has_ds_live || retained_depth_array != nullptr ||
                      retained_depth_array_gpu != nullptr,
                  r.host_data != nullptr, r.img_dim,
                  r.cls == RC::Texture, persistent_format_supported,
                  persistent_bc_block_bytes != 0, exact_unorm16_cube);
              resource_persistent_candidate = persistent_sampled_texture ||
                  retained_depth_cube_cache_candidate;
              const bool persistent_source_is_tiled =
                  persistent_sampled_texture && !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                  prosper::gpu::tile_mode_is_tiled(r.tile_mode);
              // A sampled LINEAR (tile_mode 0) 2D surface (or the selected base slice of a
              // 2D array) whose tight row is not 256-aligned
              // is pitch-padded: RDNA2 requires a sampled linear texture's row pitch to be aligned
              // (256 B here), so its real pitch aligns either texel rows or BC block rows. The
              // decode below must read it row-by-row at that pitch (else every row drifts ->
              // horizontal scramble, e.g. Dead Cells' 348-wide Motion Twin splash whose real
              // pitch is 384 texels). Guards keep
              // this narrow and regression-safe: 2D or the already-selected 2D-array base slice
              // only (volume/cube layouts untouched); tile_mode == 0 exactly, so
              // unrecognized/actually-tiled modes are NOT strided (they fall to the contiguous
              // read + auto-detile pass); exact HLE-producer provenance can select a tight guest
              // layout (AvPlayer NV12); !host_data keeps ordinary CPU-uploaded test fixtures
              // contiguous unless capture replay supplies an explicit guest-layout pitch.
              // The tightly-packed
              // LINEAR_GENERAL layout is a buffer/copy layout, not a sampled-texture layout, so it
              // does not reach here. r.size is the TIGHT extent (tw*th*bpp), so it cannot gate this.
              const uint32_t linear_row_width = persistent_bc_block_bytes
                  ? tw / 4u + static_cast<uint32_t>(tw % 4u != 0u) : tw;
              const uint32_t linear_row_count = persistent_bc_block_bytes
                  ? th / 4u + static_cast<uint32_t>(th % 4u != 0u) : th;
              const uint32_t linear_row_element_bytes = persistent_bc_block_bytes
                  ? persistent_bc_block_bytes : sampled_source_bpt;
              const size_t linear_dst_row =
                  static_cast<size_t>(linear_row_width) * linear_row_element_bytes;
              const uint32_t registered_linear_pitch = linear_dst_row <= UINT32_MAX
                  ? prosper::gpu::guest_linear_texture_row_pitch(
                        r.gpu_addr, static_cast<uint32_t>(linear_dst_row))
                  : 0;
              size_t linear_src_row = r.linear_row_pitch_bytes
                  ? r.linear_row_pitch_bytes
                  : (registered_linear_pitch
                         ? registered_linear_pitch
                         : prosper::gpu::linear_sampled_row_pitch(
                               linear_row_width, linear_row_element_bytes));
              if (const char* lp = PROSPER_ENV_VALUE("PROSPER_LINPITCH"))
                  linear_src_row =
                      (size_t)strtoull(lp, nullptr, 0) * linear_row_element_bytes;
              const bool linear_padded_read =
                  r.cls == RC::Texture && (r.img_dim == 1u || r.img_dim == 5u) &&
                  r.tile_mode == 0 &&
                  (!r.host_data || r.linear_row_pitch_bytes != 0) &&
                  !r.compression_enabled &&
                  linear_dst_row != 0 && linear_src_row > linear_dst_row;
              // Dynamic single-channel video/coverage surfaces are already exactly what a
              // VK_FORMAT_R8_UNORM sampled image consumes. Expanding every byte to RGBA on the
              // CPU multiplied Astro Bot's 3840x3240 FMV traffic by four and cost 26-31 ms per
              // frame. Keep the historical grayscale broadcast through the image-view swizzle.
              // img_dim is tested through avp_plane_is_one_layer_2d rather than
              // `== 1u`: a title may declare a single-channel plane as a ONE-LAYER 2D
              // ARRAY (dim 5, depth 1), which is the same memory and the same sampled
              // values but missed this fast path and was CPU-expanded to RGBA8 --
              // R-Type Delta (PPSA26414) ships its AvPlayer luma plane that way, so
              // every movie frame moved 4x the bytes it needed (#2034). The helper
              // also requires no layer stride and no layer mip offset, so a REAL
              // multi-layer array still fails and keeps the expanded path.
              const bool native_r8_sampled =
                  r.cls == RC::Texture && prosper::frontend::avp_plane_is_one_layer_2d(r) &&
                  r.format == prosper::gpu::DataFormat::Unorm8 &&
                  r.num_components == 1 && r.tile_mode == 0u &&
                  !r.compression_enabled && !linear_padded_read;
              // AvPlayer's exact chroma plane is interleaved RG8. Keeping it native halves
              // the upload bytes and avoids CPU RGBA expansion while the descriptor swizzle
              // preserves U/V. A pitch-padded source is copied row-by-row below.
              //
              // LINEAR ONLY, and the classifier no longer implies it: since #2731 a
              // recognised chroma plane may be GPU-TILED (Sonic Origins stages both NV12
              // planes SW_64KB_S). This branch is a straight copy with no de-swizzle, so a
              // tiled plane taken through it would upload micro-tile order as if it were
              // scanline order -- a woven picture, worse than the collapse it replaced.
              // A tiled plane instead falls to the `bpt < 4` path below, which detiles at
              // the real 2-byte element size and then preserves both U and V because
              // avplayer_chroma_layout is set.
              const bool native_rg8_sampled = avplayer_chroma_layout && r.tile_mode == 0u;
              // R11G11B10F is already a native Vulkan sampled format. Preserve the packed
              // words after detiling instead of expanding to RGBA8: the old conversion
              // clamped every HDR channel above 1.0, destroying environment-light energy.
              // GTA V's 1024x512 lighting map has 2,536 of 6,123 sampled channels above
              // 1.0 (1,791 above 16.0), so this was a semantic collapse, not quantization.
              const bool native_r11_eligible =
                  native_r11_sampled_upload_supported(r);
              const bool native_r11_sampled = native_r11_eligible &&
                  PROSPER_ENV_VALUE("PROSPER_NO_NATIVE_R11_SAMPLED") == nullptr &&
                  native_r11_sampled_selector().includes(r.gpu_addr);
              if (native_r11_eligible &&
                  PROSPER_ENV_VALUE("PROSPER_NATIVE_R11_SAMPLED_LOG") != nullptr) {
                  static std::set<uint64_t> logged_native_r11;
                  if (logged_native_r11.insert(r.gpu_addr).second)
                      std::fprintf(stderr,
                                   "[native-r11] addr=0x%llx %ux%ux%u tile=%u "
                                   "declared_mips=%u selected=%u\n",
                                   static_cast<unsigned long long>(r.gpu_addr),
                                   r.width, r.height, r.depth, r.tile_mode,
                                   r.declared_mip_levels,
                                   native_r11_sampled ? 1u : 0u);
              }
              // Native BCn (#3873): upload the guest's detiled 4x4 blocks as a Vulkan BC
              // image instead of decoding every texel to RGBA8 on the CPU -- 8x fewer
              // bytes to stage, upload and keep resident for BC1/BC4, 4x for the rest,
              // and no decode. The detiling is the one the decoder already uses, so only
              // the final step differs. Everything the renderer owns instead of guest
              // memory (RTT, depth bridge, compute image) keeps its own path.
              // A declared mip chain goes native only when its guest levels can be
              // placed exactly (tiled, power-of-two, live guest memory): the backend then
              // uploads those levels instead of generating them. Any other chain keeps the
              // decoder, whose RGBA8 level 0 the backend can still blit-generate from.
              // PROSPER_NO_NATIVE_BC_MIP_CHAIN=1 sends every chain back to the decoder.
              const uint32_t native_bc_block = prosper::gpu::bc_block_bytes(r.format);
              const prosper::gpu::MipChainPlan* native_bc_plan =
                  native_bc_block && r.declared_mip_levels > 1u && !r.host_data &&
                          native_bc_sampled_enabled() &&
                          prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                          !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                          !PROSPER_ENV_ON("PROSPER_NO_NATIVE_BC_MIP_CHAIN")
                      ? &native_bc_mip_chain_plan(r, native_bc_block) : nullptr;
              const bool native_bc_chain_modelled =
                  native_bc_plan && native_bc_plan->valid &&
                  native_bc_plan->level_count > 1u;
              const VkFormat native_bc_format = static_cast<VkFormat>(
                  native_bc_sampled_format(r, native_bc_chain_modelled));
              const bool native_bc_sampled =
                  native_bc_format != VK_FORMAT_UNDEFINED && native_bc_sampled_enabled() &&
                  !fr.is_storage_image && !guest_array && !is_cube && !is_volume &&
                  fr.sample_count <= 1u && !has_live_rtt_authority && !has_ds_live &&
                  prosper::test::backend_native_bc_sampled_supported(native_bc_format);
              const bool native_bc_mip_chain = native_bc_sampled && native_bc_chain_modelled;
              const uint32_t native_bc_chain_levels =
                  native_bc_mip_chain ? native_bc_plan->level_count : 0u;
              const uint64_t native_bc_chain_base = native_bc_mip_chain
                  ? sampled_source_addr - native_bc_plan->levels[0].byte_offset : 0u;
              // PROSPER_NATIVE_BC_LOG=1: one line per BC texture identity, naming why it
              // did or did not take the native path, for a per-title coverage census.
              if (PROSPER_ENV_ON("PROSPER_NATIVE_BC_LOG") &&
                  prosper::gpu::bc_block_bytes(r.format) != 0u) {
                  static std::set<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>> seen;
                  if (seen.size() < 200000 &&
                      seen.emplace(r.gpu_addr, tw, th, static_cast<uint32_t>(r.format)).second)
                      std::fprintf(stderr,
                          "[native-bc] addr=0x%llx fmt=%u %ux%u dim=%u depth=%u mips=%u "
                          "tile=%u native=%u shape=%u rtt=%u storage=%u chain=%u\n",
                          static_cast<unsigned long long>(r.gpu_addr),
                          static_cast<unsigned>(r.format), tw, th, r.img_dim, r.depth,
                          r.declared_mip_levels, r.tile_mode, native_bc_sampled ? 1u : 0u,
                          native_bc_format != VK_FORMAT_UNDEFINED ? 1u : 0u,
                          has_live_rtt_authority ? 1u : 0u,
                          fr.is_storage_image ? 1u : 0u, native_bc_chain_levels);
              }
              // Storage-image atomics require a typed integer Vulkan view. Keep R32_UINT
              // texels byte-exact through the existing 4-B read/detile path instead of
              // silently normalizing the view to RGBA8_UNORM.
              const bool native_r32ui_storage =
                  r.cls == RC::StorageImage && r.img_dim == 1u &&
                  r.format == prosper::gpu::DataFormat::Uint32 &&
                  r.num_components == 1 && !r.compression_enabled;
              // The portable graphics recompiler declares formatless uvec4 storage images.
              // Their UINT Sampled Type is not compatible with the renderer's historical
              // RGBA8_UNORM upload even when the same undefined read happened to return useful
              // pixels. Materialize the same raw-channel ABI as live_compute instead.
              const bool portable_raw_uvec4_storage =
                  fr.is_storage_image &&
                  reflected_binding->image_numeric_class ==
                      prosper::gpu::SpirvImageNumericClass::Uint &&
                  reflected_binding->storage_image_format == 0u;
              const uint32_t portable_storage_guest_texel =
                  portable_raw_uvec4_storage
                      ? storage_image_guest_texel_bytes(
                            r.format, r.num_components ? r.num_components : 1u)
                      : 0u;
              const bool portable_storage_shape =
                  !r.compression_enabled &&
                  !reflected_binding->image_arrayed &&
                  !reflected_binding->image_multisampled &&
                  ((reflected_binding->image_dim == 1u && r.img_dim == 1u) ||
                   (reflected_binding->image_dim == 2u && r.img_dim == 2u &&
                    r.depth != 0u));
              // A storage-image contract failure does not skip one binding -- the backend
              // returns an EMPTY batch, dropping every draw submitted with it
              // (render_runner.h, `return out;` in the pre-Vulkan validation loop). So
              // "which sub-condition failed" is the difference between one unusable
              // resource and a whole frame's geometry going missing, and the existing line
              // reports only `materialized=0`, which is the conjunction.
              if (portable_raw_uvec4_storage &&
                  (!portable_storage_guest_texel || !portable_storage_shape)) {
                  static std::set<std::tuple<uint32_t, uint32_t, uint32_t>> reported;
                  if (reported.emplace(fr.set, fr.binding,
                                       static_cast<uint32_t>(r.format)).second)
                      std::fprintf(stderr,
                          "[render] storage-image contract: set=%u binding=%u "
                          "portable-uvec4 REJECTED guest-texel=%u shape=%u "
                          "(writable=%u compressed=%u arrayed=%u multisampled=%u "
                          "reflected-dim=%u guest-dim=%u depth=%u fmt=%u comps=%u)\n",
                          fr.set, fr.binding, portable_storage_guest_texel,
                          portable_storage_shape ? 1u : 0u,
                          writable_storage_image ? 1u : 0u,
                          r.compression_enabled ? 1u : 0u,
                          reflected_binding->image_arrayed ? 1u : 0u,
                          reflected_binding->image_multisampled ? 1u : 0u,
                          reflected_binding->image_dim, r.img_dim, r.depth,
                          static_cast<unsigned>(r.format), r.num_components);
                  fr.storage_image_contract_valid = false;
              }
              if (fr.is_storage_image &&
                  reflected_binding->image_numeric_class ==
                      prosper::gpu::SpirvImageNumericClass::Uint &&
                  reflected_binding->storage_image_format ==
                      prosper::gpu::kSpirvImageFormatR32ui &&
                  !native_r32ui_storage)
                  fr.storage_image_contract_valid = false;
              if (fr.is_storage_image &&
                  reflected_binding->image_numeric_class ==
                      prosper::gpu::SpirvImageNumericClass::Sint)
                  fr.storage_image_contract_valid = false;
              const VkFormat decoded_texture_format = float32_array
                  ? VK_FORMAT_R32G32B32A32_SFLOAT : has_cpu_live_rtt
                  ? live_rtt->second.format
                  : (portable_raw_uvec4_storage
                         ? VK_FORMAT_R32G32B32A32_UINT
                     : (native_r32ui_storage ? VK_FORMAT_R32_UINT
                     : (native_r8_sampled ? VK_FORMAT_R8_UNORM
                     : (native_rg8_sampled ? VK_FORMAT_R8G8_UNORM
                     : (native_r11_sampled ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                     : (native_bc_sampled ? native_bc_format
                     : (sampled_source_f32 ? VK_FORMAT_R16G16B16A16_SFLOAT
                                           : VK_FORMAT_R8G8B8A8_UNORM)))))));
              // A LINEAR, unpadded native BC surface is uploaded byte-for-byte as the guest
              // stores it, so the cached pixels are themselves the validation source.
              const bool native_bc_linear_copy = native_bc_sampled &&
                  !persistent_source_is_tiled && !linear_padded_read && !r.in_mip_tail &&
                  !(prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                    !PROSPER_ENV_VALUE("PROSPER_NODETILE"));
              const bool persistent_source_matches_pixels =
                  native_r8_sampled || native_rg8_sampled || native_bc_linear_copy ||
                  (persistent_unorm8_texture && r.num_components == 4 &&
                   !linear_padded_read &&
                   !prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                   (!PROSPER_ENV_VALUE("PROSPER_DETILE") || atoi(PROSPER_ENV_VALUE("PROSPER_DETILE")) == 0));
              const size_t persistent_base_source_size = [&] {
                  if (!persistent_sampled_texture) return size_t{0};
                  if (is_cube)
                      return layered_cube_source_size(
                          persistent_bc_block_bytes != 0 || exact_unorm16_cube,
                          sampled_source_addr,
                          prosper::gpu::gpu_capture_resource_footprint(r));
                  // One SURFACE's bytes. For a layered array the decode reads many of
                  // these, so the layered span is applied below -- see #2998.
                  const size_t surface_bytes = [&] {
                  if (persistent_bc_block_bytes) {
                      const uint32_t bw = (tw + 3) / 4;
                      const uint32_t bh = (th + 3) / 4;
                      return persistent_source_is_tiled
                          ? prosper::gpu::tiled_elements_bytes(
                                bw, bh, persistent_bc_block_bytes, r.tile_mode)
                          : (linear_padded_read
                                 ? linear_src_row * linear_row_count
                                 : static_cast<size_t>(bw) * bh *
                                       persistent_bc_block_bytes);
                  }
                  if (persistent_packed32_texture)
                      return persistent_source_is_tiled
                          ? (is_volume
                                 ? prosper::gpu::tiled_volume_bytes(
                                       tw, th, r.depth, r.tile_mode, 4u)
                                 : prosper::gpu::tiled_surface_bytes(
                                       tw, th, r.tile_mode, persistent_pitch, 4u))
                          : static_cast<size_t>(tw) * th *
                                (is_volume ? r.depth : 1u) * 4u;
                  // fp32 sources are 4 B/component, fp16/unorm16 are 2, and unorm8 is 1.
                  const uint32_t source_component_bytes =
                      (persistent_fp32_texture || persistent_uint32_texture) ? 4u
                      : ((persistent_fp16_texture || persistent_unorm16_texture) ? 2u : 1u);
                  const uint32_t source_bpt = r.num_components * source_component_bytes;
                  if (persistent_source_is_tiled)
                      return is_volume
                          ? prosper::gpu::tiled_volume_bytes(
                                tw, th, r.depth, r.tile_mode, source_bpt)
                          : prosper::gpu::tiled_surface_bytes(
                                tw, th, r.tile_mode, persistent_pitch, source_bpt);
                  // Forced detiling treats a nominally linear Unorm8x4 descriptor as tiled; its
                  // decode source is then not the linear byte range computed here. This exclusion
                  // is unorm8-specific — fp16 (which can also land at source_bpt==4 for RG16F)
                  // always reads exactly tw*th*source_bpt and must NOT be excluded.
                  if (persistent_unorm8_texture && source_bpt == 4 &&
                      !persistent_source_matches_pixels) return size_t{0};
                  if (linear_padded_read) return linear_src_row * th;
                  return static_cast<size_t>(tw) * th *
                      (is_volume ? r.depth : 1u) * source_bpt;
                  }();
                  // An uploaded ARRAY's decode reads EVERY layer, so the range the
                  // persistent cache validates has to span them all. Returning one
                  // surface made it compare 262144 of 90177536 bytes -- 0.29% -- of Tomb
                  // Raider's 256-layer world atlas: every layer above the first then
                  // changed invisibly, and a decode taken while the atlas was about 5%
                  // filled was reused for the rest of the run, which is why interiors
                  // sampled slices that held the previous occupants of that memory
                  // (#2998). The `is_cube` branch above has always spanned its six faces;
                  // the array path was added later without the matching change.
                  //
                  // A surface_bytes of 0 is a deliberate "do not cache this" signal from
                  // the branches above, so it must not be turned into a nonzero span.
                  if (guest_array)
                      return prosper::frontend::layered_array_source_size(
                          surface_bytes, r.layer_stride_bytes, decoded_layers);
                  return surface_bytes;
              }();
              // DCC code 0xff means the base allocation contains ordinary uncompressed
              // texels.  Such an image is just as cacheable as a descriptor with DCC disabled,
              // provided we re-check the complete metadata plane before every reuse.  Other
              // metadata states remain on the existing fast-clear/unsupported paths: caching
              // them from base bytes alone would miss a metadata-only content transition.
              static const bool prune_unused_texture_preparation =
                  PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_PREPARATION_PRUNING") == nullptr;
              // These branches already own the sampled pixels. DCC describes guest
              // backing that neither winner consumes. CPU-only RTTs retain preparation:
              // their injection can fail and still needs the ordinary guest fallback.
              const uint64_t sampled_dcc_metadata_size =
                  !has_ds_live && r.compression_enabled &&
                  !(prune_unused_texture_preparation &&
                    (has_gpu_live_rtt || has_uniform_live_rtt))
                  ? prosper::gpu::gpu_capture_dcc_metadata_footprint(r)
                  : 0u;
              std::vector<uint8_t> sampled_dcc_metadata(
                  static_cast<size_t>(sampled_dcc_metadata_size), 0);
              const size_t sampled_dcc_metadata_got = sampled_dcc_metadata_size
                  ? copy_dcc_metadata(sampled_dcc_metadata.data(),
                                      sampled_dcc_metadata.size())
                  : 0u;
              if (sampled_dcc_metadata_size) {
                  ++g_texture_decode_scope.dcc_metadata_read_attempts;
                  g_texture_decode_scope.dcc_metadata_read_bytes += sampled_dcc_metadata_got;
              }
              const bool persistent_dcc_uncompressed = r.compression_enabled &&
                  sampled_dcc_metadata_got == sampled_dcc_metadata.size() &&
                  !sampled_dcc_metadata.empty() &&
                  std::all_of(sampled_dcc_metadata.begin(), sampled_dcc_metadata.end(),
                              [](uint8_t code) { return code == 0xff; });
              uint8_t persistent_dcc_clear_pixel[4]{};
              const bool persistent_dcc_fast_clear = r.compression_enabled &&
                  sampled_dcc_metadata_got == sampled_dcc_metadata.size() &&
                  prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                      persistent_dcc_clear_pixel, 1,
                      sampled_dcc_metadata.data(), sampled_dcc_metadata.size(),
                      r.num_components, r.alpha_is_on_msb);
              // A native BC chain's pixels depend on every level, so the cache validates
              // (and watches) the whole allocation, not level 0 alone.
              const bool persistent_chain_source =
                  native_bc_mip_chain && persistent_base_source_size != 0;
              const uint64_t persistent_source_addr = persistent_dcc_fast_clear
                  ? r.metadata_addr
                  : (persistent_chain_source ? native_bc_chain_base : sampled_source_addr);
              const size_t persistent_source_size = persistent_dcc_fast_clear
                  ? sampled_dcc_metadata.size()
                  : (persistent_chain_source
                         ? static_cast<size_t>(native_bc_plan->allocation_bytes)
                         : persistent_base_source_size);
              resource_persistent_source_size = persistent_source_size;
              resource_texture_source_bytes = persistent_source_size;
              if (texref_census)
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kSourceSize);
              // A successful typed-storage compute dispatch may still own this exact
              // sampled image on the shared Vulkan device. Prefer that image only for the
              // exact native formats the graphics backend preserves today. The compute cache
              // independently requires the complete descriptor key plus a current-submit
              // journal or page-watch proof; a miss falls through to the existing exact
              // guest-byte decode/cache path.
              prosper::frontend::LiveComputeImageImport compute_image_import;
              const VkFormat compute_image_format = static_cast<VkFormat>(
                  prosper::frontend::live_compute_graphics_import_native_format(
                      r.format, r.num_components));
              // A cube decoder's ordinary source span covers one face, while its retained
              // Uint16 2D-array producer is keyed by all six. Keep that exception beside
              // the importer; ordinary decoded formats continue to use the exact span the
              // persistent cache validates.
              const bool compute_image_shape =
                  (r.img_dim == 1u && r.depth == 1u) ||
                  (r.img_dim == 2u && r.depth != 0u) ||
                  (r.img_dim == 3u && r.depth == 6u);
              const bool compute_image_import_eligible =
                  !PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_COMPUTE_IMAGE_BIND") &&
                  !has_live_rtt && !has_ds_live && r.cls == RC::Texture &&
                  compute_image_shape && !r.in_mip_tail &&
                  r.declared_mip_levels == 1u && !r.srgb &&
                  (!r.depth_compare || (r.img_dim == 3u && r.depth == 6u)) &&
                  !r.host_data &&
                  (!r.compression_enabled || persistent_dcc_uncompressed) &&
                  compute_image_format != VK_FORMAT_UNDEFINED;
              // A zero decode span can trigger a full tiled capture-footprint derivation.
              // Resolve it only after the independent import prerequisites pass; keep the
              // same exact extent/alias proof for every import that can actually run.
              uint64_t compute_image_guest_bytes = 0;
              if (!prune_unused_texture_preparation || compute_image_import_eligible) {
                  ++g_texture_decode_scope.compute_import_span_queries;
                  compute_image_guest_bytes =
                      prosper::frontend::live_compute_graphics_import_guest_bytes(
                          r, persistent_source_size);
              }
              const bool compute_image_candidate =
                  compute_image_import_eligible && compute_image_guest_bytes;
              resource_compute_image_candidate = compute_image_candidate;
              if (compute_image_candidate &&
                  prosper::frontend::import_live_compute_storage_image(
                      r, compute_image_guest_bytes, compute_image_import)) {
                  const prosper::test::RenderVkCtx& render_context =
                      prosper::test::render_vk_ctx();
                  resource_compute_image_hit = compute_image_import.valid() &&
                      compute_image_import.native_format ==
                          static_cast<uint32_t>(compute_image_format) &&
                      compute_image_import.width == tw &&
                      compute_image_import.height == th &&
                      compute_image_import.depth == r.depth && render_context.ok &&
                      compute_image_import.device ==
                          static_cast<void*>(render_context.dev);
                  // #3307: a borrow the renderer then discards costs exactly what a miss
                  // costs, and the compute backend cannot see it happen.
                  prosper::frontend::live_compute_record_image_borrow_renderer_verdict(
                      resource_compute_image_hit);
                  if (!resource_compute_image_hit) compute_image_import = {};
              }
              if (resource_compute_image_hit &&
                  compute_image_import.vertical_stack_layers == 6u) {
                  // The compute image is the exact five-face copy result. A later graphics
                  // span can render the remaining face into a retained D32 attachment at
                  // the same guest cube address. Merge only attachment writes ordered
                  // after this producer; older retained faces are precisely what compute
                  // replaced and must not win merely because they still reside in cache.
                  if (r.layer_stride_bytes)
                      prosper::test::note_ds_layer_stride(
                          r.gpu_addr, tw, th, r.layer_stride_bytes);
                  const auto overlay =
                      prosper::test::select_persistent_ds_cube_depth_after(
                          r.gpu_addr, tw, th,
                          compute_image_import.producer_command_order);
                  if (overlay.present_mask) {
                      resource_compute_depth_hybrid = true;
                      resource_compute_producer_order =
                          compute_image_import.producer_command_order;
                      resource_compute_depth_overlay_mask = overlay.present_mask;
                      decode_key.renderer_overlay_version = overlay.overlay_version;
                      compute_image_import = {}; // release the lease; CPU builds the hybrid
                      resource_compute_image_hit = false;
                  }
              }
              if (texref_census)
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kComputeProbe);
              // GPU route for a fully renderer-owned depth cube: gather, quantise and
              // restack the six retained faces inside this callback's ordered batch
              // instead of the CPU bridge below. Decided after the compute probe: an
              // exact compute cube, or the compute/DS hybrid (GTA V), must win over
              // older retained faces, and both stay on their existing paths. A mixed
              // cube (fewer than six retained faces) keeps the CPU path, which decodes
              // the missing faces from guest bytes.
              std::shared_ptr<prosper::test::PersistentDsDepthCubeGpuImage>
                  retained_depth_cube_gpu;
              if (gpu_depth_cube_snapshots && producer_batch &&
                  retained_depth_cube_cache_candidate &&
                  retained_depth_cube.present_mask == 0x3fu &&
                  !resource_compute_image_hit && !resource_compute_depth_hybrid) {
                  using TexCensus = prosper::frontend::TextureReferenceCensus;
                  for (const auto& cached : depth_cube_gpu_snapshots) {
                      if (cached.base == r.gpu_addr && cached.width == tw &&
                          cached.height == th && cached.gpu->servable_to(producer_batch) &&
                          prosper::test::persistent_ds_cube_identity_matches(
                              r.gpu_addr, tw, th, *cached.gpu)) {
                          retained_depth_cube_gpu = cached.gpu;
                          if (texref_census) ++texref_census->cube_gpu_reuses;
                          break;
                      }
                  }
                  if (!retained_depth_cube_gpu) {
                      const uint64_t census_cube_read_start =
                          texref_census ? TexCensus::aux_begin() : 0;
                      const auto result = prosper::test::copy_persistent_ds_cube_depth_gpu(
                          r.gpu_addr, tw, th, *producer_batch, retained_depth_cube_gpu);
                      if (texref_census)
                          texref_census->aux_end(TexCensus::kAuxCubeGpu, census_cube_read_start);
                      if (result == prosper::test::DepthCubeGpuResult::Ready) {
                          if (depth_cube_gpu_snapshots.size() >= kDepthCubeGpuSnapshotEntries)
                              depth_cube_gpu_snapshots.erase(depth_cube_gpu_snapshots.begin());
                          depth_cube_gpu_snapshots.push_back(
                              {r.gpu_addr, tw, th, retained_depth_cube_gpu});
                      }
                  }
                  if (retained_depth_cube_gpu) {
                      // Renderer authority is now carried by the snapshot. Neither the
                      // renderer-generation cube cache nor the guest-byte decode cache
                      // may take part: there are no decoded pixels to publish.
                      retained_depth_cube_cache_candidate = false;
                      resource_persistent_candidate = false;
                  }
              }
              DepthCubeSourceLayout depth_cube_source_layout;
              if (retained_depth_cube_cache_candidate) {
                  const size_t face_bytes = persistent_source_is_tiled
                      ? prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, 2)
                      : static_cast<size_t>(tw) * th * 2u;
                  depth_cube_source_layout = {r.gpu_addr,
                      r.layer_stride_bytes ? r.layer_stride_bytes : face_bytes, face_bytes};
              }
              const bool depth_cube_source_layout_valid =
                  retained_depth_cube_cache_candidate && depth_cube_source_layout.fits(
                      persistent_source_addr, persistent_source_size);
              // The validation snapshot must cover the range validate_exact() compares:
              // persistent_source_addr, which is below sampled_source_addr for a native BC
              // mip chain (#3873) and equal to it for every other guest texture.
              auto copy_persistent_source = [&](uint8_t* dst, size_t bytes) {
                  return persistent_dcc_fast_clear
                      ? copy_dcc_metadata(dst, bytes)
                      : copy_resource(dst, persistent_source_addr, bytes);
              };
              const bool guest_persistent_cache_eligible =
                  persistent_texture_decode_cache_eligible(
                      persistent_sampled_texture && !retained_depth_cube_gpu,
                      resource_compute_image_hit || resource_compute_depth_hybrid,
                      fr.is_storage_image,
                      PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_DECODE_CACHE") != nullptr,
                      !r.compression_enabled || persistent_dcc_uncompressed ||
                          persistent_dcc_fast_clear,
                      persistent_decode_limit, persistent_source_size);
              const bool retained_depth_cube_cache_eligible =
                  retained_depth_cube_cache_candidate &&
                  !PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_DECODE_CACHE") &&
                  persistent_decode_limit != 0;
              const bool persistent_cache_eligible =
                  guest_persistent_cache_eligible ||
                  retained_depth_cube_cache_eligible;
              // Range a submit-scoped identity entry may be re-proved against across a span
              // boundary (#1691). It is exactly the range the persistent decode cache
              // validates for this identity — the same bytes `validate_exact()` compares and
              // the same range its own in-submit reuse queries — so the fast path never
              // asserts more than the cache it short-circuits. Ineligible resources
              // (captured replay backing, storage images, unsupported DCC states, cache
              // disabled) have no such established range, so they keep the pre-#1691
              // span-local lifetime instead of being retained against an unverified extent.
              const uint64_t cross_span_source_size = persistent_cache_eligible
                  ? persistent_source_size : 0u;
              static const bool cross_submit_watch_enabled =
                  !PROSPER_ENV_VALUE("PROSPER_NO_CROSS_SUBMIT_TEXTURE_WRITE_WATCH");
              // Page-protection watches have a fixed setup/query cost and may need to
              // resolve every alias of every covered page.  For small textures an exact
              // byte comparison is both simpler and cheaper; reserve dirty tracking for
              // sources large enough for it to amortize.  Keep the cutoff tunable for
              // host/platform profiling without changing the cache's correctness policy.
              static const size_t cross_submit_watch_min_bytes = [] {
                  // Same family, same sentinel: 0 makes every source large enough, so a
                  // typo would widen eligibility rather than disable the knob (#3253).
                  const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_MIN_KB");
                  const uint64_t kib = prosper::diag::env_u64_or_default_capped(
                      "PROSPER_TEXTURE_WRITE_WATCH_MIN_KB", value, 1024ull,
                      SIZE_MAX / 1024ull, "KiB");
                  return static_cast<size_t>(kib * 1024ull);
              }();
              // An explicit minimum keeps its historical meaning, including diagnostic
              // trials that deliberately enable eager watches for smaller sources.
#ifdef __linux__
              static const bool stable_small_sources =
                  !PROSPER_ENV_VALUE("PROSPER_NO_SMALL_TEXTURE_WRITE_WATCH") &&
                  !PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_MIN_KB");
#else
              // Other hosts cannot arm the fault-safe page watch. Repeated unsuccessful
              // small promotions would consume budget without avoiding any comparisons.
              constexpr bool stable_small_sources = false;
#endif
              // Retained depth cubes can advance stability from renderer-generation
              // authority without comparing guest bytes. That is not small-tier evidence.
              const bool stable_small_guest_sources =
                  stable_small_sources && !retained_depth_cube_cache_eligible;
              static const size_t cross_submit_watch_defer_min_bytes = [] {
                  // 0 means "defer nothing": should_promote_write_watch returns true
                  // immediately, arming every source on first sight. This is the knob
                  // #3253 names first, because it is the one an A/B is most likely to
                  // mistype (`=8mb`, `=8 KB`, a stray quote) -- and a bare strtoull would
                  // hand back the MOST aggressive arm instead of the one asked for.
                  const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB");
                  const uint64_t kib = prosper::diag::env_u64_or_default_capped(
                      "PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB", value, 8192ull,
                      SIZE_MAX / 1024ull, "KiB");
                  return static_cast<size_t>(kib * 1024ull);
              }();
              static const uint32_t cross_submit_watch_promotion_validations = [] {
                  // 0 makes `stable >= 0` always true, so every deferred source promotes
                  // on its first acquisition -- the aggressive end, not the off end.
                  const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS");
                  return static_cast<uint32_t>(prosper::diag::env_u64_or_default_capped(
                      "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS", value, 3ull, UINT32_MAX,
                      "unchanged validations"));
              }();
              const auto cold_watch_admission = renderer_write_watch_admission(
                  persistent_source_size, 0, cross_submit_watch_min_bytes,
                  cross_submit_watch_defer_min_bytes,
                  cross_submit_watch_promotion_validations, stable_small_guest_sources);
              const bool cross_submit_watch_eligible =
                  cross_submit_watch_enabled && cold_watch_admission.eligible;
              static const bool audit_cross_submit_watch =
                  PROSPER_ENV_VALUE("PROSPER_AUDIT_CROSS_SUBMIT_TEXTURE_WRITE_WATCH") != nullptr;
              prosper::host::GuestWriteWatch pending_source_watch;
              // `narrow_decode_done` prevents the generic 32-bpp detiler from touching an already
              // expanded narrow surface. `narrow_done` separately records the legacy coverage
              // broadcast whose RGBA pixels intentionally replace the T# swizzle. UNORM16 expands
              // to (R,0,0,1) and must retain the real descriptor swizzle, so the two facts differ.
              bool narrow_done = false;
              bool narrow_decode_done = false;
              if (texref_census)
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kPreLookup);
              auto reused = (has_live_rtt || resource_compute_image_hit)
                  ? decoded_textures.end() : decoded_textures.find(decode_key);
              // A retained entry from an EARLIER span in this submit is usable only while
              // its range is still this binding's range AND the ordered journal proves
              // nothing wrote it. Overlap (an interleaved compute/DMA/EOP write landed on
              // the backing), a moved range, and Unknown (no journal, overflowed, or a
              // different submit) all drop the entry, releasing its scratch pin, and the
              // resolve falls through to the persistent cache's own validation exactly as
              // it did before the map was widened. The journal query is skipped for a
              // same-span hit: its verdict cannot change the answer, and this runs once
              // per texture reference.
              if (reused != decoded_textures.end()) {
                  const bool same_span = reused->second.span == decode_span_ordinal;
                  // Unknown, not Unchanged: the predicate ignores this on a same-span
                  // hit, so the value is unobservable today — but Unchanged is the
                  // PERMISSIVE verdict, and if the two same-span tests ever drift apart
                  // the placeholder would silently authorise reuse. Unknown fails closed
                  // at identical cost.
                  prosper::gpu::GuestGpuWriteQuery journal_query =
                      prosper::gpu::GuestGpuWriteQuery::Unknown;
                  if (!same_span) {
                      journal_query = prosper::gpu::guest_gpu_writes_since(
                          reused->second.snapshot, reused->second.source_addr,
                          reused->second.source_size);
                  }
                  if (!submit_local_texture_decode_reusable(
                          reused->second.span, decode_span_ordinal,
                          reused->second.source_addr, reused->second.source_size,
                          persistent_source_addr, cross_span_source_size,
                          journal_query)) {
                      if (reused->second.texstore_slot < texstore_pinned.size())
                          texstore_pinned[reused->second.texstore_slot] = false;
                      decoded_textures.erase(reused);
                      reused = decoded_textures.end();
                      ++g_texture_decode_scope.invalidations;
                  }
              }
              // A native BCn entry's bytes depend on the T#'s mip level count, which the
              // decode key does not hold: two descriptors at one address that differ only
              // in last_level share a key. Reusing a 1-level entry for a 10-level chain
              // would make the backend read the missing levels past the end of the buffer
              // (#3883 review). So a BC entry is reusable only when its format AND its
              // packed level count are exactly what this reference would produce.
              const uint32_t expected_packed_mip_levels =
                  native_bc_mip_chain ? native_bc_chain_levels : 0u;
              auto native_bc_entry_matches = [&](VkFormat entry_format,
                                                 uint32_t entry_levels) {
                  if (!native_bc_sampled &&
                      prosper::test::backend_block_compressed_bytes(entry_format) == 0u)
                      return true;
                  return entry_format == decoded_texture_format &&
                         entry_levels == expected_packed_mip_levels;
              };
              if (reused != decoded_textures.end() &&
                  !native_bc_entry_matches(reused->second.texture_format,
                                           reused->second.packed_mip_levels)) {
                  if (reused->second.texstore_slot < texstore_pinned.size())
                      texstore_pinned[reused->second.texstore_slot] = false;
                  decoded_textures.erase(reused);
                  reused = decoded_textures.end();
                  ++g_texture_decode_scope.invalidations;
              }
              DecodedTexture persistent_reuse;
              const DecodedTexture* decoded_reuse = reused != decoded_textures.end()
                  ? &reused->second : nullptr;
              resource_local_reuse = decoded_reuse != nullptr;
              if (decoded_reuse) {
                  if (decoded_reuse->span == decode_span_ordinal) {
                      ++g_texture_decode_scope.same_span_reuses;
                  } else {
                      ++g_texture_decode_scope.cross_span_reuses;
                      // Unchanged was just proven up to here, so this instant is a valid new
                      // baseline. Advancing it keeps each later query scanning only the
                      // writes since the previous use instead of the whole submit journal.
                      reused->second.snapshot = prosper::gpu::guest_gpu_write_snapshot();
                  }
              }
              if (!decoded_reuse && persistent_cache_eligible) {
                  auto cached = persistent_decoded_textures.find(decode_key);
                  if (cached != persistent_decoded_textures.end() &&
                      cached->second.source_addr == persistent_source_addr &&
                      cached->second.source_size == persistent_source_size &&
                      native_bc_entry_matches(cached->second.texture_format,
                                              cached->second.packed_mip_levels) &&
                      (!retained_depth_cube_cache_eligible ||
                       (depth_cube_source_layout_valid && cached->second.depth_cube_source.active &&
                        cached->second.depth_cube_source.renderer_mask ==
                            retained_depth_cube.present_mask))) {
                      resource_texture_watch_active =
                          static_cast<bool>(cached->second.source_watch);
                      resource_texture_watch_disabled =
                          cached->second.source_watch_disabled;
                      resource_texture_watch_only = cached->second.source_watch_only;
                      resource_texture_watch_stability =
                          cached->second.source_watch_stable_validations;
                      auto record_validation = [&](bool matches, bool refusal,
                                                   size_t bytes, double milliseconds) {
                          if (!validation_census) return;
                          using Watch = TextureValidationWatch;
                          using Query = prosper::host::GuestWriteWatchQuery;
                          Watch reason = !cross_submit_watch_enabled
                              ? Watch::ControlDisabled
                              : !cross_submit_watch_eligible ? Watch::BelowMinimum
                              : cached->second.source_watch_disabled
                                  ? Watch::DisabledAfterDirty
                              : resource_texture_watch_query == static_cast<int>(Query::Dirty)
                                  ? Watch::Dirty
                              : resource_texture_watch_query == static_cast<int>(Query::Unknown)
                                  ? Watch::Unknown
                              : resource_texture_watch_query == static_cast<int>(Query::Unchanged)
                                  ? Watch::Unchanged : Watch::NotQueried;
                          validation_census->data.record(
                              refusal ? TextureValidationOutcome::WatchOnlyRefusal
                                  : matches ? TextureValidationOutcome::Match
                                            : TextureValidationOutcome::ExactFailure,
                              reason, persistent_source_size, bytes, milliseconds,
                              static_cast<bool>(cached->second.source_watch),
                              cached->second.source_watch_stable_validations);
                      };
                      auto validate_exact = [&] {
                          // A successfully promoted Linux watch may own the mutation proof
                          // without retaining a second encoded copy. Dirty/Unknown must miss;
                          // there is deliberately no probabilistic hash fallback here.
                          if (cached->second.source_watch_only) {
                              record_validation(false, true, 0, 0);
                              return false;
                          }
                          resource_texture_exact_validation = true;
                          const auto validation_start = timing_enabled
                              ? RenderClock::now() : RenderClock::time_point{};
                          bool matches = false;
                          size_t validated_bytes = 0;
                          if (cached->second.depth_cube_source.active) {
                              matches = equal_depth_cube_source(depth_cube_source_layout,
                                  cached->second.depth_cube_source, cached->second.source_prefix,
                                  safe_span, safe_equal, validated_bytes);
                          } else if (cached->second.source_matches_pixels) {
                              matches = safe_equal(
                                  cached->second.pixels ? cached->second.pixels->data() : nullptr,
                                  persistent_source_addr,
                                  persistent_source_size, validated_bytes) &&
                                  validated_bytes == cached->second.source_prefix_size;
                          } else if (!PROSPER_ENV_VALUE("PROSPER_TEXTURE_VALIDATION_SCRATCH_COPY") &&
                                     cached->second.source_prefix.size() ==
                                         persistent_source_size) {
                              // The cached prefix owns the complete encoded texture. Compare
                              // guest memory directly against it: copying the same 100+ MiB
                              // working set into a scratch buffer before memcmp doubled the
                              // validation traffic on every Evergate frame. safe_equal keeps
                              // the same sparse-page/readability guards and exact byte check.
                              matches = safe_equal(
                                  cached->second.source_prefix.data(), persistent_source_addr,
                                  persistent_source_size, validated_bytes) &&
                                  validated_bytes == persistent_source_size;
                          } else {
                              persistent_validation_scratch.resize(persistent_source_size);
                              validated_bytes = copy_persistent_source(
                                  persistent_validation_scratch.data(),
                                  persistent_source_size);
                              matches =
                                  validated_bytes == cached->second.source_prefix_size &&
                                  validated_bytes == cached->second.source_prefix.size() &&
                                  (validated_bytes == 0 || !std::memcmp(
                                      persistent_validation_scratch.data(),
                                      cached->second.source_prefix.data(), validated_bytes));
                          }
                          resource_texture_validated_bytes += validated_bytes;
                          if (timing_enabled) {
                              pending_timing.persistent_validations++;
                              pending_timing.persistent_validation_bytes += validated_bytes;
                              const double validation_ms =
                                  std::chrono::duration<double, std::milli>(
                                      RenderClock::now() - validation_start).count();
                              resource_texture_validation_ms += validation_ms;
                              record_validation(matches, false, validated_bytes, validation_ms);
                          }
                          return matches;
                      };
                      static const bool submit_reuse_enabled =
                          !PROSPER_ENV_VALUE("PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE");
                      static const bool audit_submit_reuse =
                          PROSPER_ENV_VALUE("PROSPER_AUDIT_SUBMIT_TEXTURE_VALIDATION_REUSE") != nullptr;
                      const prosper::gpu::GuestGpuWriteQuery submit_query =
                          submit_reuse_enabled
                          ? prosper::gpu::guest_gpu_writes_since(
                              cached->second.validation_snapshot, persistent_source_addr,
                              persistent_source_size)
                          : prosper::gpu::GuestGpuWriteQuery::Unknown;
                      resource_texture_submit_query =
                          static_cast<int>(submit_query);
                      if (timing_enabled && submit_reuse_enabled) {
                          if (submit_query == prosper::gpu::GuestGpuWriteQuery::Unknown) {
                              pending_timing.persistent_submit_unknown++;
                              if (prosper::gpu::guest_gpu_write_tracking_active())
                                  pending_timing.persistent_submit_stale++;
                              else
                                  pending_timing.persistent_submit_unarmed++;
                          } else if (submit_query == prosper::gpu::GuestGpuWriteQuery::Overlap)
                              pending_timing.persistent_submit_overlap++;
                      }
                      const bool submit_unchanged = submit_reuse_enabled &&
                          submit_query == prosper::gpu::GuestGpuWriteQuery::Unchanged;
                      prosper::host::GuestWriteWatchQuery watch_query =
                          prosper::host::GuestWriteWatchQuery::Unknown;
                      if (!submit_unchanged && cross_submit_watch_eligible &&
                          !cached->second.source_watch_disabled) {
                          watch_query = cached->second.source_watch.query();
                          resource_texture_watch_query =
                              static_cast<int>(watch_query);
                          if (timing_enabled) {
                              if (watch_query == prosper::host::GuestWriteWatchQuery::Dirty)
                                  pending_timing.persistent_watch_dirty++;
                              else if (watch_query == prosper::host::GuestWriteWatchQuery::Unknown)
                                  pending_timing.persistent_watch_unknown++;
                          }
                          if (watch_query == prosper::host::GuestWriteWatchQuery::Dirty &&
                              ++cached->second.source_watch_dirty_count >= 2) {
                              cached->second.source_watch.reset();
                              cached->second.source_watch_disabled = true;
                              if (timing_enabled)
                                  pending_timing.persistent_watch_disabled++;
                          } else if (watch_query ==
                                     prosper::host::GuestWriteWatchQuery::Unchanged) {
                              cached->second.source_watch_dirty_count = 0;
                          }
                      }
                      const bool watch_unchanged = !submit_unchanged &&
                          watch_query == prosper::host::GuestWriteWatchQuery::Unchanged;
                      const bool watch_ready =
                          prosper::frontend::renderer_write_watch_admission(
                              persistent_source_size,
                              cached->second.source_watch_stable_validations,
                              cross_submit_watch_min_bytes,
                              cross_submit_watch_defer_min_bytes,
                              cross_submit_watch_promotion_validations,
                              stable_small_guest_sources).ready;
                      const bool may_arm_watch = cached->second.source_watch ||
                          (!submit_unchanged && !watch_unchanged &&
                           cross_submit_watch_eligible &&
                           !cached->second.source_watch_disabled && watch_ready &&
                           write_watch_promotion_budget.try_consume(
                               persistent_source_size));
                      if (!submit_unchanged && !watch_unchanged &&
                          cross_submit_watch_eligible &&
                          !cached->second.source_watch_disabled &&
                          may_arm_watch) {
                          // Arm before reading. A concurrent CPU write during or after the
                          // authoritative comparison then dirties this registration instead of
                          // landing in an unprotected compare-to-rearm window.
                          if (!cached->second.source_watch.rearm())
                              cached->second.source_watch =
                                  prosper::host::GuestWriteWatch::create(
                                      persistent_source_addr, persistent_source_size);
                      }
                      const bool renderer_only_cube = retained_depth_cube_cache_eligible &&
                          cached->second.depth_cube_source.active &&
                          cached->second.depth_cube_source.renderer_mask == 0x3fu;
                      bool content_matches = renderer_only_cube;
                      if (renderer_only_cube) {
                          if (texref_census) ++texref_census->cube_cache_hits;
                          // The key carries the newest selected retained-depth write.
                          // Finding this entry is the exact renderer-authority proof;
                          // there are intentionally no guest bytes to compare.
                          resource_texture_exact_validation = false;
                      } else if (submit_unchanged || watch_unchanged) {
                          const bool audit = submit_unchanged
                              ? audit_submit_reuse : audit_cross_submit_watch;
                          content_matches = audit ? validate_exact() : true;
                          if (!content_matches) {
                              fprintf(stderr,
                                      "[render] %s texture validation audit failed "
                                      "addr=0x%llx bytes=%zu\n",
                                      submit_unchanged ? "in-submit" : "cross-submit-watch",
                                      (unsigned long long)r.gpu_addr,
                                      persistent_source_size);
                          } else {
                              if (submit_unchanged) {
                                  resource_persistent_submit_reuse = true;
                                  if (timing_enabled)
                                      pending_timing.persistent_submit_reuses++;
                              } else if (timing_enabled) {
                                  pending_timing.persistent_watch_reuses++;
                              }
                          }
                      } else {
                          content_matches = validate_exact();
                      }
                      if (!submit_unchanged && !watch_unchanged) {
                          cached->second.source_watch_stable_validations =
                              prosper::frontend::update_write_watch_stability(
                                  cached->second.source_watch_stable_validations,
                                  content_matches,
                                  cold_watch_admission.stability_limit);
                      }
                      resource_texture_watch_active =
                          static_cast<bool>(cached->second.source_watch);
                      resource_texture_watch_disabled =
                          cached->second.source_watch_disabled;
                      resource_texture_watch_only = cached->second.source_watch_only;
                      resource_texture_watch_stability =
                          cached->second.source_watch_stable_validations;
                      if (content_matches) {
                          cached->second.last_use = decode_generation;
                          cached->second.validation_snapshot =
                              prosper::gpu::guest_gpu_write_snapshot();
                          static const bool keep_source_snapshots =
                              PROSPER_ENV_VALUE("PROSPER_KEEP_TEXTURE_SOURCE_SNAPSHOTS") != nullptr;
                          if (!keep_source_snapshots && !cached->second.depth_cube_source.active &&
                              texture_source_snapshot_can_follow_watch(
                                  cached->second.source_matches_pixels,
                                  audit_submit_reuse || audit_cross_submit_watch,
                                  static_cast<bool>(cached->second.source_watch),
                                  cached->second.source_prefix.size(),
                                  cached->second.source_size)) {
                              const size_t released = cached->second.source_prefix.size();
                              std::vector<uint8_t>().swap(cached->second.source_prefix);
                              cached->second.source_watch_only = true;
                              persistent_decoded_texture_bytes -= released;
                          }
                          persistent_reuse = {cached->second.pixels ? cached->second.pixels->data() : nullptr,
                                              cached->second.output_height,
                                              cached->second.narrow,
                                              cached->second.persistent_id,
                                              cached->second.persistent_version,
                                              decode_span_ordinal,
                                              cached->second.validation_snapshot,
                                              cached->second.source_addr,
                                              cross_span_source_size,
                                              SIZE_MAX,
                                              cached->second.pixels};
                          persistent_reuse.texture_format =
                              cached->second.texture_format;
                          persistent_reuse.storage_image_contract_valid =
                              cached->second.storage_image_contract_valid;
                          persistent_reuse.layers = cached->second.layers;
                          persistent_reuse.packed_mip_levels =
                              cached->second.packed_mip_levels;
                          persistent_reuse.pixels_bytes = cached->second.pixels ? cached->second.pixels->size() : 0;
                          decoded_reuse = &persistent_reuse;
                          resource_persistent_hit = true;
                          if (timing_enabled) pending_timing.persistent_hits++;
                      } else {
                          resource_persistent_invalidation = true;
                          if (timing_enabled)
                              pending_timing.persistent_invalidations++;
                          // #3891 texture-validation-churn: charge the failed validation's
                          // route-specific prefix count. Direct/depth comparisons exclude
                          // the first differing chunk; scratch validation counts copied bytes.
                          // A failed validation can also mean an incomplete readable prefix,
                          // so neither source mutation nor total comparison traffic is proved.
                          if (resource_texture_exact_validation) {
                              prosper::diagnostics::perf::add(
                                  prosper::diagnostics::perf::Counter::TextureValidationFailures);
                              prosper::diagnostics::perf::add(
                                  prosper::diagnostics::perf::Counter::TextureValidationFailedBytes,
                                  resource_texture_validated_bytes);
                          }
                      }
                  } else {
                      // Establish the mutation boundary before the initial source read/decode.
                      // If registration is unsupported, the empty watch keeps all later reuse on
                      // the exact fallback.
                      if (cross_submit_watch_eligible && cold_watch_admission.ready)
                          pending_source_watch = prosper::host::GuestWriteWatch::create(
                              persistent_source_addr, persistent_source_size);
                      resource_persistent_miss = true;
                      if (timing_enabled) {
                          pending_timing.persistent_misses++;
                      }
                  }
              }
              if (texref_census)
                  texref_census->mark(prosper::frontend::TextureReferenceCensus::kLookup);
              // PROSPER_BIND_LOG=<min-cb>: log every sampled-resource binding decision for
              // 3 final callbacks (same ordinal as PROSPER_PASS_LOG) — which path serves
              // each guest address, so a black consumer input can be attributed.
              static const char* const bind_log = getenv("PROSPER_BIND_LOG");
              if (bind_log) {
                  const uint64_t at = g_pass_log_submit.load(std::memory_order_relaxed);
                  const uint64_t bl_min = std::strtoull(bind_log, nullptr, 0);
                  if (at >= bl_min && at < bl_min + 3u)
                      fprintf(stderr,
                              "[bind] cb=%llu addr=0x%llx %ux%u dim=%u fmt=%u path=%s "
                              "rtt(w=%u h=%u gpu=%d rgba=%d)\n",
                              (unsigned long long)at, (unsigned long long)r.gpu_addr,
                              tw, th, r.img_dim, (unsigned)r.format,
                              has_gpu_live_rtt          ? "gpu-bind"
                              : has_uniform_live_rtt    ? "uniform-clear"
                              : has_cpu_live_rtt        ? "cpu-rtt"
                              : resource_compute_image_hit ? "compute-bind"
                              : decoded_reuse           ? "decode-cache"
                                                        : "decode",
                              live_rtt != g_rtt.end() ? live_rtt->second.w : 0,
                              live_rtt != g_rtt.end() ? live_rtt->second.h : 0,
                              live_rtt != g_rtt.end() ? (int)live_rtt->second.gpu_valid
                                                      : -1,
                              live_rtt != g_rtt.end() ? (int)(bool)live_rtt->second.rgba
                                                      : -1);
              }
              // Sampled depth bridge (#1275): the T# addresses the depth plane of a
              // surface prosper rendered into a persistent Vulkan DS image. Guest memory
              // never receives that depth, so the guest-byte decode below would sample
              // zeros (every shadow compare passes -> unshadowed, overbright scenes).
              // Bind the retained depth image directly instead. The guest T# keeps its
              // native extent under PROSPER_RENDER_SCALE>1 while the renderer-owned image
              // is uniformly smaller; normalized depth sampling maps onto that image just
              // like the color-target bridge above. Keep the actual image extent in the
              // backend resource so its exact cache lookup remains unambiguous.
              if (has_gpu_live_rtt) {
                  fr.persistent_render_target_id = sampled_source_addr;
                  fr.tw = live_rtt->second.w;
                  fr.th = live_rtt->second.h;
                  fr.td = sampled_volume_rtt ? r.depth : 1u;
                  fr.img_dim = r.img_dim;
                  // A depth-one T# array can compile to either a base-slice 2D image or
                  // a real arrayed image. Bind the view the shader actually declared.
                  fr.guest_array = reflected_binding->image_arrayed;
                  fr.texture_format = live_rtt->second.format;
                  // CB_COLOR renders one mip view per target, while this T# samples the
                  // complete allocation. Reconstruct the target identities and retain
                  // every level that is still authoritative on the GPU. The backend will
                  // copy them into one sampled mip image. Expose only the contiguous
                  // renderer-owned prefix: inventing a missing tail by filtering the last
                  // live level corrupts temporal/bloom chains whose real producer has not
                  // run yet. Requiring at least two live levels keeps the
                  // historical zero-copy bind for ordinary single-target resources.
                  uint32_t mip_bytes_per_texel =
                      prosper::gpu::data_format_bytes(r.format) *
                      (r.num_components ? r.num_components : 1u);
                  if ((r.format == prosper::gpu::DataFormat::Float10_11_11 &&
                       r.num_components == 3u) ||
                      (r.format == prosper::gpu::DataFormat::Unorm2_10_10_10 &&
                       r.num_components == 4u))
                      mip_bytes_per_texel = 4u;
                  const bool renderer_mip_chain_selected =
                      !sampled_volume_rtt &&
                      PROSPER_ENV_VALUE("PROSPER_NO_RENDERER_MIP_CHAIN") == nullptr &&
                      renderer_mip_chain_selector().includes(sampled_source_addr);
                  const RendererMipChainLayout mip_chain = renderer_mip_chain_selected
                      ? renderer_mip_chain_layout(
                            sampled_source_addr, tw, th, mip_bytes_per_texel,
                            r.tile_mode, r.declared_mip_levels)
                      : RendererMipChainLayout{};
                  const VkFormat mip_format = prosper::test::backend_color_format(
                      live_rtt->second.format);
                  const size_t retained_mips_consumed = mip_chain.level_count > 1u
                      ? consume_renderer_mip_chain(mip_chain, fr.tw, fr.th, mip_format)
                      : 0u;
                  uint32_t live_mip_levels = 0;
                  uint32_t live_mip_mask = 0;
                  uint32_t present_mip_mask = 0;
                  if (mip_chain.level_count > 1u) {
                      for (uint32_t level = 0; level < mip_chain.level_count; ++level) {
                          const uint32_t level_w = std::max(fr.tw >> level, 1u);
                          const uint32_t level_h = std::max(fr.th >> level, 1u);
                          auto* retained = prosper::test::find_persistent_color_target(
                              mip_chain.level_ids[level], level_w, level_h,
                              mip_format, false);
                          if (retained) present_mip_mask |= 1u << level;
                          if (retained && retained->valid) {
                              fr.persistent_render_target_mip_ids[level] =
                                  mip_chain.level_ids[level];
                              ++live_mip_levels;
                              live_mip_mask |= 1u << level;
                          }
                      }
                  }
                  uint32_t exposed_mip_levels = 0;
                  while (exposed_mip_levels < mip_chain.level_count &&
                         (live_mip_mask & (1u << exposed_mip_levels)) != 0u)
                      ++exposed_mip_levels;
                  if (exposed_mip_levels > 1u &&
                      fr.persistent_render_target_mip_ids[0] ==
                          fr.persistent_render_target_id) {
                      fr.persistent_render_target_mip_count = exposed_mip_levels;
                      fr.declared_mip_levels = exposed_mip_levels;
                  } else {
                      fr.persistent_render_target_mip_ids = {};
                  }
                  if (PROSPER_ENV_VALUE("PROSPER_RENDERER_MIP_CHAIN_LOG") != nullptr &&
                      r.declared_mip_levels > 1u) {
                      static std::set<uint64_t> logged_renderer_mips;
                      if (logged_renderer_mips.insert(sampled_source_addr).second) {
                          std::fprintf(stderr,
                                       "[renderer-mips] addr=0x%llx %ux%u fmt=%u "
                                       "tile=%u declared=%u layout=%u live=%u "
                                       "present=0x%x valid=0x%x exposed=%u "
                                       "selected=%u retained=%zu shader=%llu set=%u binding=%u "
                                       "lod=%.3f..%.3f bias=%.3f filter=%u/%u/%u "
                                       "cache=%zu/%.1fMiB limit=%zu/%.1fMiB ids=",
                                       static_cast<unsigned long long>(sampled_source_addr),
                                       tw, th, static_cast<unsigned>(r.format), r.tile_mode,
                                       r.declared_mip_levels, mip_chain.level_count,
                                       live_mip_levels, present_mip_mask, live_mip_mask,
                                       exposed_mip_levels,
                                       renderer_mip_chain_selected ? 1u : 0u,
                                       retained_mips_consumed,
                                       static_cast<unsigned long long>(shader_identity),
                                       set, r.binding, r.min_lod, r.max_lod, r.lod_bias,
                                       r.mag_filter, r.min_filter, r.mip_filter,
                                       prosper::test::persistent_color_target_cache().size(),
                                       prosper::test::persistent_color_target_bytes() /
                                           (1024.0 * 1024.0),
                                       prosper::test::persistent_color_target_count_limit(),
                                       prosper::test::persistent_color_target_limit() /
                                           (1024.0 * 1024.0));
                          for (uint32_t level = 0; level < mip_chain.level_count; ++level)
                              std::fprintf(stderr, "%s0x%llx", level ? "," : "",
                                           static_cast<unsigned long long>(
                                               mip_chain.level_ids[level]));
                          std::fprintf(stderr, "\n");
                      }
                  }
                  resource_rtt_hit = true;
              } else if (has_uniform_live_rtt) {
                  fr.has_uniform_color = true;
                  fr.uniform_color = live_rtt->second.uniform_color;
                  fr.tw = live_rtt->second.w;
                  fr.th = live_rtt->second.h;
                  fr.td = 1;
                  fr.img_dim = r.img_dim;
                  fr.texture_format = live_rtt->second.format;
                  resource_rtt_hit = true;
              } else if (resource_compute_image_hit) {
                  fr.borrowed_compute_image = compute_image_import.image;
                  fr.borrowed_compute_device = compute_image_import.device;
                  fr.borrowed_compute_image_layout = compute_image_import.layout;
                  fr.borrowed_compute_image_lease =
                      std::move(compute_image_import.lease);
                  fr.borrowed_compute_vertical_stack_layers =
                      compute_image_import.vertical_stack_layers;
                  fr.tw = compute_image_import.width;
                  fr.th = compute_image_import.height *
                      std::max(compute_image_import.vertical_stack_layers, 1u);
                  fr.td = compute_image_import.vertical_stack_layers
                      ? 1u : compute_image_import.depth;
                  fr.img_dim = r.img_dim;
                  fr.texture_format = compute_image_format;
              } else if (retained_depth_array_gpu) {
                  // GPU-resident snapshot of every requested layer, filled by a command
                  // buffer queued earlier in this callback's ordered batch. The backend's
                  // borrowed-image route binds it directly and holds the lease until its
                  // submission completes; like the CPU snapshot below, it never enters a
                  // guest-byte cache.
                  fr.borrowed_compute_image = retained_depth_array_gpu->image();
                  fr.borrowed_compute_device = prosper::test::render_vk_ctx().dev;
                  fr.borrowed_compute_image_layout =
                      static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                  fr.borrowed_compute_image_lease = retained_depth_array_gpu;
                  fr.texture_format = retained_depth_array_format;
                  fr.tw = tw;
                  fr.th = th;
                  fr.td = 1;
                  fr.img_dim = r.img_dim;
                  fr.guest_array = true;
                  fr.sample_count = r.depth;
                  resource_rtt_hit = true;
              } else if (retained_depth_cube_gpu) {
                  // GPU-gathered vertical stack of all six retained faces, filled by a
                  // command buffer queued earlier in this callback's ordered batch. Same
                  // representation the CPU cube bridge uploads: R8G8B8A8_UNORM,
                  // tw x 6th, face-major. Never entered into a guest-byte cache.
                  fr.borrowed_compute_image = retained_depth_cube_gpu->image();
                  fr.borrowed_compute_device = prosper::test::render_vk_ctx().dev;
                  fr.borrowed_compute_image_layout =
                      static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                  fr.borrowed_compute_image_lease = retained_depth_cube_gpu;
                  fr.texture_format = VK_FORMAT_R8G8B8A8_UNORM;
                  fr.tw = tw;
                  fr.th = th * 6u;
                  fr.td = 1;
                  fr.img_dim = r.img_dim;
                  resource_rtt_hit = true;
              } else if (retained_depth_array) {
                  // Owned CPU snapshot spans every requested layer; it is never entered
                  // into guest-byte caches, which cannot observe renderer-only rewrites.
                  fr.tex_rgba_owner = retained_depth_array;
                  fr.tex_rgba = retained_depth_array->data();
                  fr.tex_byte_size = retained_depth_array->size();
                  fr.texture_format = retained_depth_array_format;
                  fr.tw = tw;
                  fr.th = th;
                  fr.td = 1;
                  fr.img_dim = r.img_dim;
                  fr.guest_array = true;
                  fr.sample_count = r.depth;
                  resource_rtt_hit = true;
              } else if (has_ds_live) {
                  fr.persistent_depth_target_id = r.gpu_addr;
                  fr.tw = sampled_ds.width;
                  fr.th = sampled_ds.height;
                  fr.td = 1; fr.img_dim = r.img_dim;
                  resource_rtt_hit = true;
                  if (PROSPER_ENV_ON("PROSPER_DSBRIDGE_LOG")) {
                      static int consumer_logged = 0;
                      if (consumer_logged++ < 24)
                          fprintf(stderr,
                                  "[dsbridge] consumer draw target=0x%llx samples "
                                  "0x%llx %ux%u cwm=%x scissor=[%d,%d)-[%d,%d) "
                                  "dcmp=%d func=%u\n",
                                  (unsigned long long)draw.color0_base,
                                  (unsigned long long)r.gpu_addr, tw, th,
                                  draw.ps.color_write_mask, draw.ps.scissor_left,
                                  draw.ps.scissor_top, draw.ps.scissor_right,
                                  draw.ps.scissor_bottom, (int)r.depth_compare,
                                  r.depth_compare_func);
                  }
              } else if (decoded_reuse) {
                  fr.tex_rgba = decoded_reuse->pixels;
                  fr.tex_rgba_owner = decoded_reuse->pixels_owner;
                  fr.gpu_detile = decoded_reuse->gpu_detile;
                  fr.tw = tw;
                  fr.th = decoded_reuse->output_height;
                  // #325: the layer count travels with the decoded pixels. Without this
                  // the hot path -- a cache hit, which is how a world-texture atlas is
                  // served on all but its first reference -- published a single-layer view
                  // over a buffer holding every slice, so the shader could only ever read
                  // slice 0 however correct the decode and the SPIR-V were.
                  // Guarded on `is_array` so this cannot touch the MSAA plane-array
                  // resources that share the channel: for those, sample_count is already
                  // established by the resource setup, and republishing a cached value
                  // would be a behaviour change nobody asked for.
                  if (is_array) {
                      fr.sample_count = decoded_reuse->layers;
                      fr.tex_byte_size = decoded_reuse->pixels_bytes;
                  }
                  // #2998: does the GUEST fill high slices after we decoded the atlas?
                  // The decode happens once and is then served from cache, so a guest that
                  // streams slices in later is invisible to every instrument that samples
                  // the DECODE. This samples guest memory directly on the reuse path,
                  // which runs on every reference, and reports the first time a high slice
                  // becomes non-zero. If that ever fires, the cache is serving stale
                  // pixels and the decode simply happened too early.
                  // #2998: what BACKS this atlas? If the guest maps texture data in
                  // (a file mapping, or several separate mappings), a short or partial
                  // mapping explains an allocation whose tail reads zero far better than
                  // "the game never wrote it". Printed once per address, from the host
                  // process's own map, because the guest VA is mapped into it.
                  if (is_array && r.depth > 64u && PROSPER_ENV_ON("PROSPER_SLICEMAPS")) {
                      static std::set<uint64_t> shown;
                      if (shown.insert(r.gpu_addr).second) {
                          const uint64_t lo = r.gpu_addr;
                          const uint64_t hi = r.gpu_addr +
                              (uint64_t)r.layer_stride_bytes * r.depth;
                          FILE* m = fopen("/proc/self/maps", "r");
                          char line[512];
                          uint32_t printed = 0, overlapping = 0;
                          while (m && fgets(line, sizeof line, m)) {
                              unsigned long long a = 0, b = 0;
                              if (sscanf(line, "%llx-%llx", &a, &b) != 2) continue;
                              if (b <= lo || a >= hi) continue;
                              ++overlapping;
                              if (printed >= 8u) continue;   // counted below, never hidden
                              fprintf(stderr, "[slicemaps] 0x%llx..0x%llx overlaps: %s",
                                      (unsigned long long)lo, (unsigned long long)hi, line);
                              ++printed;
                          }
                          // The COUNT is the measurement -- a span split across many
                          // mappings is precisely the partial-residency case, and a
                          // silently truncated list would read as "one clean mapping".
                          if (overlapping > printed)
                              fprintf(stderr, "[slicemaps]   ... %u overlapping mapping(s) "
                                              "total, %u shown\n", overlapping, printed);
                          if (m) fclose(m);
                          if (!printed)
                              fprintf(stderr, "[slicemaps] 0x%llx..0x%llx has NO mapping "
                                              "in /proc/self/maps\n",
                                      (unsigned long long)lo, (unsigned long long)hi);
                      }
                  }
                  // #2998: a coarse occupancy map of the GUEST allocation itself, in
                  // 256 KiB buckets, independent of any stride assumption. Every previous
                  // measurement addressed slices with layer_stride, so a wrong stride made
                  // "slice N is zero" trivially true by probing outside the data. This
                  // asks the only question that needs no layout at all: WHERE in this
                  // allocation is there content?
                  if (is_array && r.depth > 64u && PROSPER_ENV_ON("PROSPER_OCCUPANCY")) {
                      static std::set<uint64_t> done;
                      if (done.insert(r.gpu_addr).second) {
                          const uint64_t span = (uint64_t)r.layer_stride_bytes * r.depth;
                          const uint64_t bucket = 256u * 1024u;
                          const uint64_t n = span / bucket;
                          if (!n) {
                              // filled=0/0 read as "scanned and empty". It is neither.
                              fprintf(stderr,
                                      "[occupancy] 0x%llx span=%llu bytes -- NO complete "
                                      "256KiB bucket (layer_stride=%u depth=%u); "
                                      "NOTHING SCANNED\n",
                                      (unsigned long long)r.gpu_addr,
                                      (unsigned long long)span,
                                      r.layer_stride_bytes, r.depth);
                              fflush(stderr);
                          } else {
                          // Scan every byte of each bucket, not a head sample. This probed
                          // only the first 512 B of each 256 KiB bucket (0.195%), so a
                          // bucket whose content began past that window read as EMPTY --
                          // a false negative that reached TOMB_RAIDER_STATUS.md as
                          // "not in this allocation at ANY layout".
                          const uint64_t scan_n = n < 512u ? n : 512u;
                          std::vector<uint8_t> probe(64u * 1024u);
                          std::string map;
                          uint64_t filled = 0, last_filled = 0, unreadable = 0;
                          for (uint64_t i = 0; i < scan_n; ++i) {
                              bool any = false, short_read = false;
                              for (uint64_t off = 0; off < bucket && !any; off += probe.size()) {
                                  const size_t want = (size_t)std::min<uint64_t>(
                                      probe.size(), bucket - off);
                                  const size_t got = copy_resource(
                                      probe.data(), r.gpu_addr + i * bucket + off, want);
                                  for (size_t k = 0; k < got && !any; ++k)
                                      any = probe[k] != 0;
                                  if (got < want) { short_read = true; break; }
                              }
                              if (any) { ++filled; last_filled = i; }
                              else if (short_read) ++unreadable;
                              // '?' is NOT '.': prosper could not read it, so it is
                              // unknown, not known-empty.
                              map += any ? '#' : (short_read ? '?' : '.');
                          }
                          fprintf(stderr,
                                  "[occupancy] 0x%llx span=%llu MiB bucket=256KiB "
                                  "FULL-BUCKET scan filled=%llu/%llu scanned "
                                  "(of %llu total) last_filled_bucket=%llu "
                                  "unreadable=%llu ('?' = unreadable, NOT empty)\n%s\n",
                                  (unsigned long long)r.gpu_addr,
                                  (unsigned long long)(span >> 20),
                                  (unsigned long long)filled,
                                  (unsigned long long)scan_n, (unsigned long long)n,
                                  (unsigned long long)last_filled,
                                  (unsigned long long)unreadable, map.c_str());
                          fflush(stderr);
                          }
                      }
                  }
                  if (is_array && r.depth > 64u && PROSPER_ENV_ON("PROSPER_SLICEWATCH")) {
                      // Per-address, not global: with two arrays in flight a global flag
                      // lets the first silence the second's report entirely.
                      static std::map<uint64_t, uint64_t> watched_by_addr;
                      static std::set<uint64_t> said_zero, said_nz;
                      uint64_t& watched = watched_by_addr[r.gpu_addr];
                      const uint64_t stride = r.layer_stride_bytes
                          ? r.layer_stride_bytes
                          : (uint64_t)tw * th;  // conservative
                      const uint64_t probe_slice = r.depth - 8u;
                      const uint64_t addr = r.gpu_addr + probe_slice * stride;
                      uint8_t buf[256] = {0};
                      const size_t got = copy_resource(buf, addr, sizeof buf);
                      bool nz = false;
                      for (size_t k = 0; k < got && !nz; ++k) nz = buf[k] != 0;
                      ++watched;
                      if (!nz && said_zero.insert(r.gpu_addr).second) {
                          fprintf(stderr, "[slicewatch] slice %llu of 0x%llx reads ZERO "
                                          "(got=%zu of %zu) on reference %llu%s\n",
                                  (unsigned long long)probe_slice,
                                  (unsigned long long)r.gpu_addr, got, sizeof buf,
                                  (unsigned long long)watched,
                                  r.layer_stride_bytes ? "" : " [probe used the tw*th "
                                                              "stride FALLBACK, not a "
                                                              "guest-declared stride]");
                      }
                      if (nz && said_nz.insert(r.gpu_addr).second) {
                          // A transition needs a PRIOR zero reading for this address.
                          // Without one this is just the first observation, and saying
                          // "the guest filled it after our decode" would assert an
                          // ordering nothing here measured.
                          const bool saw_zero_first = said_zero.count(r.gpu_addr) != 0;
                          fprintf(stderr,
                                  saw_zero_first
                                    ? "[slicewatch] slice %llu of 0x%llx BECAME NON-ZERO "
                                      "on reference %llu -- it read ZERO earlier, so the "
                                      "guest filled it after that reading%s\n"
                                    : "[slicewatch] slice %llu of 0x%llx reads NON-ZERO on "
                                      "reference %llu -- FIRST observation, no prior ZERO "
                                      "reading, so this is NOT an observed transition%s\n",
                                  (unsigned long long)probe_slice,
                                  (unsigned long long)r.gpu_addr,
                                  (unsigned long long)watched,
                                  r.layer_stride_bytes ? "" : " [probe used the tw*th "
                                                              "stride FALLBACK, not a "
                                                              "guest-declared stride]");
                      }
                  }
                  fr.td = is_volume ? r.depth : 1u;
                  fr.img_dim = r.img_dim;
                  fr.texture_format = decoded_reuse->texture_format;
                  fr.storage_image_contract_valid =
                      decoded_reuse->storage_image_contract_valid;
                  // #1272: plain 2D guest textures only — cube outputs stack 6 faces into
                  // one 2D image (fr.th != th), and volumes keep their own path; generated
                  // mips across face/slice boundaries would bleed.
                  if (!is_volume && fr.th == th)
                      fr.declared_mip_levels = r.declared_mip_levels;
                  // native_bc_entry_matches proved the entry packs exactly this many levels;
                  // the byte count lets the backend refuse anything it cannot read.
                  if (prosper::test::backend_block_compressed_bytes(fr.texture_format)) {
                      fr.uploaded_mip_levels = decoded_reuse->packed_mip_levels;
                      fr.tex_byte_size = decoded_reuse->pixels_bytes;
                  }
                  narrow_done = decoded_reuse->narrow;
                  fr.persistent_texture_id = decoded_reuse->persistent_id;
                  fr.persistent_texture_version = decoded_reuse->persistent_version;
                  if (PROSPER_ENV_ON("PROSPER_DETILE_STATS") && resource_persistent_hit) {
                      static uint64_t bc_cube_hit_total = 0;
                      static std::unordered_map<uint64_t, uint64_t> bc_cube_hits;
                      const uint64_t hit_footprint =
                          prosper::gpu::gpu_capture_resource_footprint(r);
                      const bool expensive_bc_cube = r.img_dim == 3u &&
                          texture_decode_miss_is_expensive_block(
                              r.format == prosper::gpu::DataFormat::Bc6,
                              persistent_source_size,
                              static_cast<size_t>(std::min<uint64_t>(
                                  hit_footprint, SIZE_MAX)));
                      if (expensive_bc_cube) {
                          const uint64_t address_hit_ordinal = ++bc_cube_hits[r.gpu_addr];
                          const uint64_t global_hit_ordinal = address_hit_ordinal == 1u
                              ? ++bc_cube_hit_total : bc_cube_hit_total;
                          if (should_report_texture_decode_hit(
                                  global_hit_ordinal, address_hit_ordinal,
                                  expensive_bc_cube)) {
                              fprintf(stderr,
                                      "[detile-hit] ordinal=%llu address-ordinal=%llu "
                                      "addr=0x%llx key=0x%zx %ux%ux%u dim=%u fmt=%u/%u "
                                      "tile=%u footprint=%llu source=%zu "
                                      "cache=persistent-hit id=%llu version=%llu "
                                      "validation=%s validated=%zu submit-query=%d "
                                      "watch-query=%d watch-active=%d watch-only=%d\n",
                                      (unsigned long long)global_hit_ordinal,
                                      (unsigned long long)address_hit_ordinal,
                                      (unsigned long long)r.gpu_addr,
                                      TextureDecodeKeyHash{}(decode_key),
                                      tw, th, r.depth, r.img_dim,
                                      static_cast<unsigned>(r.format), r.num_components,
                                      r.tile_mode,
                                      (unsigned long long)hit_footprint,
                                      persistent_source_size,
                                      (unsigned long long)fr.persistent_texture_id,
                                      (unsigned long long)fr.persistent_texture_version,
                                      resource_texture_exact_validation ? "exact" : "skip",
                                      resource_texture_validated_bytes,
                                      resource_texture_submit_query,
                                      resource_texture_watch_query,
                                      static_cast<int>(resource_texture_watch_active),
                                      static_cast<int>(resource_texture_watch_only));
                              fflush(stderr);
                          }
                      }
                  }
                  // Only a persistent-cache hit needs to be recorded here; a submit-local
                  // hit is already in the map under this key, and emplacing over it would
                  // build and discard a node on every repeat reference. Persistent-hit
                  // pixels are owned by the persistent entry, so the bytes live in storage
                  // the scratch allocator never recycles and no pin is needed; the retained
                  // range is the one that cache itself validates for this identity.
                  if (!resource_local_reuse)
                      decoded_textures.insert_or_assign(
                          decode_key,
                          DecodedTexture{fr.tex_rgba, fr.th, narrow_done,
                                         fr.persistent_texture_id,
                                         fr.persistent_texture_version,
                                         decode_span_ordinal,
                          prosper::gpu::guest_gpu_write_snapshot(),
                          persistent_source_addr,
                          cross_span_source_size, SIZE_MAX,
                          fr.tex_rgba_owner,
                                         fr.texture_format,
                                         fr.storage_image_contract_valid,
                                         fr.sample_count, fr.tex_byte_size, fr.gpu_detile,
                                         fr.uploaded_mip_levels});
                  if (timing_enabled) pending_timing.texture_reuses++;
              } else {
              // PROSPER_DETILE_STATS: this branch is the texture-decode MISS path — the cache
              // had no entry for decode_key, so we are about to read guest memory and CPU-detile
              // this surface. Counting decodes per guest address answers the #1177 redundancy
              // question: if a handful of addresses accumulate huge counts, the same immutable
              // surface is re-detiled every frame (a cache-key instability / live-RTT re-decode);
              // if counts stay near 1 the cost is a large one-time working set instead.
              if (PROSPER_ENV_ON("PROSPER_DETILE_STATS")) {
                  struct DecodeShape {
                      uint32_t width = 0, height = 0, depth = 0;
                      uint32_t image_dimension = 0, format = 0, components = 0;
                      uint32_t tile_mode = 0, declared_bytes = 0;
                  };
                  struct DecodeAddressState {
                      size_t last_key_hash = 0;
                      uint64_t key_changes = 0;
                      bool has_key = false;
                  };
                  static std::unordered_map<uint64_t, uint64_t> ds_count;  // guest addr -> times decoded
                  static std::unordered_map<uint64_t, DecodeShape> ds_shape;
                  static std::unordered_map<uint64_t, DecodeAddressState> ds_address_state;
                  static uint64_t ds_total = 0;
                  // Classify WHY this decode reached the miss path (so the redundancy is actionable):
                  // rtt          - a renderer-owned RTT decoded on the CPU (has_live_rtt)
                  // notsampled   - not a candidate class (host data / format / unsupported dim)
                  // compression  - DCC-compressed source whose metadata is unsupported
                  // size0        - persistent_source_size==0 (force-detiled Unorm8x4)
                  // invalidated  - eligible AND a cache entry exists, but validate_exact() rejected it
                  // cold         - eligible but no cache entry (first use / LRU-evicted)
                  static uint64_t r_rtt = 0, r_notsampled = 0, r_compression = 0,
                                  r_size0 = 0, r_inval = 0, r_cold = 0, r_other = 0;
                  // For the rtt bucket, record WHICH has_gpu_live_rtt condition failed (so a
                  // registered render target that could be sampled straight from the GPU image
                  // instead fell back to CPU detile). storage/notvalid/dimmismatch/self/noptarget.
                  static uint64_t rtt_storage = 0, rtt_notvalid = 0, rtt_dimmismatch = 0,
                                  rtt_self = 0, rtt_noptarget = 0, rtt_unknown = 0;
                  // For the notsampled bucket, record WHY the guest texture is not a persistent-
                  // cache candidate (host data / unsupported dimension / class / format) plus a
                  // (format<<4|components) histogram of the format-excluded
                  // ones, to see whether FP16/HDR art dominates (a cache-eligibility gap, #1177).
                  static uint64_t ns_hostdata = 0, ns_notdim1 = 0, ns_notclass = 0, ns_fmt = 0;
                  static std::unordered_map<uint32_t, uint64_t> ns_fmt_hist;
                  const uint64_t address_ordinal = ++ds_count[r.gpu_addr];
                  ds_shape[r.gpu_addr] = {
                      tw, th, r.depth, r.img_dim, static_cast<uint32_t>(r.format),
                      r.num_components, r.tile_mode, r.size};
                  const uint64_t global_ordinal = ++ds_total;
                  const size_t key_hash = TextureDecodeKeyHash{}(decode_key);
                  DecodeAddressState& address_state = ds_address_state[r.gpu_addr];
                  const bool key_changed = address_state.has_key &&
                      address_state.last_key_hash != key_hash;
                  if (key_changed) ++address_state.key_changes;
                  const size_t previous_key_hash = address_state.last_key_hash;
                  address_state.last_key_hash = key_hash;
                  address_state.has_key = true;

                  const auto matching_entry = persistent_decoded_textures.find(decode_key);
                  const bool matching_cache_entry =
                      matching_entry != persistent_decoded_textures.end() &&
                      matching_entry->second.source_addr == persistent_source_addr &&
                      matching_entry->second.source_size == persistent_source_size;
                  const bool compression_supported = !r.compression_enabled ||
                      persistent_dcc_uncompressed || persistent_dcc_fast_clear;
                  const bool cache_disabled =
                      PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_DECODE_CACHE") != nullptr;
                  const TextureDecodeMissReason miss_reason = texture_decode_miss_reason(
                      has_live_rtt, persistent_sampled_texture, compression_supported,
                      persistent_source_size, cache_disabled, persistent_decode_limit,
                      matching_cache_entry, persistent_cache_eligible);
                  if (miss_reason == TextureDecodeMissReason::LiveRenderTarget) {
                      r_rtt++;
                      if (fr.is_storage_image) rtt_storage++;
                      else if (!live_rtt->second.gpu_valid) rtt_notvalid++;
                      else if (live_rtt->second.w != tw || live_rtt->second.h != th)
                          rtt_dimmismatch++;
                      else if (draw_binds_color_target(draw, sampled_source_addr, tw, th))
                          rtt_self++;
                      else if (prosper::test::find_persistent_color_target(
                                   sampled_source_addr, tw, th,
                                   live_rtt->second.format) == nullptr)
                          rtt_noptarget++;   // evicted from / absent in the backend target cache
                      else rtt_unknown++;
                  }
                  else if (miss_reason == TextureDecodeMissReason::UnsupportedCandidate) {
                      r_notsampled++;
                      if (r.host_data) ns_hostdata++;
                      else if (r.img_dim != 1u) ns_notdim1++;
                      else if (r.cls != RC::Texture) ns_notclass++;
                      else { ns_fmt++;
                          ns_fmt_hist[((uint32_t)r.format << 4) |
                                      (r.num_components & 0xFu)]++; }
                  }
                  else if (miss_reason == TextureDecodeMissReason::UnsupportedCompression)
                      r_compression++;
                  else if (miss_reason == TextureDecodeMissReason::EmptySource)
                      r_size0++;
                  else if (miss_reason == TextureDecodeMissReason::ContentInvalidated)
                      r_inval++;
                  else if (miss_reason == TextureDecodeMissReason::ColdOrEvicted)
                      r_cold++;
                  else
                      r_other++;

                  const uint64_t diagnostic_footprint =
                      prosper::gpu::gpu_capture_resource_footprint(r);
                  const size_t diagnostic_footprint_size = static_cast<size_t>(
                      std::min<uint64_t>(diagnostic_footprint, SIZE_MAX));
                  const bool expensive_bc6 = texture_decode_miss_is_expensive_block(
                      r.format == prosper::gpu::DataFormat::Bc6,
                      persistent_source_size, diagnostic_footprint_size);
                  if (should_report_texture_decode_miss(
                          global_ordinal, address_ordinal, expensive_bc6)) {
                      auto reason_name = [](TextureDecodeMissReason reason) {
                          switch (reason) {
                              case TextureDecodeMissReason::LiveRenderTarget: return "rtt";
                              case TextureDecodeMissReason::UnsupportedCandidate:
                                  return "unsupported-candidate";
                              case TextureDecodeMissReason::UnsupportedCompression:
                                  return "unsupported-compression";
                              case TextureDecodeMissReason::EmptySource: return "empty-source";
                              case TextureDecodeMissReason::CacheDisabled:
                                  return "cache-disabled";
                              case TextureDecodeMissReason::CacheLimitZero:
                                  return "cache-limit-zero";
                              case TextureDecodeMissReason::ContentInvalidated:
                                  return "content-invalidated";
                              case TextureDecodeMissReason::ColdOrEvicted:
                                  return "cold-or-evicted";
                              case TextureDecodeMissReason::Other: return "other";
                          }
                          return "unknown";
                      };
                      fprintf(stderr,
                              "[detile-miss] ordinal=%llu address-ordinal=%llu "
                              "addr=0x%llx key=0x%zx previous-key=0x%zx "
                              "key-changes=%llu reason=%s %ux%ux%u dim=%u fmt=%u/%u "
                              "tile=%u declared=%u footprint=%llu source=%zu "
                              "candidate=%d eligible=%d entry=%d "
                              "validation=%s validated=%zu submit-query=%d "
                              "watch-query=%d watch-active=%d watch-disabled=%d "
                              "watch-only=%d watch-stable=%u\n",
                              (unsigned long long)global_ordinal,
                              (unsigned long long)address_ordinal,
                              (unsigned long long)r.gpu_addr, key_hash,
                              previous_key_hash,
                              (unsigned long long)address_state.key_changes,
                              reason_name(miss_reason), tw, th, r.depth, r.img_dim,
                              static_cast<unsigned>(r.format), r.num_components,
                              r.tile_mode, r.size,
                              (unsigned long long)diagnostic_footprint,
                              persistent_source_size,
                              static_cast<int>(persistent_sampled_texture),
                              static_cast<int>(persistent_cache_eligible),
                              static_cast<int>(matching_cache_entry),
                              resource_texture_exact_validation ? "exact" : "skip",
                              resource_texture_validated_bytes,
                              resource_texture_submit_query,
                              resource_texture_watch_query,
                              static_cast<int>(resource_texture_watch_active),
                              static_cast<int>(resource_texture_watch_disabled),
                              static_cast<int>(resource_texture_watch_only),
                              resource_texture_watch_stability);
                      fflush(stderr);
                  }
                  if ((ds_total % 3000) == 0) {
                      std::vector<std::pair<uint64_t, uint64_t>> v(ds_count.begin(), ds_count.end());
                      std::sort(v.begin(), v.end(),
                                [](auto& a, auto& b) { return a.second > b.second; });
                      fprintf(stderr, "[detile-stats] %llu decodes, %zu distinct addrs; top re-decoded:",
                              (unsigned long long)ds_total, ds_count.size());
                      for (int i = 0; i < 8 && i < (int)v.size(); i++) {
                          const DecodeShape& shape = ds_shape[v[i].first];
                          fprintf(stderr,
                                  " 0x%llx x%llu[%ux%ux%u dim=%u fmt=%u/%u tile=%u "
                                  "decl=%u]",
                                  (unsigned long long)v[i].first,
                                  (unsigned long long)v[i].second,
                                  shape.width, shape.height, shape.depth,
                                  shape.image_dimension, shape.format, shape.components,
                                  shape.tile_mode, shape.declared_bytes);
                      }
                      fprintf(stderr, "\n[detile-stats] miss reasons: rtt=%llu notsampled=%llu "
                              "compression=%llu size0=%llu invalidated=%llu cold=%llu other=%llu\n",
                              (unsigned long long)r_rtt, (unsigned long long)r_notsampled,
                              (unsigned long long)r_compression, (unsigned long long)r_size0,
                              (unsigned long long)r_inval, (unsigned long long)r_cold,
                              (unsigned long long)r_other);
                      fprintf(stderr, "[detile-stats] rtt-fail: storage=%llu notvalid=%llu "
                              "dimmismatch=%llu self=%llu noptarget=%llu unknown=%llu\n",
                              (unsigned long long)rtt_storage, (unsigned long long)rtt_notvalid,
                              (unsigned long long)rtt_dimmismatch, (unsigned long long)rtt_self,
                              (unsigned long long)rtt_noptarget, (unsigned long long)rtt_unknown);
                      std::vector<std::pair<uint32_t, uint64_t>> fh(ns_fmt_hist.begin(),
                                                                   ns_fmt_hist.end());
                      std::sort(fh.begin(), fh.end(),
                                [](auto& a, auto& b) { return a.second > b.second; });
                      fprintf(stderr, "[detile-stats] notsampled-why: hostdata=%llu notdim1=%llu "
                              "notclass=%llu fmt=%llu; top fmt/comp:",
                              (unsigned long long)ns_hostdata, (unsigned long long)ns_notdim1,
                              (unsigned long long)ns_notclass, (unsigned long long)ns_fmt);
                      for (int i = 0; i < 6 && i < (int)fh.size(); i++)
                          fprintf(stderr, " fmt=%u/comp=%u x%llu",
                                  fh[i].first >> 4, fh[i].first & 0xF,
                                  (unsigned long long)fh[i].second);
                      fprintf(stderr, "\n");
                      fflush(stderr);
                  }
              }
              // Keep established persistent CPU decodes: replacing a cache miss with an
              // unretained GPU conversion would repeat that work on every later submit.
              const bool gpu_detile_2d_shape =
                  ((r.cls == RC::Texture && !persistent_cache_eligible) ||
                   (portable_raw_uvec4_storage && portable_storage_shape &&
                    fr.storage_image_contract_valid)) && r.img_dim == 1u &&
                  r.depth == 1u && r.declared_mip_levels == 1u &&
                  !r.layer_stride_bytes && !r.layer_mip_offset_bytes &&
                  !PROSPER_ENV_ON("PROSPER_NO_GPU_DETILE_2D");
              const bool gpu_detile_shape =
                  ((is_cube && !fr.is_storage_image && r.num_components == 4) || gpu_detile_2d_shape) &&
                  !r.compression_enabled && !r.in_mip_tail && !r.depth_compare &&
                  r.format == prosper::gpu::DataFormat::Float16 &&
                  (r.num_components == 2 || r.num_components == 4) &&
                  sampled_source_bpt == r.num_components * 2 && r.tile_mode == 27 &&
                  fr.sample_count == 1 && !cpu_rtt_copy_diagnostics &&
                  !PROSPER_ENV_VALUE("PROSPER_DUMP_RAWTILE") &&
                  !PROSPER_ENV_VALUE("PROSPER_SLICEMAP") &&
                  !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                  !PROSPER_ENV_ON("PROSPER_NO_HALF_QUANTIZATION") &&
                  !PROSPER_ENV_ON("PROSPER_NO_GPU_DETILE");
              // The intervening materializers can use CPU scratch for RTT or
              // retained depth. Defer its allocation only when neither owns this
              // address; failed GPU preflight allocates it at the fallback below.
              const bool defer_detile_pixels = gpu_detile_shape &&
                  !resource_compute_depth_hybrid && !resource_compute_image_hit &&
                  live_rtt == g_rtt.end() &&
                  !prosper::test::is_retained_ds_plane(r.gpu_addr);
              const size_t volume_texels = (size_t)tw * th * (is_volume ? r.depth : 1u);
              fr.texture_format = decoded_texture_format;
              const uint32_t output_bpp =
                  prosper::test::backend_color_bytes_per_pixel(decoded_texture_format);
              // A native BC surface holds ceil(w/4) x ceil(h/4) blocks, not texels
              // (output_bpp is 0 for a block format by design).
              size_t nb = native_bc_sampled
                  ? static_cast<size_t>(native_bc_mip_chain
                        ? prosper::test::backend_texture_chain_bytes(
                              decoded_texture_format, tw, th, native_bc_chain_levels)
                        : prosper::test::backend_texture_bytes(
                              decoded_texture_format, tw, th))
                  : volume_texels * output_bpp *
                        (is_cube ? 6u : (is_array ? decoded_layers : 1u));
              size_t linear_source_prefix_size = 0;
              bool decoder_source_snapshot_ready = false;
              size_t decoder_source_prefix_size = 0;
              // Only an exact encoded footprint can serve both the decoder and validation.
              // Repacked rows, detiled bytes and metadata are different source domains.
              // When used, the source watch was armed above, before this authoritative read.
              auto stage_linear_decoder_source = [&](
                  prosper::frontend::DecodeScratchPool::Lease& lease,
                  size_t bytes, bool tiled) {
                  const bool candidate = !tiled && !linear_padded_read &&
                      !is_cube && !is_array && !is_volume && !r.in_mip_tail &&
                      persistent_cache_eligible && !persistent_source_matches_pixels &&
                      !persistent_dcc_fast_clear &&
                      persistent_source_addr == sampled_source_addr &&
                      bytes != 0 && bytes == persistent_source_size;
                  if (candidate)
                      g_texture_decode_scope.decoder_snapshot_candidate_bytes += bytes;
                  if (candidate &&
                      !PROSPER_ENV_ON("PROSPER_NO_DECODE_SOURCE_SNAPSHOT_REUSE")) {
                      persistent_validation_scratch.resize(bytes);
                      decoder_source_prefix_size = copy_resource(
                          persistent_validation_scratch.data(), sampled_source_addr, bytes);
                      // The decoder consumes zeros beyond the readable prefix, but the
                      // persistent entry must retain only the bytes actually read.
                      std::fill(persistent_validation_scratch.begin() +
                                    decoder_source_prefix_size,
                                persistent_validation_scratch.end(), 0);
                      decoder_source_snapshot_ready = true;
                      g_texture_decode_scope.decoder_snapshot_reused_bytes +=
                          decoder_source_prefix_size;
                      return true;
                  }
                  lease = prosper::frontend::decode_scratch_pool().take(bytes);
                  return false;
              };
              bool generic_source_copy_deferred = false;
              ++g_texture_decode_scope.decodes;
              const size_t texture_slot = acquire_texstore_slot();
              std::vector<uint8_t>& texture_pixels = texstore[texture_slot];
              // Set false once the persistent cache takes ownership of these bytes; while it
              // is true a retained identity entry must pin `texture_slot`, or the next span
              // would decode an unrelated texture into the very slot it points at.
              bool decoded_pixels_in_texstore = false;
              if (retain_cpu_live_rtt || defer_detile_pixels) texture_pixels.clear();
              else texture_pixels.resize(nb);
              auto copy_linear_padded_rows = [&](uint8_t* dst, size_t dst_row,
                                                 uint32_t rows) {
                  size_t total = 0;
                  for (uint32_t y = 0; y < rows; ++y) {
                      uint8_t* drow = dst + (size_t)y * dst_row;
                      const size_t got = copy_resource(
                          drow, sampled_source_addr + (uint64_t)y * linear_src_row,
                          dst_row);
                      total += got;
                      if (got < dst_row) std::fill(drow + got, drow + dst_row, 0);
                  }
                  return total;
              };
              // #325: the same padded-row read, but from an arbitrary base, so an ARRAY
              // slice can use it. A linear array's rows are pitch-padded exactly as a
              // single linear surface's are -- `linear_padded_read` already names
              // img_dim 5 -- and the slice loop was reading them flat, which is what made
              // slice 0 of a linear BC array decode differently than it used to.
              auto copy_linear_padded_rows_from = [&](uint8_t* dst, uint64_t base,
                                                      size_t dst_row, uint32_t rows) {
                  size_t total = 0;
                  for (uint32_t y = 0; y < rows; ++y) {
                      uint8_t* drow = dst + (size_t)y * dst_row;
                      const size_t got = copy_resource(
                          drow, base + (uint64_t)y * linear_src_row, dst_row);
                      total += got;
                      if (got < dst_row) std::fill(drow + got, drow + dst_row, 0);
                  }
                  return total;
              };
              // PROSPER_TEXCOMMIT: log, once per texture base, how much of the sampled surface is
              // COMMITTED guest memory (the same reserved_range_state safe_copy stops at). If the
              // level's backgrounds read ~0% committed, they're GPU-DMA'd pages the CPU never
              // touched, so we read zeros -> the scene samples black (#300 black-gameplay probe).
              if (PROSPER_ENV_ON("PROSPER_TEXCOMMIT")) {
                  // Walk the SOURCE extent, never `nb` (the DECODED byte count computed just
                  // above for `texture_pixels`): for a block-compressed format `nb` is 4x+ the
                  // real guest bytes (BC7 decodes 1 source byte/texel to 4 RGBA8 bytes), and an
                  // array multiplies that again by the layer count. Scanning `nb` over
                  // `sampled_source_addr` both mis-measured "committed" (it counted whatever
                  // guest memory happened to follow the real texture) and read guest memory
                  // past the real allocation (#3053). `gpu_capture_resource_footprint()` is the
                  // same T#-driven source-byte computation the capture/replay path already uses
                  // for this resource, so it is correct here too regardless of format.
                  const size_t src_bytes = texcommit_scan_extent(
                      prosper::gpu::gpu_capture_resource_footprint(r));
                  const size_t PG = 0x10000; size_t committed = 0;
                  for (uint64_t a = sampled_source_addr;
                       a < sampled_source_addr + src_bytes; a += PG)
                      if (a >= 0x1000 && prosper_reserved_range_state(a) != 0) committed += PG;
                  static std::set<uint64_t> tcseen;
                  if (tcseen.insert(r.gpu_addr).second) {
                      // Also sample the first 8 dwords and count non-zero bytes over the whole
                      // SOURCE surface (src_bytes, not nb): zero content => the texture was
                      // allocated but never filled (a GPU-side upload/copy we don't execute);
                      // non-zero => it's a decode/tiling problem. tile_mode tells tiled vs linear.
                      uint32_t w0[8] = {0}; size_t nzb = 0;
                      if (committed) {
                          const uint8_t* p =
                              (const uint8_t*)(uintptr_t)sampled_source_addr;
                          for (int i = 0; i < 8; i++) w0[i] = ((const uint32_t*)p)[i];
                          for (size_t i = 0; i < src_bytes; i += 997) nzb += (p[i] != 0);   // sparse scan
                      }
                      fprintf(stderr, "[texcommit] tex 0x%llx %ux%u f%u tile=%u nb=%zu src=%zu "
                              "committed=%zu%% nz~%zu/%zu first=%08x %08x %08x %08x\n",
                              (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format, r.tile_mode,
                              nb, src_bytes,
                              src_bytes ? (size_t)(100 * committed / src_bytes) : 100, nzb,
                              src_bytes / 997,
                              w0[0], w0[1], w0[2], w0[3]);
                  }
              }
              // Real bytes-per-texel of the SAMPLED surface. Single/dual-channel textures (an R8
              // font/coverage atlas — this game's 2048x1024 c1 surface) are NOT 4 B/texel; reading
              // them as RGBA8 packs adjacent texels into one pixel and over-reads the allocation,
              // which is why glyph text rendered as solid white/black BLOCKS (#102). bpt drives a
              // narrow read+expand path below. StorageImage / unknown formats keep 4 B (bpt=0->4).
              uint32_t bpt = sampled_source_bpt;
              // fp16/fp32 surfaces use their real source element size below. Guest fp16 keeps
              // the historical UNORM8 conversion, while fp32 narrows to native RGBA16F so small
              // values and HDR range survive. Renderer-owned RTTs bypass both conversions.
              const bool f16 = sampled_source_f16;
              const bool f32 = sampled_source_f32;
              bool f16_done = false, f32_done = false;
              // RTT (#167): if this texture's base is a color target we rendered into, inject those
              // pixels (nearest-scaled to tw x th) instead of reading empty guest memory.
              // Direct compute imports already own the complete sampled image. Mark the
              // CPU materializer complete as well; otherwise it performs a discarded
              // guest decode (and, for a cube, synchronous retained-depth readbacks)
              // before the backend binds the borrowed image.
              bool rtt_hit = resource_compute_image_hit ||
                  (is_volume && has_gpu_live_rtt);
              // Why a sampled resource never consulted the renderer-owned RTT cache. Without
              // this, a draw that reads a target prosper rendered but takes the guest-decode
              // path instead is invisible in the log: no "sample tex" line is emitted at all,
              // and the draw silently samples empty guest memory.
              if (rtt_log && (fr.is_storage_image || !rtt_on ||
                              (is_volume && !has_gpu_live_rtt) || r.in_mip_tail))
                  fprintf(stderr,
                          "[rtt] sample tex addr=0x%llx %ux%u fmt=%u -> RTT PATH SKIPPED "
                          "(storage=%d rtt_on=%d volume=%d mip_tail=%d)\n",
                          (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format,
                          (int)fr.is_storage_image, (int)rtt_on, (int)is_volume,
                          (int)r.in_mip_tail);
              if (!fr.is_storage_image && rtt_on && !is_volume && !r.in_mip_tail) {
                  auto rit = live_rtt;
                  if (rit != g_rtt.end() && rit->second.w && rit->second.h && rit->second.rgba &&
                      !rit->second.rgba->empty()) {
                      const RttSurf& s = rit->second;
                      const uint32_t rtt_bpp = prosper::test::backend_color_bytes_per_pixel(s.format);
                      if (retain_cpu_live_rtt &&
                          (fr.tex_rgba_owner = rtt_injection_cache.materialize(
                               s.rgba, tw, th, s.w, s.h, rtt_bpp))) {
                          fr.tex_rgba = fr.tex_rgba_owner->data();
                          fr.texture_format = s.format;
                          rtt_hit = true;
                          resource_rtt_hit = true;
                      } else if (prosper::frontend::inject_rtt_pixels(
                                     texture_pixels, tw, th, *s.rgba,
                                     s.w, s.h, rtt_bpp)) {
                          fr.texture_format = s.format;
                          rtt_hit = true;
                          resource_rtt_hit = true;
                          // PROSPER_DUMP_SAMPLED_RTT (#710/#320): dump the exact RTT-layer
                          // pixels a draw samples, disambiguating a dark layer from a later
                          // composite/tint that darkens otherwise-correct input.
                          if (PROSPER_ENV_ON("PROSPER_DUMP_SAMPLED_RTT")) {
                              static std::set<uint64_t> seen;
                              if (seen.insert(sampled_source_addr).second) {
                                  const std::vector<uint8_t> inspected = inspection_rgba8(
                                      texture_pixels, tw, th, s.format);
                                  size_t nz = 0, rgbnz = 0;
                                  for (size_t p = 0; p + 3 < inspected.size(); p += 4) {
                                      if (inspected[p] || inspected[p+1] || inspected[p+2]) rgbnz++;
                                      for (int k = 0; k < 4; k++) nz += (inspected[p+k] != 0);
                                  }
                                  const char* dd = getenv("PROSPER_FRAME_DIR");
                                  char fn[512]; snprintf(fn, sizeof fn, "%s/sampledrtt_%llx_%ux%u.bmp",
                                                         dd ? dd : ".", (unsigned long long)r.gpu_addr, tw, th);
                                  prosper::test::dump_bmp(fn, inspected, tw, th);
                                  fprintf(stderr, "[sampledrtt] addr=0x%llx %ux%u rgb_nonblack=%zu/%u -> %s\n",
                                          (unsigned long long)sampled_source_addr,
                                          tw, th, rgbnz, tw*th, fn);
                              }
                          }
                      } // malformed/incomplete RTT bytes => miss; decode guest backing below
                  }
                  // Report HOW the binding resolved, not merely whether the COLOUR cache
                  // had it. This line used to print "miss" for a resource the depth or
                  // stencil bridge had already resolved, because `rtt_hit` speaks only for
                  // g_rtt -- so a correctly-bridged 4K depth buffer and a texture decoded
                  // from guest zeros were the same word. Reading a pass's inputs off this
                  // log therefore over-counted missing inputs, which is the one thing the
                  // log exists to answer.
                  // PROSPER_RTT_GUESTPEEK=1 — for each distinct sampled surface, how
                  // much of the GUEST backing is non-zero, next to how prosper resolved
                  // it. This separates two states that every other signal renders
                  // identically: a surface nothing ever wrote (guest bytes zero, cache
                  // zero) from one the guest DID write through a path prosper did not
                  // observe (guest bytes populated, cache zero) -- the second is a
                  // prosper defect and the first is not, and they call for opposite work.
                  // Read the guest VA directly rather than r.host_data: on this path
                  // host_data is null for every sampled texture (measured: zero peeks in a
                  // full routed run), so keying on it produced complete silence -- which
                  // reads as "the guest backing is empty" and is instead "we never
                  // looked". Guest code runs natively here, so a readability-checked VA is
                  // the authoritative view of that memory.
                  if (rtt_log && PROSPER_ENV_ON("PROSPER_RTT_GUESTPEEK") &&
                      sampled_source_addr &&
                      prosper::gpu::guest_readable(sampled_source_addr, 4096)) {
                      // Re-peek on a power-of-two schedule per address rather than once.
                      // A once-only peek reports the FIRST sighting, and a surface's first
                      // sighting is routinely its emptiest moment -- sampled during
                      // start-up before the guest has written it. That reads as "the guest
                      // never wrote this", which is the exact conclusion the peek exists to
                      // test, so the instrument would confirm it by construction. The
                      // schedule keeps the volume bounded (about 20 lines per surface over
                      // a million samples) while making the steady state observable.
                      static std::mutex peek_mutex;
                      static std::map<uint64_t, uint64_t> peek_counts;
                      uint64_t seen = 0;
                      {
                          std::lock_guard lock(peek_mutex);
                          seen = ++peek_counts[sampled_source_addr];
                      }
                      if ((seen & (seen - 1)) == 0) {
                          size_t window = std::min<size_t>(
                              prosper::gpu::gpu_capture_resource_footprint(r),
                              size_t(1) << 20);
                          while (window &&
                                 !prosper::gpu::guest_readable(
                                     sampled_source_addr,
                                     static_cast<uint32_t>(window)))
                              window /= 2;
                          size_t nz = 0;
                          const uint8_t* bytes = reinterpret_cast<const uint8_t*>(
                              static_cast<uintptr_t>(sampled_source_addr));
                          for (size_t i = 0; i < window; ++i) nz += bytes[i] != 0;
                          fprintf(stderr,
                                  "[rtt-guestpeek] addr=0x%llx %ux%u fmt=%u sample#%llu "
                                  "srt=0x%x sgpr=%u pc=%u "
                                  "guest non-zero=%zu/%zu bytes (%.1f%%) resolved=%s\n",
                                  (unsigned long long)sampled_source_addr, tw, th,
                                  (unsigned)r.format, (unsigned long long)seen,
                                  // Descriptor PROVENANCE, which decides whether a wrong
                                  // address can be stale at all. sgpr_base valid means the
                                  // descriptor came from THIS draw's user data and is fresh
                                  // by construction; srt_offset valid means it was loaded
                                  // from a table, which a first-fold read can pin.
                                  r.srt_offset, r.sgpr_base, r.fetch_pc,
                                  nz, window,
                                  window ? 100.0 * nz / window : 0.0,
                                  has_uniform_live_rtt ? "HIT-UNIFORM"
                                      : rtt_hit ? (has_gpu_live_rtt ? "HIT-GPU"
                                                                    : "HIT-CPU")
                                      : has_ds_live ? "DS" : "miss");
                      }
                  }
                  if (rtt_log)
                      fprintf(stderr,
                              "[rtt] sample tex addr=0x%llx %ux%u fmt=%u -> %s "
                              "(cache_size=%zu)\n",
                              (unsigned long long)sampled_source_addr, tw, th,
                              (unsigned)r.format,
                              // Which hit path, not merely that one hit. A UNIFORM-colour
                              // surface samples as one flat value everywhere, so a shader
                              // reading it produces correct geometry with no texture
                              // detail at all -- visually a smooth gradient inside a
                              // correct mask. That is indistinguishable from a real hit
                              // in a log that prints one word for all three paths, and it
                              // is exactly what GTA V's lighting output looks like.
                              has_uniform_live_rtt ? "HIT-UNIFORM"
                                  : rtt_hit ? (has_gpu_live_rtt ? "HIT-GPU" : "HIT-CPU")
                                  : has_ds_live
                                      ? (sampled_ds.aspect == VK_IMAGE_ASPECT_STENCIL_BIT
                                             ? "DS-STENCIL" : "DS-DEPTH")
                                      : "miss",
                              g_rtt.size());
              }
              // CUBE texture (#273 — DOLL's title reflection probes / skybox): decode six
              // independent thin-2D faces into the stacked w x 6h image addressed by the
              // cube-sample lowering. A selected mip is not six tightly packed levels: each
              // face owns a complete aligned mip chain, so layer_stride/mip_offset select the
              // same level from each face. Packed-tail levels retain each face-chain base and
              // use the tail coordinates. BCn and native-width fp16 follow the same layout.
              // GFX8-GFX10 embeds four self-contained 0/1 fast clears in DCC metadata. A
              // uniform metadata surface means every compression block has that value, so it
              // can be materialized without interpreting compressed base bytes. Uniform 0xff
              // means an emulated writer published ordinary uncompressed base texels and the
              // normal format/detile path below is authoritative.
              bool dcc_fast_clear_done = false;
              bool dcc_uncompressed = false;
              if (r.compression_enabled && !rtt_hit && !fr.is_storage_image) {
                  const uint64_t metadata_bytes = sampled_dcc_metadata_size;
                  const std::vector<uint8_t>& metadata = sampled_dcc_metadata;
                  const size_t metadata_got = sampled_dcc_metadata_got;
                  uint8_t clear_code = 0;
                  dcc_uncompressed = persistent_dcc_uncompressed;
                  if (!dcc_uncompressed && metadata_got == metadata.size() &&
                      prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                          texture_pixels.data(), texture_pixels.size() / 4,
                          metadata.data(), metadata.size(), r.num_components,
                          r.alpha_is_on_msb, &clear_code)) {
                      dcc_fast_clear_done = true;
                      static std::set<std::pair<uint64_t, uint64_t>> decoded_dcc_images;
                      if (decoded_dcc_images.emplace(r.gpu_addr, r.metadata_addr).second)
                          fprintf(stderr,
                                  "[render] DCC fast-clear addr=0x%llx meta=0x%llx "
                                  "%ux%ux%u fmt=%u code=0x%02x bytes=%zu\n",
                                  (unsigned long long)r.gpu_addr,
                                  (unsigned long long)r.metadata_addr,
                                  tw, th, r.depth, (unsigned)r.format, clear_code,
                                  metadata.size());
                  } else if (!dcc_uncompressed) {
                      static std::set<std::pair<uint64_t, uint64_t>> warned_dcc_images;
                      if (warned_dcc_images.emplace(r.gpu_addr, r.metadata_addr).second) {
                          // WHICH KIND this metadata is, by positive correlation with
                          // retained renderer state -- never inferred from the descriptor.
                          // T# WORD6[21] (`compression_enabled`) is set for depth/stencil
                          // and colour alike, and AMD PAL puts GetHtile256BAddr() in the
                          // same `meta_data_address` field it uses for DCC, so the
                          // descriptor cannot separate them. This branch assumed DCC,
                          // which is how GTA V's main depth (0x2052ac0000, HTILE at
                          // 0x2055310000) came to be reported as "DCC-compressed" (#2674).
                          //
                          // This is the FIRST production caller of the correlator anywhere
                          // in the tree. It was easy to assume the compute path already
                          // used it -- a query is registered for it in this file -- but
                          // registering a query is not calling one, and on master every
                          // reference to `classify_compression_metadata_kind` outside its
                          // own definition is a test.
                          //
                          // Classified HERE, inside the first-sighting guard, rather than
                          // beside the footprint: the query walks the retained DS cache and
                          // the RTT map, which is O(surfaces) and this path runs per
                          // sampled texture. Behind the guard it runs at most once per
                          // (base, metadata) pair for the process, so the hot path is
                          // unchanged.
                          //
                          // Used ONLY TO REPORT. It deliberately does not change which
                          // footprint is taken or what is authorized: the HTILE helper is
                          // fail-closed on sample_count != 4 and this surface is
                          // single-sample, so routing it there today returns 0 exactly as
                          // the DCC helper does, while changing the path every other title
                          // takes. Widening that gate needs its own evidence (#2674).
                          const prosper::gpu::CompressionMetadataKind kind =
                              prosper::gpu::classify_compression_metadata_kind(
                                  prosper::gpu::MetadataKindRequest{
                                      r.metadata_addr, r.gpu_addr, r.format,
                                      r.num_components, r.img_dim});
                          // Name the KIND, and print the inputs the sizing helpers
                          // actually gate on. The previous line called every declined
                          // surface "DCC-compressed" and omitted `samples` and
                          // `pipe_aligned` -- two of the three conditions in
                          // gfx10_htile_msaa_metadata_bytes(). The DCC sizer gates only on
                          // tile mode and bytes-per-texel; it takes `pipe_aligned` and
                          // explicitly discards it (`(void)pipe_aligned` in tile.cpp), so
                          // that field is diagnostic for the HTILE route only. Either way
                          // the old line could not say WHICH gate failed, and a reader
                          // chasing a depth surface was sent looking for a DCC defect.
                          //
                          // `first=` is printed only when bytes were actually read.
                          // metadata=0/0 means the footprint was zero, so nothing was
                          // fetched and the first byte is an unread default, not a
                          // measurement of the plane -- reporting 0x00 there stated a fact
                          // about our own buffer as though it were one about the guest.
                          //
                          // `ds_live=` disambiguates the TWO independent reasons the
                          // footprint can be zero, which the old line could not separate:
                          // the size expression is gated on `!has_ds_live` outright, or the
                          // sizing helper fail-closed on the shape. Without it,
                          // "metadata=0/0" is compatible with both and a reader is free to
                          // pick the one that suits their hypothesis -- which is how a gate
                          // analysis gets written against the wrong gate. One was.
                          //
                          // Read `ds_live=0` as "the size expression called a helper", NOT
                          // as "no retained DS image exists". The lookup above is itself
                          // gated on `!has_live_rtt && !is_storage_image && img_dim == 1 &&
                          // cls == Texture`, so a binding the gate rejects never reaches it
                          // and reads 0 for a reason that has nothing to do with residency
                          // -- on an `img_dim == 6` surface it is structurally 0.
                          //
                          // `dim=` and `ncomp=` are printed because the ROUTING, not the
                          // sample count, decides which gate applies:
                          // `dcc_metadata_footprint` reaches the HTILE sizer only for
                          // `img_dim == 6 && Float32 && num_components == 1`, and
                          // otherwise uses the DCC sizer, which fail-closes on tile mode
                          // and never looks at `sample_count`. All three conjuncts must be
                          // visible: `fmt=` cannot stand in for `ncomp=`, because it
                          // prints the DataFormat enum and `Float32` is reached from raw
                          // IMG_FMT 22/64/74/77 with one, two, three or four components.
                          // With any conjunct missing, "the sample count blocked it" and
                          // "it was never routed to the HTILE sizer at all" stay
                          // indistinguishable -- which they were, and a gate analysis was
                          // written against the wrong gate because of it.
                          const char* kind_name =
                              kind == prosper::gpu::CompressionMetadataKind::Htile
                                  ? "HTILE"
                                  : kind == prosper::gpu::CompressionMetadataKind::Dcc
                                        ? "DCC"
                                        : "UNCORRELATED";
                          char first_text[24] = "(unread)";
                          if (metadata_got)
                              std::snprintf(first_text, sizeof first_text, "0x%02x",
                                            metadata[0]);
                          fprintf(stderr,
                                  "[render] compressed sampled image kind=%s draw=%llu "
                                  "order=%llu fs=0x%llx fs-id=%llu target=0x%llx "
                                  "addr=0x%llx "
                                  "meta=0x%llx %ux%ux%u fmt=%u ncomp=%u dim=%u tile=%u "
                                  "samples=%u pipe_aligned=%u ds_live=%u is unsupported; "
                                  "metadata=%zu/%llu first=%s\n",
                                  kind_name,
                                  (unsigned long long)draw.draw_index,
                                  (unsigned long long)draw.command_order,
                                  (unsigned long long)draw.fs_guest_addr,
                                  (unsigned long long)draw.fs_identity,
                                  (unsigned long long)draw.color0_base,
                                  (unsigned long long)r.gpu_addr,
                                  (unsigned long long)r.metadata_addr,
                                  tw, th, r.depth, (unsigned)r.format, r.num_components,
                                  r.img_dim, r.tile_mode, r.sample_count,
                                  r.meta_pipe_aligned ? 1u : 0u,
                                  has_ds_live ? 1u : 0u, metadata_got,
                                  (unsigned long long)metadata_bytes, first_text);
                      }
                  }
              }
              // PROSPER_DS_UNBRIDGED_FAR=1 — DIAGNOSTIC ONLY, never a shipped default.
              //
              // A depth surface prosper rendered into a retained DS image but could not
              // bridge decodes from guest memory the renderer never wrote, i.e. zeros. For
              // a shadow map zero is the NEAR plane, so every shadow test reads "occluded"
              // and the lighting pass outputs black over a perfectly good G-buffer. Filling
              // with far instead flips that to "unoccluded", which is equally wrong as
              // output but is a decisive discriminator: if the scene lights up, the
              // unbridged depth surface is the cause; if it does not, the hypothesis is
              // dead and no cube-assembly work is justified. It detects its own invalidity
              // in both directions, and it reports whether it fired at all.
              //
              // TWO POLES, and which one means "unoccluded" is not knowable a priori: a
              // reversed-Z depth buffer stores near at 1.0, so filling 0xff is the SAME
              // claim as the zeros it was meant to contradict. The experiment is only
              // decisive if both poles are run, so the fill byte is a parameter
              // (PROSPER_DS_UNBRIDGED_FILL, default 0xff) rather than a constant.
              //
              // Restricted to cubes by PROSPER_DS_UNBRIDGED_FAR=cube. The first run filled
              // every unbridged retained DS plane including the main 4K depth, which is
              // depth-tested against by the HUD -- the radar vanished, so the arm measured
              // the confound as much as the subject.
              static const std::string far_mode =
                  PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FAR")
                      ? PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FAR") : "";
              static const uint8_t far_fill = PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FILL")
                  ? static_cast<uint8_t>(strtoul(
                        getenv("PROSPER_DS_UNBRIDGED_FILL"), nullptr, 0))
                  : 0xffu;
              // PROSPER_DS_UNBRIDGED_FILL_F32=<float> — fill FLOAT TEXELS, not bytes.
              //
              // The byte fill above is correct for a UNORM depth, where 0xff really is 1.0.
              // It is NOT correct for a Float32 depth (`fmt=1`), where 0xffffffff is
              // **NaN**: every comparison against it is false, so the "far" pole is not far,
              // it is a third failure mode. A two-pole run over {0x00, 0xff} on a Float32
              // surface therefore tests {0.0f, NaN} and never tests 1.0f at all -- which
              // silently voids the discriminator this lever exists to be. One such run was
              // published as a falsification before the arithmetic was checked.
              static const bool have_f32_fill =
                  PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FILL_F32") != nullptr;
              static const float f32_fill = have_f32_fill
                  ? strtof(PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FILL_F32"), nullptr)
                  : 1.0f;
              if (!rtt_hit && !has_ds_live && !fr.is_storage_image &&
                  !far_mode.empty() && far_mode != "0" &&
                  (far_mode != "cube" || r.img_dim == 3u) &&
                  prosper::test::is_retained_ds_plane(r.gpu_addr)) {
                  const bool f32_path = have_f32_fill &&
                      r.format == prosper::gpu::DataFormat::Float32 &&
                      texture_pixels.size() % sizeof(float) == 0;
                  if (f32_path) {
                      float* texels = reinterpret_cast<float*>(texture_pixels.data());
                      std::fill(texels, texels + texture_pixels.size() / sizeof(float),
                                f32_fill);
                  } else {
                      std::fill(texture_pixels.begin(), texture_pixels.end(), far_fill);
                  }
                  // PER-ADDRESS count, not just a global one. The report prints once per
                  // address, so a global running total cannot say how many of a given
                  // surface's missed samples the fill actually reached -- and that is
                  // exactly the number a reader needs: this lever only fires when
                  // `is_retained_ds_plane` holds, so it can cover an arbitrary fraction of
                  // the misses a run observes. Without it, "filled, and nothing changed"
                  // is not a falsification, because the fill may have reached 2 of 161.
                  static std::mutex far_mutex;
                  static std::map<uint64_t, uint64_t> far_hits;
                  static std::atomic<uint64_t> far_count{0};
                  const uint64_t n = far_count.fetch_add(1) + 1;
                  bool first = false;
                  uint64_t per_addr = 0;
                  {
                      std::lock_guard lock(far_mutex);
                      per_addr = ++far_hits[r.gpu_addr];
                      first = per_addr == 1;
                  }
                  // Re-report on a power-of-two schedule so coverage is visible without
                  // flooding: 1, 2, 4, 8 ... fills per address.
                  first = first || (per_addr & (per_addr - 1)) == 0;
                  char fill_text[32];
                  if (f32_path)
                      std::snprintf(fill_text, sizeof fill_text, "%gf", f32_fill);
                  else
                      std::snprintf(fill_text, sizeof fill_text, "0x%02x",
                                    (unsigned)far_fill);
                  if (first)
                      fprintf(stderr,
                              "[ds-far] addr=0x%llx %ux%ux%u dim=%u fmt=%u path=%s "
                              "filled=%s (unbridged retained DS plane; "
                              "fills-this-addr=%llu total=%llu)\n",
                              (unsigned long long)r.gpu_addr, tw, th, r.depth, r.img_dim,
                              (unsigned)r.format, f32_path ? "f32" : "byte",
                              fill_text, (unsigned long long)per_addr,
                              (unsigned long long)n);
                  rtt_hit = true;   // do not overwrite it with a guest-byte decode below
                  resource_rtt_hit = true;
              }
              bool cube_done = dcc_fast_clear_done && is_cube;
              // Deliberately NOT folded into cube_done: the two publish through different
              // channels, and reusing one flag would multiply an array's height by six.
              bool array_done = false;
              // CUBE DEPTH BRIDGE. A guest depth cube is ONE six-layer allocation whose
              // faces prosper retains as separate DS images (each keyed by its
              // DB_DEPTH_VIEW slice). The recompiler lowers a cube sample onto a single
              // vertically stacked w x 6h image, so the faces are gathered and restacked
              // here -- nothing can bind six images as one cube.
              //
              // Without this the sample fell through to a guest-byte decode of memory the
              // renderer never writes, i.e. zeros. For an omnidirectional shadow map zero
              // is one pole of the depth range, so every shadow test answered the same way
              // everywhere; a two-pole experiment showed the value can drive GTA V's
              // lighting output from its normal content to exactly zero, so this input is
              // load-bearing rather than cosmetic.
              //
              // The guest surface is Z16 (DB_Z_INFO.FORMAT=1) while prosper canonicalises
              // host attachments to D32_SFLOAT, so the conversion is NUMERIC -- clamp to
              // [0,1] and quantise -- not a bit reinterpretation, and it must not invert:
              // reversed-Z is already carried by the stored values and by the comparison
              // direction. CONFIDENCE: MED on the 8-bit staging below; the value is
              // faithful, the precision is not the guest's, and a shadow compare is
              // precision-sensitive. Recorded rather than hidden.
              bool cube_depth_bridged = false;
              bool renderer_cube_snapshot_ready = false;
              if (is_cube && !rtt_hit && !resource_compute_depth_hybrid &&
                  !fr.is_storage_image && r.depth == 6u &&
                  prosper::test::is_retained_ds_plane(r.gpu_addr)) {
                  static const bool mapped_depth_cube =
                      std::getenv("PROSPER_NO_MAPPED_DEPTH_CUBE") == nullptr;
                  std::array<std::vector<float>, 6> faces;
                  std::array<prosper::test::PersistentDsDepthReadback, 6> mapped_faces;
                  uint32_t slices_found = 0;
                  std::string cube_error;
                  // Publish the consumer descriptor's layer stride so the DS invalidation
                  // can address each face's own bytes. This is the only authority this
                  // title offers: it never programs DB_DEPTH_SLICE.
                  if (r.layer_stride_bytes)
                      prosper::test::note_ds_layer_stride(r.gpu_addr, tw, th,
                                                          r.layer_stride_bytes);
                  uint32_t present_mask = retained_depth_cube.present_mask;
                  uint32_t known_mask = retained_depth_cube.known_mask;
                  const bool use_mapped_depth_cube = mapped_depth_cube &&
                      prosper::test::mapped_depth_cube_payload_within_limit(tw, th, present_mask);
                  for (uint32_t face = 0; face < 6u; ++face)
                      slices_found += (present_mask >> face) & 1u;
                  // Selection above is metadata-only. Read every selected renderer face;
                  // missing faces retain the independent guest fallback below. A failure
                  // must not publish a cache entry under the selected renderer generation.
                  using TexCensus = prosper::frontend::TextureReferenceCensus;
                  const uint64_t census_cube_read_start =
                      texref_census ? TexCensus::aux_begin() : 0;
                  const bool readback_ok = retained_depth_cube_cache_candidate &&
                      (use_mapped_depth_cube
                          ? prosper::test::read_persistent_ds_cube_depth_mapped(
                                r.gpu_addr, tw, th, mapped_faces, slices_found,
                                cube_error, &present_mask, &known_mask)
                          : prosper::test::read_persistent_ds_cube_depth(
                                r.gpu_addr, tw, th, faces, slices_found, cube_error,
                                &present_mask, &known_mask));
                  if (texref_census && retained_depth_cube_cache_candidate)
                      texref_census->aux_end(TexCensus::kAuxCubeRead, census_cube_read_start);
                  // Residency DISTRIBUTION, not a first-sight snapshot. "1 of 6 faces"
                  // seen once is consistent with three different worlds: the guest
                  // amortises faces across frames, prosper's invalidation evicts them
                  // faster than they are re-rendered, or the sample simply landed
                  // mid-sequence. A histogram over every sample separates them -- if the
                  // count never reaches 6, no timing argument survives.
                  {
                      static std::mutex cube_mutex;
                      static std::map<std::pair<uint64_t, uint32_t>, uint64_t> hist;
                      static std::map<uint64_t, std::pair<uint32_t, uint32_t>> masks;
                      static uint64_t samples = 0;
                      std::lock_guard lock(cube_mutex);
                      ++hist[{r.gpu_addr, slices_found}];
                      auto& seen_masks = masks[r.gpu_addr];
                      seen_masks.first |= present_mask;   // any slice ever VALID
                      seen_masks.second |= known_mask;    // any slice ever KNOWN
                      // Increment SEQUENCED before the test: `(++samples) & (samples - 1)`
                      // reads and modifies `samples` with no sequencing between the
                      // operands of `&`, so the throttle's cadence was whatever the
                      // optimiser chose. That matters here beyond tidiness -- CLAUDE.md
                      // tells readers to reason about this power-of-two schedule when
                      // interpreting the log, and that reasoning is only sound if the
                      // schedule is defined. Do not fold these back together.
                      ++samples;
                      if ((samples & (samples - 1)) == 0 && samples >= 64) {
                          fprintf(stderr,
                                  "[cube-depth] residency after %llu cube samples "
                                  "(faces resident -> times seen):\n",
                                  (unsigned long long)samples);
                          for (const auto& e : hist)
                              fprintf(stderr,
                                      "[cube-depth]   addr=0x%llx faces=%u/6 x%llu%s\n",
                                      (unsigned long long)e.first.first, e.first.second,
                                      (unsigned long long)e.second,
                                      e.first.second == 6u ? "  <-- bridgeable" : "");
                          for (const auto& e : masks)
                              fprintf(stderr,
                                      "[cube-depth]   addr=0x%llx slices ever VALID=0x%02x "
                                      "ever KNOWN=0x%02x (bit n = slice n; 0x3f = all six)"
                                      "\n",
                                      (unsigned long long)e.first, e.second.first,
                                      e.second.second);
                      }
                  }
                  (void)cube_error;
                  if (readback_ok && texture_pixels.size() >=
                          static_cast<size_t>(tw) * th * 6u * 4u) {
                      const uint64_t census_cube_fill_start =
                          texref_census ? TexCensus::aux_begin() : 0;
                      const bool ctiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                          !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                      auto face_base = [&](uint32_t face, size_t selected_span) {
                          const uint64_t stride = r.layer_stride_bytes
                              ? r.layer_stride_bytes : selected_span;
                          return r.gpu_addr + static_cast<uint64_t>(face) * stride;
                      };
                      for (uint32_t face = 0; face < 6u; ++face) {
                          uint8_t* dst = texture_pixels.data() +
                              static_cast<size_t>(face) * tw * th * 4u;
                          const bool retained_face = use_mapped_depth_cube
                              ? mapped_faces[face].valid() : !faces[face].empty();
                          if (retained_face) {
                              const size_t count = use_mapped_depth_cube
                                  ? mapped_faces[face].count : faces[face].size();
                              const size_t pixels = std::min(static_cast<size_t>(tw) * th, count);
                              if (use_mapped_depth_cube)
                                  quantize_depth_cube_rgba8(
                                      dst, static_cast<const uint8_t*>(mapped_faces[face].mapped),
                                      pixels);
                              else
                                  quantize_depth_cube_rgba8(
                                      dst, reinterpret_cast<const uint8_t*>(faces[face].data()),
                                      pixels);
                          } else {
                              const uint32_t source_bpt = bpt ? bpt : 2u;
                              const size_t linear_bytes = static_cast<size_t>(tw) * th * source_bpt;
                              const size_t surface_bytes = ctiled
                                  ? prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, source_bpt)
                                  : linear_bytes;
                              const uint64_t selected_addr = face_base(face, surface_bytes);
                              std::vector<uint8_t> traw(surface_bytes, 0);
                              const size_t got = copy_resource(traw.data(), selected_addr, surface_bytes);
                              std::vector<uint8_t> linear(linear_bytes, 0);
                              if (got >= linear.size() && ctiled) {
                                  prosper::gpu::detile_surface(
                                      linear.data(), traw.data(), tw, th, r.tile_mode, 0, source_bpt);
                              } else if (got >= linear.size()) {
                                  std::memcpy(linear.data(), traw.data(), linear.size());
                              }
                              if (got >= linear.size()) {
                                  const uint16_t* u16 = reinterpret_cast<const uint16_t*>(linear.data());
                                  for (size_t i = 0; i < static_cast<size_t>(tw) * th; ++i) {
                                      const uint8_t q = static_cast<uint8_t>((u16[i] >> 8) & 0xffu);
                                      dst[i * 4 + 0] = q;
                                      dst[i * 4 + 1] = q;
                                      dst[i * 4 + 2] = q;
                                      dst[i * 4 + 3] = 0xffu;
                                  }
                              } else {
                                  for (size_t i = 0; i < static_cast<size_t>(tw) * th; ++i) {
                                      dst[i * 4 + 0] = 0xffu;
                                      dst[i * 4 + 1] = 0xffu;
                                      dst[i * 4 + 2] = 0xffu;
                                      dst[i * 4 + 3] = 0xffu;
                                  }
                              }
                          }
                      }
                      renderer_cube_snapshot_ready = depth_cube_source_layout_valid &&
                          present_mask == retained_depth_cube.present_mask;
                      for (uint32_t face = 0; face < 6; ++face)
                          renderer_cube_snapshot_ready &= (use_mapped_depth_cube
                              ? mapped_faces[face].count : faces[face].size()) ==
                              ((present_mask & (1u << face)) ? static_cast<size_t>(tw) * th : 0);
                      if (texref_census)
                          texref_census->aux_end(TexCensus::kAuxCubeFill, census_cube_fill_start);
                      cube_depth_bridged = true;
                      rtt_hit = true;          // do not overwrite with a guest-byte decode
                      resource_rtt_hit = true;
                      // The loop above wrote all SIX faces into texture_pixels. Publishing
                      // is driven by `cube_done` -- `fr.th = cube_done ? th * 6u : th` at
                      // the upload seam -- so leaving it false uploaded one face and
                      // silently dropped five, and the stacked-face lowering could never
                      // see them. The ordinary cube path sets this for the same reason.
                      cube_done = true;
                  }
              }
              // The existing authority checks above own renderer/compute-produced
              // data. Only an ordinary guest-backed FP16 surface reaches this conversion.
              // Cubes keep the shader's established vertically stacked RGBA8 representation.
              const bool gpu_detile_candidate = gpu_detile_shape && !rtt_hit &&
                  !dcc_fast_clear_done && !cube_depth_bridged &&
                  !resource_compute_depth_hybrid;
              if (gpu_detile_candidate) {
                  fr.gpu_detile = prosper::test::prepare_render_gpu_detile(
                      tw, th, is_cube ? 6u : 1u,
                      is_cube ? r.gpu_addr : sampled_source_addr,
                      r.layer_stride_bytes, r.layer_mip_offset_bytes, copy_resource,
                      r.num_components,
                      portable_raw_uvec4_storage ? prosper::gpu::Float16DetileOutput::RawUvec4
                                                 : prosper::gpu::Float16DetileOutput::Rgba8,
                      writable_storage_image);
                  if (fr.gpu_detile) {
                      if (timing_enabled) {
                          ++pending_timing.gpu_detile_preparations;
                          pending_timing.gpu_detile_2d_preparations += !is_cube;
                          pending_timing.gpu_detile_source_bytes += fr.gpu_detile->source_bytes;
                      }
                      cube_done = is_cube;
                      texture_pixels.clear();
                      fr.persistent_texture_id = 0;
                      fr.persistent_texture_version = 0;
                  }
              }
              // Short backing or unsupported device limits must restore CPU scratch
              // for both ordinary 2D and cube recovery before either writes pixels.
              if (defer_detile_pixels && !fr.gpu_detile) texture_pixels.resize(nb);
              if ((is_cube || is_array) && !cube_done && !rtt_hit && !dcc_fast_clear_done &&
                  !cube_depth_bridged) {
                  if (float32_array && r.compression_enabled) {
                      static thread_local uint32_t array_reject_logged = 0;
                      if (log_array_rejection(array_reject_logged, "compressed-guest"))
                          std::fprintf(stderr,
                              "[render-array-reject] binding=%u compressed Float32 guest backing addr=0x%llx\n",
                              r.binding, (unsigned long long)r.gpu_addr);
                      return {ImageDisposition::Reject, DropReason::ArrayCompressedGuest};
                  }
                  const uint32_t cb = prosper::gpu::bc_block_bytes(r.format);
                  const bool ctiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                      !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  auto face_base = [&](uint32_t face, size_t selected_span) {
                      const uint64_t stride = r.layer_stride_bytes
                          ? r.layer_stride_bytes : selected_span;
                      return r.gpu_addr + static_cast<uint64_t>(face) * stride;
                  };
                  const uint32_t slice_count = is_cube ? 6u : decoded_layers;
                  std::vector<uint32_t> slice_short(slice_count, 0);
                  uint32_t slice_short_count = 0;
                  // Reports the GUEST's declared layout only. Deliberately not the
                  // decoded slice count: that is what PROSPER_ARRAY_DECODE_BUDGET_MIB
                  // measures, and a diagnostic that prints a field a different switch
                  // decided is one whose output cannot be trusted on its own -- which
                  // check_diag_gates.py enforces (TWO-GATE).
                  if (!is_cube && PROSPER_ENV_ON("PROSPER_SLICESTRIDE")) {
                      static std::unordered_set<uint64_t> reported;
                      if (reported.insert(r.gpu_addr).second)
                          fprintf(stderr,
                                  "[slicestride] addr=0x%llx %ux%u guest_depth=%u fmt=%u "
                                  "layer_stride=%u mip_off=%u in_mip_tail=%d "
                                  "mip_tail_bytes=%u declared_mips=%u size=%u\n",
                                  (unsigned long long)r.gpu_addr, tw, th, r.depth,
                                  (unsigned)r.format, r.layer_stride_bytes,
                                  r.layer_mip_offset_bytes, (int)r.in_mip_tail,
                                  r.mip_tail_bytes, r.declared_mip_levels, r.size);
                  }
                  for (uint32_t fface = 0; fface < slice_count; fface++) {
                      uint8_t* slice = texture_pixels.data() + (size_t)fface * tw * th * output_bpp;
                      if (cb) {
                          uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                          size_t comp = (size_t)bw * bh * cb;
                          const size_t surface_bytes = ctiled
                              ? prosper::gpu::tiled_elements_bytes(
                                    bw, bh, cb, r.tile_mode)
                              : comp;
                          const size_t selected_span = r.in_mip_tail
                              ? r.mip_tail_bytes : surface_bytes;
                          const uint64_t selected_addr = face_base(fface, selected_span) +
                              (r.in_mip_tail ? 0u : r.layer_mip_offset_bytes);
                          std::vector<uint8_t> lin(comp, 0);
                          // #2998: keep the byte count. safe_copy stops at the first
                          // uncommitted 64 KiB page and leaves the tail zero-filled, so a
                          // slice prosper could not READ is byte-identical to one the
                          // guest never WROTE -- and every "the atlas is mostly empty"
                          // conclusion turns on telling those apart. The non-BC branch
                          // below already checks its `got`; this one discarded it.
                          size_t slice_got = 0;
                          if (ctiled) {
                              std::vector<uint8_t> traw(selected_span, 0);
                              slice_got = copy_resource(
                                  traw.data(), selected_addr, selected_span);
                              if (r.in_mip_tail)
                                  prosper::gpu::detile_elements_level(
                                      lin.data(), traw.data(), traw.size(), bw, bh, cb,
                                      r.tile_mode, r.mip_tail_x, r.mip_tail_y);
                              else
                                  prosper::gpu::detile_elements(
                                      lin.data(), traw.data(), traw.size(), bw, bh, cb,
                                      r.tile_mode);
                          } else if (linear_padded_read) {
                              slice_got = copy_linear_padded_rows_from(
                                  lin.data(), selected_addr,
                                  static_cast<size_t>(bw) * cb, bh);
                          } else {
                              slice_got = copy_resource(lin.data(), selected_addr, comp);
                          }
                          if (slice_short_count < slice_count &&
                              slice_got < (ctiled ? selected_span : comp))
                              slice_short[slice_short_count++] = fface;
                          std::vector<uint8_t> face((size_t)tw * th * 4, 0);
                          prosper::gpu::bc_decode_surface(face.data(), lin.data(), lin.size(), tw, th, r.format);
                          std::memcpy(slice, face.data(), face.size());
                      } else {
                          const uint32_t source_bpt = bpt;
                          const size_t linear_bytes =
                              static_cast<size_t>(tw) * th * source_bpt;
                          const size_t surface_bytes = ctiled
                              ? prosper::gpu::tiled_surface_bytes(
                                    tw, th, r.tile_mode, 0, source_bpt)
                              : linear_bytes;
                          const size_t selected_span = r.in_mip_tail
                              ? r.mip_tail_bytes
                              : (float32_array && !ctiled && linear_padded_read
                                     ? linear_src_row * th : surface_bytes);
                          const uint64_t selected_addr = face_base(fface, selected_span) +
                              (r.in_mip_tail ? 0u : r.layer_mip_offset_bytes);
                          std::vector<uint8_t> linear(linear_bytes, 0);
                          size_t nbc_got = 0, nbc_want = 0;
                          if (ctiled) {
                              std::vector<uint8_t> traw(selected_span, 0);
                              const size_t got = copy_resource(
                                  traw.data(), selected_addr, selected_span);
                              nbc_got = got; nbc_want = selected_span;
                              if (got < linear.size()) {
                                  copy_resource(
                                      linear.data(), selected_addr, linear.size());
                              } else if (r.in_mip_tail) {
                                  prosper::gpu::detile_surface_level(
                                      linear.data(), traw.data(), got, tw, th,
                                      r.tile_mode, source_bpt,
                                      r.mip_tail_x, r.mip_tail_y);
                              } else {
                                  prosper::gpu::detile_surface(
                                      linear.data(), traw.data(), tw, th,
                                      r.tile_mode, 0, source_bpt);
                              }
                          } else if (linear_padded_read) {
                              nbc_want = static_cast<size_t>(tw) * source_bpt * th;
                              // #325: use the SAME pitch the single-surface path uses
                              // (`linear_src_row`), not one recomputed from tw/bpt here.
                              // They can differ -- the single-surface figure honours a
                              // registered pitch and its own row width -- and slice 0 of
                              // an array must decode byte-identically to the way that
                              // same surface decoded before it was treated as an array.
                              nbc_got = copy_linear_padded_rows_from(
                                  linear.data(), selected_addr,
                                  static_cast<size_t>(tw) * source_bpt, th);
                          } else if (r.layer_stride_bytes) {
                              const size_t row_pitch = r.linear_row_pitch_bytes
                                  ? r.linear_row_pitch_bytes
                                  : prosper::gpu::linear_sampled_row_pitch(
                                        tw, source_bpt);
                              for (uint32_t y = 0; y < th; ++y)
                                  nbc_got += copy_resource(
                                      linear.data() + static_cast<size_t>(y) * tw * source_bpt,
                                      selected_addr + static_cast<uint64_t>(y) * row_pitch,
                                      static_cast<size_t>(tw) * source_bpt);
                              nbc_want = static_cast<size_t>(tw) * source_bpt * th;
                          } else {
                              nbc_got = copy_resource(linear.data(), selected_addr,
                                                      linear.size());
                              nbc_want = linear.size();
                          }
                          // Measured, not assumed: this branch left slice_short_count
                          // untouched, so an uncompressed array printed short_reads=0
                          // having counted nothing -- a clean result the instrument had
                          // not earned, which is the exact failure it exists to prevent.
                          if (slice_short_count < slice_count &&
                              nbc_want && nbc_got < nbc_want)
                              slice_short[slice_short_count++] = fface;
                          if (float32_array) {
                              // Preserve every source bit, including sub-half precision,
                              // signed zero and NaNs. Missing channels follow texture defaults.
                              const uint32_t nc = source_bpt / 4;
                              for (size_t texel = 0; texel < (size_t)tw * th; ++texel) {
                                  const uint32_t defaults[4] = {0, 0, 0, 0x3f800000u};
                                  uint8_t* pixel = slice + texel * 16;
                                  std::memcpy(pixel, defaults, sizeof(defaults));
                                  std::memcpy(pixel, linear.data() + texel * source_bpt,
                                              nc * sizeof(float));
                              }
                          } else if (f16) {
                              const uint32_t nc = source_bpt / 2;
                              // Same-build performance control for the scalar float path.
                              static const bool old_half_quantization =
                                  PROSPER_ENV_ON("PROSPER_NO_HALF_QUANTIZATION");
                              for (size_t texel = 0; texel < (size_t)tw * th; ++texel) {
                                  uint8_t* pixel = slice + texel * 4;
                                  for (uint32_t c = 0; c < 4; ++c) {
                                      uint16_t half = c == 3 ? 0x3c00u : 0u;
                                      if (c < nc) {
                                          std::memcpy(
                                              &half,
                                              linear.data() + texel * source_bpt + c * 2,
                                              sizeof(half));
                                      }
                                      if (old_half_quantization) {
                                          const float value = c < nc
                                              ? prosper::gpu::half_to_float(half)
                                              : (c == 3 ? 1.0f : 0.0f);
                                          pixel[c] = !std::isfinite(value) || value <= 0.0f
                                              ? 0u : (value >= 1.0f ? 255u
                                                  : static_cast<uint8_t>(value * 255.0f + 0.5f));
                                      } else {
                                          pixel[c] = prosper::gpu::half_to_unorm8(half);
                                      }
                                  }
                              }
                          } else if (source_bpt == 4) {
                              std::memcpy(slice, linear.data(), linear.size());
                          } else {
                              const uint32_t component_bytes =
                                  prosper::gpu::data_format_bytes(r.format);
                              const uint32_t nc = r.num_components ? r.num_components : 1u;
                              for (size_t texel = 0; texel < (size_t)tw * th; ++texel) {
                                  uint8_t* pixel = slice + texel * 4;
                                  pixel[0] = pixel[1] = pixel[2] = 0;
                                  pixel[3] = 255;
                                  for (uint32_t c = 0; c < std::min(nc, 4u); ++c) {
                                      if (component_bytes == 1) {
                                          pixel[c] = linear[texel * source_bpt + c];
                                      } else if (component_bytes == 2) {
                                          uint16_t value = 0;
                                          std::memcpy(
                                              &value,
                                              linear.data() + texel * source_bpt + c * 2,
                                              sizeof(value));
                                          pixel[c] = prosper::gpu::unorm16_to_unorm8(value);
                                      }
                                  }
                              }
                          }
                      }
                  }
                  if (float32_array && slice_short_count) {
                      static thread_local uint32_t array_reject_logged = 0;
                      if (log_array_rejection(array_reject_logged, "short-backing"))
                          std::fprintf(stderr,
                              "[render-array-reject] binding=%u Float32 short backing: "
                              "%u/%u slices, first=%u addr=0x%llx\n", r.binding, slice_short_count,
                              slice_count, slice_short[0], (unsigned long long)r.gpu_addr);
                      return {ImageDisposition::Reject, DropReason::ArrayShortBacking};
                  }
                  // The loop above filled `slice_count` slices. A cube publishes them through
                  // HEIGHT (fr.th = th*6) and an array through LAYERS (fr.sample_count), so the
                  // two completion flags must stay distinct.
                  if (PROSPER_ENV_ON("PROSPER_SLICEMAP") && !is_cube && slice_count > 1) {
                      // #2998: which decoded slices hold content, across the WHOLE range.
                      // A prefix cannot answer the question -- an array whose first slices
                      // decode and whose later ones are empty is identical, in every
                      // per-layer checksum of the first eight, to one that decoded fully.
                      //
                      // Two things this MUST report alongside the map, because without
                      // them its output is not interpretable:
                      //  - the sample rate. "empty" here means "no non-zero byte at a
                      //    sampled offset", and the sample is sparse, so a slice with
                      //    little content can read empty. It is a lower bound on content,
                      //    never an exact count.
                      //  - the SHORT READS. safe_copy stops at the first uncommitted page
                      //    and zero-fills the rest, so a slice prosper could not READ is
                      //    byte-identical to one the guest never WROTE. Reporting empties
                      //    without this invites exactly the wrong conclusion, which is the
                      //    one it invited from me.
                      const size_t lsz = (size_t)tw * th * output_bpp;
                      const size_t step = 61u;
                      std::string map;
                      uint32_t nonempty = 0;
                      for (uint32_t L = 0; L < slice_count; ++L) {
                          if ((size_t)(L + 1) * lsz > texture_pixels.size()) break;
                          const uint8_t* q = texture_pixels.data() + (size_t)L * lsz;
                          bool any = false;
                          for (size_t k = 0; k < lsz && !any; k += step) any = (q[k] != 0);
                          nonempty += any;
                          map += any ? '#' : '.';
                      }
                      fprintf(stderr,
                              "[slicemap] addr=0x%llx %ux%u slices=%u nonempty>=%u "
                              "(1 byte in %zu sampled) short_reads=%u%s map=%s\n",
                              (unsigned long long)r.gpu_addr, tw, th, slice_count,
                              nonempty, step, slice_short_count,
                              slice_short_count ? " <- EMPTY MAY MEAN UNREADABLE" : "",
                              map.c_str());
                      if (slice_short_count)
                          fprintf(stderr, "[slicemap]   first short slice=%u\n",
                                  slice_short[0]);
                  }
                  if (is_cube) cube_done = true; else array_done = true;
                  if (resource_compute_depth_hybrid && !decoded_reuse) {
                      static const bool mapped_depth_cube =
                          std::getenv("PROSPER_NO_MAPPED_DEPTH_CUBE") == nullptr;
                      std::array<std::vector<float>, 6> overlay_faces;
                      std::array<prosper::test::PersistentDsDepthReadback, 6> mapped_faces;
                      uint32_t overlay_mask = 0, known_mask = 0;
                      std::string overlay_error;
                      const bool use_mapped_depth_cube = mapped_depth_cube &&
                          prosper::test::mapped_depth_cube_payload_within_limit(
                              tw, th, resource_compute_depth_overlay_mask);
                      const bool overlay_ok = use_mapped_depth_cube
                          ? prosper::test::read_persistent_ds_cube_depth_after_mapped(
                                r.gpu_addr, tw, th, resource_compute_producer_order,
                                mapped_faces, overlay_mask, known_mask, overlay_error)
                          : prosper::test::read_persistent_ds_cube_depth_after(
                                r.gpu_addr, tw, th, resource_compute_producer_order,
                                overlay_faces, overlay_mask, known_mask, overlay_error);
                      if (overlay_ok && overlay_mask ==
                              resource_compute_depth_overlay_mask) {
                          for (uint32_t face = 0; face < 6u; ++face) {
                              if (!(overlay_mask & (1u << face))) continue;
                              uint8_t* dst = texture_pixels.data() +
                                  static_cast<size_t>(face) * tw * th * 4u;
                              const size_t count = use_mapped_depth_cube
                                  ? mapped_faces[face].count : overlay_faces[face].size();
                              auto quantize = [&](auto read_depth) {
                                  for (size_t i = 0;
                                       i < static_cast<size_t>(tw) * th && i < count;
                                       ++i) {
                                      const float d = std::clamp(read_depth(i), 0.0f, 1.0f);
                                      const uint8_t q = static_cast<uint8_t>(
                                          d * 255.0f + 0.5f);
                                      dst[i * 4 + 0] = q;
                                      dst[i * 4 + 1] = q;
                                      dst[i * 4 + 2] = q;
                                      dst[i * 4 + 3] = 0xffu;
                                  }
                              };
                              if (use_mapped_depth_cube)
                                  quantize([&](size_t i) { return mapped_faces[face].at(i); });
                              else
                                  quantize([&](size_t i) { return overlay_faces[face][i]; });
                          }
                          static std::mutex hybrid_log_mutex;
                          static std::map<uint64_t, uint64_t> hybrid_counts;
                          std::lock_guard lock(hybrid_log_mutex);
                          const uint64_t n = ++hybrid_counts[r.gpu_addr];
                          if ((n & (n - 1)) == 0)
                              std::fprintf(
                                  stderr,
                                  "[cube-depth] compute/DS hybrid addr=0x%llx "
                                  "producer-order=%llu overlay=0x%02x known=0x%02x "
                                  "(x%llu)\n",
                                  (unsigned long long)r.gpu_addr,
                                  (unsigned long long)
                                      resource_compute_producer_order,
                                  overlay_mask, known_mask,
                                  (unsigned long long)n);
                      } else {
                          static uint64_t hybrid_failures = 0;
                          if (((++hybrid_failures) & (hybrid_failures - 1)) == 0)
                              std::fprintf(
                                  stderr,
                                  "[cube-depth] compute/DS hybrid FAILED "
                                  "addr=0x%llx expected=0x%02x got=0x%02x "
                                  "known=0x%02x error=%s (x%llu)\n",
                                  (unsigned long long)r.gpu_addr,
                                  resource_compute_depth_overlay_mask,
                                  overlay_mask, known_mask,
                                  overlay_error.c_str(),
                                  (unsigned long long)hybrid_failures);
                      }
                  }
              }
              // Block-compressed (BC1/2/3): read the (possibly tiled) compressed blocks, block-
              // detile, and decode to RGBA8 in-place. The blocks are the tiled element (BC3 =
              // 16 bytes -> SW_4KB_S 16x16-block micro-tiles). #121.
              const uint32_t bcb = (rtt_hit || cube_done || array_done || dcc_fast_clear_done || fr.gpu_detile)
                  ? 0u : prosper::gpu::bc_block_bytes(r.format);
              if (rtt_hit || cube_done || array_done || dcc_fast_clear_done || fr.gpu_detile) { /* pixels already materialized */ }
              else if (portable_raw_uvec4_storage) {
                  // Storage-image tiling describes the compact guest texels, not the expanded
                  // 16-byte Vulkan representation. Detile at the real guest width first, then
                  // convert every channel into the raw VGPR dword carried by the uvec4 image.
                  const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                      !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  const uint32_t materialize_tile_mode = tiled ? r.tile_mode : 0u;
                  const uint32_t materialize_depth = is_volume ? r.depth : 1u;
                  const size_t source_bytes = storage_image_raw_uvec4_source_bytes(
                      r.format, r.num_components ? r.num_components : 1u,
                      tw, th, materialize_depth, materialize_tile_mode,
                      r.in_mip_tail, r.mip_tail_bytes);
                  std::vector<uint8_t> source(source_bytes);
                  bool decoded = fr.storage_image_contract_valid;
                  const size_t got = decoded && source_bytes
                      ? copy_resource(
                            source.data(), sampled_source_addr, source_bytes)
                      : 0u;
                  const bool materialized = decoded && got == source_bytes &&
                      storage_image_materialize_raw_uvec4(
                          source.data(), got, r.format,
                          r.num_components ? r.num_components : 1u,
                          tw, th, materialize_depth, materialize_tile_mode,
                          r.in_mip_tail, r.mip_tail_bytes,
                          r.mip_tail_x, r.mip_tail_y,
                      reinterpret_cast<uint32_t*>(texture_pixels.data()),
                      texture_pixels.size() / sizeof(uint32_t));
                  decoded = decoded && materialized;
                  if (!decoded) {
                      static std::set<std::tuple<uint32_t, uint32_t, uint32_t>> reported;
                      if (reported.emplace(fr.set, fr.binding,
                                           static_cast<uint32_t>(r.format)).second)
                          std::fprintf(stderr,
                              "[render] storage-image materialize failed: set=%u binding=%u "
                              "source=%zu got=%zu output=%zu fmt=%u comps=%u "
                              "extent=%ux%ux%u tile=%u tail=%u contract=%u\n",
                              fr.set, fr.binding, source_bytes, got,
                              texture_pixels.size(), static_cast<unsigned>(r.format),
                              r.num_components, tw, th, materialize_depth,
                              materialize_tile_mode, r.in_mip_tail ? 1u : 0u,
                              fr.storage_image_contract_valid ? 1u : 0u);
                      fr.storage_image_contract_valid = false;
                      std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                  }
              } else if (bcb && native_bc_mip_chain) {
                  // Every guest level, block-detiled at its own placement and packed level
                  // 0 first -- the layout backend_texture_chain_bytes describes. Levels in
                  // the shared tail read the allocation's first block at their element
                  // coordinates, exactly as a tail-selected single-level view does.
                  const bool tiled_chain = !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  size_t offset = 0;
                  for (uint32_t level = 0; level < native_bc_chain_levels; ++level) {
                      const prosper::gpu::MipChainLevel& L = native_bc_plan->levels[level];
                      const uint32_t lbw = (L.width + 3u) / 4u;
                      const uint32_t lbh = (L.height + 3u) / 4u;
                      const size_t level_bytes = static_cast<size_t>(lbw) * lbh * bcb;
                      if (!tiled_chain || offset + level_bytes > texture_pixels.size()) {
                          std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                          break;
                      }
                      uint8_t* dst = texture_pixels.data() + offset;
                      std::fill(dst, dst + level_bytes, 0);
                      const size_t span = L.in_tail ? L.tail_block_bytes : L.byte_size;
                      prosper::frontend::DecodeScratchPool::Lease src =
                          prosper::frontend::decode_scratch_pool().take(span);
                      src.zero_tail(copy_resource(
                          src.data(), native_bc_chain_base + L.byte_offset, span));
                      if (L.in_tail)
                          prosper::gpu::detile_elements_level(
                              dst, src.data(), span, lbw, lbh, bcb, r.tile_mode,
                              L.tail_x, L.tail_y);
                      else
                          prosper::gpu::detile_elements(
                              dst, src.data(), span, lbw, lbh, bcb, r.tile_mode);
                      offset += level_bytes;
                  }
                  // PROSPER_NATIVE_BC_CHAIN_AUDIT=1: does the level-1 placement hold this
                  // texture's own level 1? A real mip is close to the 2x2 box of level 0; a
                  // foreign or unwritten region is not. Logged once per identity (bounded).
                  if (PROSPER_ENV_ON("PROSPER_NATIVE_BC_CHAIN_AUDIT") &&
                      native_bc_chain_levels > 1 && tw >= 8 && th >= 8) {
                      static std::set<uint64_t> audited;
                      if (audited.size() < 400 && audited.insert(r.gpu_addr).second) {
                          std::vector<uint8_t> l0(size_t(tw) * th * 4), l1(size_t(tw / 2) * (th / 2) * 4);
                          const size_t l0_bytes = size_t((tw + 3) / 4) * ((th + 3) / 4) * bcb;
                          prosper::gpu::bc_decode_surface(l0.data(), texture_pixels.data(), l0_bytes, tw, th, r.format);
                          prosper::gpu::bc_decode_surface(l1.data(), texture_pixels.data() + l0_bytes,
                              texture_pixels.size() - l0_bytes, tw / 2, th / 2, r.format);
                          double mad = 0, mad_shift = 0; size_t n = 0;
                          for (uint32_t y = 0; y < th / 2; ++y)
                              for (uint32_t x = 0; x < tw / 2; ++x)
                                  for (uint32_t c = 0; c < 3; ++c) {
                                      auto at = [&](uint32_t xx, uint32_t yy) { return int(l0[(size_t(yy) * tw + xx) * 4 + c]); };
                                      const int box = (at(2*x,2*y) + at(2*x+1,2*y) + at(2*x,2*y+1) + at(2*x+1,2*y+1) + 2) / 4;
                                      const int v = l1[(size_t(y) * (tw / 2) + x) * 4 + c];
                                      const uint32_t sx = (x + tw / 4) % (tw / 2);
                                      const int vs = l1[(size_t(y) * (tw / 2) + sx) * 4 + c];
                                      mad += std::abs(v - box); mad_shift += std::abs(vs - box); ++n;
                                  }
                          std::fprintf(stderr, "[native-bc-audit] addr=0x%llx fmt=%u %ux%u levels=%u tile=%u "
                                       "l1_vs_box0=%.2f shifted_control=%.2f\n",
                                       static_cast<unsigned long long>(r.gpu_addr),
                                       static_cast<unsigned>(r.format), tw, th, native_bc_chain_levels,
                                       r.tile_mode, n ? mad / n : 0.0, n ? mad_shift / n : 0.0);
                      }
                  }
              } else if (bcb && native_bc_linear_copy) {
                  // Linear, unpadded native BC: one guarded copy of the guest blocks, which
                  // also serves as the persistent cache's validation source.
                  linear_source_prefix_size = copy_resource(
                      texture_pixels.data(), sampled_source_addr,
                      texture_pixels.size());
                  if (linear_source_prefix_size < texture_pixels.size())
                      std::fill(texture_pixels.begin() + linear_source_prefix_size,
                                texture_pixels.end(), 0);
              } else if (bcb) {
                  uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                  // Block-detile: tiled_elements_bytes/detile_elements now derive the 4KB
                  // micro-tile geometry from the block size (bpe) internally (#119) — 16-byte
                  // blocks -> 16x16, 8-byte -> 32x16 — so no tile_side is passed here.
                  size_t comp_bytes = (size_t)bw * bh * bcb;
                  prosper::frontend::DecodeScratchPool::Lease lin;
                  bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  const bool snapshot_source = stage_linear_decoder_source(
                      lin, comp_bytes, tiled);
                  if (tiled) {
                      size_t tbytes = prosper::gpu::tiled_elements_bytes(bw, bh, bcb, r.tile_mode);
                      prosper::frontend::DecodeScratchPool::Lease traw_lease;
                      size_t got = 0;
                      const uint8_t* traw = stage_tiled_source(
                          traw_lease, sampled_source_addr, tbytes, got);
                      if (r.in_mip_tail) {
                          lin.zero_all();
                          prosper::gpu::detile_elements_level(
                              lin.data(), traw, tbytes, bw, bh, bcb,
                              r.tile_mode, r.mip_tail_x, r.mip_tail_y);
                      } else {
                          // detile_elements zero-fills every block the (possibly short)
                          // source cannot supply, so it covers the whole destination for
                          // the element sizes its copiers handle -- pinned by test_tile.
                          if (!prosper::gpu::detile_writes_whole_destination(
                                  r.tile_mode, bcb))
                              lin.zero_all();
                          prosper::gpu::detile_elements(
                              lin.data(), traw, tbytes, bw, bh, bcb, r.tile_mode);
                      }
                  } else if (linear_padded_read) {
                      lin.zero_all();
                      copy_linear_padded_rows(
                          lin.data(), static_cast<size_t>(bw) * bcb, bh);
                  } else if (!snapshot_source) {
                      lin.zero_tail(copy_resource(
                          lin.data(), sampled_source_addr, comp_bytes));
                  }
                  if (native_bc_sampled) {
                      // Native upload: the detiled blocks ARE the texture. (A linear
                      // unpadded surface never reaches here; see native_bc_linear_copy.)
                      if (texture_pixels.size() == comp_bytes)
                          std::memcpy(texture_pixels.data(),
                                      snapshot_source
                                          ? persistent_validation_scratch.data()
                                          : lin.data(),
                                      comp_bytes);
                      else
                          std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                  } else if (!prosper::gpu::bc_decode_surface(
                          texture_pixels.data(), snapshot_source ? persistent_validation_scratch.data()
                                                                : lin.data(),
                          comp_bytes, tw, th, r.format))
                      std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
              } else if (f32) {
                  // Sampled Float32 resources cannot be reinterpreted as RGBA8: a value such as
                  // 1/8192 has bytes 00 00 00 39 and became a zero red channel. Preserve the float
                  // ordinary values and useful HDR range by narrowing each present component to
                  // native RGBA16F. This deliberately loses Float32 precision and extreme range;
                  // the backend carries the resulting half values losslessly to the sampler.
                  // Power-of-two component counts keep the source texel size compatible with the
                  // supported GFX10 surface detilers (4/8/16 B per texel).
                  const uint32_t nc = bpt / 4;
                  prosper::frontend::DecodeScratchPool::Lease flin;
                  const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                      !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  const bool snapshot_source = stage_linear_decoder_source(
                      flin, volume_texels * bpt, tiled);
                  if (tiled) {
                      const size_t tbytes = is_volume
                          ? prosper::gpu::tiled_volume_bytes(
                                tw, th, r.depth, r.tile_mode, bpt)
                          : prosper::gpu::tiled_surface_bytes(
                                tw, th, r.tile_mode, 0, bpt);
                      prosper::frontend::DecodeScratchPool::Lease traw_lease;
                      size_t got = 0;
                      const uint8_t* traw = stage_tiled_source(
                          traw_lease, sampled_source_addr, tbytes, got);
                      if (got < flin.size()) {
                          flin.zero_tail(copy_resource(
                              flin.data(), sampled_source_addr, flin.size()));
                      } else if (is_volume) {
                          flin.zero_all();
                          prosper::gpu::detile_volume(
                              flin.data(), traw, got, tw, th, r.depth,
                              r.tile_mode, bpt);
                      } else if (r.in_mip_tail) {
                          flin.zero_all();
                          prosper::gpu::detile_surface_level(
                              flin.data(), traw, got, tw, th, r.tile_mode,
                              bpt, r.mip_tail_x, r.mip_tail_y);
                      } else {
                          if (!prosper::gpu::detile_writes_whole_destination(
                                  r.tile_mode, bpt))
                              flin.zero_all();
                          prosper::gpu::detile_surface(
                              flin.data(), traw, tw, th, r.tile_mode, 0, bpt);
                      }
                  } else if (linear_padded_read) {
                      flin.zero_all();
                      copy_linear_padded_rows(flin.data(), (size_t)tw * bpt, th);
                  } else if (!snapshot_source) {
                      flin.zero_tail(copy_resource(
                          flin.data(), sampled_source_addr, flin.size()));
                  }
                  // Was a per-texel/per-channel scalar loop around float_to_half with a
                  // 4-byte and a 2-byte memcpy each -- 8.3 M conversions and 16.6 M small
                  // copies for one 1920x1080 reference, single-threaded, next to an
                  // already-vectorised sibling. The helper below IS that sibling, widened
                  // to the narrower component counts; it keeps the same (0,0,0,1) fill and
                  // the same bit-for-bit rounding/NaN-payload contract.
                  prosper::frontend::pack_float32_to_rgba16f_range(
                      snapshot_source ? persistent_validation_scratch.data() : flin.data(),
                      nc, bpt, volume_texels, texture_pixels.data());
                  f32_done = true;
              } else if (f16) {
                  // fp16 texture (#290 wall 1): read at the REAL bytes-per-texel and detile
                  // with the REAL element size — the old clamp read an 8-B/texel Float16x4
                  // surface at 4 B AND detiled it with bpe=4 against 8-B tiled elements, so
                  // the result was doubly wrong ("confetti" regions, e.g. DOLL's 960x540 /
                  // 480x270 bloom-chain buffers). Guest memory still converts half->UNORM8 on
                  // upload. That path clamps values above 1.0 and remains separate from native
                  // tiled guest-texture upload and the renderer-owned RTT fix in #773.
                  // Missing components read (0,0,0,1) per the hardware rule; the T# DST_SEL
                  // swizzle still applies. CONFIDENCE: MED — the half decode is exact and
                  // unit-tested, but the [0,1] clamp loses >1.0 bloom energy for guest-backed
                  // textures; renderer-owned RTTs retain RGBA16F through the #773 path.
                  const uint32_t nc = bpt / 2;                    // fp16 components per texel
                  // Pooled, so a 4K HDR intermediate does not mmap/fault/munmap 63 MiB per
                  // reference. It arrives holding the PREVIOUS surface, so every arm below
                  // either covers it completely or zeroes what it does not fill.
                  prosper::frontend::DecodeScratchPool::Lease hlin;
                  bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  const bool snapshot_source = stage_linear_decoder_source(
                      hlin, volume_texels * bpt, tiled);
                  if (tiled) {
                      size_t tbytes = is_volume
                          ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, r.tile_mode, bpt)
                          : prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                      prosper::frontend::DecodeScratchPool::Lease traw_lease;
                      size_t got = 0;
                      const uint8_t* traw = stage_tiled_source(
                          traw_lease, sampled_source_addr, tbytes, got);
                      if (got < hlin.size())
                          hlin.zero_tail(copy_resource(hlin.data(), sampled_source_addr,
                                                       hlin.size()));  // short backing -> linear fallback
                      else if (is_volume) { hlin.zero_all(); prosper::gpu::detile_volume(
                          hlin.data(), traw, got, tw, th, r.depth, r.tile_mode, bpt); }
                      else if (r.in_mip_tail) { hlin.zero_all(); prosper::gpu::detile_surface_level(
                          hlin.data(), traw, got, tw, th, r.tile_mode, bpt,
                          r.mip_tail_x, r.mip_tail_y); }
                      else {
                          // The only arm on Stray's hot path, and the only one whose
                          // destination coverage is asserted (test_tile). Everything else
                          // pre-zeroes, because a detiler that bounds its writes by a
                          // SHORT source would otherwise leave the previous tenant visible.
                          if (!prosper::gpu::detile_writes_whole_destination(
                                  r.tile_mode, bpt))
                              hlin.zero_all();
                          prosper::gpu::detile_surface(
                              hlin.data(), traw, tw, th, r.tile_mode, 0, bpt);
                      }
                  } else if (linear_padded_read) {
                      hlin.zero_all();
                      copy_linear_padded_rows(hlin.data(), (size_t)tw * bpt, th);
                  } else if (!snapshot_source) {
                      hlin.zero_tail(copy_resource(hlin.data(), sampled_source_addr,
                                                   hlin.size()));
                  }
                  // Same NaN/negative/positive-infinity clamp and absent-channel defaults
                  // as the historical scalar loop, exhaustively checked over all binary16
                  // inputs by test_game_compute. Large dynamic textures convert in parallel.
                  sampled_float16_to_unorm8_range(
                      snapshot_source ? persistent_validation_scratch.data() : hlin.data(),
                      nc, volume_texels, texture_pixels.data());
                  f16_done = true;   // read+detiled at the real element size already
              } else if (native_r8_sampled) {
                  // Tight linear R8 needs neither detiling nor per-texel expansion. One guarded
                  // copy feeds the backend's 1-byte staging upload; the view swizzle below makes
                  // every sampled component equal R, byte-for-byte matching the old RGBA result.
                  linear_source_prefix_size = copy_resource(
                      texture_pixels.data(), sampled_source_addr,
                      texture_pixels.size());
                  if (linear_source_prefix_size < texture_pixels.size())
                      std::fill(texture_pixels.begin() + linear_source_prefix_size,
                                texture_pixels.end(), 0);
                  narrow_decode_done = true;
                  narrow_done = true;
              } else if (native_rg8_sampled) {
                  // RG8 is already the backend's two-byte sampled representation. Strip
                  // physical row padding when present; a contiguous copy would advance the
                  // next visible row into padding and produce a diagonal chroma weave.
                  if (linear_padded_read) {
                      linear_source_prefix_size = copy_linear_padded_rows(
                          texture_pixels.data(), linear_dst_row, th);
                  } else {
                      linear_source_prefix_size = copy_resource(
                          texture_pixels.data(), sampled_source_addr,
                          texture_pixels.size());
                      if (linear_source_prefix_size < texture_pixels.size())
                          std::fill(texture_pixels.begin() + linear_source_prefix_size,
                                    texture_pixels.end(), 0);
                  }
                  narrow_decode_done = true;
                  narrow_done = true;
              } else if (bpt < 4) {
                  // Narrow (single/dual-channel) surface: read at the REAL element size and detile
                  // with the matching bpe geometry (1 B -> 64x64, 2 B -> 64x32 micro-tiles, #119),
                  // then expand to RGBA8. Legacy 8-bit coverage resources keep the grayscale
                  // broadcast so shaders can read either .r or .a, while the exact AvPlayer NV12
                  // contract preserves both bytes of its RG8 interleaved U/V plane. A UNORM16
                  // resource instead receives the format-defined missing channels (R,0,0,1), after
                  // which the real T# DST_SEL is applied below. Its R component must be normalized
                  // from both bytes; selecting byte zero makes a smooth ramp a sawtooth (#1186).
                  prosper::frontend::DecodeScratchPool::Lease nlin;
                  bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  const bool snapshot_source = stage_linear_decoder_source(
                      nlin, volume_texels * bpt, tiled);
                  if (tiled) {
                      size_t tbytes = is_volume
                          ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, r.tile_mode, bpt)
                          : prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                      prosper::frontend::DecodeScratchPool::Lease traw_lease;
                      size_t got = 0;
                      const uint8_t* traw = stage_tiled_source(
                          traw_lease, sampled_source_addr, tbytes, got);
                      // PROSPER_DUMP_RAWTILE (narrow path): the single/dual-channel RAW TILED bytes,
                      // once per address, for offline 8-bpp de-swizzle sweeps (the SDF font atlas).
                      if (PROSPER_ENV_ON("PROSPER_DUMP_RAWTILE") && got >= nlin.size() && tw <= 2048 && th <= 1024) {
                          static std::set<uint64_t> nseen;
                          if (nseen.insert(r.gpu_addr).second) {
                              std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                              char bn[512]; snprintf(bn, sizeof bn, "%s/narrowtile_%ux%u_b%u_%llx.bin",
                                                     dd.c_str(), tw, th, bpt, (unsigned long long)r.gpu_addr);
                              if (FILE* bf = fopen(bn, "wb")) { fwrite(traw, 1, tbytes, bf); fclose(bf);
                                  fprintf(stderr, "[render] narrow raw tiled -> %s (%zu, bpt=%u)\n", bn, tbytes, bpt); fflush(stderr); }
                          }
                      }
                      if (got < nlin.size())
                          nlin.zero_tail(copy_resource(nlin.data(), sampled_source_addr,
                                                       nlin.size()));  // short backing -> linear fallback
                      else if (is_volume) { nlin.zero_all(); prosper::gpu::detile_volume(
                          nlin.data(), traw, got, tw, th, r.depth, r.tile_mode, bpt); }
                      else if (r.in_mip_tail) { nlin.zero_all(); prosper::gpu::detile_surface_level(
                          nlin.data(), traw, got, tw, th, r.tile_mode, bpt,
                          r.mip_tail_x, r.mip_tail_y); }
                      else {
                          if (!prosper::gpu::detile_writes_whole_destination(
                                  r.tile_mode, bpt))
                              nlin.zero_all();
                          prosper::gpu::detile_surface(
                              nlin.data(), traw, tw, th, r.tile_mode, 0, bpt);
                      }
                  } else if (linear_padded_read) {
                      nlin.zero_all();
                      copy_linear_padded_rows(nlin.data(), (size_t)tw * bpt, th);
                  } else if (!snapshot_source) {
                      nlin.zero_tail(copy_resource(nlin.data(), sampled_source_addr,
                                                   nlin.size()));
                  }
                  const uint8_t* narrow_source = snapshot_source
                      ? persistent_validation_scratch.data() : nlin.data();
                  const bool unorm16 = r.format == prosper::gpu::DataFormat::Unorm16;
                  for (size_t t = 0; t < volume_texels; t++) {
                      uint8_t* p = &texture_pixels[t * 4];
                      if (avplayer_chroma_layout) {
                          const uint8_t* source = &narrow_source[t * bpt];
                          p[0] = source[0];
                          p[1] = source[1];
                          p[2] = r.num_components > 2 ? source[2] : 0;
                          p[3] = 255;
                      } else {
                          uint8_t v = narrow_source[t * bpt]; // first (coverage) channel
                          if (unorm16) {
                              uint16_t raw;
                              std::memcpy(&raw, &narrow_source[t * bpt], sizeof(raw));
                              v = prosper::gpu::unorm16_to_unorm8(raw);
                              p[0] = v; p[1] = p[2] = 0; p[3] = 255;
                          } else {
                              p[0] = p[1] = p[2] = p[3] = v;
                          }
                      }
                  }
                  narrow_decode_done = true;   // skip the generic 32-bpp detiler below
                  narrow_done = !unorm16;      // coverage broadcast replaces swizzle; R16 does not
              } else if (linear_padded_read) {
                  // Pitch-padded linear surface (see linear_padded_read above): read row-by-row at
                  // the 256-byte-aligned source pitch into the tight destination, dropping the
                  // per-row padding. copy_resource is fault-safe when an old capture has a short
                  // backing because its pre-v28 writer omitted the final padded rows.
                  linear_source_prefix_size = copy_linear_padded_rows(
                      texture_pixels.data(), linear_dst_row, th);
              } else {
                  // Ordinary RGBA8 detiling reads its own padded source below and writes
                  // every output byte. Preserve the raw copy for inspectors and for shapes
                  // with partial-write/conversion semantics; short backing still takes the
                  // historical recovery copy before detiling.
                  generic_source_copy_deferred = prune_unused_texture_preparation &&
                      !rtt_hit && !cube_done && !array_done && !dcc_fast_clear_done &&
                      !fr.gpu_detile && !bcb && !narrow_decode_done &&
                      !f16_done && !f32_done && !portable_raw_uvec4_storage &&
                      output_bpp == 4u &&
                      r.format == prosper::gpu::DataFormat::Unorm8 &&
                      r.num_components == 4u && !is_cube && !is_array && !is_volume &&
                      !r.in_mip_tail && prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                      prosper::gpu::detile_writes_whole_destination(r.tile_mode, 4u) &&
                      !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                      !PROSPER_ENV_ON("PROSPER_DUMP_RAWTEX") && !getenv("PROSPER_GFXLOG");
                  if (generic_source_copy_deferred) {
                      ++g_texture_decode_scope.generic_source_copy_deferrals;
                  } else {
                      const size_t got = copy_resource(
                          texture_pixels.data(), sampled_source_addr, nb);
                      g_texture_decode_scope.generic_source_copied_bytes += got;
                      linear_source_prefix_size = got;
                      if (got < nb)
                          std::fill(texture_pixels.begin() + got, texture_pixels.end(), 0);
                  }
              }
              // PROSPER_DUMP_RAWTEX: write the raw tiled RGBA bytes (pre-detile) to a .bin for
              // offline swizzle experimentation.
              if (PROSPER_ENV_ON("PROSPER_DUMP_RAWTEX") && !texture_pixels.empty()) {
                  std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                  char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_%ux%u.bin", d.c_str(), tw, th);
                  if (FILE* f = fopen(fn, "wb")) { fwrite(texture_pixels.data(), 1, nb, f); fclose(f);
                      fprintf(stderr, "[render] dumped raw tiled bytes -> %s (%zu)\n", fn, nb); fflush(stderr); }
              }
              if (getenv("PROSPER_GFXLOG")) { const uint8_t* b = texture_pixels.data();
                  size_t nz = 0; for (size_t i = 0; i < nb && i < (1u<<16); i++) nz += (b[i] != 0);
                  fprintf(stderr, "[render] tex binding=%u %ux%u first64k-nonzero=%zu\n", r.binding, tw, th, nz); }
              // AUTO-DETILE: de-swizzle a GPU-tiled sampled surface into the linear texstore,
              // driven by the T# tile_mode threaded through the resource table (r.tile_mode).
              bool auto_tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
              const char* dt = PROSPER_ENV_VALUE("PROSPER_DETILE");
              // BC textures already block-detiled + decoded above; the 32-bpp detiler must not touch them.
              if (!rtt_hit && !cube_done && !array_done && !dcc_fast_clear_done && !fr.gpu_detile && !bcb &&
                  !narrow_decode_done &&
                  !f16_done && !f32_done && !portable_raw_uvec4_storage &&
                  !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                  (auto_tiled || (!is_volume && dt && atoi(dt) != 0))) {
                  const uint32_t tmode = auto_tiled ? r.tile_mode : (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                  const uint32_t pitch = PROSPER_ENV_VALUE("PROSPER_PITCH") ? (uint32_t)atoi(PROSPER_ENV_VALUE("PROSPER_PITCH")) : 0;
                  size_t tiled_bytes = is_volume
                      ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, tmode, 4)
                      : prosper::gpu::tiled_surface_bytes(tw, th, tmode, pitch);
                  // The `got < nb` recovery below WRITES into this buffer, so the source
                  // may only be read in place when it cannot be reached -- i.e. when the
                  // padded tiled extent already covers the linear one.
                  prosper::frontend::DecodeScratchPool::Lease tiled_lease;
                  size_t got = 0;
                  const uint8_t* tiled = nullptr;
                  if (tiled_bytes >= nb) {
                      tiled = stage_tiled_source(
                          tiled_lease, sampled_source_addr, tiled_bytes, got);
                  } else {
                      tiled_lease = prosper::frontend::decode_scratch_pool().take(tiled_bytes);
                      got = copy_resource(
                          tiled_lease.data(), sampled_source_addr, tiled_bytes);
                      tiled_lease.zero_tail(got);
                      tiled = tiled_lease.data();
                  }
                  if (got < nb) {
                      if (generic_source_copy_deferred) {
                          linear_source_prefix_size = copy_resource(
                              texture_pixels.data(), sampled_source_addr, nb);
                          g_texture_decode_scope.generic_source_copied_bytes +=
                              linear_source_prefix_size;
                          std::fill(texture_pixels.begin() + linear_source_prefix_size,
                                    texture_pixels.end(), 0);
                      }
                      // Reachable only on the staged path -- the direct branch above
                      // requires tiled_bytes >= nb and then delivers all of them -- so the
                      // lease is already the buffer just filled. Re-staging if it somehow
                      // is not keeps this recovery writing to memory it owns.
                      if (tiled_lease.size() < tiled_bytes) {
                          tiled_lease = prosper::frontend::decode_scratch_pool().take(tiled_bytes);
                          got = copy_resource(
                              tiled_lease.data(), sampled_source_addr, tiled_bytes);
                          tiled_lease.zero_tail(got);
                      }
                      // The padded tiled buffer's tail (th rounded to whole 32-row tiles) runs past
                      // the real backing: fall back to the width*height linear bytes copied above
                      // rather than an all-zero buffer (which would BLANK the texture).
                      std::memcpy(tiled_lease.data(), texture_pixels.data(),
                                  std::min(nb, tiled_bytes));
                      tiled = tiled_lease.data();
                  }
                  // PROSPER_DUMP_RAWTILE: write the EXACT padded tiled bytes to a .bin (no lossy BMP
                  // round-trip) so the de-swizzle can be reversed offline against a known image (#101).
                  if (PROSPER_ENV_ON("PROSPER_DUMP_RAWTILE") && (frame_no < 200 || (tw <= 2048 && th <= 1024))) {
                      // Early frames by binding, PLUS small textures once per address (the font/UI
                      // atlas can be sampled late — capture whenever first seen).
                      static std::set<uint64_t> rawseen;
                      bool small = tw <= 2048 && th <= 1024;
                      if (!small || rawseen.insert(r.gpu_addr).second) {
                          std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                          char bn[512];
                          if (small) snprintf(bn, sizeof bn, "%s/rawtile_%ux%u_%llx.bin", dd.c_str(), tw, th, (unsigned long long)r.gpu_addr);
                          else snprintf(bn, sizeof bn, "%s/tiled_f%04d_b%u_%ux%u.bin", dd.c_str(), (int)frame_no, r.binding, tw, th);
                          if (FILE* bf = fopen(bn, "wb")) { fwrite(tiled, 1, tiled_bytes, bf); fclose(bf); }
                      }
                  }
                  if (is_volume) prosper::gpu::detile_volume(
                      texture_pixels.data(), tiled, got, tw, th, r.depth, tmode, 4);
                  else if (r.in_mip_tail) prosper::gpu::detile_surface_level(
                      texture_pixels.data(), tiled, got, tw, th, tmode, 4,
                      r.mip_tail_x, r.mip_tail_y);
                  else prosper::gpu::detile_surface(
                      texture_pixels.data(), tiled, tw, th, tmode, pitch);
              }
              // Legacy fallback for packed R11G11B10F shapes outside the exact native upload
              // contract above. The texel IS 4 bytes, so the generic read + auto-detile already
              // produced linear packed words; only unsupported shapes still narrow to RGBA8.
              if (!rtt_hit && !dcc_fast_clear_done && !portable_raw_uvec4_storage &&
                  !native_r11_sampled &&
                  r.format == prosper::gpu::DataFormat::Float10_11_11) {
                  uint8_t* tp = texture_pixels.data();
                  const size_t decoded_texels =
                      volume_texels * (cube_done ? 6u : (array_done ? decoded_layers : 1u));
                  for (size_t t = 0; t < decoded_texels; t++) {
                      uint32_t v; std::memcpy(&v, tp + t * 4, 4);
                      const float fc[3] = { prosper::gpu::f11_to_float((uint16_t)(v & 0x7FFu)),
                                            prosper::gpu::f11_to_float((uint16_t)((v >> 11) & 0x7FFu)),
                                            prosper::gpu::f10_to_float((uint16_t)((v >> 22) & 0x3FFu)) };
                      for (int c = 0; c < 3; c++)
                          tp[t * 4 + c] = (fc[c] != fc[c] || fc[c] <= 0.f) ? 0
                                        : (fc[c] >= 1.f ? 255 : (uint8_t)(fc[c] * 255.f + 0.5f));
                      tp[t * 4 + 3] = 255;
                  }
              }
              // Packed R10G10B10A2 UNORM (GFX10 IMG_FMT 50 / "2_10_10_10_UNORM"):
              // the generic 4-B read and detile above preserve packed texels; normalize each
              // field to the RGBA8 image format used by this renderer before sampling.
              if (!rtt_hit && !dcc_fast_clear_done && !portable_raw_uvec4_storage &&
                  r.format == prosper::gpu::DataFormat::Unorm2_10_10_10) {
                  uint8_t* tp = texture_pixels.data();
                  const size_t decoded_texels =
                      volume_texels * (cube_done ? 6u : (array_done ? decoded_layers : 1u));
                  for (size_t t = 0; t < decoded_texels; t++) {
                      uint32_t v; std::memcpy(&v, tp + t * 4, 4);
                      prosper::gpu::unorm2_10_10_10_to_rgba8(v, tp + t * 4);
                  }
              }
              // Focused per-consumer version probe (#586). Hash both the raw guest backing
              // and the final decoded/RTT-injected pixels so a live draw identifies whether
              // divergence precedes format conversion or enters through renderer-owned state.
              // Unset parses to 0x0; require the probe to have been requested (see the
              // matching guard on the pass hash below).
              if (resource_hash_w && resource_hash_h &&
                  resource_hash_w == tw && resource_hash_h == th) {
                  const size_t raw_size = std::min<size_t>(
                      r.size ? r.size : volume_texels * 4, 64u << 20);
                  std::vector<uint8_t> raw(raw_size, 0);
                  const size_t raw_got = copy_resource(
                      raw.data(), sampled_source_addr, raw.size());
                  auto fnv = [](const uint8_t* data, size_t size) {
                      uint64_t hash = 1469598103934665603ull;
                      for (size_t i = 0; i < size; ++i) {
                          hash ^= data[i];
                          hash *= 1099511628211ull;
                      }
                      return hash;
                  };
                  const uint64_t raw_hash = fnv(raw.data(), raw_got);
                  const uint64_t sample_hash = fnv(texture_pixels.data(), texture_pixels.size());
                  const std::vector<uint8_t> inspected = inspection_rgba8(
                      texture_pixels, tw, th, fr.texture_format);
                  size_t rgb_nonblack = 0, alpha_nonzero = 0;
                  for (size_t p = 0; p + 3 < inspected.size(); p += 4) {
                      rgb_nonblack += inspected[p] != 0 || inspected[p + 1] != 0 ||
                                      inspected[p + 2] != 0;
                      alpha_nonzero += inspected[p + 3] != 0;
                  }
                  const auto writer = prosper::gpu::last_guest_write_overlap(
                      sampled_source_addr, raw_size);
                  fprintf(stderr,
                          "[resource-version] render-submit=%llu draw=%llu order=%llu set=%u bind=%u "
                          "addr=0x%llx dims=%ux%u class=%u fmt=%u tile=%u dcc=%u meta=0x%llx rtt=%d "
                          "raw=%zu/%zu:%016llx sample=%zu:%016llx rgb_nonblack=%zu alpha_nonzero=%zu "
                          "writer=%s/%llu/%llu/%llu/0x%llx\n",
                          (unsigned long long)g_this_submit,
                          (unsigned long long)draw.draw_index,
                          (unsigned long long)draw.command_order,
                          set, r.binding, (unsigned long long)r.gpu_addr, tw, th,
                          (unsigned)r.cls, (unsigned)r.format, r.tile_mode,
                          r.compression_enabled, (unsigned long long)r.metadata_addr,
                          (int)rtt_hit,
                          raw_got, raw_size, (unsigned long long)raw_hash,
                          texture_pixels.size(), (unsigned long long)sample_hash,
                          rgb_nonblack, alpha_nonzero,
                          writer ? prosper::gpu::guest_writer_kind_name(writer->kind) : "none",
                          (unsigned long long)(writer ? writer->submit : 0),
                          (unsigned long long)(writer ? writer->item : 0),
                          (unsigned long long)(writer ? writer->order : 0),
                          (unsigned long long)(writer ? writer->identity : 0));
                  if (PROSPER_ENV_ON("PROSPER_DUMP_RESOURCE_VERSION")) {
                      static std::set<std::pair<uint64_t, uint64_t>> dumped_versions;
                      if (dumped_versions.emplace(r.gpu_addr, sample_hash).second) {
                          const char* dd = getenv("PROSPER_FRAME_DIR");
                          char fn[512];
                          snprintf(fn, sizeof fn, "%s/resource_%llx_%ux%u_%016llx.bmp",
                                   dd && *dd ? dd : ".", (unsigned long long)r.gpu_addr, tw, th,
                                   (unsigned long long)sample_hash);
                          prosper::test::dump_bmp(fn, inspected, tw, th);
                          fprintf(stderr, "[resource-version] dumped decoded sample -> %s\n", fn);
                      }
                  }
              }
              // PROSPER_PALETTELOG: compact identity/provenance trace for Unity's 256x16
              // palette textures. Unlike GFXLOG this is cheap enough for a focused render
              // window and reveals both descriptor-address and decoded-content changes.
              if (PROSPER_ENV_ON("PROSPER_PALETTELOG") && tw == 256 && th == 16 &&
                  fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                  uint64_t hash = 1469598103934665603ull;
                  size_t rgb_nonblack = 0;
                  for (size_t t = 0; t < (size_t)tw * th; ++t) {
                      const uint8_t* p = &texture_pixels[t * 4];
                      rgb_nonblack += (p[0] != 0 || p[1] != 0 || p[2] != 0);
                      for (unsigned c = 0; c < 4; ++c) {
                          hash ^= p[c]; hash *= 1099511628211ull;
                      }
                  }
                  fprintf(stderr, "[palette] binding=%u addr=0x%llx fnv=%016llx rgb_nonblack=%zu\n",
                          r.binding, (unsigned long long)r.gpu_addr,
                          (unsigned long long)hash, rgb_nonblack);
              }
              // Apply the synthetic texture after every decode/conversion step. Applying it before
              // auto-detile let the real tiled bytes overwrite the checker, producing a false-negative
              // sampling diagnosis (#522).
              const char* test_texture = PROSPER_ENV_VALUE("PROSPER_TESTTEX");
              const char* test_texture_binding = PROSPER_ENV_VALUE("PROSPER_TESTTEX_BINDING");
              const char* test_texture_draw = PROSPER_ENV_VALUE("PROSPER_TESTTEX_DRAW");
              const bool test_this_texture = test_texture &&
                  (!test_texture_binding ||
                   strtoul(test_texture_binding, nullptr, 0) == r.binding) &&
                  (!test_texture_draw ||
                   strtoull(test_texture_draw, nullptr, 0) == draw.draw_index);
              if (test_this_texture) {
                  const uint32_t slices = is_volume ? std::max(r.depth, 1u) : 1u;
                  if (!strcmp(test_texture, "zero")) {
                      std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                  } else if (fr.texture_format == VK_FORMAT_R16G16B16A16_SFLOAT) {
                      for (uint32_t z = 0; z < slices; ++z)
                          for (uint32_t y = 0; y < th; ++y)
                              for (uint32_t x = 0; x < tw; ++x) {
                                  uint16_t* p = reinterpret_cast<uint16_t*>(
                                      texture_pixels.data() +
                                      (((size_t)z * th + y) * tw + x) * 8);
                                  const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                  p[0] = prosper::gpu::float_to_half((float)x / tw);
                                  p[1] = prosper::gpu::float_to_half((float)y / th);
                                  p[2] = prosper::gpu::float_to_half(ck ? 0.8f : 0.16f);
                                  p[3] = prosper::gpu::float_to_half(1.0f);
                              }
                  } else if (fr.texture_format == VK_FORMAT_R8_UNORM) {
                      for (uint32_t z = 0; z < slices; ++z)
                          for (uint32_t y = 0; y < th; ++y)
                              for (uint32_t x = 0; x < tw; ++x) {
                                  const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                  texture_pixels[((size_t)z * th + y) * tw + x] =
                                      ck ? 200 : 40;
                              }
                  } else if (fr.texture_format == VK_FORMAT_R8G8_UNORM) {
                      for (uint32_t z = 0; z < slices; ++z)
                          for (uint32_t y = 0; y < th; ++y)
                              for (uint32_t x = 0; x < tw; ++x) {
                                  const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                  uint8_t* p = &texture_pixels[
                                      (((size_t)z * th + y) * tw + x) * 2];
                                  p[0] = ck ? 200 : 40;
                                  p[1] = ck ? 40 : 200;
                              }
                  } else if (fr.texture_format ==
                             VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                      for (uint32_t z = 0; z < slices; ++z)
                          for (uint32_t y = 0; y < th; ++y)
                              for (uint32_t x = 0; x < tw; ++x) {
                                  const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                  const float values[3] = {
                                      static_cast<float>(x) / tw,
                                      static_cast<float>(y) / th,
                                      ck ? 8.0f : 0.16f,
                                  };
                                  const uint32_t packed =
                                      static_cast<uint32_t>(
                                          prosper::gpu::float_to_f11(values[0])) |
                                      (static_cast<uint32_t>(
                                           prosper::gpu::float_to_f11(values[1])) << 11) |
                                      (static_cast<uint32_t>(
                                           prosper::gpu::float_to_f10(values[2])) << 22);
                                  std::memcpy(
                                      texture_pixels.data() +
                                          (((size_t)z * th + y) * tw + x) * 4,
                                      &packed, sizeof(packed));
                              }
                  } else {
                      for (uint32_t z = 0; z < slices; ++z)
                          for (uint32_t y = 0; y < th; ++y)
                              for (uint32_t x = 0; x < tw; ++x) {
                                  uint8_t* p = &texture_pixels[
                                      (((size_t)z * th + y) * tw + x) * 4];
                                  bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                  p[0] = (uint8_t)(255 * x / tw);
                                  p[1] = (uint8_t)(255 * y / th);
                                  p[2] = ck ? 200 : 40;
                                  p[3] = 255;
                              }
                  }
              }
              // This 256x16 sparse palette is addressed like 16 blue slices, each 16 texels wide.
              // The identity probe preserves source color through shaders using
              // u=r/17+b*15/16, v=1-g, distinguishing lookup contents from a broken
              // source/geometry/sample path (#522).
              if (PROSPER_ENV_ON("PROSPER_TESTLUT") && tw == 256 && th == 16 &&
                  fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                  for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                      uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                      p[0] = (uint8_t)((x % 16) * 255 / 15);
                      p[1] = (uint8_t)((15 - y) * 255 / 15);
                      p[2] = (uint8_t)((x / 16) * 255 / 15);
                      p[3] = 255;
                  }
              }
              // Unity's post-processing stack flattens a 32^3 grading LUT into a 1024x32
              // strip (32 red samples per blue slice). This isolates a missing LUT producer
              // from the persistent post shader and its healthy scene input (#522).
              if (PROSPER_ENV_ON("PROSPER_TESTLUT32") && tw == 1024 && th == 32 &&
                  fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                  for (uint32_t y = 0; y < th; ++y) for (uint32_t x = 0; x < tw; ++x) {
                      uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                      p[0] = (uint8_t)((x % 32) * 255 / 31);
                      p[1] = (uint8_t)(y * 255 / 31);
                      p[2] = (uint8_t)((x / 32) * 255 / 31);
                      p[3] = 255;
                  }
              }
              // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
              // bypassing the shader — reveals whether the render target is tiled or linear.
              if (PROSPER_ENV_ON("PROSPER_DUMP_TEX") && !texture_pixels.empty() && frame_no < 200 &&
                  fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                  std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                  char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                  prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                  fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
              }
              // PROSPER_DUMP_ATLAS: dump each SMALL sampled texture once per ADDRESS (find the caption
              // font among same-size UI textures). Capped.
              if (PROSPER_ENV_ON("PROSPER_DUMP_ATLAS") && !texture_pixels.empty() && tw <= 2048 && th <= 1024 &&
                  fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                  static std::unordered_map<uint64_t,int> seen; static int ndumped = 0;
                  if (seen[r.gpu_addr]++ == 0 && ndumped++ < 60) {
                      std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                      char fn[512]; snprintf(fn, sizeof fn, "%s/tex_%ux%u_%llx_c%u.bmp", d.c_str(), tw, th,
                                             (unsigned long long)r.gpu_addr, r.num_components);
                      prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                  }
              }
              // PROSPER_KILL_RING (#1186): null out the concentric-ring light-glow texture
              // (a 1024x1024 single-channel glow sprite) to A/B whether it is what draws the
              // see-through concentric-circles pattern over the world. Run with
              // PROSPER_NO_TEXTURE_DECODE_CACHE=1 so every sample takes this decode path.
              if (PROSPER_ENV_ON("PROSPER_KILL_RING") && tw == 1024 && th == 1024 &&
                  r.num_components == 1 && r.cls == RC::Texture)
                  std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
              if (!fr.tex_rgba && !fr.gpu_detile) {
                  fr.tex_rgba = texture_pixels.data();
                  decoded_pixels_in_texstore = true;
              }
              fr.tw = tw; fr.th = cube_done ? th * 6u : th;
              // #325: `sample_count` IS the backend's array-layer channel -- its own field
              // comment says so for MSAA-as-array -- and the view already selects
              // VK_IMAGE_VIEW_TYPE_2D_ARRAY whenever it exceeds 1. Set it only once the
              // slices are actually decoded: claiming layers we did not fill would upload
              // uninitialized memory, and the backend's span contract would reject it
              // anyway.
              if (array_done) {
                  fr.sample_count = decoded_layers;
                  fr.tex_byte_size = texture_pixels.size();
              }
              fr.td = is_volume ? r.depth : 1u;
              fr.img_dim = r.img_dim;
              // #1272: see the reuse path — plain 2D guest textures only.
              if (!is_volume && !cube_done)
                  fr.declared_mip_levels = r.declared_mip_levels;
              if (native_bc_mip_chain && !rtt_hit && fr.texture_format == native_bc_format)
                  fr.uploaded_mip_levels = native_bc_chain_levels;
              if (native_bc_sampled && !rtt_hit && fr.texture_format == native_bc_format)
                  fr.tex_byte_size = texture_pixels.size();
              // Failed/mismatched renderer readback must retry on the next use. Caching
              // its guest fallback under the renderer generation would freeze that fallback.
              const bool cube_cache_proven = !retained_depth_cube_cache_candidate ||
                  renderer_cube_snapshot_ready;
              if (!cube_cache_proven) ++g_texture_decode_scope.cube_snapshot_refusals;
              if (persistent_cache_eligible && !fr.gpu_detile && cube_cache_proven) {
                  DepthCubeSourceSnapshot cube_source_snapshot;
                  size_t source_prefix_size = linear_source_prefix_size;
                  if (!persistent_source_matches_pixels && persistent_source_size) {
                      if (renderer_cube_snapshot_ready) {
                          size_t read_bytes = 0;
                          source_prefix_size = capture_depth_cube_source(
                              depth_cube_source_layout, retained_depth_cube.present_mask,
                              PROSPER_ENV_ON("PROSPER_NO_CUBE_SOURCE_SNAPSHOT_PRUNING"),
                              persistent_validation_scratch, cube_source_snapshot,
                              copy_resource, read_bytes);
                          g_texture_decode_scope.late_snapshot_read_bytes += read_bytes;
                          g_texture_decode_scope.cube_snapshot_guest_bytes += source_prefix_size;
                          for (unsigned face = 0; face < 6; ++face)
                              if (retained_depth_cube.present_mask & (1u << face))
                                  g_texture_decode_scope.cube_snapshot_renderer_bytes +=
                                      depth_cube_source_layout.face_bytes;
                      } else if (decoder_source_snapshot_ready) {
                          source_prefix_size = decoder_source_prefix_size;
                      } else {
                          persistent_validation_scratch.resize(persistent_source_size);
                          source_prefix_size = copy_persistent_source(
                              persistent_validation_scratch.data(), persistent_source_size);
                          g_texture_decode_scope.late_snapshot_read_bytes += source_prefix_size;
                      }
                  }
                  auto old = persistent_decoded_textures.find(decode_key);
                  // PROSPER_NO_TEXTURE_PREFIX_INHERIT prevents reusing the outgoing
                  // allocation. Together with PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE it
                  // restores the `assign()` spelling: free the old buffer, allocate anew.
                  // It exists because
                  // WITHOUT it this optimisation has no off-switch, and a discriminator
                  // that cannot disable everything the change does is not a
                  // discriminator -- it silently under-reports the change it is
                  // attributing. Measured on Stray: an A/B armed with
                  // PROSPER_NO_DIRECT_TEXTURE_SOURCE and PROSPER_DECODE_SCRATCH_MB=0
                  // returned the texture leaf to 918.2 ms rather than the 1110.1 ms
                  // baseline, and the missing 191.9 ms was this, still switched on.
                  static const bool no_prefix_inherit =
                      PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_PREFIX_INHERIT") != nullptr;
                  // The entry this decode replaces holds a source_prefix buffer of the
                  // same shape. Taking its allocation instead of letting `assign` build a
                  // new one is worth naming on a surface the guest rewrites every frame:
                  // for Stray's 63.75 MiB HDR intermediate that pair -- free 63.75 MiB,
                  // then mmap and fault 63.75 MiB back in for the same bytes -- measured
                  // 13.00 ms per invalidation on this host, against 3.5 ms for the copy
                  // it exists to hold. The bytes stored are identical either way; only
                  // the mapping's lifetime changes.
                  std::vector<uint8_t> inherited_source_prefix;
                  uint32_t inherited_watch_dirty_count = 0;
                  uint32_t inherited_watch_stable_validations = 0;
                  bool inherited_watch_disabled = false;
                  prosper::host::GuestWriteWatch inherited_source_watch =
                      std::move(pending_source_watch);
                  uint64_t inherited_persistent_id = 0;
                  uint64_t inherited_persistent_version = 0;
                  if (old != persistent_decoded_textures.end() &&
                      old->second.source_addr == persistent_source_addr &&
                      old->second.source_size == persistent_source_size) {
                      inherited_watch_dirty_count = old->second.source_watch_dirty_count;
                      inherited_watch_stable_validations =
                          old->second.source_watch_stable_validations;
                      inherited_watch_disabled = old->second.source_watch_disabled;
                      if (!inherited_watch_disabled)
                          inherited_source_watch = std::move(old->second.source_watch);
                      inherited_persistent_id = old->second.persistent_id;
                      inherited_persistent_version = old->second.persistent_version;
                  }
                  // An identity already established earlier in this submit may be re-decoded
                  // if an interleaved guest GPU write mutated its backing. Only if it was used in
                  // the current submit (last_use == decode_generation) might earlier draws hold
                  // raw pointers to those bytes, so those allocations are retained in submit-scoped
                  // `retired_submit_pixels` until the submit scope completes (#3283).
                  // Retired bytes are tracked separately in `retired_submit_bytes` and deliberately
                  // held resident off-ledger for the submit's duration to ensure pointer safety.
                  if (old != persistent_decoded_textures.end()) {
                      if (old->second.pixels && old->second.last_use == decode_generation) {
                          retired_submit_bytes += old->second.bytes();
                          retired_submit_pixels.push_back(old->second.pixels);
                      }
                      persistent_decoded_texture_bytes -= old->second.bytes();
                      // Taken AFTER the ledger has been debited by the old entry's
                      // `bytes()`, which reads `source_prefix.size()`: moving first would
                      // debit zero and leak the entry's bytes from the budget forever.
                      if (!no_prefix_inherit)
                          inherited_source_prefix = std::move(old->second.source_prefix);
                      persistent_decoded_textures.erase(old);
                  }
                  const size_t required =
                      (persistent_source_matches_pixels ? 0 : source_prefix_size) +
                      texture_pixels.size();
                  while (required <= persistent_decode_limit &&
                         persistent_decoded_texture_bytes > persistent_decode_limit - required) {
                      auto victim = persistent_decoded_textures.end();
                      for (auto it = persistent_decoded_textures.begin();
                           it != persistent_decoded_textures.end(); ++it) {
                          if (it->second.last_use == decode_generation) continue;
                          if (victim == persistent_decoded_textures.end() ||
                              it->second.last_use < victim->second.last_use) victim = it;
                      }
                      if (victim == persistent_decoded_textures.end()) break;
                      // Victims are from prior submits (last_use < decode_generation) with no live
                      // references in the current submit, so erasing them releases their memory immediately.
                      persistent_decoded_texture_bytes -= victim->second.bytes();
                      persistent_decoded_textures.erase(victim);
                  }
                  if (required <= persistent_decode_limit &&
                      persistent_decoded_texture_bytes <= persistent_decode_limit - required) {
                      PersistentDecodedTexture cached;
                      cached.source_addr = persistent_source_addr;
                      cached.source_size = persistent_source_size;
                      cached.source_prefix_size = source_prefix_size;
                      cached.depth_cube_source = cube_source_snapshot;
                      cached.source_matches_pixels = persistent_source_matches_pixels;
                      if (!persistent_source_matches_pixels) {
                          static const bool transfer_source_snapshot =
                              PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE") == nullptr;
                          const auto handoff_start = timing_enabled
                              ? RenderClock::now() : RenderClock::time_point{};
                          auto snapshot = take_texture_source_snapshot(
                              persistent_validation_scratch, inherited_source_prefix,
                              source_prefix_size, transfer_source_snapshot);
                          const bool transferred = snapshot.transferred;
                          cached.source_prefix = std::move(snapshot.bytes);
                          if (transferred)
                              g_texture_decode_scope.source_snapshot_transferred_bytes += source_prefix_size;
                          else
                              g_texture_decode_scope.source_snapshot_copied_bytes += source_prefix_size;
                          if (timing_enabled) {
                              pending_timing.tex_source_snapshot_handoff_ms +=
                                  std::chrono::duration<double, std::milli>(
                                      RenderClock::now() - handoff_start).count();
                              if (transferred)
                                  pending_timing.tex_source_snapshot_transferred_bytes += source_prefix_size;
                              else
                                  pending_timing.tex_source_snapshot_copied_bytes += source_prefix_size;
                          }
                      }
                      cached.pixels = std::make_shared<const std::vector<uint8_t>>(
                          std::move(texture_pixels));
                      cached.output_height = fr.th;
                      cached.layers = fr.sample_count;
                      cached.narrow = narrow_done;
                      cached.last_use = decode_generation;
                      cached.persistent_id = inherited_persistent_id;
                      if (!cached.persistent_id) {
                          cached.persistent_id = ++persistent_texture_id;
                          if (!cached.persistent_id)
                              cached.persistent_id = ++persistent_texture_id;
                      }
                      cached.persistent_version = inherited_persistent_version + 1;
                      if (!cached.persistent_version) cached.persistent_version = 1;
                      cached.texture_format = fr.texture_format;
                      cached.packed_mip_levels = fr.uploaded_mip_levels;
                      cached.storage_image_contract_valid =
                          fr.storage_image_contract_valid;
                      cached.validation_snapshot =
                          prosper::gpu::guest_gpu_write_snapshot();
                      cached.source_watch_dirty_count = inherited_watch_dirty_count;
                      cached.source_watch_stable_validations =
                          inherited_watch_stable_validations;
                      cached.source_watch_disabled = inherited_watch_disabled;
                      cached.source_watch = std::move(inherited_source_watch);
                      auto [inserted, ok] = persistent_decoded_textures.emplace(
                          decode_key, std::move(cached));
                      if (ok)
                          persistent_decoded_texture_bytes += inserted->second.bytes();
                      fr.tex_rgba = inserted->second.pixels ? inserted->second.pixels->data() : nullptr;
                      fr.tex_rgba_owner = inserted->second.pixels;
                      fr.persistent_texture_id = inserted->second.persistent_id;
                      fr.persistent_texture_version = inserted->second.persistent_version;
                      decoded_pixels_in_texstore = false;
                  }
              }
              if (cube_cache_proven && !resource_rtt_hit && !resource_compute_image_hit && !has_live_rtt) {
                  const bool pin_scratch =
                      decoded_pixels_in_texstore && cross_span_source_size != 0;
                  const size_t pinned_slot = pin_scratch ? texture_slot : SIZE_MAX;
                  if (pin_scratch) {
                      texstore_pinned[texture_slot] = true;
                      ++g_texture_decode_scope.scratch_pins;
                  }
                  decoded_textures.insert_or_assign(
                      decode_key, DecodedTexture{fr.tex_rgba, fr.th, narrow_done,
                                                fr.persistent_texture_id,
                                                fr.persistent_texture_version,
                                                decode_span_ordinal,
                                                prosper::gpu::guest_gpu_write_snapshot(),
                                                persistent_source_addr,
                                                cross_span_source_size, pinned_slot,
                                                fr.tex_rgba_owner, fr.texture_format,
                                                fr.storage_image_contract_valid,
                                                fr.sample_count, fr.tex_byte_size, fr.gpu_detile,
                                                fr.uploaded_mip_levels});
              }
              }
              if (native_r32ui_storage && writable_storage_image) {
                  const uint32_t writeback_pitch = PROSPER_ENV_VALUE("PROSPER_PITCH")
                      ? static_cast<uint32_t>(atoi(getenv("PROSPER_PITCH"))) : 0u;
                  const size_t linear_bytes = static_cast<size_t>(tw) * th * 4u;
                  const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                  const size_t guest_bytes = r.in_mip_tail
                      ? r.mip_tail_bytes
                      : (tiled
                             ? prosper::gpu::tiled_surface_bytes(
                                   tw, th, r.tile_mode, writeback_pitch, 4u)
                             : linear_bytes);
                  // The exact Astro atomic surface is a base-level 2D R32_UINT image. Keep
                  // the callback fail-closed for malformed/short replay backing; losing a
                  // write is preferable to overrunning an unrelated guest allocation.
                  const bool backing_fits = guest_bytes &&
                      (!r.host_data || guest_bytes <= r.host_data_size);
                  if (backing_fits) {
                      const uint64_t guest_addr = r.gpu_addr;
                      uint8_t* const replay_data = r.host_data;
                      const uint32_t tile_mode = r.tile_mode;
                      const bool in_mip_tail = r.in_mip_tail;
                      const uint32_t mip_tail_x = r.mip_tail_x;
                      const uint32_t mip_tail_y = r.mip_tail_y;
                      fr.storage_image_writeback =
                          // `decoded_textures` and `texstore_pinned` are thread-local
                          // statics since #1691 and are deliberately NOT captured: a
                          // variable with static storage duration cannot appear in a
                          // capture list (clang rejects it outright), and the body reaches
                          // the render thread's own instances directly — which is the
                          // right identity, since this callback runs on that thread.
                          [guest_addr, replay_data, guest_bytes, linear_bytes,
                           tw, th, tile_mode,
                           writeback_pitch, in_mip_tail, mip_tail_x,
                           mip_tail_y](const uint8_t* pixels, size_t bytes) {
                              if (!pixels || bytes != linear_bytes) return;
                              uint8_t* destination = replay_data;
                              if (!destination) {
                                  if (guest_bytes > UINT32_MAX ||
                                      !prosper::gpu::guest_writable(
                                          guest_addr,
                                          static_cast<uint32_t>(guest_bytes)))
                                      return;
                                  prosper::host::guest_write_watch_notify_host_write(
                                      guest_addr, guest_bytes);
                                  destination = reinterpret_cast<uint8_t*>(
                                      static_cast<uintptr_t>(guest_addr));
                              }
                              if (in_mip_tail) {
                                  prosper::gpu::tile_surface_level(
                                      destination, guest_bytes, pixels, tw, th,
                                      tile_mode, 4u, mip_tail_x, mip_tail_y);
                              } else {
                                  prosper::gpu::tile_surface(
                                      destination, pixels, tw, th, tile_mode,
                                      writeback_pitch, 4u);
                              }
                              if (guest_addr) {
                                  // Named for the same reason as the compute writebacks:
                                  // a renderer writeback into guest memory is prosper's
                                  // own store, and reads as an anonymous `gpu` write in
                                  // every cache-invalidation diagnostic downstream.
                                  prosper::gpu::set_guest_gpu_write_origin(
                                      "renderer-writeback(rtt-guest-bytes)");
                                  prosper::gpu::notify_guest_gpu_write(
                                      guest_addr, guest_bytes);
                                  prosper::gpu::set_guest_gpu_write_origin(nullptr);
                              }
                              // The same guest range can already have a sampled-image decode
                              // under a different cache key/class. Its pixel representation
                              // cannot be patched byte-for-byte from R32_UINT, so discard every
                              // overlapping submit-local decode and let the next pass rebuild it
                              // from the now-current guest bytes.
                              if (guest_addr) {
                                  for (auto it = decoded_textures.begin();
                                       it != decoded_textures.end();) {
                                      const uint64_t cached_addr = it->first.gpu_addr;
                                      const uint64_t cached_bytes = std::max<uint64_t>(
                                          std::max<uint64_t>(it->first.size,
                                                             it->first.source_span_bytes),
                                          1u);
                                      const bool overlaps = cached_addr <= guest_addr
                                          ? guest_addr - cached_addr < cached_bytes
                                          : cached_addr - guest_addr < guest_bytes;
                                      if (overlaps) {
                                          if (it->second.texstore_slot <
                                              texstore_pinned.size())
                                              texstore_pinned[
                                                  it->second.texstore_slot] = false;
                                          it = decoded_textures.erase(it);
                                      } else {
                                          ++it;
                                      }
                                  }
                              }
                          };
                  }
              }
              if (portable_raw_uvec4_storage && writable_storage_image &&
                  fr.storage_image_contract_valid) {
                  const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                      !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                  const uint32_t writeback_tile_mode = tiled ? r.tile_mode : 0u;
                  const uint32_t writeback_depth = is_volume ? r.depth : 1u;
                  const uint32_t components =
                      r.num_components ? r.num_components : 1u;
                  const size_t guest_bytes = storage_image_raw_uvec4_source_bytes(
                      r.format, components, tw, th, writeback_depth,
                      writeback_tile_mode, r.in_mip_tail, r.mip_tail_bytes);
                  const size_t texels = static_cast<size_t>(tw) * th * writeback_depth;
                  const size_t expanded_bytes = texels <= SIZE_MAX / 16u
                      ? texels * 16u : 0u;
                  const bool backing_fits = guest_bytes && expanded_bytes &&
                      (!r.host_data || guest_bytes <= r.host_data_size);
                  if (!backing_fits) {
                      fr.storage_image_contract_valid = false;
                  } else {
                      const uint64_t guest_addr = r.gpu_addr;
                      uint8_t* const replay_data = r.host_data;
                      const auto format = r.format;
                      const bool in_mip_tail = r.in_mip_tail;
                      const uint32_t mip_tail_bytes = r.mip_tail_bytes;
                      const uint32_t mip_tail_x = r.mip_tail_x;
                      const uint32_t mip_tail_y = r.mip_tail_y;
                      fr.storage_image_writeback =
                          [guest_addr, replay_data, guest_bytes, expanded_bytes,
                           format, components, tw, th, writeback_depth,
                           writeback_tile_mode, in_mip_tail, mip_tail_bytes,
                           mip_tail_x, mip_tail_y](const uint8_t* pixels, size_t bytes) {
                              if (!pixels || bytes != expanded_bytes) return;
                              uint8_t* destination = replay_data;
                              if (!destination) {
                                  if (guest_bytes > UINT32_MAX ||
                                      !prosper::gpu::guest_writable(
                                          guest_addr,
                                          static_cast<uint32_t>(guest_bytes)))
                                      return;
                                  prosper::host::guest_write_watch_notify_host_write(
                                      guest_addr, guest_bytes);
                                  destination = reinterpret_cast<uint8_t*>(
                                      static_cast<uintptr_t>(guest_addr));
                              }
                              if (!storage_image_writeback_raw_uvec4(
                                      reinterpret_cast<const uint32_t*>(pixels),
                                      bytes / sizeof(uint32_t), format, components,
                                      tw, th, writeback_depth, writeback_tile_mode,
                                      in_mip_tail, mip_tail_bytes,
                                      mip_tail_x, mip_tail_y,
                                      destination, guest_bytes))
                                  return;
                              if (guest_addr) {
                                  prosper::gpu::set_guest_gpu_write_origin(
                                      "renderer-writeback(portable-uvec4)");
                                  prosper::gpu::notify_guest_gpu_write(
                                      guest_addr, guest_bytes);
                                  prosper::gpu::set_guest_gpu_write_origin(nullptr);
                                  for (auto it = decoded_textures.begin();
                                       it != decoded_textures.end();) {
                                      const uint64_t cached_addr = it->first.gpu_addr;
                                      const uint64_t cached_bytes = std::max<uint64_t>(
                                          std::max<uint64_t>(it->first.size,
                                                             it->first.source_span_bytes),
                                          1u);
                                      const bool overlaps = cached_addr <= guest_addr
                                          ? guest_addr - cached_addr < cached_bytes
                                          : cached_addr - guest_addr < guest_bytes;
                                      if (overlaps) {
                                          if (it->second.texstore_slot <
                                              texstore_pinned.size())
                                              texstore_pinned[
                                                  it->second.texstore_slot] = false;
                                          it = decoded_textures.erase(it);
                                      } else {
                                          ++it;
                                      }
                                  }
                              }
                          };
                  }
              }
              // Carry the decoded S# sampler state (filter/wrap/mip) so the pipeline samples
              // the way the game asked instead of using fixed LINEAR/clamp state.
              fr.mag_filter = r.mag_filter; fr.min_filter = r.min_filter; fr.mip_filter = r.mip_filter;
              // Draw/binding-scoped sampler A/B. This shares TESTTEX's selectors but leaves
              // the sampled pixels intact, isolating descriptor filtering from texture content.
              const char* test_filter = PROSPER_ENV_VALUE("PROSPER_TESTTEX_FILTER");
              const char* test_filter_binding = PROSPER_ENV_VALUE("PROSPER_TESTTEX_BINDING");
              const char* test_filter_draw = PROSPER_ENV_VALUE("PROSPER_TESTTEX_DRAW");
              const bool filter_valid = test_filter &&
                  (!strcmp(test_filter, "linear") || !strcmp(test_filter, "point"));
              if (filter_valid &&
                  (!test_filter_binding ||
                   strtoul(test_filter_binding, nullptr, 0) == r.binding) &&
                  (!test_filter_draw ||
                   strtoull(test_filter_draw, nullptr, 0) == draw.draw_index)) {
                  const uint32_t filter = !strcmp(test_filter, "linear") ? 1u : 0u;
                  fr.mag_filter = fr.min_filter = filter;
              }
              fr.addr_uvw[0] = r.addr_uvw[0]; fr.addr_uvw[1] = r.addr_uvw[1]; fr.addr_uvw[2] = r.addr_uvw[2];
              // Remaining S# sampler fields (#262): border color + LOD clamp/bias (applied where
              // valid; compare/unnorm stay on ShaderResource for shader-side lowering).
              fr.border_color_type = r.border_color_type;
              fr.min_lod = r.min_lod; fr.max_lod = r.max_lod; fr.lod_bias = r.lod_bias;
              // Anisotropy ratio (#275): applied in render_runner.h when the device supports the
              // samplerAnisotropy feature and filtering is linear. The Messenger decodes ratio 0
              // (isotropic) so this is a no-op for it; carries correct behavior for titles that
              // request anisotropic filtering.
              fr.max_aniso_ratio = r.max_aniso_ratio;
              // T# DST_SEL channel remap (#261): narrow coverage paths already broadcast to
              // every channel, so keep them identity. AvPlayer RG8 preserved U/V above and
              // still needs its T# mapping (R,G,0,1).
              if (native_r8_sampled) {
                  fr.swizzle[0]=4; fr.swizzle[1]=4;
                  fr.swizzle[2]=4; fr.swizzle[3]=4;
              }
              else if (narrow_done && !avplayer_chroma_layout) {
                  fr.swizzle[0]=4; fr.swizzle[1]=5; fr.swizzle[2]=6; fr.swizzle[3]=7;
              }
              else { for (int k=0;k<4;k++) fr.swizzle[k] = r.swizzle[k]; }
              if (resource_rtt_hit && live_rtt != g_rtt.end())
                  fr.render_target_guest_format = live_rtt->second.guest_format;
              // PROSPER_ALPHA1: force the sampled alpha to constant 1 (opaque). Diagnostic for a
              // black scene whose textures decode to real RGB but composite to nothing — if the
              // level appears with this, the alpha channel (decode or DST_SEL swizzle) is the bug (#300).
              if (PROSPER_ENV_ON("PROSPER_ALPHA1")) fr.swizzle[3] = 1;
              }
              return {ImageDisposition::Keep};
}

} // namespace prosper::frontend::submit_renderer
