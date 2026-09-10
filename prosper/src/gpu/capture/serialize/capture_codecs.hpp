#pragma once

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "gpu/capture/gpu_capture.hpp"

#include <mutex>
#include "hle/fs/save_paths.hpp"   // the effective per-title /savedata0 dir (#2734)
#include "gpu/texture/bc_decode.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "build_revision.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"
#include "gpu/present/videoout_present.hpp"

#include <algorithm>
#include <atomic>
#include <string>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/uio.h>
#include <unistd.h>
#include "host/platform/posix_shim.hpp"
#endif
#include "gpu/capture/gpu_capture_internal.hpp"

namespace prosper::gpu {

struct Writer {
    std::vector<uint8_t> data;
    void raw(const void* p, size_t n) {
        if (!n) return;
        const auto* b = static_cast<const uint8_t*>(p);
        data.insert(data.end(), b, b + n);
    }
    void u8(uint8_t v) { data.push_back(v); }
    void u32(uint32_t v) { for (unsigned i = 0; i < 4; ++i) data.push_back(uint8_t(v >> (8 * i))); }
    void u64(uint64_t v) { for (unsigned i = 0; i < 8; ++i) data.push_back(uint8_t(v >> (8 * i))); }
    void f32(float v) { u32(std::bit_cast<uint32_t>(v)); }
    void string(const std::string& s) { u32(static_cast<uint32_t>(s.size())); raw(s.data(), s.size()); }
    void words(const std::vector<uint32_t>& v) { u32(static_cast<uint32_t>(v.size())); for (auto x : v) u32(x); }
    void bytes(const std::vector<uint8_t>& v) { u64(v.size()); raw(v.data(), v.size()); }
};

struct Reader {
    const uint8_t* p = nullptr;
    size_t left = 0;
    std::string* error = nullptr;
    bool take(void* dst, size_t n) {
        if (n > left) { if (error && error->empty()) *error = "truncated capture"; return false; }
        if (!n) return true;
        std::memcpy(dst, p, n); p += n; left -= n; return true;
    }
    bool u8(uint8_t& v) { return take(&v, 1); }
    bool u32(uint32_t& v) {
        uint8_t b[4]; if (!take(b, 4)) return false;
        v = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        return true;
    }
    bool u64(uint64_t& v) {
        uint8_t b[8]; if (!take(b, 8)) return false; v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= uint64_t(b[i]) << (8 * i);
        return true;
    }
    bool f32(float& v) { uint32_t x; if (!u32(x)) return false; v = std::bit_cast<float>(x); return true; }
    bool string(std::string& s) {
        uint32_t n; if (!u32(n)) return false;
        if (n > kMaxStringBytes || n > left) { if (error) *error = "invalid string length"; return false; }
        s.assign(reinterpret_cast<const char*>(p), n); p += n; left -= n; return true;
    }
    bool words_bounded(std::vector<uint32_t>& v, uint32_t maximum, const char* length_error) {
        uint32_t n; if (!u32(n)) return false;
        if (n > maximum || uint64_t(n) * 4 > left) {
            if (error) *error = length_error;
            return false;
        }
        v.resize(n); for (auto& x : v) if (!u32(x)) return false; return true;
    }
    bool words(std::vector<uint32_t>& v) {
        return words_bounded(v, kMaxShaderWords, "invalid word-vector length");
    }
    bool bytes(std::vector<uint8_t>& v) {
        uint64_t n; if (!u64(n)) return false;
        if (n > max_blob_bytes() || n > left || n > std::numeric_limits<size_t>::max()) {
            if (error) *error = "invalid blob length"; return false;
        }
        v.resize(static_cast<size_t>(n)); return take(v.data(), v.size());
    }
};

// v43 (#1459): the raw color-state triple behind a resolved color write mask. Presence is packed
// alongside the values because "absent" and "present with value 0" resolve to opposite masks —
// write-all versus write-nothing — and only the raw registers distinguish them offline.
inline void write_color_state(Writer& w, const ResolvedPipelineState& ps) {
    const uint8_t present = (ps.has_cb_color_control ? 1u : 0u) |
                            (ps.has_cb_target_mask ? 2u : 0u) |
                            (ps.has_cb_shader_mask ? 4u : 0u);
    w.u8(present);
    w.u32(ps.cb_color_control);
    w.u32(ps.cb_target_mask);
    w.u32(ps.cb_shader_mask);
}

inline bool read_color_state(Reader& r, ResolvedPipelineState& ps) {
    uint8_t present = 0;
    if (!r.u8(present) || present > 7u) return false;
    ps.has_cb_color_control = (present & 1u) != 0;
    ps.has_cb_target_mask = (present & 2u) != 0;
    ps.has_cb_shader_mask = (present & 4u) != 0;
    return r.u32(ps.cb_color_control) && r.u32(ps.cb_target_mask) && r.u32(ps.cb_shader_mask);
}

inline void write_compute_config(Writer& w, const ComputeShaderConfig& config) {
    w.words(config.user_sgprs);
    w.u32(config.local_x); w.u32(config.local_y); w.u32(config.local_z);
    w.u8(config.exact_thread_extent ? 1u : 0u);
    w.u32(config.threads_x); w.u32(config.threads_y); w.u32(config.threads_z);
    w.u32(config.wave_size); w.u32(config.tidig_comp_cnt);
    const uint8_t flags = (config.tgid_x_en ? 1u : 0u) |
                          (config.tgid_y_en ? 2u : 0u) |
                          (config.tgid_z_en ? 4u : 0u) |
                          (config.tg_size_en ? 8u : 0u) |
                          (config.packed_r11_storage ? 16u : 0u) |
                          (config.storage_buffer_int64_atomics ? 32u : 0u);
    w.u8(flags);
    w.u32(config.lds_bytes);
    w.u32(config.native_subgroup_size);
    w.u32(config.native_storage_format_support);
}

inline bool read_compute_config(Reader& r, ComputeShaderConfig& config, uint32_t capture_version) {
    uint8_t exact = 0, flags = 0;
    const uint8_t valid_flags = capture_version >= 49 ? 0x3fu : 0x1fu;
    if (!r.words_bounded(config.user_sgprs, 32,
                         "invalid compute user-SGPR count") ||
        !r.u32(config.local_x) || !r.u32(config.local_y) ||
        !r.u32(config.local_z) || !r.u8(exact) || exact > 1u ||
        !r.u32(config.threads_x) || !r.u32(config.threads_y) ||
        !r.u32(config.threads_z) || !r.u32(config.wave_size) ||
        !r.u32(config.tidig_comp_cnt) || !r.u8(flags) || (flags & ~valid_flags) ||
        !r.u32(config.lds_bytes) || !r.u32(config.native_subgroup_size) ||
        !r.u32(config.native_storage_format_support))
        return false;
    if (capture_version < 50)
        config.compute_pgm_rsrc1 = UINT32_MAX;
    config.exact_thread_extent = exact != 0;
    config.tgid_x_en = (flags & 1u) != 0;
    config.tgid_y_en = (flags & 2u) != 0;
    config.tgid_z_en = (flags & 4u) != 0;
    config.tg_size_en = (flags & 8u) != 0;
    config.packed_r11_storage = (flags & 16u) != 0;
    config.storage_buffer_int64_atomics = capture_version >= 49 && (flags & 32u) != 0;
    return true;
}

inline void write_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u32(p.topology); w.u32(p.color0_format); w.u8(p.has_clear_color); for (float v : p.clear_color) w.f32(v);
    w.u8(p.depth_test_enable); w.u8(p.depth_write_enable); w.u32(p.depth_compare_op);
    w.f32(p.depth_clear_value); w.u32(p.stencil_clear_value); w.u8(p.stencil_enable);
    for (auto v : p.stencil_compare_op) w.u32(v); for (auto v : p.stencil_fail_op) w.u32(v);
    for (auto v : p.stencil_pass_op) w.u32(v); for (auto v : p.stencil_depth_fail_op) w.u32(v);
    for (auto v : p.stencil_ref) w.u32(v); for (auto v : p.stencil_op_val) w.u32(v);
    for (auto v : p.stencil_compare_mask) w.u32(v); for (auto v : p.stencil_write_mask) w.u32(v);
    w.u8(p.blend_enable); w.u32(p.src_color_blend_factor); w.u32(p.dst_color_blend_factor);
    w.u32(p.color_blend_op); w.u32(p.src_alpha_blend_factor); w.u32(p.dst_alpha_blend_factor);
    w.u32(p.alpha_blend_op); w.u32(p.color_write_mask); w.u8(p.has_viewport);
    w.f32(p.viewport_x); w.f32(p.viewport_y); w.f32(p.viewport_w); w.f32(p.viewport_h);
    w.f32(p.min_depth); w.f32(p.max_depth); w.u32(p.cull_mode); w.u32(p.front_face); w.u32(p.polygon_mode);
    w.u8(p.has_depth_clear); w.u8(p.has_stencil_clear); w.u32(p.db_render_control);
    w.u8(p.depth_clear_enable); w.u8(p.stencil_clear_enable);
    w.u64(p.depth_read_base); w.u64(p.depth_write_base); w.u64(p.stencil_read_base); w.u64(p.stencil_write_base);
    for (const auto& face : p.raw_stencil_op) for (auto v : face) w.u32(v);
    w.u32(p.db_shader_control); w.u8(p.stencil_test_val_export_enable); w.u8(p.stencil_op_val_export_enable);
    w.u32(p.db_depth_view); w.u32(p.db_render_override); w.u32(p.db_render_override2);
    w.u64(p.htile_data_base); w.u32(p.db_depth_size_xy); w.u32(p.db_dfsm_control);
    w.u32(p.db_depth_info); w.u32(p.db_z_info); w.u32(p.db_stencil_info);
    w.u32(p.db_depth_size); w.u32(p.db_depth_slice); w.u32(p.db_htile_surface);
    w.u32(p.db_rmi_l2_cache_control);
}

inline bool read_pipeline(Reader& r, ResolvedPipelineState& p, uint32_t version) {
    uint8_t b;
    if (!r.u32(p.topology) || !r.u32(p.color0_format) || !r.u8(b)) return false; p.has_clear_color = b != 0;
    for (float& v : p.clear_color) if (!r.f32(v)) return false;
    if (!r.u8(b)) return false; p.depth_test_enable = b != 0;
    if (!r.u8(b)) return false; p.depth_write_enable = b != 0;
    if (!r.u32(p.depth_compare_op) || !r.f32(p.depth_clear_value) || !r.u32(p.stencil_clear_value) || !r.u8(b)) return false;
    p.stencil_enable = b != 0;
    for (auto& v : p.stencil_compare_op) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_fail_op) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_pass_op) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_depth_fail_op) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_ref) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_op_val) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_compare_mask) if (!r.u32(v)) return false;
    for (auto& v : p.stencil_write_mask) if (!r.u32(v)) return false;
    if (!r.u8(b)) return false; p.blend_enable = b != 0;
    if (!r.u32(p.src_color_blend_factor) || !r.u32(p.dst_color_blend_factor) || !r.u32(p.color_blend_op) ||
        !r.u32(p.src_alpha_blend_factor) || !r.u32(p.dst_alpha_blend_factor) || !r.u32(p.alpha_blend_op) ||
        !r.u32(p.color_write_mask) || !r.u8(b)) return false;
    p.has_viewport = b != 0;
    if (!(r.f32(p.viewport_x) && r.f32(p.viewport_y) && r.f32(p.viewport_w) && r.f32(p.viewport_h) &&
          r.f32(p.min_depth) && r.f32(p.max_depth) && r.u32(p.cull_mode) && r.u32(p.front_face) &&
          r.u32(p.polygon_mode))) return false;
    if (version < 2) return true;
    if (!r.u8(b)) return false; p.has_depth_clear = b != 0;
    if (!r.u8(b) || !r.u32(p.db_render_control)) return false; p.has_stencil_clear = b != 0;
    if (!r.u8(b)) return false; p.depth_clear_enable = b != 0;
    if (!r.u8(b)) return false; p.stencil_clear_enable = b != 0;
    if (!r.u64(p.depth_read_base) || !r.u64(p.depth_write_base) || !r.u64(p.stencil_read_base) ||
        !r.u64(p.stencil_write_base)) return false;
    for (auto& face : p.raw_stencil_op) for (auto& v : face) if (!r.u32(v)) return false;
    if (!r.u32(p.db_shader_control) || !r.u8(b)) return false; p.stencil_test_val_export_enable = b != 0;
    if (!r.u8(b)) return false; p.stencil_op_val_export_enable = b != 0;
    if (version < 6) return true;
    if (!r.u32(p.db_depth_view) || !r.u32(p.db_render_override) || !r.u32(p.db_render_override2) ||
        !r.u64(p.htile_data_base) || !r.u32(p.db_depth_size_xy) || !r.u32(p.db_dfsm_control) ||
        !r.u32(p.db_depth_info) || !r.u32(p.db_z_info) || !r.u32(p.db_stencil_info) ||
        !r.u32(p.db_depth_size) || !r.u32(p.db_depth_slice) || !r.u32(p.db_htile_surface) ||
        !r.u32(p.db_rmi_l2_cache_control)) return false;
    return true;
}

inline void write_mrt1_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u32(p.color1_format); w.u8(p.has_clear_color1); for (float v : p.clear_color1) w.f32(v);
    w.u8(p.blend1_enable); w.u32(p.src_color_blend_factor1); w.u32(p.dst_color_blend_factor1);
    w.u32(p.color_blend_op1); w.u32(p.src_alpha_blend_factor1); w.u32(p.dst_alpha_blend_factor1);
    w.u32(p.alpha_blend_op1); w.u32(p.color1_write_mask);
}

inline bool read_mrt1_pipeline(Reader& r, ResolvedPipelineState& p) {
    uint8_t b = 0;
    if (!r.u32(p.color1_format) || !r.u8(b)) return false;
    p.has_clear_color1 = b != 0;
    for (float& v : p.clear_color1) if (!r.f32(v)) return false;
    if (!r.u8(b)) return false;
    p.blend1_enable = b != 0;
    return r.u32(p.src_color_blend_factor1) && r.u32(p.dst_color_blend_factor1) &&
           r.u32(p.color_blend_op1) && r.u32(p.src_alpha_blend_factor1) &&
           r.u32(p.dst_alpha_blend_factor1) && r.u32(p.alpha_blend_op1) &&
           r.u32(p.color1_write_mask);
}

inline void write_color_target_pipeline(Writer& w, const ResolvedPipelineState::ColorTarget& target) {
    w.u32(target.format); w.u8(target.has_clear);
    for (float value : target.clear) w.f32(value);
    w.u8(target.blend_enable);
    w.u32(target.src_color_blend_factor); w.u32(target.dst_color_blend_factor);
    w.u32(target.color_blend_op); w.u32(target.src_alpha_blend_factor);
    w.u32(target.dst_alpha_blend_factor); w.u32(target.alpha_blend_op);
    w.u32(target.write_mask); w.u8(target.disable_rop3);
}

inline bool read_color_target_pipeline(Reader& r, ResolvedPipelineState::ColorTarget& target) {
    uint8_t has_clear = 0, blend_enable = 0, disable_rop3 = 0;
    if (!r.u32(target.format) || !r.u8(has_clear) || has_clear > 1) return false;
    for (float& value : target.clear) if (!r.f32(value)) return false;
    if (!r.u8(blend_enable) || blend_enable > 1 ||
        !r.u32(target.src_color_blend_factor) ||
        !r.u32(target.dst_color_blend_factor) || !r.u32(target.color_blend_op) ||
        !r.u32(target.src_alpha_blend_factor) ||
        !r.u32(target.dst_alpha_blend_factor) || !r.u32(target.alpha_blend_op) ||
        !r.u32(target.write_mask) || target.write_mask > 0xfu ||
        !r.u8(disable_rop3) || disable_rop3 > 1)
        return false;
    target.has_clear = has_clear != 0;
    target.blend_enable = blend_enable != 0;
    target.disable_rop3 = disable_rop3 != 0;
    return true;
}

inline void restore_legacy_color_target_aliases(GpuCapturedDraw& draw) {
    draw.color_targets[0] = {draw.color0_base, draw.color0_width, draw.color0_height};
    draw.color_targets[1] = {draw.color1_base, draw.color1_width, draw.color1_height};
    auto& target0 = draw.ps.color_targets[0];
    target0.format = draw.ps.color0_format; target0.has_clear = draw.ps.has_clear_color;
    std::copy(std::begin(draw.ps.clear_color), std::end(draw.ps.clear_color), target0.clear);
    target0.blend_enable = draw.ps.blend_enable;
    target0.src_color_blend_factor = draw.ps.src_color_blend_factor;
    target0.dst_color_blend_factor = draw.ps.dst_color_blend_factor;
    target0.color_blend_op = draw.ps.color_blend_op;
    target0.src_alpha_blend_factor = draw.ps.src_alpha_blend_factor;
    target0.dst_alpha_blend_factor = draw.ps.dst_alpha_blend_factor;
    target0.alpha_blend_op = draw.ps.alpha_blend_op;
    target0.write_mask = draw.ps.color_write_mask;
    auto& target1 = draw.ps.color_targets[1];
    target1.format = draw.ps.color1_format; target1.has_clear = draw.ps.has_clear_color1;
    std::copy(std::begin(draw.ps.clear_color1), std::end(draw.ps.clear_color1), target1.clear);
    target1.blend_enable = draw.ps.blend1_enable;
    target1.src_color_blend_factor = draw.ps.src_color_blend_factor1;
    target1.dst_color_blend_factor = draw.ps.dst_color_blend_factor1;
    target1.color_blend_op = draw.ps.color_blend_op1;
    target1.src_alpha_blend_factor = draw.ps.src_alpha_blend_factor1;
    target1.dst_alpha_blend_factor = draw.ps.dst_alpha_blend_factor1;
    target1.alpha_blend_op = draw.ps.alpha_blend_op1;
    target1.write_mask = draw.ps.color1_write_mask;
}

inline void restore_legacy_color_target_aliases(GpuCapturedOperationFailure& diagnostic) {
    diagnostic.color_targets[0] = {
        diagnostic.color0_base, diagnostic.color0_width, diagnostic.color0_height};
    diagnostic.color_targets[1] = {
        diagnostic.color1_base, diagnostic.color1_width, diagnostic.color1_height};
    if (!diagnostic.pipeline_present) return;
    GpuCapturedDraw aliases;
    aliases.ps = diagnostic.pipeline;
    restore_legacy_color_target_aliases(aliases);
    diagnostic.pipeline.color_targets[0] = aliases.ps.color_targets[0];
    diagnostic.pipeline.color_targets[1] = aliases.ps.color_targets[1];
}

inline void write_scissor_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u8(p.has_scissor);
    w.u32(std::bit_cast<uint32_t>(p.scissor_left));
    w.u32(std::bit_cast<uint32_t>(p.scissor_top));
    w.u32(std::bit_cast<uint32_t>(p.scissor_right));
    w.u32(std::bit_cast<uint32_t>(p.scissor_bottom));
}

inline bool read_scissor_pipeline(Reader& r, ResolvedPipelineState& p) {
    uint8_t enabled = 0;
    uint32_t left = 0, top = 0, right = 0, bottom = 0;
    if (!r.u8(enabled) || enabled > 1 || !r.u32(left) || !r.u32(top) ||
        !r.u32(right) || !r.u32(bottom)) return false;
    p.has_scissor = enabled != 0;
    p.scissor_left = std::bit_cast<int32_t>(left);
    p.scissor_top = std::bit_cast<int32_t>(top);
    p.scissor_right = std::bit_cast<int32_t>(right);
    p.scissor_bottom = std::bit_cast<int32_t>(bottom);
    return true;
}

inline void write_logic_op_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u8(p.logic_op_enable);
    w.u32(p.logic_op);
}

inline bool read_logic_op_pipeline(Reader& r, ResolvedPipelineState& p) {
    uint8_t enabled = 0;
    if (!r.u8(enabled) || enabled > 1 || !r.u32(p.logic_op) || p.logic_op > 15u) return false;
    p.logic_op_enable = enabled != 0;
    return true;
}

inline void write_resource(Writer& w, const GpuCapturedResource& c) {
    const auto& r = c.resource;
    w.u32(static_cast<uint32_t>(r.cls)); w.u32(static_cast<uint32_t>(r.format));
    w.u32(r.num_components); w.u32(r.binding); w.u64(r.gpu_addr); w.u32(r.size); w.u32(r.stride);
    w.u32(r.srt_offset); w.u32(r.sgpr_base); w.u32(r.fetch_pc); w.u32(r.img_dim);
    w.u32(r.width); w.u32(r.height);
    w.u32(r.tile_mode); w.u8(r.srgb); w.u32(r.sampler_sgpr_base);
    w.u32(r.mag_filter); w.u32(r.min_filter); w.u32(r.mip_filter); for (auto v : r.addr_uvw) w.u32(v);
    w.u32(r.border_color_type); w.f32(r.min_lod); w.f32(r.max_lod); w.f32(r.lod_bias);
    w.u32(r.max_aniso_ratio); w.u32(r.depth_compare_func); w.u32(r.unnormalized);
    for (auto v : r.swizzle) w.u32(v);
    w.u32(c.blob_index); w.u64(c.blob_offset);
}

inline bool read_resource(Reader& rd, GpuCapturedResource& c, uint32_t version) {
    auto& r = c.resource; uint32_t cls, fmt; uint8_t b;
    if (!rd.u32(cls) || cls > static_cast<uint32_t>(ResourceClass::StorageImage) ||
        !rd.u32(fmt) || fmt > static_cast<uint32_t>(DataFormat::Sscaled16)) return false;
    r.cls = static_cast<ResourceClass>(cls); r.format = static_cast<DataFormat>(fmt);
    if (!rd.u32(r.num_components) || !rd.u32(r.binding) || !rd.u64(r.gpu_addr) || !rd.u32(r.size) ||
        !rd.u32(r.stride) || !rd.u32(r.srt_offset) || !rd.u32(r.sgpr_base) || !rd.u32(r.fetch_pc) ||
        !rd.u32(r.img_dim) || !rd.u32(r.width) || !rd.u32(r.height)) return false;
    r.depth = 1;
    r.sample_count = 1;
    r.depth_compare = false;
    if (!rd.u32(r.tile_mode) || !rd.u8(b)) return false;
    r.srgb = b != 0;
    if (!rd.u32(r.sampler_sgpr_base) || !rd.u32(r.mag_filter) || !rd.u32(r.min_filter) || !rd.u32(r.mip_filter)) return false;
    for (auto& v : r.addr_uvw) if (!rd.u32(v)) return false;
    if (!rd.u32(r.border_color_type) || !rd.f32(r.min_lod) || !rd.f32(r.max_lod) || !rd.f32(r.lod_bias) ||
        !rd.u32(r.max_aniso_ratio) || !rd.u32(r.depth_compare_func) || !rd.u32(r.unnormalized)) return false;
    for (auto& v : r.swizzle) if (!rd.u32(v)) return false;
    return rd.u32(c.blob_index) && rd.u64(c.blob_offset);
}

inline void write_table(Writer& w, const GpuCapturedTable& t) {
    w.u8(t.present); w.u32(static_cast<uint32_t>(t.resources.size()));
    for (const auto& r : t.resources) write_resource(w, r);
}

inline bool read_table(Reader& r, GpuCapturedTable& t, uint32_t version) {
    uint8_t b; uint32_t n; if (!r.u8(b) || !r.u32(n) || n > kMaxResources) return false;
    t.present = b != 0; t.resources.resize(n);
    for (auto& x : t.resources) if (!read_resource(r, x, version)) return false;
    if (!t.present && n != 0) { if (r.error) *r.error = "absent resource table has resources"; return false; }
    return true;
}

}  // namespace prosper::gpu
