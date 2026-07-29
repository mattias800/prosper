#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "gpu_capture.hpp"

#include <mutex>
#include "bc_decode.hpp"
#include "guest_texture_layout.hpp"
#include "rdna2_decode.hpp"
#include "rdna2_to_spirv.hpp"
#include "tile.hpp"
#include "videoout_present.hpp"

#include <algorithm>
#include <atomic>
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
#include "../host/posix_shim.hpp"
#endif

namespace prosper::gpu {
namespace {

constexpr char kMagic[8] = {'P','R','G','P','C','A','P','\0'};
// v23 (#1256): each realized draw also retains its RAW draw-packet state — the DrawIndexAuto/DrawIndex
// index_count and indexed flag as decoded from the guest, BEFORE realization. This lets gpu_replay
// --inspect-only surface raw-vs-realized offline and flag a decode/realization divergence (e.g. GTA
// #1163's non-indexed vertex-count inflation) without a live boot. Older captures read fine (the fields
// default to 0/false = "unknown").
// v24 (#1280): each resource's declared_mip_levels (the T#-declared mip-chain length) is serialized in a
// deterministic version-gated tail (like the v15 depth-compare / v16 mip-tail tails), so the byte-exact
// v1-v23 record prefix is preserved and older captures materialize with the historical default of 1. It
// was populated live but never written/read, so every reloaded .prgcap resource defaulted to 1 and a
// mip-declaring title replayed single-level.
// v25 (#1240): each realized/failed draw pipeline retains CB_COLOR_CONTROL.MODE=3 resolve intent in
// another version-gated tail, keeping every v1-v24 record prefix byte-exact.
// v26: retain SPI shader export and SX render-target downconversion formats per draw.
// v27: retain each draw's raw ShaderDrawModifier and signed GE_INDX_OFFSET. The latter is a draw
// parameter (Vulkan firstVertex/vertexOffset), not pipeline state; omitting it collapsed distinct
// ranges of a shared vertex pool onto vertex zero in both live rendering and replay.
// v28: retain the resolved byte row pitch and exact planned span of every resource. Guest-backed
// GFX10 linear sampled images use 256-byte-aligned rows, which differs from width*bytes-per-texel for
// video chroma planes.
// Older captures planned tight spans, so their materializer intentionally binds the legacy span while
// still deriving the guest pitch for best-effort rendering of the rows that were retained.
// v29: retain resolved depth-bias state per realized and failed pipeline.
// v30: each const-fold-resolved buffer fetch retains its proven vertex/instance/shader VADDR source.
// This is a per-resource trailing block so every v1-v29 record prefix stays byte-exact.
// v31: retain an optional linked vertex-main raw stream and the graphics-LDS allocation per draw.
// v32: retain the per-layer full-mip-chain stride and selected-level offset for thin 2D arrays and
// cube faces. Without them replay treated layer one's allocation-level bytes as layer zero's mip.
// v33: extend the append-only DataFormat enum with 8/16-bit USCALED and SSCALED buffer formats.
// Their numeric values follow every v1-v32 value, so old captures remain byte-identical/readable.
// v34: retain all eight hardware color-buffer slots and their fixed-function state.
// v35: retain the exact resource-descriptor metadata for each failed shader stage. This makes a
// table-dependent recompile retry deterministic offline instead of reducing the table to a count.
// v36: retain the resolved SPI_PS_INPUT_CNTL linkage (including metadata-only PARAM0 passthrough)
// and SPI_PS_INPUT_ENA/ADDR system-VGPR ABI for every realized draw. Raw graphics replay otherwise
// invents a different module even when the captured RDNA2 bytes are exact.
// v37: retain the required compute subgroup size. Native-subgroup SPIR-V reconstructs guest wave
// membership from SubgroupId/SubgroupLocalInvocationId and therefore must replay with the same
// required-size/full-subgroups pipeline contract used when the module was created.
constexpr uint32_t kVersion = 37;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint64_t kMaxFileBytes = 4ull << 30;
constexpr uint64_t kMaxBlobBytes = 1ull << 30;
constexpr uint64_t kMaxTotalBlobBytes = 3ull << 30;
constexpr uint32_t kMaxDraws = 65536;
constexpr uint32_t kMaxComputes = 65536;
constexpr uint32_t kMaxOperations = 131072;
constexpr uint32_t kMaxResources = 65536;
constexpr uint32_t kMaxShaderWords = 16u << 20;
constexpr uint32_t kMaxRawShaderWords = 0x4000; // 64 KiB per raw stage
constexpr uint32_t kMaxFailureStages = 3;
constexpr uint32_t kMaxStringBytes = 1u << 20;
constexpr uint64_t kMaxTotalRttSeedBytes = 1ull << 30;
constexpr uint64_t kMaxTotalDsSeedBytes = 1ull << 30;
constexpr uint64_t kDefaultResourceCaptureBytes = 512ull << 20;

CaptureRttSeedReader g_rtt_seed_reader;
CaptureRttSeedSnapshotReader g_rtt_seed_snapshot_reader;
ReplayRttSeedWriter g_rtt_seed_writer;
CaptureDsSeedSnapshotReader g_ds_seed_snapshot_reader;
ReplayDsSeedWriter g_ds_seed_writer;

size_t read_capture_guest_memory(uint64_t addr, uint8_t* dst, size_t bytes) {
#if defined(__linux__) || defined(__APPLE__)
    size_t done = 0;
    while (done < bytes) {
        iovec local{dst + done, bytes - done};
        iovec remote{reinterpret_cast<void*>(static_cast<uintptr_t>(addr + done)), bytes - done};
        const ssize_t read = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
        if (read < 0 && errno == EINTR) continue;
        if (read <= 0) break;
        done += static_cast<size_t>(read);
    }
    return done;
#else
    size_t done = 0;
    constexpr size_t chunk_max = 0x10000;
    while (done < bytes) {
        const size_t n = std::min(bytes - done, chunk_max);
        if (!guest_readable(addr + done, static_cast<uint32_t>(n))) break;
        std::memcpy(dst + done, reinterpret_cast<const void*>(static_cast<uintptr_t>(addr + done)), n);
        done += n;
    }
    return done;
#endif
}

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
        if (n > kMaxBlobBytes || n > left || n > std::numeric_limits<size_t>::max()) {
            if (error) *error = "invalid blob length"; return false;
        }
        v.resize(static_cast<size_t>(n)); return take(v.data(), v.size());
    }
};

void write_pipeline(Writer& w, const ResolvedPipelineState& p) {
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

bool read_pipeline(Reader& r, ResolvedPipelineState& p, uint32_t version) {
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

void write_mrt1_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u32(p.color1_format); w.u8(p.has_clear_color1); for (float v : p.clear_color1) w.f32(v);
    w.u8(p.blend1_enable); w.u32(p.src_color_blend_factor1); w.u32(p.dst_color_blend_factor1);
    w.u32(p.color_blend_op1); w.u32(p.src_alpha_blend_factor1); w.u32(p.dst_alpha_blend_factor1);
    w.u32(p.alpha_blend_op1); w.u32(p.color1_write_mask);
}

bool read_mrt1_pipeline(Reader& r, ResolvedPipelineState& p) {
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

void write_color_target_pipeline(Writer& w, const ResolvedPipelineState::ColorTarget& target) {
    w.u32(target.format); w.u8(target.has_clear);
    for (float value : target.clear) w.f32(value);
    w.u8(target.blend_enable);
    w.u32(target.src_color_blend_factor); w.u32(target.dst_color_blend_factor);
    w.u32(target.color_blend_op); w.u32(target.src_alpha_blend_factor);
    w.u32(target.dst_alpha_blend_factor); w.u32(target.alpha_blend_op);
    w.u32(target.write_mask); w.u8(target.disable_rop3);
}

bool read_color_target_pipeline(Reader& r, ResolvedPipelineState::ColorTarget& target) {
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

void restore_legacy_color_target_aliases(GpuCapturedDraw& draw) {
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

void restore_legacy_color_target_aliases(GpuCapturedOperationFailure& diagnostic) {
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

void write_scissor_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u8(p.has_scissor);
    w.u32(std::bit_cast<uint32_t>(p.scissor_left));
    w.u32(std::bit_cast<uint32_t>(p.scissor_top));
    w.u32(std::bit_cast<uint32_t>(p.scissor_right));
    w.u32(std::bit_cast<uint32_t>(p.scissor_bottom));
}

bool read_scissor_pipeline(Reader& r, ResolvedPipelineState& p) {
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

void write_logic_op_pipeline(Writer& w, const ResolvedPipelineState& p) {
    w.u8(p.logic_op_enable);
    w.u32(p.logic_op);
}

bool read_logic_op_pipeline(Reader& r, ResolvedPipelineState& p) {
    uint8_t enabled = 0;
    if (!r.u8(enabled) || enabled > 1 || !r.u32(p.logic_op) || p.logic_op > 15u) return false;
    p.logic_op_enable = enabled != 0;
    return true;
}

void write_resource(Writer& w, const GpuCapturedResource& c) {
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

bool read_resource(Reader& rd, GpuCapturedResource& c, uint32_t version) {
    auto& r = c.resource; uint32_t cls, fmt; uint8_t b;
    if (!rd.u32(cls) || cls > static_cast<uint32_t>(ResourceClass::StorageImage) ||
        !rd.u32(fmt) || fmt > static_cast<uint32_t>(DataFormat::Sscaled16)) return false;
    r.cls = static_cast<ResourceClass>(cls); r.format = static_cast<DataFormat>(fmt);
    if (!rd.u32(r.num_components) || !rd.u32(r.binding) || !rd.u64(r.gpu_addr) || !rd.u32(r.size) ||
        !rd.u32(r.stride) || !rd.u32(r.srt_offset) || !rd.u32(r.sgpr_base) || !rd.u32(r.fetch_pc) ||
        !rd.u32(r.img_dim) || !rd.u32(r.width) || !rd.u32(r.height)) return false;
    r.depth = 1;
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

void write_table(Writer& w, const GpuCapturedTable& t) {
    w.u8(t.present); w.u32(static_cast<uint32_t>(t.resources.size()));
    for (const auto& r : t.resources) write_resource(w, r);
}

bool read_table(Reader& r, GpuCapturedTable& t, uint32_t version) {
    uint8_t b; uint32_t n; if (!r.u8(b) || !r.u32(n) || n > kMaxResources) return false;
    t.present = b != 0; t.resources.resize(n);
    for (auto& x : t.resources) if (!read_resource(r, x, version)) return false;
    if (!t.present && n != 0) { if (r.error) *r.error = "absent resource table has resources"; return false; }
    return true;
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
    return a && b > std::numeric_limits<uint64_t>::max() / a ? std::numeric_limits<uint64_t>::max() : a * b;
}

uint32_t resolved_linear_row_pitch(const ShaderResource& r, uint32_t width, uint32_t bpt) {
    if (r.linear_row_pitch_bytes) return r.linear_row_pitch_bytes;
    const uint64_t tight = checked_mul(width, bpt);
    if (tight > UINT32_MAX) return UINT32_MAX;
    if (r.host_data) return static_cast<uint32_t>(tight);
    if (const uint32_t registered = guest_linear_texture_row_pitch(
            r.gpu_addr, static_cast<uint32_t>(tight)))
        return registered;
    const size_t aligned = linear_sampled_row_pitch(width, bpt);
    return aligned > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(aligned);
}

uint64_t resource_footprint_impl(const ShaderResource& r, bool legacy_linear_tight) {
    uint64_t result = r.size;
    if (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) return result;
    const uint64_t layers = r.img_dim == 3u ? 6u
        : ((r.img_dim == 2u || r.img_dim == 5u) ? std::max(r.depth, 1u) : 1u);
    uint32_t w = r.width ? r.width : 4, h = r.height ? r.height : 4;
    const uint32_t bc = bc_block_bytes(r.format);
    uint64_t decoded = 0;
    bool decoded_is_volume = false;
    if (bc) {
        uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
        decoded = tile_mode_is_tiled(r.tile_mode) ? tiled_elements_bytes(bw, bh, bc, r.tile_mode)
                                                  : checked_mul(checked_mul(bw, bh), bc);
    } else {
        uint32_t bpt = data_format_bytes(r.format) * (r.num_components ? r.num_components : 1);
        const bool native_wide = r.cls == ResourceClass::StorageImage ||
            (r.format == DataFormat::Float16 && (bpt == 2 || bpt == 4 || bpt == 8));
        if (bpt == 0 || (bpt > 4 && !native_wide)) bpt = 4;
        if (tile_mode_is_tiled(r.tile_mode)) {
            if (r.img_dim == 2u && r.depth > 1u) {
                decoded = tiled_volume_bytes(w, h, r.depth, r.tile_mode, bpt);
                decoded_is_volume = decoded != 0;
            }
            if (!decoded_is_volume)
                decoded = tiled_surface_bytes(w, h, r.tile_mode, 0, bpt);
        } else if (r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                   ((r.cls == ResourceClass::Texture && r.img_dim == 1u) ||
                    ((r.img_dim == 3u || r.img_dim == 5u) && r.layer_stride_bytes)) &&
                   !r.compression_enabled && !legacy_linear_tight) {
            // Guest-backed linear sampled images default to GFX10's 256-byte row alignment. Exact
            // HLE-producer provenance may override it (AvPlayer's CPU-staged NV12 is tight). Capturing
            // only width*height*bpt made genuinely padded video planes drift and omitted their tail.
            const uint32_t row_pitch = (r.img_dim == 3u || r.img_dim == 5u)
                ? static_cast<uint32_t>(linear_sampled_row_pitch(w, bpt))
                : resolved_linear_row_pitch(r, w, bpt);
            decoded = checked_mul(row_pitch, h);
        } else {
            decoded = checked_mul(checked_mul(w, h), bpt);
        }
    }
    if (r.layer_stride_bytes && layers > 1) {
        const uint64_t selected_bytes = r.in_mip_tail ? r.mip_tail_bytes : decoded;
        const uint64_t level_offset = r.in_mip_tail ? 0u : r.layer_mip_offset_bytes;
        decoded = checked_mul(r.layer_stride_bytes, layers - 1u);
        decoded = decoded > UINT64_MAX - level_offset
            ? UINT64_MAX : decoded + level_offset;
        decoded = decoded > UINT64_MAX - selected_bytes
            ? UINT64_MAX : decoded + selected_bytes;
    } else if (!decoded_is_volume) {
        decoded = checked_mul(decoded, layers);
    }
    return std::max(result, decoded);
}

uint64_t resource_footprint(const ShaderResource& r) {
    return resource_footprint_impl(r, false);
}

uint64_t legacy_resource_footprint(const ShaderResource& r) {
    return resource_footprint_impl(r, true);
}

uint64_t dcc_metadata_footprint(const ShaderResource& r) {
    if (!r.compression_enabled || !r.metadata_addr ||
        (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) ||
        bc_block_bytes(r.format))
        return 0;
    uint32_t bytes_per_texel = data_format_bytes(r.format) * (r.num_components ? r.num_components : 1u);
    if (!bytes_per_texel &&
        (r.format == DataFormat::Float10_11_11 || r.format == DataFormat::Unorm2_10_10_10))
        bytes_per_texel = 4;
    const uint32_t layers = r.img_dim == 3u ? 6u : (r.img_dim == 2u ? std::max(r.depth, 1u) : 1u);
    return gfx10_dcc_metadata_bytes(r.width, r.height, layers, r.tile_mode,
                                    bytes_per_texel, r.meta_pipe_aligned);
}

// Bundle manifests retain blob indices and offsets while their content lives in the bundle's
// deduplicated resource dictionary. make_capture_manifest marks those deliberately empty entries
// with the canonical empty-content hash; ordinary empty/default blobs do not bypass bounds checks.
bool capture_blob_payload_omitted(const GpuCaptureBlob& blob) {
    return blob.bytes.empty() && !blob.bytes_read &&
           blob.content_hash == gpu_capture_hash(nullptr, 0);
}

struct Interval {
    uint64_t begin = 0;
    uint64_t end = 0;
    uint32_t blob_index = 0xFFFFFFFFu;
};

bool is_compute_internal_gds(const ShaderResource& r) {
    return r.binding == kComputeInternalGdsBinding && r.gpu_addr == 0 &&
           r.cls == ResourceClass::ConstantBuffer && r.size == 64u * 1024u && r.stride == 4;
}

bool collect_intervals(const std::vector<DrawItem>& draws,
                       const std::vector<ComputeItem>& computes,
                       const std::vector<GpuState::DmaCopy>& dma_copies,
                       uint64_t resource_limit_bytes,
                       std::vector<Interval>& intervals, std::string& error) {
    uint64_t total = 0;
    auto add_table = [&](const ShaderResourceTable* t) -> bool {
        if (!t) return true;
        for (const auto& r : t->resources) {
            if (is_compute_internal_gds(r)) continue;
            uint64_t n = resource_footprint(r);
            if (n) {
                if (n > kMaxBlobBytes || r.gpu_addr > std::numeric_limits<uint64_t>::max() - n) {
                    char detail[512];
                    std::snprintf(
                        detail, sizeof(detail),
                        "resource capture range is invalid or exceeds 1 GiB: binding=%u class=%u "
                        "addr=0x%llx declared=%llu footprint=%llu format=%u components=%u "
                        "extent=%ux%ux%u img-dim=%u tile=%u layer-stride=%llu "
                        "layer-mip-offset=%llu mip-tail=%d/%llu",
                        r.binding, static_cast<unsigned>(r.cls),
                        static_cast<unsigned long long>(r.gpu_addr),
                        static_cast<unsigned long long>(r.size),
                        static_cast<unsigned long long>(n), static_cast<unsigned>(r.format),
                        r.num_components, r.width, r.height, r.depth, r.img_dim, r.tile_mode,
                        static_cast<unsigned long long>(r.layer_stride_bytes),
                        static_cast<unsigned long long>(r.layer_mip_offset_bytes),
                        r.in_mip_tail ? 1 : 0,
                        static_cast<unsigned long long>(r.mip_tail_bytes));
                    error = detail;
                    return false;
                }
                intervals.push_back({r.gpu_addr, r.gpu_addr + n});
            }
            n = dcc_metadata_footprint(r);
            if (n) {
                if (n > kMaxBlobBytes || r.metadata_addr > std::numeric_limits<uint64_t>::max() - n) {
                    error = "DCC metadata capture range is invalid or exceeds 1 GiB"; return false;
                }
                intervals.push_back({r.metadata_addr, r.metadata_addr + n});
            }
        }
        return true;
    };
    for (const auto& d : draws) if (!add_table(d.vrt.get()) || !add_table(d.prt.get())) return false;
    for (const auto& c : computes) if (!add_table(c.resources.get())) return false;
    for (const auto& copy : dma_copies) {
        if (!copy.dst || !copy.src || !copy.bytes ||
            copy.dst > std::numeric_limits<uint64_t>::max() - copy.bytes ||
            copy.src > std::numeric_limits<uint64_t>::max() - copy.bytes) {
            error = "ordered DMA capture range is invalid";
            return false;
        }
        intervals.push_back({copy.dst, copy.dst + copy.bytes});
        intervals.push_back({copy.src, copy.src + copy.bytes});
    }
    std::sort(intervals.begin(), intervals.end(), [](auto a, auto b) { return a.begin < b.begin; });
    std::vector<Interval> merged;
    for (auto x : intervals) {
        if (!merged.empty() && x.begin <= merged.back().end) merged.back().end = std::max(merged.back().end, x.end);
        else merged.push_back(x);
    }
    uint64_t largest = 0;
    for (auto x : merged) {
        uint64_t n = x.end - x.begin;
        if (total > kMaxTotalBlobBytes - n) { error = "capture resource data exceeds 3 GiB"; return false; }
        total += n;
        largest = std::max(largest, n);
    }
    if (total > resource_limit_bytes) {
        error = "capture resource data requires " + std::to_string((total + (1u << 20) - 1) >> 20) +
                " MiB across " + std::to_string(merged.size()) + " range(s), largest " +
                std::to_string((largest + (1u << 20) - 1) >> 20) + " MiB; limit is " +
                std::to_string(resource_limit_bytes >> 20) +
                " MiB (raise PROSPER_GPU_CAPTURE_MAX_MB or set "
                "PROSPER_GPU_CAPTURE_METADATA_ONLY=1)";
        return false;
    }
    intervals = std::move(merged); return true;
}

bool assign_blob_range(const std::vector<Interval>& intervals, uint64_t addr, uint64_t bytes,
                       uint32_t& index, uint64_t& offset, const char* missing,
                       std::string& error) {
    auto it = std::find_if(intervals.begin(), intervals.end(), [&](auto x) {
        return x.begin <= addr && addr <= x.end && bytes <= x.end - addr;
    });
    if (it == intervals.end()) { error = missing; return false; }
    index = it->blob_index;
    offset = addr - it->begin;
    return true;
}

bool capture_table(const ShaderResourceTable* src, const std::vector<Interval>& intervals,
                   bool include_resource_data, GpuCapturedTable& dst, std::string& error) {
    dst.present = src != nullptr;
    if (!src) return true;
    for (const auto& r : src->resources) {
        GpuCapturedResource c;
        c.resource = r;
        if (r.cls == ResourceClass::Texture && r.img_dim == 1u &&
            r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
            !r.compression_enabled && !bc_block_bytes(r.format)) {
            uint32_t bpt = data_format_bytes(r.format) *
                           (r.num_components ? r.num_components : 1u);
            const bool f16 = r.format == DataFormat::Float16 &&
                             (bpt == 2 || bpt == 4 || bpt == 8);
            if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
            c.resource.linear_row_pitch_bytes = resolved_linear_row_pitch(
                r, r.width ? r.width : 4u, bpt);
        }
        c.resource.host_data = nullptr; c.resource.host_data_size = 0;
        c.resource.dcc_metadata_host_data = nullptr;
        c.resource.dcc_metadata_host_data_size = 0;
        c.metadata_size = dcc_metadata_footprint(r);
        c.resource.dcc_metadata_size = c.metadata_size;
        uint64_t n = resource_footprint(r);
        c.captured_size = n;
        if (is_compute_internal_gds(r) && include_resource_data) {
            if (!r.host_data || r.host_data_size < n) {
                error = "compute GDS resource has no complete host backing";
                return false;
            }
            c.internal_bytes.assign(r.host_data, r.host_data + n);
        } else if (n && include_resource_data &&
            !assign_blob_range(intervals, r.gpu_addr, n, c.blob_index, c.blob_offset,
                               "resource was not assigned to a capture blob", error)) return false;
        if (c.metadata_size && include_resource_data &&
            !assign_blob_range(intervals, r.metadata_addr, c.metadata_size,
                               c.metadata_blob_index, c.metadata_blob_offset,
                               "DCC metadata was not assigned to a capture blob", error)) return false;
        dst.resources.push_back(std::move(c));
    }
    return true;
}

const char* env_or_empty(const char* name) { const char* v = std::getenv(name); return v ? v : ""; }

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") && std::strcmp(value, "off");
}

bool capture_resource_limit(uint64_t& bytes, std::string& error, uint64_t override_bytes = 0) {
    if (override_bytes) {
        if (override_bytes > kMaxTotalBlobBytes) {
            error = "capture resource limit exceeds 3 GiB";
            return false;
        }
        bytes = override_bytes;
        return true;
    }
    bytes = kDefaultResourceCaptureBytes;
    const char* value = std::getenv("PROSPER_GPU_CAPTURE_MAX_MB");
    if (!value || !*value) return true;
    char* end = nullptr;
    const uint64_t mib = std::strtoull(value, &end, 0);
    if (!end || *end || mib < 1 || mib > (kMaxTotalBlobBytes >> 20)) {
        error = "PROSPER_GPU_CAPTURE_MAX_MB must be between 1 and 3072";
        return false;
    }
    bytes = mib << 20;
    return true;
}

bool validate_rtt_seed(const GpuCaptureRttSeed& seed, std::string& error) {
    if (!seed.guest_addr || !seed.width || !seed.height) {
        error = "RTT seed has an invalid address or extent"; return false;
    }
    uint32_t bytes_per_pixel = 0;
    switch (seed.format) {
        case GpuCaptureColorFormat::Rgba8Unorm: bytes_per_pixel = 4; break;
        case GpuCaptureColorFormat::Rgba16Float: bytes_per_pixel = 8; break;
        case GpuCaptureColorFormat::R11G11B10Float: bytes_per_pixel = 4; break;
        default: error = "RTT seed has an unsupported color format"; return false;
    }
    const uint64_t pixels = checked_mul(seed.width, seed.height);
    const uint64_t bytes = checked_mul(pixels, bytes_per_pixel);
    if (bytes > kMaxBlobBytes || bytes != seed.rgba.size()) {
        error = "RTT seed byte count does not match its color format and extent"; return false;
    }
    return true;
}

auto ds_seed_key(const GpuCaptureDsSeed& seed) {
    return std::tuple(seed.depth_read_base, seed.depth_write_base,
                      seed.stencil_read_base, seed.stencil_write_base,
                      seed.htile_data_base, seed.width, seed.height,
                      static_cast<uint32_t>(seed.format));
}

bool validate_ds_seed(const GpuCaptureDsSeed& seed, std::string& error) {
    if ((!seed.depth_read_base && !seed.depth_write_base && !seed.stencil_read_base &&
         !seed.stencil_write_base && !seed.htile_data_base) || !seed.width || !seed.height) {
        error = "DS seed has an invalid identity or extent"; return false;
    }
    if (seed.format != GpuCaptureDsFormat::D32Float &&
        seed.format != GpuCaptureDsFormat::D32FloatS8) {
        error = "DS seed has an unsupported format"; return false;
    }
    if (!seed.depth_valid && !seed.stencil_valid) {
        error = "DS seed has no valid plane"; return false;
    }
    const uint64_t pixels = checked_mul(seed.width, seed.height);
    const uint64_t depth_bytes = checked_mul(pixels, 4);
    if ((seed.depth_valid && (depth_bytes > kMaxBlobBytes || depth_bytes != seed.depth.size())) ||
        (!seed.depth_valid && !seed.depth.empty())) {
        error = "DS seed depth byte count does not match its extent and validity"; return false;
    }
    if ((seed.stencil_valid &&
         (seed.format != GpuCaptureDsFormat::D32FloatS8 || pixels > kMaxBlobBytes ||
          pixels != seed.stencil.size())) ||
        (!seed.stencil_valid && !seed.stencil.empty())) {
        error = "DS seed stencil byte count does not match its extent, format, and validity";
        return false;
    }
    return true;
}

uint64_t shader_hash(const std::vector<uint32_t>& words) {
    return gpu_capture_hash(reinterpret_cast<const uint8_t*>(words.data()),
                            words.size() * sizeof(uint32_t));
}

void collect_shader_versions(GpuCaptureFile& capture) {
    capture.shader_versions.clear();
    auto add = [&](const std::vector<uint32_t>& words) {
        const uint64_t hash = shader_hash(words);
        auto it = std::find_if(capture.shader_versions.begin(), capture.shader_versions.end(),
            [&](const auto& version) { return version.content_hash == hash && version.words == words; });
        if (it == capture.shader_versions.end())
            capture.shader_versions.push_back({hash, words});
    };
    for (const auto& draw : capture.draws) {
        add(draw.vs);
        if (!draw.gs.empty()) add(draw.gs);
        add(draw.fs);
    }
    for (const auto& compute : capture.computes) add(compute.spirv);
}

uint32_t shader_version_index(const std::vector<GpuCaptureShaderVersion>& versions,
                              const std::vector<uint32_t>& words) {
    const uint64_t hash = shader_hash(words);
    auto it = std::find_if(versions.begin(), versions.end(), [&](const auto& version) {
        return version.content_hash == hash && version.words == words;
    });
    return it == versions.end() ? 0xFFFFFFFFu : static_cast<uint32_t>(it - versions.begin());
}

using OperationIdentity = std::tuple<uint8_t, uint64_t, uint64_t>;

OperationIdentity operation_identity(SubmitOperationKind kind, uint64_t source_index,
                                     uint64_t command_order) {
    return {static_cast<uint8_t>(kind), source_index, command_order};
}

bool validate_dma_copies(const GpuCaptureFile& capture, std::string& error) {
    if (capture.dma_copies.size() > kMaxOperations) {
        error = "invalid ordered DMA count";
        return false;
    }
    std::vector<uint8_t> referenced(capture.dma_copies.size(), 0);
    auto validate_blob = [&](uint32_t index, uint64_t offset, uint32_t bytes,
                             const char* invalid, const char* exceeds) {
        if (index == 0xFFFFFFFFu) {
            if (!offset) return true;
            error = invalid;
            return false;
        }
        if (index >= capture.blobs.size()) {
            error = invalid;
            return false;
        }
        const auto& blob = capture.blobs[index];
        if (!capture_blob_payload_omitted(blob) &&
            (offset > blob.bytes.size() || bytes > blob.bytes.size() - offset)) {
            error = exceeds;
            return false;
        }
        return true;
    };
    for (const auto& copy : capture.dma_copies) {
        if (!copy.dst || !copy.src || !copy.bytes || copy.bytes > kMaxBlobBytes ||
            copy.dst > std::numeric_limits<uint64_t>::max() - copy.bytes ||
            copy.src > std::numeric_limits<uint64_t>::max() - copy.bytes ||
            !validate_blob(copy.destination_blob_index, copy.destination_blob_offset, copy.bytes,
                           "ordered DMA destination references an invalid capture blob",
                           "ordered DMA destination exceeds its capture blob") ||
            !validate_blob(copy.source_blob_index, copy.source_blob_offset, copy.bytes,
                           "ordered DMA source references an invalid capture blob",
                           "ordered DMA source exceeds its capture blob")) {
            if (error.empty()) error = "invalid ordered DMA record";
            return false;
        }
    }
    for (const auto& operation : capture.operations) {
        if (operation.kind > SubmitOperationKind::DmaCopy) {
            error = "invalid operation kind";
            return false;
        }
        if (operation.kind != SubmitOperationKind::DmaCopy) continue;
        if (!operation.realized || operation.source_index >= capture.dma_copies.size() ||
            capture.dma_copies[operation.source_index].command_order != operation.command_order ||
            referenced[operation.source_index] == UINT8_MAX) {
            error = "ordered DMA operation does not match its record";
            return false;
        }
        ++referenced[operation.source_index];
    }
    if (std::find_if(referenced.begin(), referenced.end(), [](uint8_t count) { return count != 1; }) !=
        referenced.end()) {
        error = "ordered DMA record must have exactly one operation";
        return false;
    }
    return true;
}

bool capture_raw_shader_version(uint64_t addr, const CaptureMemoryReader& reader,
                                GpuCaptureFile& capture, uint64_t& raw_words,
                                std::map<uint64_t, uint32_t>& index_by_address,
                                uint32_t& index, std::string& error) {
    index = 0xFFFFFFFFu;
    if (!addr) return true;
    const auto known = index_by_address.find(addr);
    if (known != index_by_address.end()) {
        index = known->second;
        return true;
    }
    std::vector<uint32_t> words(kMaxRawShaderWords);
    const size_t bytes_read = std::min<size_t>(
        reader(addr, reinterpret_cast<uint8_t*>(words.data()),
               words.size() * sizeof(uint32_t)),
        words.size() * sizeof(uint32_t));
    words.resize(bytes_read / sizeof(uint32_t));
    if (words.empty()) {
        index_by_address.emplace(addr, index);
        return true;
    }
    std::vector<Rdna2Inst> instructions;
    const size_t consumed = rdna2_walk(words.data(), words.size(), instructions);
    // A few compiler-generated shaders address constant tables stored after S_ENDPGM through an
    // s_getpc_b64-built descriptor. Keep the same proven code span as the live recompiler cache so
    // an offline raw replay sees every byte that affected the generated SPIR-V. The span helper is
    // deliberately fail-closed: unrelated post-program guest memory is still discarded.
    const size_t recompile_span = rdna2_recompile_code_span(words.data(), words.size());
    const size_t captured_span = std::max(consumed, recompile_span);
    if (captured_span && captured_span < words.size()) words.resize(captured_span);
    const bool has_endpgm = !instructions.empty() && instructions.back().is_end;
    const uint64_t hash = shader_hash(words);
    auto existing = std::find_if(capture.raw_shader_versions.begin(),
                                 capture.raw_shader_versions.end(), [&](const auto& candidate) {
        return candidate.content_hash == hash && candidate.words == words;
    });
    if (existing != capture.raw_shader_versions.end()) {
        index = static_cast<uint32_t>(existing - capture.raw_shader_versions.begin());
        index_by_address.emplace(addr, index);
        return true;
    }
    if (capture.raw_shader_versions.size() >= kMaxResources ||
        raw_words > kMaxShaderWords - words.size()) {
        error = "raw shader data exceeds its bounded limit";
        return false;
    }
    index = static_cast<uint32_t>(capture.raw_shader_versions.size());
    raw_words += words.size();
    capture.raw_shader_versions.push_back({hash, has_endpgm, std::move(words)});
    index_by_address.emplace(addr, index);
    return true;
}

bool validate_failure_diagnostics(const GpuCaptureFile& capture, std::string& error) {
    if (capture.raw_shader_versions.size() > kMaxResources ||
        capture.failure_diagnostics.size() > kMaxOperations) {
        error = "invalid failed-operation diagnostic count";
        return false;
    }
    uint64_t raw_words = 0;
    for (const auto& shader : capture.raw_shader_versions) {
        if (shader.words.empty() || shader.words.size() > kMaxRawShaderWords ||
            raw_words > kMaxShaderWords - shader.words.size()) {
            error = "raw shader data exceeds its bounded limit";
            return false;
        }
        if (shader.content_hash != shader_hash(shader.words)) {
            error = "raw shader content hash mismatch";
            return false;
        }
        raw_words += shader.words.size();
    }

    std::set<OperationIdentity> diagnosed;
    std::vector<bool> raw_referenced(capture.raw_shader_versions.size(), false);
    for (const auto& draw : capture.draws) {
        for (uint32_t index : {draw.vs_raw_shader_index, draw.fs_raw_shader_index,
                               draw.vs_chain_raw_shader_index}) {
            if (index == 0xFFFFFFFFu) continue;
            if (index >= capture.raw_shader_versions.size()) {
                error = "realized draw references an invalid raw shader";
                return false;
            }
            raw_referenced[index] = true;
        }
    }
    for (const auto& diagnostic : capture.failure_diagnostics) {
        if (diagnostic.kind > SubmitOperationKind::Dispatch ||
            diagnostic.reason <= RealizationFailureReason::None ||
            diagnostic.reason > RealizationFailureReason::Filtered ||
            diagnostic.stages.size() > kMaxFailureStages) {
            error = "invalid failed-operation diagnostic metadata";
            return false;
        }
        const OperationIdentity identity = operation_identity(
            diagnostic.kind, diagnostic.source_index, diagnostic.command_order);
        const auto operation = std::find_if(capture.operations.begin(), capture.operations.end(),
            [&](const auto& candidate) {
                return operation_identity(candidate.kind, candidate.source_index,
                                          candidate.command_order) == identity;
            });
        if (operation == capture.operations.end() || operation->realized) {
            error = "failed-operation diagnostic does not match an unrealized operation";
            return false;
        }
        if (!diagnosed.insert(identity).second) {
            error = "duplicate failed-operation diagnostic";
            return false;
        }
        // A hardware graphics stage can be assembled from more than one separately allocated
        // program.  In particular, AGC's NGG vertex path binds a prolog that transfers control to
        // the separately registered main shader.  Keep both raw streams in a failed-operation
        // diagnostic: treating the logical stage as unique made exactly the capture needed to
        // diagnose a linked-stage failure impossible.  The (kind,address) identity must still be
        // unique so an accidentally duplicated record remains fail-visible.
        std::set<std::pair<uint8_t, uint64_t>> stage_programs;
        for (const auto& stage : diagnostic.stages) {
            if (stage.stage > ShaderProgramStage::Compute ||
                (stage.raw_shader_index != 0xFFFFFFFFu &&
                 stage.raw_shader_index >= capture.raw_shader_versions.size()) ||
                stage.resource_count > kMaxResources ||
                stage.coverage.total > kMaxRawShaderWords ||
                stage.coverage.alu > stage.coverage.total ||
                stage.coverage.exports > stage.coverage.total ||
                stage.coverage.table_dependent > stage.coverage.total ||
                stage.coverage.unsupported > stage.coverage.total ||
                (!stage.resource_table_present && stage.resource_count != 0) ||
                (!stage.program_addr && stage.raw_shader_index != 0xFFFFFFFFu) ||
                !stage_programs.emplace(static_cast<uint8_t>(stage.stage),
                                        stage.program_addr).second) {
                error = "invalid failed-stage diagnostic metadata";
                return false;
            }
            // Rewriting a v7-v34 capture cannot invent tables that the old file reduced to a
            // presence/count summary. Preserve that summary with an absent v35 table so retry
            // reports "capture predates v35" explicitly. Any table that is actually retained must
            // agree exactly with the summary.
            if ((stage.resource_table.present || !stage.resource_table.resources.empty()) &&
                (stage.resource_table.present != stage.resource_table_present ||
                 stage.resource_table.resources.size() != stage.resource_count)) {
                error = "failed-stage resource table disagrees with its diagnostic summary";
                return false;
            }
            if (stage.raw_shader_index != 0xFFFFFFFFu)
                raw_referenced[stage.raw_shader_index] = true;
            const uint32_t max_issue = static_cast<uint32_t>(DescriptorIssueCode::UnusedRuntimeBinding);
            if ((!stage.descriptor_issue_count && stage.first_descriptor_issue != 0xFFFFFFFFu) ||
                (stage.descriptor_issue_count && stage.first_descriptor_issue > max_issue)) {
                error = "invalid failed-stage descriptor diagnostic";
                return false;
            }
        }
    }
    for (const auto& operation : capture.operations) {
        if (!operation.realized && diagnosed.count(operation_identity(
                operation.kind, operation.source_index, operation.command_order)) == 0) {
            error = "unrealized operation is missing its failure diagnostic";
            return false;
        }
    }
    if (std::find(raw_referenced.begin(), raw_referenced.end(), false) != raw_referenced.end()) {
        error = "raw shader is not referenced";
        return false;
    }
    return true;
}

bool capture_failure_diagnostics(
    const std::vector<OperationRealizationFailure>& failures,
    const CaptureMemoryReader& reader, GpuCaptureFile& capture, uint64_t& raw_words,
    std::map<uint64_t, uint32_t>& raw_shader_index_by_address,
    std::string& error) {
    if (failures.size() > kMaxOperations) {
        error = "invalid failed-operation diagnostic count";
        return false;
    }
    for (const auto& failure : failures) {
        GpuCapturedOperationFailure diagnostic;
        diagnostic.kind = failure.kind;
        diagnostic.source_index = failure.index;
        diagnostic.command_order = failure.command_order;
        diagnostic.reason = failure.reason;
        diagnostic.pipeline_present = failure.pipeline_present;
        diagnostic.pipeline = failure.pipeline;
        diagnostic.color0_base = failure.color0_base;
        diagnostic.color0_width = failure.color0_width;
        diagnostic.color0_height = failure.color0_height;
        diagnostic.color1_base = failure.color1_base;
        diagnostic.color1_width = failure.color1_width;
        diagnostic.color1_height = failure.color1_height;
        diagnostic.color_targets = failure.color_targets;
        diagnostic.color_targets[0] = {
            diagnostic.color0_base, diagnostic.color0_width, diagnostic.color0_height};
        diagnostic.color_targets[1] = {
            diagnostic.color1_base, diagnostic.color1_width, diagnostic.color1_height};
        diagnostic.vertex_count = failure.vertex_count;
        diagnostic.compute_launch = failure.compute_launch;
        for (const auto& runtime_stage : failure.stages) {
            GpuCapturedStageDiagnostic stage;
            stage.stage = runtime_stage.stage;
            stage.program_addr = runtime_stage.program_addr;
            stage.recompiled = runtime_stage.recompiled;
            stage.resource_table_present = runtime_stage.resources != nullptr;
            stage.resource_count = runtime_stage.resources
                ? static_cast<uint32_t>(runtime_stage.resources->resources.size()) : 0;
            stage.coverage = runtime_stage.coverage;
            stage.descriptor_issue_count = runtime_stage.descriptor_issue_count;
            stage.first_descriptor_issue = runtime_stage.first_descriptor_issue;
            if (!capture_table(runtime_stage.resources.get(), {}, false,
                               stage.resource_table, error)) return false;
            if (!capture_raw_shader_version(stage.program_addr, reader, capture, raw_words,
                                            raw_shader_index_by_address,
                                            stage.raw_shader_index, error)) return false;
            if (stage.raw_shader_index < capture.raw_shader_versions.size()) {
                const auto& raw = capture.raw_shader_versions[stage.raw_shader_index].words;
                stage.coverage = recompile_coverage(raw.data(), raw.size());
            }
            diagnostic.stages.push_back(std::move(stage));
        }
        capture.failure_diagnostics.push_back(std::move(diagnostic));
    }
    for (const auto& operation : capture.operations) {
        if (operation.realized) continue;
        const auto existing = std::find_if(capture.failure_diagnostics.begin(),
                                           capture.failure_diagnostics.end(), [&](const auto& candidate) {
            return operation_identity(candidate.kind, candidate.source_index,
                                      candidate.command_order) ==
                   operation_identity(operation.kind, operation.source_index,
                                      operation.command_order);
        });
        if (existing == capture.failure_diagnostics.end()) {
            GpuCapturedOperationFailure unknown;
            unknown.kind = operation.kind;
            unknown.source_index = operation.source_index;
            unknown.command_order = operation.command_order;
            unknown.reason = RealizationFailureReason::Unknown;
            capture.failure_diagnostics.push_back(std::move(unknown));
        }
    }
    return validate_failure_diagnostics(capture, error);
}

} // namespace

void annotate_gpu_capture_scanout(GpuCaptureMetadata& metadata) {
    const uint64_t address = present_front_address();
    if (!address) return;
    char value[24];
    std::snprintf(value, sizeof(value), "0x%llx",
                  static_cast<unsigned long long>(address));
    const auto existing = std::find_if(metadata.renderer_env.begin(), metadata.renderer_env.end(),
        [](const auto& entry) { return entry.first == kGpuReplayScanoutAddressEnv; });
    if (existing == metadata.renderer_env.end())
        metadata.renderer_env.emplace_back(kGpuReplayScanoutAddressEnv, value);
    else
        existing->second = value;
}

uint64_t parse_gpu_replay_scanout_address(const char* value) {
    if (!value || !*value) return 0;
    char* end = nullptr;
    const uint64_t parsed = std::strtoull(value, &end, 0);
    return end && end != value && !*end ? parsed : 0;
}

uint64_t gpu_capture_resource_footprint(const ShaderResource& resource) {
    return resource_footprint(resource);
}

uint64_t gpu_capture_dcc_metadata_footprint(const ShaderResource& resource) {
    return dcc_metadata_footprint(resource);
}

uint64_t gpu_capture_hash(const uint8_t* data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) { h ^= data[i]; h *= 1099511628211ull; }
    return h;
}

const char* shader_program_stage_name(ShaderProgramStage stage) {
    switch (stage) {
        case ShaderProgramStage::Vertex: return "vertex";
        case ShaderProgramStage::Fragment: return "fragment";
        case ShaderProgramStage::Compute: return "compute";
    }
    return "unknown";
}

const char* realization_failure_reason_name(RealizationFailureReason reason) {
    switch (reason) {
        case RealizationFailureReason::None: return "none";
        case RealizationFailureReason::Unknown: return "unknown";
        case RealizationFailureReason::MissingProgram: return "missing-program";
        case RealizationFailureReason::ShaderRecompile: return "shader-recompile";
        case RealizationFailureReason::DescriptorContract: return "descriptor-contract";
        case RealizationFailureReason::NoEffect: return "no-effect";
        case RealizationFailureReason::ZeroVertices: return "zero-vertices";
        case RealizationFailureReason::Filtered: return "filtered";
    }
    return "unknown";
}

bool capture_draw_items(const std::vector<DrawItem>& items, const GpuCaptureMetadata& metadata,
                        const CaptureMemoryReader& reader, GpuCaptureFile& out, std::string& error,
                        const CaptureRttSeedReader& rtt_reader) {
    if (items.empty()) { error = "capture must contain at least one draw"; return false; }
    std::vector<SubmitOperation> operations;
    operations.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i)
        operations.push_back({SubmitOperationKind::Draw, static_cast<size_t>(items[i].draw_index),
                              items[i].command_order});
    return capture_submit_items(items, {}, operations, metadata, reader, out, error, rtt_reader);
}

bool capture_submit_items(const std::vector<DrawItem>& draws,
                          const std::vector<ComputeItem>& computes,
                          const std::vector<SubmitOperation>& operations,
                          const GpuCaptureMetadata& metadata,
                          const CaptureMemoryReader& reader, GpuCaptureFile& out,
                          std::string& error, const CaptureRttSeedReader& rtt_reader,
                          const std::vector<OperationRealizationFailure>& failures,
                          const std::vector<GpuState::DmaCopy>& dma_copies,
                          uint64_t resource_limit_bytes_override) {
    error.clear(); out = {}; out.format_version = kVersion; out.metadata = metadata;
    out.failure_diagnostics_available = true;
    if (draws.size() > kMaxDraws || computes.size() > kMaxComputes ||
        operations.size() > kMaxOperations) {
        error = "capture item or operation count is invalid";
        return false;
    }
    if (!reader) { error = "capture memory reader is missing"; return false; }
    const bool include_resource_data = !env_enabled("PROSPER_GPU_CAPTURE_METADATA_ONLY");
    uint64_t resource_limit_bytes = 0;
    if (!capture_resource_limit(resource_limit_bytes, error, resource_limit_bytes_override)) return false;
    if (!include_resource_data && std::none_of(
            out.metadata.renderer_env.begin(), out.metadata.renderer_env.end(),
            [](const auto& entry) { return entry.first == "PROSPER_GPU_CAPTURE_METADATA_ONLY"; }))
        out.metadata.renderer_env.emplace_back("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
    std::vector<Interval> intervals;
    if (include_resource_data &&
        !collect_intervals(draws, computes, dma_copies, resource_limit_bytes, intervals, error)) return false;
    for (auto& x : intervals) {
        GpuCaptureBlob b; b.guest_addr = x.begin; b.bytes.resize(static_cast<size_t>(x.end - x.begin), 0);
        b.bytes_read = std::min<uint64_t>(reader(x.begin, b.bytes.data(), b.bytes.size()), b.bytes.size());
        b.content_hash = gpu_capture_hash(b.bytes);
        auto existing = std::find_if(out.blobs.begin(), out.blobs.end(), [&](const auto& candidate) {
            return candidate.content_hash == b.content_hash && candidate.bytes == b.bytes;
        });
        if (existing == out.blobs.end()) {
            x.blob_index = static_cast<uint32_t>(out.blobs.size());
            out.blobs.push_back(std::move(b));
        } else {
            x.blob_index = static_cast<uint32_t>(existing - out.blobs.begin());
        }
    }
    uint64_t raw_shader_words = 0;
    std::map<uint64_t, uint32_t> raw_shader_index_by_address;
    for (const auto& d : draws) {
        GpuCapturedDraw c; c.vs = d.vs_words(); c.gs = d.gs_words(); c.fs = d.fs_words();
        c.ps = d.ps; c.vertex_count = d.vertex_count;
        c.instance_count = d.instance_count;
        c.raw_draw_count = d.raw_draw_count; c.raw_indexed = d.raw_indexed;   // #1256
        c.raw_draw_modifier = d.raw_draw_modifier;
        c.vertex_offset = d.vertex_offset;
        c.indices = d.indices; c.color0_base = d.color0_base;
        c.color0_width = d.color0_width; c.color0_height = d.color0_height;
        c.color1_base = d.color1_base;
        c.color1_width = d.color1_width; c.color1_height = d.color1_height;
        c.color_targets = d.color_targets;
        c.color_targets[0] = {c.color0_base, c.color0_width, c.color0_height};
        c.color_targets[1] = {c.color1_base, c.color1_width, c.color1_height};
        c.draw_index = d.draw_index; c.command_order = d.command_order;
        if (!capture_raw_shader_version(d.vs_guest_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.vs_raw_shader_index, error) ||
            !capture_raw_shader_version(d.fs_guest_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.fs_raw_shader_index, error) ||
            !capture_raw_shader_version(d.vs_chain_guest_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.vs_chain_raw_shader_index, error)) return false;
        c.vertex_lds_dwords = d.vertex_lds_dwords;
        c.pixel_inputs = d.pixel_inputs;
        c.system_inputs = d.system_inputs;
        c.has_pixel_inputs = d.has_pixel_inputs;
        c.has_system_inputs = d.has_system_inputs;
        if (!capture_table(d.vrt.get(), intervals, include_resource_data, c.vrt, error) ||
            !capture_table(d.prt.get(), intervals, include_resource_data, c.prt, error)) return false;
        out.draws.push_back(std::move(c));
    }
    for (const auto& compute : computes) {
        GpuCapturedCompute c;
        c.spirv = compute.spirv;
        c.launch = compute.launch;
        c.code_addr = compute.code_addr;
        c.dispatch_index = compute.dispatch_index;
        c.submit_no = compute.submit_no;
        c.command_order = compute.command_order;
        c.required_subgroup_size = compute.required_subgroup_size;
        if (!capture_table(compute.resources.get(), intervals, include_resource_data,
                           c.resources, error)) return false;
        out.computes.push_back(std::move(c));
    }
    out.dma_copies.reserve(dma_copies.size());
    for (const auto& copy : dma_copies) {
        GpuCapturedDmaCopy captured;
        captured.dst = copy.dst; captured.src = copy.src; captured.bytes = copy.bytes;
        captured.sels = copy.sels; captured.command_order = copy.command_order;
        captured.packet_addr = copy.packet_addr;
        if (include_resource_data &&
            (!assign_blob_range(intervals, copy.dst, copy.bytes,
                                captured.destination_blob_index,
                                captured.destination_blob_offset,
                                "ordered DMA destination was not assigned to a capture blob", error) ||
             !assign_blob_range(intervals, copy.src, copy.bytes,
                                captured.source_blob_index, captured.source_blob_offset,
                                "ordered DMA source was not assigned to a capture blob", error)))
            return false;
        out.dma_copies.push_back(captured);
    }
    std::unordered_set<uint64_t> realized_draws, realized_computes;
    for (const auto& draw : out.draws) realized_draws.insert(draw.draw_index);
    for (const auto& compute : out.computes) realized_computes.insert(compute.dispatch_index);
    for (const auto& operation : operations) {
        bool realized = false;
        switch (operation.kind) {
            case SubmitOperationKind::Draw:
                realized = realized_draws.count(operation.index) != 0;
                break;
            case SubmitOperationKind::Dispatch:
                realized = realized_computes.count(operation.index) != 0;
                break;
            case SubmitOperationKind::DmaCopy:
                if (operation.index >= out.dma_copies.size() ||
                    out.dma_copies[operation.index].command_order != operation.command_order) {
                    error = "ordered DMA operation references an invalid record";
                    return false;
                }
                realized = true;
                break;
        }
        out.operations.push_back({operation.kind, operation.index, operation.command_order, realized});
    }
    if (!validate_dma_copies(out, error)) return false;
    if (!capture_failure_diagnostics(failures, reader, out, raw_shader_words,
                                     raw_shader_index_by_address, error)) return false;
    collect_shader_versions(out);
    if (rtt_reader && include_resource_data) {
        std::vector<uint64_t> candidates;
        auto add_table = [&](const ShaderResourceTable* table) {
            if (!table) return;
            for (const auto& r : table->resources)
                if ((r.cls == ResourceClass::Texture || r.cls == ResourceClass::StorageImage) && r.gpu_addr)
                    candidates.push_back(r.gpu_addr);
        };
        for (const auto& item : draws) {
            add_table(item.vrt.get()); add_table(item.prt.get());
            for (const auto& target : item.color_targets)
                if (target.base) candidates.push_back(target.base);
            // Preserve direct callers that only populate the named compatibility fields.
            if (item.color0_base) candidates.push_back(item.color0_base);
            if (item.color1_base) candidates.push_back(item.color1_base);
        }
        for (const auto& item : computes) add_table(item.resources.get());
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        uint64_t total = 0;
        for (uint64_t addr : candidates) {
            GpuCaptureRttSeed seed;
            if (!rtt_reader(addr, seed)) continue;
            if (seed.guest_addr != addr) { error = "RTT seed reader returned the wrong guest address"; return false; }
            if (!validate_rtt_seed(seed, error)) return false;
            if (total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
                error = "capture RTT seed data exceeds 1 GiB"; return false;
            }
            total += seed.rgba.size(); out.rtt_seeds.push_back(std::move(seed));
        }
    }
    return validate_dma_copies(out, error);
}

static CaptureMemoryReader ordered_gpustate_capture_reader(const GpuState& state) {
    return [&state](uint64_t addr, uint8_t* destination, size_t bytes) -> size_t {
        size_t copied = read_capture_guest_memory(addr, destination, bytes);
        if (!bytes || addr > std::numeric_limits<uint64_t>::max() - bytes) return copied;
        const uint64_t end = addr + bytes;
        for (const auto& copy : state.dma_copies) {
            std::vector<uint8_t> source;
            if (read_live_render_target_bytes(copy.src, copy.bytes, source) !=
                    LiveTargetByteReadResult::Success || source.size() != copy.bytes)
                continue;
            const uint64_t copy_end = copy.src + copy.bytes;
            const uint64_t overlap_begin = std::max(addr, copy.src);
            const uint64_t overlap_end = std::min(end, copy_end);
            if (overlap_begin >= overlap_end) continue;
            const size_t destination_offset = static_cast<size_t>(overlap_begin - addr);
            const size_t source_offset = static_cast<size_t>(overlap_begin - copy.src);
            const size_t overlap_bytes = static_cast<size_t>(overlap_end - overlap_begin);
            std::memcpy(destination + destination_offset, source.data() + source_offset,
                        overlap_bytes);
            copied = std::max(copied, destination_offset + overlap_bytes);
        }
        return copied;
    };
}

bool capture_gpustate_submit(const GpuState& state, uint64_t submit_no,
                             uint32_t width, uint32_t height,
                             const GpuCaptureMetadata& metadata,
                             GpuCaptureFile& out, std::string& error,
                             uint64_t resource_limit_bytes) {
    const bool caplog = std::getenv("PROSPER_GPU_CAPTURE_LOG") != nullptr;
    std::vector<OperationRealizationFailure> failures, compute_failures;
    if (caplog) std::fprintf(stderr, "[cap] realize_gpustate_draws...\n");
    std::vector<DrawItem> draws = realize_gpustate_draws(state, 0x10000, 1.0f, 1.0f, &failures);
    if (caplog) std::fprintf(stderr, "[cap] realize draws=%zu; realize_compute_dispatches...\n", draws.size());
    std::vector<ComputeItem> computes = realize_compute_dispatches(state, submit_no, &compute_failures);
    if (caplog) std::fprintf(stderr, "[cap] realize computes=%zu; plan_submit_operations...\n", computes.size());
    failures.insert(failures.end(), std::make_move_iterator(compute_failures.begin()),
                    std::make_move_iterator(compute_failures.end()));
    GpuCaptureMetadata actual = metadata;
    actual.width = width;
    actual.height = height;
    actual.submit_index = submit_no;
    auto ops = plan_submit_operations(state);
    if (caplog) std::fprintf(stderr, "[cap] ops=%zu; capture_submit_items...\n", ops.size());
    // Raw guest memory is normally authoritative. A live renderer-owned target is the exception:
    // ordered DMA reads its host pixels, so overlay those exact bytes into the pre-submit closure.
    CaptureMemoryReader ordered_reader = ordered_gpustate_capture_reader(state);
    return capture_submit_items(draws, computes, ops, actual, ordered_reader, out, error,
                                g_rtt_seed_reader, failures, state.dma_copies,
                                resource_limit_bytes);
}

bool capture_gpustate_target_submit(const GpuState& state, uint64_t submit_no,
                                    uint32_t width, uint32_t height,
                                    uint32_t target_width, uint32_t target_height,
                                    const GpuCaptureMetadata& metadata,
                                    GpuCaptureFile& out, std::string& error) {
    std::vector<DrawItem> draws = realize_gpustate_draws(state);
    draws.erase(std::remove_if(draws.begin(), draws.end(), [&](const DrawItem& draw) {
        return draw.color0_width != target_width || draw.color0_height != target_height;
    }), draws.end());
    std::vector<SubmitOperation> operations;
    operations.reserve(draws.size());
    for (const auto& draw : draws)
        operations.push_back({SubmitOperationKind::Draw, static_cast<size_t>(draw.draw_index),
                              draw.command_order});
    for (size_t i = 0; i < state.dma_copies.size(); ++i)
        operations.push_back({SubmitOperationKind::DmaCopy, i,
                              state.dma_copies[i].command_order});
    std::stable_sort(operations.begin(), operations.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });
    GpuCaptureMetadata actual = metadata;
    actual.width = width; actual.height = height; actual.submit_index = submit_no;
    return capture_submit_items(draws, {}, operations, actual,
                                ordered_gpustate_capture_reader(state),
                                out, error, g_rtt_seed_reader, {}, state.dma_copies);
}

bool serialize_gpu_capture(const GpuCaptureFile& c, std::vector<uint8_t>& bytes, std::string& error) {
    error.clear(); Writer w; w.raw(kMagic, sizeof(kMagic)); w.u32(kVersion); w.u32(kEndian);
    w.u32(c.metadata.width); w.u32(c.metadata.height); w.u64(c.metadata.submit_index);
    w.string(c.metadata.revision); w.string(c.metadata.title_id); w.string(c.metadata.input_route); w.string(c.metadata.savedata_dir);
    w.u32(static_cast<uint32_t>(c.metadata.renderer_env.size()));
    for (const auto& [name, value] : c.metadata.renderer_env) { w.string(name); w.string(value); }
    w.u8(c.expected_output_valid); w.u64(c.expected_output_hash); w.u64(c.expected_output_bytes);
    w.u32(static_cast<uint32_t>(c.blobs.size()));
    for (const auto& b : c.blobs) {
        const uint64_t hash = gpu_capture_hash(b.bytes);
        if (b.content_hash && b.content_hash != hash) { error = "capture blob content hash mismatch"; return false; }
        w.u64(b.guest_addr); w.u64(b.bytes_read); w.u64(hash); w.bytes(b.bytes);
    }
    if (c.rtt_seeds.size() > kMaxResources) { error = "invalid RTT seed count"; return false; }
    uint64_t seed_total = 0; std::unordered_set<uint64_t> seed_addresses;
    w.u32(static_cast<uint32_t>(c.rtt_seeds.size()));
    for (const auto& seed : c.rtt_seeds) {
        if (!validate_rtt_seed(seed, error)) return false;
        if (!seed_addresses.insert(seed.guest_addr).second) { error = "duplicate RTT seed address"; return false; }
        if (seed_total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
            error = "capture RTT seed data exceeds 1 GiB"; return false;
        }
        seed_total += seed.rgba.size();
        w.u64(seed.guest_addr); w.u32(seed.width); w.u32(seed.height);
        w.u32(static_cast<uint32_t>(seed.format)); w.bytes(seed.rgba);
    }
    std::vector<GpuCaptureShaderVersion> versions = c.shader_versions;
    auto add_shader = [&](const std::vector<uint32_t>& words) {
        const uint64_t hash = shader_hash(words);
        auto it = std::find_if(versions.begin(), versions.end(), [&](const auto& version) {
            return version.content_hash == hash && version.words == words;
        });
        if (it == versions.end()) versions.push_back({hash, words});
    };
    for (const auto& d : c.draws) {
        add_shader(d.vs);
        if (!d.gs.empty()) add_shader(d.gs);
        add_shader(d.fs);
    }
    for (const auto& compute : c.computes) add_shader(compute.spirv);
    if (versions.size() > kMaxResources) { error = "invalid shader-version count"; return false; }
    w.u32(static_cast<uint32_t>(versions.size()));
    for (const auto& version : versions) {
        const uint64_t hash = shader_hash(version.words);
        if (version.content_hash && version.content_hash != hash) {
            error = "capture shader content hash mismatch"; return false;
        }
        w.u64(hash); w.words(version.words);
    }
    if (c.draws.size() > kMaxDraws) { error = "invalid draw count"; return false; }
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& d : c.draws) {
        const uint32_t vs_index = shader_version_index(versions, d.vs);
        const uint32_t fs_index = shader_version_index(versions, d.fs);
        if (vs_index == 0xFFFFFFFFu || fs_index == 0xFFFFFFFFu) {
            error = "draw shader is missing from the version table"; return false;
        }
        w.u32(vs_index); w.u32(fs_index);
        write_pipeline(w, d.ps); write_table(w, d.vrt); write_table(w, d.prt);
        w.u32(d.vertex_count); w.words(d.indices); w.u64(d.color0_base);
        w.u32(d.color0_width); w.u32(d.color0_height);
        w.u64(d.draw_index); w.u64(d.command_order);
    }
    if (c.computes.size() > kMaxComputes) { error = "invalid compute count"; return false; }
    w.u32(static_cast<uint32_t>(c.computes.size()));
    for (const auto& compute : c.computes) {
        const uint32_t shader_index = shader_version_index(versions, compute.spirv);
        if (shader_index == 0xFFFFFFFFu) { error = "compute shader is missing from the version table"; return false; }
        w.u32(shader_index); write_table(w, compute.resources);
        w.u32(compute.launch.threads_x); w.u32(compute.launch.threads_y); w.u32(compute.launch.threads_z);
        w.u32(compute.launch.local_x); w.u32(compute.launch.local_y); w.u32(compute.launch.local_z);
        w.u32(compute.launch.groups_x); w.u32(compute.launch.groups_y); w.u32(compute.launch.groups_z);
        w.u64(compute.code_addr); w.u64(compute.dispatch_index); w.u64(compute.submit_no);
        w.u64(compute.command_order);
    }
    if (c.operations.size() > kMaxOperations || !validate_dma_copies(c, error)) {
        if (error.empty()) error = "invalid operation count";
        return false;
    }
    w.u32(static_cast<uint32_t>(c.operations.size()));
    for (const auto& operation : c.operations) {
        w.u8(static_cast<uint8_t>(operation.kind)); w.u64(operation.source_index);
        w.u64(operation.command_order); w.u8(operation.realized);
    }
    if (!validate_failure_diagnostics(c, error)) return false;
    w.u32(static_cast<uint32_t>(c.raw_shader_versions.size()));
    for (const auto& shader : c.raw_shader_versions) {
        w.u64(shader.content_hash); w.u8(shader.has_endpgm); w.words(shader.words);
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u8(static_cast<uint8_t>(diagnostic.kind)); w.u64(diagnostic.source_index);
        w.u64(diagnostic.command_order); w.u8(static_cast<uint8_t>(diagnostic.reason));
        w.u8(diagnostic.pipeline_present);
        if (diagnostic.pipeline_present) write_pipeline(w, diagnostic.pipeline);
        w.u64(diagnostic.color0_base); w.u32(diagnostic.color0_width);
        w.u32(diagnostic.color0_height); w.u32(diagnostic.vertex_count);
        const auto& launch = diagnostic.compute_launch;
        w.u32(launch.threads_x); w.u32(launch.threads_y); w.u32(launch.threads_z);
        w.u32(launch.local_x); w.u32(launch.local_y); w.u32(launch.local_z);
        w.u32(launch.groups_x); w.u32(launch.groups_y); w.u32(launch.groups_z);
        w.u32(static_cast<uint32_t>(diagnostic.stages.size()));
        for (const auto& stage : diagnostic.stages) {
            w.u8(static_cast<uint8_t>(stage.stage)); w.u64(stage.program_addr);
            w.u32(stage.raw_shader_index); w.u8(stage.recompiled);
            w.u8(stage.resource_table_present); w.u32(stage.resource_count);
            w.u32(stage.coverage.total); w.u32(stage.coverage.alu);
            w.u32(stage.coverage.exports); w.u32(stage.coverage.unsupported);
            w.u32(stage.coverage.table_dependent);
            w.u32(static_cast<uint32_t>(stage.coverage.first_bad_fmt));
            w.u32(stage.coverage.first_bad_op); w.u32(stage.coverage.first_bad_pc);
            w.u32(stage.descriptor_issue_count); w.u32(stage.first_descriptor_issue);
        }
    }
    if (c.ds_seeds.size() > kMaxResources) { error = "invalid DS seed count"; return false; }
    uint64_t ds_seed_total = 0;
    std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> ds_seed_keys;
    w.u32(static_cast<uint32_t>(c.ds_seeds.size()));
    for (const auto& seed : c.ds_seeds) {
        if (!validate_ds_seed(seed, error)) return false;
        if (!ds_seed_keys.insert(ds_seed_key(seed)).second) {
            error = "duplicate DS seed identity"; return false;
        }
        const uint64_t plane_bytes = seed.depth.size() + seed.stencil.size();
        if (plane_bytes > kMaxTotalDsSeedBytes ||
            ds_seed_total > kMaxTotalDsSeedBytes - plane_bytes) {
            error = "capture DS seed data exceeds 1 GiB"; return false;
        }
        ds_seed_total += plane_bytes;
        w.u64(seed.depth_read_base); w.u64(seed.depth_write_base);
        w.u64(seed.stencil_read_base); w.u64(seed.stencil_write_base);
        w.u64(seed.htile_data_base); w.u32(seed.width); w.u32(seed.height);
        w.u32(static_cast<uint32_t>(seed.format));
        w.u8(seed.depth_valid); w.u8(seed.stencil_valid);
        w.bytes(seed.depth); w.bytes(seed.stencil);
    }
    // v9 extends the resource contract with the base-level depth of 3D images. Keep the resource
    // record itself byte-compatible with v1-v8 and append depths in deterministic table order, so old
    // captures remain readable without duplicating every legacy table parser.
    uint64_t resource_depth_count = 0;
    for (const auto& draw : c.draws)
        resource_depth_count += draw.vrt.resources.size() + draw.prt.resources.size();
    for (const auto& compute : c.computes) resource_depth_count += compute.resources.resources.size();
    if (resource_depth_count > UINT32_MAX) { error = "invalid resource depth count"; return false; }
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_depths = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) w.u32(captured.resource.depth);
    };
    for (const auto& draw : c.draws) { write_depths(draw.vrt); write_depths(draw.prt); }
    for (const auto& compute : c.computes) write_depths(compute.resources);
    // v10 appends MRT1 state in one deterministic extension, leaving every v1-v9 record prefix byte-
    // compatible. This also keeps old fixture construction simple: remove this tail and lower version.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u64(draw.color1_base); w.u32(draw.color1_width); w.u32(draw.color1_height);
        write_mrt1_pipeline(w, draw.ps);
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u64(diagnostic.color1_base); w.u32(diagnostic.color1_width);
        w.u32(diagnostic.color1_height);
        if (diagnostic.pipeline_present) write_mrt1_pipeline(w, diagnostic.pipeline);
    }
    // v11 preserves GFX10 DCC descriptor state in deterministic resource-table order. Like the v9
    // depth tail, this leaves the v1-v10 resource record byte-compatible while making compressed
    // base allocations explicit to offline inspection. Metadata bytes are not inferred or invented.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_dcc_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const auto& resource = captured.resource;
            if (resource.max_uncompressed_block_size > 3u ||
                resource.max_compressed_block_size > 3u) {
                error = "invalid resource DCC state"; return false;
            }
            const uint8_t flags = (resource.meta_pipe_aligned ? 1u : 0u) |
                                  (resource.write_compress_enabled ? 2u : 0u) |
                                  (resource.compression_enabled ? 4u : 0u) |
                                  (resource.alpha_is_on_msb ? 8u : 0u) |
                                  (resource.color_transform ? 16u : 0u);
            w.u8(flags);
            w.u32(resource.max_uncompressed_block_size);
            w.u32(resource.max_compressed_block_size);
            w.u64(resource.metadata_addr);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_dcc_state(draw.vrt) || !write_dcc_state(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_dcc_state(compute.resources)) return false;
    // v12 captures the exact DCC control surface as a normal content-addressed blob range. Keep the
    // references in a tail so v1-v11 resource records remain byte-compatible. An invalid blob index
    // with a non-zero size is intentional for metadata-only and legacy-upgraded captures.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_dcc_metadata = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const uint64_t expected = dcc_metadata_footprint(captured.resource);
            if (captured.metadata_size != expected ||
                (!expected && (captured.metadata_blob_index != 0xFFFFFFFFu ||
                               captured.metadata_blob_offset != 0)) ||
                (captured.metadata_blob_index == 0xFFFFFFFFu &&
                 captured.metadata_blob_offset != 0)) {
                error = "invalid resource DCC metadata reference"; return false;
            }
            if (captured.metadata_blob_index != 0xFFFFFFFFu) {
                if (captured.metadata_blob_index >= c.blobs.size()) {
                    error = "resource DCC metadata references an invalid capture blob"; return false;
                }
                const auto& capture_blob = c.blobs[captured.metadata_blob_index];
                const auto& blob = capture_blob.bytes;
                if (!capture_blob_payload_omitted(capture_blob) &&
                    (captured.metadata_blob_offset > blob.size() ||
                     expected > blob.size() - captured.metadata_blob_offset)) {
                    error = "resource DCC metadata exceeds its capture blob"; return false;
                }
            }
            w.u64(captured.metadata_size);
            w.u32(captured.metadata_blob_index);
            w.u64(captured.metadata_blob_offset);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_dcc_metadata(draw.vrt) || !write_dcc_metadata(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_dcc_metadata(compute.resources)) return false;
    // v14 appends ordered address-backed DMA_DATA records. Both endpoints reference the same
    // content-addressed pre-submit blobs used by draw/compute resources, so replay mutations are
    // visible to later consumers without changing any v1-v13 prefix.
    w.u32(static_cast<uint32_t>(c.dma_copies.size()));
    for (const auto& copy : c.dma_copies) {
        w.u64(copy.dst); w.u64(copy.src); w.u32(copy.bytes); w.u32(copy.sels);
        w.u64(copy.command_order); w.u64(copy.packet_addr);
        w.u32(copy.destination_blob_index); w.u64(copy.destination_blob_offset);
        w.u32(copy.source_blob_index); w.u64(copy.source_blob_offset);
    }
    // v15 records whether an image is declared for depth comparison. Keep this in a deterministic
    // extension tail, just like the v9 depth and v11/v12 DCC additions, so every older resource
    // record remains byte-compatible.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_depth_compare = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u8(captured.resource.depth_compare);
    };
    for (const auto& draw : c.draws) {
        write_depth_compare(draw.vrt);
        write_depth_compare(draw.prt);
    }
    for (const auto& compute : c.computes) write_depth_compare(compute.resources);
    // v16 records packed GFX10 mip-tail placement. Tail siblings share one captured allocation block,
    // so replay must retain both the common base and the selected in-block byte origin.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_mip_tail = [&](const GpuCapturedTable& table, const char* owner,
                              size_t owner_index) {
        for (size_t resource_index = 0; resource_index < table.resources.size();
             ++resource_index) {
            const auto& captured = table.resources[resource_index];
            const auto& resource = captured.resource;
            if ((resource.in_mip_tail &&
                 (resource.mip_tail_bytes != 4096u && resource.mip_tail_bytes != 65536u)) ||
                (resource.in_mip_tail && resource.mip_tail_offset >= resource.mip_tail_bytes) ||
                (!resource.in_mip_tail &&
                 (resource.mip_tail_offset != 0 || resource.mip_tail_bytes != 0 ||
                  resource.mip_tail_x != 0 || resource.mip_tail_y != 0))) {
                char detail[512]{};
                std::snprintf(detail, sizeof(detail),
                              "invalid resource mip-tail state: %s[%zu] resource[%zu] "
                              "class=%u binding=%u addr=0x%llx size=%u enabled=%u "
                              "offset=%u bytes=%u xy=(%u,%u)",
                              owner, owner_index, resource_index,
                              static_cast<unsigned>(resource.cls), resource.binding,
                              static_cast<unsigned long long>(resource.gpu_addr), resource.size,
                              resource.in_mip_tail ? 1u : 0u, resource.mip_tail_offset,
                              resource.mip_tail_bytes, resource.mip_tail_x,
                              resource.mip_tail_y);
                error = detail;
                return false;
            }
            w.u8(resource.in_mip_tail);
            w.u32(resource.mip_tail_offset);
            w.u32(resource.mip_tail_bytes);
            w.u32(resource.mip_tail_x);
            w.u32(resource.mip_tail_y);
        }
        return true;
    };
    for (size_t draw_index = 0; draw_index < c.draws.size(); ++draw_index) {
        const auto& draw = c.draws[draw_index];
        if (!write_mip_tail(draw.vrt, "draw-vs", draw_index) ||
            !write_mip_tail(draw.prt, "draw-ps", draw_index))
            return false;
    }
    for (size_t compute_index = 0; compute_index < c.computes.size(); ++compute_index)
        if (!write_mip_tail(c.computes[compute_index].resources, "compute", compute_index))
            return false;
    // v17 appends the effective guest scissor for each realized/failed draw pipeline. Keeping this
    // state in a deterministic tail preserves the byte-exact v1-v16 pipeline prefix and lets old
    // captures materialize with the historical full-target default.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) write_scissor_pipeline(w, draw.ps);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) write_scissor_pipeline(w, diagnostic.pipeline);
    // v18 appends the optional generated geometry-stage identity without changing any legacy draw
    // prefix. UINT32_MAX means the normal VS->FS fast path; otherwise the index names the shared
    // content-addressed SPIR-V version, just like the base VS/FS indices.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        const uint32_t geometry_index = draw.gs.empty()
            ? 0xFFFFFFFFu : shader_version_index(versions, draw.gs);
        if (!draw.gs.empty() && geometry_index == 0xFFFFFFFFu) {
            error = "draw geometry shader is missing from the version table"; return false;
        }
        w.u32(geometry_index);
    }
    // v19 retains the raw RDNA2 identity for both realized graphics stages. The raw streams share
    // the bounded content-addressed table introduced for v7 failure diagnostics; old captures leave
    // both indices unavailable instead of inventing source bytes.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u32(draw.vs_raw_shader_index);
        w.u32(draw.fs_raw_shader_index);
    }
    // v20 retains the hardware instance count per realized draw. Older captures default to one,
    // matching the command-processor and Vulkan defaults before IT_NUM_INSTANCES was consumed.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) w.u32(draw.instance_count);
    // v21 appends the resolved framebuffer logic operation for realized and failed draw pipelines.
    // Older captures retain the historical disabled/COPY default.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) write_logic_op_pipeline(w, draw.ps);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) write_logic_op_pipeline(w, diagnostic.pipeline);
    // v22 explicitly captures host-backed compute GDS state. Unlike guest-addressed resources,
    // this state has no valid capture interval; one shared replay instance preserves dispatch order.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_internal_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const bool internal = is_compute_internal_gds(captured.resource);
            if ((!internal && !captured.internal_bytes.empty()) ||
                (internal && !captured.internal_bytes.empty() &&
                 captured.internal_bytes.size() != resource_footprint(captured.resource))) {
                error = "invalid compute GDS capture state";
                return false;
            }
            w.bytes(captured.internal_bytes);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_internal_state(draw.vrt) || !write_internal_state(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_internal_state(compute.resources)) return false;
    // v23 (#1256) appends the raw draw-packet state (DrawIndexAuto/DrawIndex index_count + indexed flag)
    // per realized draw, decoded from the guest BEFORE realization. Older captures default to 0/false
    // ("unknown"). Kept as a trailing block so every v1-v22 record prefix stays byte-exact.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) { w.u32(draw.raw_draw_count); w.u32(draw.raw_indexed ? 1u : 0u); }
    // v24 (#1280) appends each resource's T#-declared mip-chain length (declared_mip_levels), used by the
    // backend to bound generated-mip uploads (#1272). Kept as a trailing per-resource block (same resource
    // enumeration as the v16 mip-tail tail) so every v1-v23 prefix stays byte-exact and older captures
    // materialize with the historical default of 1.
    {
        size_t resource_count = 0;
        for (const auto& draw : c.draws)
            resource_count += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            resource_count += compute.resources.resources.size();
        w.u32(static_cast<uint32_t>(resource_count));
        auto write_declared_mips = [&](const GpuCapturedTable& table) {
            for (const auto& captured : table.resources) w.u32(captured.resource.declared_mip_levels);
        };
        for (const auto& draw : c.draws) { write_declared_mips(draw.vrt); write_declared_mips(draw.prt); }
        for (const auto& compute : c.computes) write_declared_mips(compute.resources);
    }
    // v25 (#1240) appends the resolved MODE=3 MSAA-copy intent. gpu_replay drives the same renderer
    // callback as live execution; without this bit it silently skips color0 -> color1 resolves.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) w.u8(draw.ps.cb_resolve ? 1u : 0u);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) w.u8(diagnostic.pipeline.cb_resolve ? 1u : 0u);
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u32(draw.ps.spi_shader_col_format);
        w.u32(draw.ps.sx_ps_downconvert);
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present) {
        w.u32(diagnostic.pipeline.spi_shader_col_format);
        w.u32(diagnostic.pipeline.sx_ps_downconvert);
    }
    // v27: raw draw modifier plus GE_INDX_OFFSET-derived Vulkan draw parameter. Keep this in a
    // trailing block so v1-v26 prefixes remain byte-exact and older captures default both to zero.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u64(draw.raw_draw_modifier);
        w.u32(static_cast<uint32_t>(draw.vertex_offset));
    }
    // v28: exact source row pitch and capture-planned byte span for each resource. A zero pitch remains
    // meaningful for resources that are not linear sampled images; enumeration matches v16/v24.
    {
        size_t resource_count = 0;
        for (const auto& draw : c.draws)
            resource_count += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            resource_count += compute.resources.resources.size();
        w.u32(static_cast<uint32_t>(resource_count));
        auto write_linear_layout = [&](const GpuCapturedTable& table) {
            for (const auto& captured : table.resources) {
                ShaderResource resource = captured.resource;
                if (c.format_version < 28 && !resource.linear_row_pitch_bytes &&
                    resource.cls == ResourceClass::Texture && resource.img_dim == 1u &&
                    resource.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                    !resource.compression_enabled && !bc_block_bytes(resource.format)) {
                    uint32_t bpt = data_format_bytes(resource.format) *
                                   (resource.num_components ? resource.num_components : 1u);
                    const bool f16 = resource.format == DataFormat::Float16 &&
                                     (bpt == 2 || bpt == 4 || bpt == 8);
                    if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
                    resource.linear_row_pitch_bytes = resolved_linear_row_pitch(
                        resource, resource.width ? resource.width : 4u, bpt);
                }
                w.u32(resource.linear_row_pitch_bytes);
                w.u64(captured.captured_size ? captured.captured_size
                                             : (c.format_version < 28
                                                    ? legacy_resource_footprint(resource)
                                                    : resource_footprint(resource)));
            }
        };
        for (const auto& draw : c.draws) {
            write_linear_layout(draw.vrt);
            write_linear_layout(draw.prt);
        }
        for (const auto& compute : c.computes) write_linear_layout(compute.resources);
    }
    // v29 (#1349) appends the resolved depth-bias state per realized/failed pipeline. Shadow-map
    // passes program PA_SU_POLY_OFFSET_*; replaying without it re-introduces the acne the live
    // renderer now avoids.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u8(draw.ps.depth_bias_enable ? 1u : 0u);
        w.u32(std::bit_cast<uint32_t>(draw.ps.depth_bias_constant));
        w.u32(std::bit_cast<uint32_t>(draw.ps.depth_bias_slope));
        w.u32(std::bit_cast<uint32_t>(draw.ps.depth_bias_clamp));
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present) {
        w.u8(diagnostic.pipeline.depth_bias_enable ? 1u : 0u);
        w.u32(std::bit_cast<uint32_t>(diagnostic.pipeline.depth_bias_constant));
        w.u32(std::bit_cast<uint32_t>(diagnostic.pipeline.depth_bias_slope));
        w.u32(std::bit_cast<uint32_t>(diagnostic.pipeline.depth_bias_clamp));
    }

    // v30 appends the dynamic-fold-proven fetch index source in deterministic resource order. NGG
    // merged shaders can select instance_id for one V# and vertex_id for another; replay must compile
    // the same address model as the live draw.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_fetch_index_modes = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u32(static_cast<uint32_t>(captured.resource.fetch_index_mode));
    };
    for (const auto& draw : c.draws) {
        write_fetch_index_modes(draw.vrt);
        write_fetch_index_modes(draw.prt);
    }
    for (const auto& compute : c.computes) write_fetch_index_modes(compute.resources);
    // v31 retains a separately-installed NGG vertex main stage and the exact graphics-LDS
    // allocation. This lets diagnostic replay reconstruct linked prolog+main programs instead of
    // incorrectly probing the fetch prolog alone. Older captures keep both fields unavailable.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u32(draw.vs_chain_raw_shader_index);
        w.u32(draw.vertex_lds_dwords);
    }
    // v32 keeps the selected mip's placement inside every thin-2D array or cube slice. This is a
    // trailing block so v1-v31 records remain byte-exact and old captures retain their tight-layer
    // default.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_layer_layout = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const ShaderResource& resource = captured.resource;
            const bool enabled = resource.layer_stride_bytes != 0;
            if ((!enabled && resource.layer_mip_offset_bytes != 0) ||
                (enabled && ((resource.img_dim != 3 && resource.img_dim != 5) ||
                             resource.depth < 2 ||
                             resource.layer_mip_offset_bytes >= resource.layer_stride_bytes ||
                             (resource.in_mip_tail && resource.layer_mip_offset_bytes != 0)))) {
                error = "invalid resource layered mip layout";
                return false;
            }
            w.u32(resource.layer_stride_bytes);
            w.u32(resource.layer_mip_offset_bytes);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_layer_layout(draw.vrt) || !write_layer_layout(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_layer_layout(compute.resources)) return false;
    // v34 retains all hardware color-buffer slots. MRT0/MRT1 keep their byte-exact historical
    // records; this tail adds slots 2..7 so live and standalone replay execute the same deferred
    // rendering graph instead of silently discarding G-buffer exports.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
            const auto& binding = draw.color_targets[slot];
            w.u64(binding.base); w.u32(binding.width); w.u32(binding.height);
            write_color_target_pipeline(w, draw.ps.color_targets[slot]);
        }
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
            const auto& binding = diagnostic.color_targets[slot];
            w.u64(binding.base); w.u32(binding.width); w.u32(binding.height);
        }
        if (diagnostic.pipeline_present)
            for (uint32_t slot = 2; slot < kColorTargetCount; ++slot)
                write_color_target_pipeline(w, diagnostic.pipeline.color_targets[slot]);
    }
    // v35 keeps failed-stage descriptor metadata in a self-contained tail. The ordinary table
    // prefix already includes every field the recompiler uses for binding/provenance decisions;
    // blob references are intentionally unset because a failed stage is retried, not rendered.
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u32(static_cast<uint32_t>(diagnostic.stages.size()));
        for (const auto& stage : diagnostic.stages) write_table(w, stage.resource_table);
    }
    // v36 keeps the exact pixel-stage ABI used for the stored modules. Fields are conditional so a
    // draw with no programmed SPI input state remains explicit instead of gaining an invented zero
    // mapping during replay.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        if ((draw.has_pixel_inputs && !draw.pixel_inputs.valid_mask) ||
            (draw.pixel_inputs.passthrough_mask & ~draw.pixel_inputs.valid_mask) ||
            (draw.has_system_inputs && !draw.system_inputs.ena && !draw.system_inputs.addr)) {
            error = "invalid realized-draw pixel-stage ABI";
            return false;
        }
        const uint8_t flags = (draw.has_pixel_inputs ? 1u : 0u) |
                              (draw.has_system_inputs ? 2u : 0u);
        w.u8(flags);
        if (draw.has_pixel_inputs) {
            w.u32(draw.pixel_inputs.valid_mask);
            w.u32(draw.pixel_inputs.passthrough_mask);
            for (uint32_t control : draw.pixel_inputs.controls) w.u32(control);
        }
        if (draw.has_system_inputs) {
            w.u32(draw.system_inputs.ena);
            w.u32(draw.system_inputs.addr);
        }
    }
    // v37 keeps the pipeline-stage contract beside the already-stored native-subgroup module.
    // The realized translator only emits wave32/wave64 modules; reject invented values here so a
    // malformed capsule cannot ask Vulkan for an unrelated device-specific subgroup width.
    w.u32(static_cast<uint32_t>(c.computes.size()));
    for (const auto& compute : c.computes) {
        if (compute.required_subgroup_size != 0 && compute.required_subgroup_size != 32 &&
            compute.required_subgroup_size != 64) {
            error = "invalid compute required-subgroup size";
            return false;
        }
        w.u32(compute.required_subgroup_size);
    }
    if (w.data.size() > kMaxFileBytes) { error = "capture file exceeds 4 GiB"; return false; }
    bytes = std::move(w.data);
    return true;
}

bool write_gpu_capture(const std::string& path, const GpuCaptureFile& c, std::string& error) {
    std::vector<uint8_t> bytes;
    if (!serialize_gpu_capture(c, bytes, error)) return false;
    std::filesystem::path target(path), temp = target; temp += ".tmp";
    std::error_code ec; if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path(), ec);
    std::ofstream f(temp, std::ios::binary | std::ios::trunc);
    if (!f || !f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot write capture temporary file"; return false;
    }
    f.close(); std::filesystem::rename(temp, target, ec);
    if (ec) { std::filesystem::remove(target, ec); ec.clear(); std::filesystem::rename(temp, target, ec); }
    if (ec) { error = "cannot install capture file: " + ec.message(); return false; }
    return true;
}

bool read_gpu_capture(const std::string& path, GpuCaptureFile& c, std::string& error) {
    error.clear(); std::error_code ec; uint64_t size = std::filesystem::file_size(path, ec);
    if (ec || size > kMaxFileBytes || size > std::numeric_limits<size_t>::max()) { error = "invalid capture file size"; return false; }
    std::vector<uint8_t> bytes(static_cast<size_t>(size)); std::ifstream f(path, std::ios::binary);
    if (!f || !f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot read capture file"; return false;
    }
    return deserialize_gpu_capture(bytes, c, error);
}

bool deserialize_gpu_capture(const std::vector<uint8_t>& bytes, GpuCaptureFile& c, std::string& error) {
    error.clear(); c = {};
    Reader r{bytes.data(), bytes.size(), &error}; char magic[8]; uint32_t version, endian;
    if (!r.take(magic, 8)) { error = "truncated capture header"; return false; }
    if (std::memcmp(magic, kMagic, 8)) { error = "invalid capture magic"; return false; }
    if (!r.u32(version)) { error = "truncated capture version"; return false; }
    if (version < 1 || version > kVersion) {
        error = "unsupported capture version " + std::to_string(version);
        return false;
    }
    c.format_version = version;
    if (!r.u32(endian)) { error = "truncated capture byte-order marker"; return false; }
    if (endian != kEndian) { error = "unsupported capture byte order"; return false; }
    auto& m = c.metadata;
    if (!r.u32(m.width) || !r.u32(m.height) || !r.u64(m.submit_index) || !r.string(m.revision) ||
        !r.string(m.title_id) || !r.string(m.input_route) || !r.string(m.savedata_dir)) return false;
    uint32_t ne; if (!r.u32(ne) || ne > 256) { error = "invalid renderer environment count"; return false; }
    m.renderer_env.resize(ne);
    for (auto& [name, value] : m.renderer_env) if (!r.string(name) || !r.string(value)) return false;
    if (version >= 5) {
        uint8_t valid = 0;
        if (!r.u8(valid)) return false;
        c.expected_output_valid = valid != 0;
    } else {
        c.expected_output_valid = true;
    }
    if (!r.u64(c.expected_output_hash) || !r.u64(c.expected_output_bytes)) return false;
    uint32_t nb; if (!r.u32(nb) || nb > kMaxResources) { error = "invalid blob count"; return false; }
    uint64_t total = 0; c.blobs.resize(nb);
    for (auto& b : c.blobs) {
        if (!r.u64(b.guest_addr) || !r.u64(b.bytes_read) ||
            (version >= 5 && !r.u64(b.content_hash)) || !r.bytes(b.bytes)) return false;
        if (b.bytes_read > b.bytes.size() || total > kMaxTotalBlobBytes - b.bytes.size()) { error = "invalid blob metadata"; return false; }
        const uint64_t actual_hash = gpu_capture_hash(b.bytes);
        if (version >= 5 && b.content_hash != actual_hash) { error = "capture blob content hash mismatch"; return false; }
        b.content_hash = actual_hash;
        total += b.bytes.size();
    }
    if (version >= 4) {
        uint32_t ns; if (!r.u32(ns) || ns > kMaxResources) { error = "invalid RTT seed count"; return false; }
        uint64_t seed_total = 0; c.rtt_seeds.resize(ns);
        std::unordered_set<uint64_t> addresses;
        for (auto& seed : c.rtt_seeds) {
            uint32_t format = static_cast<uint32_t>(GpuCaptureColorFormat::Rgba8Unorm);
            if (!r.u64(seed.guest_addr) || !r.u32(seed.width) || !r.u32(seed.height) ||
                (version >= 13 && !r.u32(format)) || !r.bytes(seed.rgba)) return false;
            seed.format = static_cast<GpuCaptureColorFormat>(format);
            if (!validate_rtt_seed(seed, error)) return false;
            if (!addresses.insert(seed.guest_addr).second) { error = "duplicate RTT seed address"; return false; }
            if (seed_total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
                error = "capture RTT seed data exceeds 1 GiB"; return false;
            }
            seed_total += seed.rgba.size();
        }
    }
    if (version >= 5) {
        uint32_t ns = 0;
        if (!r.u32(ns) || ns > kMaxResources) { error = "invalid shader-version count"; return false; }
        c.shader_versions.resize(ns);
        for (auto& shader : c.shader_versions) {
            if (!r.u64(shader.content_hash) || !r.words(shader.words)) return false;
            if (shader.content_hash != shader_hash(shader.words)) {
                error = "capture shader content hash mismatch"; return false;
            }
        }
    }
    uint32_t nd; if (!r.u32(nd) || nd > kMaxDraws || (version < 5 && !nd)) { error = "invalid draw count"; return false; }
    c.draws.resize(nd);
    for (auto& d : c.draws) {
        if (version >= 5) {
            uint32_t vs_index = 0, fs_index = 0;
            if (!r.u32(vs_index) || !r.u32(fs_index) || vs_index >= c.shader_versions.size() ||
                fs_index >= c.shader_versions.size()) { error = "invalid draw shader-version index"; return false; }
            d.vs = c.shader_versions[vs_index].words;
            d.fs = c.shader_versions[fs_index].words;
        } else if (!r.words(d.vs) || !r.words(d.fs)) return false;
        if (!read_pipeline(r, d.ps, version) || !read_table(r, d.vrt, version) ||
            !read_table(r, d.prt, version) || !r.u32(d.vertex_count) || !r.words(d.indices) || !r.u64(d.color0_base)) return false;
        if (version >= 3 && (!r.u32(d.color0_width) || !r.u32(d.color0_height))) return false;
        if (version >= 5 && (!r.u64(d.draw_index) || !r.u64(d.command_order))) return false;
    }
    if (version >= 5) {
        uint32_t nc = 0;
        if (!r.u32(nc) || nc > kMaxComputes) { error = "invalid compute count"; return false; }
        c.computes.resize(nc);
        for (auto& compute : c.computes) {
            uint32_t shader_index = 0;
            if (!r.u32(shader_index) || shader_index >= c.shader_versions.size()) {
                error = "invalid compute shader-version index"; return false;
            }
            compute.spirv = c.shader_versions[shader_index].words;
            if (!read_table(r, compute.resources, version) ||
                !r.u32(compute.launch.threads_x) || !r.u32(compute.launch.threads_y) || !r.u32(compute.launch.threads_z) ||
                !r.u32(compute.launch.local_x) || !r.u32(compute.launch.local_y) || !r.u32(compute.launch.local_z) ||
                !r.u32(compute.launch.groups_x) || !r.u32(compute.launch.groups_y) || !r.u32(compute.launch.groups_z) ||
                !r.u64(compute.code_addr) || !r.u64(compute.dispatch_index) || !r.u64(compute.submit_no) ||
                !r.u64(compute.command_order)) return false;
        }
        uint32_t no = 0;
        if (!r.u32(no) || no > kMaxOperations) { error = "invalid operation count"; return false; }
        c.operations.resize(no);
        for (auto& operation : c.operations) {
            uint8_t kind = 0, realized = 0;
            const uint8_t max_kind = static_cast<uint8_t>(version >= 14
                ? SubmitOperationKind::DmaCopy : SubmitOperationKind::Dispatch);
            if (!r.u8(kind) || kind > max_kind ||
                !r.u64(operation.source_index) || !r.u64(operation.command_order) || !r.u8(realized)) return false;
            operation.kind = static_cast<SubmitOperationKind>(kind);
            operation.realized = realized != 0;
        }
    } else {
        collect_shader_versions(c);
        for (size_t i = 0; i < c.draws.size(); ++i) {
            c.draws[i].draw_index = i;
            c.operations.push_back({SubmitOperationKind::Draw, i, c.draws[i].command_order, true});
        }
    }
    if (version >= 7) {
        c.failure_diagnostics_available = true;
        uint32_t raw_count = 0;
        if (!r.u32(raw_count) || raw_count > kMaxResources) {
            error = "invalid raw shader count";
            return false;
        }
        c.raw_shader_versions.resize(raw_count);
        uint64_t raw_words = 0;
        for (auto& shader : c.raw_shader_versions) {
            uint8_t has_endpgm = 0;
            if (!r.u64(shader.content_hash) || !r.u8(has_endpgm) ||
                !r.words_bounded(shader.words, kMaxRawShaderWords,
                                 "invalid raw shader length")) return false;
            shader.has_endpgm = has_endpgm != 0;
            if (shader.words.empty() || raw_words > kMaxShaderWords - shader.words.size()) {
                error = "raw shader data exceeds its bounded limit";
                return false;
            }
            raw_words += shader.words.size();
        }
        uint32_t diagnostic_count = 0;
        if (!r.u32(diagnostic_count) || diagnostic_count > kMaxOperations) {
            error = "invalid failed-operation diagnostic count";
            return false;
        }
        c.failure_diagnostics.resize(diagnostic_count);
        for (auto& diagnostic : c.failure_diagnostics) {
            uint8_t kind = 0, reason = 0, pipeline_present = 0;
            if (!r.u8(kind) || kind > static_cast<uint8_t>(SubmitOperationKind::Dispatch) ||
                !r.u64(diagnostic.source_index) || !r.u64(diagnostic.command_order) ||
                !r.u8(reason) || reason <= static_cast<uint8_t>(RealizationFailureReason::None) ||
                reason > static_cast<uint8_t>(RealizationFailureReason::Filtered) ||
                !r.u8(pipeline_present)) {
                error = "invalid failed-operation diagnostic metadata";
                return false;
            }
            diagnostic.kind = static_cast<SubmitOperationKind>(kind);
            diagnostic.reason = static_cast<RealizationFailureReason>(reason);
            diagnostic.pipeline_present = pipeline_present != 0;
            if (diagnostic.pipeline_present && !read_pipeline(r, diagnostic.pipeline, version)) return false;
            if (!r.u64(diagnostic.color0_base) || !r.u32(diagnostic.color0_width) ||
                !r.u32(diagnostic.color0_height) || !r.u32(diagnostic.vertex_count)) return false;
            auto& launch = diagnostic.compute_launch;
            if (!r.u32(launch.threads_x) || !r.u32(launch.threads_y) || !r.u32(launch.threads_z) ||
                !r.u32(launch.local_x) || !r.u32(launch.local_y) || !r.u32(launch.local_z) ||
                !r.u32(launch.groups_x) || !r.u32(launch.groups_y) || !r.u32(launch.groups_z)) return false;
            uint32_t stage_count = 0;
            if (!r.u32(stage_count) || stage_count > kMaxFailureStages) {
                error = "invalid failed-stage diagnostic count";
                return false;
            }
            diagnostic.stages.resize(stage_count);
            for (auto& stage : diagnostic.stages) {
                uint8_t stage_kind = 0, recompiled = 0, table_present = 0;
                uint32_t first_bad_fmt = 0;
                if (!r.u8(stage_kind) || stage_kind > static_cast<uint8_t>(ShaderProgramStage::Compute) ||
                    !r.u64(stage.program_addr) || !r.u32(stage.raw_shader_index) ||
                    !r.u8(recompiled) || !r.u8(table_present) || !r.u32(stage.resource_count) ||
                    !r.u32(stage.coverage.total) || !r.u32(stage.coverage.alu) ||
                    !r.u32(stage.coverage.exports) || !r.u32(stage.coverage.unsupported) ||
                    !r.u32(stage.coverage.table_dependent) || !r.u32(first_bad_fmt) ||
                    !r.u32(stage.coverage.first_bad_op) || !r.u32(stage.coverage.first_bad_pc) ||
                    !r.u32(stage.descriptor_issue_count) || !r.u32(stage.first_descriptor_issue)) return false;
                stage.stage = static_cast<ShaderProgramStage>(stage_kind);
                stage.recompiled = recompiled != 0;
                stage.resource_table_present = table_present != 0;
                stage.coverage.first_bad_fmt = static_cast<int32_t>(first_bad_fmt);
            }
        }
    }
    if (version >= 8) {
        uint32_t seed_count = 0;
        if (!r.u32(seed_count) || seed_count > kMaxResources) {
            error = "invalid DS seed count"; return false;
        }
        c.ds_seeds.resize(seed_count);
        uint64_t seed_total = 0;
        std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> keys;
        for (auto& seed : c.ds_seeds) {
            uint32_t format = 0;
            uint8_t depth_valid = 0, stencil_valid = 0;
            if (!r.u64(seed.depth_read_base) || !r.u64(seed.depth_write_base) ||
                !r.u64(seed.stencil_read_base) || !r.u64(seed.stencil_write_base) ||
                !r.u64(seed.htile_data_base) || !r.u32(seed.width) || !r.u32(seed.height) ||
                !r.u32(format) || !r.u8(depth_valid) || !r.u8(stencil_valid) ||
                !r.bytes(seed.depth) || !r.bytes(seed.stencil)) return false;
            if (depth_valid > 1 || stencil_valid > 1) {
                error = "invalid DS seed plane validity"; return false;
            }
            seed.format = static_cast<GpuCaptureDsFormat>(format);
            seed.depth_valid = depth_valid != 0;
            seed.stencil_valid = stencil_valid != 0;
            if (!validate_ds_seed(seed, error)) return false;
            if (!keys.insert(ds_seed_key(seed)).second) {
                error = "duplicate DS seed identity"; return false;
            }
            const uint64_t plane_bytes = seed.depth.size() + seed.stencil.size();
            if (plane_bytes > kMaxTotalDsSeedBytes ||
                seed_total > kMaxTotalDsSeedBytes - plane_bytes) {
                error = "capture DS seed data exceeds 1 GiB"; return false;
            }
            seed_total += plane_bytes;
        }
    }
    if (version >= 9) {
        uint64_t expected_depths = 0;
        for (const auto& draw : c.draws)
            expected_depths += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_depths += compute.resources.resources.size();
        uint32_t depth_count = 0;
        if (!r.u32(depth_count) || depth_count != expected_depths) {
            error = "invalid resource depth count"; return false;
        }
        auto read_depths = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources)
                if (!r.u32(captured.resource.depth) || !captured.resource.depth ||
                    captured.resource.depth > 8192u) {
                    error = "invalid resource depth";
                    return false;
                }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_depths(draw.vrt) || !read_depths(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_depths(compute.resources)) return false;
    }
    if (version >= 10) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid MRT1 draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u64(draw.color1_base) || !r.u32(draw.color1_width) ||
                !r.u32(draw.color1_height) || !read_mrt1_pipeline(r, draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid MRT1 failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            if (!r.u64(diagnostic.color1_base) || !r.u32(diagnostic.color1_width) ||
                !r.u32(diagnostic.color1_height)) return false;
            if (diagnostic.pipeline_present && !read_mrt1_pipeline(r, diagnostic.pipeline)) return false;
        }
    }
    if (version >= 11) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid resource DCC-state count"; return false;
        }
        auto read_dcc_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                auto& resource = captured.resource;
                uint8_t flags = 0;
                if (!r.u8(flags) || (flags & ~0x1fu) ||
                    !r.u32(resource.max_uncompressed_block_size) ||
                    !r.u32(resource.max_compressed_block_size) ||
                    !r.u64(resource.metadata_addr) ||
                    resource.max_uncompressed_block_size > 3u ||
                    resource.max_compressed_block_size > 3u) {
                    error = "invalid resource DCC state"; return false;
                }
                resource.meta_pipe_aligned = (flags & 1u) != 0;
                resource.write_compress_enabled = (flags & 2u) != 0;
                resource.compression_enabled = (flags & 4u) != 0;
                resource.alpha_is_on_msb = (flags & 8u) != 0;
                resource.color_transform = (flags & 16u) != 0;
                captured.metadata_size = dcc_metadata_footprint(resource);
                resource.dcc_metadata_size = captured.metadata_size;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_dcc_state(draw.vrt) || !read_dcc_state(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_dcc_state(compute.resources)) return false;
    }
    if (version >= 12) {
        uint64_t expected_refs = 0;
        for (const auto& draw : c.draws)
            expected_refs += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_refs += compute.resources.resources.size();
        uint32_t ref_count = 0;
        if (!r.u32(ref_count) || ref_count != expected_refs) {
            error = "invalid resource DCC-metadata reference count"; return false;
        }
        auto read_dcc_metadata = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                if (!r.u64(captured.metadata_size) ||
                    !r.u32(captured.metadata_blob_index) ||
                    !r.u64(captured.metadata_blob_offset) ||
                    captured.metadata_size != dcc_metadata_footprint(captured.resource) ||
                    (!captured.metadata_size &&
                     (captured.metadata_blob_index != 0xFFFFFFFFu ||
                      captured.metadata_blob_offset != 0)) ||
                    (captured.metadata_blob_index == 0xFFFFFFFFu &&
                     captured.metadata_blob_offset != 0)) {
                    error = "invalid resource DCC metadata reference"; return false;
                }
                if (captured.metadata_blob_index != 0xFFFFFFFFu) {
                    if (captured.metadata_blob_index >= c.blobs.size()) {
                        error = "resource DCC metadata references an invalid capture blob"; return false;
                    }
                    const auto& capture_blob = c.blobs[captured.metadata_blob_index];
                    const auto& blob = capture_blob.bytes;
                    if (!capture_blob_payload_omitted(capture_blob) &&
                        (captured.metadata_blob_offset > blob.size() ||
                         captured.metadata_size > blob.size() - captured.metadata_blob_offset)) {
                        error = "resource DCC metadata exceeds its capture blob"; return false;
                    }
                }
                captured.resource.dcc_metadata_size = captured.metadata_size;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_dcc_metadata(draw.vrt) || !read_dcc_metadata(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_dcc_metadata(compute.resources)) return false;
    }
    if (version >= 14) {
        uint32_t dma_count = 0;
        if (!r.u32(dma_count) || dma_count > kMaxOperations) {
            error = "invalid ordered DMA count";
            return false;
        }
        c.dma_copies.resize(dma_count);
        for (auto& copy : c.dma_copies) {
            if (!r.u64(copy.dst) || !r.u64(copy.src) || !r.u32(copy.bytes) ||
                !r.u32(copy.sels) || !r.u64(copy.command_order) ||
                !r.u64(copy.packet_addr) || !r.u32(copy.destination_blob_index) ||
                !r.u64(copy.destination_blob_offset) || !r.u32(copy.source_blob_index) ||
                !r.u64(copy.source_blob_offset))
                return false;
        }
        if (!validate_dma_copies(c, error)) return false;
    }
    if (version >= 15) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid resource depth-compare count"; return false;
        }
        auto read_depth_compare = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint8_t enabled = 0;
                if (!r.u8(enabled) || enabled > 1) {
                    error = "invalid resource depth-compare state"; return false;
                }
                captured.resource.depth_compare = enabled != 0;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_depth_compare(draw.vrt) || !read_depth_compare(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_depth_compare(compute.resources)) return false;
    }
    if (version >= 16) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid resource mip-tail count"; return false;
        }
        auto read_mip_tail = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                auto& resource = captured.resource;
                uint8_t enabled = 0;
                if (!r.u8(enabled) || enabled > 1 ||
                    !r.u32(resource.mip_tail_offset) ||
                    !r.u32(resource.mip_tail_bytes) ||
                    !r.u32(resource.mip_tail_x) ||
                    !r.u32(resource.mip_tail_y) ||
                    (enabled && resource.mip_tail_bytes != 4096u &&
                                resource.mip_tail_bytes != 65536u) ||
                    (enabled && resource.mip_tail_offset >= resource.mip_tail_bytes) ||
                    (!enabled && (resource.mip_tail_offset != 0 ||
                                  resource.mip_tail_bytes != 0 ||
                                  resource.mip_tail_x != 0 ||
                                  resource.mip_tail_y != 0))) {
                    error = "invalid resource mip-tail state"; return false;
                }
                resource.in_mip_tail = enabled != 0;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_mip_tail(draw.vrt) || !read_mip_tail(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_mip_tail(compute.resources)) return false;
    }
    if (version >= 17) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid scissor draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!read_scissor_pipeline(r, draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid scissor failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics)
            if (diagnostic.pipeline_present && !read_scissor_pipeline(r, diagnostic.pipeline))
                return false;
    }
    if (version >= 18) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid geometry-stage draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            uint32_t geometry_index = 0;
            if (!r.u32(geometry_index)) return false;
            if (geometry_index == 0xFFFFFFFFu) continue;
            if (geometry_index >= c.shader_versions.size()) {
                error = "invalid draw geometry shader-version index"; return false;
            }
            draw.gs = c.shader_versions[geometry_index].words;
        }
    }
    if (version >= 19) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid realized raw-shader draw count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u32(draw.vs_raw_shader_index) || !r.u32(draw.fs_raw_shader_index))
                return false;
    }
    if (version >= 20) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid draw instance-count state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u32(draw.instance_count)) return false;
    }
    if (version >= 21) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid logic-op draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!read_logic_op_pipeline(r, draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid logic-op failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics)
            if (diagnostic.pipeline_present && !read_logic_op_pipeline(r, diagnostic.pipeline))
                return false;
    }
    if (version >= 22) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid compute GDS state count";
            return false;
        }
        auto read_internal_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                if (!r.bytes(captured.internal_bytes)) return false;
                const bool internal = is_compute_internal_gds(captured.resource);
                if ((!internal && !captured.internal_bytes.empty()) ||
                    (internal && !captured.internal_bytes.empty() &&
                     captured.internal_bytes.size() != resource_footprint(captured.resource))) {
                    error = "invalid compute GDS capture state";
                    return false;
                }
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_internal_state(draw.vrt) || !read_internal_state(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_internal_state(compute.resources)) return false;
    }
    if (version >= 23) {   // #1256: raw draw-packet state per realized draw
        uint32_t nd = 0;
        if (!r.u32(nd) || nd != c.draws.size()) { error = "invalid raw-draw-state count"; return false; }
        for (auto& draw : c.draws) {
            uint32_t ri = 0;
            if (!r.u32(draw.raw_draw_count) || !r.u32(ri)) return false;
            draw.raw_indexed = ri != 0;
        }
    }
    if (version >= 24) {   // #1280: per-resource T#-declared mip-chain length (trailing tail; default 1)
        size_t expected = 0;
        for (auto& draw : c.draws) expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes) expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) { error = "invalid declared-mip-levels count"; return false; }
        auto read_declared_mips = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources)
                if (!r.u32(captured.resource.declared_mip_levels)) return false;
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_declared_mips(draw.vrt) || !read_declared_mips(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_declared_mips(compute.resources)) return false;
    }
    if (version >= 25) {   // #1240: MODE=3 resolve intent for realized and failed pipelines
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid resolve draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            uint8_t enabled = 0;
            if (!r.u8(enabled) || enabled > 1) return false;
            draw.ps.cb_resolve = enabled != 0;
        }
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid resolve failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present) {
            uint8_t enabled = 0;
            if (!r.u8(enabled) || enabled > 1) return false;
            diagnostic.pipeline.cb_resolve = enabled != 0;
        }
    }
    if (version >= 26) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid color-export draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u32(draw.ps.spi_shader_col_format) || !r.u32(draw.ps.sx_ps_downconvert)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid color-export failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present)
            if (!r.u32(diagnostic.pipeline.spi_shader_col_format) ||
                !r.u32(diagnostic.pipeline.sx_ps_downconvert)) return false;
    }
    if (version >= 27) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid draw-parameter state count"; return false;
        }
        for (auto& draw : c.draws) {
            uint32_t vertex_offset = 0;
            if (!r.u64(draw.raw_draw_modifier) || !r.u32(vertex_offset)) return false;
            draw.vertex_offset = static_cast<int32_t>(vertex_offset);
        }
    }
    if (version >= 28) {
        size_t expected = 0;
        for (auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes)
            expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid linear-row-pitch count"; return false;
        }
        auto read_linear_layout = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                if (!r.u32(captured.resource.linear_row_pitch_bytes) ||
                    !r.u64(captured.captured_size)) return false;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_linear_layout(draw.vrt) || !read_linear_layout(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_linear_layout(compute.resources)) return false;
    }
    if (version >= 29) {   // #1349: resolved depth-bias state for realized and failed pipelines
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid depth-bias draw-state count"; return false;
        }
        auto read_bias = [&](ResolvedPipelineState& pipeline) {
            uint8_t enabled = 0;
            uint32_t constant = 0, slope = 0, clamp = 0;
            if (!r.u8(enabled) || enabled > 1 || !r.u32(constant) || !r.u32(slope) ||
                !r.u32(clamp)) return false;
            pipeline.depth_bias_enable   = enabled;
            pipeline.depth_bias_constant = std::bit_cast<float>(constant);
            pipeline.depth_bias_slope    = std::bit_cast<float>(slope);
            pipeline.depth_bias_clamp    = std::bit_cast<float>(clamp);
            return true;
        };
        for (auto& draw : c.draws) if (!read_bias(draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid depth-bias failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present)
            if (!read_bias(diagnostic.pipeline)) return false;
    }
    if (version >= 30) {   // dynamic-fold-proven buffer fetch index source
        size_t expected = 0;
        for (auto& draw : c.draws) expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes) expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid fetch-index-mode count"; return false;
        }
        auto read_fetch_index_modes = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint32_t mode = 0;
                if (!r.u32(mode) || mode > static_cast<uint32_t>(VertexFetchIndexMode::Instance))
                    return false;
                captured.resource.fetch_index_mode = static_cast<VertexFetchIndexMode>(mode);
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_fetch_index_modes(draw.vrt) || !read_fetch_index_modes(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_fetch_index_modes(compute.resources)) return false;
    }
    if (version >= 31) {   // linked NGG vertex main stage + graphics LDS
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid linked-vertex draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            if (!r.u32(draw.vs_chain_raw_shader_index) || !r.u32(draw.vertex_lds_dwords))
                return false;
            if (draw.vertex_lds_dwords > 16384u) {
                error = "invalid linked-vertex LDS allocation"; return false;
            }
        }
    }
    if (version >= 32) {   // selected mip placement inside each thin-2D array or cube slice
        size_t expected = 0;
        for (auto& draw : c.draws) expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes) expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid resource layered layout count"; return false;
        }
        auto read_layer_layout = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                ShaderResource& resource = captured.resource;
                if (!r.u32(resource.layer_stride_bytes) ||
                    !r.u32(resource.layer_mip_offset_bytes))
                    return false;
                const bool enabled = resource.layer_stride_bytes != 0;
                if ((!enabled && resource.layer_mip_offset_bytes != 0) ||
                    (enabled && ((resource.img_dim != 3 && resource.img_dim != 5) ||
                                 resource.depth < 2 ||
                                 resource.layer_mip_offset_bytes >= resource.layer_stride_bytes ||
                                 (resource.in_mip_tail && resource.layer_mip_offset_bytes != 0)))) {
                    error = "invalid resource layered mip layout";
                    return false;
                }
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_layer_layout(draw.vrt) || !read_layer_layout(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_layer_layout(compute.resources)) return false;
    }
    if (version >= 34) {   // complete hardware color-target bindings and pipeline state
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid complete-MRT draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
                auto& binding = draw.color_targets[slot];
                if (!r.u64(binding.base) || !r.u32(binding.width) || !r.u32(binding.height) ||
                    !read_color_target_pipeline(r, draw.ps.color_targets[slot]))
                    return false;
            }
        }
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid complete-MRT failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
                auto& binding = diagnostic.color_targets[slot];
                if (!r.u64(binding.base) || !r.u32(binding.width) || !r.u32(binding.height))
                    return false;
            }
            if (diagnostic.pipeline_present)
                for (uint32_t slot = 2; slot < kColorTargetCount; ++slot)
                    if (!read_color_target_pipeline(
                            r, diagnostic.pipeline.color_targets[slot])) return false;
        }
    }
    if (version >= 35) {   // exact failed-stage resource-descriptor metadata
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid failed-stage resource-table failure count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            uint32_t stage_count = 0;
            if (!r.u32(stage_count) || stage_count != diagnostic.stages.size()) {
                error = "invalid failed-stage resource-table stage count"; return false;
            }
            for (auto& stage : diagnostic.stages)
                if (!read_table(r, stage.resource_table, version)) return false;
        }
    }
    if (version >= 36) {   // exact resolved pixel-stage ABI for raw graphics replay
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid pixel-stage ABI draw count"; return false;
        }
        for (auto& draw : c.draws) {
            uint8_t flags = 0;
            if (!r.u8(flags) || (flags & ~3u)) {
                error = "invalid realized-draw pixel-stage ABI flags"; return false;
            }
            draw.has_pixel_inputs = (flags & 1u) != 0;
            draw.has_system_inputs = (flags & 2u) != 0;
            if (draw.has_pixel_inputs) {
                if (!r.u32(draw.pixel_inputs.valid_mask) ||
                    !r.u32(draw.pixel_inputs.passthrough_mask)) return false;
                for (uint32_t& control : draw.pixel_inputs.controls)
                    if (!r.u32(control)) return false;
                if (!draw.pixel_inputs.valid_mask ||
                    (draw.pixel_inputs.passthrough_mask & ~draw.pixel_inputs.valid_mask)) {
                    error = "invalid realized-draw pixel input mapping"; return false;
                }
            }
            if (draw.has_system_inputs) {
                if (!r.u32(draw.system_inputs.ena) || !r.u32(draw.system_inputs.addr)) return false;
                if (!draw.system_inputs.ena && !draw.system_inputs.addr) {
                    error = "invalid realized-draw pixel system inputs"; return false;
                }
            }
        }
    }
    if (version >= 37) {   // exact required-size/full-subgroups compute pipeline contract
        uint32_t compute_count = 0;
        if (!r.u32(compute_count) || compute_count != c.computes.size()) {
            error = "invalid compute subgroup-contract count"; return false;
        }
        for (auto& compute : c.computes) {
            if (!r.u32(compute.required_subgroup_size)) return false;
            if (compute.required_subgroup_size != 0 && compute.required_subgroup_size != 32 &&
                compute.required_subgroup_size != 64) {
                error = "invalid compute required-subgroup size"; return false;
            }
        }
    }
    for (auto& draw : c.draws) restore_legacy_color_target_aliases(draw);
    for (auto& diagnostic : c.failure_diagnostics)
        restore_legacy_color_target_aliases(diagnostic);
    if (version >= 7 && !validate_failure_diagnostics(c, error)) return false;
    if (r.left) { error = "capture has trailing data"; return false; }
    return true;
}

bool materialize_gpu_replay(const GpuCaptureFile& c, GpuReplayFrame& out, std::string& error) {
    error.clear();
    if (!validate_dma_copies(c, error)) return false;
    out = {}; out.metadata = c.metadata; out.blobs = c.blobs;
    out.rtt_seeds = c.rtt_seeds; out.ds_seeds = c.ds_seeds;
    out.raw_shader_versions = c.raw_shader_versions;
    out.failure_diagnostics = c.failure_diagnostics;
    out.failure_diagnostics_available = c.failure_diagnostics_available;
    out.expected_output_valid = c.expected_output_valid;
    out.expected_output_hash = c.expected_output_hash; out.expected_output_bytes = c.expected_output_bytes;
    size_t resource_reference_count = 0;
    for (const auto& draw : c.draws)
        resource_reference_count += draw.vrt.resources.size() + draw.prt.resources.size();
    for (const auto& compute : c.computes)
        resource_reference_count += compute.resources.resources.size();
    resource_reference_count += c.dma_copies.size() * 2;
    out.resource_instances.reserve(resource_reference_count * 2);
    std::map<std::pair<uint32_t, uint64_t>, size_t> instance_by_version_and_base;
    std::map<uint32_t, size_t> internal_instance_by_binding;
    auto bind_range = [&](uint32_t blob_index, uint64_t blob_offset, uint64_t guest_addr,
                          uint64_t need, uint8_t*& host_data, uint64_t& host_data_size,
                          const char* invalid_error, const char* exceeds_error,
                          const char* offset_error) {
        if (blob_index == 0xFFFFFFFFu) return true;
        if (blob_index >= out.blobs.size() || blob_offset > out.blobs[blob_index].bytes.size()) {
            error = invalid_error; return false;
        }
        const auto& blob = out.blobs[blob_index].bytes;
        if (need > blob.size() - blob_offset) { error = exceeds_error; return false; }
        if (blob_offset > guest_addr) { error = offset_error; return false; }
        const uint64_t logical_base = guest_addr - blob_offset;
        const auto key = std::make_pair(blob_index, logical_base);
        auto [it, inserted] = instance_by_version_and_base.emplace(key, out.resource_instances.size());
        if (inserted)
            out.resource_instances.push_back({logical_base, blob_index, blob});
        auto& instance = out.resource_instances[it->second].bytes;
        host_data = instance.data() + blob_offset;
        host_data_size = need;
        return true;
    };
    auto table = [&](const GpuCapturedTable& src, std::shared_ptr<ShaderResourceTable>& dst) -> bool {
        if (!src.present) { dst.reset(); return src.resources.empty(); }
        dst = std::make_shared<ShaderResourceTable>();
        for (const auto& x : src.resources) {
            ShaderResource r = x.resource; r.host_data = nullptr; r.host_data_size = 0;
            const uint64_t captured_footprint = x.captured_size ? x.captured_size
                : (c.format_version >= 28 ? resource_footprint(r) : legacy_resource_footprint(r));
            // v1-v27 captured linear images tightly. Preserve that bounded backing so old files stay
            // readable, but derive the real guest pitch for best-effort replay of every retained row.
            if (c.format_version < 28 && r.cls == ResourceClass::Texture && r.img_dim == 1u &&
                r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                !r.compression_enabled && !bc_block_bytes(r.format)) {
                uint32_t bpt = data_format_bytes(r.format) *
                               (r.num_components ? r.num_components : 1u);
                const bool f16 = r.format == DataFormat::Float16 &&
                                 (bpt == 2 || bpt == 4 || bpt == 8);
                if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
                r.linear_row_pitch_bytes = resolved_linear_row_pitch(
                    r, r.width ? r.width : 4u, bpt);
            }
            r.dcc_metadata_size = x.metadata_size;
            r.dcc_metadata_host_data = nullptr;
            r.dcc_metadata_host_data_size = 0;
            if (!x.internal_bytes.empty()) {
                auto [it, inserted] = internal_instance_by_binding.emplace(
                    r.binding, out.resource_instances.size());
                if (inserted)
                    out.resource_instances.push_back({0, 0xFFFFFFFFu, x.internal_bytes});
                auto& instance = out.resource_instances[it->second].bytes;
                if (!inserted && instance != x.internal_bytes) {
                    error = "compute GDS resources disagree on initial state";
                    return false;
                }
                r.host_data = instance.data();
                r.host_data_size = instance.size();
            } else if (!bind_range(x.blob_index, x.blob_offset, r.gpu_addr, captured_footprint,
                            r.host_data, r.host_data_size,
                            "resource references an invalid capture blob",
                            "resource footprint exceeds capture blob",
                            "resource blob offset exceeds its logical address")) return false;
            if (!bind_range(x.metadata_blob_index, x.metadata_blob_offset, r.metadata_addr,
                            x.metadata_size, r.dcc_metadata_host_data,
                            r.dcc_metadata_host_data_size,
                            "resource DCC metadata references an invalid capture blob",
                            "resource DCC metadata exceeds capture blob",
                            "resource DCC metadata blob offset exceeds its logical address"))
                return false;
            dst->resources.push_back(r);
        }
        return true;
    };
    out.items.reserve(c.draws.size());
    for (const auto& x : c.draws) {
        DrawItem d; d.vs = x.vs; d.gs = x.gs; d.fs = x.fs;
        d.ps = x.ps; d.vertex_count = x.vertex_count;
        d.instance_count = x.instance_count;
        d.raw_draw_count = x.raw_draw_count; d.raw_indexed = x.raw_indexed;   // #1256
        d.raw_draw_modifier = x.raw_draw_modifier;
        d.vertex_offset = x.vertex_offset;
        d.indices = x.indices; d.color0_base = x.color0_base;
        d.color0_width = x.color0_width; d.color0_height = x.color0_height;
        d.color1_base = x.color1_base;
        d.color1_width = x.color1_width; d.color1_height = x.color1_height;
        d.color_targets = x.color_targets;
        d.color_targets[0] = {d.color0_base, d.color0_width, d.color0_height};
        d.color_targets[1] = {d.color1_base, d.color1_width, d.color1_height};
        d.draw_index = x.draw_index; d.command_order = x.command_order;
        d.vs_raw_shader_index = x.vs_raw_shader_index;
        d.fs_raw_shader_index = x.fs_raw_shader_index;
        d.vs_chain_raw_shader_index = x.vs_chain_raw_shader_index;
        d.vertex_lds_dwords = x.vertex_lds_dwords;
        d.pixel_inputs = x.pixel_inputs;
        d.system_inputs = x.system_inputs;
        d.has_pixel_inputs = x.has_pixel_inputs;
        d.has_system_inputs = x.has_system_inputs;
        if (!table(x.vrt, d.vrt) || !table(x.prt, d.prt)) return false;
        if (d.vrt) d.vrt->vertices_per_instance = d.vertex_count;
        out.items.push_back(std::move(d));
    }
    out.computes.reserve(c.computes.size());
    for (const auto& x : c.computes) {
        ComputeItem compute;
        compute.spirv = x.spirv;
        compute.launch = x.launch;
        compute.code_addr = x.code_addr;
        compute.dispatch_index = x.dispatch_index;
        compute.submit_no = x.submit_no;
        compute.command_order = x.command_order;
        compute.required_subgroup_size = x.required_subgroup_size;
        if (!table(x.resources, compute.resources)) return false;
        out.computes.push_back(std::move(compute));
    }
    out.dma_copies.reserve(c.dma_copies.size());
    for (const auto& captured : c.dma_copies) {
        ReplayDmaCopy copy;
        copy.dst = captured.dst; copy.src = captured.src; copy.bytes = captured.bytes;
        copy.sels = captured.sels; copy.command_order = captured.command_order;
        copy.packet_addr = captured.packet_addr;
        uint8_t* source = nullptr;
        if (!bind_range(captured.destination_blob_index, captured.destination_blob_offset,
                        captured.dst, captured.bytes, copy.destination_data,
                        copy.destination_size,
                        "ordered DMA destination references an invalid capture blob",
                        "ordered DMA destination exceeds capture blob",
                        "ordered DMA destination blob offset exceeds its logical address") ||
            !bind_range(captured.source_blob_index, captured.source_blob_offset,
                        captured.src, captured.bytes, source, copy.source_size,
                        "ordered DMA source references an invalid capture blob",
                        "ordered DMA source exceeds capture blob",
                        "ordered DMA source blob offset exceeds its logical address"))
            return false;
        copy.source_data = source;
        out.dma_copies.push_back(copy);
    }
    out.operations = c.operations;
    return true;
}

void set_gpu_capture_rtt_seed_reader(CaptureRttSeedReader reader) { g_rtt_seed_reader = std::move(reader); }
bool read_gpu_capture_rtt_seed(uint64_t guest_addr, GpuCaptureRttSeed& seed, std::string& error) {
    error.clear();
    if (!g_rtt_seed_reader) { error = "live renderer has no RTT seed reader"; return false; }
    if (!g_rtt_seed_reader(guest_addr, seed)) { error = "render target is absent from live cache"; return false; }
    return validate_rtt_seed(seed, error);
}
bool capture_gpu_rtt_seed(GpuCaptureFile& capture, uint64_t guest_addr, std::string& error) {
    error.clear();
    if (!guest_addr) return true;
    if (std::any_of(capture.rtt_seeds.begin(), capture.rtt_seeds.end(),
                    [&](const GpuCaptureRttSeed& seed) {
                        return seed.guest_addr == guest_addr;
                    }))
        return true;
    GpuCaptureRttSeed seed;
    if (!read_gpu_capture_rtt_seed(guest_addr, seed, error)) return false;
    capture.rtt_seeds.push_back(std::move(seed));
    return true;
}
void set_gpu_capture_rtt_seed_snapshot_reader(CaptureRttSeedSnapshotReader reader) {
    g_rtt_seed_snapshot_reader = std::move(reader);
}
bool read_all_gpu_capture_rtt_seeds(std::vector<GpuCaptureRttSeed>& seeds, std::string& error) {
    error.clear(); seeds.clear();
    if (!g_rtt_seed_snapshot_reader) {
        error = "live renderer has no RTT seed snapshot reader"; return false;
    }
    if (!g_rtt_seed_snapshot_reader(seeds, error)) return false;
    if (seeds.size() > kMaxResources) { error = "invalid RTT seed count"; return false; }
    uint64_t total = 0;
    std::unordered_set<uint64_t> addresses;
    for (const auto& seed : seeds) {
        if (!validate_rtt_seed(seed, error)) return false;
        if (!addresses.insert(seed.guest_addr).second) {
            error = "duplicate RTT seed address"; return false;
        }
        if (seed.rgba.size() > kMaxTotalRttSeedBytes - total) {
            error = "RTT seed bytes exceed limit"; return false;
        }
        total += seed.rgba.size();
    }
    std::sort(seeds.begin(), seeds.end(), [](const auto& a, const auto& b) {
        return a.guest_addr < b.guest_addr;
    });
    return true;
}
void set_gpu_replay_rtt_seed_writer(ReplayRttSeedWriter writer) { g_rtt_seed_writer = std::move(writer); }

bool restore_gpu_replay_rtt_seeds(const std::vector<GpuCaptureRttSeed>& seeds, std::string& error) {
    error.clear();
    if (!seeds.empty() && !g_rtt_seed_writer) { error = "live renderer has no RTT seed writer"; return false; }
    for (const auto& seed : seeds) {
        if (!validate_rtt_seed(seed, error) || !g_rtt_seed_writer(seed, error)) return false;
    }
    return true;
}

void set_gpu_capture_ds_seed_snapshot_reader(CaptureDsSeedSnapshotReader reader) {
    g_ds_seed_snapshot_reader = std::move(reader);
}

bool read_all_gpu_capture_ds_seeds(std::vector<GpuCaptureDsSeed>& seeds, std::string& error) {
    error.clear(); seeds.clear();
    if (!g_ds_seed_snapshot_reader) {
        error = "live renderer has no DS seed snapshot reader"; return false;
    }
    if (!g_ds_seed_snapshot_reader(seeds, error)) return false;
    if (seeds.size() > kMaxResources) { error = "invalid DS seed count"; return false; }
    uint64_t total = 0;
    std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> keys;
    for (const auto& seed : seeds) {
        if (!validate_ds_seed(seed, error)) return false;
        if (!keys.insert(ds_seed_key(seed)).second) {
            error = "duplicate DS seed identity"; return false;
        }
        const uint64_t plane_bytes = seed.depth.size() + seed.stencil.size();
        if (plane_bytes > kMaxTotalDsSeedBytes || total > kMaxTotalDsSeedBytes - plane_bytes) {
            error = "DS seed bytes exceed limit"; return false;
        }
        total += plane_bytes;
    }
    std::sort(seeds.begin(), seeds.end(), [](const auto& a, const auto& b) {
        return ds_seed_key(a) < ds_seed_key(b);
    });
    return true;
}

bool gpu_capture_ds_seed_snapshot_available() {
    return static_cast<bool>(g_ds_seed_snapshot_reader);
}

bool capture_referenced_gpu_ds_seeds(GpuCaptureFile& capture, std::string& error) {
    error.clear();
    if (!capture.ds_seeds.empty()) {
        error = "capture already contains DS checkpoints";
        return false;
    }
    const bool metadata_only = std::any_of(
        capture.metadata.renderer_env.begin(), capture.metadata.renderer_env.end(),
        [](const auto& entry) {
            return entry.first == "PROSPER_GPU_CAPTURE_METADATA_ONLY" &&
                   !entry.second.empty() && entry.second != "0" && entry.second != "off";
        });
    if (metadata_only) return true;

    std::vector<GpuCaptureDsSeed> live;
    if (!read_all_gpu_capture_ds_seeds(live, error)) return false;
    for (auto& seed : live) {
        const bool referenced = std::any_of(
            capture.draws.begin(), capture.draws.end(), [&](const GpuCapturedDraw& draw) {
                const auto& ps = draw.ps;
                const bool uses_ds = ps.depth_test_enable || ps.depth_write_enable ||
                                     ps.depth_clear_enable || ps.stencil_enable ||
                                     ps.stencil_clear_enable;
                return uses_ds && draw.color0_width == seed.width &&
                       draw.color0_height == seed.height &&
                       ps.depth_read_base == seed.depth_read_base &&
                       ps.depth_write_base == seed.depth_write_base &&
                       ps.stencil_read_base == seed.stencil_read_base &&
                       ps.stencil_write_base == seed.stencil_write_base &&
                       ps.htile_data_base == seed.htile_data_base;
            });
        if (referenced) capture.ds_seeds.push_back(std::move(seed));
    }
    return true;
}

void set_gpu_replay_ds_seed_writer(ReplayDsSeedWriter writer) {
    g_ds_seed_writer = std::move(writer);
}

bool restore_gpu_replay_ds_seeds(const std::vector<GpuCaptureDsSeed>& seeds, std::string& error) {
    error.clear();
    if (!seeds.empty() && !g_ds_seed_writer) {
        error = "live renderer has no DS seed writer"; return false;
    }
    std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> keys;
    for (const auto& seed : seeds) {
        if (!validate_ds_seed(seed, error)) return false;
        if (!keys.insert(ds_seed_key(seed)).second) {
            error = "duplicate DS seed identity"; return false;
        }
        if (!g_ds_seed_writer(seed, error)) return false;
    }
    return true;
}

namespace {
// Interactive one-shot capture request (armed by the app hotkey, consumed on the render thread). A
// non-empty path means a grab is armed; the guard makes arm/consume race-free across threads.
std::mutex g_interactive_capture_mx;
std::string g_interactive_capture_path;
std::atomic<bool> g_output_capture_claimed{false};
std::string take_interactive_gpu_capture() {
    std::lock_guard<std::mutex> lk(g_interactive_capture_mx);
    return std::exchange(g_interactive_capture_path, std::string());
}
}  // namespace

void request_interactive_gpu_capture(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_interactive_capture_mx);
    g_interactive_capture_path = path;
}
bool interactive_gpu_capture_armed() {
    std::lock_guard<std::mutex> lk(g_interactive_capture_mx);
    return !g_interactive_capture_path.empty();
}

bool gpu_capture_output_nonzero_matches(const std::vector<uint8_t>& output, size_t min_nonzero,
                                        size_t max_nonzero, size_t* observed) {
    const size_t nonzero = static_cast<size_t>(std::count_if(
        output.begin(), output.end(), [](uint8_t value) { return value != 0; }));
    if (observed) *observed = nonzero;
    return !output.empty() && nonzero >= min_nonzero &&
           (!max_nonzero || nonzero <= max_nonzero);
}

namespace {
bool materialize_pending_gpu_capture(PendingGpuCapture& pending,
                                     const std::vector<DrawItem>& draws,
                                     const std::vector<ComputeItem>& computes,
                                     const std::vector<SubmitOperation>& operations,
                                     const GpuState* semantic_state,
                                     const std::vector<OperationRealizationFailure>* exact_failures,
                                     std::string& error) {
    const GpuCaptureMetadata metadata = pending.capture.metadata;
    const bool has_ordered_dma = semantic_state && !semantic_state->dma_copies.empty();
    bool captured = false;
    if (has_ordered_dma) {
        captured = capture_gpustate_submit(*semantic_state, metadata.submit_index,
                                           metadata.width, metadata.height, metadata,
                                           pending.capture, error);
    } else {
        // Preserve exact realized lists while retaining diagnostics for semantic operations rejected
        // before they reached those lists.
        std::vector<OperationRealizationFailure> failures = exact_failures
            ? *exact_failures : std::vector<OperationRealizationFailure>{};
        if (semantic_state && !exact_failures) {
            std::vector<OperationRealizationFailure> compute_failures;
            (void)realize_gpustate_draws(*semantic_state, 0x10000, 1.0f, 1.0f,
                                         &failures, false, false);
            (void)realize_compute_dispatches(*semantic_state, metadata.submit_index,
                                             &compute_failures);
            failures.insert(failures.end(),
                            std::make_move_iterator(compute_failures.begin()),
                            std::make_move_iterator(compute_failures.end()));
        }
        captured = capture_submit_items(draws, computes, operations, metadata,
                                        read_capture_guest_memory, pending.capture, error,
                                        g_rtt_seed_reader, failures);
    }
    if (!captured) return false;
    pending.materialized = true;
    return !gpu_capture_ds_seed_snapshot_available() ||
           capture_referenced_gpu_ds_seeds(pending.capture, error);
}
}  // namespace

std::unique_ptr<PendingGpuCapture> begin_requested_gpu_capture(
    const std::vector<DrawItem>& draws, const std::vector<ComputeItem>& computes,
    const std::vector<SubmitOperation>& operations, uint32_t width, uint32_t height,
    const GpuState* semantic_state, uint64_t submit_no, uint64_t semantic_draw_count,
    const std::vector<OperationRealizationFailure>* exact_failures,
    bool defer_materialization) {
    const uint64_t candidate_draw_count = semantic_state && semantic_draw_count != UINT64_MAX
        ? semantic_draw_count : static_cast<uint64_t>(draws.size());
    // Interactive one-shot (hotkey): consume the armed request on the first DRAWING invocation so the
    // grab lands on real frame content, and skip the env AT/AFTER/MIN selectors (a live keypress has no
    // submit index to predict). Falls through to the env PROSPER_GPU_CAPTURE path when nothing is armed.
    std::string interactive_path;
    if (candidate_draw_count > 0 && interactive_gpu_capture_armed())
        interactive_path = take_interactive_gpu_capture();
    const bool interactive = !interactive_path.empty();
    const char* env_path = std::getenv("PROSPER_GPU_CAPTURE");
    if (!interactive && (!env_path || !*env_path)) return {};
    const char* output_min_env = std::getenv("PROSPER_GPU_CAPTURE_OUTPUT_NZ_MIN");
    const bool output_triggered = !interactive && output_min_env && *output_min_env;
    if (output_triggered && g_output_capture_claimed.load(std::memory_order_acquire)) return {};
    uint64_t current = 0;
    if (!interactive) {
        static std::atomic<uint64_t> invocation_sequence{0};
        const uint64_t invocation = invocation_sequence.fetch_add(1);
        uint64_t after = 0;
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_AFTER")) after = std::strtoull(v, nullptr, 0);
        if (invocation < after) return {};
        static const auto capture_started = std::chrono::steady_clock::now();
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_AFTER_MS")) {
            const uint64_t after_ms = std::strtoull(v, nullptr, 0);
            const uint64_t elapsed_ms = static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::milliseconds>(std::chrono::steady_clock::now() - capture_started).count());
            if (elapsed_ms < after_ms) return {};
        }
        if (const char* value = std::getenv("PROSPER_GPU_CAPTURE_COMPUTE_ADDR")) {
            char* end = nullptr;
            errno = 0;
            const uint64_t wanted_code_addr = std::strtoull(value, &end, 0);
            if (errno || end == value || !end || *end) return {};
            const bool realized_contains_program = std::any_of(
                computes.begin(), computes.end(), [&](const ComputeItem& compute) {
                    return compute.code_addr == wanted_code_addr;
                });
            const bool semantic_contains_program = semantic_state && std::any_of(
                semantic_state->dispatches.begin(), semantic_state->dispatches.end(),
                [&](const GpuState::Dispatch& dispatch) {
                    return compute_dispatch_code_addr(*semantic_state, dispatch) == wanted_code_addr;
                });
            if (!realized_contains_program && !semantic_contains_program) return {};
        }
        if (const char* value = std::getenv("PROSPER_GPU_CAPTURE_SHADER_ADDR")) {
            char* end = nullptr;
            errno = 0;
            const uint64_t wanted_code_addr = std::strtoull(value, &end, 0);
            if (errno || end == value || !end || *end) return {};
            const bool realized_contains_program = std::any_of(
                draws.begin(), draws.end(), [&](const DrawItem& draw) {
                    return draw.vs_guest_addr == wanted_code_addr ||
                           draw.vs_chain_guest_addr == wanted_code_addr ||
                           draw.fs_guest_addr == wanted_code_addr;
                });
            const bool semantic_contains_program = semantic_state && std::any_of(
                semantic_state->draws.begin(), semantic_state->draws.end(),
                [&](const GpuState::Draw& draw) {
                    const GpuState& draw_state = draw.state ? *draw.state : *semantic_state;
                    const RenderState state = extract_render_state(draw_state);
                    return state.es_addr == wanted_code_addr || state.gs_addr == wanted_code_addr ||
                           state.hs_addr == wanted_code_addr || state.ps_addr == wanted_code_addr;
                });
            if (!realized_contains_program && !semantic_contains_program) return {};
        }
        if (const char* value = std::getenv("PROSPER_GPU_CAPTURE_TARGET_DIM")) {
            uint32_t wanted_width = 0, wanted_height = 0;
            char trailing = 0;
            if (std::sscanf(value, "%ux%u%c", &wanted_width, &wanted_height, &trailing) != 2 ||
                !wanted_width || !wanted_height)
                return {};
            const bool realized_contains_target = std::any_of(
                draws.begin(), draws.end(), [&](const DrawItem& draw) {
                    return draw.color0_width == wanted_width && draw.color0_height == wanted_height;
                });
            const bool semantic_contains_target = semantic_state && std::any_of(
                semantic_state->draws.begin(), semantic_state->draws.end(),
                [&](const GpuState::Draw& draw) {
                    const GpuState& draw_state = draw.state ? *draw.state : *semantic_state;
                    const RenderState state = extract_render_state(draw_state);
                    return state.color0_width == wanted_width &&
                           state.color0_height == wanted_height;
                });
            if (!realized_contains_target && !semantic_contains_target) return {};
        }
        uint64_t min_draws = 0, max_draws = std::numeric_limits<uint64_t>::max();
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_MIN_DRAWS")) min_draws = std::strtoull(v, nullptr, 0);
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_MAX_DRAWS")) max_draws = std::strtoull(v, nullptr, 0);
        if (candidate_draw_count < min_draws || candidate_draw_count > max_draws) return {};
        static std::atomic<uint64_t> sequence{0}; static std::atomic<bool> claimed{false};
        current = sequence.fetch_add(1);
        if (!output_triggered) {
            uint64_t wanted = 0;
            if (const char* at = std::getenv("PROSPER_GPU_CAPTURE_AT")) wanted = std::strtoull(at, nullptr, 0);
            if (current != wanted || claimed.exchange(true)) return {};
        }
    }
    auto pending = std::make_unique<PendingGpuCapture>();
    pending->materialized = false;
    pending->path = interactive ? interactive_path : std::string(env_path);
    pending->output_triggered = output_triggered;
    if (output_triggered) {
        pending->output_min_nonzero = static_cast<size_t>(std::strtoull(output_min_env, nullptr, 0));
        if (const char* max = std::getenv("PROSPER_GPU_CAPTURE_OUTPUT_NZ_MAX"))
            pending->output_max_nonzero = static_cast<size_t>(std::strtoull(max, nullptr, 0));
    }
    GpuCaptureMetadata m; m.width = width; m.height = height;
    m.submit_index = submit_no ? submit_no : current;
#ifdef PROSPER_GIT_REVISION
    m.revision = PROSPER_GIT_REVISION;
#else
    m.revision = "unknown";
#endif
    if (const char* revision = std::getenv("PROSPER_CAPTURE_REVISION")) m.revision = revision;
    m.title_id = env_or_empty("PROSPER_CAPTURE_TITLE"); m.input_route = env_or_empty("PROSPER_PAD_SCRIPT");
    m.savedata_dir = env_or_empty("PROSPER_SAVEDATA_DIR");
    static const char* render_env[] = {
        "PROSPER_ALPHA1", "PROSPER_CLEAR_DEBUG", "PROSPER_DEPTH_ALWAYS", "PROSPER_DETILE",
        "PROSPER_DRAW_ISO", "PROSPER_FLIP_FRONT_FACE", "PROSPER_FS_SPV", "PROSPER_FS_SPV_MATCH",
        "PROSPER_ISO_AT", "PROSPER_ISO_AT2",
        "PROSPER_ISO_RGB", "PROSPER_ISO_TOL", "PROSPER_NO_BLEND", "PROSPER_NO_CULL",
        "PROSPER_NO_DEPTH", "PROSPER_NO_STENCIL", "PROSPER_STENCIL_CLEAR", "PROSPER_STENCIL_REPLACE",
        "PROSPER_NO_SWIZZLE", "PROSPER_NODETILE",
        "PROSPER_DESCRIPTOR_VALIDATE",
        "PROSPER_PITCH", "PROSPER_RENDER_NOPS", "PROSPER_RENDER_REFVS", "PROSPER_RENDER_TESTPS",
        "PROSPER_RENDER_TESTPS_MATCH",
        "PROSPER_RTT", "PROSPER_RTT_NOSEED", "PROSPER_RTT_PERTARGET", "PROSPER_RTT_SINGLE_TARGET",
        "PROSPER_DUMP_RTGROUPS", "PROSPER_DUMP_RTGROUPS_RGBA", "PROSPER_DUMP_RTGROUPS_ADDR",
        "PROSPER_DUMP_DRAWSTEPS",
        "PROSPER_TESTTEX",
        "PROSPER_TESTLUT", "PROSPER_TESTLUT32"
    };
    for (const char* name : render_env) if (const char* value = std::getenv(name)) m.renderer_env.emplace_back(name, value);
    annotate_gpu_capture_scanout(m);
    pending->capture.metadata = m;
    // Output predicates must remain observational: do not read guest resources or renderer caches
    // for rejected candidates. The synchronous executor supplies the still-valid semantic state only
    // after this same submit's pixels match, and materialization happens once for that winner.
    if (output_triggered || defer_materialization) return pending;
    std::string error;
    if (!materialize_pending_gpu_capture(*pending, draws, computes, operations,
                                         semantic_state, exact_failures, error)) {
        std::fprintf(stderr, "[gpucap] capture failed: %s\n", error.c_str()); return {};
    }
    // A normal one-shot capture has the same temporal DS dependency as a timeline endpoint.  The
    // renderer callback runs before this submit is executed, so these are the exact pre-submit
    // planes consumed by depth/stencil reads in the capsule.  Timeline capture has always added
    // them explicitly; omitting them here made F9/environment capsules replay a different frame
    // whenever the selected submit reused persistent depth or stencil from an earlier submit.
    std::fprintf(stderr, "[gpucap] captured %s submit %llu: %zu draws, %zu computes, "
                         "%zu DMA copies, %zu operations, %zu blobs, %zu RTT seeds, "
                         "%zu DS seeds -> %s\n",
                 interactive ? "interactive" : "match",
                 static_cast<unsigned long long>(m.submit_index),
                 pending->capture.draws.size(), pending->capture.computes.size(),
                 pending->capture.dma_copies.size(), pending->capture.operations.size(),
                 pending->capture.blobs.size(),
                 pending->capture.rtt_seeds.size(), pending->capture.ds_seeds.size(),
                 pending->path.c_str());
    return pending;
}

bool finish_requested_gpu_capture(std::unique_ptr<PendingGpuCapture> pending,
                                  const std::vector<uint8_t>& output, std::string& error,
                                  const std::vector<DrawItem>* draws,
                                  const std::vector<ComputeItem>* computes,
                                  const std::vector<SubmitOperation>* operations,
                                  const GpuState* semantic_state,
                                  const std::vector<OperationRealizationFailure>* exact_failures) {
    if (!pending) return true;
    if (pending->output_triggered) {
        size_t nonzero = 0;
        if (!gpu_capture_output_nonzero_matches(output, pending->output_min_nonzero,
                                                pending->output_max_nonzero, &nonzero)) {
            std::fprintf(stderr,
                         "[gpucap] output candidate submit %llu rejected: nonzero=%zu range=%zu..%zu\n",
                         static_cast<unsigned long long>(pending->capture.metadata.submit_index), nonzero,
                         pending->output_min_nonzero, pending->output_max_nonzero);
            return true;
        }
        if (g_output_capture_claimed.load(std::memory_order_acquire)) return true;
        std::fprintf(stderr,
                     "[gpucap] output trigger matched submit %llu: nonzero=%zu range=%zu..%zu\n",
                     static_cast<unsigned long long>(pending->capture.metadata.submit_index), nonzero,
                     pending->output_min_nonzero, pending->output_max_nonzero);
        if (!draws || !operations) {
            error = "output-triggered capture is missing its synchronous submit state";
            return false;
        }
        static const std::vector<ComputeItem> no_computes;
        if (!materialize_pending_gpu_capture(*pending, *draws,
                                             computes ? *computes : no_computes,
                                             *operations, semantic_state, exact_failures, error))
            return false;
        if (g_output_capture_claimed.exchange(true, std::memory_order_acq_rel)) return true;
        std::fprintf(stderr,
                     "[gpucap] captured output match submit %llu: %zu draws, %zu computes, "
                     "%zu operations, %zu blobs, %zu RTT seeds, %zu DS seeds -> %s\n",
                     static_cast<unsigned long long>(pending->capture.metadata.submit_index),
                     pending->capture.draws.size(), pending->capture.computes.size(),
                     pending->capture.operations.size(), pending->capture.blobs.size(),
                     pending->capture.rtt_seeds.size(), pending->capture.ds_seeds.size(),
                     pending->path.c_str());
    }
    if (!pending->materialized) {
        if (!draws || !operations) {
            error = "deferred capture is missing its exact submit realization";
            return false;
        }
        static const std::vector<ComputeItem> no_computes;
        if (!materialize_pending_gpu_capture(*pending, *draws,
                                             computes ? *computes : no_computes,
                                             *operations, semantic_state, exact_failures, error))
            return false;
        std::fprintf(stderr,
                     "[gpucap] captured deferred submit %llu: %zu draws, %zu computes, "
                     "%zu operations, %zu failures -> %s\n",
                     static_cast<unsigned long long>(pending->capture.metadata.submit_index),
                     pending->capture.draws.size(), pending->capture.computes.size(),
                     pending->capture.operations.size(),
                     pending->capture.failure_diagnostics.size(), pending->path.c_str());
    }
    pending->capture.expected_output_valid = !output.empty();
    pending->capture.expected_output_bytes = output.size();
    pending->capture.expected_output_hash = output.empty() ? 0 : gpu_capture_hash(output);
    if (!write_gpu_capture(pending->path, pending->capture, error)) return false;
    if (output.empty()) {
        std::fprintf(stderr, "[gpucap] wrote %s without output oracle\n", pending->path.c_str());
    } else {
        std::fprintf(stderr, "[gpucap] wrote %s output_bytes=%zu hash=%016llx\n",
                     pending->path.c_str(), output.size(),
                     static_cast<unsigned long long>(pending->capture.expected_output_hash));
    }
    return true;
}

} // namespace prosper::gpu
