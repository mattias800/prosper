#include "gpu_capture.hpp"

#include "bc_decode.hpp"
#include "tile.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace prosper::gpu {
namespace {

constexpr char kMagic[8] = {'P','R','G','P','C','A','P','\0'};
constexpr uint32_t kVersion = 4;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint64_t kMaxFileBytes = 4ull << 30;
constexpr uint64_t kMaxBlobBytes = 1ull << 30;
constexpr uint64_t kMaxTotalBlobBytes = 3ull << 30;
constexpr uint32_t kMaxDraws = 65536;
constexpr uint32_t kMaxResources = 65536;
constexpr uint32_t kMaxShaderWords = 16u << 20;
constexpr uint32_t kMaxStringBytes = 1u << 20;
constexpr uint64_t kMaxTotalRttSeedBytes = 1ull << 30;

CaptureRttSeedReader g_rtt_seed_reader;
ReplayRttSeedWriter g_rtt_seed_writer;

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
    bool words(std::vector<uint32_t>& v) {
        uint32_t n; if (!u32(n)) return false;
        if (n > kMaxShaderWords || uint64_t(n) * 4 > left) { if (error) *error = "invalid word-vector length"; return false; }
        v.resize(n); for (auto& x : v) if (!u32(x)) return false; return true;
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
    return true;
}

void write_resource(Writer& w, const GpuCapturedResource& c) {
    const auto& r = c.resource;
    w.u32(static_cast<uint32_t>(r.cls)); w.u32(static_cast<uint32_t>(r.format));
    w.u32(r.num_components); w.u32(r.binding); w.u64(r.gpu_addr); w.u32(r.size); w.u32(r.stride);
    w.u32(r.srt_offset); w.u32(r.sgpr_base); w.u32(r.fetch_pc); w.u32(r.img_dim);
    w.u32(r.width); w.u32(r.height); w.u32(r.tile_mode); w.u8(r.srgb); w.u32(r.sampler_sgpr_base);
    w.u32(r.mag_filter); w.u32(r.min_filter); w.u32(r.mip_filter); for (auto v : r.addr_uvw) w.u32(v);
    w.u32(r.border_color_type); w.f32(r.min_lod); w.f32(r.max_lod); w.f32(r.lod_bias);
    w.u32(r.max_aniso_ratio); w.u32(r.depth_compare_func); w.u32(r.unnormalized);
    for (auto v : r.swizzle) w.u32(v);
    w.u32(c.blob_index); w.u64(c.blob_offset);
}

bool read_resource(Reader& rd, GpuCapturedResource& c) {
    auto& r = c.resource; uint32_t cls, fmt; uint8_t b;
    if (!rd.u32(cls) || cls > static_cast<uint32_t>(ResourceClass::StorageImage) ||
        !rd.u32(fmt) || fmt > static_cast<uint32_t>(DataFormat::Float10_11_11)) return false;
    r.cls = static_cast<ResourceClass>(cls); r.format = static_cast<DataFormat>(fmt);
    if (!rd.u32(r.num_components) || !rd.u32(r.binding) || !rd.u64(r.gpu_addr) || !rd.u32(r.size) ||
        !rd.u32(r.stride) || !rd.u32(r.srt_offset) || !rd.u32(r.sgpr_base) || !rd.u32(r.fetch_pc) ||
        !rd.u32(r.img_dim) || !rd.u32(r.width) || !rd.u32(r.height) || !rd.u32(r.tile_mode) || !rd.u8(b)) return false;
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

bool read_table(Reader& r, GpuCapturedTable& t) {
    uint8_t b; uint32_t n; if (!r.u8(b) || !r.u32(n) || n > kMaxResources) return false;
    t.present = b != 0; t.resources.resize(n); for (auto& x : t.resources) if (!read_resource(r, x)) return false;
    if (!t.present && n != 0) { if (r.error) *r.error = "absent resource table has resources"; return false; }
    return true;
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
    return a && b > std::numeric_limits<uint64_t>::max() / a ? std::numeric_limits<uint64_t>::max() : a * b;
}

uint64_t resource_footprint(const ShaderResource& r) {
    uint64_t result = r.size;
    if (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) return result;
    const uint64_t faces = r.img_dim == 3u ? 6u : 1u;
    uint32_t w = r.width ? r.width : 4, h = r.height ? r.height : 4;
    const uint32_t bc = bc_block_bytes(r.format);
    uint64_t decoded = 0;
    if (bc) {
        uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
        decoded = tile_mode_is_tiled(r.tile_mode) ? tiled_elements_bytes(bw, bh, bc, r.tile_mode)
                                                  : checked_mul(checked_mul(bw, bh), bc);
    } else {
        uint32_t bpt = data_format_bytes(r.format) * (r.num_components ? r.num_components : 1);
        const bool f16 = r.format == DataFormat::Float16 && (bpt == 2 || bpt == 4 || bpt == 8);
        if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
        decoded = tile_mode_is_tiled(r.tile_mode) ? tiled_surface_bytes(w, h, r.tile_mode, 0, bpt)
                                                  : checked_mul(checked_mul(w, h), bpt);
    }
    decoded = checked_mul(decoded, faces);
    return std::max(result, decoded);
}

struct Interval { uint64_t begin, end; };

bool collect_intervals(const std::vector<DrawItem>& items, std::vector<Interval>& intervals, std::string& error) {
    uint64_t total = 0;
    auto add_table = [&](const ShaderResourceTable* t) -> bool {
        if (!t) return true;
        for (const auto& r : t->resources) {
            uint64_t n = resource_footprint(r);
            if (!n) continue;
            if (n > kMaxBlobBytes || r.gpu_addr > std::numeric_limits<uint64_t>::max() - n) {
                error = "resource capture range is invalid or exceeds 1 GiB"; return false;
            }
            intervals.push_back({r.gpu_addr, r.gpu_addr + n});
        }
        return true;
    };
    for (const auto& d : items) if (!add_table(d.vrt.get()) || !add_table(d.prt.get())) return false;
    std::sort(intervals.begin(), intervals.end(), [](auto a, auto b) { return a.begin < b.begin; });
    std::vector<Interval> merged;
    for (auto x : intervals) {
        if (!merged.empty() && x.begin <= merged.back().end) merged.back().end = std::max(merged.back().end, x.end);
        else merged.push_back(x);
    }
    for (auto x : merged) {
        uint64_t n = x.end - x.begin;
        if (total > kMaxTotalBlobBytes - n) { error = "capture resource data exceeds 3 GiB"; return false; }
        total += n;
    }
    intervals = std::move(merged); return true;
}

bool capture_table(const ShaderResourceTable* src, const std::vector<Interval>& intervals,
                   GpuCapturedTable& dst, std::string& error) {
    dst.present = src != nullptr;
    if (!src) return true;
    for (const auto& r : src->resources) {
        GpuCapturedResource c; c.resource = r; c.resource.host_data = nullptr; c.resource.host_data_size = 0;
        uint64_t n = resource_footprint(r);
        if (n) {
            auto it = std::find_if(intervals.begin(), intervals.end(), [&](auto x) {
                return x.begin <= r.gpu_addr && r.gpu_addr + n <= x.end;
            });
            if (it == intervals.end()) { error = "resource was not assigned to a capture blob"; return false; }
            c.blob_index = static_cast<uint32_t>(it - intervals.begin()); c.blob_offset = r.gpu_addr - it->begin;
        }
        dst.resources.push_back(std::move(c));
    }
    return true;
}

const char* env_or_empty(const char* name) { const char* v = std::getenv(name); return v ? v : ""; }

bool validate_rtt_seed(const GpuCaptureRttSeed& seed, std::string& error) {
    if (!seed.guest_addr || !seed.width || !seed.height) {
        error = "RTT seed has an invalid address or extent"; return false;
    }
    const uint64_t pixels = checked_mul(seed.width, seed.height);
    const uint64_t bytes = checked_mul(pixels, 4);
    if (bytes > kMaxBlobBytes || bytes != seed.rgba.size()) {
        error = "RTT seed byte count does not match its RGBA extent"; return false;
    }
    return true;
}

} // namespace

uint64_t gpu_capture_hash(const uint8_t* data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) { h ^= data[i]; h *= 1099511628211ull; }
    return h;
}

bool capture_draw_items(const std::vector<DrawItem>& items, const GpuCaptureMetadata& metadata,
                        const CaptureMemoryReader& reader, GpuCaptureFile& out, std::string& error,
                        const CaptureRttSeedReader& rtt_reader) {
    error.clear(); out = {}; out.metadata = metadata;
    if (items.empty() || items.size() > kMaxDraws) { error = "capture must contain 1..65536 draws"; return false; }
    if (!reader) { error = "capture memory reader is missing"; return false; }
    std::vector<Interval> intervals;
    if (!collect_intervals(items, intervals, error)) return false;
    for (auto x : intervals) {
        GpuCaptureBlob b; b.guest_addr = x.begin; b.bytes.resize(static_cast<size_t>(x.end - x.begin), 0);
        b.bytes_read = std::min<uint64_t>(reader(x.begin, b.bytes.data(), b.bytes.size()), b.bytes.size());
        out.blobs.push_back(std::move(b));
    }
    for (const auto& d : items) {
        GpuCapturedDraw c; c.vs = d.vs; c.fs = d.fs; c.ps = d.ps; c.vertex_count = d.vertex_count;
        c.indices = d.indices; c.color0_base = d.color0_base;
        c.color0_width = d.color0_width; c.color0_height = d.color0_height;
        if (!capture_table(d.vrt.get(), intervals, c.vrt, error) || !capture_table(d.prt.get(), intervals, c.prt, error)) return false;
        out.draws.push_back(std::move(c));
    }
    if (rtt_reader) {
        std::vector<uint64_t> candidates;
        auto add_table = [&](const ShaderResourceTable* table) {
            if (!table) return;
            for (const auto& r : table->resources)
                if ((r.cls == ResourceClass::Texture || r.cls == ResourceClass::StorageImage) && r.gpu_addr)
                    candidates.push_back(r.gpu_addr);
        };
        for (const auto& item : items) {
            add_table(item.vrt.get()); add_table(item.prt.get());
            if (item.color0_base) candidates.push_back(item.color0_base);
        }
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
    return true;
}

bool write_gpu_capture(const std::string& path, const GpuCaptureFile& c, std::string& error) {
    error.clear(); Writer w; w.raw(kMagic, sizeof(kMagic)); w.u32(kVersion); w.u32(kEndian);
    w.u32(c.metadata.width); w.u32(c.metadata.height); w.u64(c.metadata.submit_index);
    w.string(c.metadata.revision); w.string(c.metadata.title_id); w.string(c.metadata.input_route); w.string(c.metadata.savedata_dir);
    w.u32(static_cast<uint32_t>(c.metadata.renderer_env.size()));
    for (const auto& [name, value] : c.metadata.renderer_env) { w.string(name); w.string(value); }
    w.u64(c.expected_output_hash); w.u64(c.expected_output_bytes);
    w.u32(static_cast<uint32_t>(c.blobs.size()));
    for (const auto& b : c.blobs) { w.u64(b.guest_addr); w.u64(b.bytes_read); w.bytes(b.bytes); }
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
        w.u64(seed.guest_addr); w.u32(seed.width); w.u32(seed.height); w.bytes(seed.rgba);
    }
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& d : c.draws) {
        w.words(d.vs); w.words(d.fs); write_pipeline(w, d.ps); write_table(w, d.vrt); write_table(w, d.prt);
        w.u32(d.vertex_count); w.words(d.indices); w.u64(d.color0_base);
        w.u32(d.color0_width); w.u32(d.color0_height);
    }
    if (w.data.size() > kMaxFileBytes) { error = "capture file exceeds 4 GiB"; return false; }
    std::filesystem::path target(path), temp = target; temp += ".tmp";
    std::error_code ec; if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path(), ec);
    std::ofstream f(temp, std::ios::binary | std::ios::trunc);
    if (!f || !f.write(reinterpret_cast<const char*>(w.data.data()), static_cast<std::streamsize>(w.data.size()))) {
        error = "cannot write capture temporary file"; return false;
    }
    f.close(); std::filesystem::rename(temp, target, ec);
    if (ec) { std::filesystem::remove(target, ec); ec.clear(); std::filesystem::rename(temp, target, ec); }
    if (ec) { error = "cannot install capture file: " + ec.message(); return false; }
    return true;
}

bool read_gpu_capture(const std::string& path, GpuCaptureFile& c, std::string& error) {
    error.clear(); c = {}; std::error_code ec; uint64_t size = std::filesystem::file_size(path, ec);
    if (ec || size > kMaxFileBytes || size > std::numeric_limits<size_t>::max()) { error = "invalid capture file size"; return false; }
    std::vector<uint8_t> bytes(static_cast<size_t>(size)); std::ifstream f(path, std::ios::binary);
    if (!f || !f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot read capture file"; return false;
    }
    Reader r{bytes.data(), bytes.size(), &error}; char magic[8]; uint32_t version, endian;
    if (!r.take(magic, 8) || std::memcmp(magic, kMagic, 8) || !r.u32(version) || version < 1 || version > kVersion ||
        !r.u32(endian) || endian != kEndian) { error = "unsupported capture header"; return false; }
    auto& m = c.metadata;
    if (!r.u32(m.width) || !r.u32(m.height) || !r.u64(m.submit_index) || !r.string(m.revision) ||
        !r.string(m.title_id) || !r.string(m.input_route) || !r.string(m.savedata_dir)) return false;
    uint32_t ne; if (!r.u32(ne) || ne > 256) { error = "invalid renderer environment count"; return false; }
    m.renderer_env.resize(ne);
    for (auto& [name, value] : m.renderer_env) if (!r.string(name) || !r.string(value)) return false;
    if (!r.u64(c.expected_output_hash) || !r.u64(c.expected_output_bytes)) return false;
    uint32_t nb; if (!r.u32(nb) || nb > kMaxResources) { error = "invalid blob count"; return false; }
    uint64_t total = 0; c.blobs.resize(nb);
    for (auto& b : c.blobs) {
        if (!r.u64(b.guest_addr) || !r.u64(b.bytes_read) || !r.bytes(b.bytes)) return false;
        if (b.bytes_read > b.bytes.size() || total > kMaxTotalBlobBytes - b.bytes.size()) { error = "invalid blob metadata"; return false; }
        total += b.bytes.size();
    }
    if (version >= 4) {
        uint32_t ns; if (!r.u32(ns) || ns > kMaxResources) { error = "invalid RTT seed count"; return false; }
        uint64_t seed_total = 0; c.rtt_seeds.resize(ns);
        std::unordered_set<uint64_t> addresses;
        for (auto& seed : c.rtt_seeds) {
            if (!r.u64(seed.guest_addr) || !r.u32(seed.width) || !r.u32(seed.height) || !r.bytes(seed.rgba)) return false;
            if (!validate_rtt_seed(seed, error)) return false;
            if (!addresses.insert(seed.guest_addr).second) { error = "duplicate RTT seed address"; return false; }
            if (seed_total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
                error = "capture RTT seed data exceeds 1 GiB"; return false;
            }
            seed_total += seed.rgba.size();
        }
    }
    uint32_t nd; if (!r.u32(nd) || !nd || nd > kMaxDraws) { error = "invalid draw count"; return false; }
    c.draws.resize(nd);
    for (auto& d : c.draws) {
        if (!r.words(d.vs) || !r.words(d.fs) || !read_pipeline(r, d.ps, version) || !read_table(r, d.vrt) ||
            !read_table(r, d.prt) || !r.u32(d.vertex_count) || !r.words(d.indices) || !r.u64(d.color0_base)) return false;
        if (version >= 3 && (!r.u32(d.color0_width) || !r.u32(d.color0_height))) return false;
    }
    if (r.left) { error = "capture has trailing data"; return false; }
    return true;
}

bool materialize_gpu_replay(const GpuCaptureFile& c, GpuReplayFrame& out, std::string& error) {
    error.clear(); out = {}; out.metadata = c.metadata; out.blobs = c.blobs; out.rtt_seeds = c.rtt_seeds;
    out.expected_output_hash = c.expected_output_hash; out.expected_output_bytes = c.expected_output_bytes;
    auto table = [&](const GpuCapturedTable& src, std::shared_ptr<ShaderResourceTable>& dst) -> bool {
        if (!src.present) { dst.reset(); return src.resources.empty(); }
        dst = std::make_shared<ShaderResourceTable>();
        for (const auto& x : src.resources) {
            ShaderResource r = x.resource; r.host_data = nullptr; r.host_data_size = 0;
            if (x.blob_index != 0xFFFFFFFFu) {
                if (x.blob_index >= out.blobs.size() || x.blob_offset > out.blobs[x.blob_index].bytes.size()) {
                    error = "resource references an invalid capture blob"; return false;
                }
                auto& b = out.blobs[x.blob_index].bytes;
                uint64_t need = resource_footprint(r);
                if (need > b.size() - x.blob_offset) { error = "resource footprint exceeds capture blob"; return false; }
                r.host_data = b.data() + x.blob_offset; r.host_data_size = need;
            }
            dst->resources.push_back(r);
        }
        return true;
    };
    out.items.reserve(c.draws.size());
    for (const auto& x : c.draws) {
        DrawItem d; d.vs = x.vs; d.fs = x.fs; d.ps = x.ps; d.vertex_count = x.vertex_count;
        d.indices = x.indices; d.color0_base = x.color0_base;
        d.color0_width = x.color0_width; d.color0_height = x.color0_height;
        if (!table(x.vrt, d.vrt) || !table(x.prt, d.prt)) return false;
        out.items.push_back(std::move(d));
    }
    return true;
}

void set_gpu_capture_rtt_seed_reader(CaptureRttSeedReader reader) { g_rtt_seed_reader = std::move(reader); }
void set_gpu_replay_rtt_seed_writer(ReplayRttSeedWriter writer) { g_rtt_seed_writer = std::move(writer); }

bool restore_gpu_replay_rtt_seeds(const std::vector<GpuCaptureRttSeed>& seeds, std::string& error) {
    error.clear();
    if (!seeds.empty() && !g_rtt_seed_writer) { error = "live renderer has no RTT seed writer"; return false; }
    for (const auto& seed : seeds) {
        if (!validate_rtt_seed(seed, error) || !g_rtt_seed_writer(seed, error)) return false;
    }
    return true;
}

std::unique_ptr<PendingGpuCapture> begin_requested_gpu_capture(const std::vector<DrawItem>& items,
                                                               uint32_t width, uint32_t height) {
    const char* path = std::getenv("PROSPER_GPU_CAPTURE"); if (!path || !*path) return {};
    static std::atomic<uint64_t> invocation_sequence{0};
    const uint64_t invocation = invocation_sequence.fetch_add(1);
    uint64_t after = 0;
    if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_AFTER")) after = std::strtoull(v, nullptr, 0);
    if (invocation < after) return {};
    uint64_t min_draws = 0, max_draws = std::numeric_limits<uint64_t>::max();
    if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_MIN_DRAWS")) min_draws = std::strtoull(v, nullptr, 0);
    if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_MAX_DRAWS")) max_draws = std::strtoull(v, nullptr, 0);
    if (items.size() < min_draws || items.size() > max_draws) return {};
    static std::atomic<uint64_t> sequence{0}; static std::atomic<bool> claimed{false};
    uint64_t current = sequence.fetch_add(1), wanted = 0;
    if (const char* at = std::getenv("PROSPER_GPU_CAPTURE_AT")) wanted = std::strtoull(at, nullptr, 0);
    if (current != wanted || claimed.exchange(true)) return {};
    auto pending = std::make_unique<PendingGpuCapture>(); pending->path = path;
    GpuCaptureMetadata m; m.width = width; m.height = height; m.submit_index = current;
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
        "PROSPER_DRAW_ISO", "PROSPER_FLIP_FRONT_FACE", "PROSPER_FS_SPV", "PROSPER_ISO_AT", "PROSPER_ISO_AT2",
        "PROSPER_ISO_RGB", "PROSPER_ISO_TOL", "PROSPER_NO_BLEND", "PROSPER_NO_CULL",
        "PROSPER_NO_DEPTH", "PROSPER_NO_STENCIL", "PROSPER_STENCIL_CLEAR", "PROSPER_STENCIL_REPLACE",
        "PROSPER_NO_SWIZZLE", "PROSPER_NODETILE",
        "PROSPER_DESCRIPTOR_VALIDATE",
        "PROSPER_PITCH", "PROSPER_RENDER_NOPS", "PROSPER_RENDER_REFVS", "PROSPER_RENDER_TESTPS",
        "PROSPER_RTT", "PROSPER_RTT_NOSEED", "PROSPER_RTT_PERTARGET", "PROSPER_RTT_SINGLE_TARGET",
        "PROSPER_TESTTEX",
        "PROSPER_TESTLUT", "PROSPER_TESTLUT32"
    };
    for (const char* name : render_env) if (const char* value = std::getenv(name)) m.renderer_env.emplace_back(name, value);
    auto reader = [](uint64_t addr, uint8_t* dst, size_t bytes) -> size_t {
        size_t done = 0; constexpr size_t chunk_max = 0x10000;
        while (done < bytes) {
            size_t n = std::min(bytes - done, chunk_max);
            if (!guest_readable(addr + done, static_cast<uint32_t>(n))) break;
            std::memcpy(dst + done, reinterpret_cast<const void*>(uintptr_t(addr + done)), n); done += n;
        }
        return done;
    };
    std::string error;
    if (!capture_draw_items(items, m, reader, pending->capture, error, g_rtt_seed_reader)) {
        std::fprintf(stderr, "[gpucap] capture failed: %s\n", error.c_str()); return {};
    }
    std::fprintf(stderr, "[gpucap] captured match %llu at invocation %llu: %zu draws, %zu blobs, %zu RTT seeds -> %s\n",
                 static_cast<unsigned long long>(current), static_cast<unsigned long long>(invocation),
                 items.size(), pending->capture.blobs.size(), pending->capture.rtt_seeds.size(), path);
    return pending;
}

bool finish_requested_gpu_capture(std::unique_ptr<PendingGpuCapture> pending,
                                  const std::vector<uint8_t>& output, std::string& error) {
    if (!pending) return true;
    pending->capture.expected_output_bytes = output.size(); pending->capture.expected_output_hash = gpu_capture_hash(output);
    if (!write_gpu_capture(pending->path, pending->capture, error)) return false;
    std::fprintf(stderr, "[gpucap] wrote %s output_bytes=%zu hash=%016llx\n", pending->path.c_str(), output.size(),
                 static_cast<unsigned long long>(pending->capture.expected_output_hash));
    return true;
}

} // namespace prosper::gpu
