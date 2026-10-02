#include "gpu/capture/fragment_compile_case.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#ifdef _WIN32
#include <windows.h>
#endif

namespace prosper::gpu {
namespace {
void check(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
uint64_t checksum(std::span<const uint8_t> bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t byte : bytes) { hash ^= byte; hash *= 1099511628211ull; }
    return hash;
}
// Complete resource metadata, intentionally not capture_codecs' historical prefix. All pointer
// fields are separate checked references. This single field walk is shared by both codec directions.
template<class A, class R> void resource(A& a, R& r) {
    a(r.cls, r.format, r.num_components, r.binding, r.gpu_addr, r.size, r.stride,
      r.srt_offset, r.sgpr_base, r.table_index_count, r.table_entry_stride,
      r.table_index_sgpr, r.table_selector_mode, r.table_load_pc);
    a.sequence(r.table_entries, kCompileCaseMaxEntries);
    a(r.direct_vsharp_sh_register_base, r.fetch_pc, r.nested_raw_snapshot_admitted,
      r.fetch_index_mode, r.bvh_box_grow, r.bvh_sort_enabled, r.flat_base_sgpr,
      r.img_dim, r.width, r.height, r.depth, r.sample_count, r.tile_mode,
      r.linear_row_pitch_bytes, r.declared_mip_levels, r.mip_chain_element_width,
      r.mip_chain_element_height, r.mip_chain_bytes_per_block, r.mip_chain_max_level,
      r.mip_chain_base_level, r.in_mip_tail, r.mip_tail_offset, r.mip_tail_bytes,
      r.mip_tail_x, r.mip_tail_y, r.layer_stride_bytes, r.layer_mip_offset_bytes,
      r.proven_zero_mip, r.srgb, r.sampler_sgpr_base, r.mag_filter, r.min_filter,
      r.mip_filter, r.addr_uvw, r.border_color_type, r.min_lod, r.max_lod, r.lod_bias,
      r.max_aniso_ratio, r.depth_compare_func, r.depth_compare, r.unnormalized,
      r.swizzle, r.max_uncompressed_block_size, r.max_compressed_block_size,
      r.meta_pipe_aligned, r.write_compress_enabled, r.compression_enabled,
      r.alpha_is_on_msb, r.color_transform, r.metadata_addr, r.dcc_metadata_size,
      r.dcc_metadata_host_data_size, r.host_data_size, r.host_data_prefix_bytes,
      r.atomic_x2_record_count, r.scalar_raw_pointer_word_hi,
      r.selected_sbuffer_soffset, r.selected_sbuffer_words, r.indirect_buffer_contract_tag,
      r.indirect_buffer_binding_bytes, r.indirect_buffer_slot_count,
      r.indirect_buffer_header_bytes, r.indirect_buffer_slot_bytes);
    auto& p = r.indirect_pointer_relocation;
    a(p.carrier_version, p.proof_schema, p.binding_bytes, p.record_count, p.segment_count,
      p.segment_directory_byte_offset, p.proof_fingerprint);
    a(r.scalar_buffer_dword_count, r.raw_register_snapshot);
}
template<class A, class C> void capsule(A& a, C& c) {
    a(c.complete, c.reason, c.compiler, c.expected_reject, c.expected_produced, c.program_address);
    a.sequence(c.code, kCompileCaseMaxWords); a.sequence(c.source, kCompileCaseMaxWords);
    a(c.has_resources, c.has_pixel_inputs, c.has_system_inputs,
      c.pixel_inputs.controls, c.pixel_inputs.valid_mask, c.pixel_inputs.passthrough_mask,
      c.pixel_inputs.consumed_mask, c.pixel_inputs.consumed_known,
      c.system_inputs.ena, c.system_inputs.addr,
      c.interpolation.parameter_locations, c.interpolation.system_locations,
      c.interpolation.attribute_mask, c.interpolation.smooth_mask,
      c.interpolation.passthrough_mask, c.interpolation.flat_mask,
      c.interpolation.requires_geometry, c.interpolation.valid,
      c.wave32, c.float_mode.available, c.float_mode.value, c.pcrel_target,
      c.trip.bound, c.trip.only_program, c.trip.only_phase, c.trip.only_ordinal,
      c.resources.vertices_per_instance);
    a.sequence(c.resources.resources, kCompileCaseMaxResources);
    a.sequence(c.blobs, kCompileCaseMaxBlobs);
    a.sequence(c.backing, kCompileCaseMaxResources);
    a(c.choices.error); a.sequence(c.choices.reads, 65536);
}
struct Writer {
    std::vector<uint8_t> bytes;
    template<class... T> void operator()(const T&... v) { (value(v), ...); }
    template<class T> void value(const T& v) {
        if constexpr (std::is_enum_v<T>) value(static_cast<std::underlying_type_t<T>>(v));
        else if constexpr (std::is_same_v<T, bool>) value(uint8_t(v));
        else if constexpr (std::is_same_v<T, float>) value(std::bit_cast<uint32_t>(v));
        else {
            static_assert(std::is_integral_v<T>);
            check(sizeof(T) <= kCompileCaseMaxFileBytes - bytes.size(), "compile-case file budget");
            using U = std::make_unsigned_t<T>; const U n = static_cast<U>(v);
            for (size_t k = 0; k < sizeof(T); ++k) bytes.push_back(uint8_t(n >> (8 * k)));
        }
    }
    template<class T, size_t N> void value(const std::array<T, N>& v) { for (const auto& x : v) value(x); }
    template<class T, size_t N> void value(const T (&v)[N]) { for (const auto& x : v) value(x); }
    void value(const std::string& v) { count(v.size(), 256); for (unsigned char ch : v) value(ch); }
    void value(const ShaderResource& v) { resource(*this, v); }
    void value(const ShaderBufferTableEntry& v) { (*this)(v.vsharp, v.gpu_addr, v.size, v.stride, v.host_data_size); }
    void value(const CompileCaseBacking& v) { (*this)(v.blob, v.offset); }
    void value(const CompileCaseResourceBacking& v) { (*this)(v.host, v.dcc); sequence(v.entries, kCompileCaseMaxEntries); }
    void value(const CompilerChoiceRead& v) { (*this)(v.choice, v.value); }
    void value(const CompileCaseBlob& v) {
        (*this)(v.owner, v.present, v.opaque, v.size, v.alias_of);
        if (v.present && !v.opaque && v.alias_of == UINT32_MAX) sequence(*v.bytes, kCompileCaseMaxPayloadBytes);
    }
    void count(size_t n, size_t cap) { check(n <= cap, "compile-case count bound"); value(uint32_t(n)); }
    template<class T> void sequence(const std::vector<T>& v, size_t cap) {
        count(v.size(), cap); for (const auto& x : v) value(x);
    }
};
struct Reader {
    std::span<const uint8_t> bytes;
    size_t position = 0, entries = 0, backing_entries = 0, payload = 0;
    template<class... T> void operator()(T&... v) { (value(v), ...); }
    template<class T> void value(T& v) {
        if constexpr (std::is_enum_v<T>) { std::underlying_type_t<T> n{}; value(n); v = static_cast<T>(n); }
        else if constexpr (std::is_same_v<T, bool>) { uint8_t n{}; value(n); check(n <= 1, "compile-case boolean"); v = n != 0; }
        else if constexpr (std::is_same_v<T, float>) { uint32_t n{}; value(n); v = std::bit_cast<float>(n); }
        else {
            static_assert(std::is_integral_v<T>);
            check(sizeof(T) <= bytes.size() - position, "truncated compile-case scalar");
            using U = std::make_unsigned_t<T>; U n = 0;
            for (size_t k = 0; k < sizeof(T); ++k) n |= U(bytes[position++]) << (8 * k);
            v = std::bit_cast<T>(n);
        }
    }
    template<class T, size_t N> void value(std::array<T, N>& v) { for (auto& x : v) value(x); }
    template<class T, size_t N> void value(T (&v)[N]) { for (auto& x : v) value(x); }
    void value(std::string& v) {
        const size_t n = count(256); check(n <= bytes.size() - position, "truncated compile-case text");
        v.assign(reinterpret_cast<const char*>(bytes.data() + position), n); position += n;
    }
    void value(ShaderResource& v) { resource(*this, v); }
    void value(ShaderBufferTableEntry& v) { (*this)(v.vsharp, v.gpu_addr, v.size, v.stride, v.host_data_size); }
    void value(CompileCaseBacking& v) { (*this)(v.blob, v.offset); }
    void value(CompileCaseResourceBacking& v) { (*this)(v.host, v.dcc); sequence(v.entries, kCompileCaseMaxEntries); }
    void value(CompilerChoiceRead& v) { (*this)(v.choice, v.value); }
    void value(CompileCaseBlob& v) {
        (*this)(v.owner, v.present, v.opaque, v.size, v.alias_of);
        if (v.present && !v.opaque && v.alias_of == UINT32_MAX) {
            v.bytes = std::make_shared<std::vector<uint8_t>>(); sequence(*v.bytes, kCompileCaseMaxPayloadBytes);
        }
    }
    size_t count(size_t cap) { uint32_t n{}; value(n); check(n <= cap, "compile-case count bound"); return n; }
    template<class T> void sequence(std::vector<T>& v, size_t cap) {
        const size_t n = count(cap);
        check(n <= bytes.size() - position, "truncated compile-case sequence");
        if constexpr (std::is_same_v<T, ShaderBufferTableEntry>) {
            check(n <= kCompileCaseMaxEntries - entries, "compile-case aggregate entries"); entries += n;
        }
        if constexpr (std::is_same_v<T, CompileCaseBacking>) {
            check(n <= kCompileCaseMaxEntries - backing_entries, "compile-case aggregate backing entries"); backing_entries += n;
        }
        if constexpr (std::is_same_v<T, uint8_t>) {
            check(n <= kCompileCaseMaxPayloadBytes - payload, "compile-case aggregate payload"); payload += n;
        }
        v.resize(n); for (auto& x : v) value(x);
    }
};
} // namespace
std::vector<uint8_t> encode_fragment_compile_case(const FragmentCompileCase& c) {
    validate_fragment_compile_case(c);
    Writer w; const uint8_t magic[8] = {'P','R','F','C','A','S','E',0};
    w(magic, uint32_t(4)); capsule(w, c);
    // Schema 1's field walk remains an unchanged prefix. Retain each marker verbatim: it is a
    // compiler input whose malformed value can cause a refusal, never replay admission authority.
    w.count(c.resources.resources.size(), kCompileCaseMaxResources);
    for (const auto& resource : c.resources.resources) w.value(resource.owned_raw_snapshot_bytes);
    // Profile follows the unchanged schema-2 marker tail, independently of guest MODE and
    // the ordered legacy FloatControls semantic read.
    w(c.float_transport.profile);
    // Independent launch flags follow the unchanged schema-3 transport tail.
    w(c.float_flags.available, c.float_flags.ieee_mode, c.float_flags.dx10_clamp);
    w(c.launch_rsrc1.available, c.launch_rsrc1.value);
    w.value(checksum(w.bytes)); return std::move(w.bytes);
}
FragmentCompileCase decode_fragment_compile_case(std::span<const uint8_t> bytes) {
    check(bytes.size() >= 20 && bytes.size() <= kCompileCaseMaxFileBytes, "compile-case file size");
    Reader trailer{bytes.last(8)}; uint64_t hash{}; trailer.value(hash);
    check(checksum(bytes.first(bytes.size() - 8)) == hash, "compile-case checksum");
    Reader r{bytes.first(bytes.size() - 8)}; uint8_t magic[8]{}; uint32_t schema{}; r(magic, schema);
    const uint8_t wanted[8] = {'P','R','F','C','A','S','E',0};
    check(std::equal(std::begin(magic), std::end(magic), std::begin(wanted)) &&
          (schema >= 1 && schema <= 4),
          "compile-case schema");
    FragmentCompileCase c; capsule(r, c);
    if (schema >= 2) {
        const size_t count = r.count(kCompileCaseMaxResources);
        check(count == c.resources.resources.size(), "compile-case marker/resource count");
        for (auto& resource : c.resources.resources) r.value(resource.owned_raw_snapshot_bytes);
    }
    if (schema >= 3) r(c.float_transport.profile);
    if (schema >= 4) {
        r(c.float_flags.available, c.float_flags.ieee_mode, c.float_flags.dx10_clamp);
        r(c.launch_rsrc1.available, c.launch_rsrc1.value);
    }
    check(r.position == r.bytes.size(), "compile-case trailing data");
    for (size_t k = 0; k < c.blobs.size(); ++k) if (c.blobs[k].alias_of != UINT32_MAX) {
        check(c.blobs[k].alias_of < k, "compile-case alias index");
        c.blobs[k].bytes = c.blobs[c.blobs[k].alias_of].bytes;
    }
    validate_fragment_compile_case(c); restore_fragment_compile_case_backing(c);
    if (schema < 3) {
        // Validate the genuine old schema before losing completeness. In particular, merely
        // relabeling a schema-3 file as schema 2 is trailing data, not a legacy compatibility arm.
        c.float_transport = {};
        c.complete = false;
        if (c.reason.empty()) c.reason = "fragment-transport-config-unavailable";
    }
    if (schema < 4) {
        // The checksum and exact legacy EOF are checked before downgrading. Relabeling a newer
        // file cannot hide its tail or convert absent flags into an explicitly recorded Unknown.
        c.float_flags = {};
        c.launch_rsrc1 = {};
        c.complete = false;
        if (c.reason.empty()) c.reason = "fragment-float-flags-unavailable";
    }
    return c;
}
FragmentCompileCase read_fragment_compile_case(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate); check(bool(f), "cannot open compile case");
    const auto size = f.tellg(); check(size >= 0 && size <= kCompileCaseMaxFileBytes, "compile-case file size");
    std::vector<uint8_t> bytes(static_cast<size_t>(size)); f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(bool(f), "cannot read complete compile case"); return decode_fragment_compile_case(bytes);
}
namespace {
void install_bytes(const std::filesystem::path& path, std::span<const uint8_t> bytes, bool replace = true) {
    static std::atomic<uint64_t> serial{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary;
    for (unsigned attempt = 0; attempt < 32; ++attempt) {
        temporary = path; temporary += ".tmp-" + std::to_string(stamp) + "-" + std::to_string(serial.fetch_add(1));
        if (std::filesystem::create_directory(temporary)) break;
        temporary.clear();
    }
    check(!temporary.empty(), "cannot reserve compile-case temporary");
    try {
        const auto payload = temporary / "payload";
        std::ofstream f(payload, std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        f.flush(); check(bool(f), "compile-case write/flush failed"); f.close(); check(bool(f), "compile-case close failed");
#ifdef _WIN32
        bool installed = false;
        for (unsigned attempt = 0; attempt < 50; ++attempt) {
            if (MoveFileExW(payload.c_str(), path.c_str(), replace ? MOVEFILE_REPLACE_EXISTING : 0)) { installed = true; break; }
            const DWORD error = GetLastError();
            if (error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        check(installed, "cannot atomically install compile case");
#else
        if (replace) std::filesystem::rename(payload, path);
        else {
            // The payload is adjacent to the destination: linking installs the complete file
            // atomically on the same filesystem and fails if the destination already exists.
            std::filesystem::create_hard_link(payload, path);
            std::filesystem::remove(payload);
        }
#endif
        std::filesystem::remove(temporary);
    } catch (...) {
        std::error_code ignored; std::filesystem::remove_all(temporary, ignored); throw;
    }
}
} // namespace
void write_fragment_compile_case(const std::filesystem::path& path, const FragmentCompileCase& c, bool replace) {
    const auto bytes = encode_fragment_compile_case(c); install_bytes(path, bytes, replace);
}
void write_fragment_compile_source(const std::filesystem::path& path, std::span<const uint32_t> words) {
    check(words.size() >= 5 && words.size() <= kCompileCaseMaxWords && words[0] == 0x07230203,
          "refused or invalid SOURCE cannot be written as SPIR-V");
    Writer w; for (auto word : words) w.value(word); install_bytes(path, w.bytes);
}
} // namespace prosper::gpu
