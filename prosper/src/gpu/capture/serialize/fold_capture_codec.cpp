#include "gpu/capture/fold_capture.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace prosper::gpu {
namespace {
void check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
uint64_t checksum(std::span<const uint8_t> data) {
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t byte : data) { hash ^= byte; hash *= 1099511628211ull; }
    return hash;
}
// Explicit fields rather than ABI memory images: padding and host enum/bool representations
// must not enter the format. Struct default equality independently covers every output field.
template<class A, class T> void buffer(A& a, T& d) {
    a(d.base, d.stride, d.num_records, d.size_bytes, d.format, d.num_components,
      d.dst_sel, d.forbid_unknown_fallback);
}
template<class A, class T> void fetch(A& a, T& d) {
    a(d.fetch_pc, d.srsrc); buffer(a, d.desc);
    a(d.desc_v3, d.direct_user_data_index, d.from_seed); buffer(a, d.unshifted_desc);
    a(d.instruction_format, d.index_mode);
}
template<class A, class T> void use(A& a, T& d) {
    a(d.kind, d.key, d.t8, d.descriptor_source_addr, d.v4, d.table_record_count,
      d.table_entry_stride, d.table_element_offset, d.table_load_pc, d.table_base,
      d.bvh4, d.instruction_format, d.zero_record_raw, d.scalar_oob_offset_known,
      d.scalar_oob_byte_offset, d.scalar_buffer_dword_count, d.optional_null_raw_load,
      d.proven_null_guarded_raw_store, d.proven_null_nullable_raw_buffer,
      d.has_samp, d.s4, d.required_size, d.atomic_x2_record_count, d.use_pc,
      d.is_storage_image, d.mimg_dim, d.proven_zero_mip, d.is_depth_compare);
}
template<class A, class T> void event(A& a, T& e) {
    a(e.kind, e.probe, e.pc, e.bytes, e.address, e.readable);
    a.sequence(e.data, 64);
}
template<class A, class T> void capsule(A& a, T& c) {
    auto& i = c.input;
    a(i.invocation, i.code_address, i.stage, i.revision);
    a.sequence(i.code, kFoldMaxCodeDwords); a.sequence(i.user, 128); a.sequence(i.system, 128);
    a(i.user_base, i.absent_system_count, i.user_present, i.system_present, i.srt_requested,
      i.dispatch_present, i.branch_exclusive_disabled, i.dispatch_target);
    auto& d = i.dispatch;
    a(d.valid, d.selector_sgpr_base, d.selector_byte_offset, d.selector_addend,
      d.selector_max, d.setpc_pc, d.merge_pc);
    a.count(d.required_dwords, kFoldMaxCodeDwords);
    a.sequence(d.target_pcs, kFoldMaxCodeDwords); a.sequence(d.setup_pcs, kFoldMaxCodeDwords);
    a(c.complete, c.error, c.evaluated_instructions, c.decoded_dwords, c.shader_constant_specialized);
    a.sequence(c.events, kFoldMaxEvents); a.sequence(c.fetches, kFoldMaxOutputs);
    a.sequence(c.uses, kFoldMaxOutputs);
}

struct Writer {
    std::vector<uint8_t> bytes;
    template<class... T> void operator()(const T&... fields) { (value(fields), ...); }
    template<class T> void value(const T& input) {
        if constexpr (std::is_enum_v<T>) value(static_cast<std::underlying_type_t<T>>(input));
        else if constexpr (std::is_same_v<T, bool>) value(uint8_t(input));
        else {
            static_assert(std::is_integral_v<T>);
            using U = std::make_unsigned_t<T>;
            check(bytes.size() + sizeof(T) <= kFoldMaxFileBytes, "fold file budget exceeded");
            const U n = static_cast<U>(input);
            for (size_t k = 0; k < sizeof(T); ++k) bytes.push_back(uint8_t(n >> (8 * k)));
        }
    }
    template<class T, size_t N> void value(const std::array<T, N>& v) { for (const auto& x : v) value(x); }
    template<class T, size_t N> void value(const T (&v)[N]) { for (const auto& x : v) value(x); }
    void value(const std::string& v) {
        count(v.size(), 128); for (unsigned char ch : v) value(ch);
    }
    void value(const DynFetch& v) { fetch(*this, v); }
    void value(const SrtUse& v) { use(*this, v); }
    void value(const FoldEvent& v) { event(*this, v); }
    void count(size_t n, size_t limit) { check(n <= limit, "fold count exceeds bound"); value(uint32_t(n)); }
    template<class T> void sequence(const std::vector<T>& v, size_t limit) {
        count(v.size(), limit); for (const auto& x : v) value(x);
    }
};
struct Reader {
    std::span<const uint8_t> bytes;
    size_t position = 0;
    template<class... T> void operator()(T&... fields) { (value(fields), ...); }
    template<class T> void value(T& output) {
        if constexpr (std::is_enum_v<T>) {
            std::underlying_type_t<T> n{}; value(n); output = static_cast<T>(n);
        } else if constexpr (std::is_same_v<T, bool>) {
            uint8_t n{}; value(n); check(n <= 1, "invalid fold boolean"); output = n != 0;
        } else {
            static_assert(std::is_integral_v<T>);
            check(sizeof(T) <= bytes.size() - position, "truncated fold scalar");
            using U = std::make_unsigned_t<T>;
            U n = 0;
            for (size_t k = 0; k < sizeof(T); ++k) n |= U(bytes[position++]) << (8 * k);
            output = std::bit_cast<T>(n);
        }
    }
    template<class T, size_t N> void value(std::array<T, N>& v) { for (auto& x : v) value(x); }
    template<class T, size_t N> void value(T (&v)[N]) { for (auto& x : v) value(x); }
    void value(std::string& v) {
        size_t n{}; count(n, 128); check(n <= bytes.size() - position, "truncated fold string");
        v.assign(reinterpret_cast<const char*>(bytes.data() + position), n); position += n;
    }
    void value(DynFetch& v) { fetch(*this, v); }
    void value(SrtUse& v) { use(*this, v); }
    void value(FoldEvent& v) { event(*this, v); }
    void count(size_t& n, size_t limit) {
        uint32_t word{}; value(word); check(word <= limit, "fold count exceeds bound"); n = word;
    }
    template<class T> void sequence(std::vector<T>& v, size_t limit) {
        size_t n{}; count(n, limit);
        // Every serialized element occupies at least one byte. Refuse impossible counts before
        // allocation; each field reader checks its full width. Per-type caps bound object memory.
        check(n <= bytes.size() - position, "truncated fold sequence");
        v.resize(n); for (auto& x : v) value(x);
    }
};
uint64_t code_checksum(const std::vector<uint32_t>& code) {
    Writer w; for (auto word : code) w.value(word); return checksum(w.bytes);
}
} // namespace

std::vector<uint8_t> encode_fold_capture(const FoldCapture& capture) {
    validate_fold_capture(capture);
    Writer writer;
    const uint8_t magic[8] = {'P','R','F','O','L','D',0,0};
    writer(magic, uint32_t(1), code_checksum(capture.input.code));
    capsule(writer, capture);
    writer.value(checksum(writer.bytes));
    return std::move(writer.bytes);
}
FoldCapture decode_fold_capture(std::span<const uint8_t> bytes) {
    check(bytes.size() >= 28 && bytes.size() <= kFoldMaxFileBytes, "invalid fold file size");
    Reader trailer{bytes.last(8)}; uint64_t expected{}; trailer.value(expected);
    check(checksum(bytes.first(bytes.size() - 8)) == expected, "fold checksum mismatch");
    Reader reader{bytes.first(bytes.size() - 8)};
    uint8_t magic[8]{}; uint32_t version{}; uint64_t code_hash{};
    reader(magic, version, code_hash);
    const uint8_t wanted[8] = {'P','R','F','O','L','D',0,0};
    check(std::equal(std::begin(magic), std::end(magic), std::begin(wanted)) && version == 1,
          "unsupported fold schema");
    FoldCapture capture; capsule(reader, capture);
    check(reader.position == reader.bytes.size(), "trailing fold data");
    validate_fold_capture(capture);
    check(code_checksum(capture.input.code) == code_hash, "fold code identity mismatch");
    return capture;
}
FoldCapture read_fold_capture(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(bool(file), "cannot open fold capture");
    const auto size = file.tellg();
    check(size >= 0 && size <= kFoldMaxFileBytes, "invalid fold file size");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(bool(file), "cannot read complete fold capture");
    return decode_fold_capture(bytes);
}
void write_fold_capture(const std::filesystem::path& path, const FoldCapture& capture) {
    const auto bytes = encode_fold_capture(capture);
    static std::atomic<uint64_t> serial{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary;
    for (unsigned attempt = 0; attempt < 32; ++attempt) {
        temporary = path;
        temporary += ".tmp-" + std::to_string(stamp) + "-" + std::to_string(serial.fetch_add(1));
        if (std::filesystem::create_directory(temporary)) break;
        temporary.clear();
    }
    check(!temporary.empty(), "cannot reserve fold temporary");
    try {
        const auto payload = temporary / "payload";
        std::ofstream file(payload, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        file.flush(); check(bool(file), "fold write/flush failed");
        file.close(); check(bool(file), "fold close failed");
        std::filesystem::rename(payload, path); // Never remove the previous destination on failure.
        std::filesystem::remove(temporary);
    } catch (...) {
        std::error_code ignored; std::filesystem::remove_all(temporary, ignored);
        throw;
    }
}
} // namespace prosper::gpu
