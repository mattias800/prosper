#include "gpu_timeline.hpp"

#include "render_state.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <utility>

namespace prosper::gpu {
namespace {

constexpr uint8_t kFileMagic[8] = {'P', 'R', 'G', 'T', 'L', 'N', '\0', '\0'};
constexpr uint8_t kRecordMagic[4] = {'T', 'L', 'R', 'C'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint32_t kMaxPayloadBytes = 1u << 20;
constexpr uint32_t kFlushInterval = 256;

enum class RecordType : uint32_t { Metadata = 1, Submit = 2, Present = 3 };

struct Bytes {
    std::vector<uint8_t> data;
    void raw(const void* src, size_t bytes) {
        const auto* p = static_cast<const uint8_t*>(src);
        data.insert(data.end(), p, p + bytes);
    }
    void u32(uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) data.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
    void u64(uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) data.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
    void string(const std::string& value) {
        u32(static_cast<uint32_t>(value.size()));
        raw(value.data(), value.size());
    }
};

struct Cursor {
    const uint8_t* data = nullptr;
    size_t left = 0;
    bool take(void* dst, size_t bytes) {
        if (bytes > left) return false;
        if (bytes) std::memcpy(dst, data, bytes);
        data += bytes;
        left -= bytes;
        return true;
    }
    bool u32(uint32_t& value) {
        uint8_t b[4];
        if (!take(b, sizeof b)) return false;
        value = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        return true;
    }
    bool u64(uint64_t& value) {
        uint8_t b[8];
        if (!take(b, sizeof b)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= uint64_t(b[i]) << (i * 8);
        return true;
    }
    bool string(std::string& value) {
        uint32_t size = 0;
        if (!u32(size) || size > left) return false;
        value.assign(reinterpret_cast<const char*>(data), size);
        data += size;
        left -= size;
        return true;
    }
};

uint64_t hash_bytes(const uint8_t* data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool write_all(FILE* file, const void* data, size_t bytes, std::string& error) {
    if (!bytes) return true;
    if (std::fwrite(data, 1, bytes, file) == bytes) return true;
    error = std::string("timeline write failed: ") + std::strerror(errno);
    return false;
}

bool append_record(FILE* file, RecordType type, uint64_t sequence, uint64_t elapsed_ns,
                   const Bytes& payload, std::string& error) {
    if (payload.data.size() > kMaxPayloadBytes) {
        error = "timeline record payload exceeds 1 MiB";
        return false;
    }
    Bytes body;
    body.u32(static_cast<uint32_t>(type));
    body.u32(static_cast<uint32_t>(payload.data.size()));
    body.u64(sequence);
    body.u64(elapsed_ns);
    body.raw(payload.data.data(), payload.data.size());
    const uint64_t checksum = hash_bytes(body.data.data(), body.data.size());
    Bytes trailer;
    trailer.u64(checksum);
    return write_all(file, kRecordMagic, sizeof kRecordMagic, error) &&
           write_all(file, body.data.data(), body.data.size(), error) &&
           write_all(file, trailer.data.data(), trailer.data.size(), error);
}

bool read_exact(std::ifstream& file, void* dst, size_t bytes, bool& clean_eof) {
    clean_eof = false;
    file.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (file.gcount() == static_cast<std::streamsize>(bytes)) return true;
    clean_eof = file.gcount() == 0 && file.eof();
    return false;
}

const char* env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

std::atomic<uint64_t> g_active_submit_no{0};

struct RuntimeRecorder {
    std::mutex mutex;
    std::unique_ptr<GpuTimelineWriter> writer;
    std::atomic<GpuTimelineWriter*> fast_writer{nullptr};
    std::atomic<uint8_t> state{0}; // 0 uninitialized, 1 active, 2 disabled/failed
    bool failed = false;

    GpuTimelineWriter* get() {
        if (GpuTimelineWriter* active = fast_writer.load(std::memory_order_acquire)) return active;
        if (state.load(std::memory_order_acquire) == 2) return nullptr;
        std::lock_guard<std::mutex> lock(mutex);
        if (writer && !failed) return writer.get();
        if (state.load(std::memory_order_relaxed) == 2) return nullptr;
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        if (!path || !*path) {
            state.store(2, std::memory_order_release);
            return nullptr;
        }
        std::error_code ec;
        const std::filesystem::path output(path);
        if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path(), ec);
        if (ec) {
            std::fprintf(stderr, "[timeline] cannot create '%s': %s\n", path, ec.message().c_str());
            failed = true;
            state.store(2, std::memory_order_release);
            return nullptr;
        }
        GpuTimelineMetadata metadata;
#ifdef PROSPER_GIT_REVISION
        metadata.revision = PROSPER_GIT_REVISION;
#else
        metadata.revision = "unknown";
#endif
        if (const char* revision = std::getenv("PROSPER_CAPTURE_REVISION")) metadata.revision = revision;
        metadata.title_id = env_or_empty("PROSPER_CAPTURE_TITLE");
        metadata.input_route = env_or_empty("PROSPER_PAD_SCRIPT");
        writer = std::make_unique<GpuTimelineWriter>();
        std::string error;
        if (!writer->open(path, metadata, error)) {
            std::fprintf(stderr, "[timeline] open failed: %s\n", error.c_str());
            writer.reset();
            failed = true;
            state.store(2, std::memory_order_release);
            return nullptr;
        }
        std::fprintf(stderr, "[timeline] recording native submit/present index -> %s\n", path);
        GpuTimelineWriter* active = writer.get();
        fast_writer.store(active, std::memory_order_release);
        state.store(1, std::memory_order_release);
        return active;
    }

    void mark_failed(const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!failed) std::fprintf(stderr, "[timeline] recording failed: %s\n", error.c_str());
        failed = true;
        fast_writer.store(nullptr, std::memory_order_release);
        state.store(2, std::memory_order_release);
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex);
        fast_writer.store(nullptr, std::memory_order_release);
        if (writer) writer->close();
        state.store(2, std::memory_order_release);
    }
};

RuntimeRecorder& runtime_recorder() {
    static RuntimeRecorder recorder;
    return recorder;
}

} // namespace

struct GpuTimelineWriter::Impl {
    FILE* file = nullptr;
    std::mutex mutex;
    uint64_t next_sequence = 0;
    uint32_t records_since_flush = 0;
    std::chrono::steady_clock::time_point start;

    uint64_t elapsed_ns() const {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    }

    bool append(RecordType type, const Bytes& payload, std::string& error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!file) {
            error = "timeline writer is not open";
            return false;
        }
        if (!append_record(file, type, next_sequence++, elapsed_ns(), payload, error)) return false;
        if (++records_since_flush >= kFlushInterval) {
            if (std::fflush(file) != 0) {
                error = std::string("timeline flush failed: ") + std::strerror(errno);
                return false;
            }
            records_since_flush = 0;
        }
        return true;
    }
};

GpuTimelineWriter::GpuTimelineWriter() : impl_(std::make_unique<Impl>()) {}
GpuTimelineWriter::~GpuTimelineWriter() { close(); }

bool GpuTimelineWriter::open(const std::string& path, const GpuTimelineMetadata& metadata,
                             std::string& error) {
    close();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->file = std::fopen(path.c_str(), "wb");
    if (!impl_->file) {
        error = std::string("cannot open timeline: ") + std::strerror(errno);
        return false;
    }
    std::setvbuf(impl_->file, nullptr, _IOFBF, 1u << 20);
    impl_->start = std::chrono::steady_clock::now();
    impl_->next_sequence = 0;
    impl_->records_since_flush = 0;
    Bytes header;
    header.raw(kFileMagic, sizeof kFileMagic);
    header.u32(kVersion);
    header.u32(kEndian);
    if (!write_all(impl_->file, header.data.data(), header.data.size(), error)) {
        std::fclose(impl_->file);
        impl_->file = nullptr;
        return false;
    }
    Bytes payload;
    payload.string(metadata.revision);
    payload.string(metadata.title_id);
    payload.string(metadata.input_route);
    if (!append_record(impl_->file, RecordType::Metadata, impl_->next_sequence++, 0, payload, error) ||
        std::fflush(impl_->file) != 0) {
        if (error.empty()) error = std::string("timeline header flush failed: ") + std::strerror(errno);
        std::fclose(impl_->file);
        impl_->file = nullptr;
        return false;
    }
    return true;
}

bool GpuTimelineWriter::append_submit(const GpuTimelineSubmit& submit, std::string& error) {
    Bytes payload;
    payload.u64(submit.submit_no);
    payload.u64(submit.process_command_order);
    payload.u64(submit.first_command_order);
    payload.u64(submit.last_command_order);
    payload.u64(submit.color0_base);
    payload.u32(submit.draw_count);
    payload.u32(submit.dispatch_count);
    payload.u32(submit.color0_width);
    payload.u32(submit.color0_height);
    return impl_->append(RecordType::Submit, payload, error);
}

bool GpuTimelineWriter::append_present(const GpuTimelinePresent& present, std::string& error) {
    Bytes payload;
    payload.u64(present.present_count);
    payload.u64(present.latest_submit_no);
    payload.u32(std::bit_cast<uint32_t>(present.buffer_index));
    payload.u64(std::bit_cast<uint64_t>(present.flip_arg));
    payload.u32(present.width);
    payload.u32(present.height);
    return impl_->append(RecordType::Present, payload, error);
}

bool GpuTimelineWriter::flush(std::string& error) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->file) return true;
    if (std::fflush(impl_->file) == 0) {
        impl_->records_since_flush = 0;
        return true;
    }
    error = std::string("timeline flush failed: ") + std::strerror(errno);
    return false;
}

void GpuTimelineWriter::close() {
    if (!impl_) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->file) return;
    std::fflush(impl_->file);
    std::fclose(impl_->file);
    impl_->file = nullptr;
}

bool read_gpu_timeline(const std::string& path, GpuTimelineFile& timeline, std::string& error) {
    timeline = {};
    error.clear();
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open timeline";
        return false;
    }
    uint8_t file_header[16];
    bool clean_eof = false;
    if (!read_exact(file, file_header, sizeof file_header, clean_eof)) {
        error = "truncated timeline header";
        return false;
    }
    Cursor header{file_header, sizeof file_header};
    uint8_t magic[8];
    uint32_t version = 0, endian = 0;
    if (!header.take(magic, sizeof magic) || std::memcmp(magic, kFileMagic, sizeof magic) != 0 ||
        !header.u32(version) || !header.u32(endian)) {
        error = "invalid timeline header";
        return false;
    }
    if (version != kVersion) {
        error = "unsupported timeline version " + std::to_string(version);
        return false;
    }
    if (endian != kEndian) {
        error = "timeline endian marker mismatch";
        return false;
    }

    bool have_metadata = false;
    uint64_t expected_sequence = 0;
    for (;;) {
        uint8_t magic_record[4];
        if (!read_exact(file, magic_record, sizeof magic_record, clean_eof)) {
            if (clean_eof) break;
            timeline.truncated_tail = true;
            break;
        }
        if (std::memcmp(magic_record, kRecordMagic, sizeof magic_record) != 0) {
            error = "invalid timeline record magic";
            return false;
        }
        uint8_t record_header[24];
        if (!read_exact(file, record_header, sizeof record_header, clean_eof)) {
            timeline.truncated_tail = true;
            break;
        }
        Cursor rc{record_header, sizeof record_header};
        uint32_t type_raw = 0, payload_size = 0;
        uint64_t sequence = 0, elapsed_ns = 0;
        if (!rc.u32(type_raw) || !rc.u32(payload_size) || !rc.u64(sequence) || !rc.u64(elapsed_ns) ||
            payload_size > kMaxPayloadBytes) {
            error = "invalid timeline record header";
            return false;
        }
        if (sequence != expected_sequence++) {
            error = "non-contiguous timeline record sequence";
            return false;
        }
        std::vector<uint8_t> payload(payload_size);
        uint8_t checksum_bytes[8];
        if (!read_exact(file, payload.data(), payload.size(), clean_eof) ||
            !read_exact(file, checksum_bytes, sizeof checksum_bytes, clean_eof)) {
            timeline.truncated_tail = true;
            break;
        }
        Cursor checksum_cursor{checksum_bytes, sizeof checksum_bytes};
        uint64_t stored_checksum = 0;
        checksum_cursor.u64(stored_checksum);
        Bytes body;
        body.raw(record_header, sizeof record_header);
        body.raw(payload.data(), payload.size());
        if (hash_bytes(body.data.data(), body.data.size()) != stored_checksum) {
            error = "timeline record checksum mismatch at sequence " + std::to_string(sequence);
            return false;
        }

        Cursor p{payload.data(), payload.size()};
        const RecordType type = static_cast<RecordType>(type_raw);
        if (type == RecordType::Metadata) {
            if (have_metadata || sequence != 0 || !p.string(timeline.metadata.revision) ||
                !p.string(timeline.metadata.title_id) || !p.string(timeline.metadata.input_route) || p.left) {
                error = "invalid timeline metadata record";
                return false;
            }
            have_metadata = true;
        } else if (type == RecordType::Submit) {
            GpuTimelineSubmit submit;
            submit.sequence = sequence;
            submit.elapsed_ns = elapsed_ns;
            if (!p.u64(submit.submit_no) || !p.u64(submit.process_command_order) ||
                !p.u64(submit.first_command_order) || !p.u64(submit.last_command_order) ||
                !p.u64(submit.color0_base) || !p.u32(submit.draw_count) ||
                !p.u32(submit.dispatch_count) || !p.u32(submit.color0_width) ||
                !p.u32(submit.color0_height) || p.left) {
                error = "invalid timeline submit record";
                return false;
            }
            timeline.submits.push_back(submit);
        } else if (type == RecordType::Present) {
            GpuTimelinePresent present;
            present.sequence = sequence;
            present.elapsed_ns = elapsed_ns;
            uint32_t buffer_index = 0;
            uint64_t flip_arg = 0;
            if (!p.u64(present.present_count) || !p.u64(present.latest_submit_no) ||
                !p.u32(buffer_index) || !p.u64(flip_arg) || !p.u32(present.width) ||
                !p.u32(present.height) || p.left) {
                error = "invalid timeline present record";
                return false;
            }
            present.buffer_index = std::bit_cast<int32_t>(buffer_index);
            present.flip_arg = std::bit_cast<int64_t>(flip_arg);
            timeline.presents.push_back(present);
        } else {
            error = "unknown timeline record type " + std::to_string(type_raw);
            return false;
        }
    }
    if (!have_metadata) {
        error = "timeline has no metadata record";
        return false;
    }
    return true;
}

void begin_gpu_timeline_submit(uint64_t submit_no) {
    static const bool requested = [] {
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        return path && *path;
    }();
    if (requested) g_active_submit_no.store(submit_no, std::memory_order_release);
}

void record_gpu_timeline_submit(const GpuState& state, uint64_t submit_no) {
    static const bool requested = [] {
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        return path && *path;
    }();
    if (!requested) return;
    GpuTimelineWriter* writer = runtime_recorder().get();
    if (!writer) return;
    GpuTimelineSubmit submit;
    submit.submit_no = submit_no;
    submit.process_command_order = state.command_order;
    submit.draw_count = static_cast<uint32_t>(std::min<size_t>(state.draws.size(), UINT32_MAX));
    submit.dispatch_count = static_cast<uint32_t>(std::min<size_t>(state.dispatches.size(), UINT32_MAX));
    submit.first_command_order = std::numeric_limits<uint64_t>::max();
    for (const auto& draw : state.draws) {
        submit.first_command_order = std::min(submit.first_command_order, draw.command_order);
        submit.last_command_order = std::max(submit.last_command_order, draw.command_order);
    }
    for (const auto& dispatch : state.dispatches) {
        submit.first_command_order = std::min(submit.first_command_order, dispatch.command_order);
        submit.last_command_order = std::max(submit.last_command_order, dispatch.command_order);
    }
    if (submit.first_command_order == std::numeric_limits<uint64_t>::max())
        submit.first_command_order = submit.last_command_order = 0;
    if (!state.draws.empty()) {
        const RenderState rs = extract_render_state(state.state_at_draw(state.draws.size() - 1));
        submit.color0_base = rs.color0_base;
        submit.color0_width = rs.color0_width;
        submit.color0_height = rs.color0_height;
    }
    std::string error;
    if (!writer->append_submit(submit, error)) runtime_recorder().mark_failed(error);
}

void record_gpu_timeline_present(uint64_t present_count, int buffer_index, int64_t flip_arg,
                                 uint32_t width, uint32_t height) {
    static const bool requested = [] {
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        return path && *path;
    }();
    if (!requested) return;
    GpuTimelineWriter* writer = runtime_recorder().get();
    if (!writer) return;
    GpuTimelinePresent present;
    present.present_count = present_count;
    present.latest_submit_no = g_active_submit_no.load(std::memory_order_acquire);
    present.buffer_index = buffer_index;
    present.flip_arg = flip_arg;
    present.width = width;
    present.height = height;
    std::string error;
    if (!writer->append_present(present, error)) runtime_recorder().mark_failed(error);
}

void flush_gpu_timeline() {
    static const bool requested = [] {
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        return path && *path;
    }();
    if (!requested) return;
    GpuTimelineWriter* writer = runtime_recorder().get();
    if (!writer) return;
    std::string error;
    if (!writer->flush(error)) runtime_recorder().mark_failed(error);
}

void close_gpu_timeline() {
    static const bool requested = [] {
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        return path && *path;
    }();
    if (requested) runtime_recorder().close();
}

} // namespace prosper::gpu
