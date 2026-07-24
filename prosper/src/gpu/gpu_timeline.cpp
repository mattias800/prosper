#include "gpu_timeline.hpp"

#include "gpu_capture.hpp"
#include "gpu_capture_bundle.hpp"
#include "gpu_dependency_graph.hpp"
#include "pm4_registers.hpp"
#include "render_state.hpp"
#include "videoout_present.hpp"
#include "writer_provenance.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>

#ifdef __linux__
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace prosper::gpu {
namespace {

constexpr uint8_t kFileMagic[8] = {'P', 'R', 'G', 'T', 'L', 'N', '\0', '\0'};
constexpr uint8_t kRecordMagic[4] = {'T', 'L', 'R', 'C'};
constexpr uint32_t kVersion = 8;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint32_t kMaxPayloadBytes = 1u << 20;
constexpr uint32_t kFlushInterval = 256;
constexpr uint32_t kMaxTargetSpans = 16384;
constexpr uint32_t kMaxDmaCopies = 16384;

enum class RecordType : uint32_t { Metadata = 1, Submit = 2, Present = 3, Detail = 4, Producer = 5 };

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

bool valid_target_spans(const GpuTimelineSubmit& submit) {
    if (submit.target_spans.empty()) return true;
    uint64_t expected_first = 0;
    for (const auto& span : submit.target_spans) {
        if (!span.draw_count || span.first_draw != expected_first) return false;
        expected_first += span.draw_count;
        if (expected_first > submit.draw_count) return false;
    }
    return submit.target_spans_truncated || expected_first == submit.draw_count;
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

struct RuntimeDetailRequest {
    std::string path;
    std::string predecessor_path;
    std::string bundle_path;
    uint32_t bundle_depth = 0;
    uint64_t bundle_max_unique_bytes = 1ull << 30;
    uint32_t bundle_target_width = 0;
    uint32_t bundle_target_height = 0;
    uint32_t bundle_start_target_width = 0;
    uint32_t bundle_start_target_height = 0;
    uint32_t bundle_checkpoint_interval = 0;
    uint32_t select_target_width = 0;
    uint32_t select_target_height = 0;
    uint32_t select_target_min_index = 0;
    uint32_t select_target_max_index = UINT32_MAX;
    uint32_t select_min_draws = 0;
    uint32_t select_max_draws = UINT32_MAX;
    uint32_t select_min_dispatches = 0;
    uint32_t select_max_dispatches = UINT32_MAX;
    bool exit_after_capture = false;
    uint64_t submit_no = 0;
    bool valid = false;
    std::atomic<bool> claimed{false};
    std::atomic<bool> predecessor_claimed{false};
    std::atomic<bool> predecessor_failed{false};
    bool bundle_start_reached = false;

    RuntimeDetailRequest() {
        const char* path_env = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE");
        const char* submit_env = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT");
        if ((!path_env || !*path_env) && (!submit_env || !*submit_env)) return;
        if (!path_env || !*path_env || !submit_env || !*submit_env) {
            std::fprintf(stderr, "[timeline] detailed capture requires both "
                         "PROSPER_GPU_TIMELINE_CAPTURE and PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT\n");
            return;
        }
        char* end = nullptr;
        errno = 0;
        const uint64_t parsed = std::strtoull(submit_env, &end, 0);
        if (errno || !end || *end || !parsed) {
            std::fprintf(stderr, "[timeline] invalid detailed-capture submit '%s'\n", submit_env);
            return;
        }
        path = path_env;
        if (const char* predecessor = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR"))
            predecessor_path = predecessor;
        if (const char* bundle = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE")) {
            bundle_path = bundle;
            bundle_depth = 2;
            if (const char* depth = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_DEPTH")) {
                char* depth_end = nullptr;
                const uint64_t value = std::strtoull(depth, &depth_end, 0);
                if (!depth_end || *depth_end || value < 2 || value > 4096) {
                    std::fprintf(stderr, "[timeline] capture bundle depth must be 2..4096\n");
                    bundle_path.clear(); bundle_depth = 0;
                } else {
                    bundle_depth = static_cast<uint32_t>(value);
                }
            }
            if (const char* budget = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB")) {
                char* budget_end = nullptr;
                const uint64_t value = std::strtoull(budget, &budget_end, 0);
                if (!budget_end || *budget_end || value < 64 || value > 4096) {
                    std::fprintf(stderr, "[timeline] capture bundle budget must be 64..4096 MiB\n");
                    bundle_path.clear(); bundle_depth = 0;
                } else {
                    bundle_max_unique_bytes = value << 20;
                }
            }
            if (const char* dimensions = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DIM")) {
                unsigned width = 0, height = 0; char tail = 0;
                if (std::sscanf(dimensions, "%ux%u%c", &width, &height, &tail) != 2 ||
                    !width || !height) {
                    std::fprintf(stderr, "[timeline] capture target dimension must be WxH\n");
                    bundle_path.clear(); bundle_depth = 0;
                } else {
                    bundle_target_width = width; bundle_target_height = height;
                }
            }
            if (const char* dimensions = std::getenv(
                    "PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM")) {
                unsigned width = 0, height = 0; char tail = 0;
                if (std::sscanf(dimensions, "%ux%u%c", &width, &height, &tail) != 2 ||
                    !width || !height) {
                    std::fprintf(stderr, "[timeline] capture start target dimension must be WxH\n");
                    bundle_path.clear(); bundle_depth = 0;
                } else {
                    bundle_start_target_width = width;
                    bundle_start_target_height = height;
                }
            }
            if (const char* interval = std::getenv(
                    "PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY")) {
                char* interval_end = nullptr;
                const uint64_t value = std::strtoull(interval, &interval_end, 0);
                if (!interval_end || *interval_end || value < 1 || value > 4096) {
                    std::fprintf(stderr,
                                 "[timeline] capture bundle checkpoint interval must be 1..4096\n");
                    bundle_path.clear(); bundle_depth = 0;
                } else {
                    bundle_checkpoint_interval = static_cast<uint32_t>(value);
                }
            }
        }
        if (const char* dimensions = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM")) {
            unsigned width = 0, height = 0; char tail = 0;
            if (std::sscanf(dimensions, "%ux%u%c", &width, &height, &tail) != 2 ||
                !width || !height) {
                std::fprintf(stderr, "[timeline] capture endpoint target dimension must be WxH\n");
                return;
            }
            select_target_width = width;
            select_target_height = height;
        }
        if (const char* range = std::getenv(
                "PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX")) {
            unsigned first = 0, last = 0; char tail = 0;
            if (std::sscanf(range, "%u:%u%c", &first, &last, &tail) != 2 || first > last) {
                std::fprintf(stderr, "[timeline] target draw index must be MIN:MAX\n");
                return;
            }
            select_target_min_index = first;
            select_target_max_index = last;
        }
        if (const char* min_draws = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS")) {
            char* min_end = nullptr;
            const uint64_t value = std::strtoull(min_draws, &min_end, 0);
            if (!min_end || *min_end || value > UINT32_MAX) {
                std::fprintf(stderr, "[timeline] capture minimum draws must be 0..4294967295\n");
                return;
            }
            select_min_draws = static_cast<uint32_t>(value);
        }
        if (const char* max_draws = std::getenv("PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS")) {
            char* max_end = nullptr;
            const uint64_t value = std::strtoull(max_draws, &max_end, 0);
            if (!max_end || *max_end || value > UINT32_MAX) {
                std::fprintf(stderr, "[timeline] capture maximum draws must be 0..4294967295\n");
                return;
            }
            select_max_draws = static_cast<uint32_t>(value);
        }
        if (select_min_draws > select_max_draws) {
            std::fprintf(stderr, "[timeline] capture minimum draws exceeds maximum draws\n");
            return;
        }
        if (const char* min_dispatches = std::getenv(
                "PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES")) {
            char* min_end = nullptr;
            const uint64_t value = std::strtoull(min_dispatches, &min_end, 0);
            if (!min_end || *min_end || value > UINT32_MAX) {
                std::fprintf(stderr,
                             "[timeline] capture minimum dispatches must be 0..4294967295\n");
                return;
            }
            select_min_dispatches = static_cast<uint32_t>(value);
        }
        if (const char* max_dispatches = std::getenv(
                "PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES")) {
            char* max_end = nullptr;
            const uint64_t value = std::strtoull(max_dispatches, &max_end, 0);
            if (!max_end || *max_end || value > UINT32_MAX) {
                std::fprintf(stderr,
                             "[timeline] capture maximum dispatches must be 0..4294967295\n");
                return;
            }
            select_max_dispatches = static_cast<uint32_t>(value);
        }
        if (select_min_dispatches > select_max_dispatches) {
            std::fprintf(stderr,
                         "[timeline] capture minimum dispatches exceeds maximum dispatches\n");
            return;
        }
        if (const char* value = std::getenv("PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE"))
            exit_after_capture = *value && std::strcmp(value, "0") && std::strcmp(value, "off");
        submit_no = parsed;
        valid = true;
        if (select_target_width)
            std::fprintf(stderr, "[timeline] semantic capture endpoint submit>=%llu "
                                 "target=%ux%u target-index=%u..%u draws=%u..%u "
                                 "dispatches=%u..%u\n",
                         static_cast<unsigned long long>(submit_no), select_target_width,
                         select_target_height, select_target_min_index,
                         select_target_max_index, select_min_draws, select_max_draws,
                         select_min_dispatches, select_max_dispatches);
    }
};

RuntimeDetailRequest& runtime_detail_request() {
    static RuntimeDetailRequest request;
    return request;
}

struct RuntimeTargetWrite {
    uint64_t submit_no = 0;
    uint64_t draw_index = 0;
    uint64_t command_order = 0;
    uint64_t addr = 0;
    uint64_t size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    bool color_has_clear = false;
    uint32_t color_clear_word0 = 0;
    uint32_t color_clear_word1 = 0;
    uint32_t color_control = 0;
    uint32_t target_mask = 0;
    uint32_t color_format = 0;
};

struct RuntimeTargetLifetime {
    const RuntimeTargetWrite* first = nullptr;
    const RuntimeTargetWrite* last = nullptr;
    uint64_t write_count = 0;
    uint64_t submit_count = 0;
};

struct RuntimeTargetKey {
    uint64_t addr = 0;
    uint64_t size = 0;
    bool operator==(const RuntimeTargetKey& other) const {
        return addr == other.addr && size == other.size;
    }
};

struct RuntimeTargetKeyHash {
    size_t operator()(const RuntimeTargetKey& key) const {
        return static_cast<size_t>(key.addr ^ (key.addr >> 32) ^ key.size ^ (key.size >> 32));
    }
};

struct RuntimeTargetAggregate {
    RuntimeTargetWrite first;
    RuntimeTargetWrite last;
    uint64_t write_count = 0;
    uint64_t submit_count = 0;
    uint64_t last_counted_submit = 0;
};

struct RuntimeProducerHistory {
    std::deque<std::vector<RuntimeTargetWrite>> submits;
    size_t capacity = 64;
    bool enabled = false;
    uint64_t first_submit_no = 0;
    uint64_t dropped_submits = 0;
    std::unordered_map<RuntimeTargetKey, RuntimeTargetAggregate, RuntimeTargetKeyHash> lifetimes;
    bool lifetime_truncated = false;

    RuntimeProducerHistory() {
        enabled = runtime_detail_request().valid;
        if (const char* value = std::getenv("PROSPER_GPU_TIMELINE_HISTORY")) {
            char* end = nullptr;
            const uint64_t parsed = std::strtoull(value, &end, 0);
            if (end && !*end && parsed) capacity = static_cast<size_t>(std::min<uint64_t>(parsed, 65536));
        }
    }

    void remember(const GpuState& state, uint64_t submit_no) {
        if (!enabled) return;
        if (!first_submit_no) first_submit_no = submit_no;
        std::vector<RuntimeTargetWrite> writes;
        writes.reserve(state.draws.size() * 2);
        for (size_t i = 0; i < state.draws.size(); ++i) {
            const RenderState rs = extract_render_state(state.state_at_draw(i));
            auto remember_target = [&](uint64_t base, uint32_t width, uint32_t height,
                                       bool has_clear, uint32_t clear0, uint32_t clear1,
                                       uint32_t format) {
                if (!base || !width || !height) return;
                writes.push_back({submit_no, i, state.draws[i].command_order, base,
                                  static_cast<uint64_t>(width) * height * 4,
                                  width, height, has_clear, clear0, clear1,
                                  rs.cb_color_control, rs.cb_target_mask, format});
                const RuntimeTargetWrite& write = writes.back();
                const RuntimeTargetKey key{write.addr, write.size};
                auto found = lifetimes.find(key);
                if (found == lifetimes.end()) {
                    if (lifetimes.size() >= 65536) {
                        lifetime_truncated = true;
                    } else {
                        RuntimeTargetAggregate aggregate;
                        aggregate.first = aggregate.last = write;
                        aggregate.write_count = aggregate.submit_count = 1;
                        aggregate.last_counted_submit = submit_no;
                        lifetimes.emplace(key, std::move(aggregate));
                    }
                } else {
                    RuntimeTargetAggregate& aggregate = found->second;
                    aggregate.last = write;
                    ++aggregate.write_count;
                    if (aggregate.last_counted_submit != submit_no) {
                        ++aggregate.submit_count;
                        aggregate.last_counted_submit = submit_no;
                    }
                }
            };
            remember_target(rs.color0_base, rs.color0_width, rs.color0_height,
                            rs.color0_has_clear, rs.color0_clear_word0,
                            rs.color0_clear_word1, rs.color0_format);
            remember_target(rs.color1_base, rs.color1_width, rs.color1_height,
                            rs.color1_has_clear, rs.color1_clear_word0,
                            rs.color1_clear_word1, rs.color1_format);
        }
        submits.push_back(std::move(writes));
        while (submits.size() > capacity) { submits.pop_front(); ++dropped_submits; }
    }

    const RuntimeTargetWrite* latest_overlap(uint64_t addr, uint64_t size) const {
        auto end = [](uint64_t base, uint64_t bytes) {
            return bytes > UINT64_MAX - base ? UINT64_MAX : base + bytes;
        };
        for (auto submit = submits.rbegin(); submit != submits.rend(); ++submit)
            for (auto write = submit->rbegin(); write != submit->rend(); ++write)
                if (addr < end(write->addr, write->size) && write->addr < end(addr, size))
                    return &*write;
        return nullptr;
    }

    RuntimeTargetLifetime lifetime_overlap(uint64_t addr, uint64_t size) const {
        auto end = [](uint64_t base, uint64_t bytes) {
            return bytes > UINT64_MAX - base ? UINT64_MAX : base + bytes;
        };
        RuntimeTargetLifetime lifetime;
        for (const auto& [key, aggregate] : lifetimes) {
            if (addr >= end(key.addr, key.size) || key.addr >= end(addr, size)) continue;
            if (!lifetime.first || aggregate.first.submit_no < lifetime.first->submit_no ||
                (aggregate.first.submit_no == lifetime.first->submit_no &&
                 aggregate.first.command_order < lifetime.first->command_order))
                lifetime.first = &aggregate.first;
            if (!lifetime.last || aggregate.last.submit_no > lifetime.last->submit_no ||
                (aggregate.last.submit_no == lifetime.last->submit_no &&
                 aggregate.last.command_order > lifetime.last->command_order))
                lifetime.last = &aggregate.last;
            lifetime.write_count += aggregate.write_count;
            lifetime.submit_count += aggregate.submit_count;
        }
        return lifetime;
    }

    uint64_t window_first_submit_no() const {
        return first_submit_no ? first_submit_no + dropped_submits : 0;
    }
};

RuntimeProducerHistory& runtime_producer_history() {
    static RuntimeProducerHistory history;
    return history;
}

struct RuntimeCaptureBundle {
    GpuCaptureBundle bundle;
    bool budget_exhausted = false;
    bool failed = false;
    bool started = false;
    std::chrono::steady_clock::time_point started_at;
    uint64_t captured_resource_bytes = 0;
    uint64_t captured_submit_count = 0;
};

RuntimeCaptureBundle& runtime_capture_bundle() {
    static RuntimeCaptureBundle state;
    return state;
}

bool runtime_capture_endpoint_matches(const RuntimeDetailRequest& request,
                                      const GpuTimelineSubmit& submit) {
    const uint64_t submit_no = submit.submit_no;
    if (submit_no < request.submit_no) return false;
    if (!request.select_target_width)
        return submit_no == request.submit_no;
    GpuTimelineSelector selector;
    selector.min_submit_no = request.submit_no;
    selector.target_width = request.select_target_width;
    selector.target_height = request.select_target_height;
    selector.target_min_draw = request.select_target_min_index;
    selector.target_max_draw = request.select_target_max_index;
    selector.min_draws = request.select_min_draws;
    selector.max_draws = request.select_max_draws;
    selector.min_dispatches = request.select_min_dispatches;
    selector.max_dispatches = request.select_max_dispatches;
    if (!gpu_timeline_submit_matches(submit, selector)) return false;
    std::fprintf(stderr, "[timeline] semantic capture selector matched submit=%llu\n",
                 static_cast<unsigned long long>(submit_no));
    return true;
}

bool same_depth_surface(const GpuTimelineDepthSurface& a, const GpuTimelineDepthSurface& b) {
    return std::tie(a.depth_read_base, a.depth_write_base,
                    a.stencil_read_base, a.stencil_write_base, a.htile_data_base,
                    a.db_depth_view, a.db_render_override, a.db_render_override2,
                    a.db_depth_size_xy, a.db_dfsm_control, a.db_depth_info,
                    a.db_z_info, a.db_stencil_info, a.db_depth_size, a.db_depth_slice,
                    a.db_htile_surface, a.db_rmi_l2_cache_control,
                    a.target_width, a.target_height) ==
           std::tie(b.depth_read_base, b.depth_write_base,
                    b.stencil_read_base, b.stencil_write_base, b.htile_data_base,
                    b.db_depth_view, b.db_render_override, b.db_render_override2,
                    b.db_depth_size_xy, b.db_dfsm_control, b.db_depth_info,
                    b.db_z_info, b.db_stencil_info, b.db_depth_size, b.db_depth_slice,
                    b.db_htile_surface, b.db_rmi_l2_cache_control,
                    b.target_width, b.target_height);
}

bool hash_guest_backing(uint64_t addr, uint64_t size, uint64_t& hash) {
    constexpr uint64_t kMaxHashBytes = 64ull << 20;
    if (!addr || !size || size > kMaxHashBytes || size > UINT32_MAX) return false;
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
#ifdef __linux__
    size_t done = 0;
    while (done < bytes.size()) {
        iovec local{bytes.data() + done, bytes.size() - done};
        iovec remote{reinterpret_cast<void*>(static_cast<uintptr_t>(addr + done)),
                     bytes.size() - done};
        ssize_t copied;
        do {
            copied = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
        } while (copied < 0 && errno == EINTR);
        if (copied <= 0) return false;
        done += static_cast<size_t>(copied);
    }
#else
    if (!guest_readable(addr, static_cast<uint32_t>(size))) return false;
    std::memcpy(bytes.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(addr)), bytes.size());
#endif
    hash = gpu_capture_hash(bytes);
    return true;
}

bool runtime_submit_has_target(const GpuState& state, uint32_t width, uint32_t height) {
    if (!width || !height) return false;
    for (size_t i = 0; i < state.draws.size(); ++i) {
        const RenderState rs = extract_render_state(state.state_at_draw(i));
        if (rs.color0_width == width && rs.color0_height == height) return true;
    }
    return false;
}

void begin_runtime_capture_bundle() {
    RuntimeCaptureBundle& state = runtime_capture_bundle();
    if (state.started) return;
    state.started = true;
    state.started_at = std::chrono::steady_clock::now();
}

bool append_runtime_capture_bundle(const GpuCaptureFile& capture, uint64_t max_unique_bytes,
                                   std::string& error) {
    RuntimeCaptureBundle& state = runtime_capture_bundle();
    if (state.budget_exhausted) { error = "capture bundle unique-byte budget was already exhausted"; return false; }
    const size_t chunks = state.bundle.chunks.size();
    const size_t hashes = state.bundle.chunk_hashes.size();
    const size_t resources = state.bundle.resources.size();
    const size_t submits = state.bundle.submits.size();
    const uint64_t logical = state.bundle.logical_bytes;
    if (!append_gpu_capture_bundle(state.bundle, capture, error)) return false;
    if (gpu_capture_bundle_unique_bytes(state.bundle) <= max_unique_bytes) {
        for (const auto& blob : capture.blobs) state.captured_resource_bytes += blob.bytes.size();
        return true;
    }
    state.bundle.chunks.resize(chunks); state.bundle.chunk_hashes.resize(hashes);
    state.bundle.resources.resize(resources);
    state.bundle.submits.resize(submits); state.bundle.logical_bytes = logical;
    state.bundle.chunk_indices_by_hash.clear();
    state.bundle.resource_indices_by_hash.clear();
    state.budget_exhausted = true;
    error = "capture bundle exceeded its unique-byte budget";
    return false;
}

void trim_runtime_capture_bundle(size_t max_submits) {
    GpuCaptureBundle& bundle = runtime_capture_bundle().bundle;
    while (bundle.submits.size() > max_submits) {
        bundle.logical_bytes -= bundle.submits.front().logical_bytes;
        bundle.submits.erase(bundle.submits.begin());
    }
}

GpuCaptureMetadata runtime_capture_metadata(uint64_t submit_no) {
    GpuCaptureMetadata metadata;
    metadata.width = present_width();
    metadata.height = present_height();
    metadata.submit_index = submit_no;
#ifdef PROSPER_GIT_REVISION
    metadata.revision = PROSPER_GIT_REVISION;
#else
    metadata.revision = "unknown";
#endif
    if (const char* revision = std::getenv("PROSPER_CAPTURE_REVISION")) metadata.revision = revision;
    metadata.title_id = env_or_empty("PROSPER_CAPTURE_TITLE");
    metadata.input_route = env_or_empty("PROSPER_PAD_SCRIPT");
    metadata.savedata_dir = env_or_empty("PROSPER_SAVEDATA_DIR");
    return metadata;
}

GpuTimelineDetail capture_detail(const GpuTimelineSubmit& submit, const std::string& path,
                                 const GpuCaptureFile& capture) {
    GpuTimelineDetail detail;
    detail.submit_no = submit.submit_no;
    detail.capture_path = path;
    detail.semantic_draw_count = submit.draw_count;
    detail.semantic_dispatch_count = submit.dispatch_count;
    detail.realized_draw_count = static_cast<uint32_t>(capture.draws.size());
    detail.realized_dispatch_count = static_cast<uint32_t>(capture.computes.size());
    detail.operation_count = static_cast<uint32_t>(capture.operations.size());
    detail.missing_operation_count = static_cast<uint32_t>(std::count_if(
        capture.operations.begin(), capture.operations.end(),
        [](const auto& operation) { return !operation.realized; }));
    detail.shader_version_count = static_cast<uint32_t>(capture.shader_versions.size());
    detail.resource_version_count = static_cast<uint32_t>(capture.blobs.size());
    for (const auto& blob : capture.blobs) detail.resource_bytes += blob.bytes.size();
    return detail;
}

void log_capture_detail(const GpuTimelineDetail& detail) {
    std::fprintf(stderr, "[timeline] captured submit %llu: draws=%u/%u dispatches=%u/%u "
                         "versions=%u shaders/%u resources (%llu bytes) -> %s\n",
                 static_cast<unsigned long long>(detail.submit_no), detail.realized_draw_count,
                 detail.semantic_draw_count, detail.realized_dispatch_count,
                 detail.semantic_dispatch_count, detail.shader_version_count,
                 detail.resource_version_count, static_cast<unsigned long long>(detail.resource_bytes),
                 detail.capture_path.c_str());
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
    if (submit.depth_surfaces.size() > 65536) {
        error = "timeline submit has too many depth surfaces";
        return false;
    }
    if (submit.target_spans.size() > kMaxTargetSpans) {
        error = "timeline submit has too many target spans";
        return false;
    }
    if (!valid_target_spans(submit)) {
        error = "timeline submit has invalid target spans";
        return false;
    }
    if (submit.dma_copies.size() > kMaxDmaCopies ||
        submit.dma_copy_count != submit.dma_copies.size()) {
        error = "timeline submit has invalid ordered DMA records";
        return false;
    }
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
    // v7 prefixes its optional tail with operation support metadata. v8 appends exact DMA records
    // after the v6 target-span suffix, leaving the complete v1-v7 prefix byte-compatible.
    if (!submit.depth_surfaces.empty() || !submit.target_spans.empty() ||
        submit.target_spans_truncated || submit.dma_copy_count || submit.capture_incomplete ||
        !submit.dma_copies.empty()) {
        payload.u32(submit.dma_copy_count);
        payload.u32(submit.capture_incomplete ? 1u : 0u);
        payload.u32(static_cast<uint32_t>(submit.depth_surfaces.size()));
        for (const auto& ds : submit.depth_surfaces) {
            payload.u64(ds.depth_read_base); payload.u64(ds.depth_write_base);
            payload.u64(ds.stencil_read_base); payload.u64(ds.stencil_write_base);
            payload.u64(ds.htile_data_base);
            payload.u32(ds.db_depth_view); payload.u32(ds.db_render_override);
            payload.u32(ds.db_render_override2); payload.u32(ds.db_depth_size_xy);
            payload.u32(ds.db_dfsm_control); payload.u32(ds.db_depth_info);
            payload.u32(ds.db_z_info); payload.u32(ds.db_stencil_info);
            payload.u32(ds.db_depth_size); payload.u32(ds.db_depth_slice);
            payload.u32(ds.db_htile_surface); payload.u32(ds.db_rmi_l2_cache_control);
            payload.u32(ds.target_width); payload.u32(ds.target_height);
            payload.u32(ds.draw_count); payload.u32(ds.depth_test_count);
            payload.u32(ds.depth_write_count); payload.u32(ds.clear_count);
            payload.u32(ds.compare_mask);
            payload.u32(ds.backing_hash_mask); payload.u64(ds.depth_backing_hash);
            payload.u64(ds.stencil_backing_hash); payload.u64(ds.htile_backing_hash);
            payload.u32(ds.backing_writer_kind); payload.u64(ds.backing_writer_sequence);
            payload.u64(ds.backing_writer_addr); payload.u64(ds.backing_writer_size);
            payload.u64(ds.backing_writer_order); payload.u64(ds.backing_writer_identity);
        }
        payload.u32(static_cast<uint32_t>(submit.target_spans.size()));
        for (const auto& span : submit.target_spans) {
            payload.u32(span.first_draw); payload.u32(span.draw_count);
            payload.u32(span.width); payload.u32(span.height);
        }
        payload.u32(submit.target_spans_truncated ? 1u : 0u);
        payload.u32(static_cast<uint32_t>(submit.dma_copies.size()));
        for (const auto& copy : submit.dma_copies) {
            payload.u64(copy.dst); payload.u64(copy.src); payload.u32(copy.bytes);
            payload.u32(copy.sels); payload.u64(copy.command_order);
            payload.u64(copy.packet_addr);
        }
    }
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

bool GpuTimelineWriter::append_detail(const GpuTimelineDetail& detail, std::string& error) {
    Bytes payload;
    payload.u64(detail.submit_no);
    payload.string(detail.capture_path);
    payload.u32(detail.semantic_draw_count);
    payload.u32(detail.semantic_dispatch_count);
    payload.u32(detail.realized_draw_count);
    payload.u32(detail.realized_dispatch_count);
    payload.u32(detail.operation_count);
    payload.u32(detail.missing_operation_count);
    payload.u32(detail.shader_version_count);
    payload.u32(detail.resource_version_count);
    payload.u64(detail.resource_bytes);
    return impl_->append(RecordType::Detail, payload, error);
}

bool GpuTimelineWriter::append_producer(const GpuTimelineProducer& producer, std::string& error) {
    Bytes payload;
    payload.u64(producer.consumer_submit_no);
    payload.u32(producer.consumer_operation);
    payload.u32(producer.future_writer_operation);
    payload.u64(producer.resource_addr);
    payload.u64(producer.resource_size);
    payload.u32(producer.resource_width);
    payload.u32(producer.resource_height);
    payload.u32(producer.resolved ? 1u : 0u);
    payload.u64(producer.producer_submit_no);
    payload.u64(producer.producer_draw_index);
    payload.u64(producer.producer_command_order);
    payload.u64(producer.producer_target_addr);
    payload.u32(producer.producer_width);
    payload.u32(producer.producer_height);
    payload.u32(static_cast<uint32_t>(producer.first_writer_kind));
    payload.u64(producer.history_first_submit_no);
    payload.u64(producer.history_first_draw_index);
    payload.u64(producer.history_first_command_order);
    payload.u64(producer.history_write_count);
    payload.u64(producer.history_submit_count);
    payload.u64(producer.history_window_first_submit_no);
    payload.u32(producer.lifetime_truncated ? 1u : 0u);
    payload.u32(producer.history_window_truncated ? 1u : 0u);
    payload.u32(producer.first_color_has_clear ? 1u : 0u);
    payload.u32(producer.first_color_clear_word0);
    payload.u32(producer.first_color_clear_word1);
    payload.u32(producer.first_color_control);
    payload.u32(producer.first_color_control_mode);
    payload.u32(producer.first_target_mask);
    payload.u32(producer.first_color_format);
    return impl_->append(RecordType::Producer, payload, error);
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
    if (version < 1 || version > kVersion) {
        error = "unsupported timeline version " + std::to_string(version);
        return false;
    }
    timeline.version = version;
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
                !p.u32(submit.color0_height)) {
                error = "invalid timeline submit record";
                return false;
            }
            const bool have_submit_tail = p.left != 0;
            if (version >= 7 && have_submit_tail) {
                uint32_t incomplete = 0;
                if (!p.u32(submit.dma_copy_count) || !p.u32(incomplete) || incomplete > 1) {
                    error = "invalid timeline unsupported-operation metadata";
                    return false;
                }
                submit.capture_incomplete = incomplete != 0;
            }
            if (version >= 5 && have_submit_tail) {
                uint32_t surface_count = 0;
                if (!p.u32(surface_count) || surface_count > 65536) {
                    error = "invalid timeline depth-surface count";
                    return false;
                }
                submit.depth_surfaces.resize(surface_count);
                for (auto& ds : submit.depth_surfaces) {
                    if (!p.u64(ds.depth_read_base) || !p.u64(ds.depth_write_base) ||
                        !p.u64(ds.stencil_read_base) || !p.u64(ds.stencil_write_base) ||
                        !p.u64(ds.htile_data_base) || !p.u32(ds.db_depth_view) ||
                        !p.u32(ds.db_render_override) || !p.u32(ds.db_render_override2) ||
                        !p.u32(ds.db_depth_size_xy) || !p.u32(ds.db_dfsm_control) ||
                        !p.u32(ds.db_depth_info) || !p.u32(ds.db_z_info) ||
                        !p.u32(ds.db_stencil_info) || !p.u32(ds.db_depth_size) ||
                        !p.u32(ds.db_depth_slice) || !p.u32(ds.db_htile_surface) ||
                        !p.u32(ds.db_rmi_l2_cache_control) || !p.u32(ds.target_width) ||
                        !p.u32(ds.target_height) || !p.u32(ds.draw_count) ||
                        !p.u32(ds.depth_test_count) || !p.u32(ds.depth_write_count) ||
                        !p.u32(ds.clear_count) || !p.u32(ds.compare_mask) ||
                        !p.u32(ds.backing_hash_mask) || !p.u64(ds.depth_backing_hash) ||
                        !p.u64(ds.stencil_backing_hash) || !p.u64(ds.htile_backing_hash) ||
                        !p.u32(ds.backing_writer_kind) || !p.u64(ds.backing_writer_sequence) ||
                        !p.u64(ds.backing_writer_addr) || !p.u64(ds.backing_writer_size) ||
                        !p.u64(ds.backing_writer_order) || !p.u64(ds.backing_writer_identity)) {
                        error = "invalid timeline depth-surface record";
                        return false;
                    }
                }
            }
            if (version >= 6 && have_submit_tail) {
                uint32_t span_count = 0, truncated = 0;
                if (!p.u32(span_count) || span_count > kMaxTargetSpans) {
                    error = "invalid timeline target-span count";
                    return false;
                }
                submit.target_spans.resize(span_count);
                for (auto& span : submit.target_spans) {
                    if (!p.u32(span.first_draw) || !p.u32(span.draw_count) ||
                        !p.u32(span.width) || !p.u32(span.height) || !span.draw_count) {
                        error = "invalid timeline target-span record";
                        return false;
                    }
                }
                if (!p.u32(truncated) || truncated > 1) {
                    error = "invalid timeline target-span truncation flag";
                    return false;
                }
                submit.target_spans_truncated = truncated != 0;
                if (!valid_target_spans(submit)) {
                    error = "invalid timeline target-span coverage";
                    return false;
                }
            }
            if (version >= 8 && have_submit_tail) {
                uint32_t dma_count = 0;
                if (!p.u32(dma_count) || dma_count > kMaxDmaCopies ||
                    dma_count != submit.dma_copy_count) {
                    error = "invalid timeline ordered DMA count";
                    return false;
                }
                submit.dma_copies.resize(dma_count);
                for (auto& copy : submit.dma_copies) {
                    if (!p.u64(copy.dst) || !p.u64(copy.src) || !p.u32(copy.bytes) ||
                        !p.u32(copy.sels) || !p.u64(copy.command_order) ||
                        !p.u64(copy.packet_addr) || !copy.dst || !copy.src || !copy.bytes ||
                        copy.dst > std::numeric_limits<uint64_t>::max() - copy.bytes ||
                        copy.src > std::numeric_limits<uint64_t>::max() - copy.bytes) {
                        error = "invalid timeline ordered DMA record";
                        return false;
                    }
                }
            }
            if (p.left) {
                error = "invalid timeline submit record size";
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
        } else if (type == RecordType::Detail && version >= 2) {
            GpuTimelineDetail detail;
            detail.sequence = sequence;
            detail.elapsed_ns = elapsed_ns;
            if (!p.u64(detail.submit_no) || !p.string(detail.capture_path) ||
                !p.u32(detail.semantic_draw_count) || !p.u32(detail.semantic_dispatch_count) ||
                !p.u32(detail.realized_draw_count) || !p.u32(detail.realized_dispatch_count) ||
                !p.u32(detail.operation_count) || !p.u32(detail.missing_operation_count) ||
                !p.u32(detail.shader_version_count) || !p.u32(detail.resource_version_count) ||
                !p.u64(detail.resource_bytes) || p.left) {
                error = "invalid timeline detail record";
                return false;
            }
            timeline.details.push_back(std::move(detail));
        } else if (type == RecordType::Producer && version >= 3) {
            GpuTimelineProducer producer;
            producer.sequence = sequence;
            producer.elapsed_ns = elapsed_ns;
            uint32_t resolved = 0, writer_kind = 0, lifetime_truncated = 0;
            uint32_t window_truncated = 0, has_clear = 0;
            if (!p.u64(producer.consumer_submit_no) || !p.u32(producer.consumer_operation) ||
                !p.u32(producer.future_writer_operation) || !p.u64(producer.resource_addr) ||
                !p.u64(producer.resource_size) || !p.u32(producer.resource_width) ||
                !p.u32(producer.resource_height) || !p.u32(resolved) ||
                !p.u64(producer.producer_submit_no) || !p.u64(producer.producer_draw_index) ||
                !p.u64(producer.producer_command_order) || !p.u64(producer.producer_target_addr) ||
                !p.u32(producer.producer_width) || !p.u32(producer.producer_height) ||
                resolved > 1) {
                error = "invalid timeline producer record";
                return false;
            }
            producer.resolved = resolved != 0;
            if (version >= 4) {
                if (!p.u32(writer_kind) || writer_kind > static_cast<uint32_t>(GpuTimelineWriterKind::WriteData) ||
                    !p.u64(producer.history_first_submit_no) ||
                    !p.u64(producer.history_first_draw_index) ||
                    !p.u64(producer.history_first_command_order) ||
                    !p.u64(producer.history_write_count) || !p.u64(producer.history_submit_count) ||
                    !p.u64(producer.history_window_first_submit_no) || !p.u32(lifetime_truncated) ||
                    !p.u32(window_truncated) ||
                    !p.u32(has_clear) || !p.u32(producer.first_color_clear_word0) ||
                    !p.u32(producer.first_color_clear_word1) || !p.u32(producer.first_color_control) ||
                    !p.u32(producer.first_color_control_mode) || !p.u32(producer.first_target_mask) ||
                    !p.u32(producer.first_color_format) || lifetime_truncated > 1 ||
                    window_truncated > 1 || has_clear > 1) {
                    error = "invalid timeline producer lifetime record";
                    return false;
                }
                producer.first_writer_kind = static_cast<GpuTimelineWriterKind>(writer_kind);
                producer.lifetime_truncated = lifetime_truncated != 0;
                producer.history_window_truncated = window_truncated != 0;
                producer.first_color_has_clear = has_clear != 0;
            }
            if (p.left) { error = "invalid timeline producer record size"; return false; }
            timeline.producers.push_back(std::move(producer));
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

bool gpu_timeline_submit_matches(const GpuTimelineSubmit& submit,
                                 const GpuTimelineSelector& selector) {
    if (submit.submit_no < selector.min_submit_no ||
        submit.draw_count < selector.min_draws || submit.draw_count > selector.max_draws ||
        submit.dispatch_count < selector.min_dispatches ||
        submit.dispatch_count > selector.max_dispatches)
        return false;
    if (!selector.target_width) return true;
    if (!selector.target_height || submit.target_spans_truncated) return false;
    for (const auto& span : submit.target_spans) {
        if (span.width != selector.target_width || span.height != selector.target_height ||
            !span.draw_count)
            continue;
        const uint64_t first = span.first_draw;
        const uint64_t last = first + span.draw_count - 1;
        if (last >= selector.target_min_draw && first <= selector.target_max_draw)
            return true;
    }
    return false;
}

// PROSPER_GPU_TIMELINE_MRT_SUBMIT=N, or the MRT_{MIN,MAX}_{DRAWS,DISPATCHES} predicates: print the
// complete raw color-target programming for every semantic draw in one selected submit. Timeline
// target spans intentionally retain only MRT0's extent, which is enough for scene selection but
// cannot prove whether a sampled GPU-only surface was produced through MRT1..7. This bounded
// diagnostic reads the draw-time register snapshots and never realizes shaders, copies resources,
// or invokes Vulkan. Semantic predicates select only their first match so a title loop stays bounded.
bool select_timeline_mrt_submit(const GpuState& state, uint64_t submit_no) {
    struct Config {
        uint64_t exact = 0;
        size_t min_draws = 0, max_draws = std::numeric_limits<size_t>::max();
        size_t min_dispatches = 0, max_dispatches = std::numeric_limits<size_t>::max();
        bool semantic = false;
    };
    static const Config config = [] {
        Config out;
        auto read = [&](const char* name, size_t& field) {
            const char* value = std::getenv(name);
            if (!value || !*value) return;
            field = static_cast<size_t>(std::strtoull(value, nullptr, 0));
            out.semantic = true;
        };
        if (const char* value = std::getenv("PROSPER_GPU_TIMELINE_MRT_SUBMIT"); value && *value)
            out.exact = std::strtoull(value, nullptr, 0);
        read("PROSPER_GPU_TIMELINE_MRT_MIN_DRAWS", out.min_draws);
        read("PROSPER_GPU_TIMELINE_MRT_MAX_DRAWS", out.max_draws);
        read("PROSPER_GPU_TIMELINE_MRT_MIN_DISPATCHES", out.min_dispatches);
        read("PROSPER_GPU_TIMELINE_MRT_MAX_DISPATCHES", out.max_dispatches);
        return out;
    }();
    if (config.exact) return submit_no == config.exact;
    if (!config.semantic || state.draws.size() < config.min_draws ||
        state.draws.size() > config.max_draws ||
        state.dispatches.size() < config.min_dispatches ||
        state.dispatches.size() > config.max_dispatches)
        return false;
    static std::atomic<uint64_t> selected{0};
    uint64_t expected = 0;
    selected.compare_exchange_strong(expected, submit_no, std::memory_order_relaxed);
    return selected.load(std::memory_order_relaxed) == submit_no;
}

void log_timeline_mrt_draw(const GpuState& draw_state, uint64_t submit_no,
                           size_t draw_index, uint64_t command_order, bool selected) {
    if (!selected) return;
    namespace P = prosper::agc::Pm4;
    auto read = [&](uint32_t reg) {
        const auto found = draw_state.cx.find(reg);
        return found == draw_state.cx.end() ? 0u : found->second;
    };
    const bool has_target_mask = draw_state.cx.count(P::CB_TARGET_MASK) != 0;
    const bool has_shader_mask = draw_state.cx.count(P::CB_SHADER_MASK) != 0;
    const uint32_t target_mask = has_target_mask ? read(P::CB_TARGET_MASK) : 0xffffffffu;
    const uint32_t shader_mask = read(P::CB_SHADER_MASK);

    std::fprintf(stderr,
                 "[timeline-mrt] submit=%llu draw=%zu order=%llu target-mask=%08x%s "
                 "shader-mask=%08x%s",
                 static_cast<unsigned long long>(submit_no), draw_index,
                 static_cast<unsigned long long>(command_order), target_mask,
                 has_target_mask ? "" : "(default)", shader_mask,
                 has_shader_mask ? "" : "(absent)");
    for (uint32_t slot = 0; slot < 8; ++slot) {
        constexpr uint32_t kColorRegisterStride = 0xf;
        const uint32_t base_reg = P::CB_COLOR0_BASE + slot * kColorRegisterStride;
        const uint32_t info_reg = P::CB_COLOR0_INFO + slot * kColorRegisterStride;
        const uint32_t attrib2_reg = P::CB_COLOR0_ATTRIB2 + slot;
        const uint32_t base = read(base_reg);
        const uint32_t base_ext = read(P::CB_COLOR0_BASE_EXT + slot);
        const uint32_t info = read(info_reg);
        const auto attrib2 = draw_state.cx.find(attrib2_reg);
        const uint32_t write_mask = (target_mask >> (slot * 4)) & 0xfu;
        const uint32_t export_mask = (shader_mask >> (slot * 4)) & 0xfu;
        // An absent CB_TARGET_MASK defaults to all ones, but that does not bind seven zero-address
        // targets. Report only a programmed surface or shader export, while still showing its mask.
        if (!base && !base_ext && !info && !export_mask) continue;
        const uint64_t address = (static_cast<uint64_t>(base) << 8) |
                                 (static_cast<uint64_t>(base_ext & 0xffu) << 40);
        const uint32_t format = PM4_FIELD(info, CB_COLOR0_INFO, FORMAT);
        std::fprintf(stderr, " c%u=0x%llx/f%u/w%x/e%x", slot,
                     static_cast<unsigned long long>(address), format, write_mask, export_mask);
        if (attrib2 != draw_state.cx.end()) {
            const uint32_t width = PM4_FIELD(attrib2->second, CB_COLOR0_ATTRIB2, MIP0_WIDTH) + 1u;
            const uint32_t height = PM4_FIELD(attrib2->second, CB_COLOR0_ATTRIB2, MIP0_HEIGHT) + 1u;
            std::fprintf(stderr, "/%ux%u", width, height);
        }
    }
    std::fputc('\n', stderr);
}

void begin_gpu_timeline_submit(uint64_t submit_no) {
    static const bool requested = [] {
        const char* path = std::getenv("PROSPER_GPU_TIMELINE");
        return path && *path;
    }();
    if (requested) g_active_submit_no.store(submit_no, std::memory_order_release);
}

// ---- Interactive frame-bundle capture (prosper-app F9) ---------------------------------------------
// Capture ONE complete displayed frame — every submit between two presents — on demand, so the produced
// .prgbundle replays faithfully (the producer submits re-run and regenerate renderer-owned RTTs a single
// .prgcap leaves black). State machine: F9 arms `armed_path`; the NEXT present promotes it to a fresh
// capturing frame; each submit appends; the following present writes the bundle and disarms. The hot
// per-submit/per-present hooks are guarded by g_interactive_frame_active (one atomic load) so normal
// play pays nothing until F9 is pressed.
namespace {
struct InteractiveFrameBundle {
    std::mutex mx;
    std::string armed_path;      // set by F9; the next present begins the window
    std::string current_path;    // path for the window currently being captured
    bool capturing = false;
    bool failed = false;
    uint64_t max_unique_bytes = 2048ull << 20;
    uint32_t frames_wanted = 1;  // one frame suffices: the capture seeds the sampled renderer-owned RTTs
                                 // with their live pixels (#1291), so a deferred/temporal-AA frame replays
                                 // faithfully from a single submit — no need to re-run producers across a
                                 // multi-frame window. Raise PROSPER_CAPTURE_FRAMES only to grab an
                                 // animation over several frames; each extra frame is a full heavy capture.
    uint32_t frames_seen = 0;    // presents observed while capturing
    uint64_t submits = 0;
    GpuCaptureBundle bundle;
};
InteractiveFrameBundle& interactive_frame_bundle() { static InteractiveFrameBundle b; return b; }
std::atomic<bool> g_interactive_frame_active{false};   // armed OR capturing

// Append one capture to `bundle` with content dedup, rolling back if it would exceed the byte budget.
bool append_capture_to_frame_bundle(GpuCaptureBundle& bundle, const GpuCaptureFile& capture,
                                    uint64_t max_unique_bytes, std::string& error) {
    const size_t chunks = bundle.chunks.size(), hashes = bundle.chunk_hashes.size(),
                 resources = bundle.resources.size(), submits = bundle.submits.size();
    const uint64_t logical = bundle.logical_bytes;
    if (!append_gpu_capture_bundle(bundle, capture, error)) return false;
    if (gpu_capture_bundle_unique_bytes(bundle) <= max_unique_bytes) return true;
    bundle.chunks.resize(chunks); bundle.chunk_hashes.resize(hashes);
    bundle.resources.resize(resources); bundle.submits.resize(submits);
    bundle.logical_bytes = logical;
    bundle.chunk_indices_by_hash.clear(); bundle.resource_indices_by_hash.clear();
    error = "frame bundle exceeded its unique-byte budget";
    return false;
}
}  // namespace

void request_interactive_capture_bundle(const std::string& path, uint32_t max_mb) {
    InteractiveFrameBundle& b = interactive_frame_bundle();
    std::lock_guard<std::mutex> lk(b.mx);
    b.armed_path = path;
    if (max_mb) b.max_unique_bytes = static_cast<uint64_t>(std::clamp<uint32_t>(max_mb, 64u, 4096u)) << 20;
    if (const char* frames = std::getenv("PROSPER_CAPTURE_FRAMES"))
        b.frames_wanted = std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(frames, nullptr, 0)), 1u, 240u);
    g_interactive_frame_active.store(true, std::memory_order_release);
}
bool interactive_capture_bundle_active() {
    return g_interactive_frame_active.load(std::memory_order_acquire);
}

// Called per submit: when a frame grab is in progress, append this submit's realized state to the bundle.
void interactive_frame_bundle_on_submit(const GpuState& state, uint64_t submit_no) {
    if (!g_interactive_frame_active.load(std::memory_order_acquire)) return;
    InteractiveFrameBundle& b = interactive_frame_bundle();
    std::lock_guard<std::mutex> lk(b.mx);
    if (!b.capturing || b.failed) return;
    GpuCaptureFile capture;
    const GpuCaptureMetadata meta = runtime_capture_metadata(submit_no);
    std::string error;
    if (!capture_gpustate_submit(state, submit_no, meta.width, meta.height, meta, capture, error) ||
        !append_capture_to_frame_bundle(b.bundle, capture, b.max_unique_bytes, error)) {
        b.failed = true;
        std::fprintf(stderr, "[grab] frame-bundle: submit %llu failed (%s); grab aborted\n",
                     static_cast<unsigned long long>(submit_no), error.c_str());
        return;
    }
    ++b.submits;
}

// Called per present (flip): starts the frame on the first present after F9, writes it on the next.
void interactive_frame_bundle_on_present() {
    if (!g_interactive_frame_active.load(std::memory_order_acquire)) return;
    InteractiveFrameBundle& b = interactive_frame_bundle();
    std::string write_path;
    GpuCaptureBundle write_bundle;
    {
        std::lock_guard<std::mutex> lk(b.mx);
        if (b.capturing) {
            ++b.frames_seen;
            if (b.failed || b.frames_seen >= b.frames_wanted) {   // window complete (or aborted)
                if (!b.failed && b.submits > 0) { write_path = b.current_path; write_bundle = std::move(b.bundle); }
                else if (b.failed)
                    std::fprintf(stderr, "[grab] frame-bundle aborted (see error above); not written\n");
                else
                    std::fprintf(stderr, "[grab] frame-bundle: window had no submits; press F9 again\n");
                b.capturing = false; b.current_path.clear(); b.bundle = GpuCaptureBundle{};
                b.submits = 0; b.frames_seen = 0; b.failed = false;
                if (b.armed_path.empty()) g_interactive_frame_active.store(false, std::memory_order_release);
            }
        } else if (!b.armed_path.empty()) {
            b.capturing = true; b.current_path = std::move(b.armed_path); b.armed_path.clear();
            b.bundle = GpuCaptureBundle{}; b.submits = 0; b.frames_seen = 0; b.failed = false;
            std::fprintf(stderr, "[grab] frame-bundle: capturing %u frames -> %s\n",
                         b.frames_wanted, b.current_path.c_str());
        }
    }
    if (!write_path.empty()) {
        std::string error;
        const size_t n = write_bundle.submits.size();
        if (write_gpu_capture_bundle(write_path, write_bundle, error))
            std::fprintf(stderr, "[grab] frame-bundle -> %s (%zu submits)\n", write_path.c_str(), n);
        else
            std::fprintf(stderr, "[grab] frame-bundle write failed: %s\n", error.c_str());
    }
}

void record_gpu_timeline_submit(const GpuState& state, uint64_t submit_no) {
    interactive_frame_bundle_on_submit(state, submit_no);
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
    submit.dma_copy_count = static_cast<uint32_t>(
        std::min<size_t>(state.dma_copies.size(), UINT32_MAX));
    submit.dma_copies.reserve(state.dma_copies.size());
    for (const auto& copy : state.dma_copies)
        submit.dma_copies.push_back({copy.dst, copy.src, copy.bytes, copy.sels,
                                     copy.command_order, copy.packet_addr});
    submit.capture_incomplete = false;
    submit.first_command_order = std::numeric_limits<uint64_t>::max();
    const bool log_mrt = select_timeline_mrt_submit(state, submit_no);
    for (size_t draw_index = 0; draw_index < state.draws.size(); ++draw_index) {
        const auto& draw = state.draws[draw_index];
        submit.first_command_order = std::min(submit.first_command_order, draw.command_order);
        submit.last_command_order = std::max(submit.last_command_order, draw.command_order);
        const GpuState& draw_state = draw.state ? *draw.state : state;
        log_timeline_mrt_draw(draw_state, submit_no, draw_index, draw.command_order, log_mrt);
        const RenderState rs = extract_render_state(draw_state);
        if (!submit.target_spans_truncated) {
            if (draw_index > UINT32_MAX) {
                submit.target_spans_truncated = true;
            } else if (!submit.target_spans.empty() &&
                submit.target_spans.back().draw_count < UINT32_MAX &&
                submit.target_spans.back().width == rs.color0_width &&
                submit.target_spans.back().height == rs.color0_height) {
                ++submit.target_spans.back().draw_count;
            } else if (submit.target_spans.size() < kMaxTargetSpans) {
                submit.target_spans.push_back({static_cast<uint32_t>(draw_index), 1,
                                               rs.color0_width, rs.color0_height});
            } else {
                submit.target_spans_truncated = true;
            }
        }
        if (!rs.z_enable && !rs.z_write_enable && !rs.stencil_enable &&
            !rs.depth_clear_enable && !rs.stencil_clear_enable)
            continue;
        GpuTimelineDepthSurface candidate;
        candidate.depth_read_base = rs.depth_read_base;
        candidate.depth_write_base = rs.depth_write_base;
        candidate.stencil_read_base = rs.stencil_read_base;
        candidate.stencil_write_base = rs.stencil_write_base;
        candidate.htile_data_base = rs.htile_data_base;
        candidate.db_depth_view = rs.db_depth_view;
        candidate.db_render_override = rs.db_render_override;
        candidate.db_render_override2 = rs.db_render_override2;
        candidate.db_depth_size_xy = rs.db_depth_size_xy;
        candidate.db_dfsm_control = rs.db_dfsm_control;
        candidate.db_depth_info = rs.db_depth_info;
        candidate.db_z_info = rs.db_z_info;
        candidate.db_stencil_info = rs.db_stencil_info;
        candidate.db_depth_size = rs.db_depth_size;
        candidate.db_depth_slice = rs.db_depth_slice;
        candidate.db_htile_surface = rs.db_htile_surface;
        candidate.db_rmi_l2_cache_control = rs.db_rmi_l2_cache_control;
        candidate.target_width = rs.color0_width;
        candidate.target_height = rs.color0_height;
        auto found = std::find_if(submit.depth_surfaces.begin(), submit.depth_surfaces.end(),
            [&](const auto& existing) { return same_depth_surface(existing, candidate); });
        if (found == submit.depth_surfaces.end()) {
            static const std::pair<uint32_t, uint32_t> hash_dimensions = [] {
                uint32_t width = 0, height = 0;
                const char* value = std::getenv("PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM");
                if (!value || std::sscanf(value, "%ux%u", &width, &height) != 2)
                    width = height = 0;
                return std::pair{width, height};
            }();
            if (candidate.target_width == hash_dimensions.first &&
                candidate.target_height == hash_dimensions.second) {
                const uint64_t pixels = static_cast<uint64_t>(candidate.target_width) *
                                        candidate.target_height;
                if (hash_guest_backing(candidate.depth_read_base, pixels * 4,
                                       candidate.depth_backing_hash))
                    candidate.backing_hash_mask |= 1u;
                if (hash_guest_backing(candidate.stencil_read_base, pixels,
                                       candidate.stencil_backing_hash))
                    candidate.backing_hash_mask |= 2u;
                // One HTILE dword summarizes an 8x8 depth block; guest allocations are 64 KiB
                // aligned/padded. Do not infer its size from the next plane address: allocations can
                // be non-contiguous, and hashing the gap produced a false "HTILE changed" signal.
                const uint64_t blocks = static_cast<uint64_t>((candidate.target_width + 7u) / 8u) *
                                        ((candidate.target_height + 7u) / 8u);
                const uint64_t htile_bytes = (blocks * 4u + 0xffffu) & ~0xffffull;
                if (hash_guest_backing(candidate.htile_data_base, htile_bytes,
                                       candidate.htile_backing_hash))
                    candidate.backing_hash_mask |= 4u;
                auto consider_writer = [&](uint64_t addr, uint64_t size) {
                    const auto writer = last_guest_write_overlap(addr, size);
                    if (!writer || writer->sequence <= candidate.backing_writer_sequence) return;
                    candidate.backing_writer_kind = static_cast<uint32_t>(writer->kind) + 1u;
                    candidate.backing_writer_sequence = writer->sequence;
                    candidate.backing_writer_addr = writer->addr;
                    candidate.backing_writer_size = writer->size;
                    candidate.backing_writer_order = writer->order;
                    candidate.backing_writer_identity = writer->identity;
                };
                consider_writer(candidate.depth_read_base, pixels * 4);
                consider_writer(candidate.stencil_read_base, pixels);
                consider_writer(candidate.htile_data_base, htile_bytes);
            }
            submit.depth_surfaces.push_back(candidate);
            found = std::prev(submit.depth_surfaces.end());
        }
        ++found->draw_count;
        found->depth_test_count += rs.z_enable;
        found->depth_write_count += rs.z_write_enable;
        found->clear_count += rs.depth_clear_enable || rs.stencil_clear_enable;
        if (rs.z_enable && rs.zfunc < 32) found->compare_mask |= 1u << rs.zfunc;
    }
    for (const auto& dispatch : state.dispatches) {
        submit.first_command_order = std::min(submit.first_command_order, dispatch.command_order);
        submit.last_command_order = std::max(submit.last_command_order, dispatch.command_order);
    }
    for (const auto& dma : state.dma_copies) {
        submit.first_command_order = std::min(submit.first_command_order, dma.command_order);
        submit.last_command_order = std::max(submit.last_command_order, dma.command_order);
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
    if (!writer->append_submit(submit, error)) {
        runtime_recorder().mark_failed(error);
        return;
    }

    RuntimeDetailRequest& request = runtime_detail_request();
    RuntimeProducerHistory& history = runtime_producer_history();
    if (request.valid && request.bundle_start_target_width && !request.bundle_start_reached &&
        runtime_submit_has_target(state, request.bundle_start_target_width,
                                  request.bundle_start_target_height)) {
        request.bundle_start_reached = true;
        std::fprintf(stderr, "[timeline] capture bundle start reached submit=%llu target=%ux%u\n",
                     static_cast<unsigned long long>(submit_no),
                     request.bundle_start_target_width, request.bundle_start_target_height);
    }
    const bool capture_endpoint = request.valid &&
        runtime_capture_endpoint_matches(request, submit);
    const uint64_t bundle_first = request.bundle_depth > request.submit_no
        ? 1 : request.submit_no - request.bundle_depth + 1;
    if (request.valid && !request.bundle_path.empty() &&
        !runtime_capture_bundle().budget_exhausted && !runtime_capture_bundle().failed &&
        submit_no >= bundle_first &&
        (!request.bundle_start_target_width || request.bundle_start_reached) &&
        !capture_endpoint) {
        begin_runtime_capture_bundle();
        GpuCaptureFile predecessor;
        const GpuCaptureMetadata metadata = runtime_capture_metadata(submit_no);
        const bool captured = request.bundle_target_width
            ? capture_gpustate_target_submit(state, submit_no, metadata.width, metadata.height,
                                             request.bundle_target_width, request.bundle_target_height,
                                             metadata, predecessor, error)
            : capture_gpustate_submit(state, submit_no, metadata.width, metadata.height,
                                      metadata, predecessor, error);
        if (!captured ||
            !append_runtime_capture_bundle(predecessor, request.bundle_max_unique_bytes, error)) {
            runtime_capture_bundle().failed = !runtime_capture_bundle().budget_exhausted;
            std::fprintf(stderr, "[timeline] bundle submit %llu capture failed: %s\n",
                         static_cast<unsigned long long>(submit_no), error.c_str());
        } else {
            if (request.select_target_width)
                trim_runtime_capture_bundle(request.bundle_depth - 1);
            RuntimeCaptureBundle& bundle_state = runtime_capture_bundle();
            ++bundle_state.captured_submit_count;
            if (request.bundle_checkpoint_interval &&
                !(bundle_state.captured_submit_count % request.bundle_checkpoint_interval)) {
                if (!write_gpu_capture_bundle(request.bundle_path, bundle_state.bundle, error)) {
                    std::fprintf(stderr,
                                 "[timeline] capture bundle checkpoint at submit %llu failed: %s\n",
                                 static_cast<unsigned long long>(submit_no), error.c_str());
                    error.clear();
                } else {
                    std::fprintf(stderr,
                                 "[timeline] checkpointed capture bundle submits=%zu through=%llu -> %s\n",
                                 bundle_state.bundle.submits.size(),
                                 static_cast<unsigned long long>(submit_no),
                                 request.bundle_path.c_str());
                }
            }
            const std::string identity = request.bundle_path + "#submit=" + std::to_string(submit_no);
            const GpuTimelineDetail detail = capture_detail(submit, identity, predecessor);
            if (!writer->append_detail(detail, error)) {
                runtime_recorder().mark_failed(error);
                history.remember(state, submit_no);
                return;
            }
            const uint64_t offset = submit_no - bundle_first;
            if (request.bundle_depth <= 64 || !offset || !(offset % 100) ||
                (!request.select_target_width && submit_no + 1 == request.submit_no))
                log_capture_detail(detail);
        }
    }
    if (request.valid && !request.select_target_width && !request.predecessor_path.empty() &&
        request.submit_no > 1 &&
        submit_no == request.submit_no - 1 && !request.predecessor_claimed.exchange(true)) {
        GpuCaptureFile predecessor;
        const GpuCaptureMetadata metadata = runtime_capture_metadata(submit_no);
        if (!capture_gpustate_submit(state, submit_no, metadata.width, metadata.height,
                                     metadata, predecessor, error) ||
            !write_gpu_capture(request.predecessor_path, predecessor, error)) {
            request.predecessor_failed.store(true);
            std::fprintf(stderr, "[timeline] predecessor submit %llu capture failed: %s\n",
                         static_cast<unsigned long long>(submit_no), error.c_str());
        } else {
            const GpuTimelineDetail detail = capture_detail(submit, request.predecessor_path, predecessor);
            if (!writer->append_detail(detail, error)) {
                runtime_recorder().mark_failed(error);
                history.remember(state, submit_no);
                return;
            }
            log_capture_detail(detail);
        }
    }
    if (!capture_endpoint || request.claimed.exchange(true)) {
        history.remember(state, submit_no);
        return;
    }
    const GpuCaptureMetadata metadata = runtime_capture_metadata(submit_no);

    GpuCaptureFile capture;
    if (!request.bundle_path.empty()) begin_runtime_capture_bundle();
    const auto capture_started = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[timeline] submit %llu detailed capture starting: semantic-draws=%zu "
                         "semantic-dispatches=%zu metadata-only=%s\n",
                 static_cast<unsigned long long>(submit_no), state.draws.size(),
                 state.dispatches.size(),
                 std::getenv("PROSPER_GPU_CAPTURE_METADATA_ONLY") ? "yes" : "no");
    if (!capture_gpustate_submit(state, submit_no, metadata.width, metadata.height,
                                 metadata, capture, error)) {
        std::fprintf(stderr, "[timeline] submit %llu detailed capture failed: %s\n",
                     static_cast<unsigned long long>(submit_no), error.c_str());
        history.remember(state, submit_no);
        if (request.exit_after_capture) {
            std::fprintf(stderr, "[timeline] selected capture failed; exiting nonzero as requested\n");
            close_gpu_timeline();
            std::fflush(nullptr);
            std::exit(2);
        }
        return;
    }
    const auto capture_realized = std::chrono::steady_clock::now();
    const uint64_t realization_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            capture_realized - capture_started).count());
    std::fprintf(stderr, "[timeline] submit %llu detailed capture realized in %llums: "
                         "draws=%zu dispatches=%zu operations=%zu blobs=%zu\n",
                 static_cast<unsigned long long>(submit_no),
                 static_cast<unsigned long long>(realization_ms), capture.draws.size(),
                 capture.computes.size(), capture.operations.size(), capture.blobs.size());
    if (gpu_capture_ds_seed_snapshot_available() &&
        !capture_referenced_gpu_ds_seeds(capture, error)) {
        std::fprintf(stderr, "[timeline] submit %llu DS checkpoint capture failed: %s\n",
                     static_cast<unsigned long long>(submit_no), error.c_str());
        history.remember(state, submit_no);
        if (request.exit_after_capture) {
            std::fprintf(stderr, "[timeline] selected capture failed; exiting nonzero as requested\n");
            close_gpu_timeline();
            std::fflush(nullptr);
            std::exit(2);
        }
        return;
    }
    if (!capture.ds_seeds.empty())
        std::fprintf(stderr, "[timeline] submit %llu retained %zu referenced DS checkpoint(s)\n",
                     static_cast<unsigned long long>(submit_no), capture.ds_seeds.size());
    if (!write_gpu_capture(request.path, capture, error)) {
        std::fprintf(stderr, "[timeline] submit %llu detailed capture write failed: %s\n",
                     static_cast<unsigned long long>(submit_no), error.c_str());
        history.remember(state, submit_no);
        if (request.exit_after_capture) {
            std::fprintf(stderr, "[timeline] selected capture failed; exiting nonzero as requested\n");
            close_gpu_timeline();
            std::fflush(nullptr);
            std::exit(2);
        }
        return;
    }
    const uint64_t total_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - capture_started).count());
    std::fprintf(stderr, "[timeline] submit %llu detailed capture written in %llums -> %s\n",
                 static_cast<unsigned long long>(submit_no),
                 static_cast<unsigned long long>(total_ms), request.path.c_str());
    GpuReplayFrame replay;
    GpuDependencyGraph graph;
    if (materialize_gpu_replay(capture, replay, error) &&
        build_gpu_dependency_graph(replay, graph, error)) {
        for (const auto& leaf : graph.external_leaves) {
            if (leaf.access.resource_class != ResourceClass::Texture &&
                leaf.access.resource_class != ResourceClass::StorageImage) continue;
            GpuTimelineProducer producer;
            producer.consumer_submit_no = submit_no;
            producer.consumer_operation = leaf.consumer_operations.front();
            producer.future_writer_operation = leaf.first_future_writer;
            producer.resource_addr = leaf.access.addr;
            producer.resource_size = leaf.access.size;
            producer.resource_width = leaf.access.width;
            producer.resource_height = leaf.access.height;
            const RuntimeTargetLifetime lifetime = history.lifetime_overlap(
                leaf.access.addr, leaf.access.size);
            producer.history_window_first_submit_no = history.window_first_submit_no();
            producer.lifetime_truncated = history.lifetime_truncated;
            producer.history_window_truncated = history.dropped_submits != 0;
            producer.history_write_count = lifetime.write_count;
            producer.history_submit_count = lifetime.submit_count;
            if (lifetime.first) {
                producer.first_writer_kind = GpuTimelineWriterKind::Graphics;
                producer.history_first_submit_no = lifetime.first->submit_no;
                producer.history_first_draw_index = lifetime.first->draw_index;
                producer.history_first_command_order = lifetime.first->command_order;
                producer.first_color_has_clear = lifetime.first->color_has_clear;
                producer.first_color_clear_word0 = lifetime.first->color_clear_word0;
                producer.first_color_clear_word1 = lifetime.first->color_clear_word1;
                producer.first_color_control = lifetime.first->color_control;
                producer.first_color_control_mode =
                    (lifetime.first->color_control >> 4) & 0x7u;
                producer.first_target_mask = lifetime.first->target_mask;
                producer.first_color_format = lifetime.first->color_format;
            }
            const RuntimeTargetWrite* match = lifetime.last
                ? lifetime.last : history.latest_overlap(leaf.access.addr, leaf.access.size);
            if (match) {
                producer.resolved = true;
                producer.producer_submit_no = match->submit_no;
                producer.producer_draw_index = match->draw_index;
                producer.producer_command_order = match->command_order;
                producer.producer_target_addr = match->addr;
                producer.producer_width = match->width;
                producer.producer_height = match->height;
            }
            if (!writer->append_producer(producer, error)) {
                runtime_recorder().mark_failed(error);
                history.remember(state, submit_no);
                return;
            }
            std::fprintf(stderr, "[timeline] temporal resource 0x%llx/%ux%u op=%u -> %s",
                         static_cast<unsigned long long>(producer.resource_addr),
                         producer.resource_width, producer.resource_height,
                         producer.consumer_operation, producer.resolved ? "producer" : "unresolved");
            if (producer.resolved)
                std::fprintf(stderr, " submit=%llu draw=%llu order=%llu target=0x%llx/%ux%u",
                             static_cast<unsigned long long>(producer.producer_submit_no),
                             static_cast<unsigned long long>(producer.producer_draw_index),
                             static_cast<unsigned long long>(producer.producer_command_order),
                             static_cast<unsigned long long>(producer.producer_target_addr),
                             producer.producer_width, producer.producer_height);
            if (lifetime.first)
                std::fprintf(stderr, " lifetime=%llu..%llu submits=%llu writes=%llu "
                                     "lifetime-truncated=%s window-truncated=%s "
                                     "first-clear=%s mode=%u mask=0x%x fmt=%u",
                             static_cast<unsigned long long>(producer.history_first_submit_no),
                             static_cast<unsigned long long>(producer.producer_submit_no),
                             static_cast<unsigned long long>(producer.history_submit_count),
                             static_cast<unsigned long long>(producer.history_write_count),
                             producer.lifetime_truncated ? "yes" : "no",
                             producer.history_window_truncated ? "yes" : "no",
                             producer.first_color_has_clear ? "programmed" : "absent",
                             producer.first_color_control_mode, producer.first_target_mask,
                             producer.first_color_format);
            std::fprintf(stderr, "\n");
        }
    } else {
        std::fprintf(stderr, "[timeline] dependency graph failed for submit %llu: %s\n",
                     static_cast<unsigned long long>(submit_no), error.c_str());
        error.clear();
    }
    const bool predecessor_requested = !request.select_target_width &&
        !request.predecessor_path.empty() && request.submit_no > 1;
    bool requested_artifacts_written = !predecessor_requested ||
        (request.predecessor_claimed.load() && !request.predecessor_failed.load());
    if (!request.bundle_path.empty()) {
        RuntimeCaptureBundle& state = runtime_capture_bundle();
        GpuCaptureBundle& bundle = state.bundle;
        if (state.failed) {
            error = "capture bundle has a failed or missing predecessor submit";
        } else if (state.budget_exhausted) {
            error = "capture bundle unique-byte budget was exhausted before the selected submit";
        }
        if (state.failed || state.budget_exhausted || !append_runtime_capture_bundle(
                capture, request.bundle_max_unique_bytes, error) ||
            !compact_gpu_capture_bundle(bundle, error) ||
            !write_gpu_capture_bundle(request.bundle_path, bundle, error)) {
            requested_artifacts_written = false;
            std::fprintf(stderr, "[timeline] capture bundle write failed: %s\n", error.c_str());
        } else {
            const uint64_t unique_bytes = gpu_capture_bundle_unique_bytes(bundle);
            const GpuCaptureBundleStats stats = gpu_capture_bundle_stats(bundle);
            const uint64_t capture_ms = state.started ? static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - state.started_at).count()) : 0;
            std::fprintf(stderr, "[timeline] capture bundle submits=%zu logical=%llu unique=%llu "
                                 "ratio=%.3f resources=%zu refs=%llu exact-reuse=%llu "
                                 "resource-logical=%llu resource-unique=%llu manifest-unique=%llu "
                                 "guest-copied=%llu capture-ms=%llu -> %s\n",
                         bundle.submits.size(), static_cast<unsigned long long>(bundle.logical_bytes),
                         static_cast<unsigned long long>(unique_bytes),
                         bundle.logical_bytes ? static_cast<double>(unique_bytes) / bundle.logical_bytes : 0.0,
                         bundle.resources.size(),
                         static_cast<unsigned long long>(stats.resource_reference_count),
                         static_cast<unsigned long long>(stats.exact_reuse_count),
                         static_cast<unsigned long long>(stats.resource_logical_bytes),
                         static_cast<unsigned long long>(stats.resource_unique_bytes),
                         static_cast<unsigned long long>(stats.manifest_unique_bytes),
                         static_cast<unsigned long long>(state.captured_resource_bytes),
                         static_cast<unsigned long long>(capture_ms),
                         request.bundle_path.c_str());
        }
    }
    const GpuTimelineDetail detail = capture_detail(submit, request.path, capture);
    if (!writer->append_detail(detail, error)) {
        runtime_recorder().mark_failed(error);
        history.remember(state, submit_no);
        return;
    }
    log_capture_detail(detail);
    history.remember(state, submit_no);
    if (request.exit_after_capture) {
        std::fprintf(stderr, requested_artifacts_written
            ? "[timeline] selected capture complete; exiting as requested\n"
            : "[timeline] one or more requested capture artifacts failed; exiting nonzero\n");
        close_gpu_timeline();
        std::fflush(nullptr);
        std::exit(requested_artifacts_written ? 0 : 2);
    }
}

void record_gpu_timeline_present(uint64_t present_count, int buffer_index, int64_t flip_arg,
                                 uint32_t width, uint32_t height) {
    interactive_frame_bundle_on_present();
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
