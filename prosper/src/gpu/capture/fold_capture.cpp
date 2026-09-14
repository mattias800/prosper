#include "gpu/capture/fold_capture.hpp"
#include "build_revision.hpp"
#include "diagnostics/env_numeric.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace prosper::gpu {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void validate_inputs(const FoldInputs& in) {
    require(!in.code.empty() && in.code.size() <= kFoldMaxCodeDwords, "invalid fold code span");
    require(in.user.size() <= 128 && in.system.size() <= 128 && in.user_base <= 128,
            "invalid fold register span");
    require(in.user_present || in.user.empty(), "absent user array has words");
    require(in.system_present || in.system.empty(), "absent system array has words");
    require(in.absent_system_count <= 128 && (!in.system_present || in.absent_system_count == 0),
            "invalid absent-system count");
    require(in.revision.size() <= 128, "oversized source revision");
    require(in.dispatch.target_pcs.size() <= kFoldMaxCodeDwords &&
            in.dispatch.setup_pcs.size() <= kFoldMaxCodeDwords &&
            in.dispatch.required_dwords <= in.code.size(), "invalid dispatch span");
}

class Recorder final : public FoldReader {
    FoldCapture& capture_;
    FoldReader* backing_;
    uint64_t payload_ = 0;
    void retain(FoldEvent event, const void* bytes = nullptr) noexcept {
        if (!capture_.complete) return;
        const uint32_t count = event.kind == FoldEventKind::Probe ? 0 : event.bytes;
        if (capture_.events.size() >= kFoldMaxEvents || payload_ + count > kFoldMaxPayloadBytes) {
            capture_.complete = false;
            return;
        }
        try {
            if (count) {
                const auto* data = static_cast<const uint8_t*>(bytes);
                event.data.assign(data, data + count);
            }
            capture_.events.push_back(std::move(event));
            payload_ += count;
        } catch (...) { capture_.complete = false; }
    }
public:
    Recorder(FoldCapture& capture, FoldReader* backing) : capture_(capture), backing_(backing) {
        branch_exclusive_disabled = capture.input.branch_exclusive_disabled;
        logical_code_address = capture.input.code_address;
    }
    bool probe(FoldProbe kind, uint32_t pc, uint64_t addr, uint32_t bytes) override {
        const bool ok = backing_ ? backing_->probe(kind, pc, addr, bytes) : guest_readable(addr, bytes);
        retain({FoldEventKind::Probe, kind, pc, bytes, addr, ok, {}});
        return ok;
    }
    uint32_t word(uint32_t pc, uint64_t addr) override {
        const uint32_t value = backing_ ? backing_->word(pc, addr)
            : *reinterpret_cast<const uint32_t*>(uintptr_t(addr));
        // Event words have an explicitly little-endian payload, independent of host byte order.
        const uint8_t bytes[4] = {uint8_t(value), uint8_t(value >> 8),
                                 uint8_t(value >> 16), uint8_t(value >> 24)};
        retain({FoldEventKind::Word, FoldProbe::Raw, pc, 4, addr, false, {}}, bytes);
        return value;
    }
    void prefix(uint32_t pc, uint64_t addr, void* destination, uint32_t bytes) override {
        if (backing_) backing_->prefix(pc, addr, destination, bytes);
        else std::memcpy(destination, reinterpret_cast<const void*>(uintptr_t(addr)), bytes);
        retain({FoldEventKind::Prefix, FoldProbe::Raw, pc, bytes, addr, false, {}}, destination);
    }
};

class ReplayReader final : public FoldReader {
    const std::vector<FoldEvent>& events_;
    size_t cursor_ = 0;
    const FoldEvent& next(FoldEventKind kind, FoldProbe probe, uint32_t pc,
                          uint64_t addr, uint32_t bytes) {
        if (cursor_ == events_.size()) throw std::runtime_error("missing fold event at " + std::to_string(cursor_));
        const auto& e = events_[cursor_];
        if (e.kind != kind || e.probe != probe || e.pc != pc || e.address != addr || e.bytes != bytes)
            throw std::runtime_error("fold request mismatch at event " + std::to_string(cursor_));
        ++cursor_;
        return e;
    }
public:
    explicit ReplayReader(const FoldCapture& c) : events_(c.events) {
        branch_exclusive_disabled = c.input.branch_exclusive_disabled;
        logical_code_address = c.input.code_address;
    }
    bool probe(FoldProbe kind, uint32_t pc, uint64_t addr, uint32_t bytes) override {
        return next(FoldEventKind::Probe, kind, pc, addr, bytes).readable;
    }
    uint32_t word(uint32_t pc, uint64_t addr) override {
        const auto& d = next(FoldEventKind::Word, FoldProbe::Raw, pc, addr, 4).data;
        return uint32_t(d[0]) | uint32_t(d[1]) << 8 | uint32_t(d[2]) << 16 | uint32_t(d[3]) << 24;
    }
    void prefix(uint32_t pc, uint64_t addr, void* destination, uint32_t bytes) override {
        const auto& d = next(FoldEventKind::Prefix, FoldProbe::Raw, pc, addr, bytes).data;
        std::memcpy(destination, d.data(), bytes);
    }
    void finish() const { require(cursor_ == events_.size(), "unconsumed fold events"); }
};

std::vector<DynFetch> evaluate(const FoldInputs& in, FoldReader& reader, std::vector<SrtUse>* uses) {
    // Preserve non-null empty-array identity without reading outside an empty vector.
    const uint32_t empty = 0;
    const uint32_t* user = !in.user_present ? nullptr : in.user.empty() ? &empty : in.user.data();
    const uint32_t* system = !in.system_present ? nullptr : in.system.empty() ? &empty : in.system.data();
    return resolve_dynamic_fetch(in.code.data(), in.code.size(), user,
        static_cast<uint32_t>(in.user.size()), in.user_base, uses, in.dispatch_target,
        in.dispatch_present ? &in.dispatch : nullptr, system,
        in.system_present ? static_cast<uint32_t>(in.system.size()) : in.absent_system_count, &reader);
}
void finish_capture(FoldCapture& c, const FoldReader& reader) {
    c.evaluated_instructions = reader.evaluated_instructions;
    c.decoded_dwords = reader.decoded_dwords;
    c.shader_constant_specialized = reader.shader_constant_specialized;
    if (!c.complete) c.error = "recording budget exceeded or allocation failed";
    require(reader.evaluations == 1, "capture did not execute exactly one fold");
}
} // namespace

void validate_fold_capture(const FoldCapture& c) {
    validate_inputs(c.input);
    require(c.complete && c.error.empty(), "incomplete fold capture");
    require(c.events.size() <= kFoldMaxEvents && c.fetches.size() <= kFoldMaxOutputs &&
            c.uses.size() <= kFoldMaxOutputs, "fold capture count limit");
    require(c.input.srt_requested || c.uses.empty(), "unexpected SRT outputs");
    require(c.decoded_dwords <= c.input.code.size() && c.evaluated_instructions <= c.input.code.size(),
            "invalid evaluated code span");
    uint64_t total = 0;
    for (const auto& e : c.events) {
        require(e.pc < c.input.code.size(), "event PC outside code");
        require(e.bytes != 0 && e.address <= UINT64_MAX - (e.bytes - 1), "invalid event address span");
        if (e.kind == FoldEventKind::Probe) {
            require(e.probe <= FoldProbe::NullableOutput && e.data.empty(), "invalid probe event");
            const uint32_t cap = e.probe == FoldProbe::OptionalTable ? 272u :
                                 e.probe == FoldProbe::NullableOutput ? 40u : 64u;
            require(e.bytes <= cap, "oversized fold probe");
        } else {
            require(e.kind == FoldEventKind::Word || e.kind == FoldEventKind::Prefix, "unknown fold event");
            require(e.probe == FoldProbe::Raw && !e.readable && e.bytes <= 64 &&
                    e.bytes % 4 == 0 && e.data.size() == e.bytes, "invalid fold payload");
            require(e.kind != FoldEventKind::Word || e.bytes == 4, "invalid word event");
        }
        total += e.data.size();
        require(total <= kFoldMaxPayloadBytes, "fold payload budget exceeded");
    }
}

FoldCapture capture_fold(FoldInputs input, FoldReader* backing) {
    validate_inputs(input);
    FoldCapture capture;
    capture.input = std::move(input);
    Recorder reader(capture, backing);
    capture.fetches = evaluate(capture.input, reader, capture.input.srt_requested ? &capture.uses : nullptr);
    finish_capture(capture, reader);
    return capture;
}

FoldReplayResult replay_fold(const FoldCapture& c, bool compare_outputs) {
    validate_fold_capture(c);
    ReplayReader reader(c);
    std::vector<SrtUse> uses;
    auto fetches = evaluate(c.input, reader, c.input.srt_requested ? &uses : nullptr);
    reader.finish();
    require(reader.evaluations == 1, "replay did not execute exactly one fold");
    if (compare_outputs) {
        require(fetches == c.fetches, "DynFetch output mismatch");
        require(uses == c.uses, "SrtUse output mismatch");
        require(reader.evaluated_instructions == c.evaluated_instructions &&
                reader.decoded_dwords == c.decoded_dwords &&
                reader.shader_constant_specialized == c.shader_constant_specialized,
                "fold analysis mismatch");
    }
    return {std::move(fetches), std::move(uses), reader.evaluated_instructions};
}

bool try_capture_live_fold(const uint32_t* code, size_t dwords,
        const uint32_t* user, uint32_t nuser, uint32_t user_base, std::vector<SrtUse>* uses,
        uint32_t target, const PcrelDispatchInfo* dispatch,
        const uint32_t* system, uint32_t nsystem, std::vector<DynFetch>& result) {
    struct Settings {
        const char* directory = nullptr;
        uint64_t limit = 16, skip = 0;
        bool valid = true;
        Settings() {
            if (const char* dir = std::getenv("PROSPER_FOLD_CAPTURE_DIR")) directory = dir;
            for (auto [name, destination, maximum] : {
                    std::tuple{"PROSPER_FOLD_CAPTURE_LIMIT", &limit, uint64_t(256)},
                    std::tuple{"PROSPER_FOLD_CAPTURE_SKIP", &skip, UINT64_MAX}}) {
                if (const char* value = std::getenv(name)) {
                    uint64_t parsed = 0;
                    if (!prosper::diag::parse_u64_strict(value, &parsed) || parsed > maximum) valid = false;
                    else *destination = parsed;
                }
            }
            if (!directory || !*directory) valid = false;
            if (!valid) std::fprintf(stderr, "[fold-capture] invalid capture settings; disabled\n");
        }
    };
    static const Settings settings;
    if (!settings.valid) return false;
    static std::atomic<uint64_t> calls{0};
    const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed);
    if (call < settings.skip || call - settings.skip >= settings.limit) return false;
    FoldCapture capture;
    try {
        require(!uses || uses->empty(), "capture requires an empty initial SRT vector");
        require(!g_dyntrace_force && !std::getenv("PROSPER_DYNTRACE") &&
                !std::getenv("PROSPER_DESCR_COHERENCE"), "incompatible tracing/coherence mode");
        require(code && dwords && dwords <= kFoldMaxCodeDwords && nuser <= 128 && nsystem <= 128 &&
                user_base <= 128 && (user || !nuser), "capture input bound/refusal");
        require(guest_readable(uint64_t(uintptr_t(code)), static_cast<uint32_t>(dwords * 4)),
                "declared code span not readable for capture");
        require(!nuser || guest_readable(uint64_t(uintptr_t(user)), nuser * 4), "user input unreadable");
        require(!system || !nsystem || guest_readable(uint64_t(uintptr_t(system)), nsystem * 4),
                "system input unreadable");
        auto& in = capture.input;
        in.invocation = call; in.code_address = uint64_t(uintptr_t(code));
        in.revision = prosper::embedded_build_revision();
        in.code.assign(code, code + dwords);
        if (nuser) in.user.assign(user, user + nuser);
        if (system && nsystem) in.system.assign(system, system + nsystem);
        in.user_base = user_base; in.user_present = user != nullptr; in.system_present = system != nullptr;
        in.srt_requested = uses != nullptr; in.dispatch_target = target; in.dispatch_present = dispatch != nullptr;
        if (dispatch) {
            require(dispatch->target_pcs.size() <= kFoldMaxCodeDwords &&
                    dispatch->setup_pcs.size() <= kFoldMaxCodeDwords &&
                    dispatch->required_dwords <= dwords, "capture dispatch bound/refusal");
            in.dispatch = *dispatch;
        }
        if (!system) in.absent_system_count = nsystem;
        in.branch_exclusive_disabled = std::getenv("PROSPER_NO_BRANCH_EXCLUSIVE") != nullptr;
        validate_inputs(in);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[fold-capture] refused call=%llu: %s\n", (unsigned long long)call, error.what());
        return false; // No fold or resource reader has executed yet.
    }
    const size_t before = uses ? uses->size() : 0;
    Recorder reader(capture, nullptr);
    result = evaluate(capture.input, reader, uses); // Exactly once, with direct publication to caller.
    try {
        capture.fetches = result;
        if (uses) capture.uses.assign(uses->begin() + before, uses->end());
        finish_capture(capture, reader);
        // Reserve an independent session directory across simultaneous process launches.
        static const std::filesystem::path session = [&] {
            const auto root = std::filesystem::path(settings.directory);
            std::filesystem::create_directories(root);
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            for (unsigned attempt = 0; attempt < 64; ++attempt) {
                auto dir = root / ("session-" + std::to_string(stamp) + "-" + std::to_string(attempt));
                if (std::filesystem::create_directory(dir)) return dir;
            }
            throw std::runtime_error("cannot reserve fold capture session");
        }();
        const auto path = session / ("fold-" + std::to_string(call) + ".prfold");
        write_fold_capture(path, capture);
        std::fprintf(stderr, "[fold-capture] complete call=%llu events=%zu instructions=%llu file=%s\n",
            (unsigned long long)call, capture.events.size(),
            (unsigned long long)capture.evaluated_instructions, path.filename().string().c_str());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[fold-capture] unusable call=%llu: %s\n", (unsigned long long)call, error.what());
    }
    return true; // Recording failure must not rerun the fold or discard its real outputs.
}
} // namespace prosper::gpu
