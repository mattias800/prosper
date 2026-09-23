// Opt-in, bounded cross-thread provenance for a compute-output -> graphics-present probe.
#pragma once

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

namespace prosper::frontend {

inline uint64_t compute_probe_now_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct ComputePresentProbeSpec {
    bool requested = false;
    bool armed = false;
    uint64_t code = 0;
    uint64_t after_ms = 0;
    uint32_t stride = 0;
};

// 0xCODE:ms:DEADLINE:STRIDE; all numeric fields must be exact and bounded.
inline ComputePresentProbeSpec parse_compute_present_probe(const char* value) {
    ComputePresentProbeSpec out;
    if (!value) return out;
    out.requested = true;
    const std::string_view s(value);
    if (!s.starts_with("0x")) return out;
    const auto code_end = s.find(':');
    if (code_end == std::string_view::npos || !s.substr(code_end).starts_with(":ms:"))
        return out;
    const auto deadline_end = s.find(':', code_end + 4);
    if (deadline_end == std::string_view::npos ||
        s.find(':', deadline_end + 1) != std::string_view::npos) return out;
    const auto parse = [](std::string_view part, int base, uint64_t& result) {
        if (part.empty() || part.front() == '+' || part.front() == '-') return false;
        const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), result, base);
        return error == std::errc{} && end == part.data() + part.size();
    };
    uint64_t stride = 0;
    if (!parse(s.substr(2, code_end - 2), 16, out.code) || !out.code ||
        !parse(s.substr(code_end + 4, deadline_end - code_end - 4), 10, out.after_ms) ||
        !parse(s.substr(deadline_end + 1), 10, stride) ||
        out.after_ms > 120'000 || !stride || stride > 64) return out;
    out.stride = static_cast<uint32_t>(stride);
    out.armed = true;
    return out;
}

inline bool compute_probe_modes_compatible(bool menu_trace_requested,
                                           bool broad_readback_requested) {
    return !menu_trace_requested && !broad_readback_requested;
}

inline bool compute_probe_targets_valid(const std::vector<uint64_t>& targets) {
    if (targets.size() != 4) return false;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (!targets[i]) return false;
        for (size_t j = 0; j < i; ++j)
            if (targets[i] == targets[j]) return false;
    }
    return true;
}

inline bool probe_ranges_overlap(uint64_t a, uint64_t a_bytes,
                                 uint64_t b, uint64_t b_bytes) {
    if (!a || !b || !a_bytes || !b_bytes ||
        a > UINT64_MAX - a_bytes || b > UINT64_MAX - b_bytes) return false;
    return a < b + b_bytes && b < a + a_bytes;
}

inline bool compute_probe_sample_due(uint32_t matching_callbacks, uint32_t stride,
                                     uint32_t completed_scene_probes) {
    return matching_callbacks && stride && completed_scene_probes < 4 &&
           (matching_callbacks - 1) % stride == 0;
}

inline bool compute_probe_charge(uint64_t& charged, uint64_t bytes, uint64_t cap) {
    if (!bytes || charged > cap || bytes > cap - charged) return false;
    charged += bytes;
    return true;
}

inline std::optional<uint64_t> compute_probe_bmp_bytes(uint32_t width, uint32_t height) {
    if (!width || !height || width > UINT32_MAX / 3) return std::nullopt;
    const uint64_t row = (uint64_t)width * 3;
    const uint64_t padded = (row + 3) & ~3ull;
    if (height > (UINT32_MAX - 54) / padded) return std::nullopt;
    return 54 + padded * height;
}

struct ComputePresentEvent {
    uint64_t id = 0;
    uint64_t code = 0;
    uint64_t submit = 0;
    uint64_t dispatch = 0;
    uint64_t command_order = 0; // queue-local; never compare with a graphics order
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint64_t nonzero_channels = 0;
    uint64_t guest_nonzero_bytes = 0;
    uint64_t published_ms = 0;
    uint64_t admitted_at_publish = 0;
};

inline bool compute_probe_binding_matches(const ComputePresentEvent& event,
                                          uint64_t actual_ps, uint64_t selected_ps,
                                          uint64_t actual_resource_address) {
    return event.id && selected_ps && actual_ps == selected_ps &&
           actual_resource_address == event.address;
}

class ComputePresentProbe {
public:
    void configure(ComputePresentProbeSpec spec) {
        spec_ = spec;
        enabled_.store(spec.armed, std::memory_order_release);
    }
    bool enabled() const { return enabled_.load(std::memory_order_acquire); }
    bool phase_armed() const { return phase_.load(std::memory_order_acquire); }
    bool accepts(uint64_t code) const {
        return enabled() && phase_armed() && code == spec_.code;
    }
    bool arm_phase(uint64_t elapsed_ms) {
        if (!enabled() || elapsed_ms < spec_.after_ms) return false;
        return !phase_.exchange(true, std::memory_order_acq_rel);
    }
    bool expire_no_producer(uint64_t elapsed_ms) {
        if (!phase_armed() || elapsed_ms < spec_.after_ms + 10'000) return false;
        std::lock_guard lock(mx_);
        if (state_ != State::Idle) return false;
        state_ = State::Expired;
        return true;
    }
    uint32_t stride() const { return spec_.stride; }
    uint64_t admitted_callback() { return admitted_.fetch_add(1, std::memory_order_relaxed) + 1; }
    bool publishable() const {
        if (!enabled() || !phase_armed()) return false;
        std::lock_guard lock(mx_);
        return state_ == State::Idle;
    }
    bool should_scan(uint64_t code) const {
        return accepts(code) && publishable();
    }

    // The first completed nonzero producer owns the slot. Later producers are ignored until the
    // bounded attempt resolves; an intervening notified write to its address invalidates it.
    std::optional<ComputePresentEvent> publish(ComputePresentEvent event) {
        if (!accepts(event.code) || !event.address || !event.bytes || !event.nonzero_channels ||
            !event.guest_nonzero_bytes ||
            event.address > UINT64_MAX - event.bytes) return std::nullopt;
        std::lock_guard lock(mx_);
        if (state_ != State::Idle) { ++ignored_; return std::nullopt; }
        event.id = 1; // one bounded event attempt per process
        event.admitted_at_publish = admitted_.load(std::memory_order_relaxed);
        event_ = event;
        state_ = State::Pending;
        return event;
    }
    std::optional<ComputePresentEvent> invalidate_write(uint64_t address, uint64_t bytes) {
        if (!enabled()) return std::nullopt;
        std::lock_guard lock(mx_);
        if (state_ != State::Pending || !address || !bytes)
            return std::nullopt;
        // An invalid notified range cannot establish disjointness. Refuse this provenance
        // attempt instead of wrapping the end address and silently accepting it.
        if (address <= UINT64_MAX - bytes &&
            !probe_ranges_overlap(event_.address, event_.bytes, address, bytes))
            return std::nullopt;
        state_ = State::Invalid;
        return event_;
    }
    std::optional<ComputePresentEvent> expire(uint64_t now_ms, uint64_t admitted) {
        if (!enabled()) return std::nullopt;
        std::lock_guard lock(mx_);
        if (state_ != State::Pending) return std::nullopt;
        if (now_ms >= event_.published_ms && admitted >= event_.admitted_at_publish &&
            now_ms - event_.published_ms <= 2000 &&
            admitted - event_.admitted_at_publish <= 256) return std::nullopt;
        state_ = State::Expired;
        return event_;
    }
    std::optional<ComputePresentEvent> pending() const {
        if (!enabled()) return std::nullopt;
        std::lock_guard lock(mx_);
        return state_ == State::Pending ? std::optional(event_) : std::nullopt;
    }
    std::optional<uint32_t> admit_scene_probe(uint64_t id) {
        std::lock_guard lock(mx_);
        if (state_ != State::Pending || event_.id != id) return std::nullopt;
        ++matching_callbacks_;
        if (!compute_probe_sample_due(matching_callbacks_, spec_.stride, scene_probes_))
            return std::nullopt;
        return ++scene_probes_;
    }
    bool charge_raw(uint64_t id, uint64_t bytes) {
        std::lock_guard lock(mx_);
        return state_ == State::Pending && event_.id == id &&
            compute_probe_charge(raw_charged_, bytes, 160ull << 20);
    }
    bool charge_bmp(uint64_t id, uint64_t bytes) {
        std::lock_guard lock(mx_);
        return state_ == State::Pending && event_.id == id &&
            compute_probe_charge(bmp_charged_, bytes, 256ull << 20);
    }
    bool finish(uint64_t id) {
        std::lock_guard lock(mx_);
        if (state_ != State::Pending || event_.id != id) return false;
        state_ = State::Done;
        return true;
    }
    uint64_t ignored() const {
        std::lock_guard lock(mx_);
        return ignored_;
    }

private:
    enum class State { Idle, Pending, Invalid, Expired, Done };
    ComputePresentProbeSpec spec_{}; // configured once before compute registration
    std::atomic<bool> enabled_{false};
    std::atomic<bool> phase_{false};
    std::atomic<uint64_t> admitted_{0};
    mutable std::mutex mx_;
    State state_ = State::Idle;
    ComputePresentEvent event_{};
    uint64_t ignored_ = 0;
    uint32_t matching_callbacks_ = 0;
    uint32_t scene_probes_ = 0;
    uint64_t raw_charged_ = 0;
    uint64_t bmp_charged_ = 0;
};

// A single shared instance across the live compute and renderer translation units.
inline ComputePresentProbe& compute_present_probe() {
    static ComputePresentProbe probe;
    return probe;
}

} // namespace prosper::frontend
