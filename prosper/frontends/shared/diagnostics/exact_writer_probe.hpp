// Opt-in, bounded policy for bracketing one graphics writer with exact target readbacks.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <string_view>

namespace prosper::frontend {

struct ExactWriterProbeSpec {
    bool requested = false;
    bool armed = false;
    uint64_t ps = 0;
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t after_ms = 0;
};

// 0xPS:0xINPUT:0xOUTPUT:ms:AFTER. Every field is exact; a partial parse must refuse.
inline ExactWriterProbeSpec parse_exact_writer_probe(const char* value) {
    ExactWriterProbeSpec spec;
    if (!value) return spec;
    spec.requested = true;
    std::string_view rest(value);
    auto hex = [&](uint64_t& output) {
        if (!rest.starts_with("0x")) return false;
        const size_t end = rest.find(':');
        if (end == std::string_view::npos || end <= 2) return false;
        const auto part = rest.substr(2, end - 2);
        const auto [last, error] = std::from_chars(part.data(), part.data() + part.size(),
                                                  output, 16);
        rest.remove_prefix(end + 1);
        return error == std::errc{} && last == part.data() + part.size() && output;
    };
    if (!hex(spec.ps) || !hex(spec.input) || !hex(spec.output) ||
        spec.input == spec.output || !rest.starts_with("ms:")) return spec;
    rest.remove_prefix(3);
    if (rest.empty() || rest.front() == '+' || rest.front() == '-') return spec;
    const auto [last, error] = std::from_chars(rest.data(), rest.data() + rest.size(),
                                              spec.after_ms, 10);
    spec.armed = error == std::errc{} && last == rest.data() + rest.size() &&
                 spec.after_ms <= 120'000;
    return spec;
}

enum class ExactWriterMatch : uint8_t {
    WrongPs, WrongTarget, MultiDraw, NoColorWrite, Exact
};

inline ExactWriterMatch exact_writer_match(uint64_t actual_ps, uint64_t actual_target,
                                           size_t original_draws, size_t adjusted_draws,
                                           bool color0_write, const ExactWriterProbeSpec& spec) {
    if (actual_ps != spec.ps) return ExactWriterMatch::WrongPs;
    if (actual_target != spec.output) return ExactWriterMatch::WrongTarget;
    if (original_draws != 1 || adjusted_draws != 1) return ExactWriterMatch::MultiDraw;
    if (!color0_write) return ExactWriterMatch::NoColorWrite;
    return ExactWriterMatch::Exact;
}

inline bool exact_writer_scene_input_visible(size_t raw_nonzero, size_t rgb_nonblack) {
    return raw_nonzero != 0 && rgb_nonblack != 0;
}

enum class ExactWriterOutputChange : uint8_t { Unknown, Identical, Changed };

inline ExactWriterOutputChange exact_writer_output_change(std::span<const uint8_t> before,
                                                           std::span<const uint8_t> after) {
    if (before.empty() || after.empty() || before.size() != after.size())
        return ExactWriterOutputChange::Unknown;
    for (size_t i = 0; i < before.size(); ++i)
        if (before[i] != after[i]) return ExactWriterOutputChange::Changed;
    return ExactWriterOutputChange::Identical;
}

// Different semantic submits can use different threads. The one-shot ownership and artifact caps
// are process-wide; one ordered submit's split callbacks are called synchronously on its executor
// thread (execute_ordered_items_impl::flush_span). No lock is held across Vulkan or g_rtt access.
class ExactWriterProbe {
public:
    static constexpr uint32_t kAttempts = 4;
    static constexpr uint64_t kRawCap = 160ull << 20;
    static constexpr uint64_t kBmpCap = 128ull << 20;

    void configure(ExactWriterProbeSpec spec) {
        std::lock_guard lock(mu_);
        spec_ = spec;
        state_ = State::Idle;
        attempts_ = 0;
        raw_bytes_ = bmp_bytes_ = 0;
    }

    // 0 means unarmed, too early, busy, exhausted, or already complete.
    uint32_t claim(uint64_t elapsed_ms) {
        std::lock_guard lock(mu_);
        if (!spec_.armed || elapsed_ms < spec_.after_ms || state_ != State::Idle ||
            attempts_ >= kAttempts) return 0;
        state_ = State::Busy;
        return ++attempts_;
    }

    bool charge_raw(uint64_t bytes) { return charge(bytes, kRawCap, raw_bytes_); }
    bool charge_bmp(uint64_t bytes) { return charge(bytes, kBmpCap, bmp_bytes_); }

    bool mark_bracketed() {
        std::lock_guard lock(mu_);
        if (state_ != State::Busy) return false;
        state_ = State::PendingFinal;
        return true;
    }

    void finish_failed_attempt() {
        std::lock_guard lock(mu_);
        if (state_ != State::Busy) return;
        state_ = attempts_ == kAttempts ? State::Exhausted : State::Idle;
    }

    bool finish_final(bool published) {
        std::lock_guard lock(mu_);
        if (state_ != State::PendingFinal) return false;
        state_ = published ? State::Complete : State::Exhausted;
        return published;
    }

    bool bracketed() const {
        std::lock_guard lock(mu_);
        return state_ == State::PendingFinal || state_ == State::Complete;
    }

    bool complete() const {
        std::lock_guard lock(mu_);
        return state_ == State::Complete;
    }

    bool exhausted() const {
        std::lock_guard lock(mu_);
        return state_ == State::Exhausted;
    }

    bool expired(uint64_t elapsed_ms) {
        std::lock_guard lock(mu_);
        if (!spec_.armed || state_ != State::Idle ||
            spec_.after_ms > UINT64_MAX - 10'000 ||
            elapsed_ms < spec_.after_ms + 10'000) return false;
        state_ = State::Exhausted;
        return true;
    }

    uint32_t attempts() const {
        std::lock_guard lock(mu_);
        return attempts_;
    }

private:
    enum class State : uint8_t { Idle, Busy, PendingFinal, Complete, Exhausted };
    bool charge(uint64_t bytes, uint64_t cap, uint64_t& count) {
        std::lock_guard lock(mu_);
        if ((state_ != State::Busy && state_ != State::PendingFinal) ||
            !bytes || count > cap || bytes > cap - count)
            return false;
        count += bytes;
        return true;
    }

    mutable std::mutex mu_;
    ExactWriterProbeSpec spec_{};
    State state_ = State::Idle;
    uint32_t attempts_ = 0;
    uint64_t raw_bytes_ = 0;
    uint64_t bmp_bytes_ = 0;
};

inline uint32_t claim_exact_writer(ExactWriterProbe& probe, ExactWriterMatch match,
                                   uint64_t elapsed_ms) {
    return match == ExactWriterMatch::Exact ? probe.claim(elapsed_ms) : 0;
}

} // namespace prosper::frontend
