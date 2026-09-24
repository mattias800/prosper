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
    // A moving guest allocation may be selected by exact extent, but only when the shader,
    // input binding, one-draw shape, and color write also match.
    bool output_by_extent = false;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint64_t after_ms = 0;
};

struct ExactWriterSecondInputSpec {
    bool requested = false;
    bool armed = false;
    uint32_t binding = 0;
    uint64_t address = 0;
};

// BINDING:0xADDRESS. The binding names one reflected pixel resource, not just
// any descriptor whose guest range happens to overlap the requested image.
inline ExactWriterSecondInputSpec parse_exact_writer_second_input(const char* value) {
    ExactWriterSecondInputSpec spec;
    if (!value) return spec;
    spec.requested = true;
    const std::string_view text(value);
    const size_t colon = text.find(':');
    if (colon == std::string_view::npos || !colon ||
        !text.substr(colon + 1).starts_with("0x")) return spec;
    const auto binding = text.substr(0, colon);
    const auto address = text.substr(colon + 3);
    if (address.empty()) return spec;
    const auto [binding_end, binding_error] = std::from_chars(
        binding.data(), binding.data() + binding.size(), spec.binding, 10);
    const auto [address_end, address_error] = std::from_chars(
        address.data(), address.data() + address.size(), spec.address, 16);
    spec.armed = binding_error == std::errc{} &&
                 binding_end == binding.data() + binding.size() &&
                 address_error == std::errc{} &&
                 address_end == address.data() + address.size() && spec.address;
    return spec;
}

enum class ExactWriterSecondInputVerdict : uint8_t {
    Ready, MissingBinding, MissingImage, NoVisibleRgb
};

inline ExactWriterSecondInputVerdict exact_writer_second_input_verdict(
    bool exact_gpu_binding, bool readback_ok, size_t raw_nonzero, size_t rgb_nonblack) {
    if (!exact_gpu_binding) return ExactWriterSecondInputVerdict::MissingBinding;
    if (!readback_ok) return ExactWriterSecondInputVerdict::MissingImage;
    if (!raw_nonzero || !rgb_nonblack)
        return ExactWriterSecondInputVerdict::NoVisibleRgb;
    return ExactWriterSecondInputVerdict::Ready;
}

inline const char* exact_writer_second_input_verdict_name(
    ExactWriterSecondInputVerdict verdict) {
    switch (verdict) {
    case ExactWriterSecondInputVerdict::Ready: return "ready";
    case ExactWriterSecondInputVerdict::MissingBinding: return "missing-exact-gpu-binding";
    case ExactWriterSecondInputVerdict::MissingImage: return "missing-or-unreadable-exact-image";
    case ExactWriterSecondInputVerdict::NoVisibleRgb: return "no-visible-RGB";
    }
    return "unknown";
}

// 0xPS:0xINPUT:0xOUTPUT:ms:AFTER or 0xPS:0xINPUT:auto:WxH:ms:AFTER.
// Every field is exact; a partial parse must refuse.
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
    if (!hex(spec.ps) || !hex(spec.input)) return spec;
    if (rest.starts_with("auto:")) {
        spec.output_by_extent = true;
        rest.remove_prefix(5);
        const size_t x = rest.find('x');
        const size_t colon = rest.find(':');
        if (x == std::string_view::npos || colon == std::string_view::npos ||
            x == 0 || x + 1 >= colon) return spec;
        const auto width = rest.substr(0, x);
        const auto height = rest.substr(x + 1, colon - x - 1);
        const auto [width_end, width_error] = std::from_chars(
            width.data(), width.data() + width.size(), spec.output_width);
        const auto [height_end, height_error] = std::from_chars(
            height.data(), height.data() + height.size(), spec.output_height);
        if (width_error != std::errc{} || width_end != width.data() + width.size() ||
            height_error != std::errc{} || height_end != height.data() + height.size() ||
            !spec.output_width || !spec.output_height ||
            spec.output_width > 8192 || spec.output_height > 8192) return spec;
        rest.remove_prefix(colon + 1);
    } else if (!hex(spec.output) || spec.input == spec.output) return spec;
    if (!rest.starts_with("ms:")) return spec;
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
                                           uint32_t actual_width, uint32_t actual_height,
                                           size_t original_draws, size_t adjusted_draws,
                                           bool color0_write, const ExactWriterProbeSpec& spec) {
    if (actual_ps != spec.ps) return ExactWriterMatch::WrongPs;
    if (!actual_target || actual_target == spec.input ||
        (spec.output_by_extent
             ? actual_width != spec.output_width || actual_height != spec.output_height
             : actual_target != spec.output)) return ExactWriterMatch::WrongTarget;
    if (original_draws != 1 || adjusted_draws != 1) return ExactWriterMatch::MultiDraw;
    if (!color0_write) return ExactWriterMatch::NoColorWrite;
    return ExactWriterMatch::Exact;
}

inline bool exact_writer_scene_input_visible(size_t raw_nonzero, size_t rgb_nonblack) {
    return raw_nonzero != 0 && rgb_nonblack != 0;
}

// A saved module can be valid for one draw yet wrong for earlier pipeline variants of the same
// guest program. When requested, admit it only while the exact-writer probe owns that draw.
inline bool exact_writer_file_override_selected(bool gated, bool probe_armed,
                                                uint64_t draw_index, uint64_t selected_index) {
    return !gated || (probe_armed && selected_index != UINT64_MAX &&
                      draw_index == selected_index);
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
