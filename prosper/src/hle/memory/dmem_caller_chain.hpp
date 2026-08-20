#pragma once

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace prosper {

enum class DmemCallerChainState : uint8_t {
    Disabled,
    Known,
    Unknown,
    Overflow,
};

struct DmemCallerChainResult {
    DmemCallerChainState state = DmemCallerChainState::Disabled;
    uint32_t id = 0;
    bool first = false;
};

struct DmemCallerChainFrame {
    const char* module = nullptr;
    uint64_t offset = 0;
};

inline void append_dmem_caller_chain_number(std::string& line, uint64_t value, int base) {
    char digits[32] = {};
    const auto converted = std::to_chars(digits, digits + sizeof(digits), value, base);
    if (converted.ec == std::errc{}) line.append(digits, converted.ptr);
}

// Build the whole definition before taking the output lock. Besides keeping the lock hold short,
// this makes it impossible for another first-seen chain to splice its ID or frames into this line.
inline std::string format_dmem_caller_chain_definition(
        uint32_t id, uint64_t len, const DmemCallerChainFrame* frames, size_t frame_count,
        int scanned_slots, int requested_slots) {
    std::string line;
    line.reserve(256);
    line += "[dmem-caller] caller-chain=";
    append_dmem_caller_chain_number(line, id, 10);
    line += " alloc_main_dmem len=0x";
    append_dmem_caller_chain_number(line, len, 16);
    line += " from";
    for (size_t i = 0; i < frame_count; ++i) {
        line.push_back(' ');
        line += frames[i].module ? frames[i].module : "<unknown-module>";
        line += "+0x";
        append_dmem_caller_chain_number(line, frames[i].offset, 16);
    }
    // A clamped scan reports a SHORTER chain, not a shallower one — say which it was (#1755).
    if (scanned_slots < requested_slots) {
        line += " [scan clamped to ";
        append_dmem_caller_chain_number(line, static_cast<uint64_t>(scanned_slots), 10);
        line.push_back('/');
        append_dmem_caller_chain_number(line, static_cast<uint64_t>(requested_slots), 10);
        line += " slots by stack top]";
    }
    line.push_back('\n');
    return line;
}

// A single stdio call already holds the FILE lock for its duration; the explicit diagnostic lock
// additionally keeps this contract visible and prevents future multi-call emitters from recreating
// #1859's corrupted ID-to-chain table. All platforms use this exact writer.
inline void write_dmem_caller_chain_line(FILE* stream, std::string_view line) {
    static std::mutex output_mutex;
    std::lock_guard lock(output_mutex);
    (void)std::fwrite(line.data(), 1, line.size(), stream);
}

inline void write_dmem_caller_chain_definition(
        FILE* stream, uint32_t id, uint64_t len, const DmemCallerChainFrame* frames,
        size_t frame_count, int scanned_slots, int requested_slots) {
    const std::string line = format_dmem_caller_chain_definition(
        id, len, frames, frame_count, scanned_slots, requested_slots);
    write_dmem_caller_chain_line(stream, line);
}

// Correlation belongs on every enabled allocation event, not only the event that owns the one-time
// full-chain line. Keeping this decision beside the interner gives the old suppression defect a
// focused, mutation-testable seam.
inline bool dmem_caller_chain_correlates_allocation(const DmemCallerChainResult& result) {
    return result.state != DmemCallerChainState::Disabled;
}

// Bounded, run-local identity for PROSPER_DMEM_CALLER's existing first-two-return-address key.
// IDs start at one and remain stable for the life of this registry. Unknown stacks and a full
// registry are explicit states: neither can be mistaken for a missing diagnostic line.
template <size_t Capacity = 64>
class DmemCallerChainInterner {
public:
    static_assert(Capacity > 0);

    static constexpr size_t capacity() { return Capacity; }

    DmemCallerChainResult intern(uint64_t first, uint64_t second) {
        std::lock_guard lock(mutex_);
        if (!first && !second) {
            const bool is_first = !unknown_seen_;
            unknown_seen_ = true;
            return {DmemCallerChainState::Unknown, 0, is_first};
        }
        const std::pair<uint64_t, uint64_t> key{first, second};
        for (size_t i = 0; i < size_; ++i)
            if (keys_[i] == key)
                return {DmemCallerChainState::Known, static_cast<uint32_t>(i + 1), false};
        if (size_ == Capacity) {
            const bool is_first = !overflow_seen_;
            overflow_seen_ = true;
            return {DmemCallerChainState::Overflow, 0, is_first};
        }
        keys_[size_] = key;
        ++size_;
        return {DmemCallerChainState::Known, static_cast<uint32_t>(size_), true};
    }

    size_t size() const {
        std::lock_guard lock(mutex_);
        return size_;
    }

private:
    mutable std::mutex mutex_;
    std::array<std::pair<uint64_t, uint64_t>, Capacity> keys_{};
    size_t size_ = 0;
    bool unknown_seen_ = false;
    bool overflow_seen_ = false;
};

} // namespace prosper
