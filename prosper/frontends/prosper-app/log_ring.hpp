#pragma once
// log_ring.hpp — a bounded, thread-safe line buffer behind the library's Game Log panel.
//
// One producer (the stdout/stderr reader thread main.cpp starts) appends; the UI thread snapshots
// once per frame. std::mutex/deque only — no SDL, no ImGui — so the capacity and truncation rules
// are unit-tested without a window.

#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::frontend {

class LogRing {
public:
    static constexpr size_t kMaxLines = 512;   // old lines fall off; the panel is a tail, not storage
    static constexpr size_t kMaxLine = 2048;   // one pathological line cannot eat the whole budget

    void push(const std::string& line) {
        std::lock_guard<std::mutex> lock(mu_);
        lines_.push_back(line.size() > kMaxLine ? line.substr(0, kMaxLine) : line);
        while (lines_.size() > kMaxLines) lines_.pop_front();
    }

    std::vector<std::string> snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return std::vector<std::string>(lines_.begin(), lines_.end());
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        lines_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return lines_.size();
    }

    bool empty() const { return size() == 0; }

private:
    mutable std::mutex mu_;
    std::deque<std::string> lines_;
};

} // namespace prosper::frontend
