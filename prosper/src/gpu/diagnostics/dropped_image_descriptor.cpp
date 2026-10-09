// dropped_image_descriptor.cpp -- see the header.
#include "gpu/diagnostics/dropped_image_descriptor.hpp"

#include "diagnostics/exit_census.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

namespace prosper::gpu {

const char* fold_t8_decline_reason(const FoldT8Admission& a) {
    if (!a.words_known) return "words-unknown";
    if (a.branchy_x16 && !a.mapped_t8) return "branchy-x16-unproven";
    if (!a.have_t8 && !a.mapped_t8 && !a.seed_provenance) return "no-provenance";
    if (!a.have_t8 && !a.mapped_t8 && !a.seed_publishable) return "direct-seed-implausible";
    return nullptr;
}

std::string format_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                            const std::array<uint32_t, 8>& w, const char* reason) {
    // The fields the plausibility and null gates read, extracted directly so the line does not
    // depend on (or re-run) the decoder whose verdict it is reporting: Base40 (stored >>8, word0
    // plus word1[7:0]), TYPE (word3[31:28]) and the four DST_SEL fields (word3[11:0]).
    const uint64_t base = ((uint64_t(w[0]) | (uint64_t(w[1] & 0xffu) << 32))) << 8;
    char srsrc_text[16];
    if (srsrc >= 0)
        std::snprintf(srsrc_text, sizeof srsrc_text, "s%d", srsrc);
    else
        std::snprintf(srsrc_text, sizeof srsrc_text, "unknown");
    char pc_text[16];
    if (pc == UINT32_MAX)
        std::snprintf(pc_text, sizeof pc_text, "none");
    else
        std::snprintf(pc_text, sizeof pc_text, "%u", pc);
    char line[400];
    std::snprintf(line, sizeof line,
                  "[t8-dropped] program=0x%llx pc=%s srsrc=%s reason=%s base=0x%llx type=%u "
                  "dst_sel=0x%03x raw %08x %08x %08x %08x %08x %08x %08x %08x",
                  static_cast<unsigned long long>(program), pc_text, srsrc_text,
                  reason ? reason : "unknown", static_cast<unsigned long long>(base),
                  (w[3] >> 28) & 0xfu, w[3] & 0xfffu, w[0], w[1], w[2], w[3], w[4], w[5], w[6],
                  w[7]);
    return line;
}

namespace {

// One bounded, deduped report site. Each caller owns its own instance, so the two witnesses keep
// separate caps.
class BoundedSiteReport {
public:
    // True the first time (program, pc, reason) is seen, until the cap. After the cap every call
    // returns without the lock: a title that hits the site on every draw pays one relaxed load per
    // call, never a process-global mutex (P4).
    bool admit(uint64_t program, uint32_t pc, const char* reason) {
        if (full_.load(std::memory_order_relaxed)) return false;
        // Keyed by the reason's characters through a string_view of a literal: a repeat below the
        // cap is a lookup only, with no node or string allocation.
        const Key key{program, pc, std::string_view(reason ? reason : "unknown")};
        std::lock_guard<std::mutex> lock(mutex_);
        if (seen_.size() >= kDroppedImageDescriptorMaxReports) {
            full_.store(true, std::memory_order_relaxed);
            return false;
        }
        if (!seen_.insert(key).second) return false;
        if (seen_.size() >= kDroppedImageDescriptorMaxReports)
            full_.store(true, std::memory_order_relaxed);
        return true;
    }

private:
    using Key = std::tuple<uint64_t, uint32_t, std::string_view>;
    std::atomic<bool> full_{false};
    std::mutex mutex_;
    std::set<Key> seen_;
};

}   // namespace

bool note_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                   const std::array<uint32_t, 8>& words, const char* reason) {
    static BoundedSiteReport report;
    if (!report.admit(program, pc, reason)) return false;
    const std::string line = format_dropped_image_descriptor(program, pc, srsrc, words, reason);
    std::fprintf(stderr, "%s\n", line.c_str());
    return true;
}

std::string format_unbound_image_descriptor(uint64_t program, uint32_t pc,
                                            const std::array<uint32_t, 8>& words,
                                            const char* reason) {
    // The same fields as the dropped line; only the verdict differs.
    std::string line = format_dropped_image_descriptor(program, pc, -1, words, reason);
    constexpr std::string_view kDropped = "[t8-dropped]";
    if (line.compare(0, kDropped.size(), kDropped) == 0)
        line.replace(0, kDropped.size(), "[t8-unbound]");
    return line + " -> null image (skippable instruction)";
}

namespace {

// Every null binding is counted, not only the first per site, and the total is printed at exit next
// to the other RUN TOTAL lines: a draw this rule rescues leaves the dropped-draw census, so without
// a count of its own the conversion would be invisible in the summary people read first (#4801).
std::atomic<uint64_t> g_unbound_uses{0};
std::atomic<uint64_t> g_unbound_sites{0};

bool print_unbound_total() {
    const uint64_t uses = g_unbound_uses.load(std::memory_order_relaxed);
    if (!uses) return false;
    std::fprintf(stderr,
                 "[t8-unbound] RUN TOTAL null-bound uses=%llu sites=%llu (T# words that are not a "
                 "descriptor, read only by a scalar-skippable instruction; #4796)\n",
                 static_cast<unsigned long long>(uses),
                 static_cast<unsigned long long>(g_unbound_sites.load(std::memory_order_relaxed)));
    return true;
}

}   // namespace

uint64_t unbound_image_descriptor_uses() {
    return g_unbound_uses.load(std::memory_order_relaxed);
}

bool note_unbound_image_descriptor(uint64_t program, uint32_t pc,
                                   const std::array<uint32_t, 8>& words, const char* reason) {
    // Not std::atexit: prosper leaves through _exit(), so only the exit-census hook runs
    // (exit_census.hpp). Registered on first use, never during static initialisation.
    static const bool registered = [] {
        prosper::diagnostics::register_census(nullptr, print_unbound_total);
        return true;
    }();
    (void)registered;
    g_unbound_uses.fetch_add(1, std::memory_order_relaxed);
    static BoundedSiteReport report;
    if (!report.admit(program, pc, reason)) return false;
    g_unbound_sites.fetch_add(1, std::memory_order_relaxed);
    const std::string line = format_unbound_image_descriptor(program, pc, words, reason);
    std::fprintf(stderr, "%s\n", line.c_str());
    return true;
}

}   // namespace prosper::gpu
