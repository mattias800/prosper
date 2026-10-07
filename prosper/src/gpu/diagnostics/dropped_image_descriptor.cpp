// dropped_image_descriptor.cpp -- see the header.
#include "gpu/diagnostics/dropped_image_descriptor.hpp"

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

bool note_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                   const std::array<uint32_t, 8>& words, const char* reason) {
    // After the cap every call returns here without the lock: a title that declines an image use
    // on every draw pays one relaxed load per decline, never a process-global mutex (P4).
    static std::atomic<bool> full{false};
    if (full.load(std::memory_order_relaxed)) return false;
    static std::mutex mutex;
    // Keyed by the reason's characters through a string_view of a literal: a repeat below the cap
    // is a lookup only, with no node or string allocation.
    using Key = std::tuple<uint64_t, uint32_t, std::string_view>;
    static std::set<Key> seen;
    const Key key{program, pc, std::string_view(reason ? reason : "unknown")};
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (seen.size() >= kDroppedImageDescriptorMaxReports) {
            full.store(true, std::memory_order_relaxed);
            return false;
        }
        if (seen.find(key) != seen.end()) return false;
        seen.insert(key);
        if (seen.size() >= kDroppedImageDescriptorMaxReports)
            full.store(true, std::memory_order_relaxed);
    }
    const std::string line = format_dropped_image_descriptor(program, pc, srsrc, words, reason);
    std::fprintf(stderr, "%s\n", line.c_str());
    return true;
}

}   // namespace prosper::gpu
