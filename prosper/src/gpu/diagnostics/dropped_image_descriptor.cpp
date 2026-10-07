// dropped_image_descriptor.cpp -- see the header.
#include "gpu/diagnostics/dropped_image_descriptor.hpp"

#include <cstdio>
#include <mutex>
#include <set>
#include <tuple>

namespace prosper::gpu {

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
    char line[400];
    std::snprintf(line, sizeof line,
                  "[t8-dropped] program=0x%llx pc=%u srsrc=%s reason=%s base=0x%llx type=%u "
                  "dst_sel=0x%03x raw %08x %08x %08x %08x %08x %08x %08x %08x",
                  static_cast<unsigned long long>(program), pc, srsrc_text,
                  reason ? reason : "unknown", static_cast<unsigned long long>(base),
                  (w[3] >> 28) & 0xfu, w[3] & 0xfffu, w[0], w[1], w[2], w[3], w[4], w[5], w[6],
                  w[7]);
    return line;
}

bool note_dropped_image_descriptor(uint64_t program, uint32_t pc, int srsrc,
                                   const std::array<uint32_t, 8>& words, const char* reason) {
    static std::mutex mutex;
    static std::set<std::tuple<uint64_t, uint32_t, std::string>> seen;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (seen.size() >= kDroppedImageDescriptorMaxReports) return false;
        if (!seen.emplace(program, pc, reason ? reason : "unknown").second) return false;
    }
    const std::string line = format_dropped_image_descriptor(program, pc, srsrc, words, reason);
    std::fprintf(stderr, "%s\n", line.c_str());
    return true;
}

}   // namespace prosper::gpu
