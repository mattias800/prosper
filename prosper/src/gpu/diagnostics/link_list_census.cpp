#include "gpu/diagnostics/link_list_census.hpp"

#include "gpu/diagnostics/watch_list.hpp"

#include <algorithm>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace prosper::gpu {
namespace {

// The largest link value whose successor word is still INSIDE the buffer. Every link above it reads
// out of range and is therefore answered with an architectural zero, so they all behave identically
// and none of them needs a memo slot. This is what keeps the memo bounded by the buffer rather than
// by the 2^32 values a link can take.
uint64_t memo_slot_count(size_t words, const LinkListEncoding& enc) {
    if (words <= enc.next_dword_offset) return 0;
    return ((words - enc.next_dword_offset - 1) / enc.record_stride_dwords) + 1;
}

// One step of the walk, modelling `bounded_cbuf_load`: an index at or past the end reads ZERO, which
// is a valid link and not an exit.
uint32_t step(std::span<const uint32_t> words, uint32_t link, const LinkListEncoding& enc,
              bool* out_of_range) {
    const uint64_t idx =
        static_cast<uint64_t>(link) * enc.record_stride_dwords + enc.next_dword_offset;
    if (idx < words.size()) {
        if (out_of_range) *out_of_range = false;
        return words[static_cast<size_t>(idx)];
    }
    if (out_of_range) *out_of_range = true;
    return 0u;
}

// Shared classifier for both censuses. Memoized over the in-range links, so the whole buffer is
// classified in O(records) rather than O(records * chain length).
class LinkClassifier {
public:
    LinkClassifier(std::span<const uint32_t> words, const LinkListEncoding& enc)
        : words_(words), enc_(enc), slots_(static_cast<size_t>(memo_slot_count(words.size(), enc))) {
        state_.assign(slots_, 0);
        depth_.assign(slots_, 0);
        oob_.assign(slots_, 0);
        seen_at_.assign(slots_, UINT32_MAX);
        on_cycle_.assign(slots_, 0);
    }

    struct Verdict {
        LinkWalkVerdict verdict = LinkWalkVerdict::Terminates;
        uint32_t depth = 0;
        bool out_of_range = false;
        uint32_t closing_link = 0;   // meaningful only when Cyclic
        uint32_t closing_next = 0;
    };

    Verdict classify(uint32_t start) {
        path_.clear();
        path_oob_.clear();
        uint32_t link = start;
        Verdict v;
        uint32_t suffix_depth = 0;
        bool suffix_oob = false;
        LinkWalkVerdict verdict = LinkWalkVerdict::Terminates;
        while (true) {
            if (link == enc_.terminator) {
                verdict = LinkWalkVerdict::Terminates;
                suffix_depth = 0;
                suffix_oob = false;
                break;
            }
            if (link < slots_) {
                if (state_[link]) {
                    verdict = state_[link] == 1 ? LinkWalkVerdict::Terminates : LinkWalkVerdict::Cyclic;
                    suffix_depth = depth_[link];
                    suffix_oob = oob_[link] != 0;
                    break;
                }
                if (seen_at_[link] != UINT32_MAX) {
                    // Closed a ring on THIS walk. Everything from the first sighting onwards lies on
                    // the cycle; everything before it merely leads into one.
                    for (size_t k = seen_at_[link]; k < path_.size(); ++k) mark_cycle_node(path_[k]);
                    v.closing_link = link;
                    v.closing_next = step(words_, link, enc_, nullptr);
                    verdict = LinkWalkVerdict::Cyclic;
                    suffix_depth = 0;
                    suffix_oob = false;
                    break;
                }
                seen_at_[link] = static_cast<uint32_t>(path_.size());
                path_.push_back(link);
            } else if (slots_ == 0) {
                // Every link reads out of range, so every link steps to zero, and zero does too.
                // There is no memo slot to detect the repeat with, so decide it here.
                v.closing_link = link;
                v.closing_next = 0;
                verdict = LinkWalkVerdict::Cyclic;
                suffix_depth = 0;
                suffix_oob = true;
                break;
            }
            bool oob = false;
            const uint32_t next = step(words_, link, enc_, &oob);
            if (link < slots_) path_oob_.push_back(oob ? 1u : 0u);
            else if (oob) suffix_oob_seen_ = true;   // an unmemoized hop; folded in below
            link = next;
        }
        // Unwind: every node on the path shares the suffix's verdict, and its depth and
        // out-of-range-ness are its own step plus the suffix's.
        const uint8_t state = verdict == LinkWalkVerdict::Terminates ? 1u : 2u;
        uint32_t d = suffix_depth;
        bool acc = suffix_oob || suffix_oob_seen_;
        for (size_t k = path_.size(); k-- > 0;) {
            if (state == 1u) d += 1u;
            acc = acc || path_oob_[k] != 0;
            state_[path_[k]] = state;
            depth_[path_[k]] = state == 1u ? d : 0u;
            oob_[path_[k]] = acc ? 1u : 0u;
            seen_at_[path_[k]] = UINT32_MAX;
        }
        suffix_oob_seen_ = false;
        v.verdict = verdict;
        v.depth = state == 1u ? (path_.empty() ? suffix_depth : d) : 0u;
        v.out_of_range = acc;
        return v;
    }

    uint32_t cycle_node_count() const { return cycle_node_count_; }
    const std::vector<uint32_t>& cycle_samples() const { return cycle_samples_; }
    size_t slots() const { return slots_; }

private:
    std::span<const uint32_t> words_;
    LinkListEncoding enc_;
    size_t slots_ = 0;
    std::vector<uint8_t> state_, oob_;
    std::vector<uint32_t> depth_, seen_at_;
    std::vector<uint32_t> path_;
    std::vector<uint8_t> path_oob_;
    // A bitmap rather than a set: a wholly cyclic 4-million-record pool is exactly the case this
    // instrument exists to find, and a std::set of four million nodes would cost more memory than
    // the buffer being scanned. The count stays exact; only six members are retained.
    std::vector<uint8_t> on_cycle_;
    uint32_t cycle_node_count_ = 0;
    std::vector<uint32_t> cycle_samples_;
    bool suffix_oob_seen_ = false;

    void mark_cycle_node(uint32_t node) {
        if (node >= on_cycle_.size() || on_cycle_[node]) return;
        on_cycle_[node] = 1u;
        ++cycle_node_count_;
        if (cycle_samples_.size() < 6u) cycle_samples_.push_back(node);
    }
};

void fill_samples(LinkListCensus& out, uint32_t cycle_node_count,
                  const std::vector<uint32_t>& cycle_samples,
                  std::span<const uint32_t> words, const LinkListEncoding& enc) {
    out.cycle_nodes = cycle_node_count;
    for (uint32_t node : cycle_samples) {
        if (out.sample_count >= 6u) break;
        out.sample_start[out.sample_count] = node;
        out.sample_link[out.sample_count] = node;
        out.sample_next[out.sample_count] = step(words, node, enc, nullptr);
        ++out.sample_count;
    }
}

LinkListCensus census(std::span<const uint32_t> words, std::span<const uint32_t> starts,
                      bool self_walk, const LinkListEncoding& enc) {
    LinkListCensus out;
    if (!enc.valid()) return out;
    out.words = static_cast<uint32_t>(std::min<size_t>(words.size(), UINT32_MAX));
    out.records = static_cast<uint32_t>(words.size() / enc.record_stride_dwords);
    LinkClassifier classifier(words, enc);
    const size_t n = self_walk ? classifier.slots() : starts.size();
    for (size_t i = 0; i < n; ++i) {
        const uint32_t start = self_walk ? static_cast<uint32_t>(i) : starts[i];
        const auto v = classifier.classify(start);
        ++out.starts;
        if (v.verdict == LinkWalkVerdict::Terminates) {
            ++out.terminating;
            out.longest = std::max(out.longest, v.depth);
        } else {
            ++out.cyclic;
        }
        if (v.out_of_range) ++out.oob_starts;
    }
    fill_samples(out, classifier.cycle_node_count(), classifier.cycle_samples(), words, enc);
    return out;
}

}  // namespace

BufferWordHistogram histogram_words(std::span<const uint32_t> words, uint32_t terminator) {
    BufferWordHistogram h;
    h.words = static_cast<uint32_t>(std::min<size_t>(words.size(), UINT32_MAX));
    for (size_t i = 0; i < words.size(); ++i) {
        const uint32_t w = words[i];
        if (w == 0u) { ++h.zero; continue; }
        if (w == terminator) { ++h.terminator; continue; }
        if (h.other == 0) {
            h.min_other = h.max_other = w;
            h.first_other_index = static_cast<uint32_t>(std::min<size_t>(i, UINT32_MAX));
            h.first_other_value = w;
        } else {
            h.min_other = std::min(h.min_other, w);
            h.max_other = std::max(h.max_other, w);
        }
        ++h.other;
    }
    return h;
}

LinkWalkVerdict walk_terminates(std::span<const uint32_t> words, uint32_t start,
                                const LinkListEncoding& enc, uint64_t visited_budget,
                                uint32_t* closing_link, bool* went_out_of_range) {
    if (closing_link) *closing_link = 0;
    if (went_out_of_range) *went_out_of_range = false;
    if (!enc.valid()) return LinkWalkVerdict::Terminates;
    uint32_t link = start;
    for (uint64_t taken = 0; taken < visited_budget; ++taken) {
        if (link == enc.terminator) return LinkWalkVerdict::Terminates;
        bool oob = false;
        const uint32_t next = step(words, link, enc, &oob);
        if (oob && went_out_of_range) *went_out_of_range = true;
        if (closing_link) *closing_link = link;
        link = next;
    }
    return LinkWalkVerdict::Cyclic;
}

LinkListCensus census_self_walk(std::span<const uint32_t> words, const LinkListEncoding& enc) {
    return census(words, {}, true, enc);
}

LinkListCensus census_head_walk(std::span<const uint32_t> words, std::span<const uint32_t> heads,
                                const LinkListEncoding& enc) {
    return census(words, heads, false, enc);
}

namespace {

// Strict unsigned parse. Rejects an empty string, leading whitespace/sign, and trailing junk --
// a setting nobody can arm by accident is worth more than one that is forgiving, because the
// forgiving one produces a census under an encoding the reader did not ask for.
bool parse_strict_u64(const char* text, uint64_t& out) {
    if (!text || !*text) return false;
    if (*text == '+' || *text == '-' || *text == ' ' || *text == '\t') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(text, &end, 0);
    if (errno != 0 || !end || *end != '\0' || end == text) return false;
    out = static_cast<uint64_t>(v);
    return true;
}

}  // namespace

DrawLinkScanSettings parse_draw_link_scan_settings(const char* stride, const char* next,
                                                   const char* terminator, const char* max_scans,
                                                   const char* heads, const char* records) {
    DrawLinkScanSettings s;
    uint64_t v = 0;
    const auto take = [&](const char* text, uint64_t lo, uint64_t hi, uint64_t& dst) {
        if (!text || !*text) return;
        if (!parse_strict_u64(text, v) || v < lo || v > hi) { s.valid = false; return; }
        dst = v;
    };
    uint64_t stride_v = s.encoding.record_stride_dwords;
    uint64_t next_v = s.encoding.next_dword_offset;
    uint64_t term_v = s.encoding.terminator;
    uint64_t max_v = s.max_scans_per_buffer;
    uint64_t heads_v = s.heads_binding;
    uint64_t records_v = s.records_binding;
    take(stride, 1, 4096, stride_v);
    take(next, 0, 4095, next_v);
    take(terminator, 0, 0xffffffffull, term_v);
    take(max_scans, 1, 1u << 20, max_v);
    take(heads, 0, 0xfffffffeull, heads_v);
    take(records, 0, 0xfffffffeull, records_v);
    s.encoding.record_stride_dwords = static_cast<uint32_t>(stride_v);
    s.encoding.next_dword_offset = static_cast<uint32_t>(next_v);
    s.encoding.terminator = static_cast<uint32_t>(term_v);
    s.max_scans_per_buffer = static_cast<uint32_t>(max_v);
    s.heads_binding = static_cast<uint32_t>(heads_v);
    s.records_binding = static_cast<uint32_t>(records_v);
    // A successor that sits outside its own record is not a stride/offset pair any encoding can
    // mean, so reject it here rather than emit a census whose records overlap.
    if (!s.encoding.valid()) s.valid = false;
    // Naming one half of the cross-buffer walk and not the other is a half-armed instrument; the
    // walk needs both, and a silent fallback to the self-walk would read as "the cross walk found
    // nothing".
    if ((s.heads_binding == UINT32_MAX) != (s.records_binding == UINT32_MAX)) s.valid = false;
    return s;
}

DrawLinkScanSelector::ConfigureResult DrawLinkScanSelector::configure(const char* spec) {
    addresses_.clear();
    if (!spec || !*spec) return ConfigureResult::Unset;
    std::vector<uint64_t> parsed;
    if (!parse_hex_watch_list(spec, parsed)) return ConfigureResult::Malformed;
    addresses_ = std::move(parsed);
    return ConfigureResult::Armed;
}

bool DrawLinkScanSelector::matches(uint64_t vs_addr, uint64_t vs_chain_addr,
                                   uint64_t fs_addr) const {
    if (addresses_.empty()) return false;
    const auto has = [this](uint64_t address) {
        return address != 0 &&
               std::find(addresses_.begin(), addresses_.end(), address) != addresses_.end();
    };
    return has(vs_addr) || has(vs_chain_addr) || has(fs_addr);
}

uint32_t DrawLinkScanSelector::should_scan(uint64_t program, uint32_t set, uint32_t binding,
                                           uint32_t max_scans, bool* exhausted) {
    if (exhausted) *exhausted = false;
    const auto key = std::make_pair(program, (static_cast<uint64_t>(set) << 32) | binding);
    std::lock_guard lock(mutex_);
    uint32_t& seen = counts_[key];
    if (seen >= max_scans) {
        // Report the transition exactly once: the caller prints "capped" on the first refusal so a
        // reader can tell a silent instrument from a satisfied one.
        if (seen == max_scans && exhausted) *exhausted = true;
        if (seen < UINT32_MAX) ++seen;
        return 0;
    }
    return ++seen;
}

DrawLinkScanSelector& draw_link_scan_selector() {
    // Held by reference, not returned by value: the selector owns a mutex and is therefore neither
    // copyable nor movable, so it must be configured in place.
    static DrawLinkScanSelector selector;
    static const bool configured = [] {
        const char* spec = std::getenv("PROSPER_DRAW_LINKSCAN");
        const auto result = selector.configure(spec);
        if (result == DrawLinkScanSelector::ConfigureResult::Malformed) {
            std::fprintf(stderr,
                         "[linkscan] ignoring malformed PROSPER_DRAW_LINKSCAN=\"%s\" -- expected a "
                         "comma-separated list of non-zero 0x addresses. NOTHING is scanned\n",
                         spec ? spec : "");
        } else if (result == DrawLinkScanSelector::ConfigureResult::Armed) {
            const auto& cfg = draw_link_scan_settings();
            std::fprintf(stderr,
                         "[linkscan] PROSPER_DRAW_LINKSCAN=%s -> %zu program(s); encoding "
                         "stride=%u next=+%u term=0x%08x max-scans-per-buffer=%u heads=%d "
                         "records=%d%s\n",
                         spec, selector.size(), cfg.encoding.record_stride_dwords,
                         cfg.encoding.next_dword_offset, cfg.encoding.terminator,
                         cfg.max_scans_per_buffer,
                         cfg.heads_binding == UINT32_MAX ? -1 : (int)cfg.heads_binding,
                         cfg.records_binding == UINT32_MAX ? -1 : (int)cfg.records_binding,
                         cfg.valid ? "" : "  *** SETTINGS REJECTED -- nothing is scanned");
        }
        return true;
    }();
    (void)configured;
    return selector;
}

const DrawLinkScanSettings& draw_link_scan_settings() {
    static const DrawLinkScanSettings settings = parse_draw_link_scan_settings(
        std::getenv("PROSPER_DRAW_LINKSCAN_STRIDE"), std::getenv("PROSPER_DRAW_LINKSCAN_NEXT"),
        std::getenv("PROSPER_DRAW_LINKSCAN_TERM"), std::getenv("PROSPER_DRAW_LINKSCAN_MAX"),
        std::getenv("PROSPER_DRAW_LINKSCAN_HEADS"), std::getenv("PROSPER_DRAW_LINKSCAN_RECORDS"));
    return settings;
}

bool draw_link_scan_enabled() {
    return draw_link_scan_selector().armed() && draw_link_scan_settings().valid;
}

}  // namespace prosper::gpu
