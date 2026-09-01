#include "gpu/diagnostics/draw_program_skip.hpp"

#include "gpu/diagnostics/diag_ratelimit.hpp"
#include "gpu/diagnostics/watch_list.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace prosper::gpu {
namespace {

// First 8 occurrences of each (program, stage), then every power of two. The ordinal is on every
// line, so the tail stays readable without the cap ever masquerading as the rate
// (diag_ratelimit.hpp).
constexpr uint64_t kFirstN = 8;

}  // namespace

const char* draw_program_stage_name(DrawProgramStage stage) {
    switch (stage) {
        case DrawProgramStage::Vertex:      return "vs";
        case DrawProgramStage::VertexChain: return "vs-chain";
        case DrawProgramStage::Fragment:    return "ps";
        case DrawProgramStage::None:        break;
    }
    return "none";
}

DrawProgramSkipSelector::ConfigureResult DrawProgramSkipSelector::configure(const char* spec) {
    addresses_.clear();
    if (!spec || !*spec) return ConfigureResult::Unset;
    std::vector<uint64_t> parsed;
    if (!parse_hex_watch_list(spec, parsed)) return ConfigureResult::Malformed;
    addresses_ = std::move(parsed);
    return ConfigureResult::Armed;
}

DrawSkipDecision DrawProgramSkipSelector::evaluate(uint64_t vs_addr, uint64_t vs_chain_addr,
                                                  uint64_t fs_addr) {
    DrawSkipDecision decision;
    // The disarmed fast path: one empty() test, no lock, no allocation. This runs per draw on every
    // default boot, so it must cost nothing measurable when nothing is selected.
    if (addresses_.empty()) return decision;

    // Ordered vertex, chain, pixel. A guest address names exactly one program, so at most one of
    // these can match and the order only decides which name is reported when a caller passes the
    // same address twice (a chained NGG program whose ES and main addresses coincide).
    const auto matches = [this](uint64_t address) {
        return address != 0 &&
               std::find(addresses_.begin(), addresses_.end(), address) != addresses_.end();
    };
    if (matches(vs_addr)) {
        decision.stage = DrawProgramStage::Vertex;
        decision.address = vs_addr;
    } else if (matches(vs_chain_addr)) {
        decision.stage = DrawProgramStage::VertexChain;
        decision.address = vs_chain_addr;
    } else if (matches(fs_addr)) {
        decision.stage = DrawProgramStage::Fragment;
        decision.address = fs_addr;
    } else {
        return decision;
    }
    decision.skip = true;

    std::lock_guard lock(mutex_);
    const uint64_t ordinal =
        ++counts_[{decision.address, static_cast<uint8_t>(decision.stage)}];
    ++total_;
    decision.ordinal = ordinal;
    decision.print = diag_should_print(ordinal, kFirstN);
    return decision;
}

uint64_t DrawProgramSkipSelector::skipped_total() const {
    std::lock_guard lock(mutex_);
    return total_;
}

DrawProgramCensus::Sighting DrawProgramCensus::observe(uint64_t vs_addr, uint64_t vs_chain_addr,
                                                      uint64_t fs_addr) {
    Sighting sighting;
    std::lock_guard lock(mutex_);
    const Key key{vs_addr, vs_chain_addr, fs_addr};
    const auto [it, inserted] = counts_.emplace(key, 0u);
    sighting.ordinal = ++it->second;
    sighting.first = inserted;
    sighting.distinct = counts_.size();
    // First sighting always prints — that is the census. Recurrences print at powers of two so a
    // program that suddenly dominates a frame is still visible without a per-draw firehose.
    sighting.print = inserted || diag_should_print(sighting.ordinal, 1);
    return sighting;
}

uint64_t DrawProgramCensus::distinct_programs() const {
    std::lock_guard lock(mutex_);
    return static_cast<uint64_t>(counts_.size());
}

DrawProgramSkipSelector& draw_program_skip_selector() {
    // Two statics rather than one initialized from a lambda's return: the selector holds a mutex, so
    // it is neither copyable nor movable and cannot be returned by value. Both initializations are
    // still thread-safe (magic statics), and the second runs exactly once.
    static DrawProgramSkipSelector selector;
    [[maybe_unused]] static const bool configured_once = [] {
        DrawProgramSkipSelector& s = selector;
        const char* spec = std::getenv("PROSPER_SKIP_DRAW_PROGRAM");
        switch (s.configure(spec)) {
            case DrawProgramSkipSelector::ConfigureResult::Unset:
                break;   // silence is correct here: the default boot must not print
            case DrawProgramSkipSelector::ConfigureResult::Malformed:
                std::fprintf(stderr,
                             "[draw-skip] ignoring malformed PROSPER_SKIP_DRAW_PROGRAM=\"%s\" "
                             "(expected 0x-prefixed hex program addresses, comma separated) "
                             "-- NOT armed, every draw runs\n", spec);
                break;
            case DrawProgramSkipSelector::ConfigureResult::Armed:
                std::fprintf(stderr,
                             "[draw-skip] PROSPER_SKIP_DRAW_PROGRAM=%s -> %zu program(s) will be "
                             "declined; every draw using one is withheld from the GPU\n",
                             spec, s.size());
                break;
        }
        return true;
    }();
    return selector;
}

DrawProgramCensus& draw_program_census() {
    static DrawProgramCensus census;
    return census;
}

bool draw_program_census_enabled() {
    static const bool enabled = [] {
        const char* spec = std::getenv("PROSPER_DRAW_PROGRAM_CENSUS");
        const bool on = spec && *spec && std::strcmp(spec, "0") != 0;
        if (on)
            std::fprintf(stderr,
                         "[draw-program] PROSPER_DRAW_PROGRAM_CENSUS armed -- one line per distinct "
                         "(vs, vs-chain, ps) program triple, then powers of two\n");
        return on;
    }();
    return enabled;
}

}  // namespace prosper::gpu
