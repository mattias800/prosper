// PROSPER_TEXREF_CENSUS=1 — counter-only census of the live renderer's per-texture-reference
// resolution chain (#3873 plan item 2).
//
// Two questions, both of which decide whether a per-reference memo could pay off and what it would
// have to be invalidated by:
//   * REPETITION: how often does the exact decoded T#/S# identity (plus the reflected binding
//     contract) recur within one submit, and across adjacent submits -- and when it recurs, does the
//     resolved FrameResource come out identical? A repeat whose outcome DIFFERS inside one submit is
//     direct evidence that an earlier draw changed state the chain depends on (an RTT publication, a
//     compute producer, a guest write), so a memo keyed on the identity alone would serve stale data.
//   * WHERE THE TIME GOES: TSC cycles between fixed stage marks inside the chain, split by the
//     outcome class the path decided. rdtsc, not clock_gettime: PROSPER_RENDER_TIMING's own clocks
//     cost ~9% of the hot thread, so a census built on them would mostly measure itself.
//
// Off by default; every hook is a single predictable branch on a function-local static.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <x86intrin.h>

namespace prosper::frontend {

struct TextureReferenceCensus {
    // Stage boundaries, in path order. A reference that skips a stage (e.g. the MSAA branch) leaves
    // the mark unset and its time is charged to the next mark that IS set.
    enum Stage : int {
        kSetupKey = 0,     // FrameResource + AvPlayer classification + decode key
        kRttProbe,         // live RTT lookup, lazy readback, direct/uniform serve gates
        kDepthProbe,       // sampled depth bridge, Float32 arrays, retained depth cubes
        kSourceSize,       // format predicates, source span, DCC metadata read
        kComputeProbe,     // compute-image import
        kPreLookup,        // write-watch admission constants
        kLookup,           // submit-local reuse + persistent cache lookup/validation
        kFill,             // FrameResource fill / decode on miss
        kStages
    };
    // The three "other" classes are references decoded from guest bytes without the persistent
    // decode cache: other_rttauth -- a live RTT entry kept renderer authority but could not serve
    // the reference; other_nocand -- the shape/format is not a cache candidate at all (e.g. a
    // non-BC cube); other -- a candidate the runtime gates refused (compression, budget, storage).
    enum Class : int {
        kRtt = 0, kCompute, kPersistSubmit, kPersistHit, kPersistMiss, kPersistInvalid, kLocal,
        kOtherRttAuthority, kOtherNotCandidate, kOther, kClasses
    };
    static const char* class_name(int c) {
        static const char* names[kClasses] = {"rtt", "compute", "persist_submit", "persist_hit",
                                              "persist_miss", "persist_invalid", "local",
                                              "other_rttauth", "other_nocand", "other"};
        return names[c];
    }

    static bool enabled() {
        static const bool on = std::getenv("PROSPER_TEXREF_CENSUS") != nullptr;
        return on;
    }

    uint64_t stamps[kStages + 1] = {};
    uint32_t set_mask = 0;
    bool active = false;

    void begin() {
        active = true;
        set_mask = 1u;
        stamps[0] = __rdtsc();
    }
    void mark(int stage_end) {
        if (!active) return;
        stamps[stage_end + 1] = __rdtsc();
        set_mask |= 1u << (stage_end + 1);
    }

    struct PerClass {
        uint64_t refs = 0;
        uint64_t cycles[kStages] = {};
        uint64_t repeats_in_submit = 0;
        uint64_t repeats_changed = 0;   // same key earlier in this submit, different outcome
        uint64_t seen_prev_submit = 0;
        uint64_t prev_submit_changed = 0;
    };
    PerClass cls[kClasses];
    // Named sub-steps inside a stage, timed by the caller with aux_begin()/aux_end(). Counted
    // across all classes; they explain a stage, they are not a partition of it.
    // kAuxCubeRead / kAuxCubeFill split the retained depth-cube CPU bridge: the six-face readback
    // and the CPU quantise into the stacked RGBA8 payload. kAuxCubeGpu is the GPU route's
    // recording of the same gather (tests/fixtures/retained_depth_cube_gpu.h).
    enum Aux : int {
        kAuxDsFlush = 0, kAuxDsIdentity, kAuxDsRead, kAuxDsScan, kAuxCubeRead, kAuxCubeFill,
        kAuxCubeGpu, kAux
    };
    uint64_t aux_cycles[kAux] = {};
    uint64_t aux_count[kAux] = {};
    // Retained depth-cube references served by the persistent decode cache without a bridge.
    uint64_t cube_cache_hits = 0;
    // GPU-route cube snapshots served again inside their owning batch.
    uint64_t cube_gpu_reuses = 0;
    static uint64_t aux_begin() { return enabled() ? __rdtsc() : 0; }
    void aux_end(int which, uint64_t start) {
        if (!start) return;
        aux_cycles[which] += __rdtsc() - start;
        ++aux_count[which];
    }
    uint64_t submits = 0;
    uint64_t distinct_keys = 0;
    int submit = -1;
    std::unordered_map<uint64_t, uint64_t> this_submit;   // key -> outcome
    std::unordered_map<uint64_t, uint64_t> prev_submit;
    std::chrono::steady_clock::time_point window_start{};
    uint64_t window_tsc = 0;

    void finish(int submit_ordinal, uint64_t key, uint64_t outcome, int klass) {
        if (!active) return;
        active = false;
        const uint64_t end = __rdtsc();
        stamps[kStages] = end;
        set_mask |= 1u << kStages;
        if (window_tsc == 0) {
            window_tsc = stamps[0];
            window_start = std::chrono::steady_clock::now();
        }
        if (submit_ordinal != submit) {
            if (submit >= 0) {
                ++submits;
                distinct_keys += this_submit.size();
                prev_submit.swap(this_submit);
            }
            this_submit.clear();
            submit = submit_ordinal;
        }
        PerClass& pc = cls[klass];
        ++pc.refs;
        // Charge each stage the time since the previous SET mark, attributing skipped stages' time
        // (zero-width) to the next set boundary.
        uint64_t previous = stamps[0];
        for (int s = 0; s < kStages; ++s) {
            if (!(set_mask & (1u << (s + 1)))) continue;
            const uint64_t now = stamps[s + 1];
            if (now >= previous) pc.cycles[s] += now - previous;
            previous = now;
        }
        auto [it, inserted] = this_submit.try_emplace(key, outcome);
        if (!inserted) {
            ++pc.repeats_in_submit;
            if (it->second != outcome) {
                ++pc.repeats_changed;
                it->second = outcome;
            }
        }
        if (const auto prev = prev_submit.find(key); prev != prev_submit.end()) {
            ++pc.seen_prev_submit;
            if (prev->second != outcome) ++pc.prev_submit_changed;
        }
        maybe_report(end);
    }

    void maybe_report(uint64_t now_tsc) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - window_start).count();
        if (seconds < 5.0) return;
        const double tsc_per_us = static_cast<double>(now_tsc - window_tsc) / (seconds * 1e6);
        uint64_t total_refs = 0;
        double total_us = 0;
        for (const auto& pc : cls) {
            total_refs += pc.refs;
            for (uint64_t c : pc.cycles) total_us += c / tsc_per_us;
        }
        std::fprintf(stderr,
                     "[texref-census] window=%.2fs submits=%llu refs=%llu distinct/submit=%.1f "
                     "chain=%.1fms (%.2fus/ref) tsc=%.0fMHz\n",
                     seconds, (unsigned long long)submits, (unsigned long long)total_refs,
                     submits ? (double)distinct_keys / submits : 0.0, total_us / 1000.0,
                     total_refs ? total_us / total_refs : 0.0, tsc_per_us);
        static const char* stage_names[kStages] = {"setup", "rtt", "depth", "source", "compute",
                                                   "prelookup", "lookup", "fill"};
        for (int c = 0; c < kClasses; ++c) {
            const PerClass& pc = cls[c];
            if (!pc.refs) continue;
            double class_us = 0;
            for (uint64_t cyc : pc.cycles) class_us += cyc / tsc_per_us;
            std::fprintf(stderr,
                         "[texref-census]   %-15s refs=%llu %.2fus/ref repeat_in_submit=%llu "
                         "(changed=%llu) seen_prev_submit=%llu (changed=%llu) stages_us/ref:",
                         class_name(c), (unsigned long long)pc.refs, class_us / pc.refs,
                         (unsigned long long)pc.repeats_in_submit,
                         (unsigned long long)pc.repeats_changed,
                         (unsigned long long)pc.seen_prev_submit,
                         (unsigned long long)pc.prev_submit_changed);
            for (int s = 0; s < kStages; ++s)
                std::fprintf(stderr, " %s=%.2f", stage_names[s],
                             pc.cycles[s] / tsc_per_us / pc.refs);
            std::fprintf(stderr, "\n");
        }
        static const char* aux_names[kAux] = {"ds_flush", "ds_identity", "ds_read", "ds_scan",
                                             "cube_read", "cube_fill", "cube_gpu"};
        bool any_aux = false;
        for (int a = 0; a < kAux; ++a) any_aux |= aux_count[a] != 0;
        any_aux |= cube_cache_hits != 0 || cube_gpu_reuses != 0;
        if (any_aux) {
            std::fprintf(stderr, "[texref-census]   aux:");
            for (int a = 0; a < kAux; ++a)
                std::fprintf(stderr, " %s=%llu/%.1fms", aux_names[a],
                             (unsigned long long)aux_count[a], aux_cycles[a] / tsc_per_us / 1000.0);
            std::fprintf(stderr, " cube_cache_hits=%llu cube_gpu_reuses=%llu\n",
                         (unsigned long long)cube_cache_hits, (unsigned long long)cube_gpu_reuses);
        }
        cube_cache_hits = 0;
        cube_gpu_reuses = 0;
        for (auto& c : aux_cycles) c = 0;
        for (auto& c : aux_count) c = 0;
        for (auto& pc : cls) pc = PerClass{};
        submits = 0;
        distinct_keys = 0;
        window_start = now;
        window_tsc = now_tsc;
    }
};

inline TextureReferenceCensus& texture_reference_census() {
    static thread_local TextureReferenceCensus census;
    return census;
}

}  // namespace prosper::frontend
