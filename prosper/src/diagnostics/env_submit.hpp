// env_submit.hpp — diagnostic switches re-read once per SUBMIT, for per-draw paths.
//
// WHY THIS EXISTS, and why it is not `env_cache.hpp`.
//
// `PROSPER_ENV_ON` samples getenv once for the life of the process. That is correct for a
// boot-time switch and wrong for any variable a test arms at runtime: the read is frozen before
// the arm, and such a test does not fail, it goes VACUOUS. `tools/env/check_cached_env.py` refuses
// those names for exactly that reason, which leaves them as live `getenv` calls -- and several of
// them sit on per-DRAW paths. Measured on a routed Grand Theft Auto V window after #3705 cached
// everything that could be cached: 58,304,770 remaining `getenv` calls, of which `PROSPER_GFXLOG`
// is 13.4 M and `PROSPER_RENDER_TIMING` 10.0 M, both read per draw per stage. `getenv` itself is
// 1.92% of the render thread in the retained native profile, and a name that is ABSENT -- the
// normal state of a diagnostic switch -- costs a full scan of the environment, so the misses are
// the expensive ones.
//
// THE CONTRACT, established from the arming sites rather than assumed. Every test that arms one of
// these names does so BETWEEN operations and then calls the production path:
//
//     set_env("PROSPER_RENDER_TIMING", "1");
//     render_draws_rgba(...);                       // tests/gpu/test_texture_sample_render.cpp:712
//     const auto timing = backend_render_timing_stats();
//
// So the weakest requirement that keeps those tests non-vacuous is: **a change must be visible at
// the next submit, and need not be visible part-way through one**. That is what this samples. It
// is deliberately NOT a claim that mid-submit visibility is unnecessary in general.
//
// THE FALLBACK IS THE OLD BEHAVIOUR, and that is the point. Outside every `SubmitEnvScope` this
// macro is a plain live `getenv` -- so it narrows to the scoped region and changes nothing
// anywhere else. That matters because `build_stage_table` and `realize_draw_item` are also reached
// from `realize_retained_draw`, `diagnose_resource_provenance` and several tests that call them
// directly. Without the fallback each of those would sample once and hold that value until some
// unrelated thread happened to open a submit -- process-lifetime behaviour on exactly the paths
// nobody was thinking about, which is #2214 again with extra steps. With it, adding a scope is an
// opt-in optimisation and forgetting one costs speed, never correctness.
//
// WHY THE GENERATION IS GLOBAL AND THE CACHE IS PER THREAD. `realize_gpustate_draws_parallel`
// realizes a submit's draws on worker threads. A thread_local generation would be bumped only on
// the submitting thread, so every worker would sample once and never again -- process-lifetime
// behaviour for exactly the threads that do the per-draw work. The generation is therefore one
// atomic counter; each call site keeps its own thread_local (generation, value) pair, so a worker
// re-reads the first time it observes a new submit and not afterwards. Two submits genuinely in
// flight at once can therefore make a worker re-sample mid-submit: that is more eager than the
// contract, never less, so it cannot hide an arm.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace prosper::diag {

namespace detail {
// Monotone window id. Bumped on every scope entry, so entering invalidates every site's cache.
inline std::atomic<uint64_t>& submit_env_generation() {
    static std::atomic<uint64_t> generation{1};
    return generation;
}
// How many scopes are open anywhere. Zero means "no submit in progress" -> read live.
inline std::atomic<int>& submit_env_depth() {
    static std::atomic<int> depth{0};
    return depth;
}
}  // namespace detail

// The current sampling window, or 0 when no submit is in progress.
inline uint64_t submit_env_window() {
    if (detail::submit_env_depth().load(std::memory_order_relaxed) <= 0) return 0;
    return detail::submit_env_generation().load(std::memory_order_relaxed);
}

// RAII at the top of a submit. Nesting is harmless: an inner scope starts another sampling window,
// which is still no weaker than the per-submit contract above.
struct SubmitEnvScope {
    SubmitEnvScope() {
        detail::submit_env_generation().fetch_add(1, std::memory_order_relaxed);
        detail::submit_env_depth().fetch_add(1, std::memory_order_relaxed);
    }
    ~SubmitEnvScope() { detail::submit_env_depth().fetch_sub(1, std::memory_order_relaxed); }
    SubmitEnvScope(const SubmitEnvScope&) = delete;
    SubmitEnvScope& operator=(const SubmitEnvScope&) = delete;
};

}  // namespace prosper::diag

// Presence check re-sampled once per submit per thread, and read live outside any submit. THE UNIT
// IS THE CALL SITE, as it is for PROSPER_ENV_ON: each textual expansion owns its own thread_local
// pair, so two sites reading the same name sample independently and neither can freeze the other.
#define PROSPER_ENV_ON_PER_SUBMIT(name) ([]() -> bool {                                     \
        static thread_local uint64_t prosper_env_submit_seen = 0;                           \
        static thread_local bool prosper_env_submit_value = false;                          \
        const uint64_t prosper_env_submit_now = prosper::diag::submit_env_window();          \
        if (prosper_env_submit_now == 0) return std::getenv(name) != nullptr;               \
        if (prosper_env_submit_seen != prosper_env_submit_now) {                            \
            prosper_env_submit_seen = prosper_env_submit_now;                               \
            prosper_env_submit_value = std::getenv(name) != nullptr;                        \
        }                                                                                   \
        return prosper_env_submit_value; }())
