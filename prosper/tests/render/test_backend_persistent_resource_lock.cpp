// The backend's persistent-resource domain lock (#2953).
//
// THE DEFECT. `render_draw_pass_rgba` owns a set of process-lifetime `static` containers -- the
// persistent pipeline, pipeline-layout, colour-target, depth/stencil and texture caches, their byte
// totals and their generation counters -- and reached every one of them with no synchronisation.
// #2953 recorded a host SIGSEGV on a guest job thread inside `persistent_texture_images.find()`,
// in libstdc++'s `_M_equals` reached through `_M_find_before_node` with a null node: the shape a
// `find()` takes when it races another thread's rehash.
//
// WHAT THESE ARMS PROVE, AND IN WHICH ORDER.
//
//   Arm 1 is the POSITIVE CONTROL, and it runs first on purpose. It drives the same threads through
//   the same shape with NO guard and asserts the detector fires. Without it, arms 2 and 3 are
//   unfalsifiable: a clean zero would be equally well explained by "the guard works" and by "these
//   threads never actually overlapped", and the second explanation is the one a passing test would
//   quietly adopt. It is built by hand out of atomics rather than by removing the guard from the
//   production path, so it observes no undefined behaviour of its own.
//
//   Arm 2 asserts the guard's contract directly: N threads, a shared `std::unordered_map` mutated
//   under the guard, no observed overlap, and the map intact afterwards.
//
//   Arm 3 is the one that binds the PRODUCTION path. N threads call `render_triangle_rgba`
//   concurrently, each with a persistent-texture resource so the exact `find()` from the crash
//   report is on every thread's route. It asserts (a) the domain really was contended --
//   `peak_in_flight >= 2`, so the arm cannot pass vacuously -- and (b) zero overlaps, which is only
//   possible if `render_draw_pass_rgba` takes the guard. Remove the guard from that function and
//   this arm reddens.
//
// HONESTY ABOUT DETERMINISM. Arms 1 and 3 depend on threads genuinely running at the same time, so
// they are probabilistic in principle. Two things make them reliable in practice, and one makes
// them safe when they are not. `in_flight` is incremented BEFORE the mutex acquire, so a thread
// blocked on the guard still counts -- with a rendezvous before the call and a critical section
// that records and submits a command buffer, every waiting thread is observed. And the
// `peak_in_flight >= 2` assertion is what turns "the threads did not overlap" from a silent pass
// into a visible failure: this test cannot report success on a run that never exercised the case.
// What would make it fully deterministic is a scheduling hook inside the critical section, which
// would mean shipping a test-only seam through the live render path; the in-flight assertion buys
// the same protection without one.

#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

namespace {

constexpr int kThreads = 4;
constexpr int kIterations = 64;

// Release every thread at once, so the window in which they can overlap is as wide as the work
// itself rather than as narrow as the thread-creation loop.
class Rendezvous {
public:
    explicit Rendezvous(int expected) : expected_(expected) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (++arrived_ >= expected_) { open_ = true; cv_.notify_all(); return; }
        cv_.wait(lock, [this] { return open_; });
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int expected_;
    int arrived_ = 0;
    bool open_ = false;
};

// The same depth/overlap detector the production guard uses, over storage this test owns. Arm 1
// runs it WITHOUT a lock to show it can report an overlap at all.
struct OverlapDetector {
    std::atomic<int> depth{0};
    std::atomic<int> peak{0};
    std::atomic<uint64_t> overlaps{0};

    void enter() {
        const int now = depth.fetch_add(1, std::memory_order_acq_rel) + 1;
        int seen = peak.load(std::memory_order_relaxed);
        while (seen < now && !peak.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {}
        if (now > 1) overlaps.fetch_add(1, std::memory_order_relaxed);
    }
    void leave() { depth.fetch_sub(1, std::memory_order_acq_rel); }
};

}  // namespace

int main() {
    // ---- 1. POSITIVE CONTROL: the detector reports an overlap when nothing excludes ------------
    // Built by hand out of atomics, outside the production guard, so this cannot inherit whatever
    // the guard does. A zero here would mean the following two arms are measuring nothing.
    {
        OverlapDetector unguarded;
        Rendezvous gate(kThreads);
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&] {
                gate.arrive_and_wait();
                for (int i = 0; i < kIterations; ++i) {
                    unguarded.enter();
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                    unguarded.leave();
                }
            });
        for (auto& thread : threads) thread.join();
        check(unguarded.peak.load() >= 2,
              "control: unguarded threads are inside the section together");
        check(unguarded.overlaps.load() > 0,
              "control: the overlap detector reports a non-zero count when nothing excludes");
    }

    // ---- 2. The guard excludes, and a shared map survives the same contention -------------------
    // The map operation is the one from the crash report: repeated find() against another thread's
    // insertions, which is what rehashes under a lock-free reader.
    {
        const uint64_t overlaps_before = prosper::test::backend_persistent_resource_overlaps().load();
        prosper::test::backend_persistent_resource_peak_in_flight().store(0);

        std::unordered_map<uint64_t, uint64_t> shared;
        std::atomic<uint64_t> found{0};
        Rendezvous gate(kThreads);
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] {
                gate.arrive_and_wait();
                for (int i = 0; i < kIterations; ++i) {
                    const prosper::test::BackendPersistentResourceGuard guard;
                    const uint64_t key = static_cast<uint64_t>(t) * kIterations + i;
                    shared.emplace(key, key * 3 + 1);
                    for (int probe = 0; probe <= t; ++probe) {
                        const auto entry = shared.find(static_cast<uint64_t>(probe) * kIterations + i);
                        if (entry != shared.end() && entry->second == entry->first * 3 + 1) ++found;
                    }
                }
            });
        for (auto& thread : threads) thread.join();

        check(prosper::test::backend_persistent_resource_peak_in_flight().load() >= 2,
              "guarded threads really contended for the domain (the arm is not vacuous)");
        check(prosper::test::backend_persistent_resource_overlaps().load() == overlaps_before,
              "no two threads were ever inside the guarded section at once");
        check(shared.size() == static_cast<size_t>(kThreads) * kIterations,
              "every insertion survived the contention -- the shared map is intact");
        check(found.load() > 0, "the guarded find() returned live entries (the probes ran)");
        check(prosper::test::backend_persistent_resource_depth().load() == 0 &&
                  prosper::test::backend_persistent_resource_in_flight().load() == 0,
              "the guard's counters return to zero when every thread has left");
    }

    // ---- 3. THE PRODUCTION ARM: render_draw_pass_rgba takes the guard ---------------------------
    // Each thread renders a triangle that samples a PERSISTENT texture, so
    // `persistent_texture_images.find()` -- the exact call in the #2953 backtrace -- is on every
    // thread's route. Distinct persistent ids per thread keep the workload realistic (four
    // identities competing for one cache) rather than four lookups of one key.
    if (!prosper::test::render_vk_ctx().ok) {
        std::fprintf(stderr,
                     "note: no usable Vulkan device -- skipping the concurrent render arm; arms 1 "
                     "and 2 above still ran\n");
    } else {
        const uint64_t overlaps_before = prosper::test::backend_persistent_resource_overlaps().load();
        prosper::test::backend_persistent_resource_peak_in_flight().store(0);

        const std::vector<uint32_t> vert(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv) / 4);
        const std::vector<uint32_t> frag(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv) / 4);
        constexpr uint32_t W = 32, H = 32;
        constexpr uint32_t kTexW = 8, kTexH = 8;
        constexpr int kRenderIterations = 6;

        std::vector<std::vector<uint8_t>> images(kThreads);
        std::vector<uint8_t> texels(static_cast<size_t>(kTexW) * kTexH * 4, 0x40);
        Rendezvous gate(kThreads);
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] {
                gate.arrive_and_wait();
                for (int i = 0; i < kRenderIterations; ++i) {
                    std::vector<prosper::test::FrameResource> resources;
                    for (uint32_t set = 0; set < 2; ++set) {
                        prosper::test::FrameResource cbuf;
                        cbuf.binding = 2; cbuf.set = set;
                        resources.push_back(std::move(cbuf));
                        prosper::test::FrameResource vbuf;
                        vbuf.binding = 3; vbuf.set = set;
                        resources.push_back(std::move(vbuf));
                    }
                    prosper::test::FrameResource tex;
                    tex.binding = 0;
                    tex.set = 1;
                    tex.tex_rgba = texels.data();
                    tex.tex_byte_size = texels.size();
                    tex.tw = kTexW;
                    tex.th = kTexH;
                    // Non-zero: this is what routes the upload through the persistent texture cache
                    // and therefore through the find() that crashed.
                    tex.persistent_texture_id = 0x2953'0000ull + static_cast<uint64_t>(t);
                    resources.push_back(std::move(tex));
                    images[t] = prosper::test::render_triangle_rgba(
                        vert, frag, W, H, nullptr, nullptr, nullptr, nullptr, &resources);
                }
            });
        for (auto& thread : threads) thread.join();

        const int peak = prosper::test::backend_persistent_resource_peak_in_flight().load();
        std::fprintf(stderr, "note: peak threads wanting the domain during the render arm: %d\n", peak);
        check(peak >= 2,
              "concurrent render_draws_rgba calls really contended for the domain "
              "(the arm is not vacuous)");
        check(prosper::test::backend_persistent_resource_overlaps().load() == overlaps_before,
              "render_draw_pass_rgba never ran two threads through its persistent caches at once");
        bool every_frame_rendered = true;
        for (int t = 0; t < kThreads; ++t)
            if (images[t].size() != static_cast<size_t>(W) * H * 4) every_frame_rendered = false;
        check(every_frame_rendered, "every concurrent render returned a complete frame");
        check(prosper::test::backend_persistent_resource_depth().load() == 0 &&
                  prosper::test::backend_persistent_resource_in_flight().load() == 0,
              "the render path leaves the guard's counters at zero");
    }

    if (failures) { std::fprintf(stderr, "== FAIL: %d ==\n", failures); return 1; }
    std::fprintf(stderr, "== PASS ==\n");
    return 0;
}
