// Classified draw disposition: why a draw the guest issued did not reach the GPU.
//
// The defect this guards is a silence. Six distinct drop paths folded into one `skipped_draw()`
// counter, two of them printing nothing at all and one printing once per process, so "the world is
// black" and "146 draws wanted wave64 on a 32-wide device" were the same observation. This suite
// pins the properties that make the replacement trustworthy rather than merely present:
//
//   * a healthy pass is SILENT -- otherwise the instrument is noise and gets turned off;
//   * a drop names its MECHANISM, because that is what makes the fix general instead of
//     title-shaped;
//   * the census DETECTS ITS OWN INVALIDITY. `seen` and `recorded` are counted by routes that
//     share no counter, so a drop path added without a reason shows up as UNACCOUNTED instead of
//     silently rebalancing a tidy total. Without this arm a future edit could add a seventh drop
//     path and the report would keep printing plausible numbers.

#include "gpu/diagnostics/draw_disposition.hpp"
#include "diagnostics/exit_reports.hpp"
#include "diagnostics/perf/perf_ledger.hpp"

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

using namespace prosper::gpu;

static int failures = 0;
static void check(bool ok, const char* name) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

// Capture actual report output, including the registered exit flush below.
template<typename Report>
static std::string capture_report(Report report) {
    fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    FILE* tmp = tmpfile();
    if (!tmp) { printf("FAIL: tmpfile()\n"); failures++; return {}; }
    dup2(fileno(tmp), STDERR_FILENO);
    report();
    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    rewind(tmp);
    std::string out;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), tmp)) > 0) out.append(buf, n);
    fclose(tmp);
    return out;
}

static std::string capture_pass() {
    return capture_report([] { draw_disposition_census().report_pass(); });
}

int main() {
    auto& c = draw_disposition_census();
    bool empty_reported = true;
    const std::string empty = capture_report([&] { empty_reported = c.report_totals(); });
    check(!empty_reported && empty.empty(), "an unused exit snapshot remains silent");

    // --- names -------------------------------------------------------------------------------
    bool names_ok = true;
    for (int i = 0; i < static_cast<int>(DrawDrop::Count); i++) {
        const char* n = draw_drop_name(static_cast<DrawDrop>(i));
        if (!n || !*n || std::strcmp(n, "unknown") == 0) names_ok = false;
    }
    check(names_ok, "every DrawDrop has a stable non-placeholder name");

    namespace perf = prosper::diagnostics::perf;
    const auto unaccounted = [] {
        return perf::ledger().counters[static_cast<size_t>(perf::Counter::DrawsUnaccounted)].load();
    };
    const uint64_t balanced_before = unaccounted();
    // --- a healthy pass is silent ---------------------------------------------------------
    c.note_seen(4);
    c.note_recorded(4);
    check(capture_pass().empty(), "a pass whose every seen draw was recorded prints nothing");

    // --- a drop names its mechanism -------------------------------------------------------
    c.note_seen(3);
    c.note_recorded(2);
    c.note_dropped(DrawDrop::SubgroupFeatures);
    {
        const std::string out = capture_pass();
        check(out.find("subgroup-features=1") != std::string::npos,
              "a dropped draw is reported under its mechanism name");
        check(out.find("ord=1") != std::string::npos,
              "the reason carries its 1-based ordinal (diag_ratelimit contract)");
        check(out.find("UNACCOUNTED") == std::string::npos,
              "a fully accounted pass does not claim an accounting gap");
        check(out.find("BLACK-PASS") == std::string::npos,
              "a pass that recorded draws is not flagged black");
    }

    // --- the self-validation arm ----------------------------------------------------------
    // Simulates exactly the future edit this instrument must survive: a draw leaves the pass
    // without being recorded AND without naming a reason. The census must say so.
    // #3891: the two passes above add up, so they left the perf ledger's counter alone.
    check(unaccounted() == balanced_before, "passes that add up leave DrawsUnaccounted alone");
    const uint64_t unaccounted_before = unaccounted();
    c.note_seen(10);
    c.note_recorded(7);
    c.note_dropped(DrawDrop::ShaderRejected);
    c.note_dropped(DrawDrop::BufferResources);
    {
        const std::string out = capture_pass();
        check(out.find("UNACCOUNTED=1") != std::string::npos,
              "an unnamed drop path is reported as UNACCOUNTED, not absorbed into the total");
    }
    // #3891 unaccounted-draws: the same blind spot reaches the perf-alarm ledger.
    check(unaccounted() - unaccounted_before == 1,
          "the unaccounted draw is added to the perf ledger's DrawsUnaccounted counter");

    // --- double counting must be VISIBLE, not plausible ------------------------------------
    // The defect this guards shipped once: a seventh census site was added that was not disjoint
    // from the six, so every setup-loop drop was counted twice -- once under its real reason and
    // once as pipeline-creation. It survived review of two live titles because both happened to
    // drop zero draws, so the census reported clean totals on the only runs anyone looked at.
    // Here the arithmetic is forced: 4 seen, 2 recorded, 3 dropped. A census that merely summed
    // would print a tidy-looking total; this one must report the overcount as NEGATIVE.
    c.note_seen(4);
    c.note_recorded(2);
    c.note_dropped(DrawDrop::SubgroupFeatures);
    c.note_dropped(DrawDrop::ShaderRejected);
    c.note_dropped(DrawDrop::PipelineCreation);
    {
        const std::string out = capture_pass();
        check(out.find("UNACCOUNTED=-1") != std::string::npos,
              "counting one drop under two reasons reports a negative unaccounted, not a total");
    }
    check(unaccounted() - unaccounted_before == 2,
          "an overcount (negative unaccounted) reaches the ledger as its magnitude");

    // --- a black pass is always reported --------------------------------------------------
    c.note_seen(5);
    for (int i = 0; i < 5; i++) c.note_dropped(DrawDrop::ShaderRejected);
    {
        const std::string out = capture_pass();
        check(out.find("BLACK-PASS") != std::string::npos,
              "a pass that saw draws and recorded none is flagged BLACK-PASS");
        check(out.find("shader-rejected=5") != std::string::npos,
              "the black pass names what it dropped");
    }

    // --- totals are process-lifetime ------------------------------------------------------
    check(c.seen() == 26 && c.recorded() == 15,
          "process-lifetime seen/recorded totals survive per-pass resets");
    // Split so a failure localizes: 1 (pass 3) + 5 (pass 4) shader-rejected, and
    // 1 subgroup + 1 shader + 1 buffer + 5 shader = 8 drops overall.
    check(c.dropped(DrawDrop::ShaderRejected) == 7,
          "per-reason drop totals accumulate across passes");
    check(c.dropped(DrawDrop::SubgroupFeatures) == 2 &&
          c.dropped(DrawDrop::BufferResources) == 1,
          "drops are attributed to the reason that was named, not pooled");
    check(c.dropped_total() == 11, "aggregate drop total is the sum of the per-reason totals");

    // Balance is not proof of quiescence, even before a worker starts preparing the next pass.
    const std::string balanced = capture_report([&] { c.report_totals(); });
    check(balanced.find("RUN SNAPSHOT quiescence=unverified") != std::string::npos,
          "a balanced aggregate does not claim a quiescent final total");
    check(balanced.find("snapshot-delta=") == std::string::npos,
          "balanced independent loads do not invent a snapshot delta");

    // A real counter producer reaches a pass, then waits unjoined across the registered flush.
    // The rendezvous establishes this state without hoping to catch a timing-dependent race.
    std::mutex mutex;
    std::condition_variable changed;
    bool ready = false, release = false, worker_finished = false;
    std::thread worker([&] {
        c.note_seen();
        std::unique_lock lock(mutex);
        ready = true;
        changed.notify_one();
        changed.wait(lock, [&] { return release; });
        lock.unlock();
        c.note_recorded();
        lock.lock();
        worker_finished = true;
    });
    {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return ready; });
        check(!worker_finished && c.pass_seen_for_scope() == 1,
              "counter-producing worker remains live with an in-flight pass before flush");
    }
    const uint64_t exit_before = unaccounted();
    const std::string live = capture_report([] { prosper::diagnostics::flush_exit_reports(); });
    check(live.find("RUN SNAPSHOT quiescence=unverified") != std::string::npos &&
          live.find("snapshot-delta=1") != std::string::npos,
          "registered flush labels the live exit difference as an unverified snapshot");
    check(live.find("UNACCOUNTED") == std::string::npos && unaccounted() == exit_before,
          "an in-flight exit difference is not a completed-pass error or alarm");
    {
        std::scoped_lock lock(mutex);
        check(!worker_finished, "worker has not finished at the registered flush boundary");
        release = true;
    }
    changed.notify_one();
    worker.join();
    check(capture_pass().empty() && unaccounted() == exit_before,
          "joined healthy pass retains quiet completed-pass accounting");
    const std::string joined = capture_report([&] { c.report_totals(); });
    check(joined.find("snapshot-delta=") == std::string::npos &&
          joined.find("quiescence=unverified") != std::string::npos,
          "balanced post-join loads still require an explicit quiescence proof");

    printf("%s\n", failures ? "FAILURES" : "ALL PASS");
    return failures ? 1 : 0;
}
