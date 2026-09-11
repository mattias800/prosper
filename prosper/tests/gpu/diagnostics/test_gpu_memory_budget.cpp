// gpu_memory_budget — "how much device memory does prosper itself hold, per heap?"
//
// The instrument exists because nothing in prosper answered that (#3533), and what is pinned here is
// the one design decision the answer rests on: **the counter is keyed by the VkDeviceMemory handle,
// not by a size the free site passes back.** The free sites in live_compute.cpp hold only a handle —
// they do not know the allocation's size or its heap — so a size-carrying free would be six places
// to keep in agreement, and a wrong one there does not fail loudly: it silently attributes bytes to
// the wrong heap, or drifts the count until the line reads as a leak that is not there.
//
// Every case below is constructed by hand rather than driven through the allocator, per the
// charter's positive-control rule: the property under test is exactly the bookkeeping the allocator
// would be trusted to get right.
#include "gpu/diagnostics/gpu_memory_budget.hpp"

#include "fixtures/test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static constexpr uint64_t kMiB = 1024ull * 1024ull;

// Read back whatever the instrument printed. The counters being right is only half of it: this
// file's first version asserted eighteen things about the counters, all true, while the growth line
// -- the instrument's entire reason to exist -- printed NOTHING for the life of the process. The
// threshold compare-exchange compared against a substituted value rather than the stored one, so it
// failed every time, and no assertion about a counter can see that.
static std::string captured_output(const char* path) {
    std::fflush(stderr);
    FILE* file = std::fopen(path, "rb");
    if (!file) return std::string();
    std::string text;
    char buffer[4096];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) text.append(buffer, got);
    std::fclose(file);
    return text;
}

int main() {
    // A 1 MiB step so the ordinary test sizes below cross it, and stderr into a file so the crossings
    // can be asserted rather than assumed. Both must happen before any other call: the step is read
    // from the environment once, lazily, on the first allocation.
    // A per-process scratch path, never a fixed name in the CWD. ctest runs cases under -j, several
    // worktrees on this box run ctest at once, and a shared path would make two processes fight over
    // one file -- whose symptom is a CONTENT assertion failing on correct code, the failure most
    // likely to send a reader hunting a defect that does not exist. test_scratch.h exists for
    // exactly this and its header records the four issues it came from; an earlier version of this
    // test walked straight back into it.
    const std::string capture_file = prosper_test::test_scratch_file("gpu_memory_budget_stderr.txt");
    const char* const capture_path = capture_file.c_str();
#if defined(_WIN32)
    _putenv_s("PROSPER_GPU_MEM_LOG_MIB", "1");
#else
    setenv("PROSPER_GPU_MEM_LOG_MIB", "1", 1);
#endif
    const bool captured = std::freopen(capture_path, "w+", stderr) != nullptr;

    // The layout every case below is read against. Types 0-3 map to heaps 0-3 so a case can name the
    // heap it means; types 4-6 alias heaps 0-2, because a real device has several types per heap and
    // an implementation that confused the two indices would pass a 1:1 table; and type 7 deliberately
    // names a heap BEYOND heap_count, which is the one malformed layout a driver could hand over.
    const uint32_t type_heap[8] = {0, 1, 2, 3, 0, 1, 2, 9};
    const uint64_t heap_sizes[4] = {64 * kMiB, 32 * kMiB, 4096 * kMiB, 64 * kMiB};
    set_device_heaps(type_heap, 8, heap_sizes, 4);

    // A second, different layout must not renumber the heaps under a count that is already running.
    const uint32_t other_heap[2] = {1, 0};
    const uint64_t other_sizes[2] = {1 * kMiB, 1 * kMiB};
    set_device_heaps(other_heap, 2, other_sizes, 2);

    // Heap 0: a free must give back exactly what ITS OWN allocation took.
    //
    // THREE DIFFERENT SIZES, and every free is checked, because a smaller sequence does not
    // discriminate. An earlier version freed the 5 MiB allocation first and asserted the remainder:
    // a deliberately broken implementation subtracting a CONSTANT 5 MiB passed it unchanged, and
    // the clamp at zero absorbed the drift by the end of the test, so the arm reported "no failure"
    // while the lever had certainly moved. The sizes below (5, 7, then 2 on a reused handle) leave
    // no constant that satisfies both checked frees: 12-C == 5 demands C == 7, and 7-C == 5 demands
    // C == 2.
    note_device_alloc(0x1000, 0, 5 * kMiB);
    note_device_alloc(0x2000, 0, 7 * kMiB);
    CHECK(device_bytes_held(0) == 12 * kMiB, "two live allocations sum");
    note_device_free(0x2000);
    CHECK(device_bytes_held(0) == 5 * kMiB,
          "a free subtracts its OWN size, not the other live record's");

    // The peak survives the free. Held says what is outstanding now; peak says what was demanded,
    // which is the only one of the two that can explain a failure after the fact.
    CHECK(device_peak_bytes(0) == 12 * kMiB, "peak is a high-water mark, not the live total");

    // A free of a handle that was never counted changes nothing. The driver is free to hand back a
    // handle value prosper has already returned, and live_compute routes some frees for allocations
    // this counter never saw; neither may drive the count negative or wrap to 16 exabytes.
    note_device_free(0x2000);                 // already freed
    note_device_free(0xdeadbeef);             // never allocated
    note_device_free(0);                      // VK_NULL_HANDLE
    CHECK(device_bytes_held(0) == 5 * kMiB, "an unmatched free is a no-op, not a wrap");

    // A handle REUSED after its free counts once, at its NEW size. Without the erase in
    // note_device_free the reuse reads as a double count; with an insert that declines to overwrite
    // an existing key, the free below would give back the stale 7 MiB instead of 2 MiB -- the
    // leak-that-is-not-there, or its mirror image, a count that drains to zero while memory is held.
    note_device_alloc(0x2000, 0, 2 * kMiB);
    CHECK(device_bytes_held(0) == 7 * kMiB, "a reused handle is counted again");
    note_device_free(0x2000);
    CHECK(device_bytes_held(0) == 5 * kMiB, "the reused handle's free subtracts its NEW size");
    note_device_free(0x1000);
    CHECK(device_bytes_held(0) == 0, "every live allocation released returns the heap to zero");
    CHECK(device_peak_bytes(0) == 12 * kMiB, "peak is unchanged by releasing everything");

    // Heap 3: the same handle counted TWICE with no free between, which is the only path that
    // reaches the map's overwrite. It is not hypothetical -- it is what the counter sees when the
    // driver reuses a handle whose free did not route through this counter. The live record must
    // describe the CURRENT allocation, so the free gives back 4 MiB and the 3 MiB whose free was
    // never seen stays on the books. That residue is the honest reading rather than a defect: those
    // bytes really were allocated and this counter really never saw them released.
    //
    // An insert that declined to overwrite (emplace) would keep the stale 3 MiB record and give
    // that back instead, leaving 4 MiB. Without this case that mutation passes the whole file.
    note_device_alloc(0x9000, 3, 3 * kMiB);
    note_device_alloc(0x9000, 3, 4 * kMiB);
    CHECK(device_bytes_held(3) == 7 * kMiB, "a handle re-counted without a free between adds both");
    note_device_free(0x9000);
    CHECK(device_bytes_held(3) == 3 * kMiB,
          "its free gives back the CURRENT record, not the stale one");

    // Heaps are independent: a Vulkan device has several, and prosper's DEVICE_LOCAL pressure is not
    // its host-visible staging. Attributing one to the other is the failure mode that made the
    // instrument worth writing.
    note_device_alloc(0x3000, 1, 4 * kMiB);
    CHECK(device_bytes_held(1) == 4 * kMiB && device_bytes_held(0) == 0, "heaps count separately");

    // A type index the layout does not describe, a type whose heap index the layout does not
    // describe, and two degenerate inputs. All are DROPPED rather than folded into heap 0: counting
    // an unresolvable allocation somewhere would print a confident wrong attribution, which is the
    // failure this whole instrument exists to stop repeating.
    note_device_alloc(0x4000, 99, 8 * kMiB);   // type beyond memoryTypeCount
    note_device_alloc(0x4100, 7, 8 * kMiB);    // type 7 names heap 9, beyond memoryHeapCount
    note_device_alloc(0x5000, 1, 0);           // zero-byte allocation
    note_device_alloc(0, 1, 8 * kMiB);         // VK_NULL_HANDLE
    CHECK(device_bytes_held(1) == 4 * kMiB, "unresolvable and degenerate allocations are ignored");
    CHECK(device_bytes_held(0) == 0, "an unresolvable allocation does not spill into heap 0");
    note_device_free(0x4000);
    note_device_free(0x4100);
    note_device_free(0x5000);
    CHECK(device_bytes_held(1) == 4 * kMiB, "freeing what was never counted is still a no-op");

    // Two memory types on ONE heap accumulate on that heap. A 1:1 type-to-heap assumption reads
    // type 4 as heap 4 and this stays at zero.
    note_device_alloc(0x6000, 4, 3 * kMiB);    // type 4 -> heap 0
    CHECK(device_bytes_held(0) == 3 * kMiB, "a second memory type on heap 0 lands on heap 0");
    note_device_free(0x6000);
    CHECK(device_bytes_held(0) == 0, "and its free comes off the same heap");

    // These run on prosper's real allocation path, which is multi-threaded. Concurrent alloc/free
    // pairs on one heap must land back exactly where they started -- a lost update here is a count
    // that drifts over a long run, which is indistinguishable from the leak the line would report.
    {
        constexpr int kThreads = 8, kPerThread = 500;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; t++) {
            threads.emplace_back([t] {
                for (int i = 0; i < kPerThread; i++) {
                    const uint64_t handle = 0x100000ull + (uint64_t)t * kPerThread + (uint64_t)i;
                    note_device_alloc(handle, 2, kMiB);
                    note_device_free(handle);
                }
            });
        }
        for (auto& th : threads) th.join();
        CHECK(device_bytes_held(2) == 0, "concurrent alloc/free pairs balance exactly");
        CHECK(device_peak_bytes(2) >= kMiB, "the concurrent run registered a peak at all");
    }

    // The reporting path must survive a null occasion string and must not disturb what it reports
    // on -- a reporter that mutated the counters would make every number printed after the first one
    // wrong. An earlier version of this case asserted `CHECK(true, ...)` beside these calls, which
    // is no assertion at all, and it was hiding an out-of-bounds read in the TEST: it declared 99
    // heaps over a 2-entry array and printed 134212650 MiB of stack residue for heaps 4-15.
    report_device_memory("test");
    report_device_memory(nullptr);
    CHECK(device_bytes_held(1) == 4 * kMiB && device_bytes_held(2) == 0,
          "reporting does not disturb the counters it reports");
    CHECK(device_peak_bytes(0) == 12 * kMiB, "reporting does not disturb the peak either");

    // The second layout above was refused, so heap 1 must still be the 32 MiB heap it was declared
    // as -- not the 1 MiB one the second call offered. Without that refusal every figure printed
    // after a second device appeared would be measured against the wrong denominator.
    CHECK(device_bytes_held(1) == 4 * kMiB && device_peak_bytes(1) == 4 * kMiB,
          "a second device's layout did not renumber the heaps");

    // Pressure labels. heap 3 is 64 MiB and holds 3 MiB of deliberate residue from the overwrite
    // case above, so 42 MiB more puts it at 70%; heap 1 is 32 MiB holding 4 MiB, so 26 MiB more puts
    // it past 90%. Both thresholds are asserted on the EMITTED text further down rather than on a
    // return value, because the label exists only to be read in a log.
    note_device_alloc(0xa000, 3, 42 * kMiB);
    note_device_alloc(0xb000, 1, 26 * kMiB);

    // The step is a RATCHET: once a heap has printed at 13 MiB it stays quiet below 13 MiB forever,
    // even after falling back to zero and climbing again. That is deliberate -- it bounds the output
    // of a workload that cycles -- but it means the line reports a new HIGH-WATER MARK, not current
    // growth, and a reader watching for "prosper is allocating again" will not see it. Pinned here
    // so the behaviour is a decision rather than an accident, and stated in the header.
    note_device_free(0xb000);
    note_device_alloc(0xb100, 1, 26 * kMiB);   // same size again, below the mark: must stay silent

    // A report of the LOADED state. The report above ran before these allocations, so it could not
    // carry a pressure label; without this one, deleting the label from report_device_memory passes
    // every assertion in the file because only the growth line exercises it.
    report_device_memory("loaded");

    // Everything above is about counters. THIS is about output, and it is the arm that catches a
    // dead instrument whose arithmetic is perfect.
    if (!captured) {
        printf("  [FAIL] could not capture stderr, so the emission arms did not run\n");
        fails++;
    } else {
        const std::string output = captured_output(capture_path);
        CHECK(output.find("[gpu-mem] prosper holds") != std::string::npos,
              "a heap crossing its step actually PRINTS a growth line");
        CHECK(output.find("on heap 1 of 32 MiB") != std::string::npos,
              "the growth line names the heap and its real size");
        CHECK(output.find("[gpu-mem] test: heap 0 is 64 MiB") != std::string::npos,
              "report_device_memory prints the per-heap standing");
        CHECK(output.find("not other processes, the compositor, or driver overhead") !=
                  std::string::npos,
              "every report carries the caveat about what it does NOT count");
        CHECK(output.find("(over 70% of the heap)") != std::string::npos,
              "a heap past 70% says so on the line");
        // ...and on the REPORT line specifically, which is a different call site. Deleting the
        // label from report_device_memory used to pass every assertion in this file, because the
        // growth line alone carried it.
        CHECK(output.find("loaded: heap 3 is 64 MiB; prosper holds 45 MiB (peak 45 MiB)  "
                          "(over 70% of the heap)") != std::string::npos,
              "report_device_memory carries the pressure label too, not just the growth line");
        CHECK(output.find("*** OVER 90% OF THE HEAP ***") != std::string::npos,
              "a heap past 90% is shouted rather than mentioned");
        // The ratchet: exactly one line for heap 1 at 30 MiB, not a second one when the same 26 MiB
        // is freed and re-allocated. Counting occurrences is the only way to see a duplicate.
        size_t at = 0, repeats = 0;
        const std::string mark = "holds 30 MiB on heap 1";
        while ((at = output.find(mark, at)) != std::string::npos) { repeats++; at += mark.size(); }
        CHECK(repeats == 1, "re-reaching a mark already printed does not print it again");
        std::remove(capture_path);
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
