// A compute full-overwrite destination at the persistent colour-target cache's COUNT bound (#3873).
//
// ensure_persistent_color_target_for_compute_overwrite used to refuse outright whenever the cache
// was at its entry bound, because only the ordered render call knew whether a graphics batch could
// still reference an unpinned target. On GTA V the cache sits at its bound through gameplay, so one
// refusal latched a 14 MiB compute result onto the CPU snapshot path for a whole run. It now evicts
// under the graphics admission predicate, with the pending-batch half read from the process-wide
// count BackendSubmissionBatch keeps. These arms pin both halves of that predicate:
//   * at the bound with nothing pending, creation succeeds by evicting the LEAST recently used
//     unpinned target, and the cache stays at its bound;
//   * while any graphics batch holds unsubmitted commands, creation is refused and nothing is
//     evicted (the reason the old code refused at all);
//   * a pinned target is never the victim, and with every older target pinned, creation is refused;
//   * the pending count follows enqueue/discard, so a discarded batch releases the gate.
// The count bound is 4 (PROSPER_BACKEND_TARGET_CACHE_COUNT, set by the ctest entry).
#include "fixtures/render_runner.h"

#include <cstdio>
#include <cstdlib>

using namespace prosper::test;

namespace {
int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++failures; } \
                        else std::printf("[ok] %s\n", m); } while (0)
constexpr uint32_t W = 16, H = 16;
constexpr VkFormat F = VK_FORMAT_R8G8B8A8_UNORM;

bool present(uint64_t id) {
    const PersistentColorTargetImage* t = find_persistent_color_target(id, W, H, F, false);
    return t && t->image;
}
// A just-created compute destination is not yet VALID (nothing has written it), so the public
// pin helper, which requires a valid target, refuses it; pin through the cache entry itself.
bool pin(uint64_t id, int delta) {
    PersistentColorTargetImage* t = find_persistent_color_target(id, W, H, F, false);
    if (!t) return false;
    t->pin_count += delta;
    return true;
}
bool create(uint64_t id) {
    return ensure_persistent_color_target_for_compute_overwrite(id, W, H, F) != nullptr;
}
}  // namespace

int main() {
    if (!render_vk_ctx().ok) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }
    if (persistent_color_target_count_limit() != 4) {
        std::printf("[FAIL] PROSPER_BACKEND_TARGET_CACHE_COUNT=4 was not applied (limit=%zu)\n",
                    persistent_color_target_count_limit());
        return 1;
    }
    for (uint64_t id = 1; id <= 4; ++id) create(id);
    CHECK(present(1) && present(2) && present(3) && present(4) &&
              persistent_color_target_cache().size() == 4,
          "four targets fill the cache to its count bound");

    // Nothing pending: the fifth evicts the least recently used (1), not the newest.
    CHECK(create(5), "at the bound with no batch pending, a compute destination is created");
    CHECK(!present(1) && present(2) && present(5) && persistent_color_target_cache().size() == 4,
          "...by evicting the least recently used target; the cache stays at its bound");

    // A graphics batch with unsubmitted commands: refuse, evict nothing.
    {
        BackendSubmissionBatch batch;
        batch.enqueue(VK_NULL_HANDLE);   // recorded but never submitted: discarded below
        CHECK(backend_pending_submission_batches().load() == 1,
              "an enqueued batch counts as pending process-wide");
        CHECK(!create(6), "while a graphics batch is pending, creation at the bound is refused");
        CHECK(present(2) && present(3) && present(4) && present(5) &&
                  persistent_color_target_cache().size() == 4,
              "...and nothing was evicted, nor was a half-created entry left behind");
        batch.discard();
        CHECK(backend_pending_submission_batches().load() == 0,
              "discarding the batch releases the pending count");
    }
    CHECK(create(6), "with the batch gone, the same creation succeeds");
    CHECK(!present(2) && present(6), "...evicting the next least recently used target (2)");

    // Pins: the oldest (3) is pinned, so the victim is the next one (4).
    CHECK(pin(3, +1), "pin the oldest target");
    CHECK(create(7) && present(3) && !present(4),
          "a pinned target is skipped; the next least recently used one is evicted");
    // Everything older than the newest is pinned: no victim, so creation is refused.
    CHECK(pin(5, +1) && pin(6, +1),
          "pin two more");
    CHECK(!create(8) && persistent_color_target_cache().size() == 4 && present(7),
          "with every evictable target pinned or newest, creation is refused (never over the bound)");
    pin(3, -1);
    pin(5, -1);
    pin(6, -1);

    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
