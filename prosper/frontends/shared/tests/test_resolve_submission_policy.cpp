#include "shared/live/resolve_submission_policy.hpp"

#include <cstdio>

using prosper::frontend::resolve_copy_may_batch;
using prosper::frontend::terminal_resolve_must_flush;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    CHECK(resolve_copy_may_batch(true, false));

    // PROSPER_NO_BACKEND_BATCH_SUBMITS and PROSPER_NO_BATCHED_RESOLVE_COPY independently preserve
    // the synchronous copy. Deferring a copy while later render groups submit directly would change
    // render(S=v1) -> copy(S,D) -> render(S=v2) into render(v1) -> render(v2) -> copy(v2,D).
    CHECK(!resolve_copy_may_batch(false, false));
    CHECK(!resolve_copy_may_batch(true, true));
    CHECK(!resolve_copy_may_batch(false, true));

    // A final resolve has no later ordinary render call to flush the batch. The presentation tail
    // may run only after this boundary; nonterminal resolves stay queued with later render work.
    CHECK(terminal_resolve_must_flush(true, true, true));
    CHECK(!terminal_resolve_must_flush(true, false, true));
    CHECK(!terminal_resolve_must_flush(true, true, false));
    CHECK(!terminal_resolve_must_flush(false, true, true));

    if (!failures) std::printf("resolve_submission_policy: OK\n");
    return failures ? 1 : 0;
}
