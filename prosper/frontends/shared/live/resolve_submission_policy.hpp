#pragma once

namespace prosper::frontend {

// A resolve copy may join the renderer's ordered submission only when ordinary render groups use
// that same batch. Otherwise a later synchronous render could overtake the deferred copy and change
// its source before the copy reaches the queue.
inline constexpr bool resolve_copy_may_batch(bool backend_submits_batched,
                                             bool force_synchronous_copy) {
    return backend_submits_batched && !force_synchronous_copy;
}

// Ordinary terminal render groups flush from inside the backend. A terminal resolve returns before
// that call, so it owns the equivalent boundary before the presentation/readback tail may inspect
// speculative destination state.
inline constexpr bool terminal_resolve_must_flush(bool backend_submits_batched,
                                                  bool final_group,
                                                  bool batch_pending) {
    return backend_submits_batched && final_group && batch_pending;
}

} // namespace prosper::frontend
