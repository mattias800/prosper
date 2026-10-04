#pragma once

#include "host/memory/guest_memory_topology.hpp"
#include <cstddef>
#include <cstdint>

namespace prosper {

// Test-only observation after an exclusive mapping attempt has actually found a held reader.
// Null in production; the callback must not call HLE mapping functions.
void set_guest_mapping_mutation_contention_observer_for_test(void (*observer)());

// Copy host bytes into a fully committed direct-memory mapping through prosper's authoritative
// physical backing. This models a device write without weakening the guest VA's CPU protection.
// Private, untracked, malformed, and cross-mapping destinations fail closed.
bool guest_memory_gpu_write_supported(uint64_t destination, size_t bytes);
bool guest_memory_gpu_write(uint64_t destination, const void* source, size_t bytes);

// Linux direct-memory copy through an emulator-private writable shared alias. The callback is
// called only after the destination's topology is proved and its watched guest aliases are marked
// GPU-dirty; the guest VA protection stays armed. False means no complete copy was published and
// the caller must take its ordinary writeback path. The callback must only copy the supplied
// disjoint host-staging bytes; it runs under the topology lock and must not enter mapping/HLE APIs.
// Other hosts conservatively decline.
using GuestMemoryByteCopy = void (*)(void* destination, const void* source, size_t bytes);
bool guest_memory_gpu_write_alias(uint64_t destination, const void* source, size_t bytes,
                                  GuestMemoryByteCopy copy);

// Unit-test witness that the protected/backing-aware device-write path actually ran. Output bytes
// alone cannot prove that lever moved: a host memmove can coincidentally choose the correct direction
// for one pair of physically aliased but VA-disjoint views.
uint64_t guest_memory_gpu_write_successes_for_test();
uint64_t guest_memory_gpu_alias_write_successes_for_test();

// #2384: does an AMPR constructor's `a2` say the command buffer keeps a byte cursor? Exposed for
// test because the answer is a PREDICATE over guest-supplied bit patterns, and the only way to show
// the widening is strictly additive is to assert it over the shapes the working titles pass -- which
// a boot test cannot do without those titles' dumps. The implementation lives in hle_kernel_mem.cpp.
bool ampr_cb_tracks_offset_arg_for_test(uint64_t a2);

} // namespace prosper
