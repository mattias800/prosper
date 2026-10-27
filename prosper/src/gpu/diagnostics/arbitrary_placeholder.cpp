// Adapt the compiler-owned observation port to the always-on performance observer.
#include <cstdint>
#include "gpu/recompiler/rdna2_arbitrary_placeholder_observation.hpp"
#include "diagnostics/perf/arbitrary_placeholder.hpp"

namespace prosper::gpu {
void observe_arbitrary_placeholder_check() {
    diagnostics::perf::note_arbitrary_placeholder_check();
}

void observe_arbitrary_placeholder_admission(uint64_t program, uint64_t fresh, uint64_t loop) {
    diagnostics::perf::note_arbitrary_placeholder_admission(program, fresh, loop);
}
} // namespace prosper::gpu
