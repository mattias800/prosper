// Arbitrary data does not discharge an unbounded/writing-address or scalar-read obligation.
#include <cstdint>
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <vector>
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
void append_unproved_loop_reads(PlaceholderRoots& roots, const SpirvCompute& b, uint32_t header,
                                const std::vector<int>& words, bool weak_allowed) {
    for (int word : words) {
        if (weak_allowed && !b.loop_blanket_roots_strong.contains(blanket_root(header, word)))
            continue;
        const bool vector = word >= kVectorBlanketBase;
        const int reg = vector ? (word - kVectorBlanketBase) % kArbitraryOutsideRootOffset : word;
        roots.insert({PlaceholderOrigin::UnprovedStrongRead,
                      {vector ? PlaceholderFile::Vector : PlaceholderFile::Scalar, reg},
                      header});
    }
}
} // namespace prosper::gpu
