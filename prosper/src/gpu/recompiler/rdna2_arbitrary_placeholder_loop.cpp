// Arbitrary data cannot supply an unsafe memory read or an unmodelled scalar publication.
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
        const bool address = b.loop_blanket_roots_strong.contains(blanket_root(header, word));
        const bool scalar = !weak_allowed;
        if (!address && !scalar) continue;
        const bool vector = word >= kVectorBlanketBase;
        const int reg = vector ? (word - kVectorBlanketBase) % kArbitraryOutsideRootOffset : word;
        const PlaceholderWord source{vector ? PlaceholderFile::Vector : PlaceholderFile::Scalar,
                                     reg};
        if (address) roots.insert({PlaceholderOrigin::UnprovedStrongRead, source, header});
        if (scalar) roots.insert({PlaceholderOrigin::UnmodelledScalarPublication, source, header});
    }
}
} // namespace prosper::gpu
