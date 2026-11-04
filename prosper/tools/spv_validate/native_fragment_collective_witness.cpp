// The chosen execution modes distinguish the isolated native retry from legacy success.
#include "native_fragment_collective_witness.hpp"
#include "fixtures/native_fragment_readfirst.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include <cstdint>
#include <vector>

namespace prosper::spv_validation {
std::vector<uint32_t> native_fragment_collective_witness(bool wave32, bool whole_quad) {
    namespace fixture = prosper::test::native_fragment_readfirst;
    using namespace prosper::gpu;
    auto result = prosper::test::native_pixel_parameter::compile(
        fixture::program(true, false, 2.5f, true, whole_quad, wave32), fixture::mapping(), wave32);
    if (fragment_spirv_reconvergence(result.module) != FragmentReconvergence::Both ||
        !arbitrary_placeholder_reads(result.module).total())
        return {};
    return result.module;
}
} // namespace prosper::spv_validation
