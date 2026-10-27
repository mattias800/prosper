// Exercise the delegating entry point itself, including its chosen admission metadata.
#include <cstdint>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <vector>
#include "arbitrary_placeholder_witness.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"

namespace prosper::spv_validation {
std::vector<uint32_t> arbitrary_placeholder_witness() {
    namespace fixture = prosper::test::arbitrary_placeholder;
    using namespace prosper::gpu;
    const auto words = fixture::scalar_relay();
    const auto resources = fixture::table();
    const auto config = fixture::config();
    const RecompileDiagnosticContext context{RecompileDiagnosticStage::Compute, 0};
    auto module = recompile_with_arbitrary_placeholders(context, [&] {
        return recompile_compute(words.data(), words.size(), &resources, config, context);
    });
    if (!arbitrary_placeholder_reads(module).total()) return {};
    return module;
}
} // namespace prosper::spv_validation
