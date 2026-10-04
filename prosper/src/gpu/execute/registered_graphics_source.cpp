#include "gpu/execute/registered_graphics_source_internal.hpp"
#include "gpu/execute/shader_cache_internal.hpp"

#include <map>

namespace prosper::gpu {
GraphicsReadSource
coupled_graphics_read_source(const std::shared_ptr<const DecodedShader>& decoded) {
    if (!decoded || decoded->code.empty()) return {};
    using Source = std::weak_ptr<const DecodedShader>;
    static thread_local std::map<Source, std::shared_ptr<const RegisteredNativeGraphicsAnalysis>,
                                 std::owner_less<Source>>
        native;
    const Source identity = decoded;
    static const bool bypass = PROSPER_ENV_ON("PROSPER_NO_SHADER_ANALYSIS_CACHE");
    std::shared_ptr<const RegisteredNativeGraphicsAnalysis> analysis;
    if (!bypass)
        if (const auto found = native.find(identity); found != native.end())
            analysis = found->second;
    if (!analysis) {
        // No arbitrary shader-count warm eviction. Retire genuinely expired source versions on
        // cold insertion; values retain weak decoder owners and immutable native code only.
        for (auto item = native.begin(); item != native.end();)
            if (item->first.expired())
                item = native.erase(item);
            else
                ++item;
        auto derived = acquire_shader_analysis(decoded->code.data(), decoded->code.size());
        analysis = std::shared_ptr<const RegisteredNativeGraphicsAnalysis>(
            new RegisteredNativeGraphicsAnalysis(decoded, std::move(derived)));
        if (!bypass) native.emplace(identity, analysis);
    }
    GraphicsReadSource result;
    result.words = SharedShaderWords(decoded, &decoded->code);
    result.chains = std::shared_ptr<const std::vector<RawNestedWideChain>>(
        decoded, &decoded->owned_nested_wide_chains);
    result.packet_requirements = std::shared_ptr<const FragmentPacketVgprRequirements>(
        decoded, &decoded->packet_requirements);
    result.vertex_effects =
        std::shared_ptr<const OriginalGraphicsStageEffects>(decoded, &decoded->original_effects[0]);
    result.fragment_effects =
        std::shared_ptr<const OriginalGraphicsStageEffects>(decoded, &decoded->original_effects[1]);
    result.decoded = decoded;
    result.native_analysis = std::move(analysis);
    return result;
}
}   // namespace prosper::gpu
