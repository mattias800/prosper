#include "gpu/execute/native_graphics_source_lineage.hpp"
#include "gpu/execute/registered_graphics_source_internal.hpp"

namespace prosper::gpu {
bool NativeGraphicsStageCompilation::matches(const GraphicsReadSource& source,
                                             const SharedShaderWords& module) const {
    return source.words && source.decoded && source.native_analysis && module && !module->empty() &&
           source_.words.get() == source.words.get() && !source_.words.owner_before(source.words) &&
           !source.words.owner_before(source_.words) &&
           source_.decoded.get() == source.decoded.get() &&
           !source_.decoded.owner_before(source.decoded) &&
           !source.decoded.owner_before(source_.decoded) &&
           source_.native_analysis == source.native_analysis &&
           source_.registered_header == source.registered_header &&
           source_.header_snapshot == source.header_snapshot &&
           source.native_analysis->belongs_to(source.decoded) && module_.get() == module.get() &&
           !module_.owner_before(module) && !module.owner_before(module_);
}
} // namespace prosper::gpu
