// Checked text relocation; these small terminal/constant helpers retain the original statements.
#include "gpu/recompiler/ir/lower/buffer_access_routes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
namespace prosper::gpu::ir {
bool BufferAccessRoutes::finish() {
    if (indirect_pointer_descriptor_source) {
        const auto root_lo = rs.vreg.find(in.dst.value);
        const auto word1 = rs.vreg.find(in.dst.value + 1);
        if (root_lo == rs.vreg.end() || word1 == rs.vreg.end()) {
            ok = false;
            return true;
        }
        b.capture_indirect_pointer_descriptor_source(val(in.src[0]), root_lo->second, word1->second,
                                                     rs.exec_narrowed, rs.exec);
    }
    return true;
}
float BufferAccessRoutes::normalization_range() const {
    float norm = 0.0f;
    switch (fmt) {
        case DataFormat::Unorm8: norm = 255.0f; break;
        case DataFormat::Snorm8: norm = 127.0f; break;
        case DataFormat::Unorm16: norm = 65535.0f; break;
        case DataFormat::Snorm16: norm = 32767.0f; break;
        default: break;
    }
    return norm;
}
} // namespace prosper::gpu::ir
