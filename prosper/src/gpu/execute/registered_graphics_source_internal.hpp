#pragma once
#include "gpu/execute/gpu_execute.hpp"

namespace prosper::gpu {
// Native analysis is derived cold from the exact immutable original decoder version. This
// sealed association is not a second address lookup or a same-content assertion at handoff.
// It retains only a WEAK original owner, so a resident native payload cannot keep that owner
// alive merely by retaining its compile-analysis words.
class RegisteredNativeGraphicsAnalysis {
    friend GraphicsReadSource
    coupled_graphics_read_source(const std::shared_ptr<const DecodedShader>&);
    std::weak_ptr<const DecodedShader> original_;
    const DecodedShader* original_address_ = nullptr;
    SharedShaderAnalysis analysis_;
    RegisteredNativeGraphicsAnalysis(const std::shared_ptr<const DecodedShader>& original,
                                     SharedShaderAnalysis analysis)
        : original_(original), original_address_(original.get()), analysis_(std::move(analysis)) {}

public:
    bool belongs_to(const std::shared_ptr<const DecodedShader>& original) const {
        return original && original.get() == original_address_ &&
               !original_.owner_before(original) && !original.owner_before(original_);
    }
    const SharedShaderAnalysis& analysis() const { return analysis_; }
};

// Input comes from the actual normal stage producer's byte-validated registered observation.
// The helper itself grants no read permission, stage eligibility, resource or draw admission.
GraphicsReadSource coupled_graphics_read_source(const std::shared_ptr<const DecodedShader>&);
}   // namespace prosper::gpu
