// Current portable symbolic halves refine dispatcher marks without publishing scalar DATA.
#pragma once

#include "gpu/recompiler/rdna2_spill_slot_domain.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

namespace prosper::gpu {
struct SpirvCompute;
struct SpilledMaskHalves;
struct SpilledVectorSourceMarks;
// A dispatcher ACTIVE entry can omit guest lanes even with a complete physical workgroup.
// Preserve the producer's stronger refusal context across nested body emission.
class PortableMaskHalfEntryScope {
public:
    PortableMaskHalfEntryScope(SpirvCompute&, bool);
    ~PortableMaskHalfEntryScope();

private:
    SpirvCompute& b_;
    bool previous_;
};
void refine_portable_mask_dispatch(const SpirvCompute&, const Rdna2Inst&, bool,
                                   const SpilledVectorSourceMarks&, SpilledMaskHalves&,
                                   SpillSlotDomains&);
} // namespace prosper::gpu
