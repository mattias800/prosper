// Current portable symbolic halves refine dispatcher marks without publishing scalar DATA.
#pragma once

#include "gpu/recompiler/rdna2_spill_slot_domain.hpp"
#include <cstdint>

namespace prosper::gpu {
struct RegState;
struct SpirvCompute;
struct SpilledMaskHalves;
struct SpilledVectorSourceMarks;
void seed_portable_mask_dispatch(const SpirvCompute&, const RegState&, uint32_t, SpilledMaskHalves&,
                                 SpillSlotDomains&);
void refine_portable_mask_dispatch(const SpirvCompute&, const Rdna2Inst&, bool,
                                   const SpilledVectorSourceMarks&, SpilledMaskHalves&,
                                   SpillSlotDomains&);
} // namespace prosper::gpu
