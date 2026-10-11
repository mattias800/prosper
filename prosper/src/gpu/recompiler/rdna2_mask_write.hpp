#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
// A 64-bit mask write architecturally overwrites the destination SGPR pair ("D = ..." in every
// ISA mask-op description): any tracked data-domain SSA value or descriptor-provenance tag for
// those registers is stale afterwards. Erase both so a later data read / SRSRC resolution cannot
// alias the pre-mask-op value (the mask itself lives in sreg_bool / rs.vcc / rs.exec).
inline void mask_write_clobbers_pair(RegState& rs, int dst) {
    rs.sreg.erase(dst);
    rs.sreg.erase(dst + 1);
    rs.sreg_srt.erase(dst);
    rs.sreg_srt.erase(dst + 1);
}

inline void saveexec_write_clobbers_pair(RegState& rs, int dst) {
    mask_write_clobbers_pair(rs, dst);
    // CONFIDENCE: HIGH — every B64 SAVEEXEC unconditionally writes D=OLD_EXEC to both
    // physical destination words. An earlier opaque M0 value no longer occupies either word.
    // Ordinary/conditional mask writers keep their existing, separate lifetime contracts.
    rs.sreg_entry_m0.erase(dst);
    rs.sreg_entry_m0.erase(dst + 1);
}
} // namespace prosper::gpu
