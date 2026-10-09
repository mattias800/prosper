#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;

// Records the architectural DATA value of S_BFM_B64 (SOP2 opcode 0x25) for an
// SGPR destination pair, alongside the lane-mask view the emitter keeps in
// sreg_bool. A BFM-defined pair is exactly the shape a scalar-data consumer
// (e.g. s_cselect_b64's pair projection) needs; without it the pair reads as
// unwritten and the consumer refuses. EXEC/VCC destinations keep their
// mask-only form: those registers have no DATA slots in the model. Returns
// whether DATA was recorded; when the sources are not defined DATA (or are
// unresolvable) it records nothing and the mask path proceeds unchanged, so a
// fabricated word can never become a tracked zero here.
bool record_bfm_b64_data(SpirvCompute& b, RegState& rs, const Rdna2Inst& in);

}  // namespace prosper::gpu
