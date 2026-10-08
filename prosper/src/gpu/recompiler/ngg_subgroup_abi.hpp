// ngg_subgroup_abi.hpp -- static admission of a LINKED merged ES+GS NGG program for the subgroup
// shell (#3135 P2, design section 4). Pure analysis over decoded instructions: no SPIR-V, no Vulkan.
//
// The shell supplies exactly the launch state the hardware contract establishes (launch research on
// #3135): s3 (merged wave info, without the GS wave id in [23:16]), user SGPRs s8.. from push
// constants, s0:s1 only when the caller knows the user-data address, VGPRs v0..v3/v5/v8, and an
// EXEC the program must write before it reads it. Everything else is refused here, by name:
//
//   ngg-abi-read-s0-s1                 s0:s1 read before written without a known user-data address
//   ngg-abi-read-s2                    s2 (NGG group info; ISA and compilers disagree on it)
//   ngg-abi-read-s3-gs-wave-id         a launch s3 read whose demanded bits include [23:16]
//   ngg-abi-read-s4-s5                 off-chip LDS base / scratch offset
//   ngg-abi-read-s6-s7                 the GS program address (only the link may use it)
//   ngg-abi-read-undefined-sgpr        any other SGPR (s8+user_sgprs.. included) read before written
//   ngg-abi-read-undefined-vcc / -m0 / -scc / ngg-abi-read-ttmp
//   ngg-abi-unclassified-vector-width  a vector source whose register count is not classified
//   ngg-abi-exec-read-before-write     EXEC (explicitly, or implicitly by a vector instruction)
//   ngg-abi-read-v4                    adjacency offsets 4/5 read before written on the lane, for
//                                      an ADJACENCY input topology only (see adjacency_input)
//   ngg-abi-read-v6-v7                 ES user VGPRs read before written on the lane
//   ngg-side-effect                    memory stores/atomics, GDS/GWS/ordered count, scratch
//   ngg-sendmsg-missing                no GS_ALLOC_REQ at all
//   ngg-sendmsg-unsupported            any s_sendmsg/s_sendmsghalt other than GS_ALLOC_REQ (msg 9)
//   ngg-sendmsg-m0-unproven            GS_ALLOC_REQ whose M0 is not written on every path by a
//                                      scalar ALU instruction
//   ngg-export-in-cycle / ngg-export-compressed / ngg-export-target-unsupported /
//   ngg-export-pos1-channels (POS1 other than the layer, z) /
//   ngg-export-duplicate-target / ngg-export-prim-channels / ngg-export-missing-prim /
//   ngg-export-missing-pos0 / ngg-export-too-many-params
//   ngg-abi-control-transfer           s_setpc/s_swappc/s_call/M0-relative moves/subvector loops
//   ngg-abi-cfg-unresolved             a branch target that is not a decoded instruction
//   ngg-abi-undecoded                  an instruction the decoder could not identify
//
// "Read before written" is a MUST analysis over the CFG: a register counts as written only when
// every path from entry writes it. A VGPR counts as written for a lane only through a full-dword
// write while that lane is in EXEC: either under a provably full EXEC, under an EXEC that has only
// narrowed since the write, or under an EXEC restored (s_mov_b64 exec, sP) from a pair that saved
// the EXEC of the write (s_mov_b64 sP, exec or a SAVEEXEC) and was not overwritten since (#3135 P6:
// the compiler's if/else idiom, which Kena's indexed producer uses for v7). The read inventory is fail-closed where it is unsure: an unknown scalar
// source width reads two registers, DS data fields are always read, a MIMG address is charged
// through v7, and a vector source whose width is not classified refuses the program. Not
// inventoried: cross-lane reads (v_readlane, ds_bpermute, permlane) of lanes outside the EXEC a
// definition covered. CONFIDENCE: MED on the completeness of the 64-bit opcode lists.
#pragma once

#include "gpu/recompiler/ngg_export_record.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

struct NggSubgroupAbiLaunch {
    uint32_t user_sgprs = 0;   // s8 .. s8+user_sgprs-1 hold user data
    bool user_data_address_known = false;   // s0:s1 hold SPI_SHADER_USER_DATA_ADDR_LO/HI_GS
    // The draw's input primitives carry adjacency, so launch v4 holds the offsets of vertices 4/5
    // (#3135 launch research: v0 = offsets 0/1, v1 = 2/3, v2 = PrimitiveID, v3 = GS invocation,
    // v4 = 4/5; radeonsi, radv and LLPC agree). Without adjacency there are no vertices 4/5: v4
    // carries no input, the hardware leaves whatever an earlier wave held there, and a correct
    // program cannot depend on it -- so the shell's 0 is one of the values the hardware may present
    // and a read of it is admitted. Kena's compiler treats v4 accordingly: in every refused program
    // its first reference is a write, as a temporary (#4746). ngg_draw_admission admits triangle
    // lists and strips only, so no live draw sets this. CONFIDENCE: MED (the hardware's v4 for a
    // non-adjacency input is inferred from the layout, not observed).
    bool adjacency_input = false;
};

struct NggSubgroupAbiFacts {
    std::string refusal;   // empty when admitted; "reason=<name> pc=<pc> ..."
    std::string reason;   // the bare reason name
    NggExportRecordLayout layout;   // valid only when admitted
    std::vector<uint32_t> alloc_request_pcs;   // every s_sendmsg GS_ALLOC_REQ
    bool ok() const { return refusal.empty(); }
};

// `ins` is the complete decoded linked program (rdna2_walk of the linked words).
NggSubgroupAbiFacts analyze_ngg_subgroup_abi(const std::vector<Rdna2Inst>& ins,
                                             const NggSubgroupAbiLaunch& launch);

}   // namespace prosper::gpu
