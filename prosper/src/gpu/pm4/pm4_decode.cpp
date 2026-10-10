// pm4_decode.cpp — see pm4_decode.hpp. Pure PM4 type-3 stream walker.
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <set>
#include "gpu/pm4/pm4_decode.hpp"

namespace prosper::gpu {

namespace {
// A valid type-3 PM4 header has 0b11 in bits 30..31 (0xC0000000). Anything else (e.g. a zeroed pad
// dword, or a type-0/2 packet we don't emit) ends the walk.
inline bool is_type3(uint32_t h) { return (h & 0xC0000000u) == 0xC0000000u; }

// Fields carried by hle_agc.cpp's PM4() encoding.
inline uint32_t hdr_len(uint32_t h) { return ((h >> 16) & 0x3fffu) + 2u; }  // dwords incl. header
inline uint32_t hdr_op (uint32_t h) { return (h >> 8) & 0xffu; }
inline uint32_t hdr_r  (uint32_t h) { return (h >> 2) & 0x3fu; }

uint64_t lo_hi(const uint32_t* p) { return (uint64_t)p[0] | ((uint64_t)p[1] << 32); }
}  // namespace

size_t decode_pm4(const uint32_t* buf, size_t dwords, std::vector<Pm4Command>& out) {
    size_t i = 0;
    while (i < dwords) {
        uint32_t h = buf[i];
        if (!is_type3(h)) {
            // PM4 TYPE-2 (bits[31:30] == 0b10, e.g. 0x80000000) is a single-dword filler NOP — the CP
            // skips it and keeps executing. sceAgcCbNop(dcb, 1) emits one (a 1-dword pad cannot be a
            // type-3 packet, whose minimum is 2 dwords). Skip it and continue rather than ending the
            // walk, so a 1-dword pad does not truncate the rest of the submit (#401). Any OTHER
            // non-type-3 dword (type-0 register writes we never emit, zero padding, garbage) still
            // ends the walk as before.
            if ((h & 0xC0000000u) == 0x80000000u) { i += 1; continue; }
            break;                                     // not a packet -> stop (pad/garbage)
        }
        uint32_t len = hdr_len(h);
        if (len == 0 || i + len > dwords) break;        // truncated packet -> stop

        Pm4Command c;
        c.header  = h;
        c.op      = hdr_op(h);
        c.r       = hdr_r(h);
        c.len     = len;
        c.payload = (len > 1) ? &buf[i + 1] : nullptr;
        const uint32_t* pl = c.payload;
        const uint32_t npl = len - 1;                   // payload dword count

        using K = Pm4Command::Kind;
        if (c.op == IT_DISPATCH_DIRECT && npl == 4) {
            // Hardware PM4 DISPATCH_DIRECT (GFX10, 5 dwords): [0..2] = DIM_X/Y/Z, [3] =
            // COMPUTE_DISPATCH_INITIATOR. The custom R_DISPATCH_DIRECT modifier is that same
            // register's bit layout (USE_THREAD_DIMENSIONS, CS_W32_EN, ...), so the initiator
            // feeds resolve_compute_launch() unchanged. Other lengths stay Unknown.
            c.kind = K::DispatchDirect;
            c.threads_x = pl[0];
            c.threads_y = pl[1];
            c.threads_z = pl[2];
            c.dispatch_modifier = pl[3];
        } else if (c.op == IT_DISPATCH_INDIRECT && npl == 2) {
            // Hardware PM4 DISPATCH_INDIRECT, graphics-ring form (GFX10, 3 dwords): [0] =
            // DATA_OFFSET from the compute indirect base set by SET_BASE, [1] =
            // COMPUTE_DISPATCH_INITIATOR.
            c.kind = K::DispatchIndirect;
            c.indirect_offset = pl[0];
            c.dispatch_modifier = pl[1];
        } else if (c.op == IT_DISPATCH_INDIRECT && npl == 3) {
            // Hardware PM4 DISPATCH_INDIRECT, compute-queue (MEC) form (4 dwords): [0..1] = the
            // argument buffer's whole address, [2] = COMPUTE_DISPATCH_INITIATOR. A compute queue has
            // no SET_BASE, so the address is absolute -- the same contract as R_DISPATCH_INDIRECT_ADDR.
            // CONFIDENCE: MED (published MEC layout; not yet observed in a PS5 capture).
            c.kind = K::DispatchIndirect;
            c.indirect_address = lo_hi(pl) & ~3ull;
            c.indirect_address_absolute = true;
            c.dispatch_modifier = pl[2];
        } else if (c.op == IT_SET_BASE && npl == 3 && (pl[0] & 0xfu) == 1u) {
            // Hardware PM4 SET_BASE (GFX10, 4 dwords): [0] = BASE_INDEX, [1..2] = address lo/hi.
            // BASE_INDEX 1 is the indirect-argument base; header bit 1 (SHADER_TYPE) says whose:
            // 0 = graphics (DRAW_*_INDIRECT), 1 = compute (DISPATCH_INDIRECT) -- the same 0/1
            // convention as the custom R_SET_BASE_INDIRECT_ARGS. Other base indices (GDS / CE
            // partition bases, display-list patch table) stay Unknown.
            c.kind = K::SetBaseIndirectArgs;
            c.indirect_shader_type = (h >> 1) & 1u;
            c.indirect_base = lo_hi(&pl[1]);
        } else if (c.op == IT_DRAW_INDEX_AUTO && npl == 2) {
            // Hardware PM4 DRAW_INDEX_AUTO (GFX10, 3 dwords): [0] = INDEX_COUNT, [1] =
            // VGT_DRAW_INITIATOR. The initiator is not a ShaderDrawModifier, so di_modifier keeps
            // its zero value (the R_DRAW_INDEX_AUTO legacy three-dword rule).
            c.kind = K::DrawIndexAuto;
            c.index_count = pl[0];
        } else if ((c.op == IT_DRAW_INDIRECT || c.op == IT_DRAW_INDEX_INDIRECT) && npl == 4) {
            // Hardware PM4 DRAW_INDIRECT / DRAW_INDEX_INDIRECT (GFX10, 5 dwords, see
            // AGC_PACKET_SIZES.md): [0] = DATA_OFFSET from the graphics indirect base, [1] =
            // BASE_VTX_LOC, [2] = START_INST_LOC, [3] = VGT_DRAW_INITIATOR. The argument layout
            // follows from the opcode, exactly as for R_DRAW_INDIRECT / R_DRAW_INDEX_INDIRECT.
            c.kind = c.op == IT_DRAW_INDIRECT ? K::DrawIndirect : K::DrawIndexIndirect;
            c.indirect_offset = pl[0];
        } else if (c.op == IT_INDEX_BASE && npl == 2) {
            // Hardware PM4 INDEX_BASE (GFX10, 3 dwords): [0..1] = index buffer address lo/hi.
            c.kind = K::SetIndexBase;
            c.ib_addr = lo_hi(pl);
        } else if (c.op == IT_WRITE_DATA && npl >= 4 && (pl[0] & (1u << 16)) == 0 &&
                   (((pl[0] >> 8) & 0xfu) == 2u || ((pl[0] >> 8) & 0xfu) == 5u)) {
            // Hardware PM4 WRITE_DATA (GFX10): [0] = CONTROL, [1..2] = DST_ADDR lo/hi, [3..] =
            // data. Only memory destinations decode -- CONTROL.DST_SEL 2 (TC_L2) or 5 (memory) --
            // with consecutive addressing (WR_ONE_ADDR, bit 16, clear). A register, GDS or
            // single-address destination stays Unknown: its "address" is not a guest pointer.
            c.kind = K::WriteData;
            c.wd_addr = lo_hi(&pl[1]);
            c.wd_declared_num = npl - 3;
            c.wd_num = c.wd_declared_num;
            c.wd_data = &pl[3];
            c.wd_valid = true;
        } else if (c.op == IT_WAIT_REG_MEM && npl == 6 && ((pl[0] >> 4) & 3u) == 1u &&
                   ((pl[0] >> 6) & 3u) == 0u) {
            // Hardware PM4 WAIT_REG_MEM (GFX10, 7 dwords): [0] = FUNCTION[2:0] | MEM_SPACE[5:4] |
            // OPERATION[7:6] | ENGINE_SEL, [1..2] = POLL_ADDRESS lo/hi, [3] = REFERENCE, [4] = MASK,
            // [5] = POLL_INTERVAL. FUNCTION uses the numbering wait_regmem_value_satisfied()
            // already implements (0 always .. 6 greater). Only a plain memory poll decodes:
            // MEM_SPACE 0 polls a register (its "address" is a register offset) and OPERATION != 0
            // writes before waiting, so either stays Unknown. The 32-bit MASK zero-extends, so the
            // qword the processor reads compares on its low dword only.
            c.kind = K::WaitRegMem;
            c.wm_func = pl[0] & 7u;
            c.wm_addr = lo_hi(&pl[1]) & ~3ull;
            c.wm_ref = pl[3];
            c.wm_mask = pl[4];
            c.wm_valid = true;
        } else if (c.op == IT_INDEX_TYPE) {
            c.kind = K::SetIndexType;
            if (npl >= 1) c.index_size = pl[0];
        } else if (c.op == IT_NUM_INSTANCES) {
            c.kind = K::SetNumInstances;
            if (npl >= 1) c.instance_count = pl[0];
        } else if (c.op == IT_EVENT_WRITE) {
            c.kind = K::EventWrite;
            if (npl >= 1) c.event_type = pl[0] & 0xffu;
            // Address-carrying (timestamp/label) variant: [1..2] = address lo/hi (#132). Address-less
            // pipeline-sync events leave event_addr == 0 (a no-op in the CommandProcessor).
            if (npl >= 3) c.event_addr = (uint64_t)pl[1] | ((uint64_t)pl[2] << 32);
        } else if (c.op == IT_SET_CONTEXT_REG || c.op == IT_SET_SH_REG ||
                   c.op == IT_SET_UCONFIG_REG) {
            // SET_*_REG sets a RANGE: pl[0] = start register offset, pl[1..npl-1] = consecutive
            // values. Capture the class from the opcode so single-register Cx/Uc builders do not
            // silently disappear and SH range packets keep their existing behavior (#395 F5).
            c.kind = K::SetRegDirect;
            c.reg_class = c.op == IT_SET_CONTEXT_REG ? RegClass::Cx
                        : c.op == IT_SET_SH_REG ? RegClass::Sh : RegClass::Uc;
            if (npl >= 2) {
                c.reg_offset = pl[0]; c.reg_value = pl[1];
                c.reg_count = npl - 1; c.reg_data = &pl[1];
            }
        } else if (c.op == IT_ACQUIRE_MEM && npl == 7) {
            // Hardware PM4 ACQUIRE_MEM (GFX10: header + COHER_CNTL, COHER_SIZE, COHER_SIZE_HI,
            // COHER_BASE, COHER_BASE_HI, POLL_INTERVAL, GCR_CNTL = 8 dwords), as the PS5's own
            // libSceAgc emits it: a cache flush/invalidate barrier with no memory write. Only the
            // GFX10 length decodes; any other length stays Unknown.
            c.kind = K::AcquireMem;
        } else if (c.op == IT_RELEASE_MEM && npl == 7) {
            // Hardware PM4 RELEASE_MEM (GFX10, 8 dwords): [0]=EVENT_CNTL, [1]=DATA_CNTL,
            // [2..3]=ADDRESS_LO/HI, [4..5]=DATA_LO/HI, [6]=INT_CTXID. DATA_CNTL[31:29] is DATA_SEL
            // (0 none, 1 low 32 bits, 2 64 bits, 3 GPU clock) -- the same selector numbering the
            // custom R_RELEASE_MEM payload carries, so honor_eop_write() applies unchanged. No #312
            // build snapshot exists in this shape, so the generation guard stays inert for it.
            // Only the GFX10 length decodes; any other length stays Unknown. CONFIDENCE: MED.
            c.kind = K::ReleaseMem;
            c.rel_data_sel = (pl[1] >> 29) & 7u;
            c.rel_addr = lo_hi(&pl[2]);
            c.rel_value = lo_hi(&pl[4]);
            c.rel_value_valid = true;
        } else if ((c.op == IT_LOAD_SH_REG || c.op == IT_LOAD_CONTEXT_REG ||
                    c.op == IT_LOAD_CONTEXT_REG_INDEX || c.op == IT_LOAD_UCONFIG_REG) &&
                   npl == 4 && (pl[0] & 3u) == 0 && (pl[2] & 0x80000000u) != 0) {
            // Hardware PM4 indirect register loads, as the PS5's own libSceAgc emits them (#4822):
            //   [0..1] = 64-bit guest address of a ShaderReg {offset, value} array (lo/hi),
            //   [2]    = bit 31 set: the array holds offset/value pairs,
            //   [3]    = number of pairs.
            // Seen in a Black Flag (PPSA28183) submit captured on PS5 hardware: 330 0x63 packets
            // loading user-data SGPRs, plus 0x64 and 0x9F context loads. Only that observed shape
            // decodes: an index/offset address mode (low address bits), the contiguous-range data
            // format (bit 31 clear) or another length stays Unknown rather than being read as
            // pairs. CONFIDENCE: MED (one title's capture; 0x64's role is inferred from it).
            c.kind = K::SetRegsIndirect;
            c.reg_class = (c.op == IT_LOAD_SH_REG)        ? RegClass::Sh
                          : (c.op == IT_LOAD_UCONFIG_REG) ? RegClass::Uc
                                                          : RegClass::Cx;
            c.regs_vaddr = lo_hi(pl);
            c.num_regs = pl[3];
        } else if (c.op == IT_INDIRECT_BUFFER && npl == 13) {
            // sceAgcCbBranch's packet (#4540). Other lengths of this opcode are not emitted by any
            // prosper builder and stay Unknown.
            c.kind = K::CondIndirectBuffer;
            c.cib_mode = pl[0] & 3u;
            c.cib_func = (pl[0] >> 8) & 7u;
            c.cib_compare_addr = lo_hi(&pl[1]) & ~7ull;
            c.cib_mask = lo_hi(&pl[3]);
            c.cib_reference = lo_hi(&pl[5]);
            c.cib_then_addr = lo_hi(&pl[7]) & ~3ull;
            c.cib_then_dwords = pl[9] & 0xfffffu;
            c.cib_else_addr = lo_hi(&pl[10]) & ~3ull;
            c.cib_else_dwords = pl[12] & 0xfffffu;
        } else if (c.op == IT_NOP) {
            switch (c.r) {
                case R_DRAW_RESET:    c.kind = K::DrawReset;    break;
                case R_WAIT_FLIP_DONE:c.kind = K::WaitFlipDone; break;
                case R_PUSH_MARKER:
                    c.kind = K::PushMarker;
                    c.marker_label = npl ? reinterpret_cast<const char*>(pl) : "";
                    break;
                case R_POP_MARKER:    c.kind = K::PopMarker;    break;
                case R_ACQUIRE_MEM:   c.kind = K::AcquireMem;   break;
                case R_WRITE_DATA:
                    c.kind = K::WriteData;
                    // payload: [0]=dst, [1..2]=addr lo/hi, [3]=num_dwords, [4..]=inline data dwords.
                    if (npl >= 3) c.wd_addr = (uint64_t)pl[1] | ((uint64_t)pl[2] << 32);
                    if (npl >= 4) {
                        c.wd_declared_num = pl[3];
                        const uint32_t avail = (npl > 4) ? (npl - 4) : 0;
                        c.wd_num = std::min(c.wd_declared_num, avail); // never read past the packet
                        c.wd_data = (c.wd_num > 0) ? &pl[4] : nullptr;
                        c.wd_valid = c.wd_declared_num <= avail;
                    }
                    break;
                case R_WAIT_MEM_64:
                    c.kind = K::WaitRegMem;
                    // payload: [0..1]=addr lo/hi, [2..3]=mask lo/hi, [4..5]=reference lo/hi,
                    // [6]=compare_function (Kyty GraphicsDcbWaitRegMem layout; see agc_dcb_wait_reg_mem).
                    if (npl >= 7) {
                        c.wm_addr = lo_hi(pl);
                        c.wm_mask = (uint64_t)pl[2] | ((uint64_t)pl[3] << 32);
                        c.wm_ref  = (uint64_t)pl[4] | ((uint64_t)pl[5] << 32);
                        c.wm_func = pl[6];
                        c.wm_valid = true;
                    }
                    break;
                case R_RELEASE_MEM:
                    c.kind = K::ReleaseMem;
                    // payload: [0..1]=label addr lo/hi, [2]=data_sel, [3..4]=64-bit value (see agc_cb_release_mem)
                    if (npl >= 2) c.rel_addr = lo_hi(pl);
                    if (npl >= 3) c.rel_data_sel = pl[2];
                    if (npl >= 5) { c.rel_value = (uint64_t)pl[3] | ((uint64_t)pl[4] << 32); c.rel_value_valid = true; }
                    // #312 build-time snapshot of the destination qword. TWO packet shapes exist and
                    // the payload length is the discriminator:
                    //   npl == 7 — current 8-dword packet (#1748, sized to hardware RELEASE_MEM):
                    //              pl[6] holds only the HIGH half, the only half the generation guard
                    //              reads (it masks with 0xffffffff00000000).
                    //   npl >= 8 — historical 9-dword packet, still present in every capture recorded
                    //              before #1748: pl[6..7] hold the full qword.
                    if (npl >= 8) {
                        c.rel_build_pre = lo_hi(pl + 6);
                        c.rel_build_pre_valid = true;
                    } else if (npl >= 7) {
                        c.rel_build_pre = (uint64_t)pl[6] << 32;
                        c.rel_build_pre_valid = true;
                    }
                    break;
                case R_FLIP:
                    c.kind = K::Flip;
                    // payload: [0]=videoout handle, [1]=buffer index, [2]=flip mode, [3..4]=64-bit
                    // flipArg (see agc_dcb_set_flip).
                    if (npl >= 5) {
                        c.flip_handle = pl[0];
                        c.flip_bufidx = (int32_t)pl[1];
                        c.flip_mode   = pl[2];
                        c.flip_arg    = (int64_t)((uint64_t)pl[3] | ((uint64_t)pl[4] << 32));
                        c.flip_valid  = true;
                    }
                    break;
                case R_DRAW_INDEX:
                    // payload: [0]=index_count, [1..2]=index-buffer addr lo/hi, [3..4]=64-bit draw
                    // modifier lo/hi (see agc_dcb_draw_index + the ABI evidence in pm4_decode.hpp).
                    c.kind = K::DrawIndex;
                    if (npl >= 1) c.index_count = pl[0];
                    if (npl >= 3) c.di_index_addr = lo_hi(pl + 1);
                    if (npl >= 5) { c.di_modifier = lo_hi(pl + 3); c.di_valid = true; }
                    break;
                case R_DRAW_INDEX_AUTO:
                    c.kind = K::DrawIndexAuto;
                    if (npl >= 1) c.index_count = pl[0];
                    // Custom HLE payload: [0]=count, [1..2]=64-bit ShaderDrawModifier. Legacy
                    // three-dword hardware/test packets have only count+initiator and retain 0.
                    if (npl >= 3) c.di_modifier = lo_hi(pl + 1);
                    break;
                case R_DISPATCH_DIRECT:
                    c.kind = K::DispatchDirect;
                    // Custom HLE payload: [0..2]=raw dimensions x/y/z, [3..4]=64-bit modifier.
                    if (npl >= 3) { c.threads_x = pl[0]; c.threads_y = pl[1]; c.threads_z = pl[2]; }
                    // A short legacy/test packet has no modifier and deliberately retains the
                    // zero-initialized value, whose documented interpretation is workgroup mode.
                    if (npl >= 5) c.dispatch_modifier = (uint64_t)pl[3] | ((uint64_t)pl[4] << 32);
                    break;
                case R_SET_BASE_INDIRECT_ARGS:
                    c.kind = K::SetBaseIndirectArgs;
                    if (npl >= 3) {
                        c.indirect_shader_type = pl[0];
                        c.indirect_base = lo_hi(pl + 1);
                    }
                    break;
                case R_STALL_COMMAND_BUFFER_PARSER:
                    c.kind = K::StallCommandBufferParser;
                    break;
                case R_DRAW_INDEX_INDIRECT:
                    c.kind = K::DrawIndexIndirect;
                    if (npl >= 1) c.indirect_offset = pl[0];
                    if (npl >= 3) c.di_modifier = lo_hi(pl + 1);
                    break;
                case R_DRAW_INDIRECT:
                    // sceAgcDcbDrawIndirect (#2929). Same payload shape as the indexed sibling —
                    // [0] = byte offset into the graphics argument base, [1..2] = the 64-bit
                    // ShaderDrawModifier — but a distinct kind, because the arguments it points at
                    // are the four-dword non-indexed form and no index buffer is bound.
                    c.kind = K::DrawIndirect;
                    if (npl >= 1) c.indirect_offset = pl[0];
                    if (npl >= 3) c.di_modifier = lo_hi(pl + 1);
                    break;
                case R_DISPATCH_INDIRECT:
                    c.kind = K::DispatchIndirect;
                    if (npl >= 1) c.indirect_offset = pl[0];
                    if (npl >= 3) c.dispatch_modifier = lo_hi(pl + 1);
                    break;
                case R_DISPATCH_INDIRECT_ADDR:
                    // sceAgcAcbDispatchIndirect: payload [0..1] = the argument buffer's whole
                    // 64-bit address, [2..3] = the 64-bit ShaderDispatchModifier. The ACB has no
                    // SetBase in the whole library, so there is nothing for an offset to be
                    // relative to; a short packet is left non-absolute and therefore resolves to
                    // address 0, which the executor already rejects, rather than silently becoming
                    // a base-relative offset.
                    c.kind = K::DispatchIndirect;
                    if (npl >= 2) {
                        c.indirect_address = lo_hi(pl);
                        c.indirect_address_absolute = true;
                    }
                    if (npl >= 4) c.dispatch_modifier = lo_hi(pl + 2);
                    break;
                case R_INDEX_BASE:
                    // sceAgcDcbSetIndexBuffer -> bind the index buffer base address. payload: [0..1]=addr.
                    c.kind = K::SetIndexBase;
                    if (npl >= 2) c.ib_addr = lo_hi(pl);
                    break;
                case R_INDEX_COUNT:
                    // sceAgcDcbSetIndexCount -> set the bound index count. payload: [0]=count.
                    c.kind = K::SetIndexCount;
                    if (npl >= 1) c.index_count = pl[0];
                    break;
                case R_DRAW_INDEX_OFFSET:
                    // sceAgcDcbDrawIndexOffset -> indexed draw from a start offset using the bound
                    // index base + count. payload: [0]=start-index offset, [1]=explicit index count
                    // (0 => use the SetIndexCount state, threaded in by GpuState::apply).
                    c.kind = K::DrawIndexOffset;
                    if (npl >= 1) c.index_offset = pl[0];
                    if (npl >= 2) c.index_count  = pl[1];
                    break;
                case R_JUMP:
                    // sceAgcDcbJump (#319): call-with-length of a side command segment.
                    // payload: [0..1]=target addr lo/hi, [2]=dword count. The predication flag is
                    // HEADER bit 0 — the hardware type-3 PREDICATE bit — since #3676, which took the
                    // packet from 5 dwords to the INDIRECT_BUFFER's 4 (see hle_agc.cpp kDwJump).
                    //
                    // pl[3] is still read when a 5-dword packet arrives, so a capture recorded
                    // before #3676 replays with its predication intact rather than silently losing
                    // it. The two can never both be set by a live guest: this build emits 4.
                    c.kind = K::Jump;
                    if (npl >= 3) {
                        c.jump_addr   = lo_hi(pl);
                        c.jump_dwords = pl[2];
                        c.jump_pred   = (h & 1u) | (npl >= 4 ? pl[3] : 0u);
                        c.jump_valid  = true;
                    }
                    break;
                case R_DMA_DATA:
                    // sceAgcDcbDmaData / sceAgcAcbDmaData (#312): CP-DMA. TWO packet shapes exist and
                    // the payload length is the discriminator, exactly as for R_RELEASE_MEM above:
                    //   npl == 6 — current 7-dword packet (#1756, sized to hardware DMA_DATA):
                    //              [0..1]=dst lo/hi, [2..3]=srcOrImm lo/hi, [4]=numBytes, [5]=sels.
                    //              The #312 build snapshot is NOT carried; there is no spare slot at
                    //              the hardware size, so dd_build_pre_valid stays false.
                    //   npl >= 8 — historical 9-dword packet, in every capture recorded before #1756:
                    //              the same six fields plus [6..7] = the destination qword as it read
                    //              when the packet was built.
                    c.kind = K::DmaData;
                    if (npl >= 6) {
                        c.dd_dst   = lo_hi(pl);
                        c.dd_src   = lo_hi(pl + 2);
                        c.dd_bytes = pl[4];
                        c.dd_sels  = pl[5];
                        c.dd_valid = true;
                    }
                    if (npl >= 8) {
                        c.dd_build_pre = lo_hi(pl + 6);
                        c.dd_build_pre_valid = true;
                    }
                    break;
                case R_SET_PRED:
                    // sceAgcDcbSetPredication (#319): begin/end a predication window.
                    // payload: [0..1]=condition addr lo/hi (0 = end), [2]=control word
                    // (op + flag arguments, pack_set_predication_control).
                    c.kind = K::SetPredication;
                    if (npl >= 3) {
                        c.pred_addr  = lo_hi(pl);
                        c.pred_op    = pl[2];
                        c.pred_valid = true;
                    }
                    break;
                case R_CX_REGS_INDIRECT:
                case R_SH_REGS_INDIRECT:
                case R_UC_REGS_INDIRECT:
                    c.kind = K::SetRegsIndirect;
                    c.reg_class = (c.r == R_CX_REGS_INDIRECT) ? RegClass::Cx
                                : (c.r == R_SH_REGS_INDIRECT) ? RegClass::Sh : RegClass::Uc;
                    // payload: [0]=num_regs, [1..2]=regs vaddr lo/hi (Set*RegsIndirect in hle_agc).
                    if (npl >= 1) c.num_regs = pl[0];
                    if (npl >= 3) c.regs_vaddr = lo_hi(pl + 1);
                    break;
                default:
                    // A prosper-built packet whose sub-op nothing decodes. This fell to Unknown in
                    // silence, and it is a different miss from the raw-opcode one reported below: the
                    // header IS ours, so the guest went through an HLE builder and the gap is on our
                    // side -- a builder that emits an R_ value the decoder never learned, whose
                    // packet is then dropped with no trace at all.
                    //
                    // That silence is exactly what an investigation into a surface "written by
                    // nothing" cannot afford: every other instrument observes DECODED work, so a
                    // packet lost here is invisible to all of them. Reported once per distinct
                    // sub-op, ungated, matching the raw-opcode reporter beside it.
                    {
                        static std::mutex sub_mutex;
                        static std::set<uint32_t> seen_sub;
                        bool first = false;
                        {
                            std::lock_guard<std::mutex> lock(sub_mutex);
                            first = seen_sub.insert(c.r).second;
                        }
                        if (first)
                            std::fprintf(stderr,
                                         "[pm4] undecoded prosper sub-op r=0x%02x op=0x%02x "
                                         "len=%u dwords\n",
                                         c.r, c.op, (unsigned)(npl + 1));
                    }
                    c.kind = K::Unknown;
                    break;
            }
        } else {
            c.kind = K::Unknown;
            // #1124: PPSA02664 re-arms cloned workload DMA templates by writing raw PM4 headers the
            // game computed itself, which prosper's custom-packet fold has never met. Surface every
            // distinct unknown raw type-3 opcode ONCE (with length + leading payload) so a guest
            // bypassing the HLE builders is loud instead of silently skipped.
            static uint32_t unk_n = 0;              // benign-race diagnostic counter
            static uint32_t unk_ops[16];
            const uint32_t idx = unk_n;             // snapshot so the write index is provably < 16
            bool seen = false;
            for (uint32_t k = 0; k < idx && k < 16; k++) if (unk_ops[k] == c.op) { seen = true; break; }
            if (!seen && idx < 16) {
                unk_ops[idx] = c.op; unk_n = idx + 1;
                std::fprintf(stderr, "[pm4] unknown raw type-3 opcode 0x%02x len=%u dwords @%p:", c.op,
                             (unsigned)(npl + 1), (const void*)&buf[i]);
                for (uint32_t k = 0; k < npl && k < 8; k++) std::fprintf(stderr, " %08x", pl[k]);
                std::fprintf(stderr, "\n");
            }
        }

        out.push_back(c);
        i += len;
    }
    return i;
}

} // namespace prosper::gpu
