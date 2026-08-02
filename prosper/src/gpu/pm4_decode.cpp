// pm4_decode.cpp — see pm4_decode.hpp. Pure PM4 type-3 stream walker.
#include <cstdio>
#include "pm4_decode.hpp"

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
        if (c.op == IT_INDEX_TYPE) {
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
                        c.wd_num  = pl[3];
                        uint32_t avail = (npl > 4) ? (npl - 4) : 0;   // dwords actually present in the packet
                        if (c.wd_num > avail) c.wd_num = avail;       // never read past the packet
                        c.wd_data = (c.wd_num > 0) ? &pl[4] : nullptr;
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
                case R_DISPATCH_INDIRECT:
                    c.kind = K::DispatchIndirect;
                    if (npl >= 1) c.indirect_offset = pl[0];
                    if (npl >= 3) c.dispatch_modifier = lo_hi(pl + 1);
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
                    // payload: [0..1]=target addr lo/hi, [2]=dword count, [3]=predicated flag.
                    c.kind = K::Jump;
                    if (npl >= 4) {
                        c.jump_addr   = lo_hi(pl);
                        c.jump_dwords = pl[2];
                        c.jump_pred   = pl[3];
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
                    // payload: [0..1]=condition addr lo/hi (0 = end), [2]=raw op.
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
                default: c.kind = K::Unknown; break;
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
