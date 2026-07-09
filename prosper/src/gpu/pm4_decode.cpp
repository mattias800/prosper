// pm4_decode.cpp — see pm4_decode.hpp. Pure PM4 type-3 stream walker.
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
        if (!is_type3(h)) break;                       // not a packet -> stop (pad/garbage)
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
        } else if (c.op == IT_EVENT_WRITE) {
            c.kind = K::EventWrite;
            if (npl >= 1) c.event_type = pl[0] & 0xffu;
            // Address-carrying (timestamp/label) variant: [1..2] = address lo/hi (#132). Address-less
            // pipeline-sync events leave event_addr == 0 (a no-op in the CommandProcessor).
            if (npl >= 3) c.event_addr = (uint64_t)pl[1] | ((uint64_t)pl[2] << 32);
        } else if (c.op == IT_SET_SH_REG) {
            // SET_SH_REG sets a RANGE: pl[0] = start register offset, pl[1..npl-1] = consecutive values
            // (this is how the driver uploads the whole user-data descriptor block in one packet — e.g. a
            // len-22 packet at GS_0 loads s0..s20). Capture the full range, not just the first register.
            c.kind = K::SetShRegDirect;
            if (npl >= 2) {
                c.sh_reg_offset = pl[0]; c.sh_reg_value = pl[1];
                c.sh_reg_count = npl - 1; c.sh_reg_data = &pl[1];
            }
        } else if (c.op == IT_NOP) {
            switch (c.r) {
                case R_DRAW_RESET:    c.kind = K::DrawReset;    break;
                case R_WAIT_FLIP_DONE:c.kind = K::WaitFlipDone; break;
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
                    break;
                case R_DISPATCH_DIRECT:
                    c.kind = K::DispatchDirect;
                    // payload: [0..2]=threadgroups x/y/z, [3..4]=64-bit dispatch modifier.
                    if (npl >= 3) { c.tg_x = pl[0]; c.tg_y = pl[1]; c.tg_z = pl[2]; }
                    if (npl >= 5) c.dispatch_modifier = (uint64_t)pl[3] | ((uint64_t)pl[4] << 32);
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
        }

        out.push_back(c);
        i += len;
    }
    return i;
}

} // namespace prosper::gpu
