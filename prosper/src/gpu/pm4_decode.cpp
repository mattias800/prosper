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
        } else if (c.op == IT_SET_SH_REG) {
            c.kind = K::SetShRegDirect;
            if (npl >= 2) { c.sh_reg_offset = pl[0]; c.sh_reg_value = pl[1]; }
        } else if (c.op == IT_NOP) {
            switch (c.r) {
                case R_DRAW_RESET:    c.kind = K::DrawReset;    break;
                case R_WAIT_FLIP_DONE:c.kind = K::WaitFlipDone; break;
                case R_ACQUIRE_MEM:   c.kind = K::AcquireMem;   break;
                case R_WRITE_DATA:    c.kind = K::WriteData;    break;
                case R_WAIT_MEM_64:   c.kind = K::WaitRegMem;   break;
                case R_RELEASE_MEM:
                    c.kind = K::ReleaseMem;
                    // payload: [0..1]=dst label addr lo/hi, [2]=value a6, [3]=value a7 (see agc_cb_release_mem)
                    if (npl >= 2) c.rel_addr = lo_hi(pl);
                    if (npl >= 3) c.rel_v6 = pl[2];
                    if (npl >= 4) c.rel_v7 = pl[3];
                    break;
                case R_FLIP:          c.kind = K::Flip;         break;
                case R_DRAW_INDEX_AUTO:
                    c.kind = K::DrawIndexAuto;
                    if (npl >= 1) c.index_count = pl[0];
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
