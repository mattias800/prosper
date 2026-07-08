// pm4_decode.hpp — decode a libSceAgc "Gen5" Draw Command Buffer (a dword stream of PM4 type-3
// packets, as emitted by the Dcb functions in hle_agc.cpp) into a list of semantic operations.
//
// This is the front of the eventual CommandProcessor (AGC -> Vulkan, milestone M4): the game builds
// a command buffer via the Dcb functions, then submits it; the CommandProcessor walks the stream and
// translates each op to host GPU state/commands. The decoder here is pure (no I/O, no guest state),
// so it is unit-testable in isolation against streams built by the real Dcb builders (test_pm4_decode).
//
// PM4 encoding (Kyty Pm4.h, matches hle_agc.cpp's PM4()):
//   header = 0xC0000000 | ((len-2) & 0x3fff)<<16 | (op & 0xff)<<8 | (r & 0x3f)<<2
// where len = total dwords incl. header, op = IT_* opcode, r = custom sub-op carried in IT_NOP.
#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// IT_* opcodes and the R_* sub-opcodes carried inside IT_NOP (mirror hle_agc.cpp).
enum : uint32_t {
    IT_NOP = 0x10, IT_INDEX_TYPE = 0x2A, IT_EVENT_WRITE = 0x46, IT_SET_SH_REG = 0x76,
};
enum : uint32_t {
    R_DRAW_INDEX = 0x03, R_DRAW_INDEX_AUTO = 0x04, R_DRAW_RESET = 0x05, R_WAIT_FLIP_DONE = 0x06,
    R_SH_REGS_INDIRECT = 0x11, R_CX_REGS_INDIRECT = 0x12, R_UC_REGS_INDIRECT = 0x13,
    R_ACQUIRE_MEM = 0x14, R_WRITE_DATA = 0x15, R_WAIT_MEM_64 = 0x16, R_FLIP = 0x17,
    R_RELEASE_MEM = 0x18, R_NUM = 0x40,
};

// The register set a Set*RegistersIndirect packet targets.
enum class RegClass { Cx, Sh, Uc };

// A decoded command. `kind` names the semantic operation; the union-ish fields carry its operands.
// `header`/`op`/`r`/`len`/`payload` always describe the raw packet (payload = the len-1 dwords after
// the header), so unknown packets are still walkable and inspectable.
struct Pm4Command {
    enum class Kind {
        DrawReset, WaitFlipDone, SetShRegDirect, SetRegsIndirect, SetIndexType,
        DrawIndex, DrawIndexAuto, EventWrite, AcquireMem, WriteData, WaitRegMem, Flip, ReleaseMem, Unknown,
    } kind = Kind::Unknown;

    uint32_t        header = 0;
    uint32_t        op = 0, r = 0, len = 0;   // op, sub-op, total dwords (incl. header)
    const uint32_t* payload = nullptr;        // points at header+1 (len-1 dwords)

    // Decoded operands (only the ones relevant to `kind` are meaningful):
    RegClass reg_class = RegClass::Cx;   // SetRegsIndirect
    uint32_t num_regs = 0;               // SetRegsIndirect
    uint64_t regs_vaddr = 0;             // SetRegsIndirect: guest addr of the register array
    uint32_t index_count = 0;            // DrawIndexAuto / DrawIndex
    uint32_t index_size = 0;             // SetIndexType

    // DrawIndex (sceAgcDcbDrawIndex -> R_DRAW_INDEX, laid out by hle_agc.cpp agc_dcb_draw_index).
    // Packet payload: [0]=index_count, [1..2]=index-buffer guest address (lo/hi), [3..4]=64-bit draw
    // modifier (lo/hi). Field roles cross-checked against Kyty: GraphicsDrawIndex (Graphics.cpp:313)
    // emits exactly cmd[1]=index_count, cmd[2..3]=index_addr lo/hi under R_DRAW_INDEX, and its consumer
    // cp_op_draw_index (GraphicsRun.cpp:2757) reads buffer[0]=index_count, buffer[1..2]=index_addr —
    // CONFIDENCE: HIGH for count/address. [3..4] follows the Gen5 Dcb convention of a trailing 64-bit
    // ShaderDrawModifier (cf. GraphicsDcbDrawIndexAuto (buf, index_count, modifier), Graphics.cpp:1971);
    // Kyty's Gen4 form carries 32-bit flags+type there instead — CONFIDENCE: MED, decoded raw.
    // The index ELEMENT SIZE is not in this packet: it comes from the preceding SetIndexType
    // (GpuState::index_type), which the per-draw snapshot already captures.
    uint64_t di_index_addr = 0;          // DrawIndex: guest address of the index buffer
    uint64_t di_modifier = 0;            // DrawIndex: raw 64-bit draw-modifier bits
    bool     di_valid = false;           // DrawIndex: payload was long enough to carry addr+modifier
    uint32_t event_type = 0;             // EventWrite
    uint32_t sh_reg_offset = 0, sh_reg_value = 0;  // SetShRegDirect (sh_reg_value = first value)
    uint32_t sh_reg_count = 0;                 // SetShRegDirect: # of consecutive registers this packet sets
    const uint32_t* sh_reg_data = nullptr;     // SetShRegDirect: -> the value dwords (count of them) in-packet

    // ReleaseMem (EOP fence) — laid out by hle_agc.cpp agc_cb_release_mem, whose args are now pinned to the
    // AGC ABI sceAgcCbReleaseMem(buf, action, gcr_cntl, dst, cache_policy, address, data_sel, data, …)
    // (Kyty GraphicsCbReleaseMem, Graphics.cpp:1763). Packet payload: [0..1]=label address (lo/hi),
    // [2]=data_sel, [3..4]=64-bit fence value (lo/hi), [5]=event action. data_sel (Kyty allows {2,3};
    // shadPS4 DataSelect): 1=write 32-bit value, 2=write 64-bit value, 3=write 64-bit GPU clock.
    // CONFIDENCE: HIGH — every field decoded straight from the packet the builder wrote.
    uint64_t rel_addr = 0;               // ReleaseMem: destination label address
    uint32_t rel_data_sel = 0;           // ReleaseMem: DATA_SEL (1/2/3)
    uint64_t rel_value = 0;              // ReleaseMem: 64-bit fence value (for data_sel 1/2)
    bool rel_value_valid = false;        // ReleaseMem: packet was long enough to carry rel_value

    // WaitRegMem — laid out by hle_agc.cpp agc_dcb_wait_reg_mem per sceAgcDcbWaitRegMem(buf, size,
    // compare_func, op, cache_policy, address, reference, mask, poll_cycles) (Kyty
    // GraphicsDcbWaitRegMem, Graphics.cpp:2096). Payload: [0..1]=addr, [2..3]=mask, [4..5]=reference,
    // [6]=compare_function (PM4 WAIT_REG_MEM: 0=always 1=< 2=<= 3=== 4=!= 5=>= 6=>), [7]=interval.
    uint64_t wm_addr = 0;                // WaitRegMem: label address to poll
    uint64_t wm_mask = 0;                // WaitRegMem: AND-mask applied to the memory value
    uint64_t wm_ref  = 0;                // WaitRegMem: reference value
    uint32_t wm_func = 0;                // WaitRegMem: compare function
    bool     wm_valid = false;           // WaitRegMem: packet carried the full operand set

    // WriteData — laid out by hle_agc.cpp agc_dcb_write_data per sceAgcDcbWriteData(buf, dst, cache_policy,
    // address_or_offset, data*, num_dwords, …) (Kyty GraphicsDcbWriteData, Graphics.cpp:2061). Packet
    // payload: [0]=dst, [1..2]=destination address (lo/hi), [3]=num_dwords, [4..]=inline data dwords.
    uint64_t wd_addr = 0;                // WriteData: destination address
    uint32_t wd_num = 0;                 // WriteData: number of dwords to write
    const uint32_t* wd_data = nullptr;   // WriteData: -> the inline data dwords within the packet

    // Flip — laid out by hle_agc.cpp agc_dcb_set_flip per sceAgcDcbSetFlip(buf, video_out_handle,
    // display_buffer_index, flip_mode, flip_arg). Packet payload: [0]=videoout handle, [1]=display
    // buffer index, [2]=flip mode, [3..4]=64-bit flipArg. The GPU processing this packet IS the flip
    // moment: it must advance the videoout flip status (count/flipArg/currentBuffer) the game polls.
    uint32_t flip_handle = 0;            // Flip: sceVideoOut handle
    int32_t  flip_bufidx = -1;           // Flip: display buffer index (-1 = not decoded)
    uint32_t flip_mode = 0;              // Flip: flip mode
    int64_t  flip_arg = 0;               // Flip: the game's 64-bit flip argument
    bool     flip_valid = false;         // Flip: payload was long enough to carry the fields
};

// Decode `dwords` dwords starting at `buf` into `out` (appended). Returns the number of dwords
// consumed (== dwords on a clean stream). Stops early and returns the consumed count if it hits a
// dword that is not a valid type-3 header (top two bits != 0b11) or a packet that would overrun.
size_t decode_pm4(const uint32_t* buf, size_t dwords, std::vector<Pm4Command>& out);

} // namespace prosper::gpu
