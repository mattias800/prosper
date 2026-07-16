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
    R_PUSH_MARKER = 0x0b, R_POP_MARKER = 0x0c,
    R_SH_REGS_INDIRECT = 0x11, R_CX_REGS_INDIRECT = 0x12, R_UC_REGS_INDIRECT = 0x13,
    R_ACQUIRE_MEM = 0x14, R_WRITE_DATA = 0x15, R_WAIT_MEM_64 = 0x16, R_FLIP = 0x17,
    R_RELEASE_MEM = 0x18, R_DMA_DATA = 0x19, R_DISPATCH_DIRECT = 0x1a,
    // Gen5 indexed-draw state + draw (issue #232, DOLL/UE4 geometry path). These three builders
    // (sceAgcDcbSetIndexBuffer / SetIndexCount / DrawIndexOffset) are the ONLY draw path UE4 uses;
    // they were unimplemented->0 (no packet appended) so every scene draw was silently dropped.
    R_INDEX_BASE = 0x1b, R_INDEX_COUNT = 0x1c, R_DRAW_INDEX_OFFSET = 0x1d,
    // Command-buffer call + GPU predication (issue #319, DOLL/UE4 backbuffer composite).
    // sceAgcDcbJump appends R_JUMP = "execute `jump_dwords` dwords at `jump_addr`, then resume"
    // (a call-with-length: live capture shows each target segment is exactly the passed size with
    // no jump-back packet, and the parent stream continues after — so the CP must return).
    // sceAgcDcbSetPredication appends R_SET_PRED = "begin (addr!=0) / end (addr==0) a predication
    // window over the following packets, condition = the 64-bit value at addr".
    R_JUMP = 0x1e, R_SET_PRED = 0x1f,
    R_NUM = 0x40,
};

// The register set a Set*RegistersIndirect packet targets.
enum class RegClass { Cx, Sh, Uc };

// A decoded command. `kind` names the semantic operation; the union-ish fields carry its operands.
// `header`/`op`/`r`/`len`/`payload` always describe the raw packet (payload = the len-1 dwords after
// the header), so unknown packets are still walkable and inspectable.
struct Pm4Command {
    enum class Kind {
        DrawReset, WaitFlipDone, PushMarker, PopMarker, SetShRegDirect, SetRegsIndirect, SetIndexType,
        DrawIndex, DrawIndexAuto, EventWrite, AcquireMem, WriteData, WaitRegMem, Flip, ReleaseMem,
        DispatchDirect, SetIndexBase, SetIndexCount, DrawIndexOffset, Jump, SetPredication,
        DmaData, Unknown,
    } kind = Kind::Unknown;

    uint64_t stream_order = 0;          // assigned before apply and retained by deferred effects
    uint32_t        header = 0;
    uint32_t        op = 0, r = 0, len = 0;   // op, sub-op, total dwords (incl. header)
    const uint32_t* payload = nullptr;        // points at header+1 (len-1 dwords)

    // PushMarker label inside the packet payload. The builder always NUL-terminates it; consumers
    // that retain commands past the source stream's lifetime must copy it.
    const char* marker_label = nullptr;

    // Decoded operands (only the ones relevant to `kind` are meaningful):
    RegClass reg_class = RegClass::Cx;   // SetRegsIndirect
    uint32_t num_regs = 0;               // SetRegsIndirect
    uint64_t regs_vaddr = 0;             // SetRegsIndirect: guest addr of the register array
    uint32_t index_count = 0;            // DrawIndexAuto / DrawIndex
    uint32_t index_size = 0;             // SetIndexType

    // CbDispatch (sceAgcCbDispatch -> R_DISPATCH_DIRECT, hle_agc.cpp). This is a prosper custom
    // packet, not raw hardware DISPATCH_DIRECT: payload [0..2] retains raw API dimensions x/y/z.
    // Dispatch modifier bit 5 selects total-thread dimensions; when clear, they are already
    // workgroup counts. [3..4] is the 64-bit dispatch modifier from the CS shader's specials block.
    uint32_t threads_x = 0, threads_y = 0, threads_z = 0;
    uint64_t dispatch_modifier = 0;          // DispatchDirect modifier bits

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

    // Gen5 indexed-draw state (issue #232). SetIndexBase: [0..1]=index buffer addr lo/hi.
    // SetIndexCount: [0]=index count. DrawIndexOffset: [0]=start-index offset, [1]=draw index count
    // (0 => use the SetIndexCount state). These carry no register state; GpuState::apply threads the
    // bound base+count through to the emitted draw.
    uint64_t ib_addr = 0;                // SetIndexBase: index buffer guest address
    uint32_t index_offset = 0;           // DrawIndexOffset: first index (start location)
    uint32_t event_type = 0;             // EventWrite
    uint64_t event_addr = 0;             // EventWrite: destination address (0 = address-less sync event) (#132)
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
    uint64_t rel_build_pre = 0;          // ReleaseMem: destination qword when this packet was built
    bool rel_build_pre_valid = false;    // ReleaseMem: extended packet carried the build snapshot

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
    // Jump (sceAgcDcbJump -> R_JUMP, hle_agc.cpp agc_dcb_jump; #319). Payload: [0..1]=target
    // address lo/hi, [2]=segment dword count, [3]=predicated flag (set by sceAgcSetPacketPredication
    // patching the built packet). The CommandProcessor folds the target segment inline (call-with-
    // length), gated by the active predication window when the flag is set.
    uint64_t jump_addr = 0;              // Jump: target segment guest address
    uint32_t jump_dwords = 0;            // Jump: segment length in dwords
    uint32_t jump_pred = 0;              // Jump: 1 = packet-predicated (participates in the window)
    bool     jump_valid = false;         // Jump: payload carried the full operand set

    // SetPredication (sceAgcDcbSetPredication -> R_SET_PRED; #319). Payload: [0..1]=condition
    // address lo/hi (0 = end the window), [2]=raw predication op (live capture: 3 on begin, 0 on end).
    uint64_t pred_addr = 0;              // SetPredication: 64-bit condition address (0 = window end)
    uint32_t pred_op = 0;                // SetPredication: raw op field
    bool     pred_valid = false;         // SetPredication: payload carried the operands

    // DmaData (sceAgcDcbDmaData / sceAgcAcbDmaData -> R_DMA_DATA; issue #312). The CP-DMA engine
    // packet. ABI re-pinned from three eboot callsites (see agc_dcb_dma_data): a1 = source address
    // OR immediate value (the patcher family is named sceAgcDmaDataPatchSetSrcAddressOrOffsetOr-
    // IMMEDIATE), a4 = DESTINATION address (a callsite passes a freshly-malloc'd buffer there),
    // stack arg9 = byte count. DOLL's RHI translate loop emits DmaData(src=0, dst=<per-chunk fence
    // label>, 4 bytes) per segment — the GPU-side label INIT (label := 0) of the consumed-marker
    // protocol whose completion leg is ReleaseMem(label <- 1). Packet payload:
    // [0..1]=dst lo/hi, [2..3]=srcOrImm lo/hi, [4]=numBytes, [5]=sels (a2 | a3<<8),
    // [6..7]=the destination qword captured at build time (stale-generation identity, #312).
    uint64_t dd_dst = 0;                 // DmaData: destination address
    uint64_t dd_src = 0;                 // DmaData: source address OR immediate value
    uint32_t dd_bytes = 0;               // DmaData: byte count (0 = unrecovered -> not executed)
    uint32_t dd_sels = 0;                // DmaData: raw selector args (a2 | a3<<8)
    bool     dd_valid = false;           // DmaData: payload carried the full operand set
    uint64_t dd_build_pre = 0;           // DmaData: destination qword when this exact packet was built
    bool     dd_build_pre_valid = false; // DmaData: extended packet carried the build snapshot

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
