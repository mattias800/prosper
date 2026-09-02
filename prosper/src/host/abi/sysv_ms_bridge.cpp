#include "host/abi/sysv_ms_bridge.hpp"
#include <algorithm>
#include <cstring>

namespace prosper::abi {

namespace {

// SysV integer argument registers, in the order the convention consumes them.
constexpr uint8_t kSysvIntRegs[6] = { kRdi, kRsi, kRdx, kRcx, kR8, kR9 };
// Microsoft x64 integer argument registers, likewise.
constexpr uint8_t kMsIntRegs[4]   = { kRcx, kRdx, kR8, kR9 };
constexpr unsigned kSysvSseRegs   = 8;   // xmm0..xmm7

struct Buf {
    uint8_t* p;
    size_t   n = 0;
    void b(uint8_t v) { p[n++] = v; }
    void u32(uint32_t v) { std::memcpy(p + n, &v, 4); n += 4; }
    void u64(uint64_t v) { std::memcpy(p + n, &v, 8); n += 8; }
};

// ModRM + SIB for a [rsp + disp] operand carrying `reg` in the reg field. rsp as a base always needs
// a SIB byte (0x24), and disp8 is used whenever it reaches.
void mem_rsp(Buf& b, uint8_t reg, uint32_t disp) {
    const uint8_t r = reg & 7;
    if (disp < 0x80) { b.b(uint8_t(0x40 | (r << 3) | 4)); b.b(0x24); b.b(uint8_t(disp)); }
    else             { b.b(uint8_t(0x80 | (r << 3) | 4)); b.b(0x24); b.u32(disp); }
}
void mov_reg_mem(Buf& b, uint8_t dst, uint32_t disp) {           // mov dst64, [rsp+disp]
    b.b(uint8_t(0x48 | (dst >= 8 ? 4 : 0))); b.b(0x8B); mem_rsp(b, dst, disp);
}
void mov_mem_reg(Buf& b, uint32_t disp, uint8_t src) {           // mov [rsp+disp], src64
    b.b(uint8_t(0x48 | (src >= 8 ? 4 : 0))); b.b(0x89); mem_rsp(b, src, disp);
}
void mov_reg_reg(Buf& b, uint8_t dst, uint8_t src) {             // mov dst64, src64
    b.b(uint8_t(0x48 | (src >= 8 ? 4 : 0) | (dst >= 8 ? 1 : 0)));
    b.b(0x89); b.b(uint8_t(0xC0 | ((src & 7) << 3) | (dst & 7)));
}
void movsd_xmm_mem(Buf& b, uint8_t xmm, uint32_t disp) {         // movsd xmm, [rsp+disp]
    b.b(0xF2); if (xmm >= 8) b.b(0x44); b.b(0x0F); b.b(0x10); mem_rsp(b, xmm, disp);
}
void movsd_mem_xmm(Buf& b, uint32_t disp, uint8_t xmm) {         // movsd [rsp+disp], xmm
    b.b(0xF2); if (xmm >= 8) b.b(0x44); b.b(0x0F); b.b(0x11); mem_rsp(b, xmm, disp);
}
void movaps_xmm_xmm(Buf& b, uint8_t dst, uint8_t src) {          // movaps dst, src
    if (dst >= 8 || src >= 8) b.b(uint8_t(0x40 | (dst >= 8 ? 4 : 0) | (src >= 8 ? 1 : 0)));
    b.b(0x0F); b.b(0x28); b.b(uint8_t(0xC0 | ((dst & 7) << 3) | (src & 7)));
}
void sub_rsp(Buf& b, uint32_t imm) {
    b.b(0x48);
    if (imm < 0x80) { b.b(0x83); b.b(0xEC); b.b(uint8_t(imm)); }
    else            { b.b(0x81); b.b(0xEC); b.u32(imm); }
}
void add_rsp(Buf& b, uint32_t imm) {
    b.b(0x48);
    if (imm < 0x80) { b.b(0x83); b.b(0xC4); b.b(uint8_t(imm)); }
    else            { b.b(0x81); b.b(0xC4); b.u32(imm); }
}
void movabs_rax(Buf& b, uint64_t v) { b.b(0x48); b.b(0xB8); b.u64(v); }   // movabs rax, imm64
void call_rax(Buf& b)               { b.b(0xFF); b.b(0xD0); }
void movabs_r11(Buf& b, uint64_t v) { b.b(0x49); b.b(0xBB); b.u64(v); }   // movabs r11, imm64
void call_r11(Buf& b)               { b.b(0x41); b.b(0xFF); b.b(0xD3); }

// The Microsoft outgoing argument area, in bytes: four home slots minimum, one 8-byte slot per
// argument beyond that.
uint32_t outgoing_bytes(const CallSignature& sig) {
    return 8u * std::max<unsigned>(4u, sig.count);
}

// The stub's own frame, in bytes. Above the outgoing area sit two save slots the return checkpoint
// needs (rax, and xmm0 when the handler returns one). The guest entered with rsp%16==8 (the call
// pushed a return address onto a 16-aligned stack), and Microsoft x64 requires rsp%16==0 at a call
// site, so the frame itself must be 8 mod 16.
uint32_t frame_bytes(const CallSignature& sig) {
    uint32_t frame = outgoing_bytes(sig) + 16;
    if (frame % 16 != 8) frame += 8;
    return frame;
}

} // namespace

ArgLocation sysv_arg_location(const CallSignature& sig, unsigned arg) {
    unsigned ints = 0, sses = 0, spilled = 0;
    for (unsigned i = 0; i <= arg && i < sig.count; ++i) {
        const bool sse = sig.arg_class(i) == ArgClass::Sse;
        const bool in_reg = sse ? (sses < kSysvSseRegs) : (ints < 6);
        if (i == arg) {
            if (!in_reg) return { ArgLocation::Kind::Stack, 0, uint8_t(spilled) };
            return sse ? ArgLocation{ ArgLocation::Kind::SseReg, uint8_t(sses), 0 }
                       : ArgLocation{ ArgLocation::Kind::IntReg, kSysvIntRegs[ints], 0 };
        }
        if (!in_reg) ++spilled;
        else if (sse) ++sses;
        else ++ints;
    }
    return {};   // past the end of the signature -> Kind::None
}

ArgLocation ms_arg_location(const CallSignature& sig, unsigned arg) {
    if (arg >= sig.count) return {};   // past the end of the signature -> Kind::None
    if (arg >= 4) return { ArgLocation::Kind::Stack, 0, uint8_t(arg) };
    return sig.arg_class(arg) == ArgClass::Sse
               ? ArgLocation{ ArgLocation::Kind::SseReg, uint8_t(arg), 0 }
               : ArgLocation{ ArgLocation::Kind::IntReg, kMsIntRegs[arg], 0 };
}

size_t emit_legacy_integer_prologue(uint8_t* out) {
    // Guest a0..a9 -> Microsoft a0..a9, integer registers only. Hand-scheduled rather than produced
    // by the general placer below because it is the path all ~700 integer/pointer handlers take and
    // it has to stay inside the stub slot: the four guest stack words are re-pushed with a two-byte
    // memory push each, and the register moves run in descending destination order so a register is
    // never overwritten before it is read.
    static const uint8_t seq[] = {
        0x50,                              // push rax             ; alignment pad
        0xFF,0x74,0x24,0x28,               // push [rsp+0x28]      ; original guest a9
        0xFF,0x74,0x24,0x28,               // push [rsp+0x28]      ; original guest a8
        0xFF,0x74,0x24,0x28,               // push [rsp+0x28]      ; original guest a7
        0xFF,0x74,0x24,0x28,               // push [rsp+0x28]      ; original guest a6
        0x48,0x83,0xEC,0x30,               // sub rsp,0x30         ; shadow + a4/a5
        0x4C,0x89,0x44,0x24,0x20,          // mov [rsp+0x20],r8    ; MS 5th arg = a4
        0x4C,0x89,0x4C,0x24,0x28,          // mov [rsp+0x28],r9    ; MS 6th arg = a5
        0x49,0x89,0xC9,                    // mov r9,rcx           ; MS 4th = a3
        0x49,0x89,0xD0,                    // mov r8,rdx           ; MS 3rd = a2
        0x48,0x89,0xF2,                    // mov rdx,rsi          ; MS 2nd = a1
        0x48,0x89,0xF9,                    // mov rcx,rdi          ; MS 1st = a0
    };
    std::memcpy(out, seq, sizeof seq);
    return sizeof seq;
}

size_t emit_guest_abi_tailjump(uint8_t* out, uint64_t handler) {
    Buf b{ out };
    movabs_rax(b, handler);
    b.b(0xFF); b.b(0xE0);          // jmp rax
    return b.n;
}

size_t emit_sysv_to_ms_bridge(uint8_t* out, const BridgeParams& params) {
    Buf b{ out };
    const CallSignature& sig = params.signature;

    // A handler compiled in the guest's own convention needs no conversion at all — the guest's frame
    // is already the one it reads, including a variadic frame no CallSignature could have described
    // (#3246). Checked before the signature, because the two are mutually exclusive: a guest-ABI
    // handler's arguments are not the bridge's business, and a declared signature would only mislead.
    if (params.guest_abi) return emit_guest_abi_tailjump(out, params.handler);

    if (!sig.needs_conversion()) {
        // Historical path, byte for byte. The pad + four copied guest stack words + the 0x30 outgoing
        // area consume 0x58 bytes (8 mod 16), so a normally-entered stub reaches the handler call
        // with rsp%16==0.
        b.n += emit_legacy_integer_prologue(out);
        movabs_rax(b, params.handler);
        call_rax(b);
        mov_mem_reg(b, 0x20, kRax);                       // stash the return value across the checkpoint
        movabs_r11(b, params.checkpoint);
        call_r11(b);
        if (params.return_hook) { movabs_r11(b, params.return_hook); call_r11(b); }
        mov_reg_mem(b, kRax, 0x20);
        add_rsp(b, 0x58);
        b.b(0xC3);
        return b.n;
    }

    // Signature-driven path. Reached only by a handler that declares a floating-point argument or
    // return, i.e. one the historical path cannot place.
    //
    // Refuse a signature wider than the emitter is sized for, rather than writing past the caller's
    // staging buffer. signature_of rejects this at compile time, but register_fn_with_signature
    // takes a raw CallSignature, so the guard has to live next to the buffer it protects. An
    // impossible length is the fail-visible answer: install_stubs compares it against the slot,
    // reports "generated Windows ABI bridge exceeds stub_size", and nothing has been written.
    if (sig.count > kMaxArgs) return kMaxBridgeBytes + 1;

    const uint32_t frame = frame_bytes(sig);
    const uint32_t rax_slot = outgoing_bytes(sig);
    const uint32_t xmm_slot = rax_slot + 8;
    // Where the guest left argument words it spilled: past this frame and past the return address.
    const auto guest_stack_off = [&](unsigned spilled_slot) { return frame + 8 + 8u * spilled_slot; };

    sub_rsp(b, frame);

    // Place arguments in DESCENDING position. Two properties make that sufficient with no scratch
    // spill and no cycle breaking:
    //   * Every stack destination (position >= 4) is emitted before any register destination, so a
    //     register source is still intact when a stack slot reads it. The outgoing area lies BELOW
    //     the guest's own spilled words, so a store can never clobber a stack source either.
    //   * A register destination is never a source of a still-pending move. Pending moves have a
    //     smaller position i', and read SysV register index n' <= i'. The Microsoft destination
    //     registers are rcx=SysV[3], rdx=SysV[2], r8=SysV[4], r9=SysV[5] at positions 0,1,2,3 — each
    //     needs n' >= 2 while i' < 3, so n' <= i' cannot hold. The xmm file is simpler: an SSE
    //     argument's SysV index never exceeds its position, so a destination xmm is only ever the
    //     source of a move at an EQUAL or LOWER position, and the equal case is the identity.
    for (unsigned i = sig.count; i-- > 0;) {
        const ArgLocation src = sysv_arg_location(sig, i);
        const ArgLocation dst = ms_arg_location(sig, i);
        const uint32_t src_off = src.kind == ArgLocation::Kind::Stack ? guest_stack_off(src.slot) : 0;
        switch (dst.kind) {
        case ArgLocation::Kind::None:
            break;   // unreachable: the loop never runs past sig.count
        case ArgLocation::Kind::IntReg:
            if (src.kind == ArgLocation::Kind::Stack) mov_reg_mem(b, dst.reg, src_off);
            else if (src.reg != dst.reg)              mov_reg_reg(b, dst.reg, src.reg);
            break;
        case ArgLocation::Kind::SseReg:
            if (src.kind == ArgLocation::Kind::Stack) movsd_xmm_mem(b, dst.reg, src_off);
            else if (src.reg != dst.reg)              movaps_xmm_xmm(b, dst.reg, src.reg);
            break;
        case ArgLocation::Kind::Stack: {
            const uint32_t dst_off = 8u * dst.slot;
            if (src.kind == ArgLocation::Kind::SseReg) movsd_mem_xmm(b, dst_off, src.reg);
            else if (src.kind == ArgLocation::Kind::IntReg) mov_mem_reg(b, dst_off, src.reg);
            else { mov_reg_mem(b, kRax, src_off); mov_mem_reg(b, dst_off, kRax); }
            break;
        }
        }
    }

    movabs_rax(b, params.handler);
    call_rax(b);
    // The checkpoint (and the optional return hook) are ordinary host calls, free to clobber every
    // volatile register — which on Microsoft x64 includes xmm0. A handler that returns a float or a
    // double leaves its result there, so save it alongside rax or the guest reads whatever the
    // checkpoint happened to leave behind.
    mov_mem_reg(b, rax_slot, kRax);
    if (sig.sse_return) movsd_mem_xmm(b, xmm_slot, 0);
    movabs_r11(b, params.checkpoint);
    call_r11(b);
    if (params.return_hook) { movabs_r11(b, params.return_hook); call_r11(b); }
    if (sig.sse_return) movsd_xmm_mem(b, 0, xmm_slot);
    mov_reg_mem(b, kRax, rax_slot);
    add_rsp(b, frame);
    b.b(0xC3);
    return b.n;
}

} // namespace prosper::abi
