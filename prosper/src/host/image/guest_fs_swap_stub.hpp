#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace prosper {

// Emit the Linux/macOS import stub that calls a host handler with the host FS base, then restores
// the guest FS base. Kept in a header so its machine-code contract can be checked without executing
// instructions that require FSGSBASE support.
inline size_t emit_guest_fs_swap_stub(uint8_t* p, uint32_t idx, uint64_t fn, bool unimpl) {
    uint8_t* s = p;
    if (unimpl) { *p++ = 0xBF; std::memcpy(p, &idx, 4); p += 4; }             // mov edi, idx
    *p++ = 0x49; *p++ = 0xBA; std::memcpy(p, &fn, 8); p += 8;                // movabs r10, fn
    *p++=0xF3; *p++=0x49; *p++=0x0F; *p++=0xAE; *p++=0xC3;
    *p++=0x41; *p++=0x81; *p++=0xBB; uint32_t mo=0x108; std::memcpy(p,&mo,4); p+=4;
    uint32_t magic=0x50524F53u; std::memcpy(p,&magic,4); p+=4;
    *p++=0x75; uint8_t* jne_rel = p++;
    // Save r11 and re-push args 10/9/8/7. Five qwords align the handler call site; the saved r11
    // is at [handler_rsp+0x28].
    *p++=0x41; *p++=0x53;
    *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;
    *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;
    *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;
    *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;
    *p++=0x49; *p++=0x8B; *p++=0x83; uint32_t ho=0x100; std::memcpy(p,&ho,4); p+=4;
    *p++=0xF3; *p++=0x48; *p++=0x0F; *p++=0xAE; *p++=0xD0;
    *p++=0x41; *p++=0xFF; *p++=0xD2;
    *p++=0x48; *p++=0x83; *p++=0xC4; *p++=0x20;
    *p++=0x41; *p++=0x5B;
    *p++=0xF3; *p++=0x49; *p++=0x0F; *p++=0xAE; *p++=0xD3;
    *p++=0xC3;
    *jne_rel = static_cast<uint8_t>(p - (jne_rel + 1));
    *p++=0x41; *p++=0xFF; *p++=0xE2;
    return static_cast<size_t>(p - s);
}

} // namespace prosper
