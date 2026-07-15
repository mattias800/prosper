#pragma once

#include <cstdint>

namespace prosper {

// Recover the guest %fs saved by exec_image_linux.cpp's guest import-stub path. The entry shim
// passes the HLE entry %rsp here: return-to-stub at +0, forwarded args7/8/9 at +8/+0x10/+0x18,
// an alignment pad at +0x20, saved r11 (guest %fs) at +0x28, and the guest return address at +0x30.
// Host-context tail calls have no such frame and return 0.
inline uint64_t callback_guest_fs_from_entry_stack(uint64_t entry_rsp) {
    uint64_t shim_ret = entry_rsp ? *(const uint64_t*)(uintptr_t)entry_rsp : 0;
    if (shim_ret < 0x600000000ull || shim_ret >= 0x700000000ull) return 0;

    uint64_t fs = *(const uint64_t*)(uintptr_t)(entry_rsp + 0x28);
    // Match the import stub's TCB check before installing the value as %fs. This makes a malformed
    // frame fail closed instead of interpreting a forwarded argument as a thread pointer.
    if (fs < 0x10000 || (fs & 0xfull) || *(const uint64_t*)(uintptr_t)fs != fs ||
        *(const uint32_t*)(uintptr_t)(fs + 0x108) != 0x50524F53u)
        return 0;
    return fs;
}

} // namespace prosper
