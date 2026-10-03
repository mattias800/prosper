#include "host/memory/committed_section.hpp"

#include <windows.h>

#include <algorithm>

namespace prosper::host {

bool committed_section_range(uint64_t address, uint64_t bytes, uint64_t view_base) {
    if (address < 0x1000 || !bytes || address > UINT64_MAX - bytes || !view_base) return false;
    const uint64_t end = address + bytes;
    while (address < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(address)), &info,
                         sizeof(info)) != sizeof(info) ||
            info.State != MEM_COMMIT || info.Type != MEM_MAPPED ||
            reinterpret_cast<uintptr_t>(info.AllocationBase) != view_base ||
            (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
            (info.AllocationProtect & (PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY)))
            return false;
        // VirtualQuery keeps reporting MEM_MAPPED after COW becomes private. Both current and
        // original COW refuse, but this is not a history detector: HLE producers never install COW.
        const DWORD access = info.Protect & 0xffu;
        if (access != PAGE_READONLY && access != PAGE_READWRITE && access != PAGE_EXECUTE_READ &&
            access != PAGE_EXECUTE_READWRITE)
            return false;
        const uint64_t begin = reinterpret_cast<uintptr_t>(info.BaseAddress);
        if (!info.RegionSize || begin > address || begin > UINT64_MAX - info.RegionSize ||
            begin + info.RegionSize <= address)
            return false;
        address = std::min(end, begin + info.RegionSize);
    }
    return true;
}

}   // namespace prosper::host
