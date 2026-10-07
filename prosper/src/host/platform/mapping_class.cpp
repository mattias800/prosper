// mapping_class.cpp -- see mapping_class.hpp.
#include "host/platform/mapping_class.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace prosper::host {

MappingClass classify_host_mapping(uint64_t address) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(address)), &mbi, sizeof(mbi)) == 0 ||
        mbi.State == MEM_FREE)
        return MappingClass::Untracked;
    switch (mbi.Type) {
        case MEM_PRIVATE: return MappingClass::Private;
        case MEM_MAPPED:  return MappingClass::MappedView;
        default:          return MappingClass::Other;
    }
#else
    // Platforms with the real page-protection write watch need no such split.
    (void)address;
    return MappingClass::Other;
#endif
}

}  // namespace prosper::host
