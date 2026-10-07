// mapping_class_win.cpp -- Windows backend of mapping_class.hpp. Compiled only on Windows (the build
// selects the backend by file; see the filter next to PROSPER_SRC in CMakeLists.txt).
#include "host/platform/mapping_class.hpp"

#include <windows.h>

namespace prosper::host {

MappingClass classify_host_mapping(uint64_t address) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(address)), &mbi, sizeof(mbi)) == 0 ||
        mbi.State == MEM_FREE)
        return MappingClass::Untracked;
    switch (mbi.Type) {
        case MEM_PRIVATE: return MappingClass::Private;
        case MEM_MAPPED:  return MappingClass::MappedView;
        default:          return MappingClass::Other;
    }
}

bool host_mapping_classification_informative() { return true; }

}  // namespace prosper::host
