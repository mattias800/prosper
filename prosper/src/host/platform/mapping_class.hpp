#pragma once
// mapping_class.hpp -- which KIND of host memory backs an address.
//
// Windows `GetWriteWatch` tracks private allocations only and rejects every section view with
// ERROR_INVALID_PARAMETER, so a fault-free write watch there is bounded by how much of the guest's
// memory is MEM_PRIVATE versus MEM_MAPPED. This answers that question straight from the OS, with no
// dependence on how the memory HLE tracks its own mappings. Used by the validation census
// (the [validation-census] exit summary).
#include <cstdint>

namespace prosper::host {

enum class MappingClass : int {
    Untracked = 0,    // not committed host memory (free or unmapped)
    Private = 1,      // MEM_PRIVATE: GetWriteWatch can track it
    MappedView = 2,   // MEM_MAPPED: a section view (direct memory); GetWriteWatch rejects it
    Other = 3,        // MEM_IMAGE, or a platform with no such distinction
};

MappingClass classify_host_mapping(uint64_t address);

}  // namespace prosper::host
