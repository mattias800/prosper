// mapping_class_posix.cpp -- non-Windows backend of mapping_class.hpp. Compiled everywhere except
// Windows. These platforms have the real page-protection write watch, so the private/section split
// the census exists to measure does not arise and every address is `Other`.
#include "host/platform/mapping_class.hpp"

namespace prosper::host {

MappingClass classify_host_mapping(uint64_t /*address*/) { return MappingClass::Other; }

}  // namespace prosper::host
