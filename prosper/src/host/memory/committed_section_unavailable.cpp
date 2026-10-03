#include "host/memory/committed_section.hpp"

namespace prosper::host {

bool committed_section_range(uint64_t, uint64_t, uint64_t) {
    return false;
}

} // namespace prosper::host
