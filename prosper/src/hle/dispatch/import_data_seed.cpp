#include "import_data_seed.hpp"
#include "nid.hpp"

#include <cstring>

namespace prosper {

bool seed_import_data(const std::string& nid, uint8_t* slot, size_t size) {
    if (!slot) return false;
    // Hashed at first use rather than written as a literal, for the reason nid_census's --self-check
    // exists: a hard-coded 11-character string is unverifiable by inspection, while `nid_hash` is
    // the same function the loader resolves imports with. If the two ever disagreed, everything
    // would disagree together instead of this one table silently missing.
    static const std::string kStackChkGuard = nid_hash("__stack_chk_guard");
    if (nid == kStackChkGuard && size >= sizeof(uint64_t)) {
        const uint64_t v = kStackGuardValue;
        memcpy(slot, &v, sizeof v);
        return true;
    }
    return false;
}

} // namespace prosper
