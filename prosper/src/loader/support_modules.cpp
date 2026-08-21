// support_modules.cpp — see support_modules.hpp.

#include "loader/support_modules.hpp"

#include <algorithm>

namespace prosper {

std::string support_module_lib_name(const std::string& path) {
    std::string lib = path;
    if (const auto slash = lib.find_last_of("/\\"); slash != std::string::npos)
        lib = lib.substr(slash + 1);
    if (const auto dot = lib.find_last_of('.'); dot != std::string::npos && dot != 0)
        lib = lib.substr(0, dot);
    return lib;
}

std::vector<size_t> unimported_support_module_indices(
    const std::vector<LinkInput>& in,
    const std::vector<std::vector<std::string>>& imports_by_index) {
    std::vector<size_t> drop;

    // Nothing opted in: no work, and in particular no scan. The common case is every title that
    // ships none of these files.
    const bool any = std::any_of(in.begin(), in.end(),
                                 [](const LinkInput& e) { return e.only_if_imported; });
    if (!any) return drop;

    // Collect the imports of the NON-candidates only. Taking a candidate's own imports would let
    // two bundled support PRXs that import each other keep each other alive with no title code
    // wanting either.
    std::vector<std::string> imported;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i].only_if_imported) continue;
        if (i >= imports_by_index.size()) continue;
        imported.insert(imported.end(), imports_by_index[i].begin(), imports_by_index[i].end());
    }

    // Descending, so the caller's erase loop stays valid.
    for (size_t i = in.size(); i-- > 0; ) {
        if (!in[i].only_if_imported) continue;
        const std::string lib = support_module_lib_name(in[i].path);
        if (std::find(imported.begin(), imported.end(), lib) == imported.end())
            drop.push_back(i);
    }
    return drop;
}

} // namespace prosper
