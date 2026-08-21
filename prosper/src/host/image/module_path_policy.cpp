#include "host/image/module_path_policy.hpp"

#include <algorithm>
#include <cctype>

namespace prosper {
namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Normalize separators and collapse repeated slashes. Trailing slashes are dropped so a dump root
// given as ".../PPSA24651-app0/" compares equal to one given without.
std::string normalize(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        const char n = (c == '\\') ? '/' : c;
        if (n == '/' && !out.empty() && out.back() == '/') continue;
        out.push_back(n);
    }
    while (out.size() > 1 && out.back() == '/') out.pop_back();
    return out;
}

std::vector<std::string> split(const std::string& p) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : p) {
        if (c == '/') {
            if (!cur.empty()) parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) parts.push_back(cur);
    return parts;
}

// A rejected path gets a reason that says what the file appears to BE, not just that a string did
// not match. `fakelib` and a `libSce*` basename are the two signals that actually occur, and the
// difference between "prosper does not link from here" and "this is a replacement Sony library"
// is the difference between a puzzling log line and an actionable one.
std::string describe_rejection(const std::vector<std::string>& rel_parts) {
    const std::string dir = rel_parts.empty() ? std::string() : lower(rel_parts.front());
    const std::string base = rel_parts.empty() ? std::string() : rel_parts.back();
    const bool sony_named = lower(base).rfind("libsce", 0) == 0;

    if (dir == "fakelib") {
        return sony_named
            ? "third-party replacement of a Sony library, in `fakelib/`. prosper answers "
              "entitlement and add-content queries from its own local inventory and never delegates "
              "them to a bundled module"
            : "in `fakelib/`, a directory prosper never links from";
    }
    if (sony_named) {
        return "a Sony-named library outside `sce_module/`; prosper links Sony libraries only from "
               "the dump's own `sce_module/` directory";
    }
    return "in a directory prosper does not link modules from";
}

}  // namespace

ModulePathDecision classify_module_path(const std::string& dump_root, const std::string& path) {
    ModulePathDecision d;

    const std::string root = normalize(dump_root);
    const std::string full = normalize(path);

    if (root.empty() || full.empty()) {
        d.verdict = ModulePathVerdict::OutsideDumpRoot;
        d.reason = "empty dump root or module path";
        return d;
    }

    // Must sit strictly under the root. Compare on a component boundary so `/dump-evil/x` is not
    // accepted as living under `/dump`.
    if (full.size() <= root.size() || full.compare(0, root.size(), root) != 0 ||
        full[root.size()] != '/') {
        d.verdict = ModulePathVerdict::OutsideDumpRoot;
        d.reason = "not inside the dump directory";
        return d;
    }

    const std::vector<std::string> parts = split(full.substr(root.size() + 1));
    if (parts.empty()) {
        d.verdict = ModulePathVerdict::OutsideDumpRoot;
        d.reason = "resolves to the dump directory itself";
        return d;
    }
    // Refuse rather than resolve: `a/../../etc/passwd` escapes, and lexical rejection keeps this
    // function pure. `.` is refused for the same reason — it should never appear in a path the
    // loader built, and accepting it would mean two spellings of one location.
    for (const auto& p : parts) {
        if (p == ".." || p == ".") {
            d.verdict = ModulePathVerdict::OutsideDumpRoot;
            d.reason = "contains a `" + p + "` component";
            return d;
        }
    }

    if (parts.size() == 1) {
        for (const char* f : kPermittedRootFiles) {
            if (lower(parts[0]) == lower(f)) {
                d.verdict = ModulePathVerdict::Permitted;
                return d;
            }
        }
        d.verdict = ModulePathVerdict::DirectoryNotPermitted;
        d.reason = "in the dump root, which holds no linkable module other than eboot.bin";
        return d;
    }

    std::string dir;
    for (size_t i = 0; i + 1 < parts.size(); i++) {
        if (i) dir += '/';
        dir += parts[i];
    }
    for (const char* permitted : kPermittedModuleDirs) {
        if (lower(dir) != lower(permitted)) continue;
        // Permitted directory — but a Sony-named library is allowed only out of sce_module/.
        // See the kSonyLibraryPrefix comment in the header for why the Media/* directories are
        // held to a stricter rule than their being on the allowlist would suggest.
        const bool is_sce_module = lower(dir) == "sce_module";
        if (!is_sce_module && lower(parts.back()).rfind(kSonyLibraryPrefix, 0) == 0) {
            d.verdict = ModulePathVerdict::SonyLibraryOutsideSceModule;
            d.reason = "a Sony-named library under `" + dir +
                       "/`, which prosper auto-links wholesale; Sony libraries are linked only from "
                       "the dump's own `sce_module/` directory";
            return d;
        }
        d.verdict = ModulePathVerdict::Permitted;
        return d;
    }

    d.verdict = ModulePathVerdict::DirectoryNotPermitted;
    d.reason = describe_rejection(parts);
    return d;
}

std::vector<RejectedModule> enforce_module_path_policy(const std::string& dump_root,
                                                       std::vector<LinkInput>& in) {
    std::vector<RejectedModule> rejected;
    for (size_t i = in.size(); i-- > 0;) {
        const ModulePathDecision d = classify_module_path(dump_root, in[i].path);
        if (d.permitted()) continue;
        rejected.push_back({ in[i].path, d.reason });
        in.erase(in.begin() + (ptrdiff_t)i);
    }
    // Iterated backwards so erasing is safe; hand the caller the list in list order.
    std::reverse(rejected.begin(), rejected.end());
    return rejected;
}

}  // namespace prosper
