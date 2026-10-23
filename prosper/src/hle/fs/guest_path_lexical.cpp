#include "hle/fs/guest_path_lexical_internal.hpp"
#include <stddef.h>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace prosper {
// clang-format off: preserve the byte-identical moved definitions.
    // Lexically normalize a virtual sub-path (the part AFTER a /app0|/temp0|/savedata0 root),
    // treating both '/' and '\\' as separators. Collapses "." and applies ".." within the sub-path.
    // Returns the normalized sub-path (leading '/', e.g. "/a/../b" -> "/b", "" for the root itself) if
    // it stays AT OR BELOW the root, or std::nullopt if a ".." climbs above it (a sandbox escape).
    // Purely lexical — no filesystem access. Only invoked when the sub-path contains "..", so a normal
    // path never reaches it and is composed byte-for-byte as before. CONFIDENCE: HIGH (pure string op).
    std::optional<std::string> sandbox_normalize_subpath(const std::string& sub) {
        std::vector<std::string> stack;
        size_t i = 0;
        auto is_sep = [](char c) { return c == '/' || c == '\\'; };
        while (i < sub.size()) {
            while (i < sub.size() && is_sep(sub[i])) i++;
            size_t j = i; while (j < sub.size() && !is_sep(sub[j])) j++;
            std::string comp = sub.substr(i, j - i);
            i = j;
            if (comp.empty() || comp == ".") continue;
            if (comp == "..") { if (stack.empty()) return std::nullopt; stack.pop_back(); continue; }
            stack.push_back(comp);
        }
        std::string out;
        for (const auto& c : stack) { out += '/'; out += c; }
        return out;
    }
    // Clamp-normalize a RELATIVE guest path (#1226): same lexical walk as sandbox_normalize_subpath,
    // but a ".." with nothing left to pop is dropped (the POSIX "/.." == "/" rule at the virtual
    // root) instead of failing. Only for rootless relative spellings — an ABSOLUTE guest path that
    // climbs above its mount stays a denied sandbox escape (#1205). Returns the normalized path
    // with no leading separator ("a/../../b/c" -> "b/c").
    std::string clamp_normalize_relative(const std::string& sub) {
        std::vector<std::string> stack;
        size_t i = 0;
        auto is_sep = [](char c) { return c == '/' || c == '\\'; };
        while (i < sub.size()) {
            while (i < sub.size() && is_sep(sub[i])) i++;
            size_t j = i; while (j < sub.size() && !is_sep(sub[j])) j++;
            std::string comp = sub.substr(i, j - i);
            i = j;
            if (comp.empty() || comp == ".") continue;
            if (comp == "..") { if (!stack.empty()) stack.pop_back(); continue; }
            stack.push_back(std::move(comp));
        }
        std::string out;
        for (const auto& c : stack) { if (!out.empty()) out += '/'; out += c; }
        return out;
    }
    bool mount_path_matches(const std::string& path, const char* mount) {
        const size_t len = std::strlen(mount);
        return path.compare(0, len, mount) == 0 &&
               (path.size() == len || path[len] == '/' || path[len] == '\\');
    }
    // A path component that is "." or "..", split on either guest separator.
    bool has_dot_component(const std::string& p) {
        size_t i = 0;
        while (i < p.size()) {
            while (i < p.size() && (p[i] == '/' || p[i] == '\\')) ++i;
            size_t j = i;
            while (j < p.size() && p[j] != '/' && p[j] != '\\') ++j;
            const size_t len = j - i;
            if ((len == 1 && p[i] == '.') || (len == 2 && p[i] == '.' && p[i + 1] == '.'))
                return true;
            i = j;
        }
        return false;
    }
// clang-format on
} // namespace prosper
