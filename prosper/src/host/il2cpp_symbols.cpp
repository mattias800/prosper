// il2cpp_symbols.cpp — see il2cpp_symbols.hpp for the format and for what this deliberately omits.
#include "il2cpp_symbols.hpp"

#include "boot_program.hpp"   // BOOT_IL2CPP / BOOT_PSNCORE: the IL2CPP module's guest aperture

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace prosper {
namespace il2cpp {
namespace {

constexpr const char* kMagic = "prosper-il2cpp-symtab v1";
constexpr const char* kEnvVar = "PROSPER_IL2CPP_SYMBOLS";

struct Entry {
    uint64_t rva;
    std::string name;
};

struct Table {
    std::vector<Entry> entries;   // non-decreasing by rva, ties in emitter order (see below)
    uint64_t window = 0;
};

std::mutex g_mutex;
std::shared_ptr<const Table> g_table;      // guarded by g_mutex
SymbolTableStatus g_status;                // guarded by g_mutex
bool g_env_probe_done = false;             // guarded by g_mutex

// Read the required `key=0x…` / `key=…` field out of the magic line. Returns false when the key is
// absent or unparseable, so a header that does not state its own semantics is a load failure rather
// than a silent default.
bool header_field(const std::string& line, const char* key, uint64_t* out) {
    const std::string needle = std::string(key) + "=";
    size_t at = line.find(needle);
    if (at == std::string::npos) return false;
    const char* start = line.c_str() + at + needle.size();
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(start, &end, 0);
    if (end == start || errno == ERANGE) return false;
    *out = (uint64_t)value;
    return true;
}

// The name may contain spaces (IL2CPP spells generics `Foo<A, B>$$Bar`), so everything after the
// first whitespace run is the name VERBATIM. Anything that tokenises here truncates 1.5% of
// PPSA24651's methods into names that still look plausible.
bool parse_entry(const std::string& line, Entry* out) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long rva = std::strtoull(line.c_str(), &end, 16);
    if (end == line.c_str() || errno == ERANGE) return false;
    size_t name_at = (size_t)(end - line.c_str());
    if (name_at >= line.size() || (line[name_at] != ' ' && line[name_at] != '\t')) return false;
    while (name_at < line.size() && (line[name_at] == ' ' || line[name_at] == '\t')) ++name_at;
    if (name_at >= line.size()) return false;
    out->rva = (uint64_t)rva;
    out->name = line.substr(name_at);
    return true;
}

// One line, every time, on success and on failure alike. A resolver whose only failure signal is
// producing no names is indistinguishable from one that ran and found nothing.
void announce(const SymbolTableStatus& status) {
    if (status.loaded)
        std::fprintf(stderr, "[il2cpp-sym] loaded %zu symbols from %s (window=0x%llx)\n",
                     status.count, status.source.c_str(), (unsigned long long)status.window);
    else
        std::fprintf(stderr, "[il2cpp-sym] NOT LOADED from %s: %s "
                             "(guest addresses stay unsymbolicated)\n",
                     status.source.c_str(), status.error.c_str());
}

}  // namespace

const char* resolve_state_token(ResolveState state) {
    switch (state) {
        case ResolveState::NotConfigured: return "not-configured";
        case ResolveState::Unavailable:   return "unavailable";
        case ResolveState::OutsideModule: return "outside-module";
        case ResolveState::NoMatch:       return "no-managed-method";
        case ResolveState::Resolved:      return "resolved";
    }
    return "unknown";
}

bool load_symbol_table(const std::string& path, std::string* err) {
    SymbolTableStatus status;
    status.attempted = true;
    status.source = path;
    std::shared_ptr<Table> table;

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        status.error = "cannot open file";
    } else {
        table = std::make_shared<Table>();
        std::string line;
        bool have_header = false;
        uint64_t declared_count = 0;
        bool have_declared_count = false;
        size_t line_no = 0;
        while (status.error.empty() && std::getline(in, line)) {
            ++line_no;
            if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF tolerance
            if (line.empty()) continue;
            if (!have_header) {
                if (line.compare(0, std::strlen(kMagic), kMagic) != 0) {
                    // The overwhelmingly likely mistake is pointing this at a raw script.json.
                    status.error = "first line is not '" + std::string(kMagic) +
                                   "' — produce this file with "
                                   "tools/il2cpp/resolve.py --emit-symtab <script.json> <out>";
                    break;
                }
                if (!header_field(line, "window", &table->window) || table->window == 0) {
                    status.error = "header has no usable window= field";
                    break;
                }
                have_declared_count = header_field(line, "count", &declared_count);
                have_header = true;
                continue;
            }
            if (line[0] == '#') continue;
            Entry entry;
            if (!parse_entry(line, &entry)) {
                std::ostringstream why;
                why << "malformed entry at line " << line_no << " (want '<hex-rva> <name>')";
                status.error = why.str();
                break;
            }
            // Non-decreasing is REQUIRED rather than repaired by sorting here. resolve.py bisects a
            // list sorted by (address, name) and takes the last entry at or below the query; if this
            // side re-sorted with any other tie rule the two implementations would disagree on
            // exactly the addresses where several methods share a start. Preserving emitter order
            // and refusing anything else keeps the agreement structural.
            if (!table->entries.empty() && entry.rva < table->entries.back().rva) {
                std::ostringstream why;
                why << "entries are not sorted by rva (line " << line_no << ")";
                status.error = why.str();
                break;
            }
            table->entries.push_back(std::move(entry));
        }
        if (status.error.empty() && !have_header)
            status.error = "empty file (no '" + std::string(kMagic) + "' header)";
        if (status.error.empty() && have_declared_count &&
            declared_count != (uint64_t)table->entries.size()) {
            std::ostringstream why;
            why << "header declares count=" << declared_count << " but " << table->entries.size()
                << " entries were read (truncated file?)";
            status.error = why.str();
        }
        if (status.error.empty() && table->entries.empty())
            status.error = "header is valid but the table has no entries";
    }

    if (status.error.empty()) {
        status.loaded = true;
        status.count = table->entries.size();
        status.window = table->window;
    } else {
        table.reset();
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_table = table;
        g_status = status;
        g_env_probe_done = true;   // an explicit load supersedes the environment probe
    }
    announce(status);
    if (err) *err = status.error;
    return status.loaded;
}

bool load_symbol_table_from_env() {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_table) return true;
    }
    const char* path = std::getenv(kEnvVar);
    if (!path || !*path) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_env_probe_done = true;
        return false;
    }
    return load_symbol_table(path, nullptr);
}

void ensure_symbol_table_loaded() {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_env_probe_done) return;
    }
    load_symbol_table_from_env();
}

void clear_symbol_table() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_table.reset();
    g_status = SymbolTableStatus{};
    g_env_probe_done = false;
}

SymbolTableStatus symbol_table_status() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_status;
}

Resolution resolve_rva(uint64_t rva) {
    std::shared_ptr<const Table> table;
    bool attempted = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        table = g_table;
        attempted = g_status.attempted;
    }
    Resolution out;
    if (!table) {
        out.state = attempted ? ResolveState::Unavailable : ResolveState::NotConfigured;
        return out;
    }
    // Last entry with entry.rva <= rva — the same record Python's `bisect_right(keys, off) - 1`
    // selects, including the tie rule (see the sort note in load_symbol_table).
    auto it = std::upper_bound(table->entries.begin(), table->entries.end(), rva,
                               [](uint64_t value, const Entry& e) { return value < e.rva; });
    if (it == table->entries.begin()) { out.state = ResolveState::NoMatch; return out; }
    --it;
    if (rva - it->rva >= table->window) { out.state = ResolveState::NoMatch; return out; }
    out.state = ResolveState::Resolved;
    out.name = it->name;
    out.offset = rva - it->rva;
    return out;
}

Resolution resolve_guest_va(uint64_t va) {
    if (va < BOOT_IL2CPP || va >= BOOT_PSNCORE) {
        Resolution out;
        out.state = ResolveState::OutsideModule;
        return out;
    }
    return resolve_rva(va - BOOT_IL2CPP);
}

std::string annotation_for_guest_va(uint64_t va) {
    ensure_symbol_table_loaded();
    const Resolution resolution = resolve_guest_va(va);
    switch (resolution.state) {
        case ResolveState::Resolved: {
            std::string text = " " + resolution.name;
            if (resolution.offset) {
                char suffix[32];
                std::snprintf(suffix, sizeof suffix, "+0x%llx",
                              (unsigned long long)resolution.offset);
                text += suffix;
            }
            return text;
        }
        case ResolveState::NoMatch:     return " <no-managed-method>";
        case ResolveState::Unavailable: return " <il2cpp-symbols-unavailable>";
        // No table was ever asked for, or the address is not IL2CPP code at all: say nothing, so a
        // default run's diagnostics are byte-for-byte what they were before this existed.
        case ResolveState::NotConfigured:
        case ResolveState::OutsideModule:
            break;
    }
    return std::string();
}

}  // namespace il2cpp
}  // namespace prosper
