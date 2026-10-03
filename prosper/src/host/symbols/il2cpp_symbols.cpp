// il2cpp_symbols.cpp — see il2cpp_symbols.hpp for the format and for what this deliberately omits.
#include "host/symbols/il2cpp_symbols.hpp"

#include "host/image/boot_program.hpp"   // BOOT_IL2CPP / BOOT_PSNCORE: the IL2CPP module's guest aperture

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
#include <string>
#include <vector>

namespace prosper {
namespace il2cpp {
namespace {

constexpr const char* kMagic = "prosper-il2cpp-symtab v1";
constexpr const char* kEnvVar = "PROSPER_IL2CPP_SYMBOLS";

struct Entry {
    uint64_t rva;
    std::string name;
    size_t candidate_count = 1;
};

struct Table {
    std::vector<Entry> entries;   // non-decreasing by (rva, name); both halves enforced (see below)
    uint64_t window = 0;
};

enum class BoundsState { Unknown, Absent, Present };

std::mutex g_mutex;
std::shared_ptr<const Table> g_table;      // guarded by g_mutex
SymbolTableStatus g_status;                // guarded by g_mutex
bool g_env_probe_done = false;             // guarded by g_mutex
BoundsState g_bounds_state = BoundsState::Unknown;   // guarded by g_mutex
LoadedModuleBounds g_bounds{};                     // guarded by g_mutex
uint64_t g_generation = 0;                          // guarded by g_mutex

// Called under g_mutex. Sorted entries make the last RVA the only upper-bound check needed.
std::string bounds_error(const Table& table) {
    if (g_bounds_state == BoundsState::Absent) return "no IL2CPP module is loaded";
    if (g_bounds_state != BoundsState::Present || table.entries.back().rva < g_bounds.max_rva)
        return {};
    std::ostringstream why;
    why << "highest symbol RVA 0x" << std::hex << table.entries.back().rva
        << " is outside loaded IL2CPP RVA range [0x" << g_bounds.min_rva
        << ", 0x" << g_bounds.max_rva << ")";
    return why.str();
}

struct Snapshot {
    std::shared_ptr<const Table> table;
    bool attempted;
    BoundsState bounds_state;
    LoadedModuleBounds bounds;
};

Snapshot symbol_snapshot() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return {g_table, g_status.attempted, g_bounds_state, g_bounds};
}

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

namespace {

bool load_symbol_table_for_generation(const std::string& path, std::string* err,
                                      uint64_t generation) {
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
                                   "' -- produce this file with "
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
            //
            // BOTH halves of that key are checked, because only the rva half used to be. The header
            // in il2cpp_symbols.hpp promised (rva, name) while a file whose tied entries were in any
            // order at all was accepted -- so the one situation the tie rule exists for was the one
            // situation nothing verified. It matters because resolve() answers a shared rva with the
            // LAST entry of the group: reorder a tie group and the lookup returns a different, and
            // equally plausible-looking, method name.
            //
            // Comparing the names as bytes here matches resolve.py comparing them by code point:
            // std::char_traits<char> orders as unsigned char, and UTF-8 is order-preserving, so
            // byte order and code-point order coincide for every name either side can write.
            if (!table->entries.empty()) {
                const Entry& prev = table->entries.back();
                const bool ordered = entry.rva > prev.rva ||
                                     (entry.rva == prev.rva && !(entry.name < prev.name));
                if (!ordered) {
                    std::ostringstream why;
                    why << "entries are not sorted by (rva, name) (line " << line_no << ")";
                    status.error = why.str();
                    break;
                }
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
        // Count raw records once; identical duplicate rows are not deduplicated.
        for (size_t first = 0; first < table->entries.size();) {
            size_t end = first + 1;
            while (end < table->entries.size() &&
                   table->entries[end].rva == table->entries[first].rva)
                ++end;
            for (size_t i = first; i < end; ++i) table->entries[i].candidate_count = end - first;
            first = end;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (generation != g_generation) {
            status.error = "symbol load superseded by session reset";
        } else {
            if (status.error.empty()) status.error = bounds_error(*table);
            if (status.error.empty()) {
                status.loaded = true;
                status.count = table->entries.size();
                status.window = table->window;
            } else {
                table.reset();
            }
            g_table = table;
            g_status = status;
            g_env_probe_done = true;   // an explicit load supersedes the environment probe
        }
    }
    announce(status);
    if (err) *err = status.error;
    return status.loaded;
}

bool load_from_env(bool once_only) {
    uint64_t generation;
    std::string path;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_table) return true;
        if (once_only && g_env_probe_done) return false;
        generation = g_generation;
        const char* env_path = std::getenv(kEnvVar);
        if (env_path) path = env_path;   // copy the live value before releasing the session lock
        g_env_probe_done = true;
    }
    return !path.empty() && load_symbol_table_for_generation(path, nullptr, generation);
}

Resolution resolve_snapshot_rva(const Snapshot& snapshot, uint64_t rva) {
    Resolution out;
    if (snapshot.bounds_state == BoundsState::Absent ||
        (snapshot.bounds_state == BoundsState::Present &&
         (rva < snapshot.bounds.min_rva || rva >= snapshot.bounds.max_rva))) {
        out.state = ResolveState::OutsideModule;
        return out;
    }
    const auto& table = snapshot.table;
    if (!table) {
        out.state = snapshot.attempted ? ResolveState::Unavailable : ResolveState::NotConfigured;
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
    out.candidate_count = it->candidate_count;
    return out;
}

}  // namespace

bool load_symbol_table(const std::string& path, std::string* err) {
    uint64_t generation;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        generation = g_generation;
    }
    return load_symbol_table_for_generation(path, err, generation);
}

bool load_symbol_table_from_env() {
    return load_from_env(false);
}

void ensure_symbol_table_loaded() {
    load_from_env(true);
}

void clear_symbol_table() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_generation;
    g_table.reset();
    g_status = SymbolTableStatus{};
    g_env_probe_done = false;
    g_bounds_state = BoundsState::Unknown;
    g_bounds = {};
}

void publish_loaded_module_bounds(std::optional<LoadedModuleBounds> bounds) {
    SymbolTableStatus rejected;
    bool invalidated = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (bounds && bounds->min_rva < bounds->max_rva) {
            g_bounds_state = BoundsState::Present;
            g_bounds = *bounds;
        } else {
            g_bounds_state = BoundsState::Absent;
            g_bounds = {};
        }
        if (g_table) {
            const std::string error = bounds_error(*g_table);
            if (!error.empty()) {
                g_table.reset();
                g_status.loaded = false;
                g_status.count = 0;
                g_status.window = 0;
                g_status.error = error;
                rejected = g_status;
                invalidated = true;
            }
        }
    }
    if (invalidated) announce(rejected);
}

SymbolTableStatus symbol_table_status() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_status;
}

Resolution resolve_rva(uint64_t rva) {
    return resolve_snapshot_rva(symbol_snapshot(), rva);
}

Resolution resolve_guest_va(uint64_t va) {
    const Snapshot snapshot = symbol_snapshot();
    if (va < BOOT_IL2CPP ||
        (snapshot.bounds_state == BoundsState::Unknown && va >= BOOT_PSNCORE)) {
        Resolution out;
        out.state = ResolveState::OutsideModule;
        return out;
    }
    return resolve_snapshot_rva(snapshot, va - BOOT_IL2CPP);
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
            if (resolution.candidate_count > 1)
                text += " (+" + std::to_string(resolution.candidate_count - 1) +
                        " more at this address)";
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
