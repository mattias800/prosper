// il2cpp_symbols.hpp — name the C# method containing a live guest PC, during the run (#2551).
//
// prosper could already do this OFFLINE: tools/il2cpp/prx_to_elf.py flattens the compiled-C# PRX,
// Il2CppDumper produces script.json, and tools/il2cpp/resolve.py maps an RVA to a method name. What
// was missing is the in-process form, so a fault backtrace prints
//
//     [app] guest backtrace: 0x4402140d0 (Il2cpp+0x2140d0 SonicBloom.Koreo.Koreographer$$GetMusicSampleTime+0xf0)
//
// instead of a bare address the reader has to go and correlate by hand, per crash, per title.
//
// WHAT THIS DELIBERATELY DOES NOT DO: it does not parse script.json, and it does not parse
// global-metadata.dat. Both would be re-derivations of work resolve.py already does correctly, and
// the IL2CPP metadata layout is version-dependent (prosper's titles are not all on one Unity
// version), so a parser validated against one title says little about another. Instead resolve.py
// emits a flat symbol table -- `resolve.py --emit-symtab` -- from its OWN loader, and this reads
// that. The two implementations therefore share the record set and the lookup rule by construction,
// and `il2cpp_symtab_agreement` (tools/il2cpp/test_symtab_agreement.py) pins that they agree
// address-for-address.
//
// FILE FORMAT (v1), produced by tools/il2cpp/resolve.py --emit-symtab:
//
//     prosper-il2cpp-symtab v1 window=0x8000 count=87851
//     # <free-form comment lines>
//     1e3e00 Some.Namespace.Type$$Method
//     1e3f10 Some.Namespace.Container<Key, Value>$$Insert
//
// One entry per line: the module RVA in lowercase hex (no 0x), one space, then the method name
// VERBATIM to end of line. The name may contain spaces -- 1,326 of PPSA24651's 87,851 methods do,
// because IL2CPP spells generic arguments `Foo<A, B>$$Bar` -- so a whitespace-tokenising parser
// silently truncates them. Entries are sorted by (rva, name), which is the order resolve.py's own
// bisect operates on.
//
// The magic line is REQUIRED and carries the acceptance `window=`: the file states its own lookup
// semantics rather than each side hard-coding a constant that can drift. A file without it is
// rejected loudly; it is almost certainly a raw script.json, and silently reading zero symbols out
// of one is exactly the "answered nothing when it meant it did not run" failure this guards.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper {
namespace il2cpp {

// Why a resolution produced no name. Kept as five distinct states, not a bool, because the whole
// value of the resolver depends on "I have no symbol table" being readable as different from "I
// looked and there is no managed method there".
enum class ResolveState {
    NotConfigured,   // no symbol table was ever requested (PROSPER_IL2CPP_SYMBOLS unset)
    Unavailable,     // one WAS requested and could not be loaded -- see symbol_table_status().error
    OutsideModule,   // the address is not in the IL2CPP module's guest aperture at all
    NoMatch,         // a table is loaded and no method covers this offset
    Resolved,        // `name` + `offset` are meaningful
};

// A stable lowercase token per state, for logs and tests: "not-configured", "unavailable",
// "outside-module", "no-managed-method", "resolved".
const char* resolve_state_token(ResolveState state);

struct Resolution {
    ResolveState state = ResolveState::NotConfigured;
    std::string name;      // only when state == Resolved
    uint64_t offset = 0;   // bytes past the method's first instruction; only when state == Resolved
};

struct SymbolTableStatus {
    bool attempted = false;     // some load was tried (successfully or not)
    bool loaded = false;        // a table is available right now
    std::string source;         // the path the last attempt used
    std::string error;          // why it is not loaded; empty exactly when loaded
    size_t count = 0;           // symbols available
    uint64_t window = 0;        // nearest-preceding acceptance window declared by the file
};

// Load `path`. Replaces whatever was loaded before. On both success and failure this prints one
// `[il2cpp-sym]` line naming the source and the symbol count -- always, because a resolver that
// can report neither what it loaded nor how much of it is a resolver whose silence is unreadable.
// Returns true on success; on failure *err (when non-null) gets the reason.
bool load_symbol_table(const std::string& path, std::string* err);

// Load from PROSPER_IL2CPP_SYMBOLS if it is set and non-empty. Reads the environment LIVE on every
// call (never PROSPER_ENV_VALUE): tools/env/check_cached_env.py refuses a cached read of any name a
// test arms with setenv, and test_il2cpp_symbols does exactly that. Returns true if a table is
// loaded when it returns. A table that is already loaded is kept and no re-read happens.
bool load_symbol_table_from_env();

// Run load_symbol_table_from_env() at most once. This is what the address-printing paths call, so
// the environment probe costs one getenv for the whole process rather than one per printed frame.
void ensure_symbol_table_loaded();

// Drop the table and re-arm the once-only probe. For tests.
void clear_symbol_table();

SymbolTableStatus symbol_table_status();

// Resolve a module-relative RVA (what script.json's `Address` field holds, and what a prosper
// `Il2cpp+0x<offset>` label prints).
Resolution resolve_rva(uint64_t rva);

// Resolve an absolute guest virtual address. Addresses outside the IL2CPP module's aperture return
// OutsideModule rather than being folded into some other module's offset space (#1659's lesson: a
// single wide range labelled every module in it "eboot+", i.e. wrong binary, not just wrong offset).
Resolution resolve_guest_va(uint64_t va);

// " Name+0x<offset>" / " <no-managed-method>" / " <il2cpp-symbols-unavailable>" / "" -- the suffix
// describe_code_address appends to an `Il2cpp+0x…` label. Empty exactly when no claim can be made
// (NotConfigured or OutsideModule), so an unconfigured run's diagnostics are byte-identical to
// today's.
std::string annotation_for_guest_va(uint64_t va);

}  // namespace il2cpp
}  // namespace prosper
