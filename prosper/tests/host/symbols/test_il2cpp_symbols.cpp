// test_il2cpp_symbols.cpp — the runtime IL2CPP symbol resolver (#2551).
//
// Modes:
//   (no args)                       self-checking unit test; exit code is truth
//   --probe <symtab> <rva> [...]    print one machine-readable line per rva, for the cross-
//                                   implementation agreement test (tools/il2cpp/test_symtab_agreement.py)
//   --describe <guest-va> [...]     print describe_code_address() for each address, i.e. the REAL
//                                   production label a fault backtrace prints. Honors
//                                   PROSPER_IL2CPP_SYMBOLS, so this is how a live capture is checked
//                                   against a real title's symbol table by hand.
//   --boot-bounds <valid|oversized|absent>  one synthetic real boot per fresh process, without
//                                         imports, initializers, or executing the guest entry.
//
// Every assertion below is paired with a MUTATION ARM — an input differing in exactly the property
// under test, whose expected answer differs. An arm is only worth having if no other branch of the
// resolver could produce it, so each one names what it excludes.
#include "host/symbols/il2cpp_symbols.hpp"
#include "host/image/boot_program.hpp"
#include "host/image/exec_image.hpp"   // describe_code_address: the production label being symbolicated
#include "fixtures/synth_prx.h"
#include "fixtures/test_scratch.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace prosper::il2cpp;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) ++g_failures;
}

void check_eq(const std::string& got, const std::string& want, const std::string& what) {
    check(got == want, what + " (got \"" + got + "\", want \"" + want + "\")");
}

std::string write_temp(const std::string& name, const std::string& body) {
    std::ofstream out(name, std::ios::binary);
    out << body;
    out.close();
    return name;
}

// MinGW/MSVC have no setenv/unsetenv. `_putenv_s(name, "")` removes the variable there, which is
// what the "not configured" arm needs -- an empty value is also what load_symbol_table_from_env()
// treats as unset, so the two spellings agree.
void set_test_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}
void clear_test_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

std::string state_of(uint64_t rva) { return resolve_state_token(resolve_rva(rva).state); }

std::string resolved_name(uint64_t rva) {
    const Resolution r = resolve_rva(rva);
    return r.state == ResolveState::Resolved ? r.name : std::string("<unresolved>");
}

uint64_t resolved_offset(uint64_t rva) { return resolve_rva(rva).offset; }

// ---------------------------------------------------------------------------------------------

void test_nearest_preceding(const std::string& fixture) {
    check(load_symbol_table(fixture, nullptr), "fixture loads");
    check(symbol_table_status().count == 5,
          "fixture reports 5 symbols, got " + std::to_string(symbol_table_status().count));

    // Inside the first method.
    check_eq(resolved_name(0x100208), "Prosper.Fixture.Alpha$$Start", "0x100208 -> first method");
    check(resolved_offset(0x100208) == 0x8, "0x100208 offset is +0x8");
    // ARM: one byte BELOW the second method's start still belongs to the first. Distinguishes
    // "last entry <= query" from "closest entry by distance", which would hand 0x10030f to the
    // SECOND method (0x1 away) instead of the first (0x10f away).
    check_eq(resolved_name(0x10030f), "Prosper.Fixture.Alpha$$Start", "0x10030f -> still the first method");
    check(resolved_offset(0x10030f) == 0x10f, "0x10030f offset is +0x10f");
    // ARM: exactly AT the second method's start it switches, with offset 0. Distinguishes a
    // `<` bound from a `<=` one; an upper_bound off by one entry answers the first method here.
    check_eq(resolved_name(0x100310),
             "Prosper.Fixture.Container<Key, Value>$$Insert",
             "0x100310 -> second method");
    check(resolved_offset(0x100310) == 0, "0x100310 offset is +0x0");
}

void test_name_with_spaces() {
    // The whole name, verbatim to end of line.
    const std::string want =
        "Prosper.Fixture.Container<Key, Value>$$Insert";
    check_eq(resolved_name(0x100310), want, "generic name survives the space");
    // ARM: the exact string a whitespace-tokenising parser (sscanf "%s", istream >>) produces.
    // Nothing else in this table can yield it, and it is the failure that looks most like success.
    check(resolved_name(0x100310) != "Prosper.Fixture.Container<Key,",
          "the name is NOT truncated at the first space");
}

void test_window_boundary() {
    // 0x900000 is isolated; window is 0x8000.
    check_eq(resolved_name(0x900000 + 0x7fff), "Prosper.Fixture.Isolated$$Run", "+0x7fff is inside the window");
    // ARM: one byte further is NOT attributed to it. Distinguishes a windowed lookup from an
    // unbounded nearest-preceding one, which would happily claim every address up to 0xa00000.
    check_eq(state_of(0x900000 + 0x8000), "no-managed-method", "+0x8000 falls outside the window");
    check_eq(state_of(0x900000 + 0x100), "resolved", "+0x100 is inside the window");
}

void test_below_first_symbol() {
    // ARM pair: the first entry starts at 0x100200, so anything below it has no method at all.
    // Distinguishes "step back from upper_bound, refusing to step past begin()" from an
    // implementation that clamps to the first entry (which would name every low address).
    check_eq(state_of(0x1000), "no-managed-method", "an rva below every symbol does not resolve");
    check_eq(state_of(0x100200), "resolved", "the first symbol's own start does resolve");
}

void test_tie_rule() {
    // Two methods share 0xa00000. resolve.py bisects a (address, name)-sorted list and takes the
    // LAST at or below the query, so the tie resolves to the lexicographically later name.
    // ARM: asserting it is NOT the earlier name pins the direction; an implementation using
    // lower_bound, or one that re-sorted the table with a different tie rule, gives Prosper.Fixture.Tied$$Method.
    check_eq(resolved_name(0xa00000), "Prosper.Fixture.Tied$$MethodZzz", "a tied address takes the last entry");
    check(resolved_name(0xa00000) != "Prosper.Fixture.Tied$$Method", "a tied address is NOT the first entry");
}

void test_aperture() {
    const uint64_t rva = 0x100208;
    check(resolve_guest_va(prosper::BOOT_IL2CPP + rva).state == ResolveState::Resolved,
          "an address in the IL2CPP aperture resolves");
    check_eq(resolve_guest_va(prosper::BOOT_IL2CPP + rva).name, "Prosper.Fixture.Alpha$$Start",
             "...to the same method resolve_rva names");
    // ARM: the SAME offset in the eboot's aperture must not be symbolicated. #1659: a single wide
    // range labelled every module in it "eboot+", i.e. wrong binary, not merely wrong offset. An
    // implementation that masked or modulo'd the address instead of range-checking resolves this.
    check_eq(resolve_state_token(resolve_guest_va(prosper::BOOT_EBOOT + rva).state),
             "outside-module", "the same offset in the eboot aperture is NOT symbolicated");
    check_eq(resolve_state_token(resolve_guest_va(prosper::BOOT_PSNCORE + rva).state),
             "outside-module", "an address just above the aperture is NOT symbolicated");
}

void test_annotation_strings() {
    // Resolved at a method start: no "+0x0" noise.
    check_eq(annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x100200), " Prosper.Fixture.Alpha$$Start",
             "annotation at a method start omits the offset");
    // ARM: one byte in, the offset appears. Only the `resolution.offset != 0` branch produces it.
    check_eq(annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x100201), " Prosper.Fixture.Alpha$$Start+0x1",
             "annotation one byte in carries +0x1");
    // Outside the module: no claim at all, so a non-IL2CPP title's diagnostics are unchanged.
    check_eq(annotation_for_guest_va(prosper::BOOT_EBOOT + 0x1000), "",
             "an eboot address gets no annotation");
}

// The heart of the correctness bar: "cannot resolve" must be readable as different from "resolved
// to nothing". These three states are produced by three different situations and must never
// collapse into one string.
void test_three_negative_states(const std::string& fixture) {
    clear_symbol_table();
    const std::string not_configured_token = state_of(0x100208);
    const std::string not_configured_annotation =
        annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x100208);
    check_eq(not_configured_token, "not-configured", "no table was ever requested");
    check(!symbol_table_status().attempted, "status().attempted is false before any load");
    check_eq(not_configured_annotation, "", "an unconfigured run's output is unchanged");

    std::string err;
    const bool loaded = load_symbol_table("il2cpp_symtab_does_not_exist.symtab", &err);
    const std::string unavailable_token = state_of(0x100208);
    const std::string unavailable_annotation =
        annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x100208);
    check(!loaded, "a missing symbol file fails to load");
    check(!err.empty(), "...with a non-empty reason: \"" + err + "\"");
    check(symbol_table_status().attempted && !symbol_table_status().loaded,
          "status() reports attempted-but-not-loaded");
    check(symbol_table_status().count == 0, "...and zero symbols");
    check_eq(unavailable_token, "unavailable", "a failed load reads as unavailable");
    check_eq(unavailable_annotation, " <il2cpp-symbols-unavailable>",
             "...and says so at the point of use");

    check(load_symbol_table(fixture, nullptr), "fixture reloads");
    const std::string nomatch_token = state_of(0x1000);
    const std::string nomatch_annotation = annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x1000);
    check_eq(nomatch_token, "no-managed-method", "a loaded table with no covering method");
    check_eq(nomatch_annotation, " <no-managed-method>", "...says THAT, not silence");

    // ARM: the three are pairwise distinct, in both the token and the printed annotation. A
    // resolver that answered "" or "unknown" for all three would satisfy every individual
    // assertion above and fail here — this is the only check that can see the collapse.
    check(not_configured_token != unavailable_token && unavailable_token != nomatch_token &&
              not_configured_token != nomatch_token,
          "the three non-resolving states have three distinct tokens");
    check(not_configured_annotation != unavailable_annotation &&
              unavailable_annotation != nomatch_annotation &&
              not_configured_annotation != nomatch_annotation,
          "...and three distinct annotations");
}

void test_rejects_a_raw_script_json() {
    // The mistake this exists for: pointing PROSPER_IL2CPP_SYMBOLS at script.json itself. Reading
    // zero symbols out of it and reporting "no managed method" would be the exact failure the
    // three-state design is meant to prevent.
    const std::string path = write_temp("il2cpp_symtab_test_rawjson.symtab",
                                        "{\"ScriptMethod\":[{\"Address\":1980928,\"Name\":\"X\"}]}\n");
    std::string err;
    check(!load_symbol_table(path, &err), "a raw script.json is REFUSED");
    check(err.find("prosper-il2cpp-symtab v1") != std::string::npos,
          "...naming the header it wanted: \"" + err + "\"");
    check(symbol_table_status().count == 0, "...and loads zero symbols");

    // ARM: the same body BEHIND a valid header loads. This proves the refusal is the header check
    // and not "anything unusual fails" — without it, a resolver that rejected every file would pass.
    const std::string ok_path = write_temp(
        "il2cpp_symtab_test_rawjson_ok.symtab",
        "prosper-il2cpp-symtab v1 window=0x8000 count=1\n"
        "1e3e00 {\"ScriptMethod\":[{\"Address\":1980928,\"Name\":\"X\"}]}\n");
    check(load_symbol_table(ok_path, nullptr), "the same text behind a valid header loads");
    check(symbol_table_status().count == 1, "...as exactly one symbol");
    std::remove(path.c_str());
    std::remove(ok_path.c_str());
}

void test_rejects_truncation() {
    const std::string path = write_temp("il2cpp_symtab_test_short.symtab",
                                        "prosper-il2cpp-symtab v1 window=0x8000 count=3\n"
                                        "1000 A$$a\n"
                                        "2000 B$$b\n");
    std::string err;
    check(!load_symbol_table(path, &err), "a file with fewer entries than count= is REFUSED");
    check(err.find("count=3") != std::string::npos, "...quoting the mismatch: \"" + err + "\"");
    // ARM: the identical entries with an honest count load. Isolates the count check from the
    // parse; a resolver that rejected two-entry files for any other reason fails here.
    const std::string ok = write_temp("il2cpp_symtab_test_short_ok.symtab",
                                      "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                      "1000 A$$a\n"
                                      "2000 B$$b\n");
    check(load_symbol_table(ok, nullptr), "the same entries with count=2 load");
    check(symbol_table_status().count == 2, "...as two symbols");
    std::remove(path.c_str());
    std::remove(ok.c_str());
}

void test_rejects_unsorted() {
    const std::string path = write_temp("il2cpp_symtab_test_unsorted.symtab",
                                        "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                        "2000 B$$b\n"
                                        "1000 A$$a\n");
    std::string err;
    check(!load_symbol_table(path, &err), "an out-of-order table is REFUSED");
    check(err.find("sorted") != std::string::npos, "...saying why: \"" + err + "\"");
    // ARM: the same two entries in ascending order load and resolve. Without this, "refuses
    // everything" would pass the assertion above.
    const std::string ok = write_temp("il2cpp_symtab_test_sorted_ok.symtab",
                                      "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                      "1000 A$$a\n"
                                      "2000 B$$b\n");
    check(load_symbol_table(ok, nullptr), "the same entries in order load");
    check_eq(resolved_name(0x2000), "B$$b", "...and resolve");
    std::remove(path.c_str());
    std::remove(ok.c_str());
}

// The TIE half of the (rva, name) key, which the loader used to trust rather than check. This is
// the only place the tie rule is observable: resolve() answers a shared rva with the LAST entry of
// the group, so a reordered tie group returns a different, equally plausible method name — the
// failure mode #2514 is open about. Both directions are asserted, because "refuses everything" and
// "refuses the right thing" are the same result on the rejection arm alone.
void test_rejects_unsorted_ties() {
    const std::string path = write_temp("il2cpp_symtab_test_tieorder.symtab",
                                        "prosper-il2cpp-symtab v1 window=0x8000 count=3\n"
                                        "1000 A$$a\n"
                                        "2000 Zeta$$z\n"
                                        "2000 Alpha$$a\n");   // descending WITHIN the tie group
    std::string err;
    check(!load_symbol_table(path, &err), "a table whose tied entries are out of order is REFUSED");
    check(err.find("sorted") != std::string::npos, "...saying why: \"" + err + "\"");

    // ARM 1: the same three entries with the tie group ascending are accepted, and the answer at
    // the shared address is the LAST of the group — the property the rule protects.
    const std::string ok = write_temp("il2cpp_symtab_test_tieorder_ok.symtab",
                                      "prosper-il2cpp-symtab v1 window=0x8000 count=3\n"
                                      "1000 A$$a\n"
                                      "2000 Alpha$$a\n"
                                      "2000 Zeta$$z\n");
    check(load_symbol_table(ok, nullptr), "the same entries with the tie group in order load");
    check_eq(resolved_name(0x2000), "Zeta$$z", "...and a tied address resolves to the LAST entry");

    // ARM 2: EQUAL names at a shared rva are legal — resolve.py's sort is non-strict, so a duplicate
    // record must not be rejected. Without this arm the check above would also pass if the
    // comparison had been written as a strict `>`, which would refuse real emitter output.
    const std::string dup = write_temp("il2cpp_symtab_test_tieorder_dup.symtab",
                                       "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                       "2000 Same$$s\n"
                                       "2000 Same$$s\n");
    check(load_symbol_table(dup, nullptr), "two identical records at one rva are accepted");
    check_eq(resolved_name(0x2000), "Same$$s", "...and resolve");

    // ARM 3: the SIGNEDNESS discriminator, and it is the only arm here that can fail for a reason
    // the others cannot see. resolve.py sorts names by Unicode code point; this side compares
    // std::string, i.e. bytes. Those agree only because UTF-8 is order-preserving AND
    // std::char_traits<char> orders as UNSIGNED char. If the byte comparison were signed, every
    // non-ASCII lead byte (>= 0x80) would sort as negative — before all ASCII — and the two sides
    // would disagree about exactly the names this project already has.
    //
    // "Z" is U+005A and "Ä" is U+00C4, so code-point order puts Z first and this file is correctly
    // ascending: it must LOAD. Under signed bytes 0xC3 reads as -61 < 'Z', the pair looks
    // descending, and it would be refused. So a pass here means unsigned, and a failure names the
    // cause instead of surfacing as a mystery rejection of a real title's table.
    //
    // Constructed by hand on purpose. PPSA24651 has 87,851 methods and ZERO tied addresses, so no
    // sample drawn from it can exercise the tie rule at all — the case is structurally
    // inexpressible there, and a control built from that data would have passed while testing
    // nothing.
    // The literals are split after every hex escape so no following character can be absorbed into
    // it: "\xC3\x84" is U+00C4 (A-umlaut) in UTF-8.
    const std::string utf8_tie = write_temp("il2cpp_symtab_test_tieorder_utf8.symtab",
                                            "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                            "2000 Z" "\xC3\x84" "$$ascii_first\n"
                                            "2000 " "\xC3\x84" "Z$$nonascii_second\n");
    std::string utf8_err;
    check(load_symbol_table(utf8_tie, &utf8_err),
          "a tie group ascending by CODE POINT loads -- the name comparison is unsigned, so it "
          "agrees with resolve.py (\"" + utf8_err + "\")");
    check_eq(resolved_name(0x2000), "\xC3\x84" "Z$$nonascii_second",
             "...and the last entry of that group is the answer");

    std::remove(path.c_str());
    std::remove(ok.c_str());
    std::remove(dup.c_str());
    std::remove(utf8_tie.c_str());
}

void test_rejects_missing_window() {
    const std::string path = write_temp("il2cpp_symtab_test_nowindow.symtab",
                                        "prosper-il2cpp-symtab v1 count=1\n"
                                        "1000 A$$a\n");
    std::string err;
    check(!load_symbol_table(path, &err), "a header without window= is REFUSED");
    check(err.find("window") != std::string::npos, "...saying so: \"" + err + "\"");
    // ARM: the window is READ from the file, not hard-coded here. A file declaring window=0x10
    // must stop resolving at +0x10 even though the production window is 0x8000 — no constant
    // baked into this side can produce that answer.
    const std::string tiny = write_temp("il2cpp_symtab_test_tinywindow.symtab",
                                        "prosper-il2cpp-symtab v1 window=0x10 count=1\n"
                                        "1000 A$$a\n");
    check(load_symbol_table(tiny, nullptr), "a window=0x10 table loads");
    check(symbol_table_status().window == 0x10, "...and status reports window=0x10");
    check_eq(state_of(0x100f), "resolved", "...resolving at +0xf");
    check_eq(state_of(0x1010), "no-managed-method", "...and stopping at +0x10");
    std::remove(path.c_str());
    std::remove(tiny.c_str());
}

void test_utf8_name_roundtrip() {
    // 8 of PPSA24651's 87,851 method names are non-ASCII. Written as explicit bytes so the test's
    // own source encoding cannot be what makes it pass.
    const std::string name = "N\xc3\xa4mespace.T\xc3\xbfpe$$M\xc3\xa9thod";
    const std::string path = write_temp("il2cpp_symtab_test_utf8.symtab",
                                        "prosper-il2cpp-symtab v1 window=0x8000 count=1\n"
                                        "1000 " + name + "\n");
    check(load_symbol_table(path, nullptr), "a table with a non-ASCII name loads");
    check(resolved_name(0x1000) == name, "...and the bytes round-trip exactly");
    // ARM: byte length, not character count — a parser that re-encoded or truncated at the first
    // high byte cannot produce this number.
    check(resolved_name(0x1000).size() == name.size(),
          "...with the same byte length (" + std::to_string(name.size()) + ")");
    std::remove(path.c_str());
}

void test_env_path(const std::string& fixture) {
    clear_symbol_table();
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    ensure_symbol_table_loaded();
    check(!symbol_table_status().attempted, "an unset PROSPER_IL2CPP_SYMBOLS attempts no load");
    check_eq(state_of(0x100208), "not-configured",
             "...and reads as not-configured, NOT as a failed load");

    clear_symbol_table();
    set_test_env("PROSPER_IL2CPP_SYMBOLS", fixture);
    ensure_symbol_table_loaded();
    // ARM against the arm above: the same call, the same code path, one environment variable
    // different, and now the table is present. Only the env branch can produce this pair.
    check(symbol_table_status().loaded, "PROSPER_IL2CPP_SYMBOLS pointing at the fixture loads it");
    check(symbol_table_status().count == 5, "...with all 5 symbols");
    check_eq(resolved_name(0x100208), "Prosper.Fixture.Alpha$$Start", "...and resolves through it");
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
}

// The wiring, not the resolver: this drives prosper::describe_code_address() — the single function
// every guest-address diagnostic already funnels through — and asserts the annotation reaches it.
// Without this the resolver could be perfect and the feature still absent at every call site.
void test_describe_code_address_wiring(const std::string& fixture) {
    clear_symbol_table();
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    // ARM (baseline): with no symbol table the label must be byte-identical to what prosper printed
    // before #2551. This is the whole "unconfigured runs are unchanged" contract, and only the
    // NotConfigured branch produces it — any of the other four states appends something.
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x100208), "Il2cpp+0x100208",
             "unconfigured: the label is unchanged");

    check(load_symbol_table(fixture, nullptr), "fixture loads for the wiring check");
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x100208),
             "Il2cpp+0x100208 Prosper.Fixture.Alpha$$Start+0x8",
             "configured: the same label now names the C# method");
    // A non-IL2CPP guest module keeps its bare label even with a table loaded.
    check_eq(prosper::describe_code_address(prosper::BOOT_EBOOT + 0x100208), "eboot+0x100208",
             "an eboot address is untouched by a loaded table");
    // And a covered-by-nothing IL2CPP address says so rather than going quiet.
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x10), "Il2cpp+0x10 <no-managed-method>",
             "an IL2CPP address with no method says so in the label");
}

// #2650: the first entry is shared between valid and oversized tables. A lookup inside the
// actual image must lose that plausible name when the LAST entry proves the table mismatched.
const char* const kBoundsInside = "Prosper.Bounds.Shared$$Inside";

std::string bounds_table(uint64_t first, uint64_t last, const char* edge) {
    std::ostringstream body;
    body << "prosper-il2cpp-symtab v1 window=0x8000 count=2\n" << std::hex
         << first << ' ' << kBoundsInside << '\n' << last << ' ' << edge << '\n';
    return body.str();
}

std::string write_bounds_table(const char* name, uint64_t first, uint64_t last,
                               const char* edge = "Prosper.Bounds.Edge$$Valid") {
    const std::string path = prosper_test::test_scratch_file(name);
    std::ofstream out(path, std::ios::binary);
    out << bounds_table(first, last, edge);
    out.close();
    check(!out.fail(), std::string("bounds fixture writes: ") + name);
    return path;
}

void check_bounds_refusal(const std::string& source, const std::string* caller_error,
                          const char* token, const std::string& arm) {
    const SymbolTableStatus status = symbol_table_status();
    check(status.attempted && !status.loaded && status.count == 0 && status.window == 0,
          arm + ": rejected status has no retained symbols");
    check_eq(status.source, source, arm + ": rejected status retains its source");
    check(status.error.find(token) != std::string::npos,
          arm + ": status names the loaded-module refusal");
    if (caller_error)
        check_eq(*caller_error, status.error, arm + ": caller receives the status error");
}

void test_loaded_bounds_reload() {
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    clear_symbol_table();
    publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
    const std::string valid = write_bounds_table("bounds-valid.symtab", 0x100, 0x3fff);
    const std::string oversized = write_bounds_table("bounds-oversized.symtab", 0x100, 0x4000);
    check(load_symbol_table(valid, nullptr), "loaded bounds: last byte is accepted");
    check_eq(resolved_name(0x108), kBoundsInside, "loaded bounds: shared low entry resolves");
    check(resolved_offset(0x108) == 8, "loaded bounds: shared low entry keeps its offset");
    check_eq(resolved_name(0x3fff), "Prosper.Bounds.Edge$$Valid",
             "loaded bounds: final admitted RVA resolves");

    std::string error;
    check(!load_symbol_table(oversized, &error), "loaded bounds: exclusive end is refused");
    check_bounds_refusal(oversized, &error, "loaded IL2CPP RVA range", "exclusive end");
    check(error.find("0x4000") != std::string::npos,
          "exclusive end: caller error includes the offending RVA");
    check_eq(state_of(0x108), "unavailable", "exclusive end: plausible low entry is unavailable");
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
             "Il2cpp+0x108 <il2cpp-symbols-unavailable>",
             "exclusive end: refusal reaches the production annotation");
    check_eq(state_of(0x4000), "outside-module", "loaded bounds: raw query at end is outside");
    check_eq(annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x4000), "",
             "loaded bounds: absolute query at end claims no name");
    check(load_symbol_table(valid, nullptr), "loaded bounds: valid reload recovers");
    check(symbol_table_status().count == 2 && symbol_table_status().error.empty(),
          "loaded bounds: recovery restores exactly the valid table");
    check_eq(resolved_name(0x108), kBoundsInside, "loaded bounds: recovery resolves the shared entry");
}

void test_bounds_publication_and_env() {
    const std::string valid = write_bounds_table("bounds-env-valid.symtab", 0x100, 0x3fff);
    const std::string oversized = write_bounds_table("bounds-env-oversized.symtab", 0x100, 0x4000);
    clear_symbol_table();
    set_test_env("PROSPER_IL2CPP_SYMBOLS", oversized);
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
             "Il2cpp+0x108 Prosper.Bounds.Shared$$Inside+0x8",
             "early env: Unknown session accepts an offline table");
    publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
    check_bounds_refusal(oversized, nullptr, "loaded IL2CPP RVA range", "early env publication");
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
             "Il2cpp+0x108 <il2cpp-symbols-unavailable>",
             "early env: publication invalidates a table after the once-only probe");

    clear_symbol_table();
    set_test_env("PROSPER_IL2CPP_SYMBOLS", valid);
    ensure_symbol_table_loaded();
    publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
    check(symbol_table_status().loaded && symbol_table_status().count == 2,
          "early env: publication keeps an in-bounds table");
    check_eq(resolved_name(0x3fff), "Prosper.Bounds.Edge$$Valid",
             "early env: retained table resolves at the last mapped byte");
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    clear_symbol_table();
    publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108), "Il2cpp+0x108",
             "loaded bounds: unset environment keeps the production label bare");
    check(!symbol_table_status().attempted, "loaded bounds: unset environment attempts no load");
}

void test_nonzero_bss_bounds() {
    // Independently constructed pure BSS, without the synthetic PRX generator. Its end is an
    // ELF RVA (0xc000), not the span (0x4000), and the file contains no backing bytes at all.
    prosper::Module module;
    prosper::Segment bss;
    bss.type = prosper::PT_LOAD;
    bss.flags = 6;
    bss.vaddr = 0x8000;
    bss.memsz = 0x4000;
    module.segments.push_back(bss);
    prosper::LoadedImage image;
    std::string error;
    const bool built = prosper::build_image(module, prosper::BOOT_IL2CPP, image, &error);
    check(built, "nonzero BSS: independent image builds");
    if (!built) return;
    check(image.min_vaddr == 0x8000 && image.max_vaddr == 0xc000 && image.mem.size() == 0x4000,
          "nonzero BSS: actual extent differs from both the span and file size");
    clear_symbol_table();
    publish_loaded_module_bounds(LoadedModuleBounds{image.min_vaddr, image.max_vaddr});
    const std::string valid = write_bounds_table("bounds-bss-valid.symtab", 0x8100, 0xbfff);
    const std::string oversized = write_bounds_table("bounds-bss-oversized.symtab", 0x8100, 0xc000);
    check(load_symbol_table(valid, nullptr), "nonzero BSS: highest RVA below actual end loads");
    check_eq(resolved_name(0x8108), kBoundsInside, "nonzero BSS: raw RVAs are relative to image base");
    check(resolve_guest_va(image.base + 0x8108).state == ResolveState::Resolved,
          "nonzero BSS: guest VA uses image base without subtracting the nonzero minimum");
    check_eq(state_of(0x7fff), "outside-module", "nonzero BSS: query below actual minimum is outside");
    check_eq(state_of(0xc000), "outside-module", "nonzero BSS: query at actual end is outside");
    const std::string low_record = write_bounds_table("bounds-bss-low-record.symtab", 0x100, 0x8100);
    check(load_symbol_table(low_record, nullptr),
          "nonzero BSS: a low record alone does not turn the partial mismatch check into identity validation");
    check_eq(state_of(0x100), "outside-module", "nonzero BSS: accepted low record cannot name an outside query");
    check_eq(resolved_name(0x8108), "Prosper.Bounds.Edge$$Valid",
             "nonzero BSS: accepted sibling record still resolves inside the loaded extent");
    check(!load_symbol_table(oversized, &error), "nonzero BSS: exclusive RVA end is refused");
    check_bounds_refusal(oversized, &error, "loaded IL2CPP RVA range", "nonzero BSS end");
    check(load_symbol_table(valid, nullptr), "nonzero BSS: valid reload recovers");
}

void test_absent_bounds_and_reset(const std::string& fixture) {
    clear_symbol_table();
    publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
    const std::string valid = write_bounds_table("bounds-absent-valid.symtab", 0x100, 0x3fff);
    check(load_symbol_table(valid, nullptr), "absent publication: valid sibling first loads");
    publish_loaded_module_bounds(std::nullopt);
    check_bounds_refusal(valid, nullptr, "no IL2CPP module is loaded", "absent publication");
    check_eq(state_of(0x108), "outside-module", "absent module: formerly inside raw query is outside");
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108), "Il2cpp+0x108",
             "absent module: production labels make no symbol claim");
    std::string error;
    check(!load_symbol_table(fixture, &error), "absent module: even a valid offline table is refused");
    check_bounds_refusal(fixture, &error, "no IL2CPP module is loaded", "absent direct load");
    clear_symbol_table();
    check(load_symbol_table(fixture, nullptr), "reset: Unknown session restores offline fixture loading");
    check_eq(resolved_name(0xa00000), "Prosper.Fixture.Tied$$MethodZzz",
             "reset: offline query beyond the old loaded extent resolves again");
    clear_symbol_table();
}

#ifndef _WIN32
struct PendingSymbolLoad {
    std::atomic<bool> done{false};
    bool loaded = false;
    std::string error;
};

void require_worker_drain(std::thread& worker, const std::shared_ptr<PendingSymbolLoad>& state) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!state->done.load() && std::chrono::steady_clock::now() < end)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(state->done.load(), "POSIX stale parse: worker drains within the bounded deadline");
    if (!state->done.load()) {
        // The worker retains its state. Avoid joining a blocked reader or tearing down globals
        // underneath it; a fixture failure must remain bounded even when the loader is broken.
        std::fflush(nullptr);
        std::_Exit(1);
    }
    worker.join();
}

std::shared_ptr<PendingSymbolLoad> complete_stale_parse(
    const std::string& old_payload, const std::function<void()>& reset_while_blocked) {
    const std::string fifo = prosper_test::test_scratch_file("bounds-pending.fifo");
    const bool created = ::mkfifo(fifo.c_str(), 0600) == 0;
    check(created, "POSIX stale parse: disposable FIFO is created");
    if (!created) return nullptr;
    clear_symbol_table();
    auto state = std::make_shared<PendingSymbolLoad>();
    std::thread worker([state, fifo] {
        state->loaded = load_symbol_table(fifo, &state->error);
        state->done.store(true);
    });
    int writer = -1;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (writer < 0 && std::chrono::steady_clock::now() < end) {
        writer = ::open(fifo.c_str(), O_WRONLY | O_NONBLOCK);
        if (writer >= 0 || (errno != ENXIO && errno != EINTR)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(writer >= 0, "POSIX stale parse: nonblocking writer observes the old reader");
    if (writer < 0) {
        std::fflush(nullptr);
        std::_Exit(1);
    }
    // load_symbol_table captures its session before opening the FIFO. A writer can open only
    // once that old reader exists, and no payload is supplied until the new session is installed.
    reset_while_blocked();
    const long atomic_limit = ::fpathconf(writer, _PC_PIPE_BUF);
    check(atomic_limit > 0 && old_payload.size() <= static_cast<size_t>(atomic_limit),
          "POSIX stale parse: synthetic payload fits one atomic pipe write");
    ssize_t written;
    do { written = ::write(writer, old_payload.data(), old_payload.size()); }
    while (written < 0 && errno == EINTR && std::chrono::steady_clock::now() < end);
    check(written == static_cast<ssize_t>(old_payload.size()), "POSIX stale parse: old payload is delivered");
    ::close(writer);
    require_worker_drain(worker, state);
    check(!state->loaded, "POSIX stale parse: superseded parse reports refusal to its caller");
    check(state->error.find("symbol load superseded by session reset") != std::string::npos,
          "POSIX stale parse: caller receives the session-reset reason");
    ::unlink(fifo.c_str());
    return state;
}

void test_reset_fences_pending_parse() {
    const std::string current = write_bounds_table("bounds-current.symtab", 0x100, 0x3fff);
    const auto state = complete_stale_parse(
        bounds_table(0x100, 0x3ffe, "Prosper.Bounds.Old$$Stale"), [&] {
            clear_symbol_table();
            publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
            check(load_symbol_table(current, nullptr), "POSIX stale parse: new-session valid table loads");
        });
    if (!state) return;
    const SymbolTableStatus status = symbol_table_status();
    check(status.loaded && status.count == 2 && status.error.empty() && status.source == current,
          "POSIX stale parse: old completion preserves the current status and source");
    check_eq(resolved_name(0x3fff), "Prosper.Bounds.Edge$$Valid",
             "POSIX stale parse: old completion cannot replace the current final method");
    // Shrinking the extent invalidates the current table without resetting its completed probe.
    // An in-bounds replacement in the environment must wait for an explicit load or session reset.
    publish_loaded_module_bounds(LoadedModuleBounds{0, 0x3fff});
    const std::string recovery = write_bounds_table("bounds-probe-recovery.symtab", 0x100, 0x3ffe);
    set_test_env("PROSPER_IL2CPP_SYMBOLS", recovery);
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
             "Il2cpp+0x108 <il2cpp-symbols-unavailable>",
             "POSIX stale parse: subsequent annotation retains the completed env probe");
    check(load_symbol_table(recovery, nullptr), "POSIX stale parse: explicit valid reload recovers");
    check_eq(resolved_name(0x3ffe), "Prosper.Bounds.Edge$$Valid",
             "POSIX stale parse: recovery resolves the new final method");
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    clear_symbol_table();
}

void test_reset_fences_failed_env_probe() {
    const auto state = complete_stale_parse("not a prosper symbol table\n", [] {
        clear_symbol_table();
        publish_loaded_module_bounds(LoadedModuleBounds{0, 0x4000});
    });
    if (!state) return;
    const SymbolTableStatus status = symbol_table_status();
    check(!status.attempted && !status.loaded && status.count == 0 && status.source.empty() &&
              status.error.empty(), "POSIX stale failure: old parse leaves new session unattempted");
    check_eq(state_of(0x108), "not-configured", "POSIX stale failure: old error cannot become new status");
    const std::string current = write_bounds_table("bounds-new-env.symtab", 0x100, 0x3fff);
    set_test_env("PROSPER_IL2CPP_SYMBOLS", current);
    ensure_symbol_table_loaded();
    check(symbol_table_status().loaded && symbol_table_status().source == current,
          "POSIX stale failure: old completion cannot consume the new session's env probe");
    check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
             "Il2cpp+0x108 Prosper.Bounds.Shared$$Inside+0x8",
             "POSIX stale failure: new env-backed label resolves after the old failure");
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    clear_symbol_table();
}
#else
void test_reset_fences_pending_parse() {
    std::printf("[skip] POSIX FIFO stale-parse handshake is unavailable on Windows\n");
}
void test_reset_fences_failed_env_probe() {
    std::printf("[skip] POSIX FIFO stale-failure env-probe control is unavailable on Windows\n");
}
#endif

void record_guest_entry(bool, void* opaque) { ++*static_cast<int*>(opaque); }

bool write_boot_modules(const std::filesystem::path& root, bool include_il2cpp) {
    std::error_code error;
    std::filesystem::create_directories(root / "Media/Modules", error);
    check(!error, "boot fixture: module directories created");
    if (error) return false;
    prosper_test::SynthModuleSpec spec;
    spec.init = false;
    std::string reason;
    const bool eboot = prosper_test::write_synth_prx((root / "eboot.bin").string(), spec, &reason);
    check(eboot, "boot fixture: synthetic eboot writes");
    if (!eboot) return false;
    const bool il2cpp = !include_il2cpp || prosper_test::write_synth_prx(
        (root / "Media/Modules/Il2cppUserAssemblies.prx").string(), spec, &reason);
    check(il2cpp, "boot fixture: synthetic IL2CPP presence matches the requested arm");
    return il2cpp;
}

void check_boot_image(const prosper::Program& program, bool present) {
    const prosper::LoadedImage* image = nullptr;
    for (const auto& candidate : program.imgs)
        if (candidate.base == prosper::BOOT_IL2CPP) image = &candidate;
    check((image != nullptr) == present, "real boot: IL2CPP image presence matches the fixture");
    if (!image) return;
    check(image->min_vaddr == 0 && image->max_vaddr == 0x4000 && image->mem.size() == 0x4000,
          "real boot: loaded image end includes the materialized BSS/alignment extent");
    if (image->mem.size() >= 4) {
        const auto* mapped = reinterpret_cast<const uint8_t*>(image->base + image->min_vaddr);
        check(std::memcmp(mapped, image->mem.data(), 4) == 0 && mapped[0] == 0x7f,
              "real boot: image bytes are actually mapped at the guest address");
    }
}

int boot_bounds_mode(const std::string& mode) {
    const bool absent = mode == "absent";
    const bool oversized = mode == "oversized";
    if (!absent && !oversized && mode != "valid") return 2;
    const auto root = prosper_test::test_scratch_path("symbol-boot");
    if (!write_boot_modules(root, !absent)) return 1;
    const std::string table = write_bounds_table("bounds-boot.symtab", 0x100,
                                                 oversized ? 0x4000 : 0x3fff);
    set_test_env("PROSPER_IL2CPP_SYMBOLS", table);
    prosper::Program program;
    std::string error;
    int guest_entries = 0;
    bool callback_ran = false;
    prosper::set_guest_execution_thread_enter_test_hook(record_guest_entry, &guest_entries);
    const bool booted = prosper::boot_program(root.string(), program, &error, [&] {
        callback_ran = true;
        check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
                 "Il2cpp+0x108 Prosper.Bounds.Shared$$Inside+0x8",
                 "real boot: pre-map environment lookup accepts the Unknown-session table");
        check(symbol_table_status().loaded && symbol_table_status().count == 2,
              "real boot: early table is loaded before actual image publication");
    });
    prosper::set_guest_execution_thread_enter_test_hook(nullptr);
    check(booted, "real boot: synthetic modules boot successfully");
    if (!booted) {
        std::fprintf(stderr, "synthetic boot failed: %s\n", error.c_str());
        clear_test_env("PROSPER_IL2CPP_SYMBOLS");
        return 1;
    }
    check(callback_ran, "real boot: production pre-map callback runs");
    check(guest_entries == 0 && program.init_fns.empty() && program.slots.empty() &&
              program.data_slots.empty(), "real boot: fixture executes no guest entry or initializer");
    check(program.mods.size() == (absent ? 1U : 2U) && program.imgs.size() == program.mods.size(),
          "real boot: only the requested synthetic modules are linked");
    check_boot_image(program, !absent);
    if (absent) {
        check_bounds_refusal(table, nullptr, "no IL2CPP module is loaded", "real boot absent");
        check_eq(state_of(0x108), "outside-module", "real boot absent: raw query is outside");
        check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108), "Il2cpp+0x108",
                 "real boot absent: publication removes the early plausible method name");
    } else if (oversized) {
        check_bounds_refusal(table, nullptr, "loaded IL2CPP RVA range", "real boot oversized");
        check_eq(state_of(0x108), "unavailable", "real boot oversized: low query is unavailable");
        check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
                 "Il2cpp+0x108 <il2cpp-symbols-unavailable>",
                 "real boot oversized: publication removes the early plausible method name");
        check_eq(state_of(0x4000), "outside-module", "real boot oversized: exclusive end is outside");
    } else {
        check(symbol_table_status().loaded && symbol_table_status().count == 2,
              "real boot valid: publication retains the accepted early table");
        check_eq(resolved_name(0x3fff), "Prosper.Bounds.Edge$$Valid",
                 "real boot valid: symbol beyond physical EOF resolves inside the mapped image");
        check_eq(prosper::describe_code_address(prosper::BOOT_IL2CPP + 0x108),
                 "Il2cpp+0x108 Prosper.Bounds.Shared$$Inside+0x8",
                 "real boot valid: production label keeps the accepted method name");
    }
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}

int describe_mode(int argc, char** argv) {
    for (int i = 2; i < argc; ++i) {
        const uint64_t va = std::strtoull(argv[i], nullptr, 0);
        std::printf("0x%llx  %s\n", (unsigned long long)va,
                    prosper::describe_code_address(va).c_str());
    }
    return 0;
}

int probe_mode(int argc, char** argv) {
    std::string err;
    if (!load_symbol_table(argv[2], &err)) {
        std::fprintf(stderr, "probe: %s\n", err.c_str());
        return 2;
    }
    for (int i = 3; i < argc; ++i) {
        const uint64_t rva = std::strtoull(argv[i], nullptr, 0);
        const Resolution r = resolve_rva(rva);
        if (r.state == ResolveState::Resolved)
            std::printf("%llx resolved %s +0x%llx\n", (unsigned long long)rva, r.name.c_str(),
                        (unsigned long long)r.offset);
        else
            std::printf("%llx %s -\n", (unsigned long long)rva, resolve_state_token(r.state));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--boot-bounds") == 0) return boot_bounds_mode(argv[2]);
    if (argc >= 4 && std::strcmp(argv[1], "--probe") == 0) return probe_mode(argc, argv);
    if (argc >= 3 && std::strcmp(argv[1], "--describe") == 0) return describe_mode(argc, argv);

    const std::string fixture = argc >= 2 ? argv[1] : "data/il2cpp_symtab_fixture.symtab";
    std::printf("[info] fixture: %s\n", fixture.c_str());

    test_nearest_preceding(fixture);
    test_name_with_spaces();
    test_window_boundary();
    test_below_first_symbol();
    test_tie_rule();
    test_aperture();
    test_annotation_strings();
    test_three_negative_states(fixture);
    test_rejects_a_raw_script_json();
    test_rejects_truncation();
    test_rejects_unsorted();
    test_rejects_unsorted_ties();
    test_rejects_missing_window();
    test_utf8_name_roundtrip();
    test_env_path(fixture);
    test_describe_code_address_wiring(fixture);
    test_loaded_bounds_reload();
    test_bounds_publication_and_env();
    test_nonzero_bss_bounds();
    test_absent_bounds_and_reset(fixture);
    test_reset_fences_pending_parse();
    test_reset_fences_failed_env_probe();

    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
