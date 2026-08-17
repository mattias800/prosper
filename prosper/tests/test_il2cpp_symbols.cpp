// test_il2cpp_symbols.cpp — the runtime IL2CPP symbol resolver (#2551).
//
// Two modes:
//   (no args)                       self-checking unit test; exit code is truth
//   --probe <symtab> <rva> [...]    print one machine-readable line per rva, for the cross-
//                                   implementation agreement test (tools/il2cpp/test_symtab_agreement.py)
//   --describe <guest-va> [...]     print describe_code_address() for each address, i.e. the REAL
//                                   production label a fault backtrace prints. Honors
//                                   PROSPER_IL2CPP_SYMBOLS, so this is how a live capture is checked
//                                   against a real title's symbol table by hand.
//
// Every assertion below is paired with a MUTATION ARM — an input differing in exactly the property
// under test, whose expected answer differs. An arm is only worth having if no other branch of the
// resolver could produce it, so each one names what it excludes.
#include "host/il2cpp_symbols.hpp"
#include "host/boot_program.hpp"
#include "host/exec_image.hpp"   // describe_code_address: the production label being symbolicated

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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
             "…to the same method resolve_rva names");
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
    check(!err.empty(), "…with a non-empty reason: \"" + err + "\"");
    check(symbol_table_status().attempted && !symbol_table_status().loaded,
          "status() reports attempted-but-not-loaded");
    check(symbol_table_status().count == 0, "…and zero symbols");
    check_eq(unavailable_token, "unavailable", "a failed load reads as unavailable");
    check_eq(unavailable_annotation, " <il2cpp-symbols-unavailable>",
             "…and says so at the point of use");

    check(load_symbol_table(fixture, nullptr), "fixture reloads");
    const std::string nomatch_token = state_of(0x1000);
    const std::string nomatch_annotation = annotation_for_guest_va(prosper::BOOT_IL2CPP + 0x1000);
    check_eq(nomatch_token, "no-managed-method", "a loaded table with no covering method");
    check_eq(nomatch_annotation, " <no-managed-method>", "…says THAT, not silence");

    // ARM: the three are pairwise distinct, in both the token and the printed annotation. A
    // resolver that answered "" or "unknown" for all three would satisfy every individual
    // assertion above and fail here — this is the only check that can see the collapse.
    check(not_configured_token != unavailable_token && unavailable_token != nomatch_token &&
              not_configured_token != nomatch_token,
          "the three non-resolving states have three distinct tokens");
    check(not_configured_annotation != unavailable_annotation &&
              unavailable_annotation != nomatch_annotation &&
              not_configured_annotation != nomatch_annotation,
          "…and three distinct annotations");
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
          "…naming the header it wanted: \"" + err + "\"");
    check(symbol_table_status().count == 0, "…and loads zero symbols");

    // ARM: the same body BEHIND a valid header loads. This proves the refusal is the header check
    // and not "anything unusual fails" — without it, a resolver that rejected every file would pass.
    const std::string ok_path = write_temp(
        "il2cpp_symtab_test_rawjson_ok.symtab",
        "prosper-il2cpp-symtab v1 window=0x8000 count=1\n"
        "1e3e00 {\"ScriptMethod\":[{\"Address\":1980928,\"Name\":\"X\"}]}\n");
    check(load_symbol_table(ok_path, nullptr), "the same text behind a valid header loads");
    check(symbol_table_status().count == 1, "…as exactly one symbol");
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
    check(err.find("count=3") != std::string::npos, "…quoting the mismatch: \"" + err + "\"");
    // ARM: the identical entries with an honest count load. Isolates the count check from the
    // parse; a resolver that rejected two-entry files for any other reason fails here.
    const std::string ok = write_temp("il2cpp_symtab_test_short_ok.symtab",
                                      "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                      "1000 A$$a\n"
                                      "2000 B$$b\n");
    check(load_symbol_table(ok, nullptr), "the same entries with count=2 load");
    check(symbol_table_status().count == 2, "…as two symbols");
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
    check(err.find("sorted") != std::string::npos, "…saying why: \"" + err + "\"");
    // ARM: the same two entries in ascending order load and resolve. Without this, "refuses
    // everything" would pass the assertion above.
    const std::string ok = write_temp("il2cpp_symtab_test_sorted_ok.symtab",
                                      "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                      "1000 A$$a\n"
                                      "2000 B$$b\n");
    check(load_symbol_table(ok, nullptr), "the same entries in order load");
    check_eq(resolved_name(0x2000), "B$$b", "…and resolve");
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
    check(err.find("sorted") != std::string::npos, "…saying why: \"" + err + "\"");

    // ARM 1: the same three entries with the tie group ascending are accepted, and the answer at
    // the shared address is the LAST of the group — the property the rule protects.
    const std::string ok = write_temp("il2cpp_symtab_test_tieorder_ok.symtab",
                                      "prosper-il2cpp-symtab v1 window=0x8000 count=3\n"
                                      "1000 A$$a\n"
                                      "2000 Alpha$$a\n"
                                      "2000 Zeta$$z\n");
    check(load_symbol_table(ok, nullptr), "the same entries with the tie group in order load");
    check_eq(resolved_name(0x2000), "Zeta$$z", "…and a tied address resolves to the LAST entry");

    // ARM 2: EQUAL names at a shared rva are legal — resolve.py's sort is non-strict, so a duplicate
    // record must not be rejected. Without this arm the check above would also pass if the
    // comparison had been written as a strict `>`, which would refuse real emitter output.
    const std::string dup = write_temp("il2cpp_symtab_test_tieorder_dup.symtab",
                                       "prosper-il2cpp-symtab v1 window=0x8000 count=2\n"
                                       "2000 Same$$s\n"
                                       "2000 Same$$s\n");
    check(load_symbol_table(dup, nullptr), "two identical records at one rva are accepted");
    check_eq(resolved_name(0x2000), "Same$$s", "…and resolve");
    std::remove(path.c_str());
    std::remove(ok.c_str());
    std::remove(dup.c_str());
}

void test_rejects_missing_window() {
    const std::string path = write_temp("il2cpp_symtab_test_nowindow.symtab",
                                        "prosper-il2cpp-symtab v1 count=1\n"
                                        "1000 A$$a\n");
    std::string err;
    check(!load_symbol_table(path, &err), "a header without window= is REFUSED");
    check(err.find("window") != std::string::npos, "…saying so: \"" + err + "\"");
    // ARM: the window is READ from the file, not hard-coded here. A file declaring window=0x10
    // must stop resolving at +0x10 even though the production window is 0x8000 — no constant
    // baked into this side can produce that answer.
    const std::string tiny = write_temp("il2cpp_symtab_test_tinywindow.symtab",
                                        "prosper-il2cpp-symtab v1 window=0x10 count=1\n"
                                        "1000 A$$a\n");
    check(load_symbol_table(tiny, nullptr), "a window=0x10 table loads");
    check(symbol_table_status().window == 0x10, "…and status reports window=0x10");
    check_eq(state_of(0x100f), "resolved", "…resolving at +0xf");
    check_eq(state_of(0x1010), "no-managed-method", "…and stopping at +0x10");
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
    check(resolved_name(0x1000) == name, "…and the bytes round-trip exactly");
    // ARM: byte length, not character count — a parser that re-encoded or truncated at the first
    // high byte cannot produce this number.
    check(resolved_name(0x1000).size() == name.size(),
          "…with the same byte length (" + std::to_string(name.size()) + ")");
    std::remove(path.c_str());
}

void test_env_path(const std::string& fixture) {
    clear_symbol_table();
    clear_test_env("PROSPER_IL2CPP_SYMBOLS");
    ensure_symbol_table_loaded();
    check(!symbol_table_status().attempted, "an unset PROSPER_IL2CPP_SYMBOLS attempts no load");
    check_eq(state_of(0x100208), "not-configured",
             "…and reads as not-configured, NOT as a failed load");

    clear_symbol_table();
    set_test_env("PROSPER_IL2CPP_SYMBOLS", fixture);
    ensure_symbol_table_loaded();
    // ARM against the arm above: the same call, the same code path, one environment variable
    // different, and now the table is present. Only the env branch can produce this pair.
    check(symbol_table_status().loaded, "PROSPER_IL2CPP_SYMBOLS pointing at the fixture loads it");
    check(symbol_table_status().count == 5, "…with all 5 symbols");
    check_eq(resolved_name(0x100208), "Prosper.Fixture.Alpha$$Start", "…and resolves through it");
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

    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
