// hle_registry_dump — write out the HLE registry prosper ACTUALLY builds, as ground truth for
// `tools/re/hle_handler_map.py` (#2070).
//
// `hle_handler_map.py` answers "which Sony NIDs collapse onto one prosper handler?" by parsing
// `src/hle/**`. A parser is the wrong kind of authority for that question in two specific ways, and
// both fail quietly:
//
//   * A registration SHAPE it does not recognise is invisible. It does not error — the census just
//     comes back smaller, and a smaller census of a collapse reads as good news.
//   * It reimplements the C preprocessor to decide which `#if` arm is live. `hle_kernel_mem.cpp`
//     defines `register_kernel_mem_hle()` twice, ~3,200 lines apart, in the two arms of one
//     conditional, so getting that wrong moves real numbers — it is exactly what turned five
//     single-name handlers into "shared" ones in the first published version of this measurement.
//     Only a compiled binary can settle which arm won, because only the compiler evaluated it.
//
// So this runs the real `register_builtin_hle()` and prints the resulting table. No boot, no game
// dump, no GPU, no window: registration is pure setup, which is what makes reconciliation cheap
// enough to be a ctest rather than an occasional manual chore.
//
// Output is one TSV row per NID, plus a trailing count so a truncated file cannot be mistaken for a
// short table:
//
//     <nid>\t<real|placeholder>\t<handler address, hex>\t<display name>
//     # 1153 registrations
//
// CAVEAT on the address column, and it only ever errs in one direction. Two DIFFERENT handlers with
// identical machine code (`s_ok` and `k_attr_noop` are both `{ return 0; }`) may be folded to one
// address by an identical-code-folding linker. That can make distinct handlers look shared; it can
// never make shared handlers look distinct. Grouping by address is therefore an upper bound on the
// collapse, while the NID column is exact — see `test_hle_registry_reconcile.py`, which relies on
// exactly that asymmetry and reports folding rather than tripping over it.
#include "../../src/hle/dispatch.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    prosper::register_builtin_hle();

    std::vector<prosper::RegisteredFn> rows = prosper::Hle::registrations();
    // Sort so a diff between two runs is a real difference, not unordered_map iteration order.
    std::sort(rows.begin(), rows.end(),
              [](const prosper::RegisteredFn& a, const prosper::RegisteredFn& b) {
                  return a.nid < b.nid;
              });

    FILE* out = stdout;
    if (argc > 1) {
        out = fopen(argv[1], "w");
        if (!out) {
            fprintf(stderr, "hle_registry_dump: cannot write %s\n", argv[1]);
            return 2;
        }
    }
    for (const prosper::RegisteredFn& r : rows)
        fprintf(out, "%s\t%s\t%p\t%s\n", r.nid.c_str(), r.placeholder ? "placeholder" : "real",
                r.fn, r.name.c_str());
    fprintf(out, "# %zu registrations\n", rows.size());
    if (out != stdout) fclose(out);

    // A zero-row dump would reconcile against a parser that found nothing and both would agree, so
    // refuse rather than emit an empty table that reads as a clean answer.
    if (rows.empty()) {
        fprintf(stderr, "hle_registry_dump: register_builtin_hle() registered NOTHING — refusing\n");
        return 2;
    }
    return 0;
}
