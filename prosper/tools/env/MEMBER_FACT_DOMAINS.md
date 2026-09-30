# Diagnostic member-fact domains

The scanner joins context members by **name**, not C++ type. An owning test fixture can declare
the same names as a production reference context and preset those references without environment
gates. Adding that fixture used to retire real production aliases: #3892's native image fixture
changed 122 findings / 65 keys to 111 / 54 without changing the production gate behavior.

An explicit CMake source-role declaration separates these member facts:

```cmake
add_executable(test_example tests/example.cpp)
add_test(NAME example COMMAND test_example)
set_property(TARGET test_example PROPERTY PROSPER_DIAG_MEMBER_FACT_DOMAIN TEST_LOCAL)
```

This is a **maintained assertion that this translation unit is nonshipping**, reviewed with its
build/source changes. It is not an inferred complete CMake/C++ reachability proof. Add a marker
only when that assertion is true. A marker removes neither the file nor its diagnostic findings
from collection, inventory, local scanning, call-site analysis or baseline enforcement.

## Admission and visible refusal

`member_fact_domains.py` recognizes literal CMake commands outside comments, quoted payloads and
bracket payloads. The supported marker is exactly the five unquoted arguments above, with a
literal `test_` target name and `TEST_LOCAL` value, outside function/macro definitions. It requires
one literal single-source `add_executable` and one literal four-argument
`add_test(NAME name COMMAND target)`. The source must exist, be collected, normalize beneath
`tests/`, end in `.cpp`, and use no symlink component. Conditions are not evaluated: this declares
the source's role even when its test is disabled in a particular configuration.

Unannotated sources remain shared. Duplicate/stale/unsupported markers, missing declarations,
files or registrations, multiple/variable/glob sources, and headers visibly refuse admission.
Every refused file stays shared and locally scanned. The CLI prints accepted and refused records
with target, canonical source, location and reason; a refused registration fails the gate even
when its diagnostic keys happen to match the baseline.

The following enumerated safeguards also revoke admission:

- Extra `target_sources`, literal source reuse in another executable/library, alias targets,
  inbound `target_link_libraries`, install/export, and property/source-property modifications
  naming the target or source. The test's ordinary outbound backend links are allowed.
- Source-valued `set`/`list`/`file`/`configure_file` constructions naming the source, literal globs
  whose directory prefix could collect it, and opaque include/subdirectory/EVAL or unsupported
  commands that mention the target/source. Canonical paths compare identities across relative
  spellings and CMake file scopes; opaque/genex source mentions conservatively match the basename.
- Literal CPP inclusion from scanned sources/headers or `.inc` files, resolved relative to the
  including file or repository source root. Headers themselves always retain shared facts,
  including the shipped `tests/fixtures/render_runner.h`.

These guards **do not execute CMake or preprocess C++**. Unrelated opaque functions/macros,
variable/genex expansion, generated/copied sources, dynamically included CMake, macro includes,
compiler include paths and dependency closure are outside their claimed completeness. A reviewer
must still verify the source-role assertion against the actual build. Unknown supported-role
syntax refuses isolation; absence of an enumerated misuse is not a reachability proof. An evaluated
target/source manifest and compiler dependency closure would be needed for that stronger claim.

## Fact and scan behavior

Both member declaration kinds **and designated-initializer writes** are partitioned together.
Shared sources use the shared table. Admitted test translation units use the original all-file
table, preserving their existing conservative test-local semantics. All headers, unannotated test
files and refused sources remain shared. Predicates, cached getters, definitions, printing
functions, global flags, call sites and local diagnostic scans still see every collected file.

The member-name join remains lexical within each table: any same-named value member or unknown
initializer in that table retires the alias. This introduces no type precision. `--selftest`
checks the two independent collision modes, domain-disabled and marker-absent calibration,
production/header retirement, exact local findings/inventory and literal admission/refusal.
It runs in the existing normal and hostile-TMPDIR CTest arms and on every full tree scan.

## Ruled out

- **The image status IIFE caused the lost findings.** Holding the fixture addition fixed while
  restoring the original builder still produced 111 / 54; keeping the status introduction while
  omitting only the fixture restored all 122 / 65. The changed member facts came from the fixture,
  independently of the new resource-status wrapper (#3892).
- **An assignment initializer made the scanner consume the entire image branch.** Its statement
  reader has a 24-line bound. A parenthesized direct-initialization probe also produced 111 / 54;
  changing production spelling did not restore the missing provenance (#3892).
- **Renaming only the owning fixture fields would fix the join.** Prefixing all 47 fields removed
  the value-name collisions but left unknown designated context initializers. The same findings
  remained absent; both declaration and initializer facts need the same declared domain (#3892).
