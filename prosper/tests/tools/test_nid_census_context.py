#!/usr/bin/env python3
"""Exercise nid_census's real CLI against small, disposable ELF modules (#3554 item 1).

Every byte is constructed from ELF64 fields and Prosper's Sony dynamic-tag contract; no dump
content is used. Imports and sibling exports provide positive controls for the reported selection,
so an empty or unreadable fixture cannot pass as a successfully measured zero.
"""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


DATA_NID = "CensusDataA"
LOCAL_NID = "CensusLocal"
FUNC_NID = "CensusFuncA"
TSV_HEADER = "nid\tname\tsym_type\tregistered\ttitles\tmodules\tlibs\ttitle_list\tname_src"
failures = 0


def check(condition, label):
    global failures
    print(f"  [{'ok' if condition else 'FAIL'}] {label}")
    if not condition:
        failures += 1


def write_module(path, imports=(), exports=()):
    """Hand-build one raw ELF with complete dynamic/symbol tables, without executing its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = bytearray(0x800)
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<16sHHIQQQIHHHHHH", image, 0, ident, 0xFE18, 0x3E, 1,
                     0, 0x40, 0, 0, 0x40, 0x38, 2, 0, 0, 0)
    # PT_LOAD contains both tables; PT_DYNAMIC names a bounded run inside it.
    struct.pack_into("<IIQQQQQQ", image, 0x40, 1, 6, 0, 0, 0, len(image), len(image), 0x1000)
    struct.pack_into("<IIQQQQQQ", image, 0x78, 2, 6, 0x180, 0x180, 0, 0x80, 0x80, 8)
    symbols = [(nid, kind, True) for nid, kind in exports]
    symbols += [(nid, kind, False) for nid, kind in imports]
    strings = bytearray(b"\0")
    for index, (nid, kind, defined) in enumerate(symbols, 1):
        name_offset = len(strings)
        strings.extend(f"{nid}#A#A\0".encode("ascii"))
        struct.pack_into("<IBBHQQ", image, 0x280 + index * 24, name_offset,
                         0x10 | kind, 0, int(defined), 0x100 if defined else 0, 8)
    image[0x480:0x480 + len(strings)] = strings
    tags = [(5, 0x480), (10, len(strings)), (6, 0x280), (11, 24),
            (0x6100003F, (len(symbols) + 1) * 24), (0, 0)]
    for index, (tag, value) in enumerate(tags):
        struct.pack_into("<qQ", image, 0x180 + index * 16, tag, value)
    path.write_bytes(image)


def run_census(binary, *arguments):
    env = os.environ.copy()
    env.pop("PROSPER_NO_PLUGIN_AUTOLINK", None)
    result = subprocess.run([str(binary), *(str(arg) for arg in arguments)],
                            capture_output=True, env=env, timeout=5)
    # Native Windows printf uses CRLF; retain byte-exact POSIX output for the path-control arm.
    if os.name == "nt":
        result.stdout = result.stdout.replace(b"\r\n", b"\n")
        result.stderr = result.stderr.replace(b"\r\n", b"\n")
    # Keep CR and Unicode separators intact: records are delimited by emitted LF bytes.
    result.stdout = result.stdout.decode("utf-8")
    result.stderr = result.stderr.decode("utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"census returned {result.returncode}: {result.stderr}")
    return result


def input_label(path):
    label = []
    for char in str(path):
        if char == "\\":
            label.append("\\\\")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            label.append(f"\\x{ord(char):02x}")
        else:
            label.append(char)
    return "".join(label)


def context_line(path, mode, read, failed, prefix=""):
    return f"{prefix}input: {input_label(path)} -> {mode} ({read} read, {failed} unreadable)"


def data_line(path, count, prefix=""):
    return f"{prefix}data input: {input_label(path)} -> {count} unresolved DATA binding(s)"


def tsv_rows(output):
    lines = [line for line in output.split("\n") if line and not line.startswith("# ")]
    check(bool(lines) and lines[0] == TSV_HEADER, "TSV control: the legacy data header is unchanged")
    check(all(len(line.split("\t")) == 9 for line in lines),
          "TSV control: every noncomment line retains nine columns (incl. name_src)")
    return {columns[0]: columns for columns in (line.split("\t") for line in lines[1:])}


def exercise_selection(binary, root, importer, provider):
    linked = run_census(binary, root, "--data-only")
    check(not linked.stderr, "root control: all selected fixtures parse without warnings")
    check(DATA_NID in linked.stdout and LOCAL_NID not in linked.stdout and FUNC_NID not in linked.stdout,
          "root control: a sibling export excludes one DATA import while an unlinked decoy does not")
    check("1 binding(s) excluded as satisfied by a sibling module's export" in linked.stdout,
          "root control: the actual sibling exclusion is counted")
    check("over 2 module(s) read, 0 unreadable" in linked.stdout,
          "root control: exactly the two loader-selected modules are read")
    check(context_line(root, "link set, 2 modules", 2, 0) in linked.stdout,
          "root context: identify the actual two-module link set")
    check(data_line(root, 1) in linked.stdout,
          "root DATA context: identify the original root input and unresolved count")

    explicit = run_census(binary, importer, provider, "--data-only")
    check(not explicit.stderr and DATA_NID in explicit.stdout and LOCAL_NID in explicit.stdout,
          "module control: separate explicit inputs do not acquire sibling exports")
    check("0 binding(s) excluded as satisfied by a sibling module's export" in explicit.stdout,
          "module control: per-module exclusion remains zero")
    expected = [context_line(importer, "single module", 1, 0),
                context_line(provider, "single module", 1, 0)]
    check([line for line in explicit.stdout.split("\n") if line.startswith("input: ")] == expected,
          "module context: identify each explicit module in argument order")
    check([line for line in explicit.stdout.split("\n") if line.startswith("data input: ")] ==
          [data_line(importer, 2), data_line(provider, 0)],
          "module DATA context: retain importer count and zero-count provider as separate inputs")
    check("#lbl" in explicit.stdout and "groups inputs by basename" in explicit.stdout,
          "aggregate context: qualify the unchanged basename-grouped human count")


def exercise_single_and_failed(binary, scratch):
    root = scratch / "one-module-root"
    write_module(root / "eboot.bin", imports=[(FUNC_NID, 2)])
    valid = run_census(binary, root)
    check(not valid.stderr and FUNC_NID in valid.stdout and "over 1 module(s) read, 0 unreadable" in valid.stdout,
          "one-module control: the root's complete fixture contributes a real import")
    check(context_line(root, "link set, 1 module", 1, 0) in valid.stdout,
          "one-module context: a dump root remains a link set when only one module is selected")
    broken = root / "Media" / "Plugins" / "Unreadable.prx"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"synthetic invalid ELF")
    failed_root = run_census(binary, root)
    check(FUNC_NID in failed_root.stdout and "over 1 module(s) read, 1 unreadable" in failed_root.stdout and
          "[warn]" in failed_root.stderr and "WARNING:" in failed_root.stdout,
          "failed-root control: a valid import survives beside an explicitly unreadable module")
    check(context_line(root, "link set, 2 modules", 1, 1) in failed_root.stdout,
          "failed-root context: distinguish selected modules from successful parses")
    failed_module = run_census(binary, broken, "--data-only")
    check("over 0 module(s) read, 1 unreadable" in failed_module.stdout and "[warn]" in failed_module.stderr,
          "failed-module control: the regular file is selected and its parse failure is visible")
    check(context_line(broken, "single module", 0, 1) in failed_module.stdout and
          data_line(broken, 0) in failed_module.stdout,
          "failed-module context: retain the failed single-module input and its zero DATA count")


def exercise_repeated_labels(binary, scratch):
    importer = scratch / "first" / "same.prx"
    empty = scratch / "second" / "same.prx"
    write_module(importer, imports=[(DATA_NID, 1)])
    write_module(empty)
    result = run_census(binary, importer, empty, "--data-only", "--tsv")
    rows = tsv_rows(result.stdout)
    check(not result.stderr and set(rows) == {DATA_NID} and rows[DATA_NID][2:6] == ["OBJECT", "0", "1", "1"],
          "same-basename control: valid DATA row preserves legacy registration and aggregate values")
    check("1 DATA binding(s)" in result.stdout and "over 1 of 2 input(s)" in result.stdout,
          "same-basename DATA context: count actual inputs rather than merged basenames")
    expected = [data_line(importer, 1, "# "), data_line(empty, 0, "# ")]
    check([line for line in result.stdout.split("\n") if line.startswith("# data input: ")] == expected,
          "same-basename DATA context: retain both original arguments including the zero")

    repeated = run_census(binary, importer, importer, "--data-only", "--tsv")
    rows = tsv_rows(repeated.stdout)
    check(not repeated.stderr and rows.get(DATA_NID, [])[4:6] == ["1", "2"],
          "repeated-input control: existing basename and binding aggregates remain unchanged")
    check(repeated.stdout.split("\n").count(data_line(importer, 1, "# ")) == 2 and
          "over 2 of 2 input(s)" in repeated.stdout,
          "repeated-input DATA context: report each argument and its own binding count")


def exercise_tsv(binary, root, importer, provider):
    result = run_census(binary, root, importer, provider, "--data-only", "--tsv")
    rows = tsv_rows(result.stdout)
    check(not result.stderr and set(rows) == {DATA_NID, LOCAL_NID} and
          rows[DATA_NID][4:6] == ["2", "2"] and rows[LOCAL_NID][4:6] == ["1", "1"],
          "mixed TSV control: root and separate modules retain the actual resolution results")
    expected = [context_line(root, "link set, 2 modules", 2, 0, "# "),
                context_line(importer, "single module", 1, 0, "# "),
                context_line(provider, "single module", 1, 0, "# ")]
    check([line for line in result.stdout.split("\n") if line.startswith("# input: ")] == expected,
          "mixed TSV context: selection labels are comments in original input order")
    check([line for line in result.stdout.split("\n") if line.startswith("# data input: ")] ==
          [data_line(root, 1, "# "), data_line(importer, 2, "# "), data_line(provider, 0, "# ")],
          "mixed TSV DATA context: label every input independently including zeros")


def exercise_path_controls(binary, scratch):
    if os.name == "nt":
        print("  [scope] POSIX path-control fixture omitted: native Windows forbids these filename bytes")
        return
    # Keep the basename ordinary, so legacy TSV rows remain valid even without the new metadata.
    # NEL/U+2028/U+2029 are valid UTF-8 path text, not LF delimiters in the TSV byte stream.
    parent = "controls-\n\r\t\v\f\x01\x1f\x7f\\-π\u0085\u2028\u2029"
    module = scratch / parent / "ordinary.prx"
    write_module(module, imports=[(DATA_NID, 1)])
    result = run_census(binary, module, "--data-only", "--tsv")
    rows = tsv_rows(result.stdout)
    check(not result.stderr and rows.get(DATA_NID, [])[2:6] == ["OBJECT", "0", "1", "1"],
          "path control: the valid module contributes the unchanged ordinary-basename DATA row")
    # Literal golden escapes keep this edge oracle independent of both escaping implementations.
    escaped_parent = r"controls-\x0a\x0d\x09\x0b\x0c\x01\x1f\x7f\\-" + "π\u0085\u2028\u2029"
    expected_label = input_label(scratch) + "/" + escaped_parent + "/ordinary.prx"
    records = result.stdout.split("\n")
    check([line for line in records if line.startswith("# input: ")] ==
          [f"# input: {expected_label} -> single module (1 read, 0 unreadable)"],
          "path context: escape ASCII controls and backslash while preserving UTF-8 on one record")
    check([line for line in records if line.startswith("# data input: ")] ==
          [f"# data input: {expected_label} -> 1 unresolved DATA binding(s)"],
          "path DATA context: apply the same escaping to the independent input label")


def exercise_names(binary, scratch, importer):
    # A flat `NID name` database is a SECONDARY source: its provenance must be visible (TSV
    # `# names:` comment) and its names shown only when they hash back to the NID. The fixture NIDs
    # are arbitrary constants, so a flat entry for one is not a valid preimage and is dropped; this
    # arm pins the wiring and the marker, not a specific resolved name.
    names_csv = scratch / "names.csv"
    names_csv.write_text(f"{DATA_NID} sceFixtureName\n", encoding="utf-8")
    result = run_census(binary, importer, "--names", str(names_csv), "--data-only", "--tsv")
    comment = next((line for line in result.stdout.split("\n") if line.startswith("# names:")), "")
    check(
        "source=flat database (secondary)" in comment,
        "names provenance: a flat database is reported as a secondary source in TSV",
    )
    check(
        "dropped=1" in comment,
        "names verification: an unverifiable flat pair is dropped, not shown as authoritative",
    )
    rows = tsv_rows(result.stdout)
    check(
        DATA_NID in rows and rows[DATA_NID][1] == "?" and rows[DATA_NID][-1] == "-",
        "names row: a dropped flat name leaves the row unnamed with name_src '-'",
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_nid_census_context.py <nid_census executable>")
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="prosper-nid-census-") as temporary:
        scratch = Path(temporary)
        root = scratch / "dump fixture"
        importer = root / "eboot.bin"
        provider = root / "Media" / "Plugins" / "ContextProvider.prx"
        write_module(importer, imports=[(DATA_NID, 1), (LOCAL_NID, 1), (FUNC_NID, 2)])
        write_module(provider, exports=[(LOCAL_NID, 1)])
        write_module(root / "not-linked" / "decoy.prx", exports=[(DATA_NID, 1), (FUNC_NID, 2)])
        exercise_selection(binary, root, importer, provider)
        exercise_single_and_failed(binary, scratch)
        exercise_repeated_labels(binary, scratch)
        exercise_tsv(binary, root, importer, provider)
        exercise_path_controls(binary, scratch)
        exercise_names(binary, scratch, importer)
    print(f"== FAIL: {failures} ==" if failures else "== PASS ==")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
