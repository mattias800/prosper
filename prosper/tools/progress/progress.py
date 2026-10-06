"""Static HLE coverage census, counted per registered Sony NID.

Answers "of the Sony entry points prosper registers a handler for, how many get a real handler and
how many only a placeholder?", grouped by `src/hle/<area>/`. No dump, no build, no GPU.

The registration surface comes from `tools/re/hle_handler_map.py`'s parser, not from a regex of its
own. That parser discovers every registration shape (direct `Hle::register_*` calls, file-local
wrapper macros, lambdas), evaluates `#if` arms for one `--platform`, and reports what it could not
claim. A first version of this tool keyed on the `HLE()` macro instead and saw about 60% of the
surface (#4532): plain functions, wrapper macros and lambdas fell out of both sides of the ratio, and
the `#define HLE(name)` line itself was counted as a handler.

On top of the parser this tool resolves the registrations whose NID is not a literal: elements of
a file-local constant table (`kUlt[kIdxInitialize].nid`), and a fold over a
`std::index_sequence` (`register_agc_tracers<0>(std::make_index_sequence<32>{})`), which
registers one NID per pack element and so must count as many NIDs, not as one site.

A NID is DONE when at least one registration installs it through a non-placeholder API
(`register_fn`, `register_typed`, `register_guest_abi`, ...), and PLACEHOLDER when every
registration of it is `register_placeholder`. A real handler beats a placeholder regardless of file
order: `register_placeholder` records a shadow when it overwrites a real entry, and ctest
`test_hle_no_shadow` fails on any shadow at boot, so a placeholder registered after a real handler
cannot survive the suite. "Done" says a real handler is installed, not that it is correct.

Exit 0 when the census is complete, 3 when a registration site was unclaimed or a NID expression
stayed unresolved (every count is then a lower bound), 2 when the scan was refused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "re"))
import hle_handler_map as H  # noqa: E402  (composed, not copied)

PLACEHOLDER_API = "register_placeholder"
INT_TYPES = r"(?:std::)?(?:size_t|int|unsigned|uint32_t|uint64_t|int32_t|int64_t)"


# ---------------------------------------------------------------- constant tables


def _elements(inner: str) -> list[str]:
    """Top-level comma-separated items of a brace body, without a trailing empty item."""
    args, _end = H.split_args("(" + inner + ")", 0)
    items = [a.strip() for a in (args or [])]
    return items[:-1] if items and not items[-1] else items


class FileTables:
    """Compile-time constants, arrays, struct layouts and index_sequence templates of one file."""

    def __init__(self, text: str) -> None:
        """Parse `text`, already comment-stripped and blanked to one platform arm."""
        self.text = text
        self.structs: dict[str, list[str]] = {}
        for m in re.finditer(r"\bstruct\s+(\w+)\s*\{([^{}]*)\}", text):
            fields = []
            for decl in m.group(2).split(";"):
                decl = decl.split("=")[0].strip()
                ident = re.findall(r"[A-Za-z_]\w*", decl)
                if ident:
                    fields.append(ident[-1])
            self.structs[m.group(1)] = fields
        self.arrays: dict[str, tuple[str, list[str]]] = {}
        for m in re.finditer(r"([\w:]+)\s*(?:const\s+)?(\w+)\s*\[\s*\w*\s*\]\s*=\s*\{", text):
            close = H._matching_brace(text, m.end() - 1)
            if close is not None:
                self.arrays[m.group(2)] = (m.group(1), _elements(text[m.end() : close]))
        self.consts: dict[str, int] = {}
        pending: dict[str, str] = {}
        for m in re.finditer(rf"\b(?:constexpr|const)\s+{INT_TYPES}\s+([^;{{}}]+);", text):
            for item in _elements(m.group(1)):
                name, _, expr = item.partition("=")
                if expr and re.fullmatch(r"\s*\w+\s*", name):
                    pending[name.strip()] = expr.strip()
        for _round in range(8):  # fixpoint: constants defined in terms of other constants
            for name, expr in list(pending.items()):
                value = self.evaluate(expr, {})
                if value is not None:
                    self.consts[name] = value
                    del pending[name]
        self.templates = self._index_sequence_templates()

    def evaluate(self, expr: str, binding: dict[str, int]) -> int | None:
        """Integer value of `expr` from known constants and `binding`, or None."""
        e = re.sub(
            r"sizeof\s*\(\s*(\w+)\s*\)\s*/\s*sizeof\s*\(\s*\1\s*\[\s*0\s*\]\s*\)",
            lambda m: (
                str(len(self.arrays[m.group(1)][1])) if m.group(1) in self.arrays else m.group(0)
            ),
            expr,
        )
        e = re.sub(
            r"std::size\s*\(\s*(\w+)\s*\)",
            lambda m: (
                str(len(self.arrays[m.group(1)][1])) if m.group(1) in self.arrays else m.group(0)
            ),
            e,
        )
        e = re.sub(r"\b(\d+)(?:u|ull|ul|z|uz)?\b", r"\1", e, flags=re.I)

        def ident(m: re.Match) -> str:
            name = m.group(0)
            if name in binding:
                return str(binding[name])
            return str(self.consts[name]) if name in self.consts else name

        e = re.sub(r"\b[A-Za-z_]\w*\b", ident, e)
        if not re.fullmatch(r"[\d\s+\-*/()%]+", e):
            return None
        try:
            return int(eval(e.replace("/", "//"), {"__builtins__": {}}, {}))  # noqa: S307 — digits and operators only
        except Exception:  # noqa: BLE001
            return None

    def string(self, expr: str, binding: dict[str, int]) -> str | None:
        """String literal named by `expr`: a literal, `ARR[i]`, or `ARR[i].field`."""
        lit = H.resolve_string(expr)
        if lit is not None:
            return lit
        m = re.fullmatch(r"\s*(\w+)\s*\[(.+)\]\s*(?:\.\s*(\w+))?\s*", H.unwrap(expr))
        if not m or m.group(1) not in self.arrays:
            return None
        elem_type, elements = self.arrays[m.group(1)]
        index = self.evaluate(m.group(2), binding)
        if index is None or not 0 <= index < len(elements):
            return None
        element = elements[index]
        if m.group(3):
            fields = self.structs.get(elem_type.split("::")[-1])
            if not fields or m.group(3) not in fields or not element.startswith("{"):
                return None
            parts = _elements(element[1:-1])
            pos = fields.index(m.group(3))
            if pos >= len(parts):
                return None
            element = parts[pos]
        return H.resolve_string(element)

    def _index_sequence_templates(self) -> list[tuple[str, tuple[int, int], list[dict[str, int]]]]:
        """(name, body span, bindings per call) for each `template<..., size_t... Is>` function
        taking `std::index_sequence<Is...>`. Each call `F<A>(std::make_index_sequence<N>{})` binds
        the non-pack parameters to its explicit arguments and the pack to every value in 0..N-1."""
        out = []
        pattern = (
            r"template\s*<([^<>]*)>\s*(?:[\w:]+\s+)+?(\w+)\s*\(\s*(?:std::)?index_sequence"
            r"\s*<\s*(\w+)\s*\.\.\.\s*>\s*\w*\s*\)\s*\{"
        )
        for m in re.finditer(pattern, self.text):
            close = H._matching_brace(self.text, m.end() - 1)
            if close is None:
                continue
            params = [p.strip() for p in m.group(1).split(",")]
            fixed = [re.findall(r"\w+", p)[-1] for p in params if "..." not in p]
            pack = m.group(3)
            bindings = []
            call = r"\b%s\s*<([^<>]*)>\s*\(\s*(?:std::)?make_index_sequence\s*<([^<>]+)>\s*\{\s*\}\s*\)"
            for c in re.finditer(call % re.escape(m.group(2)), self.text):
                actuals = _elements(c.group(1))
                count = self.evaluate(c.group(2), {})
                values = [self.evaluate(a, {}) for a in actuals]
                if count is None or len(values) != len(fixed) or None in values:
                    bindings.append(None)  # a call the tool cannot evaluate: stays unresolved
                    continue
                base = dict(zip(fixed, values, strict=False))
                bindings += [dict(base, **{pack: i}) for i in range(count)]
            out.append((m.group(2), (m.end() - 1, close), bindings))
        return out

    def bindings_at(self, line: int) -> list[dict[str, int] | None]:
        """Pack bindings for a site on `line`: one empty binding outside any template."""
        offset = sum(len(x) + 1 for x in self.text.split("\n")[: line - 1])
        for _name, (start, end), bindings in self.templates:
            if start <= offset <= end:
                return bindings or [None]
        return [{}]


# ---------------------------------------------------------------- the census


class Reg:
    """One registered NID: where it is registered, through which API, and under what label."""

    def __init__(self, nid: str, api: str, label: str, handler: str, file: str, line: int) -> None:
        """Record one registration of `nid`."""
        self.nid, self.api, self.label, self.handler = nid, api, label, handler
        self.file, self.line = file, line


def area_of(file: str) -> str:
    """`src/hle/<area>/` of a file path relative to `src/hle`."""
    parts = Path(file).parts
    return parts[0] if len(parts) > 1 else "."


def scan(src_hle: Path, platform: str) -> dict:
    """Every registration under `src_hle` for `platform`, literal and table-resolved."""
    sc = H.scan_tree(str(src_hle), platform)
    regs = [Reg(r.nid, r.api, r.name or r.handler, r.handler, r.file, r.line) for r in sc.regs]
    tables: dict[str, FileTables] = {}
    resolved_sites = 0
    unresolved = []
    for fn, line, _shape, api, args in sc.unresolved_sites:
        if fn not in tables:
            raw = (src_hle / fn).read_text(errors="ignore")
            tables[fn] = FileTables(H.platform_text(raw, platform)[0])
        t = tables[fn]
        site_ok = True
        for binding in t.bindings_at(line):
            nid = t.string(args[0], binding) if binding is not None else None
            if nid is None or not H.NID_RE.match(nid):
                site_ok = False
                continue
            handler = H.unwrap(args[1])
            if binding:  # glog_thunk<Offset + Is> -> glog_thunk<35>
                handler = re.sub(
                    r"<([^<>]*)>",
                    lambda m, b=binding, t=t: (
                        "<%s>"
                        % (
                            t.evaluate(m.group(1), b)
                            if t.evaluate(m.group(1), b) is not None
                            else m.group(1)
                        )
                    ),
                    handler,
                )
            label = t.string(args[2], binding) or handler
            regs.append(Reg(nid, api, label, handler, fn, line))
        if site_ok:
            resolved_sites += 1
        else:
            unresolved.append(f"{fn}:{line} {args[0]}")
    return {
        "regs": regs,
        "files": len(sc.files),
        "sites_claimed": len(sc.regs) + len(sc.unresolved_sites),
        "unclaimed": [f"{f}:{ln} {why}" for f, ln, why, _t in sc.unclaimed],
        "literal_nids": len({r.nid for r in sc.regs}),
        "literal_handlers": len({r.handler for r in sc.regs}),
        "table_sites": resolved_sites,
        "unresolved": unresolved,
    }


def collect(root: Path, platform: str) -> dict:
    """Per-NID done/placeholder census of `<root>/prosper/src/hle` for `platform`."""
    s = scan(root / "prosper" / "src" / "hle", platform)
    by_nid: dict[str, list[Reg]] = {}
    for r in s["regs"]:
        by_nid.setdefault(r.nid, []).append(r)
    groups: dict[str, dict] = {}
    for nid, regs in by_nid.items():
        real = [r for r in regs if r.api != PLACEHOLDER_API]
        state = "done" if real else "todo"
        owner = max(real or regs, key=lambda r: (r.file, r.line))
        g = groups.setdefault(area_of(owner.file), {"done_nids": [], "todo_nids": []})
        g[state + "_nids"].append({"nid": nid, "name": owner.label, "handler": owner.handler})
    out_groups = []
    for name in sorted(groups):
        g = groups[name]
        for key in ("done_nids", "todo_nids"):
            g[key].sort(key=lambda e: (e["name"], e["nid"]))
        out_groups.append(
            {"name": name, "done": len(g["done_nids"]), "todo": len(g["todo_nids"]), **g}
        )
    done = sum(g["done"] for g in out_groups)
    total = len(by_nid)
    complete = not s["unclaimed"] and not s["unresolved"]
    return {
        "platform": platform,
        "complete": complete,
        "done": done,
        "todo": total - done,
        "total": total,
        "percent": round(100 * done / total, 2) if total else 0.0,
        "coverage": {
            "files_scanned": s["files"],
            "sites_claimed": s["sites_claimed"],
            "sites_unclaimed": s["unclaimed"],
            "literal_nids": s["literal_nids"],
            "literal_handlers": s["literal_handlers"],
            "table_resolved_sites": s["table_sites"],
            "table_resolved_new_nids": total - s["literal_nids"],
            "unresolved_nid_exprs": s["unresolved"],
        },
        "groups": out_groups,
    }


def text(data: dict) -> str:
    """Coverage header, one line per area, and the total."""
    c = data["coverage"]
    lines = [
        f"# HLE progress census, platform={data['platform']}, per registered Sony NID",
        f"#   .cpp files scanned: {c['files_scanned']}",
        f"#   registration sites claimed: {c['sites_claimed']}",
        f"#   NIDs from literal sites: {c['literal_nids']}"
        " (hle_handler_map's 'distinct NIDs registered')",
        "#   NIDs added from constant tables/index_sequence folds:"
        f" {c['table_resolved_new_nids']} (from {c['table_resolved_sites']} sites)",
        f"#   sites UNCLAIMED: {len(c['sites_unclaimed'])}",
        f"#   NID expressions UNRESOLVED: {len(c['unresolved_nid_exprs'])}",
    ]
    lines += [f"#     unclaimed  {u}" for u in c["sites_unclaimed"]]
    lines += [f"#     unresolved {u}" for u in c["unresolved_nid_exprs"]]
    if not data["complete"]:
        lines.append("# INCOMPLETE: every count below is a lower bound")
    lines += [
        f"{g['name']}: {g['done']}/{g['done'] + g['todo']} done ({g['todo']} placeholder-only)"
        for g in data["groups"]
    ]
    lines.append(f"Total: {data['done']}/{data['total']} NIDs done ({data['percent']}%)")
    return "\n".join(lines) + "\n"


def entries(data: dict, state: str) -> dict[str, str]:
    """NID -> 'area::name' for one state, for diffing."""
    return {
        e["nid"]: "{}::{} ({})".format(g["name"], e["name"], e["nid"])
        for g in data["groups"]
        for e in g.get(state + "_nids", [])
    }


def compare(base: dict, head: dict) -> str:
    """Markdown delta between two progress.json files, keyed by NID."""
    base_done, head_done = entries(base, "done"), entries(head, "done")
    base_todo, head_todo = entries(base, "todo"), entries(head, "todo")
    base_all, head_all = {**base_todo, **base_done}, {**head_todo, **head_done}
    implemented = {n: head_done[n] for n in head_done.keys() - base_done.keys()}
    declared = {n: head_todo[n] for n in head_todo.keys() - base_all.keys()}
    regressed = {n: head_todo[n] for n in base_done.keys() & head_todo.keys()}
    removed = {n: base_all[n] for n in base_all.keys() - head_all.keys()}
    delta = round(head["percent"] - base["percent"], 2)
    # ASCII verdict tag: Windows consoles decode stdout as cp1252, where the
    # obvious emoji/icons are unencodable and crash the diff (measured).
    icon = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
    counts = ", ".join(
        f"{n:+} {label}"
        for n, label in (
            (len(implemented), "implemented"),
            (len(declared), "registered as placeholders"),
            (-len(regressed), "reverted"),
            (-len(removed), "removed"),
        )
        if n
    )
    head_line = f"{icon} HLE: {head['percent']}% ({delta:+}%"
    head_line += f", {counts}" if counts else ""
    lines = [head_line + ")", ""]
    if base.get("platform") != head.get("platform"):
        lines += [
            f"WARNING: comparing platform {base.get('platform')} against {head.get('platform')}",
            "",
        ]
    for flag, side in ((base, "base"), (head, "head")):
        if not flag.get("complete", False):
            lines += [f"WARNING: the {side} census is incomplete; its counts are lower bounds", ""]
    for title, items in (
        ("implemented", implemented),
        ("registered as placeholders", declared),
        ("went back to placeholders", regressed),
        ("removed", removed),
    ):
        if not items:
            continue
        lines.append(f"{title} ({len(items)}):")
        ordered = sorted(items.values())
        lines += [f"- {label}" for label in ordered[:50]]
        if len(ordered) > 50:
            lines.append(f"- ... {len(ordered) - 50} more")
        lines.append("")
    if len(lines) == 2:
        lines.append("No HLE coverage change.")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Parse arguments and run the census or the diff."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="checkout root holding prosper/"
    )
    parser.add_argument(
        "--platform",
        choices=sorted(H.PLATFORM_DEFINES),
        help="which #if arm to evaluate (required for a census; registrations differ per platform)",
    )
    parser.add_argument("--output", type=Path, help="directory receiving progress.json")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="diff two progress.json files as markdown",
    )
    args = parser.parse_args()
    if args.compare:
        # Local dev tool: operator-supplied paths are the whole job (precedent:
        # new_probe/verify_report.py). NOSONAR marks the taint-accepted lines.
        base = json.loads(args.compare[0].read_text())  # NOSONAR
        head = json.loads(args.compare[1].read_text())  # NOSONAR
        print(compare(base, head), end="")
        return 0
    if not args.platform:
        parser.error("--platform is required for a census")
    try:
        data = collect(args.root, args.platform)
    except SystemExit as refused:  # hle_handler_map refuses a tree it cannot read
        print(refused, file=sys.stderr)
        return 2
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)  # NOSONAR
        (args.output / "progress.json").write_text(json.dumps(data, indent=2) + "\n")
    print(text(data), end="")
    return 0 if data["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
