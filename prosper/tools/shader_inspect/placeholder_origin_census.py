#!/usr/bin/env python3
"""Count diagnostic roots in an existing corpus survey; never run or modify shaders."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import shader_corpus

ROOT = re.compile(r"\borigin=([a-z-]+) word=([^ ;\]]+) at=([^ ;\]]+)")


def census(survey: dict) -> dict:
    programs = {}
    origins = Counter()
    roots_by_origin = Counter()
    combinations = Counter()
    missing = []
    for name, row in sorted(survey["programs"].items()):
        if Path(name).is_absolute():
            raise ValueError("survey program names must be relative")
        basis, answer, _ = shader_corpus.decided(row, row)
        reason = answer.get("reject", "")
        policy_one = (
            "fabricated-fresh-mask-source" in reason
            or "unmodelled-scalar-publication" in reason
            or "hard-placeholder-read" in reason
            or ("loop-carried" in reason and "fabricated" in reason)
        )
        if answer["verdict"] != shader_corpus.REFUSED or not policy_one:
            continue
        roots = sorted(set(ROOT.findall(reason)))
        kinds = sorted({root[0] for root in roots})
        if not roots:
            missing.append(name)
        origins.update(kinds)
        roots_by_origin.update(root[0] for root in roots)
        combinations.update([" + ".join(kinds) if kinds else "unclassified"])
        programs[name] = {
            "basis": basis,
            "origins": kinds,
            "roots": [dict(zip(("origin", "word", "at"), root, strict=True)) for root in roots],
        }
    return {
        "entries_surveyed": len(survey["programs"]),
        "policy_one_refusals": len(programs),
        "programs_by_origin": dict(sorted(origins.items())),
        "roots_by_origin": dict(sorted(roots_by_origin.items())),
        "origin_combinations": dict(sorted(combinations.items())),
        "unclassified": missing,
        "programs": programs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("survey", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = census(json.loads(args.survey.read_text()))
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"policy-1 refusals: {report['policy_one_refusals']}")
    for origin, count in report["programs_by_origin"].items():
        print(f"{origin}: {count} programs (groups can overlap)")
    print(f"unclassified: {len(report['unclassified'])}")
    return 1 if report["unclassified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
