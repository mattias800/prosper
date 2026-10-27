"""Origin measurements must retain mixed roots and expose missing diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import placeholder_origin_census as poc  # noqa: E402


def entry(reason, verdict="refused"):
    return {"stage": {"verdict": verdict, "reject": reason}}


def test_multiple_roots_are_separate_and_program_counts_do_not_double_count():
    reason = (
        "fabricated-fresh-mask-source placeholder-roots=["
        "origin=unloaded word=v2 at=entry;origin=unloaded word=v3 at=entry;"
        "origin=spill word=spill-v20:3 at=entry]"
    )
    report = poc.census({"programs": {"compute/a.bin": entry(reason)}})
    assert report["programs_by_origin"] == {"spill": 1, "unloaded": 1}
    assert report["roots_by_origin"] == {"spill": 1, "unloaded": 2}
    assert report["origin_combinations"] == {"spill + unloaded": 1}
    assert report["unclassified"] == []


def test_missing_origin_is_visible_instead_of_becoming_unloaded():
    report = poc.census({"programs": {"compute/a.bin": entry("fabricated-fresh-mask-source")}})
    assert report["unclassified"] == ["compute/a.bin"]
    assert report["programs_by_origin"] == {}


def test_loop_rejection_and_unknown_launch_are_counted():
    reason = (
        "loop-carried v0 is fabricated on the back edge placeholder-roots=["
        "origin=unknown-launch word=s4 at=entry]"
    )
    report = poc.census({"programs": {"compute/a.bin": entry(reason)}})
    assert report["programs_by_origin"] == {"unknown-launch": 1}


def test_live_replay_is_the_measurement_instead_of_the_synthetic_stage():
    row = entry("fabricated-fresh-mask-source")
    row["replay"] = {"verdict": "accepted", "reject": ""}
    report = poc.census({"programs": {"compute/a.bin": row}})
    assert report["entries_surveyed"] == 1
    assert report["policy_one_refusals"] == 0


def test_missing_scalar_publication_stays_in_the_policy_origin_census():
    reason = (
        "unmodelled-scalar-publication placeholder-roots=["
        "origin=unmodelled-scalar-publication word=s20 at=3;"
        "origin=unloaded word=s60 at=entry]"
    )
    report = poc.census({"programs": {"compute/a.bin": entry(reason)}})
    assert report["policy_one_refusals"] == 1
    assert report["programs_by_origin"] == {"unloaded": 1, "unmodelled-scalar-publication": 1}
    assert report["unclassified"] == []


def test_unrelated_hard_data_read_remains_visible_in_the_policy_census():
    reason = "hard-placeholder-read placeholder-roots=[origin=unknown-launch word=s12 at=entry]"
    report = poc.census({"programs": {"compute/a.bin": entry(reason)}})
    assert report["policy_one_refusals"] == 1
    assert report["programs_by_origin"] == {"unknown-launch": 1}
    assert report["unclassified"] == []
