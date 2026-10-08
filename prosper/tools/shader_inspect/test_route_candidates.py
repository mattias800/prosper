"""pytest arms for route_candidates.py: the route-2 candidate classifier over reason names.

Each arm builds its inputs by hand from shader_inspect's own reason vocabulary (kWaveReasonNames in
shader_inspect.cpp) and one arm reads that table to prove the classifier knows every name the
decoder can emit, so a renamed reason cannot silently turn a candidate into an uncovered one.
"""

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import route_candidates as rc  # noqa: E402


def test_each_rewrite_class_is_a_candidate_on_its_own():
    assert rc.classify(["readlane64"]) == (rc.CANDIDATE, ["uniform-readlane"], [])
    assert rc.classify(["wave-any"]) == (rc.CANDIDATE, ["uniform-vote"], [])
    assert rc.classify(["wave-ballot"]) == (rc.CANDIDATE, ["compaction-ballot"], [])


def test_a_lane_id_is_covered_only_alongside_a_ballot():
    # The slot-allocation shape: MBCNT plus the ballot it indexes.
    assert rc.classify(["lane-id", "wave-ballot"]) == (rc.CANDIDATE, ["compaction-ballot"], [])
    # Alone it is lane-dependent data no rewrite makes independent (the mutation arm).
    assert rc.classify(["lane-id"]) == (rc.OTHER_ROUTE, [], ["lane-id"])
    assert rc.classify(["lane-id", "wave-any"]) == (rc.OTHER_ROUTE, ["uniform-vote"], ["lane-id"])


def test_a_scalar_reduce_is_a_wave_any_result_so_it_is_covered_with_one():
    # The recompiler reports scalar-reduce for a WaveAny that reaches a scalar consumer, and the
    # uniform-vote rewrite is exact whoever consumes the vote.
    assert rc.classify(["wave-any", "scalar-reduce"]) == (rc.CANDIDATE, ["uniform-vote"], [])
    # Alone it is not a vote at all: the mutation arm that keeps the pairing from over-covering.
    assert rc.classify(["scalar-reduce"]) == (rc.OTHER_ROUTE, [], ["scalar-reduce"])
    assert rc.classify(["wave-ballot", "scalar-reduce"]) == (
        rc.OTHER_ROUTE,
        ["compaction-ballot"],
        ["scalar-reduce"],
    )


@pytest.mark.parametrize("blocker", ["shuffle", "dpp-row16", "permlane32"])
def test_one_uncovered_reason_keeps_the_shader_out(blocker):
    verdict, _, uncovered = rc.classify(["wave-any", blocker])
    assert verdict == rc.OTHER_ROUTE
    assert uncovered == [blocker]


def test_an_unknown_reason_bit_falls_out_of_candidate():
    verdict, _, uncovered = rc.classify(["wave-any", "unknown-bit9"])
    assert verdict == rc.OTHER_ROUTE
    assert uncovered == ["unknown-bit9"]


def test_an_empty_set_is_none_not_candidate():
    for field in ("none", "absent", "", None):
        assert rc.classify(rc.parse_names(field))[0] == rc.NONE


def test_tally_counts_each_verdict_class_and_blocker():
    verdicts, classes, blockers = rc.tally(
        ["wave-any", "wave-any,readlane64", "shuffle,wave-ballot", "lane-id", "none"]
    )
    assert verdicts[rc.CANDIDATE] == 2
    assert verdicts[rc.OTHER_ROUTE] == 2
    assert verdicts[rc.NONE] == 1
    assert classes == {"uniform-vote": 2, "uniform-readlane": 1}
    assert blockers == {"shuffle": 1, "lane-id": 1}
    text = rc.report(["wave-any", "shuffle"])
    assert "candidate (every reason has a rewrite class) = 1" in text
    assert "admits nothing" in text


def test_every_name_the_decoder_emits_is_known_to_the_classifier():
    source = (HERE.parent / "shader_inspect.cpp").read_text(encoding="utf-8")
    emitted = set(re.findall(r'\{kFragmentWaveReason\w+,\s*"([a-z0-9-]+)"\}', source))
    assert emitted, "kWaveReasonNames not found in shader_inspect.cpp"
    known = set(rc.CANDIDATE_CLASS) | {rc.PAIRED_WITH_BALLOT, rc.PAIRED_WITH_ANY}
    assert set(rc.CANDIDATE_CLASS) <= emitted, "a candidate name the decoder no longer emits"
    assert rc.PAIRED_WITH_BALLOT in emitted
    assert rc.PAIRED_WITH_ANY in emitted
    for name in emitted - known:
        # Every other name must classify as uncovered, never silently as a candidate.
        assert rc.classify([name])[0] == rc.OTHER_ROUTE


def test_the_census_csv_reader_selects_analysed_wide_wave_rows(tmp_path):
    header = "name,recompiled,table_dependent,endpgm,required_subgroup_size,reasons,names,reason_set_admissible,error"
    rows = [
        "a_PS_1.bin,1,0,1,64,0x2,wave-any,1,",  # wide, candidate
        "b_PS_2.bin,1,0,1,32,0x0,none,1,",  # not wide: excluded
        "c_PS_3.bin,0,1,0,0,,,0,",  # not lowered: excluded, not 'requires nothing'
        "d_PS_4.bin,1,0,1,64,0x20,shuffle,0,",  # wide, other-route
        "e_PS_5.bin,0,0,0,0,,,0,no sentinel",  # error: excluded
    ]
    path = tmp_path / "census.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    assert rc.names_from_census_csv(str(path)) == ["wave-any", "shuffle"]
    assert rc.main([str(path)]) == 0


def test_the_census_argument_must_be_an_existing_csv_file(tmp_path):
    text = tmp_path / "census.txt"
    text.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        rc.resolve_census_csv(str(text))
    with pytest.raises(ValueError):
        rc.resolve_census_csv(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        rc.resolve_census_csv(str(tmp_path / "missing.csv"))
    assert rc.main([str(tmp_path / "missing.csv")]) == 2
