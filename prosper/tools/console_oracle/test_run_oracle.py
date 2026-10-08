"""Tests for run_oracle.py: the parts that need no console.

Covers the cases-file parser, the C header generator (round-tripped through Python's own string
literal rules, which match C's for the escapes used), the payload-output parser, the golden file
format, and the committed data under tests/data/console_oracle: every cases file must have a golden
whose spec columns still match, and every measured case must have run on the console.
"""

import ast
from pathlib import Path

import pytest
import run_oracle as ro

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "console_oracle"


def test_parse_cases_skips_comments_and_blank_lines():
    cases = ro.parse_cases(
        "# c\n\nrtc_a\tlibSceRtc.sprx\tsceRtcIsLeapYear\ti:2000\nrtc_b\t-\tstrlen\ts:x\tr64\n"
    )
    assert [c.id for c in cases] == ["rtc_a", "rtc_b"]
    assert cases[0].expect == "" and cases[1].expect == "r64"


@pytest.mark.parametrize(
    "text, why",
    [
        ("only\ttwo\tfields\n", "expected id"),
        ("bad id\t-\tf\t-\n", "bad case id"),
        ("a\t-\tf\t-\na\t-\tg\t-\n", "duplicate"),
    ],
)
def test_parse_cases_rejects_bad_input(text, why):
    with pytest.raises(ValueError, match=why):
        ro.parse_cases(text)


def test_c_header_round_trips_special_characters():
    text = 'id\tlib\tfunc\ts:a\\x2cb "quoted" \\ back\targ\n# comment\r\n'
    header = ro.render_cases_header(text)
    body = header.split("kCases[] =\n", 1)[1].rsplit(";", 1)[0]
    literal = "".join(ast.literal_eval(line.strip()) for line in body.splitlines())
    assert literal == text


def test_c_header_for_empty_cases_is_valid_c():
    assert 'kCases[] =\n    ""' in ro.render_cases_header("")


def test_parse_output_picks_result_lines_out_of_deploy_noise():
    out = ro.parse_output(
        "prospero-clang -Wall ...\n# lib_oracle v1\n"
        "R\tc1\tok\t0x0000000000000001\t-\ta0=aa\n"
        "R\tc2\tfault:11\t0x0000000000000000\t-\n"
        "garbage line\n# done ran=2\n"
    )
    assert out.done and out.ran == 2
    assert out.results["c1"] == ["ok", "0x0000000000000001", "-", "a0=aa"]
    assert out.results["c2"][0] == "fault:11"


def test_parse_output_without_done_line_marks_the_run_incomplete():
    out = ro.parse_output("R\tc1\tok\t0x0\t-\n")
    assert not out.done and out.ran is None


def test_golden_round_trip_and_missing_case_marker():
    cases = ro.parse_cases("a\tlibX.sprx\tf\ti:1\nb\tlibX.sprx\tg\t-\tret\n")
    output = ro.parse_output("R\ta\tok\t0x0000000000000005\t-\ta0=00\n# done ran=1\n")
    text = ro.build_golden("x.cases.tsv", cases, output, "2026-10-08")
    assert text.startswith(ro.GOLDEN_MAGIC)
    rows = {r["id"]: r for r in ro.parse_golden(text)}
    assert rows["a"]["status"] == "ok" and rows["a"]["ret"] == "0x0000000000000005"
    assert rows["a"]["buffers"] == ["a0=00"]
    assert (
        rows["b"]["status"] == "missing"
    )  # a case the payload never reached is visible, not dropped
    assert rows["b"]["expect"] == "ret"


def test_golden_spec_problems_reports_drift_in_every_direction():
    cases_text = "a\tlibX.sprx\tf\ti:1\nb\tlibX.sprx\tg\t-\n"
    output = ro.parse_output("R\ta\tok\t0x0\t-\nR\tb\tok\t0x0\t-\n")
    golden = ro.build_golden("x", ro.parse_cases(cases_text), output, "d")
    assert ro.golden_spec_problems(golden, cases_text) == []
    edited = "a\tlibX.sprx\tf\ti:2\nc\tlibX.sprx\th\t-\n"  # a's args changed, b removed, c added
    problems = "\n".join(ro.golden_spec_problems(golden, edited))
    assert "a: args differs" in problems
    assert "b: in golden, absent from cases file" in problems
    assert "c: in cases file, absent from golden" in problems


def test_scrub_replaces_only_the_values_the_replay_never_compares():
    # `ret` and `none` cases do not compare buffers: their bytes (a clock reading, say) must not be kept.
    assert ro.scrub_volatile("ret", ["ok", "0x0", "-", "a0=b565d514531de300"]) == [
        "ok",
        "0x0",
        "-",
        "a0=volatile",
    ]
    assert ro.scrub_volatile("r64,none", ["ok", "0x5f259c06", "-", "a0=aa", "a1=bb"]) == [
        "ok",
        "0x5f259c06",
        "-",
        "a0=volatile",
        "a1=volatile",
    ]
    # a retoff case compares the offset, so the raw pointer in the return register is dropped
    assert ro.scrub_volatile("r64,retoff:0", ["ok", "0x000000088000a2e2", "ptr+2"]) == [
        "ok",
        ro.RAW_RET_PLACEHOLDER,
        "ptr+2",
    ]
    # a case that compares everything keeps everything
    kept = ["ok", "0x5", "-", "a0=0102"]
    assert ro.scrub_volatile("", kept) == kept
    assert ro.scrub_volatile("r64", kept) == kept


def test_scrub_is_idempotent_and_keeps_headers():
    text = (
        "# header\n"
        "a\tlib\tf\ti:1\tret\tok\t0x0\t-\ta0=ff\n"
        "b\tlib\tg\ts:x\tr64,retoff:0\tok\t0x7ffd1234\tptr+1\n"
    )
    once = ro.scrub_golden_text(text)
    assert once.splitlines()[0] == "# header"
    assert "a0=volatile" in once and "0x7ffd1234" not in once
    assert ro.scrub_golden_text(once) == once


def test_short_golden_line_is_an_error():
    with pytest.raises(ValueError, match="short golden line"):
        ro.parse_golden("a\tb\tc\n")


def test_run_on_console_rejects_a_directory_that_is_not_an_sdk(tmp_path):
    with pytest.raises(SystemExit, match="not an installed ps5-payload-sdk"):
        ro.run_on_console("", str(tmp_path), "host", 9021, 5)


# The committed data. These are the checks that stop a cases file and its measurement drifting apart.
CASES_FILES = sorted(DATA.glob("*.cases.tsv"))


def test_there_are_committed_cases():
    assert CASES_FILES, f"no *.cases.tsv under {DATA}"


@pytest.mark.parametrize("cases_path", CASES_FILES, ids=lambda p: p.name)
def test_committed_golden_matches_its_cases_and_was_fully_measured(cases_path):
    golden_path = cases_path.with_name(cases_path.name.replace(".cases.tsv", ".golden.tsv"))
    assert golden_path.exists(), f"{golden_path.name} missing: run run_oracle.py on a console"
    golden = golden_path.read_text()
    assert ro.golden_spec_problems(golden, cases_path.read_text()) == []
    not_ok = [(r["id"], r["status"]) for r in ro.parse_golden(golden) if r["status"] != "ok"]
    assert not_ok == [], f"cases the console could not run: {not_ok}"


@pytest.mark.parametrize("cases_path", CASES_FILES, ids=lambda p: p.name)
def test_committed_golden_holds_no_uncompared_volatile_values(cases_path):
    golden_path = cases_path.with_name(cases_path.name.replace(".cases.tsv", ".golden.tsv"))
    golden = golden_path.read_text()
    assert ro.scrub_golden_text(golden) == golden, (
        f"{golden_path.name} holds values the replay never compares (a capture-time reading or a "
        "process address); run run_oracle.py --scrub on it"
    )
