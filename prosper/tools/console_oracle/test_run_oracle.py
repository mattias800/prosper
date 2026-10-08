"""Tests for run_oracle.py: the parts that need no console.

Covers the cases-file parser, the C header generator (round-tripped through Python's own string
literal rules, which match C's for the escapes used), the payload-output parser, the golden file
format, the path and argument validation, and the committed data under tests/data/console_oracle:
every cases file must have a golden whose spec columns still match, and every measured case must have
run on the console.
"""

import ast

import pytest
import run_oracle as ro

DATA = ro.DATA_DIR


def test_parse_cases_skips_comments_and_blank_lines():
    cases = ro.parse_cases(
        "# c\n\nrtc_a\tlibSceRtc.sprx\tsceRtcIsLeapYear\ti:2000\nrtc_b\t-\tstrlen\ts:x\tr64\n"
    )
    assert [c.id for c in cases] == ["rtc_a", "rtc_b"]
    assert cases[0].expect == ""
    assert cases[1].expect == "r64"


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
    assert out.done
    assert out.ran == 2
    assert out.results["c1"] == ["ok", "0x0000000000000001", "-", "a0=aa"]
    assert out.results["c2"][0] == "fault:11"


def test_parse_output_without_done_line_marks_the_run_incomplete():
    out = ro.parse_output("R\tc1\tok\t0x0\t-\n")
    assert not out.done
    assert out.ran is None


def test_golden_round_trip_and_missing_case_marker():
    cases = ro.parse_cases("a\tlibX.sprx\tf\ti:1\nb\tlibX.sprx\tg\t-\tret\n")
    output = ro.parse_output("R\ta\tok\t0x0000000000000005\t-\ta0=00\n# done ran=1\n")
    text = ro.build_golden("x.cases.tsv", cases, output, "2026-10-08")
    assert text.startswith(ro.GOLDEN_MAGIC)
    rows = {r["id"]: r for r in ro.parse_golden(text)}
    assert rows["a"]["status"] == "ok"
    assert rows["a"]["ret"] == "0x0000000000000005"
    assert rows["a"]["buffers"] == ["a0=00"]
    # a case the payload never reached is visible, not dropped
    assert rows["b"]["status"] == "missing"
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
    assert "a0=volatile" in once
    assert "0x7ffd1234" not in once
    assert ro.scrub_golden_text(once) == once


def test_short_golden_line_is_an_error():
    with pytest.raises(ValueError, match="short golden line"):
        ro.parse_golden("a\tb\tc\n")


# --- argument and path validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "host, sdk",
    [
        ("console; ls x", "/opt/ps5-payload-sdk"),
        ("console name", "/opt/ps5-payload-sdk"),
        ("192.168.0.2", "/opt/ps5 payload sdk"),
        ("192.168.0.2", "$(whoami)"),
        ("192.168.0.2", "/opt/sdk;ls"),
    ],
)
def test_unsafe_host_or_sdk_is_rejected_before_anything_runs(host, sdk):
    with pytest.raises(SystemExit, match="only"):
        ro.run_on_console("", sdk, host, 9021, 5)


def test_plain_host_and_sdk_are_accepted():
    ro.check_connection_args("192.168.0.2", "/opt/ps5-payload-sdk")
    ro.check_connection_args("my-console.local", "/mnt/c/dev/ps5/ps5-payload-sdk")
    ro.check_connection_args("fe80::1", "relative/sdk")


def test_a_failed_build_is_reported_with_an_sdk_hint(monkeypatch):
    class Failed:
        returncode = 2
        stdout = ""
        stderr = "Makefile:9: toolchain/prospero.mk: No such file or directory"

    monkeypatch.setattr(ro.subprocess, "run", lambda *args, **kwargs: Failed())
    with pytest.raises(SystemExit, match="installed ps5-payload-sdk"):
        ro.run_on_console("", "/opt/not-an-sdk", "192.168.0.2", 9021, 5)


def test_family_paths_stay_inside_the_data_directory():
    cases, golden = ro.family_paths("rtc")
    base = ro.DATA_DIR.resolve()
    assert cases == base / "rtc.cases.tsv"
    assert golden == base / "rtc.golden.tsv"


@pytest.mark.parametrize("family", ["../x", "a/b", "a\\b", "a.b", "", "rtc\n", "..", "x y"])
def test_a_family_that_is_not_a_bare_name_is_rejected(family):
    with pytest.raises(ValueError, match="bad family name"):
        ro.family_paths(family)


# --- the committed data -------------------------------------------------------------------------
# These are the checks that stop a cases file and its measurement drifting apart.
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
