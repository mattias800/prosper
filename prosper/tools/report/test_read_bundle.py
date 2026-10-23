"""Tests for read_bundle.py. Synthetic bundles, no title, no app."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_bundle as rb  # noqa: E402


def bundle(**over):
    record = {
        "type": "report",
        "schema": 1,
        "tool": "prosper-app",
        "build_revision": "a" * 40,
        "build_source_identity": "b" * 64,
        "os": "linux",
        "title_id": "PPSA00000",
        "switches": {},
    }
    record.update(over)
    return record


def write_bundle(tmp_path, record):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


def test_valid_bundle_loads_and_triages(tmp_path):
    record = rb.load_bundle(write_bundle(tmp_path, bundle()))
    lines = rb.triage_lines(record)
    assert lines == [
        "tool: prosper-app",
        "build: aaaaaaaaaaaa (source identity present: True)",
        "os: linux",
        "title: PPSA00000",
        "switches: 0 PROSPER_* in force",
    ]


def test_empty_title_and_switches_are_reported_not_defaulted(tmp_path):
    record = rb.load_bundle(write_bundle(tmp_path, bundle(title_id="", switches={})))
    lines = rb.triage_lines(record)
    assert any("no title identity" in line for line in lines)
    assert any("0 PROSPER_*" in line for line in lines)


def test_unknown_schema_wrong_type_and_malformed_bundles_refuse(tmp_path):
    good = write_bundle(tmp_path, bundle())
    assert rb.load_bundle(good)["schema"] == 1
    for change in ({"schema": 2}, {"type": "run"}, {"build_revision": ""}, {"switches": []}):
        bad_record = bundle(**change)
        bad = write_bundle(tmp_path, bad_record)
        with pytest.raises(rb.BundleError):
            rb.load_bundle(bad)
    missing = tmp_path / "missing.json"
    with pytest.raises(rb.BundleError):
        rb.load_bundle(str(missing))
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(rb.BundleError):
        rb.load_bundle(str(garbage))


def test_cli_refuses_without_printing_triage(tmp_path, capsys):
    assert rb.main([]) == 2
    assert rb.main([write_bundle(tmp_path, bundle())]) == 0
    out = capsys.readouterr().out
    assert "prosper-app" in out
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert rb.main([str(bad)]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"schema": True},
        {"schema": 1.0},
        {"tool": []},
        {"tool": ""},
        {"build_source_identity": {}},
        {"build_source_identity": ""},
        {"os": "unknown"},
        {"title_id": None},
        {"title_id": "../game"},
        {"switches": {"PATH": "1"}},
        {"switches": {"PROSPER_GFXLOG": 1}},
    ],
)
def test_malformed_section_contracts_refuse(tmp_path, change):
    with pytest.raises(rb.BundleError):
        rb.load_bundle(write_bundle(tmp_path, bundle(**change)))


def test_invalid_utf8_refuses_without_a_traceback(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_bytes(b'{"type":"report","tool":"\xff"}')
    assert rb.main([str(path)]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("filename", "displayed"),
    [
        (
            "missing-\x1b]52;c;payload\x07\nforged.json",
            "missing-\\u001b]52;c;payload\\u0007\\nforged.json",
        ),
        ("missing-\r\t\x7f\x9b.json", "missing-\\r\\t\\u007f\\u009b.json"),
        ("missing-\u2028\u202e.json", "missing-\\u2028\\u202e.json"),
    ],
)
def test_cli_refusal_displays_untrusted_filename_without_terminal_controls(
    tmp_path, capsys, filename, displayed
):
    # Missing files still carry their supplied name in the refusal diagnostic. A report
    # filename must not execute terminal commands or forge another diagnostic line.
    assert rb.main([str(tmp_path / filename)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("refused:")
    assert len(captured.err.splitlines()) == 1
    assert displayed in captured.err


def test_deeply_nested_switch_value_refuses_without_a_traceback(tmp_path, capsys):
    # Some supported runtimes reach the JSON decoder's recursion limit first; others can
    # parse this depth and reject the non-string switch value. Both must refuse cleanly.
    nested_value = "[" * 20000 + "0" + "]" * 20000
    record = json.dumps(bundle(switches={}))[:-2]
    path = tmp_path / "report.json"
    path.write_text(record + '"PROSPER_NESTED":' + nested_value + "}}", encoding="utf-8")
    assert rb.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("refused:")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("tool", "\x1b]52;c;payload\x07\nspoof", "tool: \\u001b]52;c;payload\\u0007\\nspoof"),
        ("tool", "x\r\ty\x7f\x9b", "tool: x\\r\\ty\\u007f\\u009b"),
        ("tool", "name\u2028\u202e", "tool: name\\u2028\\u202e"),
        (
            "build_revision",
            "\x1b[31mforged\nignored",
            "build: \\u001b[31mforged\\n (source identity present: True)",
        ),
        (
            "build_revision",
            "x\r\ty\x7f\x9b",
            "build: x\\r\\ty\\u007f\\u009b (source identity present: True)",
        ),
        (
            "build_revision",
            "name\u2028\u202e",
            "build: name\\u2028\\u202e (source identity present: True)",
        ),
    ],
)
def test_cli_displays_untrusted_metadata_without_terminal_controls(
    tmp_path, capsys, field, value, expected
):
    # A shared report is untrusted input. OSC/CSI, newlines and Unicode formatting must be
    # visible data, never terminal commands or an extra triage line.
    path = write_bundle(tmp_path, bundle(**{field: value}))
    assert rb.main([path]) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 5
    assert expected in output.splitlines()
