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
    assert any("prosper-app" in line for line in lines)
    assert any("PPSA00000" in line for line in lines)


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
