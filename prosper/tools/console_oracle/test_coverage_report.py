"""Tests for coverage_report.py: offline coverage and candidate analysis.

Uses synthetic inputs to verify:
- Classifier categorisation:
  - 'Save' -> excluded-state
  - 'Net' -> excluded-network
  - Plain name -> candidate
- Committed/synthetic cases coverage:
  - Function present in cases file is not reported as uncovered
- Mutation testing:
  - Mutating the classifier (e.g. removing 'Save' pattern) makes the test fail
- Filtering by library (--only) and --registered-only behavior.
"""

from __future__ import annotations

import pathlib
import pytest

import prosper.tools.console_oracle.coverage_report as cr


def test_classifier_puts_name_containing_save_in_excluded_state():
    """Verify that names containing Save are categorized as excluded-state."""
    assert cr.classify_export("sceSaveDataMount", "libSceSaveData.sprx") == cr.CLASS_STATE
    assert cr.classify_export("sceKernelSaveContext", "libkernel.sprx") == cr.CLASS_STATE
    assert cr.classify_export("customSaveHelper", "?") == cr.CLASS_STATE


def test_classifier_puts_name_containing_net_in_excluded_network():
    """Verify that names containing Net are categorized as excluded-network."""
    assert cr.classify_export("sceNetSocket", "libSceNet.sprx") == cr.CLASS_NETWORK
    assert cr.classify_export("NetInit", "?") == cr.CLASS_NETWORK
    assert cr.classify_export("sceNetPoolCreate", "?") == cr.CLASS_NETWORK


def test_classifier_puts_plain_name_in_candidate():
    """Verify that neutral, non-excluded names are categorized as candidate."""
    assert cr.classify_export("sceRtcGetCurrentTick", "libSceRtc.sprx") == cr.CLASS_CANDIDATE
    assert cr.classify_export("sceKernelGetProcessTime", "libkernel.sprx") == cr.CLASS_CANDIDATE
    assert cr.classify_export("custom_plain_function", "?") == cr.CLASS_CANDIDATE


def test_classifier_other_categories():
    """Verify identity and floating-point/variadic classifications."""
    assert cr.classify_export("sceUserServiceGetUserId", "libSceUserService.sprx") == cr.CLASS_IDENTITY
    assert cr.classify_export("sceKernelGetDeviceId", "libkernel.sprx") == cr.CLASS_IDENTITY
    assert cr.classify_export("printf", "libSceLibcInternal.sprx") == cr.CLASS_FP_VARIADIC
    assert cr.classify_export("vsnprintf", "libSceLibcInternal.sprx") == cr.CLASS_FP_VARIADIC
    assert cr.classify_export("asprintf", "?") == cr.CLASS_FP_VARIADIC
    assert cr.classify_export("sceKernelDebugRaiseException", "libkernel.sprx") == cr.CLASS_STATE


def test_name_present_in_cases_file_is_not_reported_as_uncovered():
    """Verify that functions already covered by a cases file are recorded as covered."""
    db_entries = [
        ("NID0001", "funcCovered"),
        ("NID0002", "funcUncovered"),
    ]
    cases_funcs = {
        "funcCovered": "libTest.sprx",
    }
    reg_names = {"funcCovered", "funcUncovered"}

    data = cr.generate_coverage(
        db_entries=db_entries,
        cases_funcs=cases_funcs,
        reg_names=reg_names,
    )

    # funcCovered should be under libTest.sprx and marked covered
    assert "libTest.sprx" in data
    assert data["libTest.sprx"]["cases_covered"] == 1
    assert data["libTest.sprx"]["candidates"] == []

    # funcUncovered should be under '?' and categorized as candidate
    assert "?" in data
    assert data["?"]["cases_covered"] == 0
    assert "funcUncovered" in data["?"]["candidates"]


def test_filter_registered_only_limits_candidates():
    """Verify --registered-only filters candidate lists to registered HLE functions."""
    db_entries = [
        ("NID0001", "funcRegisteredCandidate"),
        ("NID0002", "funcUnregisteredCandidate"),
    ]
    cases_funcs = {}
    reg_names = {"funcRegisteredCandidate"}

    # Without registered_only: both are candidates
    data_all = cr.generate_coverage(
        db_entries=db_entries,
        cases_funcs=cases_funcs,
        reg_names=reg_names,
        registered_only=False,
    )
    assert len(data_all["?"]["candidates"]) == 2

    # With registered_only: only the registered function is in candidates
    data_reg = cr.generate_coverage(
        db_entries=db_entries,
        cases_funcs=cases_funcs,
        reg_names=reg_names,
        registered_only=True,
    )
    assert data_reg["?"]["candidates"] == ["funcRegisteredCandidate"]


def test_nid_libs_maps_library():
    """Verify that --nid-libs properly attributes library to uncovered entries."""
    db_entries = [
        ("NID_SAMPLE", "sceAudioOutInit"),
    ]
    cases_funcs = {}
    reg_names = {"sceAudioOutInit"}
    nid_libs = {"NID_SAMPLE": "libSceAudioOut.sprx"}

    data = cr.generate_coverage(
        db_entries=db_entries,
        cases_funcs=cases_funcs,
        reg_names=reg_names,
        nid_libs=nid_libs,
    )

    assert "libSceAudioOut.sprx" in data
    assert data["libSceAudioOut.sprx"]["db_exports"] == 1
    assert "sceAudioOutInit" in data["libSceAudioOut.sprx"]["candidates"]


def test_classifier_mutation_fails(monkeypatch):
    """Demonstrate that mutating the classifier (removing Save) causes test failure."""
    # Mutate STATE_PATTERNS by removing "Save"
    mutated_patterns = [p for p in cr.STATE_PATTERNS if p != "Save"]
    monkeypatch.setattr(cr, "STATE_PATTERNS", mutated_patterns)

    # With 'Save' removed from patterns, sceSaveDataMount must fail to classify as excluded-state
    classification = cr.classify_export("sceSaveDataMount", "libSceSaveData.sprx")
    assert classification != cr.CLASS_STATE
