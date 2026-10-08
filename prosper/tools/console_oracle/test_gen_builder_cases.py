"""Tests for gen_builder_cases.py: which exports are probed as builders, and the cases they become."""

import gen_builder_cases as g
import pytest
import run_oracle as ro

STUB = """
.global sceAgcDcbDrawIndexAuto
.type sceAgcDcbDrawIndexAuto @function
.global sceAgcDcbDrawIndexAutoGetSize
.global sceAgcAcbCopyData
.global sceAgcCbNop
.global sceAgcDcbSetIndexSize
.global sceAgcDcbSetIndexSize
.global sceAgcSetCxRegIndirectPatchSetAddress
.global sceAgcDcbRewindPatchThing
.global sceAgcAcbRewind
.global sceAgcCreateShader
.global sceAgcInit
.global sceAgcGetRegisterDefaults
.global sceAgcDcbwhatever
"""


def test_exported_names_are_unique_and_sorted():
    names = g.exported_names(STUB)
    assert names == sorted(set(names))
    assert names.count("sceAgcDcbSetIndexSize") == 1


def test_only_command_buffer_builders_are_probed():
    got = g.builders(g.exported_names(STUB))
    assert got == [
        "sceAgcAcbCopyData",
        "sceAgcCbNop",
        "sceAgcDcbDrawIndexAuto",
        "sceAgcDcbSetIndexSize",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "sceAgcDcbDrawIndexAutoGetSize",  # a size query takes no buffer: measured in the agc family
        "sceAgcSetCxRegIndirectPatchSetAddress",  # patchers store through a pointer into an old packet
        "sceAgcAcbRewind",  # rewind stores through the packet it rewinds
        "sceAgcCreateShader",  # reads a caller-supplied structure, would dereference garbage
        "sceAgcInit",
        "sceAgcDcbwhatever",  # not a builder name: the packet word must start with a capital
    ],
)
def test_names_that_are_not_safe_to_probe_are_excluded(name):
    assert g.builders([name]) == []


def test_every_builder_gets_the_three_probes_and_the_cases_parse():
    text = g.render(g.exported_names(STUB))
    cases = ro.parse_cases(text)
    assert len(cases) == 4 * len(g.PROBES)
    first = cases[0]
    assert first.lib == "libSceAgc.sprx"
    assert first.expect == "r64,retoff:0"
    assert first.args.startswith(f"dcb:{g.DCB_DWORDS},")
    # id carries the builder and the probe, and is unique across the file (parse_cases enforces that)
    assert {c.id for c in cases if c.func == "sceAgcCbNop"} == {
        "agcbuild_CbNop_a",
        "agcbuild_CbNop_b",
        "agcbuild_CbNop_c",
    }


def test_every_probe_has_at_most_five_value_arguments_after_the_buffer():
    # the payload passes six registers: the descriptor plus five values
    for _, args in g.PROBES:
        assert len(args.split(",")) == 5


def test_the_probe_values_are_distinct_enough_to_read_off_a_packet():
    a = [int(t.split(":")[1], 16) for t in dict(g.PROBES)["a"].split(",")]
    assert len(set(a)) == len(a)
    assert all(v > 0xFFFF for v in a)  # recognisable among small packet fields
