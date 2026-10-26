"""Pin truthful stack capture and exact fixture replay inputs for the #5104 diagnostic."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "system_error_trace", Path(__file__).with_name("windows_system_error_trace.py")
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def test_capture_requires_breakpoint_exit_and_an_actual_throw_frame():
    stack = "PROSPER_SYSTEM_ERROR_STACK\n#0  std::__throw_system_error (int) at error.cc:1\n"
    assert trace.captured_stack(42, stack)
    assert not trace.captured_stack(0, stack)
    assert not trace.captured_stack(42, "PROSPER_SYSTEM_ERROR_STACK\nNo stack.\n")
    assert not trace.captured_stack(42, "#0 std::__throw_system_error (int)\n")
    assert not trace.captured_stack(42, stack.replace("#0", "#1"))


@pytest.mark.parametrize("mode,captured", [("throw", True), ("normal", False), ("signal", False)])
def test_real_debugger_does_not_require_a_main_symbol(tmp_path, mode, captured):
    compiler, debugger = shutil.which("g++"), shutil.which("gdb")
    if not compiler or not debugger:
        pytest.skip("real debugger control requires g++ and gdb")
    source = tmp_path / "control.cpp"
    source.write_text(
        "#include <bits/functexcept.h>\n#include <csignal>\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc > 1 && argv[1][0] == 't') std::__throw_system_error(22);\n"
        "  if (argc > 1 && argv[1][0] == 's') std::raise(SIGSEGV);\n"
        "  return 0;\n}\n"
    )
    executable = tmp_path / ("control.exe" if os.name == "nt" else "control")
    subprocess.run(
        [compiler, "-O0", "-s", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    env = dict(os.environ, DEBUGINFOD_URLS="")
    # Establish the missing-symbol condition independently of the script under test.
    status, output = trace.run_debugger(
        [debugger, "--batch", "--nx", "-ex", "info address main", "--args", str(executable)],
        str(tmp_path),
        env,
    )
    assert status == 1 and '"main"' in output, output
    script = Path(__file__).with_name("system_error_stack.gdb")
    status, output = trace.run_debugger(
        [
            debugger,
            "--batch",
            "--nx",
            "-iex",
            "set auto-load off",
            "-x",
            str(script),
            "--args",
            str(executable),
            mode,
        ],
        str(tmp_path),
        env,
    )
    assert trace.captured_stack(status, output) == captured, output
    if captured:
        assert status == 42
    else:
        assert status == 0 and "PROSPER_NO_SYSTEM_ERROR_STACK" in output, output
        assert "PROSPER_SYSTEM_ERROR_STACK\n" not in output


def test_only_actual_failed_fixture_names_are_selected():
    names = [
        "WaitProducerTrace.One",
        "WaitProducerTraceDisabled.One",
        "WaitProducerTraceSelective.One",
        "WaitProducerTraceBudget.MissingLatestTagAndBoundedRecordsStayExplicit",
        "WaitProducerTrace.Passed",
        "NotWaitProducerTrace.One",
        "indirect_pointer_descriptor_range",
    ]
    tests = [{"name": name, "command": ["owned executable", name]} for name in names]
    selected = trace.selected_tests({"tests": tests}, set(names) - {"WaitProducerTrace.Passed"})
    assert [test["name"] for test in selected] == names[:4]
    assert selected[0]["command"] == tests[0]["command"]


def test_replay_population_is_bounded():
    tests = [{"name": f"WaitProducerTrace.Case{i}"} for i in range(8)]
    assert len(trace.selected_tests({"tests": tests}, {test["name"] for test in tests})) == 4


def test_environment_unsets_preserve_disabled_and_batch_contracts(monkeypatch):
    monkeypatch.setenv("PROSPER_WAIT_PRODUCER_TRACE", "inherited")
    monkeypatch.setenv("PROSPER_NO_BATCH_RENDERER_WRITE_DRAIN", "inherited")
    test = {
        "properties": [
            {"name": "WORKING_DIRECTORY", "value": "directory with spaces"},
            {"name": "ENVIRONMENT", "value": ["PROSPER_POST_SUBMIT_VISIBILITY=1", "KEEP=a=b"]},
            {
                "name": "ENVIRONMENT_MODIFICATION",
                "value": [
                    "PROSPER_WAIT_PRODUCER_TRACE=unset:",
                    "PROSPER_NO_BATCH_RENDERER_WRITE_DRAIN=unset:",
                ],
            },
        ]
    }
    cwd, env = trace.test_context(test)
    assert cwd == "directory with spaces"
    assert env["PROSPER_POST_SUBMIT_VISIBILITY"] == "1"
    assert env["KEEP"] == "a=b"
    assert "PROSPER_WAIT_PRODUCER_TRACE" not in env
    assert "PROSPER_NO_BATCH_RENDERER_WRITE_DRAIN" not in env
    assert os.environ["PROSPER_WAIT_PRODUCER_TRACE"] == "inherited"


def test_unknown_environment_modification_is_unverified():
    with pytest.raises(ValueError, match="unsupported environment modification"):
        trace.test_context(
            {
                "properties": [
                    {"name": "ENVIRONMENT_MODIFICATION", "value": ["X=path_list_prepend:other"]}
                ]
            }
        )


def test_debugger_output_and_failure_exit_are_retained():
    code = "import sys; print('not a captured stack'); sys.exit(7)"
    status, output = trace.run_debugger([sys.executable, "-c", code], None, dict(os.environ))
    assert status == 7
    assert "not a captured stack" in output
    assert not trace.captured_stack(status, output)


def test_debugger_timeout_is_unverified_and_waits_for_owned_process(capsys):
    # The descendant keeps the captured pipe open. Killing only its parent cannot
    # reach EOF within the drain timeout, so that omission must fail this assertion.
    child = "import time; print('TRACE_DESCENDANT_READY', flush=True); time.sleep(20)"
    parent = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}], "
        "stdout=sys.stdout, stderr=sys.stderr).wait()"
    )
    with pytest.raises(RuntimeError, match="timed out; owned process tree stopped"):
        trace.run_debugger(
            [sys.executable, "-c", parent],
            None,
            dict(os.environ),
            timeout=5,
        )
    assert "TRACE_DESCENDANT_READY" in capsys.readouterr().out


@pytest.mark.parametrize(
    "status,output,expected",
    [
        (42, "PROSPER_SYSTEM_ERROR_STACK\n#0 std::__throw_system_error (int)\n", 0),
        (0, "PROSPER_SYSTEM_ERROR_STACK\n#0 std::__throw_system_error (int)\n", 1),
        (42, "PROSPER_NO_SYSTEM_ERROR_STACK\n", 1),
    ],
)
def test_cli_keeps_original_failure_and_uses_discovered_command(
    tmp_path, monkeypatch, status, output, expected
):
    inventory = tmp_path / "Testing/Temporary/LastTestsFailed.log"
    inventory.parent.mkdir(parents=True)
    original = "17:WaitProducerTrace.One\n18:indirect_pointer_descriptor_range\n"
    inventory.write_text(original)
    command = ["owned executable with spaces", "--gtest_filter=WaitProducerTrace.One"]
    manifest = {
        "tests": [
            {
                "name": "WaitProducerTrace.One",
                "command": command,
                "properties": [{"name": "WORKING_DIRECTORY", "value": str(tmp_path)}],
            }
        ]
    }
    monkeypatch.setattr(
        trace.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(manifest))
    )
    calls = []

    def debugger(args, cwd, env):
        calls.append((args, cwd))
        return status, output

    monkeypatch.setattr(trace, "run_debugger", debugger)
    assert trace.main(["--build", str(tmp_path), "--gdb", "selected gdb"]) == expected
    assert len(calls) == (1 if expected == 0 else 2)
    assert calls[0][0][0] == "selected gdb"
    assert calls[0][0][-2:] == command
    assert calls[0][1] == str(tmp_path)
    assert inventory.read_text() == original
