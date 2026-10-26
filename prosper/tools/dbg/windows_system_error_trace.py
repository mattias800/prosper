"""Capture #5104 stacks from failed CPU PM4 registrations without changing CTest's verdict.

Run after ordinary CTest: --build BUILD [--gdb gdb]. Replays at most four failed
registrations twice each, with their discovered commands, directory and environment.
Exit 0 means a stack was captured or no matching failure needed diagnosis; 1 means
no stack was captured, and 2 means the apparatus could not complete. None clears CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


def selected_tests(manifest, failed):
    """Keep actual failed producer fixtures, never a passing or similarly named test."""
    return [
        test
        for test in manifest["tests"]
        if test["name"] in failed
        and re.fullmatch(
            r"WaitProducerTrace(?:Budget|Disabled|Selective)?\.[A-Za-z0-9_]+", test["name"]
        )
    ][:4]


def test_context(test):
    """Reproduce CTest's environment; refuse modifications we cannot reproduce."""
    properties = {item["name"]: item["value"] for item in test.get("properties", [])}
    env = dict(os.environ)
    for item in properties.get("ENVIRONMENT", []):
        key, value = item.split("=", 1)
        env[key] = value
    for item in properties.get("ENVIRONMENT_MODIFICATION", []):
        key, modification = item.split("=", 1)
        operation, value = modification.split(":", 1)
        if operation == "unset":
            env.pop(key, None)
        elif operation == "set":
            env[key] = value
        else:
            raise ValueError(f"unsupported environment modification: {operation}")
    return properties.get("WORKING_DIRECTORY"), env


def run_debugger(command, cwd, env, timeout=45):
    """Bound the debugger and drain its owned process tree before returning on timeout."""
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        **options,
    )
    try:
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    killed = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    raise RuntimeError("could not confirm debugger process-tree cleanup") from error
                if killed.returncode:
                    raise RuntimeError("could not confirm debugger process-tree cleanup") from None
            else:
                os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=10)
            print(output, end="")
            raise RuntimeError("debugger timed out; owned process tree stopped") from None
        return process.returncode, output
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def captured_stack(status, output):
    return (
        status == 42
        and "PROSPER_SYSTEM_ERROR_STACK\n" in output
        and re.search(r"^#0\s+.*__throw_system_error", output, re.MULTILINE) is not None
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--gdb", default="gdb")
    args = parser.parse_args(argv)
    try:
        build = args.build.resolve(strict=True)
        failed_file = build / "Testing/Temporary/LastTestsFailed.log"
        if not failed_file.exists():
            print("No failed CTest inventory; no PM4 replay performed.")
            return 0
        failed = {
            line.split(":", 1)[1] for line in failed_file.read_text().splitlines() if ":" in line
        }
        listing = subprocess.run(
            ["ctest", "--test-dir", str(build), "--show-only=json-v1"],  # ctest-gate: listing
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        tests = selected_tests(json.loads(listing.stdout), failed)
        if not tests:
            print("No failed producer-trace registration; no PM4 replay performed.")
            return 0
        script = Path(__file__).with_name("system_error_stack.gdb").resolve()
        for test in tests:
            cwd, env = test_context(test)
            command = [
                args.gdb,
                "--batch",
                "--nx",
                "-iex",
                "set auto-load off",
                "-x",
                str(script),
                "--args",
                *test["command"],
            ]
            for attempt in range(1, 3):
                print(f"System-error diagnostic: {test['name']} attempt={attempt}", flush=True)
                status, output = run_debugger(command, cwd, env)
                print(output, end="", flush=True)
                print(f"Debugger exit={status}; original CTest result is unchanged.", flush=True)
                if captured_stack(status, output):
                    print("Captured a system-error stack; this does not establish a repair.")
                    return 0
        print("No system-error stack captured. Replays do not clear the original failures.")
        return 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"System-error diagnostic UNVERIFIED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
