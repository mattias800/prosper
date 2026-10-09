# tools/console_capture

A client for `ps5debug-NG`, the homebrew debugger on a developer's own unlocked PS5. It reads
runtime state (foreground app, process list and maps, bounded memory reads) and can arm ONE software
breakpoint to capture a GPU submit. That breakpoint is the only thing that changes console state (it
writes a trap into the target process), so it is opt-in: `PS5DebugClient(..., allow_breakpoint=True)`,
slot 0 only.

Console data is optional human evidence. No test, CI job or agent workflow may depend on a console:
the tests (`prosper/tests/tools/test_console_capture.py`) drive the client against a loopback fake
server. The client writes nothing to disk; captured bytes belong outside the repository.
