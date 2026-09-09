"""Check actual lifecycle records and exact PCM fixture with instrumentation on and off."""
import os
import re
import subprocess
import sys


def fields(line):
    return dict(re.findall(r"([a-zA-Z_]+)=([^\s]+)", line))


def number(row, key):
    return int(row[key], 0)


for enabled in (False, True):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PROSPER_AUDIO")}
    env["PROSPER_AUDIO_LIFECYCLE"] = "1" if enabled else "0"
    if "--negative-disabled" in sys.argv:
        env["PROSPER_AUDIO_LIFECYCLE"] = "0"
    result = subprocess.run([sys.argv[1], "--lifecycle-publication"], env=env,
                            capture_output=True, text=True, timeout=30, check=True)
    expected = fields(next(line for line in result.stdout.splitlines() if line.startswith("[lifecycle-fixture]")))
    ports = [fields(line) for line in result.stderr.splitlines() if line.startswith("[audio-lifecycle-port]")]
    contexts = [fields(line) for line in result.stderr.splitlines() if line.startswith("[audio-lifecycle-context]")]
    if not enabled:
        assert not ports and not contexts, result.stderr
        continue
    assert len(ports) == 4 and len(contexts) == 2, result.stderr
    by_port = {number(row, "port"): row for row in ports}
    assert len(by_port) == 4
    for name, publications, valid, kind, reason in (
            ("old", 2, 1, 0, "port-destroy"), ("new", 1, 1, 0, "context-destroy"),
            ("aux", 1, 1, 1, "context-destroy"), ("next_port", 0, 0, 0, "context-destroy")):
        handle = number(expected, name)
        row = by_port[handle]
        context = number(expected, "next_context" if name == "next_port" else "context")
        assert number(row, "context") == context and number(row, "sink") == 17
        assert number(row, "generation") == (handle >> 16) & 0xffffffff
        assert number(row, "context_generation") == (context >> 16) & 0xffffffff
        assert number(row, "type") == kind and number(row, "data_format") == 0x200
        assert row["event"] == reason and number(row, "publications") == publications
        assert number(row, "valid_publications") == valid
        if valid:
            assert 0 < number(row, "last_valid_pcm_ns") <= number(row, "last_attribute_ns") <= number(row, "steady_ns")
        else:
            assert number(row, "last_valid_pcm_ns") == number(row, "last_attribute_ns") == 0
    assert number(by_port[number(expected, "old")], "frames") == 0
    assert number(by_port[number(expected, "old")], "pending") == 1
    assert number(by_port[number(expected, "new")], "frames") == 64
    assert number(by_port[number(expected, "new")], "pending") == 0
    assert number(by_port[number(expected, "aux")], "pending") == 1
    for row in contexts:
        assert number(row, "entered_ns") <= number(row, "cleared_ns") <= number(row, "close_return_ns")
    assert [number(row, "implicit_ports") for row in contexts] == [2, 1]
    assert [number(row, "sink_was_open") for row in contexts] == [1, 0]
print("Lifecycle observer: real HLE publication/retirement records and unchanged PCM verified on/off")
