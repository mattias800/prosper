#!/usr/bin/env python3
"""test_eboot_shader_elfs — the eboot shader-ELF extractor must find only AMDGPU ELFs.

Covers tools/re/eboot_shader_elfs.py: a synthetic eboot (never real game bytes) with
a hand-built minimal ELF64 (e_machine 0xE0, 6 sections incl. .shader_header and
.shader_text) is found, extracted, and manifested; a non-AMDGPU ELF in the same
position is ignored (this is the arm that fails if the e_machine check is broken);
truncated and malformed hits are rejected rather than raising; and the manifest
carries hashes only, never host paths.
"""

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "re"))

import eboot_shader_elfs as ese

S_NOP = bytes((0x00, 0x00, 0x80, 0xBF))
S_ENDPGM = bytes((0x00, 0x00, 0x81, 0xBF))
PAD = bytes((0x55, 0xAA))


def build_shader_elf(
    machine=0xE0, ei_class=2, header_bytes=None, text_bytes=None, section_names=None
):
    """Hand-build a minimal ELF64 with e_shnum == 6 like the embedded images."""
    header_bytes = b"H" * 120 if header_bytes is None else header_bytes
    text_bytes = S_NOP * 3 + S_ENDPGM if text_bytes is None else text_bytes
    section_names = (
        (".shader_header", ".shader_text", ".note_a", ".note_b")
        if section_names is None
        else section_names
    )
    names = [b""] + [n.encode() for n in section_names] + [b".shstrtab"]
    blobs = [b"", header_bytes, text_bytes, b"aa", b"bb", b""]
    strtab = b"\x00" + b"\x00".join(names[1:]) + b"\x00"
    blobs[5] = strtab
    name_off = {}
    at = 0
    for name in names:
        name_off[name] = at
        at += len(name) + 1
    shoff = 64
    data_at = shoff + 6 * 64
    shdrs = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    offsets = []
    for i in range(1, 6):
        offsets.append(data_at)
        data_at += len(blobs[i])
    for i in range(1, 6):
        shdrs.append(
            struct.pack(
                "<IIQQQQIIQQ",
                name_off[names[i]],
                1,
                0,
                0,
                offsets[i - 1],
                len(blobs[i]),
                0,
                0,
                1,
                0,
            )
        )
    ident = b"\x7fELF" + bytes((ei_class, 1, 1, 0)) + bytes(8)
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH", ident, 2, machine, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 6, 5
    )
    return ehdr + b"".join(shdrs) + b"".join(blobs[1:])


def run_tool(eboot_bytes, tmp_path, *extra_args):
    eboot = tmp_path / "eboot.bin"
    eboot.write_bytes(eboot_bytes)
    out = tmp_path / "store"
    rc = ese.main(
        ["--eboot", str(eboot), "--title-id", "PPSA00000", "--out", str(out), *extra_args]
    )
    assert rc == 0
    manifest = out / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    return out, records


def test_extracts_nested_elf_and_manifest(tmp_path):
    elf = build_shader_elf()
    out, records = run_tool(PAD * 512 + elf + PAD * 256, tmp_path, "--no-llvm")
    assert len(records) == 1
    rec = records[0]
    assert rec == {
        "source": "eboot",
        "title_id": "PPSA00000",
        "file_offset": 1024,
        "elf_size": len(elf),
        "sections": [".shader_header", ".shader_text"],
        "header_size": 120,
        "text_size": 16,
        "sha256": hashlib.sha256(elf).hexdigest(),
        "text_sha256": hashlib.sha256(S_NOP * 3 + S_ENDPGM).hexdigest(),
        "ends_with_s_endpgm": True,
        "llvm_invalid": None,
        "status": "unvalidated",
    }
    assert (out / f"{rec['sha256']}.elf").read_bytes() == elf
    assert (out / f"{rec['text_sha256']}.bin").read_bytes() == S_NOP * 3 + S_ENDPGM


def test_manifest_carries_hashes_not_host_paths(tmp_path):
    elf = build_shader_elf()
    out, records = run_tool(PAD * 64 + elf, tmp_path, "--no-llvm")
    assert len(records) == 1
    text = (out / "manifest.jsonl").read_text(encoding="utf-8")
    assert str(out) not in text
    assert str(tmp_path) not in text
    assert "\\" not in text and re.search(r"[A-Za-z]:", text) is None


def test_ignores_non_amdgpu_elf(tmp_path):
    x86_elf = build_shader_elf(machine=0x3E)
    out, records = run_tool(PAD * 64 + x86_elf + PAD * 64, tmp_path, "--no-llvm")
    assert records == []
    assert list(out.glob("*.elf")) == [] and list(out.glob("*.bin")) == []


def test_ignores_32bit_elf(tmp_path):
    elf32 = build_shader_elf(ei_class=1)
    out, records = run_tool(PAD * 64 + elf32 + PAD * 64, tmp_path, "--no-llvm")
    assert records == []
    assert list(out.glob("*.elf")) == []


def test_truncated_magic_at_eof_is_not_a_hit(tmp_path):
    out, records = run_tool(PAD * 64 + b"\x7fELF", tmp_path, "--no-llvm")
    assert records == []


def test_malformed_candidate_is_rejected_not_raised(tmp_path):
    elf = bytearray(build_shader_elf())
    struct.pack_into("<Q", elf, 0x28, 0xFFFFFFFFFFFF)  # e_shoff past end of file
    out, records = run_tool(PAD * 32 + bytes(elf) + PAD * 32, tmp_path, "--no-llvm")
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "rejected"
    assert rec["file_offset"] == 64
    assert rec["sha256"] is None and rec["elf_size"] == 0
    assert list(out.glob("*.elf")) == []


def test_missing_text_section(tmp_path):
    elf = build_shader_elf(section_names=(".shader_header", ".note_a", ".note_b", ".note_c"))
    out, records = run_tool(PAD * 16 + elf + PAD * 16, tmp_path, "--no-llvm")
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "missing-section"
    assert rec["sections"] == [".shader_header"]
    assert rec["header_size"] == 120
    assert rec["text_size"] == 0 and rec["text_sha256"] is None
    assert rec["ends_with_s_endpgm"] is False
    assert (out / f"{rec['sha256']}.elf").read_bytes() == elf


def test_reports_missing_s_endpgm_without_error(tmp_path):
    elf = build_shader_elf(text_bytes=S_NOP * 4)
    _out, records = run_tool(elf, tmp_path, "--no-llvm")
    assert len(records) == 1
    assert records[0]["ends_with_s_endpgm"] is False
    assert records[0]["status"] == "unvalidated"


def test_identical_elfs_share_one_file(tmp_path):
    elf = build_shader_elf()
    out, records = run_tool(elf + PAD * 64 + elf, tmp_path, "--no-llvm")
    assert len(records) == 2
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["file_offset"] == 0
    assert records[1]["file_offset"] == len(elf) + 128
    assert len(list(out.glob("*.elf"))) == 1


def test_rejects_bad_title_id(tmp_path):
    eboot = tmp_path / "eboot.bin"
    eboot.write_bytes(PAD * 64)
    with pytest.raises(SystemExit):
        ese.main(
            [
                "--eboot",
                str(eboot),
                "--title-id",
                "NOTATITLE",
                "--out",
                str(tmp_path / "store"),
                "--no-llvm",
            ]
        )


def test_llvm_decode_counts_invalid_instructions():
    if not ese.llvm_available():
        pytest.skip("llvm-mc unreachable from this host")
    assert ese.llvm_invalid_count(S_NOP * 3 + S_ENDPGM) == 0
    assert ese.llvm_invalid_count(bytes((0xFF,)) * 16) > 0


def test_llvm_validated_status_end_to_end(tmp_path):
    if not ese.llvm_available():
        pytest.skip("llvm-mc unreachable from this host")
    _out, records = run_tool(PAD * 16 + build_shader_elf(), tmp_path)
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["llvm_invalid"] == 0


def test_default_store_is_under_home_not_a_windows_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(ese.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(tmp_path)
    eboot = tmp_path / "eboot.bin"
    eboot.write_bytes(PAD * 16 + build_shader_elf())
    assert ese.main(["--eboot", str(eboot), "--title-id", "PPSA00000", "--no-llvm"]) == 0
    store = home / "prosper-work" / "eboot_shaders" / "PPSA00000"
    assert (store / "manifest.jsonl").is_file()
    assert len(list(store.glob("*.elf"))) == 1
    # nothing may be created relative to the working directory (e.g. a literal "C:\..." name)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["eboot.bin", "home"]


def test_without_llvm_records_degrade_to_unvalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(ese.shutil, "which", lambda _name: None)
    assert ese.llvm_available() is False
    assert ese.llvm_invalid_count(S_ENDPGM) is None
    _out, records = run_tool(PAD * 16 + build_shader_elf(), tmp_path)
    assert len(records) == 1
    assert records[0]["status"] == "unvalidated"
    assert records[0]["llvm_invalid"] is None


def test_native_llvm_mc_is_preferred_over_wsl(monkeypatch):
    seen = []

    def fake_which(name):
        return {"llvm-mc": "/opt/llvm/bin/llvm-mc", "wsl": "/usr/bin/wsl"}.get(name)

    class Done:
        returncode = 0
        stdout = "invalid instruction encoding\ninvalid instruction encoding\n"
        stderr = ""

    def fake_run(argv, **_kw):
        seen.append(argv)
        return Done()

    monkeypatch.setattr(ese.shutil, "which", fake_which)
    monkeypatch.setattr(ese.subprocess, "run", fake_run)
    assert ese.llvm_available() is True
    assert ese.llvm_invalid_count(S_ENDPGM) == 2
    assert all(argv[0] == "/opt/llvm/bin/llvm-mc" for argv in seen)
