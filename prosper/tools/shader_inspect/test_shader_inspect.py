#!/usr/bin/env python3
"""Regression test for shader_inspect's resource-table honesty (#1571).

shader_inspect reads a RAW shader dump, which carries instructions but no descriptors, so it can
never supply a ShaderResourceTable. The stage recompilers then legitimately refuse to lower any
instruction that resolves a V#/T#/S# through that table. Before #1571 the tool reported that refusal
as `status=rejected`, i.e. as if the SHADER were unsupported — a tool limitation misattributed as a
shader defect. It was wrong for 109 of 114 shaders that had provably recompiled and rendered live.

These cases pin the corrected contract. Case 1 is the one that fails without the fix.

Usage: test_shader_inspect.py <path-to-shader_inspect>
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

S_ENDPGM = 0xBF810000
S_NOP = 0xBF800000


def s_load_dwordx4(sdata: int, sbase_pair: int, offset: int) -> tuple:
    """`s_load_dwordx4 s[sdata:sdata+3], s[base:base+1], offset` with SOFFSET = SGPR_NULL (125).

    This is the canonical constant-buffer load emitted by real title shaders: SMEM opcode 0x02,
    immediate offset, and the SOFFSET field set to NULL rather than a register.
    """
    word0 = (0x3D << 26) | (0x02 << 18) | ((sdata & 0x7F) << 6) | (sbase_pair & 0x3F)
    word1 = (125 << 25) | (offset & 0x1FFFFF)
    return word0, word1


def run(binary: str, words, stage=None, extra=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shader.bin"
        path.write_bytes(struct.pack("<%dI" % len(words), *words))
        cmd = [binary, str(path)]
        if extra:
            cmd += list(extra)
        if stage:
            cmd += ["--stage", stage]
        done = subprocess.run(cmd, capture_output=True, text=True)
        return done.returncode, done.stdout


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_shader_inspect.py <path-to-shader_inspect>", file=sys.stderr)
        return 2
    binary = sys.argv[1]
    failures = []

    def check(name, condition, detail=""):
        if condition:
            print("ok   - %s" % name)
        else:
            print("FAIL - %s %s" % (name, detail))
            failures.append(name)

    # A minimal graphics shader whose only real work is a constant-buffer load.
    cbuf_load = list(s_load_dwordx4(sdata=8, sbase_pair=0, offset=0)) + [S_ENDPGM]

    # --- Case 1: the required contract -------------------------------------------------------
    # A graphics stage with a constant-buffer load must NOT be reported as an unsupported-instruction
    # rejection when no resource table is supplied. Without the fix this printed `status=rejected`.
    for stage in ("fragment", "vertex"):
        code, out = run(binary, cbuf_load, stage)
        check("%s cbuf load is not reported as a shader rejection" % stage,
              "status=rejected" not in out, "\n" + out)
        check("%s cbuf load reports the tool limitation explicitly" % stage,
              "status=undetermined-no-resource-table" in out, "\n" + out)
        check("%s cbuf load names the limitation in words" % stage,
              "TOOL LIMITATION" in out and "NOT EVIDENCE OF A SHADER DEFECT" in out, "\n" + out)
        # The note must not overcorrect into reassurance: `stage_undetermined` is set whenever a
        # table-dependent instruction is present, so a genuine defect can be present too and would
        # look identical. The wording has to say so, or it invites waving real bugs away.
        check("%s cbuf load does not claim the shader is fine" % stage,
              "equally not evidence that it is fine" in out, "\n" + out)
        check("%s cbuf load calls the rejection unattributable" % stage,
              "UNATTRIBUTABLE" in out, "\n" + out)
        check("%s cbuf load counts the table-dependent instruction" % stage,
              "table_dependent=1" in out, "\n" + out)
        check("%s cbuf load exits 3 (undetermined), never 0" % stage,
              code == 3, "got exit %d" % code)

    # --- Case 2: the classification is stage-sensitive, not blanket ---------------------------
    # Compute passes allow_smem=true unconditionally, so the SAME stream needs no table there and
    # must still recompile normally. This is what proves case 1 is about the missing table and not
    # about the instruction being unsupported.
    code, out = run(binary, cbuf_load, "compute")
    check("identical stream recompiles ok in compute",
          "status=ok" in out, "\n" + out)
    check("compute reports no table-dependent instructions",
          "table_dependent=0" in out, "\n" + out)
    check("compute cbuf load exits 0", code == 0, "got exit %d" % code)

    # --- Case 3: genuine defects are still genuine --------------------------------------------
    # A stream that never reaches s_endpgm is a real defect and must not be softened into the new
    # undetermined status, even though the tool still has no resource table.
    code, out = run(binary, [S_NOP], "compute")
    check("stream without s_endpgm is still a genuine failure", code == 1,
          "got exit %d" % code)
    check("stream without s_endpgm is not called undetermined",
          "undetermined" not in out, "\n" + out)

    # A shader with no table-dependent instruction at all keeps the plain verdict vocabulary: it is
    # attributable, so it must never be labelled undetermined.
    code, out = run(binary, [S_ENDPGM], "fragment")
    check("table-free shader is never labelled undetermined",
          "undetermined" not in out, "\n" + out)

    # --- Case 4: usage errors unchanged --------------------------------------------------------
    done = subprocess.run([binary], capture_output=True, text=True)
    check("no arguments still exits 2", done.returncode == 2,
          "got exit %d" % done.returncode)
    check("usage text warns about the missing resource table",
          "resource table" in done.stderr, "\n" + done.stderr)

    # ---- #3464: offline wave-reason census -----------------------------------------------
    #
    # A fragment shader that votes (compare, then branch on the result) requires the guest wave
    # width for a reason that is RECOVERABLE at 32 lanes -- two half-waves union to the same
    # answer -- and prosper already admits exactly this class. The census must say so in a form
    # a script can filter, so a dump of a title's shaders becomes a table rather than 86 manual
    # inspections.
    vote_words = (
        0x7E040280,              # v_mov_b32 v2,0
        0x7E060280,              # v_mov_b32 v3,0
        0x7C020300,              # v_cmp_lt_f32 vcc,v0,v1
        0xBF13806A,              # s_cmp_lg_u64 vcc,0
        0xBF840001,              # s_cbranch_scc0 -> +1
        0x7E040281,              # v_mov_b32 v2,1
        0x7E080280,              # v_mov_b32 v4,0
        0x7E0A0280,              # v_mov_b32 v5,0
        0xF800180F, 0x05040302,  # exp mrt0 v2,v3,v4,v5 done vm
        S_ENDPGM,
    )
    code, out = run(binary, vote_words, extra=["--wave-reasons"])
    check("wave-reasons census emits its sentinel",
          "wave-reasons-end" in out, "\n" + out)
    check("wave-reasons census reports the required width",
          "required-subgroup-size=64" in out, "\n" + out)
    check("wave-reasons census names the reason bits",
          "reasons=0x2" in out and "wave-any" in out, "\n" + out)
    # The actionable column, and it is deliberately a fact about SHIPPING code rather than a
    # judgement: render_runner.h admits a fragment shader at the host's native wave32 only when
    # its reason set equals kFragmentWaveReasonWaveAny exactly. A reason set of precisely WaveAny
    # is therefore admitted today, and the census must say so in the same terms, so a count taken
    # over a title's shaders describes the emulator someone actually runs.
    check("wave-reasons census reports native-wave32 admission",
          "native-wave32-admitted=1" in out, "\n" + out)

    # A shader with no wave op at all must report a real, empty census -- not silence, which a
    # consumer cannot distinguish from a crash.
    plain_words = (
        0x7E040280, 0x7E060280, 0x7E080280, 0x7E0A0280,
        0xF800180F, 0x05040302,
        S_ENDPGM,
    )
    code, out = run(binary, plain_words, extra=["--wave-reasons"])
    check("a wave-free shader still emits a census",
          "wave-reasons-end" in out and "required-subgroup-size=0" in out, "\n" + out)
    # A shader that requires no width at all never reaches the gate, so it is not "admitted" by
    # it. Conflating "needs nothing" with "needs something prosper can supply" would inflate every
    # admission count by the entire population of ordinary shaders.
    check("a wave-free shader is not counted as gate-admitted",
          "native-wave32-admitted=0" in out, "\n" + out)

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
