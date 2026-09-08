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
    check("wave-reasons census reports reason-set admissibility",
          "reason-set-admissible=1" in out, "\n" + out)

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
    check("a wave-free shader is not reason-set admissible",
          "reason-set-admissible=0" in out, "\n" + out)

    # ---- #3464: the census must read a SPIR-V module too ------------------------------------
    #
    # A raw dump carries no descriptors, so a texture-sampling shader cannot be lowered and
    # contributes no reason data -- 99.3% of GTA V's pixel-shader database. gpu_replay has the
    # real table and `--dump-shader DRAW:fs` writes the recompiled SPIR-V, so reading a .spv is
    # what makes this census reach the shaders the issue is actually about.
    #
    # Built by hand from the header plus two OpModuleProcessed strings, NOT by recompiling: a
    # fixture from the recompiler would share its assumptions with the reader under test.
    def module_processed(text):
        raw = text.encode("utf-8") + b"\x00"
        raw += b"\x00" * (-len(raw) % 4)
        payload = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
        return [((1 + len(payload)) << 16) | 330] + payload   # Op_ModuleProcessed = 330

    def spirv_module(size, reasons):
        words = [0x07230203, 0x00010300, 0, 1, 0]   # magic, version, generator, bound, schema
        words += module_processed("Prosper.FragmentSubgroupSize=%d" % size)
        words += module_processed("Prosper.FragmentSubgroupWhy=%d" % reasons)
        return words

    # lane-id + wave-ballot: the reason set that blocks every one of GTA V's skipped shaders.
    code, out = run(binary, spirv_module(64, 0x41), extra=["--wave-reasons"])
    check("a SPIR-V module is accepted by the census",
          "wave-reasons-end" in out and "input=spirv" in out, "\n" + out)
    check("a SPIR-V module reports its recorded width",
          "required-subgroup-size=64" in out, "\n" + out)
    check("a SPIR-V module reports its recorded reasons",
          "reasons=0x41" in out and "lane-id" in out and "wave-ballot" in out, "\n" + out)
    # 0x41 is not equal to kFragmentWaveReasonWaveAny, so the shipping gate drops it.
    check("a SPIR-V module blocked by lane-id is not reason-set admissible",
          "reason-set-admissible=0" in out, "\n" + out)

    # WaveAny alone IS admitted, so the column cannot be reading a constant.
    code, out = run(binary, spirv_module(64, 0x2), extra=["--wave-reasons"])
    check("a wave-any-only SPIR-V module is reason-set admissible",
          "reason-set-admissible=1" in out, "\n" + out)

    # No marker at all: absent is not none, and must never be printed as a mask.
    code, out = run(binary, [0x07230203, 0x00010300, 0, 1, 0], extra=["--wave-reasons"])
    check("an unmarked SPIR-V module reports absent, not 0x0",
          "reasons=absent" in out and "reasons=0x0" not in out, "\n" + out)
    check("an unmarked SPIR-V module is not reason-set admissible",
          "reason-set-admissible=0" in out, "\n" + out)

    # ---- #3464 review: the census must not claim ADMISSION it cannot determine -------------
    #
    # render_runner.h:7128-7133 is a five-conjunct condition and only the innermost equality
    # (:7140) is a module property. `bd.allow_native_fragment_vote_width` defaults false
    # (:565) and is set from `title_id == "PPSA04263"` alone (live_renderer.cpp:1216, :7591),
    # and two more conjuncts depend on the HOST's subgroup limits. An offline tool knows none
    # of those, so it must report the reason-set test under its own name and say plainly that
    # admission needs more -- otherwise every non-GTA-V count reads as admitted when the
    # renderer in fact drops the shader.
    code, out = run(binary, spirv_module(64, 0x2), extra=["--wave-reasons"])
    check("the census states that admission is not decided by the module alone",
          "gate-undecided=" in out, "\n" + out)
    check("the census names the title allowlist as an undecided conjunct",
          "title-allowlist" in out, "\n" + out)
    check("the census names the host subgroup limits as an undecided conjunct",
          "host-subgroup" in out, "\n" + out)
    # The two conjuncts a module CAN decide are reported rather than silently assumed to pass.
    check("the census reports the internal-GDS conjunct",
          "internal-gds=" in out, "\n" + out)
    check("the census reports the required subgroup features",
          "subgroup-features=0x" in out, "\n" + out)

    # And the old name must be gone: a consumer keying on it would otherwise keep reading an
    # admission claim that is no longer made.
    check("the overclaiming field name is gone",
          "native-wave32-admitted" not in out, "\n" + out)
    # ---- #3464 review: a corrupt module must not be counted as "needs nothing" --------------
    #
    # `required-subgroup-size=0 reasons=absent` is what a genuine wave-free shader reports AND
    # what a truncated one reports, so a corrupt dump lands in the population that says #3464
    # does not affect it. The marker search already walks word/opcode pairs, so whether that
    # walk lands exactly on the end is free to report -- and it is the difference between a
    # module that says nothing and a module that could not be read.
    code, out = run(binary, spirv_module(64, 0x2), extra=["--wave-reasons"])
    check("a well-formed module reports a clean walk",
          "spirv-walk=ok" in out, "\n" + out)

    # An instruction claiming more words than remain: the walk cannot complete.
    truncated = [0x07230203, 0x00010300, 0, 1, 0, (999 << 16) | 330]
    code, out = run(binary, truncated, extra=["--wave-reasons"])
    check("an overrunning instruction is reported as a bad walk",
          "spirv-walk=truncated" in out, "\n" + out)
    check("a bad walk still emits the sentinel",
          "wave-reasons-end" in out, "\n" + out)

    # A zero word count would loop forever if the walk were unguarded; it must terminate AND
    # be reported as unreadable rather than as a shader with no requirement.
    zero_len = [0x07230203, 0x00010300, 0, 1, 0, 0]
    code, out = run(binary, zero_len, extra=["--wave-reasons"])
    check("a zero-length instruction is reported as a bad walk",
          "spirv-walk=truncated" in out, "\n" + out)

    # And the header alone IS a clean walk -- zero instructions is a complete stream, not a
    # corrupt one. Without this the fix would just relabel every wave-free module as corrupt.
    code, out = run(binary, [0x07230203, 0x00010300, 0, 1, 0], extra=["--wave-reasons"])
    check("a header-only module is a clean walk, not a truncated one",
          "spirv-walk=ok" in out, "\n" + out)
    # ---- #3464 re-review: pin the FAILED-recompile honesty contract ------------------------
    #
    # A shader needing a resource table cannot be lowered here, and the census must then print
    # the sentinel with recompiled=0 and NO census row. The row is what a consumer tallies, so
    # emitting one for a shader that was never read would put it in the population that says
    # #3464 does not affect it -- the same failure as the unrecompiled column, one layer down.
    # This was reported as covered in an earlier round and was not; nothing would have reddened.
    code, out = run(binary, cbuf_load, extra=["--wave-reasons"])
    check("a table-dependent shader still emits the sentinel",
          "wave-reasons-end" in out, "\n" + out)
    check("a failed recompile reports recompiled=0",
          "recompiled=0" in out, "\n" + out)
    check("a failed recompile emits NO census row",
          "required-subgroup-size=" not in out, "\n" + out)
    # NOT `reason-set-admissible=1 not in out`, which was the first draft: that is implied by
    # the row-absence arm above, and in the one world where THAT breaks the printed value is 0
    # anyway -- an arm satisfied for a reason other than the one it names. The gate-undecided
    # line is a SEPARATE printf and can be moved out of the recompiled guard on its own, so
    # asserting on its absence covers a failure the row-absence arm cannot see.
    check("a failed recompile emits no gate-undecided line either",
          "gate-undecided=" not in out, "\n" + out)

    # And the SPIR-V sentinel must say recompiled=n/a -- nothing was recompiled, it was read.
    # Load-bearing: wave_reason_census.py distinguishes the two on this field.
    code, out = run(binary, spirv_module(64, 0x2), extra=["--wave-reasons"])
    check("a SPIR-V input reports recompiled=n/a, never a count",
          "recompiled=n/a" in out and "recompiled=1" not in out, "\n" + out)
    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
