#!/usr/bin/env python3
"""Prove vote_reaches_output can SEE a vote reaching a colour output, before any zero is believed.

`vote_reaches_output.py` reported 0 of 19 modules across two titles. That is the answer hoped for,
which is exactly when a zero must not be taken on trust: an analyser that structurally cannot express
the case reports the same zero as one that looked and found nothing.

So the modules below are assembled BY HAND, not by recompiling a shader and not by the tool's own
helpers. Each is a minimal SPIR-V module built from raw words, so the positive cases exist
independently of anything the analyser or the emitter does.

Four shapes, chosen because each is a distinct way the closure could fail:
  1. vote -> OpStore to Output                    (the direct case)
  2. vote -> OpSelect -> store                    (one SSA hop)
  3. vote -> store to a Function local -> OpLoad -> store to Output   (through memory)
  4. vote used only in a branch condition         (the NEGATIVE control: must NOT be flagged)

Shape 4 is the one that keeps the test honest. Without it, an analyser that flagged everything would
pass shapes 1-3 and look perfect.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
TOOL = HERE.parent / "vote_reaches_output.py"

MAGIC, VERSION = 0x07230203, 0x00010300
failures = []


def check(name, ok, detail=""):
    if ok:
        print("ok   - %s" % name)
    else:
        failures.append(name)
        print("FAIL - %s %s" % (name, detail))


def inst(op, *operands):
    return [((1 + len(operands)) << 16) | op] + list(operands)


def module(body_ids, body):
    """A 5-word header plus `body`, with a bound above every id used."""
    return [MAGIC, VERSION, 0, max(body_ids) + 1, 0] + body


def write(words, path):
    Path(path).write_bytes(struct.pack("<%dI" % len(words), *words))


# ids: 1 t_bool 2 t_float 3 t_v4 4 ptr_out 5 out_var 6 scope 7 vote 8 val_a 9 val_b
#      10 selected 11 ptr_fn 12 local 13 loaded
def mod_direct():
    b = (inst(20, 1) + inst(22, 2, 32) + inst(23, 3, 2, 4)
         + inst(32, 4, 3, 3) + inst(59, 4, 5, 3)                   # OpVariable Output
         + inst(43, 2, 6, 3)                                       # a constant (scope)
         + inst(335, 1, 7, 6, 6)                                   # OpGroupNonUniformAny -> 7
         + inst(62, 5, 7))                                         # OpStore out, vote
    return module([1, 2, 3, 4, 5, 6, 7], b)


def mod_one_hop():
    b = (inst(20, 1) + inst(22, 2, 32) + inst(23, 3, 2, 4)
         + inst(32, 4, 3, 3) + inst(59, 4, 5, 3) + inst(43, 2, 6, 3)
         + inst(335, 1, 7, 6, 6)
         + inst(43, 2, 8, 0) + inst(43, 2, 9, 1)
         + inst(169, 2, 10, 7, 8, 9)                               # OpSelect using the vote
         + inst(62, 5, 10))
    return module([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], b)


def mod_through_memory():
    b = (inst(20, 1) + inst(22, 2, 32) + inst(23, 3, 2, 4)
         + inst(32, 4, 3, 3) + inst(59, 4, 5, 3) + inst(43, 2, 6, 3)
         + inst(32, 11, 1, 7)                                      # ptr Function bool
         + inst(59, 11, 12, 7)                                     # OpVariable Function
         + inst(335, 1, 7, 6, 6)
         + inst(62, 12, 7)                                         # store vote -> local
         + inst(61, 1, 13, 12)                                     # load  local -> 13
         + inst(62, 5, 13))                                        # store 13 -> Output
    return module([1, 2, 3, 4, 5, 6, 7, 11, 12, 13], b)


def mod_branch_only():
    b = (inst(20, 1) + inst(22, 2, 32) + inst(23, 3, 2, 4)
         + inst(32, 4, 3, 3) + inst(59, 4, 5, 3) + inst(43, 2, 6, 3)
         + inst(335, 1, 7, 6, 6)
         + inst(43, 2, 8, 0)
         + inst(250, 7, 6, 6)                                      # OpBranchConditional on the vote
         + inst(62, 5, 8))                                         # store a CONSTANT -> Output
    return module([1, 2, 3, 4, 5, 6, 7, 8], b)


def mod_through_extinst():
    """vote -> OpExtInst argument -> store. Exercises the opcode whose LITERAL instruction number
    caused 8 false positives; the fix must still follow its genuine id arguments."""
    b = (inst(20, 1) + inst(22, 2, 32) + inst(23, 3, 2, 4)
         + inst(32, 4, 3, 3) + inst(59, 4, 5, 3) + inst(43, 2, 6, 3)
         + inst(335, 1, 7, 6, 6)
         + inst(11, 20)                                            # OpExtInstImport -> set id 20
         + inst(12, 2, 21, 20, 4, 7)                               # OpExtInst set=20 instr=4 arg=vote
         + inst(62, 5, 21))
    return module([1, 2, 3, 4, 5, 6, 7, 20, 21], b)


def mod_through_composite():
    """vote -> OpCompositeConstruct -> OpCompositeExtract -> store. Exercises the literal-index
    opcode; word 3 is a genuine id and must still be followed."""
    b = (inst(20, 1) + inst(22, 2, 32) + inst(23, 3, 2, 4)
         + inst(32, 4, 3, 3) + inst(59, 4, 5, 3) + inst(43, 2, 6, 3)
         + inst(335, 1, 7, 6, 6)
         + inst(80, 3, 30, 7, 7, 7, 7)                             # OpCompositeConstruct from vote
         + inst(81, 2, 31, 30, 0)                                  # OpCompositeExtract idx 0
         + inst(62, 5, 31))
    return module([1, 2, 3, 4, 5, 6, 7, 30, 31], b)


def run(words):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.spv"
        write(words, p)
        done = subprocess.run([sys.executable, str(TOOL), str(p)],
                              capture_output=True, text=True)
        return done.returncode, done.stdout


def main() -> int:
    for name, mod in (("a vote stored straight to a colour output", mod_direct()),
                      ("a vote reaching an output through OpSelect", mod_one_hop()),
                      ("a vote reaching an output through a local variable", mod_through_memory()),
                      ("a vote reaching an output through an OpExtInst argument",
                       mod_through_extinst()),
                      ("a vote reaching an output through composite construct/extract",
                       mod_through_composite())):
        rc, out = run(mod)
        check("DETECTS %s" % name, rc == 1 and "REACHES OUTPUT" in out, out)

    rc, out = run(mod_branch_only())
    check("does NOT flag a vote used only as a branch condition",
          rc == 0 and "no output path" in out, out)

    # A module the tool cannot parse must never be counted as clean.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "junk.spv"
        p.write_bytes(b"not spirv at all, really")
        done = subprocess.run([sys.executable, str(TOOL), str(p)], capture_output=True, text=True)
        check("an unreadable module is NOT reported clean",
              done.returncode == 1 and "UNREADABLE" in done.stdout, done.stdout)

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
