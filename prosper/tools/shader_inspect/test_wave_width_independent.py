#!/usr/bin/env python3
"""Prove `wave_width_independent` can SEE the cases that must be refused, before any pass is trusted.

The predicate's POSITIVE answer is load-bearing: it is what lets a draw run at the host's native
32-lane fragment subgroup instead of being skipped. So the tests that matter are the ones that must
come back NOT PROVEN, and the sharpest of them is the counterexample that retired this tool's
predecessor:

    let the guest predicate be true in the upper 32 lanes and false in the lower. Initialise a
    colour output, branch on any(P), and store a constant in the taken block. A 64-lane wave takes
    the branch and every lane sees the constant; two independent 32-lane groups disagree, and the
    constant lands in one half only.

The vote's VALUE never reaches the store there -- only the branch does -- so a value-reachability
predicate calls that module safe. `mod_branch_only(SC_INPUT)` is exactly that module, and it is the
first check below.

Every module here is assembled BY HAND from raw words, not by recompiling a shader and not through
the tool's own helpers, so the cases exist independently of anything the analyser or the emitter
does. A control drawn from the same source as the null would test the discriminator, not the domain.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
TOOL = HERE.parent / "wave_width_independent.py"

MAGIC, VERSION = 0x07230203, 0x00010300
SC_UNIFORM_CONSTANT, SC_INPUT, SC_OUTPUT, SC_FUNCTION, SC_STORAGE_BUFFER = 0, 1, 3, 7, 12
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
    return [MAGIC, VERSION, 0, max(body_ids) + 1, 0] + body


def write(words, path):
    Path(path).write_bytes(struct.pack("<%dI" % len(words), *words))


# Shared preamble. ids: 1 bool 2 float 3 vec4 4 ptr_out 5 out_var 6 scope 11 const
# `source_storage` selects where the vote's predicate comes from, which is the whole experiment:
# Input is per-pixel and divergent, StorageBuffer is a constant-buffer scalar and uniform.
def preamble(source_storage):
    return (inst(20, 1)                       # OpTypeBool
            + inst(22, 2, 32)                 # OpTypeFloat 32
            + inst(23, 3, 2, 4)               # OpTypeVector
            + inst(32, 4, 3, SC_OUTPUT)       # OpTypePointer Output vec4
            + inst(59, 4, 5, SC_OUTPUT)       # OpVariable Output
            + inst(43, 2, 6, 3)               # OpConstant (used as the scope literal)
            + inst(43, 2, 11, 1)              # OpConstant (the value a taken block stores)
            + inst(32, 7, 1, source_storage)  # OpTypePointer <storage> bool
            + inst(59, 7, 8, source_storage)  # OpVariable <storage>
            + inst(61, 1, 9, 8))              # OpLoad -> %9, the vote's predicate


PRE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]


def mod_branch_only(storage):
    """any(P) guards a block that stores a constant into the colour output."""
    b = (preamble(storage)
         + inst(335, 1, 10, 6, 9)                       # OpGroupNonUniformAny -> %10
         + inst(248, 20)                                # OpLabel (entry)
         + inst(247, 22, 0)                             # OpSelectionMerge %22
         + inst(250, 10, 21, 22)                        # OpBranchConditional on the vote
         + inst(248, 21) + inst(62, 5, 11) + inst(249, 22)   # then: OpStore out, const
         + inst(248, 22) + inst(253))                   # merge, OpReturn
    return module(PRE_IDS + [10, 20, 21, 22], b)


def mod_phi_of_constants(storage):
    """The reviewer's second shape: the two arms merge in a phi that then reaches the output."""
    b = (preamble(storage)
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20)
         + inst(247, 23, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(249, 23)
         + inst(248, 22) + inst(249, 23)
         + inst(248, 23)
         + inst(245, 2, 24, 11, 21, 6, 22)             # OpPhi: different value per edge
         + inst(62, 5, 24)                             # ...stored to the colour output
         + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22, 23, 24], b)


def mod_value_to_output(storage):
    """The one case the predecessor DID catch: the vote's value itself is stored."""
    b = (preamble(storage)
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20) + inst(62, 5, 10) + inst(253))
    return module(PRE_IDS + [10, 20], b)


def mod_guards_storage_buffer_write():
    """The effect the region performs is a STORAGE BUFFER write, not a colour store.

    A predicate that only looked for `Output` stores would call this safe. It is not: the write
    happens in one 32-lane group and not the other, and the rest of the draw can read it.
    """
    b = (preamble(SC_INPUT)
         + inst(32, 40, 2, SC_STORAGE_BUFFER)          # OpTypePointer StorageBuffer float
         + inst(59, 40, 41, SC_STORAGE_BUFFER)         # OpVariable StorageBuffer
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20)
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(62, 41, 11) + inst(249, 22)   # then: store into the buffer
         + inst(248, 22) + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22, 40, 41], b)


def mod_uniform_source_but_shader_writes_memory():
    """`mod_branch_only(SC_STORAGE_BUFFER)` with ONE extra instruction: an unrelated store into a
    second storage buffer, outside the vote's region.

    That instruction is the whole experiment. The admitted version and this one differ by nothing
    else, so if the verdict does not flip, the "uniform" test is not asking whether the value is
    invocation-invariant -- it is only asking which storage class the pointer had. A buffer this
    draw writes is not a constant.
    """
    b = (preamble(SC_STORAGE_BUFFER)
         + inst(32, 40, 2, SC_STORAGE_BUFFER)
         + inst(59, 40, 41, SC_STORAGE_BUFFER)
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20)
         + inst(62, 41, 11)                            # the extra instruction: a UAV write
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(62, 5, 11) + inst(249, 22)
         + inst(248, 22) + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22, 40, 41], b)


def mod_guards_discard():
    """The region discards the pixel. Whether the fragment survives is the vote's answer."""
    b = (preamble(SC_INPUT)
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20)
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(252)                   # then: OpKill
         + inst(248, 22) + inst(62, 5, 11) + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22], b)


def mod_dead_vote():
    """A divergent vote whose region has no observable effect and whose merge carries nothing."""
    b = (preamble(SC_INPUT)
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20)
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(129, 2, 25, 11, 11) + inst(249, 22)   # then: an unused OpFAdd
         + inst(248, 22) + inst(62, 5, 11) + inst(253))               # merge stores a CONSTANT
    return module(PRE_IDS + [10, 20, 21, 22, 25], b)


def mod_unstructured():
    """A vote-conditioned branch with no merge instruction: the region has no bound to trust."""
    b = (preamble(SC_INPUT)
         + inst(335, 1, 10, 6, 9)
         + inst(248, 20)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(249, 22)
         + inst(248, 22) + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22], b)


def mod_uniform_ballot():
    """A BALLOT over a uniform value. Uniformity must NOT clear it: the result is a per-lane mask
    whose width is the subgroup's, so 32 lanes report half a mask as though it were whole."""
    b = (preamble(SC_STORAGE_BUFFER)
         + inst(21, 30, 32, 0)                         # OpTypeInt 32 unsigned
         + inst(23, 31, 30, 4)                         # OpTypeVector uint 4
         + inst(339, 31, 10, 6, 9)                     # OpGroupNonUniformBallot -> %10
         + inst(248, 20)
         + inst(81, 30, 32, 10, 0)                     # OpCompositeExtract .x
         + inst(62, 5, 32)
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 32], b)


def mod_no_votes():
    b = preamble(SC_INPUT) + inst(248, 20) + inst(62, 5, 11) + inst(253)
    return module(PRE_IDS + [20], b)


def run(words):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.spv"
        write(words, p)
        done = subprocess.run([sys.executable, str(TOOL), str(p)], capture_output=True, text=True)
        return done.returncode, done.stdout


def expect_refused(name, words):
    rc, out = run(words)
    check("REFUSES %s" % name, rc == 1 and "NOT PROVEN" in out, out)


def expect_admitted(name, words):
    rc, out = run(words)
    check("admits %s" % name, rc == 0 and "width-independent" in out, out)


def main() -> int:
    # --- the cases that must be refused ---------------------------------------------------------
    expect_refused("a divergent vote guarding a store to the colour output",
                   mod_branch_only(SC_INPUT))
    expect_refused("a divergent vote whose arms merge in a phi that reaches the output",
                   mod_phi_of_constants(SC_INPUT))
    expect_refused("a divergent vote whose value is stored to the colour output",
                   mod_value_to_output(SC_INPUT))
    expect_refused("a vote-conditioned branch with no merge instruction", mod_unstructured())
    expect_refused("a BALLOT over a uniform value reaching the output", mod_uniform_ballot())
    expect_refused("a divergent vote guarding a STORAGE BUFFER write",
                   mod_guards_storage_buffer_write())
    expect_refused("a divergent vote guarding a discard", mod_guards_discard())
    expect_refused("a vote over a storage buffer THIS SHADER WRITES",
                   mod_uniform_source_but_shader_writes_memory())

    # --- the cases that must be admitted, each for a DIFFERENT reason ----------------------------
    # Arm (a). Same three shapes, same analysis, one word changed: the predicate now comes from a
    # constant buffer rather than a per-pixel input. If any of these came back refused the uniform
    # arm would be doing nothing, and if the three above came back admitted it would be doing too
    # much -- the pair is what makes either result meaningful.
    expect_admitted("a UNIFORM vote guarding a store to the colour output",
                    mod_branch_only(SC_STORAGE_BUFFER))
    expect_admitted("a UNIFORM vote whose arms merge in a phi that reaches the output",
                    mod_phi_of_constants(SC_STORAGE_BUFFER))
    expect_admitted("a UNIFORM vote whose value is stored to the colour output",
                    mod_value_to_output(SC_STORAGE_BUFFER))
    # Arm (b).
    expect_admitted("a divergent vote that cannot influence any output", mod_dead_vote())
    expect_admitted("a module with no wave vote at all", mod_no_votes())

    # A module the tool cannot parse must never be counted as proven.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "junk.spv"
        p.write_bytes(b"not spirv at all, really")
        done = subprocess.run([sys.executable, str(TOOL), str(p)], capture_output=True, text=True)
        check("an unreadable module is NOT reported proven",
              done.returncode == 1 and "UNREADABLE" in done.stdout, done.stdout)

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
