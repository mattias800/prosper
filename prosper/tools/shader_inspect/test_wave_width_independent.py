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
            + inst(59, 7, 8, source_storage))  # OpVariable <storage>


# The entry block. The divergent load lives INSIDE it: a review noted that emitting it before the
# first OpLabel put it in no block at all, so the region machinery could not see anything that
# depended on it -- which quietly weakens every fixture whose point is control flow.
def entry():
    return inst(248, 20) + inst(61, 1, 9, 8)   # OpLabel, then OpLoad -> %9


PRE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]


def mod_branch_only(storage):
    """any(P) guards a block that stores a constant into the colour output."""
    b = (preamble(storage)
         + inst(335, 1, 10, 6, 9)                       # OpGroupNonUniformAny -> %10
         + entry()
         + inst(247, 22, 0)                             # OpSelectionMerge %22
         + inst(250, 10, 21, 22)                        # OpBranchConditional on the vote
         + inst(248, 21) + inst(62, 5, 11) + inst(249, 22)   # then: OpStore out, const
         + inst(248, 22) + inst(253))                   # merge, OpReturn
    return module(PRE_IDS + [10, 20, 21, 22], b)


def mod_phi_of_constants(storage):
    """The reviewer's second shape: the two arms merge in a phi that then reaches the output."""
    b = (preamble(storage)
         + inst(335, 1, 10, 6, 9)
         + entry()
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
         + entry() + inst(62, 5, 10) + inst(253))
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
         + entry()
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
         + entry()
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
         + entry()
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(252)                   # then: OpKill
         + inst(248, 22) + inst(62, 5, 11) + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22], b)


def mod_phi_before_vote(identical):
    """A divergent branch reconverges in a phi, and the PHI is what the vote reduces.

    Both incoming values are uniform constants, so a rule that asked only "are the incoming values
    uniform?" called the phi uniform and admitted the module -- while the phi is exactly the
    divergent predicate that selected it. With `identical`, both edges carry the same constant, and
    then which edge ran genuinely cannot matter.

    The pair is the experiment: one operand id differs between them and nothing else does, so a phi
    rule that is inert and one that is too permissive both fail rather than look correct.
    """
    lower_value = 11 if identical else 6
    b = (preamble(SC_INPUT)                            # %9 is a per-pixel load: divergent
         + entry()
         + inst(247, 23, 0)
         + inst(250, 9, 21, 22)                        # branch on the DIVERGENT value itself
         + inst(248, 21) + inst(249, 23)
         + inst(248, 22) + inst(249, 23)
         + inst(248, 23)
         + inst(245, 1, 24, 11, 21, lower_value, 22)   # OpPhi -> %24
         + inst(335, 1, 10, 6, 24)                     # the vote reduces the PHI
         + inst(62, 5, 10)
         + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22, 23, 24], b)


def mod_vote_over_a_local(storage):
    """A vote over a value read back through an ACCESS CHAIN.

    With `storage` Function, the array is a per-invocation local: the module writes a divergent
    value into it and reads it back, so the vote is divergent however uniform the *pointer* is. A
    Function variable with a uniform initializer used to inherit that initializer's uniformity, the
    chain inherited it from the variable, and the load inherited it from the chain -- so the vote
    cleared on a property of where it pointed rather than of what was there.

    With `storage` StorageBuffer, the same chain reads a constant buffer and the vote really is
    uniform. The pair differs by one storage-class word, so a rule that stopped trusting chains
    altogether fails just as loudly as the one that trusted them too far.
    """
    b = (preamble(SC_INPUT)                            # %9 is a per-pixel load: divergent
         + inst(21, 30, 32, 0)                         # OpTypeInt 32 unsigned
         + inst(43, 30, 31, 4)                         # OpConstant uint 4 -- a uniform index
         + inst(28, 32, 1, 31)                         # OpTypeArray bool 4
         + inst(32, 33, storage, 32)                   # OpTypePointer <storage> arr
         + inst(44, 32, 34, 11, 11, 11, 11)            # OpConstantComposite arr (uniform)
         + inst(59, 33, 35, storage, 34)               # OpVariable <storage> WITH initializer
         + inst(32, 36, storage, 1)                    # OpTypePointer <storage> bool
         + entry()
         + inst(65, 36, 37, 35, 31)                    # OpAccessChain var[4]
         + ([] if storage != SC_FUNCTION else inst(62, 37, 9))   # store the DIVERGENT value
         + inst(61, 1, 38, 37)                         # read it back through the chain
         + inst(335, 1, 10, 6, 38)                     # vote over what was read
         + inst(62, 5, 10)
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38], b)


def mod_vote_reaches_a_switch():
    """The vote reaches an OpSwitch SELECTOR, and each case stores a different constant.

    Modelling only a switch's successors and not its selector left this with no control dependence
    at all: nothing in it is an OpBranchConditional, so the whole construct was invisible. The CFG
    emitter really does build a switch on the guest PC, so this is not a theoretical shape.
    """
    b = (preamble(SC_INPUT) + entry()
         + inst(335, 1, 10, 6, 9)                       # vote over the divergent load
         + inst(169, 2, 25, 10, 11, 6)                  # OpSelect -> a selector derived from it
         + inst(247, 28, 0)                             # OpSelectionMerge %28
         + inst(251, 25, 26, 0, 27)                     # OpSwitch %25 default %26, case 0 -> %27
         + inst(248, 26) + inst(62, 5, 11) + inst(249, 28)
         + inst(248, 27) + inst(62, 5, 6) + inst(249, 28)
         + inst(248, 28) + inst(253))
    return module(PRE_IDS + [10, 20, 25, 26, 27, 28], b)


def mod_taint_through_two_chains(same_value):
    """A vote-tainted value is stored into a local through ONE access chain and read back through
    ANOTHER into the same slot.

    Keying the taint by pointer id lost the dependence, because the two chains are different ids --
    and the emitter mints a fresh chain per guest-scratch access, so one slot really is reached
    through several. `same_value=False` stores the vote; True stores a constant, and then there is
    genuinely nothing to carry.
    """
    stored = 10 if not same_value else 11
    b = (preamble(SC_INPUT)
         + inst(21, 30, 32, 0) + inst(43, 30, 31, 4)
         + inst(28, 32, 1, 31)                          # OpTypeArray bool 4
         + inst(32, 33, SC_FUNCTION, 32)
         + inst(59, 33, 34, SC_FUNCTION)                # a Function local array
         + inst(32, 35, SC_FUNCTION, 1)
         + entry()
         + inst(335, 1, 10, 6, 9)                       # the vote
         + inst(65, 35, 36, 34, 31)                     # chain A into slot 4
         + inst(62, 36, stored)                         # ...store through A
         + inst(65, 35, 37, 34, 31)                     # chain B into the SAME slot
         + inst(61, 1, 38, 37)                          # ...read back through B
         + inst(62, 5, 38)                              # ...to the colour output
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38], b)


def mod_slot_through_copyobject(tainted_store):
    """One slot reached through an access chain and an `OpCopyObject` of it.

    `OpCopyObject` derives a pointer without being an access chain, so a root map that followed only
    chains gave the copy its own identity and lost the store/load dependence. `tainted_store` selects
    whether the value written is the vote or a constant, which is the only difference between the
    refusal and its control.
    """
    stored = 10 if tainted_store else 11
    b = (preamble(SC_INPUT)
         + inst(21, 30, 32, 0) + inst(43, 30, 31, 4)
         + inst(28, 32, 1, 31)
         + inst(32, 33, SC_FUNCTION, 32) + inst(59, 33, 34, SC_FUNCTION)
         + inst(32, 35, SC_FUNCTION, 1)
         + entry()
         + inst(335, 1, 10, 6, 9)
         + inst(65, 35, 36, 34, 31)                    # chain into slot 4
         + inst(62, 36, stored)
         + inst(83, 35, 37, 36)                        # OpCopyObject of that pointer
         + inst(61, 1, 38, 37)                         # read back through the COPY
         + inst(62, 5, 38)
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38], b)


def mod_region_performs_atomic(op):
    """A vote-guarded region performs an atomic.

    `op` is the whole experiment: `OpAtomicIAdd` (234) sits inside the numeric window the observable
    check used, `OpAtomicFAddEXT` (6035) does not, and the modules are otherwise identical -- so a
    membership test written as a range admitted one and refused the other.
    """
    b = (preamble(SC_INPUT)
         + inst(21, 30, 32, 0) + inst(43, 30, 31, 4)
         + inst(32, 33, SC_STORAGE_BUFFER, 30) + inst(59, 33, 34, SC_STORAGE_BUFFER)
         + entry()
         + inst(335, 1, 10, 6, 9)
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(op, 30, 39, 34, 6, 6, 31) + inst(249, 22)
         + inst(248, 22) + inst(62, 5, 11) + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22, 30, 31, 33, 34, 39], b)


def mod_region_stores_through_second_chain():
    """The CONTROL-DEPENDENCE twin of the two-chain fixture.

    The value stored inside the vote-guarded region is a CONSTANT, so no data flow carries the vote;
    what carries it is whether the store ran at all. The data-flow twin already existed, this path
    did not -- and they reach the taint through different code.
    """
    b = (preamble(SC_INPUT)
         + inst(21, 30, 32, 0) + inst(43, 30, 31, 4)
         + inst(28, 32, 1, 31)
         + inst(32, 33, SC_FUNCTION, 32) + inst(59, 33, 34, SC_FUNCTION)
         + inst(32, 35, SC_FUNCTION, 1)
         + entry()
         + inst(335, 1, 10, 6, 9)
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(65, 35, 36, 34, 31) + inst(62, 36, 11) + inst(249, 22)
         + inst(248, 22)
         + inst(65, 35, 37, 34, 31)                    # a SECOND chain into the same slot
         + inst(61, 1, 38, 37)
         + inst(62, 5, 38)
         + inst(253))
    return module(PRE_IDS + [10, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38], b)


def mod_atomic_writes_the_vote(swap):
    """A vote-derived value written to a UAV by an atomic, in straight-line code.

    The leaving-scan tested word 3 -- an atomic's POINTER -- while the value it writes is further
    along, so the scan fired and read the wrong word. `swap` exchanges pointer and value: the swapped
    form was already refused, which is what proved the scan ran at all.
    """
    ptr, val = (34, 10) if not swap else (10, 34)
    b = (preamble(SC_INPUT)
         + inst(21, 30, 32, 0) + inst(43, 30, 31, 4)
         + inst(32, 33, SC_STORAGE_BUFFER, 30) + inst(59, 33, 34, SC_STORAGE_BUFFER)
         + entry()
         + inst(335, 1, 10, 6, 9)
         + inst(234, 30, 39, ptr, 6, 6, val)           # OpAtomicIAdd
         + inst(62, 5, 11)
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 33, 34, 39], b)


def mod_copy_memory_carries_taint(via_copy):
    """A tainted local moved to the colour output.

    `via_copy` selects OpCopyMemory, which moves a value with no OpLoad/OpStore pair for the closure
    to follow; False is the same module written as a load and a store, which was always refused. The
    pair is the experiment.
    """
    b = (preamble(SC_INPUT)
         + inst(32, 30, SC_FUNCTION, 1) + inst(59, 30, 31, SC_FUNCTION)
         + entry()
         + inst(335, 1, 10, 6, 9)
         + inst(62, 31, 10)                            # store the vote into a local
         + (inst(63, 5, 31) if via_copy                # OpCopyMemory out
            else inst(61, 1, 32, 31) + inst(62, 5, 32))
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 32], b)


def mod_dead_vote():
    """A divergent vote whose region has no observable effect and whose merge carries nothing."""
    b = (preamble(SC_INPUT)
         + inst(335, 1, 10, 6, 9)
         + entry()
         + inst(247, 22, 0)
         + inst(250, 10, 21, 22)
         + inst(248, 21) + inst(129, 2, 25, 11, 11) + inst(249, 22)   # then: an unused OpFAdd
         + inst(248, 22) + inst(62, 5, 11) + inst(253))               # merge stores a CONSTANT
    return module(PRE_IDS + [10, 20, 21, 22, 25], b)


def mod_unstructured():
    """A vote-conditioned branch with no merge instruction: the region has no bound to trust."""
    b = (preamble(SC_INPUT)
         + inst(335, 1, 10, 6, 9)
         + entry()
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
         + entry()
         + inst(81, 30, 32, 10, 0)                     # OpCompositeExtract .x
         + inst(62, 5, 32)
         + inst(253))
    return module(PRE_IDS + [10, 20, 30, 31, 32], b)


def mod_no_votes():
    b = preamble(SC_INPUT) + entry() + inst(62, 5, 11) + inst(253)
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
    expect_refused("a vote over a phi whose DIVERGENT branch chose between uniform constants",
                   mod_phi_before_vote(identical=False))
    expect_refused("a vote over a LOCAL written per lane and read back through an access chain",
                   mod_vote_over_a_local(SC_FUNCTION))
    expect_refused("a vote reaching an OpSwitch selector whose cases store different constants",
                   mod_vote_reaches_a_switch())
    expect_refused("a vote stored and reloaded through TWO chains into one local slot",
                   mod_taint_through_two_chains(same_value=False))
    expect_refused("a vote reaching one slot through an access chain and an OpCopyObject of it",
                   mod_slot_through_copyobject(tainted_store=True))
    expect_refused("a vote-guarded region performing an INTEGER atomic",
                   mod_region_performs_atomic(234))
    expect_refused("a vote-guarded region performing a FLOAT atomic (outside the old range)",
                   mod_region_performs_atomic(6035))
    expect_refused("a vote-guarded region whose store is read back through a second chain",
                   mod_region_stores_through_second_chain())
    expect_refused("a vote-derived value written to a UAV by an atomic",
                   mod_atomic_writes_the_vote(swap=False))
    expect_refused("...and with the atomic's pointer and value exchanged",
                   mod_atomic_writes_the_vote(swap=True))
    expect_refused("a tainted local moved to the colour output by OpCopyMemory",
                   mod_copy_memory_carries_taint(via_copy=True))
    expect_refused("...and the same module written as a load and a store",
                   mod_copy_memory_carries_taint(via_copy=False))

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
    expect_admitted("a vote over a phi whose edges all carry the SAME value",
                    mod_phi_before_vote(identical=True))
    expect_admitted("a vote over a constant buffer read through an access chain",
                    mod_vote_over_a_local(SC_STORAGE_BUFFER))
    expect_admitted("two chains into one slot carrying a value the vote never touched",
                    mod_taint_through_two_chains(same_value=True))
    expect_admitted("a chain and an OpCopyObject carrying a value the vote never touched",
                    mod_slot_through_copyobject(tainted_store=False))

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
