#!/usr/bin/env python3
"""Would this fragment module answer the same over two 32-lane groups as over one 64-lane wave?

This is the admission question for #3464's native-wave32 allowance, and it is NOT the question the
reason-set classifier asks. That classifier keys on how a vote was PRODUCED -- a module whose reason
set is exactly `kFragmentWaveReasonWaveAny` is called "control-flow only" -- and the property it
needs is about what the vote can DO.

A vote clears on either of two independent grounds:

  (a) **its operand is provably wave-uniform.** `Any(P)` reduces P over whatever lanes the group
      holds, so when every invocation of the draw agrees on P the reduction IS P -- at 64 lanes, at
      32, at any width. This is the arm that carries the corpus: the guest's `s_cbranch_execz`-style
      tests are overwhelmingly comparisons of constant-buffer scalars.

  (b) **its result cannot influence a colour output** -- through data flow OR through control
      dependence. A dead vote may be answered any way at all.

Both halves of (b) matter, and the second is a correction rather than a refinement. An earlier
version of this tool asked only whether a vote's VALUE could reach an `Output` store. That misses
the case the whole allowance turns on: a vote used only as a branch condition never flows into a
store, yet it decides WHICH store runs. With the guest predicate true in the upper 32 lanes and
false in the lower, a 64-lane `any()` takes a branch two independent 32-lane groups disagree about,
and whatever the taken block writes lands in one half only. Measured over 49 real fragment modules
from four titles, value reachability clears 49 of 49 while control dependence clears 1 -- so the
weaker question could not see the population at all. With arm (a) added, 38 of the 49 clear.

The error direction is chosen throughout: a module that cannot be parsed, a construct with no merge,
an opcode nobody classified -- each answers "not proven", which costs a draw its native-width fast
path rather than shipping a wrong pixel.

    python wave_width_independent.py <dir-of-spv> [...]

Exit 0 if every module is provably width-independent, 1 if any is not, 2 on usage/read error. Prints
one line per module so a title can be judged shader by shader rather than in aggregate.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spirv_operands import id_operand_indices, result_index  # noqa: E402

MAGIC = 0x07230203

OP_LABEL, OP_BRANCH, OP_BRANCH_COND = 248, 249, 250
OP_SELECTION_MERGE, OP_LOOP_MERGE, OP_SWITCH = 247, 246, 251
OP_PHI, OP_VARIABLE, OP_STORE, OP_LOAD, OP_ACCESS_CHAIN = 245, 59, 62, 61, 65
OP_GROUP_NON_UNIFORM_ANY = 335      # matches the emitter's own Op_GroupNonUniformAny
OP_GROUP_NON_UNIFORM_BALLOT = 339
OP_IMAGE_WRITE, OP_KILL, OP_FUNCTION_CALL = 99, 252, 57

STORAGE_UNIFORM_CONSTANT, STORAGE_INPUT, STORAGE_UNIFORM = 0, 1, 2
STORAGE_OUTPUT, STORAGE_FUNCTION, STORAGE_PUSH, STORAGE_SB = 3, 7, 9, 12
UNIFORM_STORAGE = {STORAGE_UNIFORM, STORAGE_SB, STORAGE_PUSH, STORAGE_UNIFORM_CONSTANT}

# OpConstantTrue/False, OpConstant, OpConstantComposite/Sampler/Null, and the spec-constant forms.
CONSTANT_OPS = {41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52}

# Values that differ between the invocations of one draw. Naming the DIVERGENT sources rather than
# the uniform ones is deliberate: an opcode nobody thought about has to earn uniformity through its
# operands instead of inheriting it, so the unknown case fails towards "not proven".
DIVERGENT_OPS = (
    set(range(87, 107)) |          # OpImageSample* / Fetch / Gather / Read / Query*
    set(range(207, 216)) |         # derivatives: dPdx/dPdy/Fwidth, coarse and fine
    set(range(227, 243)) |         # atomics
    set(range(333, 367)) |         # every subgroup instruction
    {OP_FUNCTION_CALL}
)


def instructions(words):
    off = 5
    while off < len(words):
        n = words[off] >> 16
        op = words[off] & 0xffff
        if not n or off + n > len(words):
            return
        yield op, words[off:off + n]
        off += n


class Module:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        if len(raw) < 20 or struct.unpack_from("<I", raw, 0)[0] != MAGIC:
            raise ValueError("%s is not SPIR-V" % path)
        words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
        self.insts = list(instructions(words))

        self.var_storage = {w[2]: w[3] for op, w in self.insts
                            if op == OP_VARIABLE and len(w) >= 4}
        self.outputs = {v for v, sc in self.var_storage.items() if sc == STORAGE_OUTPUT}
        self.locals = {v for v, sc in self.var_storage.items() if sc == STORAGE_FUNCTION}
        # Access chains inherit their base's class, transitively; one pass is not enough.
        changed = True
        while changed:
            changed = False
            for op, w in self.insts:
                if op != OP_ACCESS_CHAIN or len(w) < 4:
                    continue
                if w[3] in self.outputs and w[2] not in self.outputs:
                    self.outputs.add(w[2])
                    changed = True
                if w[3] in self.locals and w[2] not in self.locals:
                    self.locals.add(w[2])
                    changed = True

        self.votes = []                             # (result, operand)
        self.ballots = set()
        for op, w in self.insts:
            if op == OP_GROUP_NON_UNIFORM_ANY and len(w) >= 5:
                self.votes.append((w[2], w[4]))
            elif op == OP_GROUP_NON_UNIFORM_BALLOT and len(w) >= 5:
                self.votes.append((w[2], w[4]))
                self.ballots.add(w[2])

        self._build_cfg()

    def _build_cfg(self):
        self.blocks, self.succ, self.merge_of, self.branch_cond = {}, {}, {}, {}
        current = None
        for op, w in self.insts:
            if op == OP_LABEL and len(w) >= 2:
                current = w[1]
                self.blocks[current] = []
                continue
            if current is None:
                continue
            self.blocks[current].append((op, w))
            if op in (OP_SELECTION_MERGE, OP_LOOP_MERGE) and len(w) >= 2:
                self.merge_of[current] = w[1]
            elif op == OP_BRANCH and len(w) >= 2:
                self.succ[current] = [w[1]]
            elif op == OP_BRANCH_COND and len(w) >= 4:
                self.branch_cond[current] = w[1]
                self.succ[current] = [w[2], w[3]]
            elif op == OP_SWITCH and len(w) >= 3:
                self.succ[current] = [w[2]] + [w[i] for i in range(4, len(w), 2)]

    # -- arm (a) -----------------------------------------------------------------------------
    def _writes_memory(self):
        """Does this shader write buffer or image memory -- anything but its own locals and colour?

        A storage buffer the same draw writes is not uniform: one invocation's store changes what
        another's load returns. Constant buffers, push constants and uniform constants are read-only
        and keep their uniformity regardless.
        """
        for op, w in self.insts:
            if op == OP_IMAGE_WRITE or 227 <= op <= 242:
                return True
            if op == OP_STORE and len(w) >= 3 and w[1] not in self.locals \
                    and w[1] not in self.outputs:
                return True
        return False

    def _uniform_storage(self, sc):
        if sc == STORAGE_SB:
            return not self._writes_memory()
        return sc in (STORAGE_UNIFORM, STORAGE_PUSH, STORAGE_UNIFORM_CONSTANT)

    def uniform_values(self):
        uni = set()
        for op, w in self.insts:
            if op in CONSTANT_OPS and len(w) >= 3:
                uni.add(w[2])
            if op == OP_VARIABLE and len(w) >= 4 and self._uniform_storage(w[3]):
                uni.add(w[2])
        changed = True
        while changed:
            changed = False
            for op, w in self.insts:
                ri = result_index(op)
                if ri is None or len(w) <= ri or w[ri] in uni:
                    continue
                if op == OP_GROUP_NON_UNIFORM_ANY:
                    # A vote's own result is uniform once its operand is: an all-lanes reduction of
                    # a value every lane already agrees on. Chained votes resolve by fixed point.
                    if len(w) >= 5 and w[4] in uni:
                        uni.add(w[ri])
                        changed = True
                    continue
                if op in DIVERGENT_OPS:
                    continue
                if op == OP_LOAD:
                    if len(w) >= 4:
                        ptr = w[3]
                        sc = self.var_storage.get(ptr)
                        if (sc is not None and self._uniform_storage(sc)) \
                                or (ptr in uni and sc is None):
                            uni.add(w[ri])
                            changed = True
                    continue
                if op == OP_PHI:
                    # IDENTICAL incoming values, not merely uniform ones -- a phi chooses between
                    # its edges, so uniform values say nothing when the branch choosing is
                    # divergent. `q = phi(true, false)` after `if (divergent)` is exactly the
                    # divergent predicate, and both constants are uniform. See the C++ twin.
                    vals = [w[i] for i in range(3, len(w) - 1, 2)]
                    if vals and all(v == vals[0] for v in vals) and vals[0] in uni:
                        uni.add(w[ri])
                        changed = True
                    continue
                operands = [w[i] for i in id_operand_indices(op, len(w)) if i < len(w)]
                if operands and all(o in uni for o in operands):
                    uni.add(w[ri])
                    changed = True
        return uni

    # -- arm (b) -----------------------------------------------------------------------------
    def _region(self, head):
        """Blocks strictly inside head's construct, and the merge that bounds it."""
        merge = self.merge_of.get(head)
        seen, stack = set(), list(self.succ.get(head, []))
        while stack:
            b = stack.pop()
            if b == merge or b not in self.blocks or b in seen:
                continue
            seen.add(b)
            stack.extend(self.succ.get(b, []))
        return seen, merge

    def _dataflow(self, tainted, tainted_ptrs):
        changed = True
        while changed:
            changed = False
            for op, w in self.insts:
                if op == OP_STORE and len(w) >= 3:
                    if w[2] in tainted and w[1] in self.locals and w[1] not in tainted_ptrs:
                        tainted_ptrs.add(w[1])
                        changed = True
                    continue
                if op == OP_LOAD and len(w) >= 4:
                    if w[3] in tainted_ptrs and w[2] not in tainted:
                        tainted.add(w[2])
                        changed = True
                    continue
                ri = result_index(op)
                if ri is None or len(w) <= ri or w[ri] in tainted:
                    continue
                if any(w[i] in tainted for i in id_operand_indices(op, len(w)) if i < len(w)):
                    tainted.add(w[ri])
                    changed = True

    def _observable(self, op, w):
        """Does this instruction let the rest of the draw see that it ran?

        A store through anything that is not a Function-storage pointer -- a colour attachment, a
        storage buffer, an image. Restricting this to Output would make a UAV write invisible to the
        analysis, the same shape of blind spot as the control-dependence one, one level down.
        """
        if op == OP_STORE:
            return len(w) >= 3 and w[1] not in self.locals
        if op == 63:                                  # OpCopyMemory
            return len(w) >= 3 and w[1] not in self.locals
        if op == 64:                                  # OpCopyMemorySized
            return len(w) >= 4 and w[1] not in self.locals
        if op in (OP_IMAGE_WRITE, OP_KILL, 4416, 5380, 224, 225):
            return True                               # image write, kill, terminate, demote, barrier
        return 227 <= op <= 242                       # atomics

    def influences_output(self, seed):
        tainted, tainted_ptrs = {seed}, set()
        # A bounded fixed point, and the bound is conservative on purpose: falling out of the loop
        # still growing means the closure was NOT complete, so the honest answer is "it might" --
        # returning False there would admit a module on an unfinished analysis.
        settled = False
        for _ in range(32):
            self._dataflow(tainted, tainted_ptrs)
            for op, w in self.insts:                 # a tainted value leaving the shader
                if op == OP_STORE and len(w) >= 3 and w[1] not in self.locals and w[2] in tainted:
                    return True
                if op == OP_IMAGE_WRITE and len(w) >= 4 and (w[2] in tainted or w[3] in tainted):
                    return True
                if 227 <= op <= 242 and len(w) >= 4 and w[3] in tainted:
                    return True
            grew = False
            for head, cond in self.branch_cond.items():
                if cond not in tainted:
                    continue
                if head not in self.merge_of:
                    return True     # unstructured: the region has no bound we can trust
                inside, merge = self._region(head)
                for b in inside:
                    for op, w in self.blocks[b]:
                        if self._observable(op, w):
                            return True
                        # A store into a local inside the region is control-dependent too: whether
                        # it happened at all is the vote's answer.
                        if op == OP_STORE and len(w) >= 3 and w[1] in self.locals \
                                and w[1] not in tainted_ptrs:
                            tainted_ptrs.add(w[1])
                            grew = True
                for op, w in self.blocks.get(merge, []):
                    if op != OP_PHI or len(w) < 5 or w[2] in tainted:
                        continue
                    if any(w[i] != w[3] for i in range(5, len(w) - 1, 2)):
                        tainted.add(w[2])
                        grew = True
            if not grew:
                settled = True
                break
        return not settled


def analyse(path):
    """(width_independent, n_votes, n_unproved) for one module."""
    m = Module(path)
    if not m.votes:
        return True, 0, 0
    uniform = m.uniform_values()
    unproved = 0
    for result, operand in m.votes:
        # A ballot is never cleared by uniformity: its result is a per-lane bit MASK whose width is
        # the subgroup's, so a 32-lane ballot reports half a mask as though it were whole however
        # uniform the value being balloted. Only deadness can clear one.
        if result not in m.ballots and operand in uniform:
            continue
        if not m.influences_output(result):
            continue
        unproved += 1
    return unproved == 0, len(m.votes), unproved


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wave_width_independent.py <dir-of-spv> [...]", file=sys.stderr)
        return 2
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        files.extend(sorted(p.glob("*.spv")) if p.is_dir() else [p])
    if not files:
        print("no .spv files found", file=sys.stderr)
        return 2

    bad = 0
    for f in files:
        try:
            ok, votes, unproved = analyse(f)
        except Exception as exc:              # a module we cannot parse is not a proven one
            print("  %-52s UNREADABLE (%s)" % (f.name[:52], exc))
            bad += 1
            continue
        verdict = "width-independent" if ok else "NOT PROVEN (%d vote(s))" % unproved
        print("  %-52s votes=%-3d %s" % (f.name[:52], votes, verdict))
        bad += not ok

    print("\n%d of %d module(s) cannot be proved width-independent." % (bad, len(files)))
    if bad:
        print("Those keep the exact-width contract: their votes can change a pixel when a 64-lane")
        print("guest wave is answered by two independent 32-lane groups.")
    else:
        print("Every vote here is either wave-uniform or unable to influence a colour output, so")
        print("none can change a pixel by being answered at a narrower width.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
