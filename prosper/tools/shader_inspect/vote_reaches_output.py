#!/usr/bin/env python3
"""Does a fragment module's wave vote reach a colour output?

This is the admission question for #3464's wave32 allowance, and it is NOT the question the shipping
classifier asks. That classifier admits a module whose reason set is exactly
`kFragmentWaveReasonWaveAny`, on the argument that a control-flow-only vote is width-agnostic. Two
things undermine relying on it alone:

  - "control-flow only" is never tested. The only site raising `kFragmentWaveReasonScalarReduce` is
    gated on a six-opcode SCC-as-data list, and the taint has known escapes: `sel()` drops the taint
    that `bsel()` propagates, and a vote routed into VCC is read per-lane by `v_cndmask_b32` with no
    marker. So a module can consume a vote as data and still present as `0x2`.
  - measured, the reason bits point the wrong way. Evergate's admitted modules DO feed votes into
    `OpSelect`/`OpPhi` and render correctly; Alex Kidd's are branch-only.

So this asks the question that actually matters for correctness: **can the vote's value influence a
colour output?** A vote that cannot reach an `Output` store is safe to evaluate at a narrower width
however it is consumed, because nothing it touches leaves the shader.

The closure is forward and deliberately over-approximate: SSA operand edges, plus store/load through
`Function`-storage variables (a vote stashed in a local and reloaded later still counts). Over-
approximating is the right direction — a false "reaches output" costs a title its place on the
allowlist, a false "does not" would ship a wrong render.

    python vote_reaches_output.py <dir-of-spv> [...]

Exit 0 if no module's vote reaches an output, 1 if any does, 2 on usage/read error. Prints one line
per module so a title can be judged shader by shader rather than in aggregate.
"""
import struct
import sys
from pathlib import Path

MAGIC = 0x07230203
OP_VARIABLE = 59
OP_STORE = 62
OP_LOAD = 61
OP_ACCESS_CHAIN = 65
OP_GROUP_NON_UNIFORM_ANY = 335      # matches the emitter's own Op_GroupNonUniformAny
OP_GROUP_NON_UNIFORM_BALLOT = 339
STORAGE_OUTPUT = 3
STORAGE_FUNCTION = 7

# Instructions whose result id sits at word 2 (type, result, ...). Branches, stores and merges have
# no result; treating them as if they did would taint an unrelated id.
NO_RESULT = {62, 249, 250, 247, 248, 251, 252, 253, 254, 246, 245 - 245}


def instructions(words):
    off = 5
    while off < len(words):
        n = words[off] >> 16
        op = words[off] & 0xffff
        if not n or off + n > len(words):
            return
        yield op, words[off:off + n]
        off += n


def analyse(path):
    """(reaches_output, n_votes, n_output_stores) for one module."""
    raw = Path(path).read_bytes()
    if len(raw) < 20 or struct.unpack_from("<I", raw, 0)[0] != MAGIC:
        raise ValueError("%s is not SPIR-V" % path)
    words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    insts = list(instructions(words))

    outputs, functions = set(), set()
    for op, w in insts:
        if op == OP_VARIABLE and len(w) >= 4:
            if w[3] == STORAGE_OUTPUT:
                outputs.add(w[2])
            elif w[3] == STORAGE_FUNCTION:
                functions.add(w[2])

    tainted = {w[2] for op, w in insts
               if op in (OP_GROUP_NON_UNIFORM_ANY, OP_GROUP_NON_UNIFORM_BALLOT) and len(w) >= 3}
    votes = len(tainted)

    # Access chains into an output variable are themselves output pointers.
    for op, w in insts:
        if op == OP_ACCESS_CHAIN and len(w) >= 4 and w[3] in outputs:
            outputs.add(w[2])

    # Forward closure to a fixed point. Re-run because SSA order does not guarantee one pass
    # suffices once loads through locals are followed.
    tainted_ptrs = set()
    changed = True
    while changed:
        changed = False
        for op, w in insts:
            if op == OP_STORE and len(w) >= 3:
                ptr, val = w[1], w[2]
                if val in tainted and ptr in functions and ptr not in tainted_ptrs:
                    tainted_ptrs.add(ptr)
                    changed = True
                continue
            if op == OP_LOAD and len(w) >= 4:
                if w[3] in tainted_ptrs and w[2] not in tainted:
                    tainted.add(w[2])
                    changed = True
                continue
            if op in NO_RESULT or len(w) < 3:
                continue
            result = w[2]
            if result in tainted:
                continue
            if any(t in tainted for t in w[3:]):
                tainted.add(result)
                changed = True

    output_stores = 0
    for op, w in insts:
        if op == OP_STORE and len(w) >= 3 and w[1] in outputs and w[2] in tainted:
            output_stores += 1
    return output_stores > 0, votes, output_stores


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-3], file=sys.stderr)
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
            reaches, votes, stores = analyse(f)
        except Exception as exc:                      # a module we cannot parse is not a clean one
            print("  %-52s UNREADABLE (%s)" % (f.name[:52], exc))
            bad += 1
            continue
        verdict = "REACHES OUTPUT (%d store(s))" % stores if reaches else "no output path"
        print("  %-52s votes=%-3d %s" % (f.name[:52], votes, verdict))
        bad += reaches

    print("\n%d of %d module(s) let a vote reach a colour output." % (bad, len(files)))
    if bad:
        print("Those cannot be admitted at a narrower wave width on this argument: the vote's value")
        print("can influence a pixel, so evaluating it over 32 lanes instead of 64 can change it.")
    else:
        print("No vote reaches a colour output, so none of these can change a pixel by being")
        print("evaluated at a narrower width, however each is consumed internally.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
