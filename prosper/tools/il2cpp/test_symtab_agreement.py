#!/usr/bin/env python3
"""il2cpp_symtab_agreement — prosper's RUNTIME resolver must answer exactly what resolve.py answers.

#2551 added in-process IL2CPP symbolication (src/host/symbols/il2cpp_symbols.cpp). It deliberately does not
re-derive resolve.py's mapping: resolve.py emits the symbol table (--emit-symtab) and the runtime
side looks up in it. But "shares the input" is not "gives the same answer" — the two sides still
have independent implementations of the nearest-preceding rule, its window, and its tie handling,
and each of those is a place they can silently drift apart. A runtime resolver that produces
plausible-looking names nobody checked is worse than none.

So this drives both, over the same synthesized corpus, through their real command-line interfaces:

    script.json --(resolve.py --emit-symtab)--> symtab --(test_il2cpp_symbols --probe)--> names
    script.json --(resolve.py <addrs>)-------------------------------------------------> names

and requires them to agree address-for-address, including on the addresses that resolve to nothing.

No game dump and no Il2CppDumper are involved: the corpus is synthesized here, so this runs
everywhere and cannot silently skip. The real-title agreement (PPSA24651, 87,851 methods) is
recorded in tools/il2cpp/README.md as a measurement, not as a test — reproducing it needs a dump.

Usage: test_symtab_agreement.py <path to the test_il2cpp_symbols binary>
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLVE = os.path.join(HERE, 'resolve.py')
WINDOW = 0x8000

# Anchor addresses the named checks assert against. Kept clear of the generated clusters (whose
# last method starts at 0x400000 + 39*0x30000 = 0xb50000) and of each other; synth_script_json
# verifies that rather than trusting this comment.
GENERIC_AT = 0x800000
TIED_AT = 0x900000
UTF8_AT = 0xd00000

RESOLVED_RE = re.compile(r'^(.*)  \(\+0x([0-9a-f]+)\)$')

failures = 0


def check(ok, what):
    global failures
    print('%s %s' % ('[ok]  ' if ok else '[FAIL]', what))
    if not ok:
        failures += 1


def synth_script_json(path):
    """A method table with the shapes that make the two implementations disagree if either is
    wrong: tight clusters (nearest-preceding boundaries), gaps wider than the window (so some
    probes MUST resolve to nothing), a shared address (the tie rule), a name containing ', '
    (generics), and a non-ASCII name."""
    methods = []
    rva = 0x1e3e00
    for i in range(120):
        methods.append({'Address': rva, 'Name': 'Cluster%03d.Type$$Method' % i})
        rva += 0x40 if i % 3 else 0x2000        # tight runs punctuated by wider steps
    rva = 0x400000
    for i in range(40):
        methods.append({'Address': rva, 'Name': 'Sparse%02d.Type$$Method' % i})
        rva += 0x30000                          # every gap far wider than the window
    methods.append({'Address': GENERIC_AT,
                    'Name': 'Synth.Generic.Container<Key, Value>$$Insert'})
    methods.append({'Address': TIED_AT, 'Name': 'Tied$$Method'})
    methods.append({'Address': TIED_AT, 'Name': 'Tied$$MethodZzz'})    # same address, later name
    methods.append({'Address': UTF8_AT, 'Name': 'Nämespace.Tÿpe$$Méthod'})
    # Entries without an Address are what resolve.py's load() filters out; include one so both
    # sides see the same post-filter record set rather than differing by it.
    methods.append({'Name': 'NoAddress$$Method'})

    # The named checks below assert on WHICH method three specific addresses resolve to, so they
    # are only meaningful while nothing else in the generated corpus sits at those addresses. The
    # generator's own arithmetic put a Sparse method on top of the UTF-8 one once already, and the
    # only symptom was one confusing failure naming a method the check had never heard of. Make
    # the collision loud and immediate instead of leaving it to be diagnosed downstream.
    counts = {}
    for m in methods:
        if 'Address' in m:
            counts[m['Address']] = counts.get(m['Address'], 0) + 1
    expected = {GENERIC_AT: 1, TIED_AT: 2, UTF8_AT: 1}
    for addr, want in expected.items():
        if counts.get(addr, 0) != want:
            raise SystemExit('test_symtab_agreement: anchor 0x%x occurs %d time(s), expected %d — '
                             'the generated corpus collides with it, so the named checks below '
                             'would assert against the wrong method'
                             % (addr, counts.get(addr, 0), want))

    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'ScriptMethod': methods}, f)
    return [m['Address'] for m in methods if 'Address' in m]


def probe_addresses(starts):
    """Boundaries first: every place the two implementations could round differently."""
    probes = []
    for a in sorted(set(starts)):
        probes += [a, a + 1, a + WINDOW - 1, a + WINDOW]
    probes += [0, 1, 0x1e3dff, UTF8_AT + WINDOW, 0xffffffff]
    return sorted(set(p for p in probes if p >= 0))


def run(args):
    """Both sides carry non-ASCII method names, so the pipe encoding is pinned rather than taken
    from the locale: under LC_ALL=C a text-mode pipe decodes as ASCII and the run dies on the one
    record this test added to cover that case."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    return subprocess.run(args, capture_output=True, encoding='utf-8', check=True, env=env).stdout


def run_offline(script_path, probes):
    """resolve.py's own CLI, exactly as tools/il2cpp/README.md documents it."""
    args = [sys.executable, RESOLVE, script_path] + ['0x%x' % p for p in probes]
    out = run(args)
    answers = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        token, rest = line.split(None, 1)
        rva = int(token, 16)
        m = RESOLVED_RE.match(rest.strip())
        answers[rva] = (m.group(1), int(m.group(2), 16)) if m else None
    return answers


def run_runtime(binary, symtab, probes):
    args = [binary, '--probe', symtab] + ['0x%x' % p for p in probes]
    out = run(args)
    answers = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        rva_text, rest = line.split(None, 1)
        rva = int(rva_text, 16)
        if rest.startswith('resolved '):
            body = rest[len('resolved '):]
            name, _, off = body.rpartition(' +0x')
            answers[rva] = (name, int(off, 16))
        else:
            answers[rva] = None
    return answers


def compare(offline, runtime, probes):
    """Returns the list of disagreements. Kept separate so the mutation arm below can prove it is
    capable of reporting one — a comparator that always returns [] would make every check pass."""
    bad = []
    for p in probes:
        if offline.get(p, 'missing') != runtime.get(p, 'missing'):
            bad.append((p, offline.get(p, 'missing'), runtime.get(p, 'missing')))
    return bad


def check_usage_message():
    """resolve.py's no-argument run must print its usage, not the literal string `None`.

    `main()` answers a bad invocation with `print(__doc__)`, which is a usage message only while the
    header is a docstring. Rewritten as `#` comments it becomes `None` — the tool still exits 1, so
    nothing looks broken and the caller is simply told nothing. That is #2399, and it came back in
    #2642 on this very file, in the same change that added the `--emit-symtab` line being asserted
    below. This is the behavioural half of the guard; tools/ci/check_usage_text.py (ctest
    `tools_usage_text`) asks the same question of every tool in the tree at once.
    """
    p = subprocess.run([sys.executable, RESOLVE], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    check(p.returncode != 0, 'resolve.py refuses a no-argument invocation (exit %d)' % p.returncode)
    check(out != '' and out != 'None',
          'resolve.py prints real usage text, not %r' % (out[:40] or ''))
    check('usage' in out.lower() and 'script.json' in out,
          'the usage text names the invocation shape')
    # The regression is specifically that a usage block gets edited and then never printed, so
    # assert the CURRENT feature is in the text a caller actually receives.
    check('--emit-symtab' in out, 'the printed usage documents --emit-symtab')


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = argv[1]
    if not os.path.exists(binary):
        print('[FAIL] probe binary not found: %s' % binary)
        return 2

    check_usage_message()

    with tempfile.TemporaryDirectory(prefix='il2cpp_symtab_agree_') as work:
        script_path = os.path.join(work, 'script.json')
        symtab = os.path.join(work, 'out.symtab')
        starts = synth_script_json(script_path)
        subprocess.run([sys.executable, RESOLVE, script_path, '--emit-symtab', symtab], check=True)

        with open(symtab, encoding='utf-8') as f:
            header = f.readline().strip()
        check(header.startswith('prosper-il2cpp-symtab v1'), 'emitted symtab carries the magic line')
        check('window=0x%x' % WINDOW in header,
              'emitted header declares the window: %r' % header)

        probes = probe_addresses(starts)
        offline = run_offline(script_path, probes)
        runtime = run_runtime(binary, symtab, probes)

        check(len(offline) == len(probes) and len(runtime) == len(probes),
              'both sides answered all %d probes (offline=%d runtime=%d)'
              % (len(probes), len(offline), len(runtime)))

        # The domain must actually contain both outcomes, or "they agree" would be vacuous: a pair
        # of resolvers that both answered None everywhere would pass a bare agreement check.
        resolved = sum(1 for p in probes if offline.get(p) is not None)
        unresolved = len(probes) - resolved
        check(resolved > 0 and unresolved > 0,
              'the probe set exercises both outcomes (%d resolved, %d unresolved)'
              % (resolved, unresolved))

        bad = compare(offline, runtime, probes)
        check(not bad, 'runtime and offline agree on all %d probes%s'
              % (len(probes), '' if not bad else '; first 5: %r' % bad[:5]))

        # Named checks on the shapes, so a regression says WHICH rule broke rather than only that
        # some address disagreed.
        tie = runtime.get(TIED_AT)
        check(tie is not None and tie[0] == 'Tied$$MethodZzz',
              'runtime takes the later name at a tied address: %r' % (tie,))
        check(offline.get(TIED_AT) == tie, '…and offline agrees: %r' % (offline.get(TIED_AT),))
        generic = runtime.get(GENERIC_AT)
        check(generic is not None and generic[0].endswith('>$$Insert') and ', ' in generic[0],
              'the generic name survives with its ", " intact: %r' % (generic,))
        utf8 = runtime.get(UTF8_AT)
        check(utf8 is not None and utf8[0] == 'Nämespace.Tÿpe$$Méthod',
              'a non-ASCII name round-trips through the symtab: %r' % (utf8,))
        check(offline.get(UTF8_AT + WINDOW) is None and runtime.get(UTF8_AT + WINDOW) is None,
              'both sides refuse an address one window past the last method')

        # MUTATION ARM 1 — the comparator can report a disagreement. Without this, a silently
        # broken compare() (or a parse that produced two empty dicts) would make every agreement
        # check above pass while proving nothing at all.
        salted = dict(runtime)
        victim = next(p for p in probes if runtime.get(p) is not None)
        salted[victim] = ('DeliberatelyWrong$$Name', 0)
        forced = compare(offline, salted, probes)
        check(len(forced) == 1 and forced[0][0] == victim,
              'the comparator DOES report a planted disagreement at 0x%x: %r' % (victim, forced))

        # MUTATION ARM 2 — the two sides move TOGETHER when the input moves. Agreeing on a fixed
        # probe set is also what two resolvers that both ignore their input would do; this requires
        # that shifting a probe across a method boundary changes both answers, identically.
        base = sorted(set(starts))[5]
        near, far = base + 1, base + WINDOW
        check(offline.get(near) is not None and offline.get(far) != offline.get(near),
              'the offline answer changes across the window at 0x%x' % base)
        check(runtime.get(near) == offline.get(near) and runtime.get(far) == offline.get(far),
              'the runtime answer changes with it, to the same values')

    print('%s: %d failure(s)' % ('FAILED' if failures else 'PASSED', failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
