#!/usr/bin/env python3
"""A tool that prints `__doc__` as its usage message must HAVE a `__doc__`.

`print(__doc__)` is this repository's house style for "you invoked me wrongly, here is how to invoke
me". It is one token, it never drifts from the header, and it is silently conditional on something
no reader checks: a module whose header is made of `#` comments has `__doc__ == None`, so the usage
message is the literal string `None`. The tool still exits non-zero, so nothing looks broken -- the
caller is simply told nothing and guesses.

That has now happened three times, in three different files:

  * xref.py (#2399/#2558) -- `--help` printed `None`, the reporter guessed `--addr`, and the guess
    cleared the arity check, parsed a 27 MB module, decoded 527,707 sites and printed an empty
    exit-0 answer for an address with two real references.
  * waitgraph.py and il2cpp/resolve.py -- found by sweeping for the same shape while fixing #2558.
  * il2cpp/resolve.py AGAIN (#2642), four hours later. That PR's own purpose was to document
    `--emit-symtab` in the usage block, and it rewrote the header as `#` comments on the way past,
    so the text it existed to publish was the text that stopped printing. Nine CI checks were green:
    the only guard in the tree, `tools/re/test_xref.py:399`, asserts on xref.py by name.

A per-tool assertion cannot keep up with that, because the defect arrives in whichever tool nobody
is currently looking at. So this asks the question of the whole tree at once: for every Python
module under the scanned root that READS `__doc__`, does it have one?

The AST is the right instrument rather than a regex over `print(__doc__)`, because the dependency
takes several shapes that all break the same way -- `print(__doc__)`, `print(__doc__.rstrip())`,
`argparse.ArgumentParser(description=__doc__.splitlines()[0])`. The last two raise AttributeError on
None rather than printing it, which is louder but no more correct.

Deliberately narrow, so the check stays trusted and never fires on a correct file:

  * only a LOAD of the bare name `__doc__` counts. `SOMEMODULE.__doc__` is an attribute of another
    object -- that is what test_xref.py:399 does, and it is an assertion, not a usage message.
  * `__doc__ = "..."` (a store) is not a read.
  * a module that never mentions `__doc__` is never asked for a docstring. Most tools here are
    libraries or fixtures and have no usage message to print.

usage:
  python3 check_usage_text.py --root <dir> [--root <dir> ...]   # scan; exit 1 on any violation
  python3 check_usage_text.py --selftest                        # prove the scanner still fires
"""
import argparse
import ast
import os
import sys
import tempfile

# Directories whose contents are not this project's source. A CMake build directory lives at
# prosper/build-linux, i.e. INSIDE the scanned root, and can hold generated or vendored Python.
PRUNE_NAMES = {
    '.git', '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    'node_modules', '.venv', 'venv', 'third_party',
}


def _is_build_dir(path):
    """A configured CMake build tree, wherever it was placed and whatever it was named."""
    return os.path.exists(os.path.join(path, 'CMakeCache.txt'))


def reads_doc(tree):
    """True if the module evaluates the bare name `__doc__` -- see the narrowing notes above."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == '__doc__' and isinstance(node.ctx, ast.Load):
            return True
    return False


def scan(roots):
    """Returns (violations, examined) -- violations are (path, reason) pairs.

    `examined` is the count of modules that read `__doc__`, and it is returned so a caller can
    refuse a run that examined nothing. This checker's failure mode is silence: a walk that
    matched no files, or a detector that stopped detecting, reports a clean tree forever.
    """
    violations = []
    examined = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in PRUNE_NAMES
                                 and not _is_build_dir(os.path.join(dirpath, d)))
            for name in sorted(filenames):
                if not name.endswith('.py'):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=path)
                except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                    # Unparseable is reported, never skipped: a file this cannot read is a file
                    # whose usage text it cannot vouch for.
                    violations.append((path, 'could not be parsed: %s' % exc))
                    continue
                if not reads_doc(tree):
                    continue
                examined.append(path)
                doc = ast.get_docstring(tree)
                if doc is None:
                    violations.append((path, 'reads __doc__ but has no module docstring, so its '
                                             'usage message is the literal string "None"'))
                elif not doc.strip():
                    violations.append((path, 'reads __doc__ but its module docstring is empty, so '
                                             'its usage message is blank'))
    return violations, examined


SELFTEST_CASES = {
    # (filename, source, must_be_reported)
    'violator.py': ("#!/usr/bin/env python3\n"
                    "# tool — does a thing.\n"
                    "# usage: tool <arg>\n"
                    "import sys\n"
                    "def main():\n"
                    "    print(__doc__)\n"
                    "    return 1\n", True),
    'violator_argparse.py': ("import argparse\n"
                             "# usage lives only in comments\n"
                             "p = argparse.ArgumentParser(description=__doc__.splitlines()[0])\n",
                             True),
    'violator_empty.py': ('""""""\n'
                          "def main():\n"
                          "    print(__doc__)\n", True),
    # --- false-positive arms. A checker that fires on a correct file gets disabled rather than
    # --- heeded, so half of this selftest is files it MUST leave alone.
    'compliant.py': ('#!/usr/bin/env python3\n'
                     '"""tool — does a thing.\n'
                     '\n'
                     'usage: tool <arg>\n'
                     '"""\n'
                     "def main():\n"
                     "    print(__doc__)\n"
                     "    return 1\n", False),
    'no_doc_use.py': ("#!/usr/bin/env python3\n"
                      "# An ordinary module with a comment header and no usage message. It must not\n"
                      "# be required to carry a docstring it never prints.\n"
                      "def helper():\n"
                      "    return 42\n", False),
    'attribute_only.py': ("#!/usr/bin/env python3\n"
                          "# test_xref.py:399's shape: asserting on ANOTHER module's __doc__.\n"
                          "import importlib\n"
                          "OTHER = importlib.import_module('json')\n"
                          "assert OTHER.__doc__ is not None\n", False),
    'store_only.py': ("#!/usr/bin/env python3\n"
                      "# Assigning __doc__ is not reading it.\n"
                      "__doc__ = 'set at runtime'\n", False),
}


def selftest():
    """Build the two directions by hand and require the scanner to separate them.

    The positive cases are written here rather than drawn from the repository on purpose: a control
    sampled from the same tree the scan just called clean can only confirm the machinery runs on
    cases that tree contains. These are constructed instances of the class being tested.
    """
    failures = 0
    with tempfile.TemporaryDirectory(prefix='usage_text_selftest_') as work:
        for name, (src, _) in SELFTEST_CASES.items():
            with open(os.path.join(work, name), 'w', encoding='utf-8') as f:
                f.write(src)
        violations, examined = scan([work])
        reported = {os.path.basename(p) for p, _ in violations}

        for name, (_, must_report) in sorted(SELFTEST_CASES.items()):
            ok = (name in reported) == must_report
            print('%s selftest: %s is %s' % ('[ok]  ' if ok else '[FAIL]', name,
                                             'REPORTED' if must_report else 'left alone'))
            if not ok:
                failures += 1

        expected_examined = {'violator.py', 'violator_argparse.py', 'violator_empty.py',
                             'compliant.py'}
        got_examined = {os.path.basename(p) for p in examined}
        ok = got_examined == expected_examined
        print('%s selftest: exactly the __doc__ readers were examined (%s)'
              % ('[ok]  ' if ok else '[FAIL]', sorted(got_examined)))
        if not ok:
            failures += 1

    print('selftest: %d failure(s)' % failures)
    return failures


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', action='append', default=[],
                    help='directory to scan recursively; repeatable')
    ap.add_argument('--selftest', action='store_true',
                    help='run the scanner against hand-built compliant and violating modules')
    args = ap.parse_args(argv[1:])

    rc = 0
    if args.selftest:
        rc |= 1 if selftest() else 0
    if not args.root:
        if not args.selftest:
            ap.error('nothing to do: pass --root and/or --selftest')
        return rc

    missing = [r for r in args.root if not os.path.isdir(r)]
    if missing:
        print('check_usage_text: not a directory: %s' % ', '.join(missing), file=sys.stderr)
        return 2

    violations, examined = scan(args.root)
    if not examined:
        # "0 violations" and "the walk found nothing" are the same output otherwise, and this whole
        # file exists because of a check that could not tell those apart.
        print('check_usage_text: examined 0 modules under %s — the scan matched nothing, which is '
              'an instrument failure, not a clean tree.' % ', '.join(args.root), file=sys.stderr)
        return 2

    for path, reason in violations:
        print('%s: %s' % (path, reason), file=sys.stderr)
    print('check_usage_text: %d module(s) read __doc__, %d violation(s)'
          % (len(examined), len(violations)))
    return rc | (1 if violations else 0)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
