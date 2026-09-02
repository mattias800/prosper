#!/usr/bin/env python3
"""Self-checking tests for wt_stash.py (exit code is truth).

The first arm is the one that gives the rest their meaning. It builds two linked worktrees by hand
and runs PLAIN `git stash` in each, and it asserts that the collision from #3174 actually happens:
worktree A pops and receives worktree B's content, while B's entry disappears from the stack. That
is a positive instance of the failure class, constructed outside the tool under test. Every later
arm claims the tool removes that failure, and none of those claims means anything if this fixture
cannot produce the failure in the first place -- so this arm is checked FIRST and its own failure is
reported as VOID rather than as a pass.

Everything is built from git primitives rather than from wt_stash's own view of the world, because
a fixture derived from the thing under test inherits its blind spots.

Run directly, or via ctest as wt_stash_tool.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = str(Path(__file__).resolve().parent / 'wt_stash.py')

FAILURES: list[str] = []
VOID = False

ENV = dict(os.environ)
ENV.update({
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_SYSTEM': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 't@e.st',
    'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 't@e.st',
    'GIT_TERMINAL_PROMPT': '0',
})
for stale in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'):
    ENV.pop(stale, None)


def check(name, got, want, extra=''):
    if got == want:
        print('  ok   {}'.format(name))
    else:
        msg = '{}: got {!r}, want {!r}'.format(name, got, want)
        if extra:
            msg += ' ({})'.format(extra)
        print('  FAIL {}'.format(msg))
        FAILURES.append(msg)


def git(cwd, *args, check_rc=True):
    p = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, env=ENV)
    if check_rc and p.returncode != 0:
        raise RuntimeError('git {} failed in {}: {}'.format(' '.join(args), cwd, p.stderr.strip()))
    return p.stdout


def rev(cwd, ref):
    """The sha a ref points at, or None -- never an exception, so an arm can assert on absence."""
    rc, out, _ = git_rc(cwd, 'rev-parse', '--verify', '--quiet', ref)
    return out.strip() if rc == 0 else None


def git_rc(cwd, *args):
    p = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, env=ENV)
    return p.returncode, p.stdout, p.stderr


def tool(cwd, *args):
    p = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True,
                       cwd=str(cwd), env=ENV)
    return p.returncode, p.stdout + p.stderr


def read(path):
    return Path(path).read_text()


def write(path, text):
    Path(path).write_text(text)


def make_repo(root):
    """A repo with a main checkout and two linked worktrees on their own branches.

    This mirrors the shape CLAUDE.md prescribes: each agent in .claude/worktrees/<slug> on its own
    fix/issue-NN-* branch, all sharing one .git common directory.
    """
    main = Path(root) / 'main-checkout'
    main.mkdir()
    git(main.parent, 'init', '-q', '-b', 'main', str(main))
    write(main / 'shared.txt', 'base\n')
    write(main / 'other.txt', 'base\n')
    git(main, 'add', '-A')
    git(main, 'commit', '-qm', 'base')
    a = Path(root) / 'wtA'
    b = Path(root) / 'wtB'
    git(main, 'worktree', 'add', '-q', str(a), '-b', 'fix/issue-3051-lane-a')
    git(main, 'worktree', 'add', '-q', str(b), '-b', 'fix/issue-3068-lane-b')
    return main, a, b


# --- arm 1: the collision is real ------------------------------------------------------------


def arm_plain_git_stash_collides():
    """POSITIVE CONTROL. Plain `git stash` in two worktrees; A's pop must take B's work."""
    global VOID
    print('[arm] plain git stash across two worktrees (positive control for #3174)')
    with tempfile.TemporaryDirectory() as root:
        main, a, b = make_repo(root)
        write(a / 'shared.txt', 'LANE-A\n')
        write(b / 'shared.txt', 'LANE-B\n')
        git(a, 'stash', 'push', '-q', '-m', 'lane A wip')
        git(b, 'stash', 'push', '-q', '-m', 'lane B wip')

        seen_from_a = git(a, 'stash', 'list')
        seen_from_b = git(b, 'stash', 'list')
        check('refs/stash is the SAME stack in both worktrees', seen_from_a, seen_from_b)
        check('both entries are on the one shared stack', len(seen_from_a.splitlines()), 2)

        git(a, 'stash', 'pop', '-q')
        got = read(a / 'shared.txt')
        check('lane A pops lane B\'s content into its own worktree', got, 'LANE-B\n')
        left = git(main, 'stash', 'list').splitlines()
        check('lane B\'s entry was consumed by lane A\'s pop', len(left), 1)
        check('what is left on the stack is lane A\'s own entry',
              'issue-3051' in left[0] if left else False, True)

        if FAILURES:
            VOID = True


# --- arm 2: the tool removes it --------------------------------------------------------------


def arm_wt_stash_isolates():
    print('[arm] wt_stash push/pop across the same two worktrees')
    with tempfile.TemporaryDirectory() as root:
        main, a, b = make_repo(root)
        write(a / 'shared.txt', 'LANE-A\n')
        write(b / 'shared.txt', 'LANE-B\n')

        rc, out = tool(a, 'push', '-m', 'lane A wip')
        check('lane A push exit', rc, 0, out)
        rc, out = tool(b, 'push', '-m', 'lane B wip')
        check('lane B push exit', rc, 0, out)

        check('lane A worktree reverted', read(a / 'shared.txt'), 'base\n')
        check('lane B worktree reverted', read(b / 'shared.txt'), 'base\n')

        ref = 'refs/worktree/prosper-stash/default'
        sha_a = rev(a, ref)
        sha_b = rev(b, ref)
        check('lane A parked into the per-worktree namespace', sha_a is not None, True)
        check('lane B parked into the per-worktree namespace', sha_b is not None, True)
        check('one ref NAME, two different values', sha_a != sha_b, True,
              '{} vs {}'.format(sha_a, sha_b))

        check('the shared refs/stash was never written', rev(main, 'refs/stash'), None)

        # Pop in the interleaved order that broke plain git stash: B parked last, A pops first.
        rc, out = tool(a, 'pop')
        check('lane A pop exit', rc, 0, out)
        check('lane A got ITS OWN content back', read(a / 'shared.txt'), 'LANE-A\n')
        check('lane B\'s slot survived lane A\'s pop', rev(b, ref), sha_b)

        rc, out = tool(b, 'pop')
        check('lane B pop exit', rc, 0, out)
        check('lane B got ITS OWN content back', read(b / 'shared.txt'), 'LANE-B\n')

        check('lane A slot is gone after pop', rev(a, ref), None)


def arm_push_refuses_occupied_slot():
    print('[arm] push refuses an occupied slot rather than stacking')
    with tempfile.TemporaryDirectory() as root:
        main, a, _b = make_repo(root)
        write(a / 'shared.txt', 'first\n')
        rc, out = tool(a, 'push', '-m', 'first')
        check('first push exit', rc, 0, out)
        first = git(a, 'rev-parse', 'refs/worktree/prosper-stash/default').strip()
        write(a / 'shared.txt', 'second\n')
        rc, out = tool(a, 'push', '-m', 'second')
        check('second push into the same slot is refused', rc, 2)
        check('refusal names the occupancy', 'already occupied' in out, True, out)
        check('the first snapshot was not displaced',
              git(a, 'rev-parse', 'refs/worktree/prosper-stash/default').strip(), first)
        check('the working tree was left alone by the refusal',
              read(a / 'shared.txt'), 'second\n')
        rc, out = tool(a, 'push', '-m', 'second', '--slot', 'other')
        check('a named second slot is accepted', rc, 0, out)
        check('working tree reverted by the accepted push', read(a / 'shared.txt'), 'base\n')


def arm_push_refuses_untracked():
    print('[arm] push refuses silently-unsnapshotted untracked files')
    with tempfile.TemporaryDirectory() as root:
        main, a, _b = make_repo(root)
        write(a / 'shared.txt', 'LANE-A\n')
        write(a / 'notes.txt', 'untracked\n')
        rc, out = tool(a, 'push')
        check('push refused while an untracked file is present', rc, 2)
        check('refusal names the file', 'notes.txt' in out, True, out)
        rc, _, _ = git_rc(a, 'rev-parse', '--verify', '--quiet',
                          'refs/worktree/prosper-stash/default')
        check('nothing was parked by the refused push', rc, 1)
        check('working tree untouched by the refusal', read(a / 'shared.txt'), 'LANE-A\n')

        rc, out = tool(a, 'push', '--allow-untracked')
        check('--allow-untracked proceeds', rc, 0, out)
        check('tracked change was parked', read(a / 'shared.txt'), 'base\n')
        check('untracked file stayed in the working tree', read(a / 'notes.txt'), 'untracked\n')


def arm_pathspec_park():
    print('[arm] parking named paths leaves the rest of the worktree alone')
    with tempfile.TemporaryDirectory() as root:
        main, a, _b = make_repo(root)
        write(a / 'shared.txt', 'PARK-ME\n')
        write(a / 'other.txt', 'KEEP-ME\n')
        rc, out = tool(a, 'push', '-m', 'A/B one file', 'shared.txt')
        check('pathspec push exit', rc, 0, out)
        check('named path reverted', read(a / 'shared.txt'), 'base\n')
        check('unnamed path untouched', read(a / 'other.txt'), 'KEEP-ME\n')
        rc, out = tool(a, 'pop')
        check('pathspec pop exit', rc, 0, out)
        check('named path restored', read(a / 'shared.txt'), 'PARK-ME\n')
        check('unnamed path still untouched', read(a / 'other.txt'), 'KEEP-ME\n')
        check('pop did not stage the restored path',
              'shared.txt' in git(a, 'diff', '--cached', '--name-only'), False)

        rc, out = tool(a, 'push', 'no/such/file.txt')
        check('a pathspec matching nothing is refused', rc, 2)
        check('refusal says so', 'no tracked file matches' in out, True, out)


def arm_drop_and_missing_slot():
    print('[arm] drop stays recoverable; a missing slot is an error, not a no-op')
    with tempfile.TemporaryDirectory() as root:
        main, a, _b = make_repo(root)
        rc, out = tool(a, 'pop')
        check('pop of a slot that does not exist fails', rc, 2)
        check('and says which slot', 'no such slot' in out, True, out)

        write(a / 'shared.txt', 'LANE-A\n')
        tool(a, 'push', '-m', 'x')
        sha = git(a, 'rev-parse', 'refs/worktree/prosper-stash/default').strip()
        rc, out = tool(a, 'drop')
        check('drop exit', rc, 0, out)
        check('drop printed the sha it detached', sha in out, True, out)
        rc, _, _ = git_rc(a, 'rev-parse', '--verify', '--quiet',
                          'refs/worktree/prosper-stash/default')
        check('slot is gone', rc, 1)
        git(a, 'update-ref', 'refs/worktree/prosper-stash/default', sha)
        rc, out = tool(a, 'pop')
        check('the printed recovery command really recovers it', rc, 0, out)
        check('recovered content', read(a / 'shared.txt'), 'LANE-A\n')


def arm_slots_are_addressed_by_name():
    """A named slot is the property that makes this un-race-able; pin it, do not assume it.

    Added because a mutation that made `pop` fall back to "whatever slot exists" passed the rest of
    this suite: every other arm happens to pop a slot that is present, so a top-of-stack fallback
    is invisible to them. That fallback is `git stash pop`'s exact semantics, which is what #3174
    is about.
    """
    print('[arm] a slot is addressed by NAME -- no top-of-stack fallback')
    with tempfile.TemporaryDirectory() as root:
        main, a, _b = make_repo(root)
        write(a / 'shared.txt', 'ONE\n')
        tool(a, 'push', '-m', 'one', '--slot', 'one')
        write(a / 'shared.txt', 'TWO\n')
        tool(a, 'push', '-m', 'two', '--slot', 'two')
        one = rev(a, 'refs/worktree/prosper-stash/one')
        two = rev(a, 'refs/worktree/prosper-stash/two')
        check('two distinct slots exist', one is not None and two is not None and one != two, True)

        rc, out = tool(a, 'pop', '--slot', 'three')
        check('popping an absent slot fails even though others exist', rc, 2)
        check('and says which slot', 'no such slot' in out, True, out)
        check('slot one untouched', rev(a, 'refs/worktree/prosper-stash/one'), one)
        check('slot two untouched', rev(a, 'refs/worktree/prosper-stash/two'), two)
        check('working tree untouched', read(a / 'shared.txt'), 'base\n')

        rc, out = tool(a, 'apply', '--slot', 'three')
        check('apply of an absent slot fails too', rc, 2)
        rc, out = tool(a, 'drop', '--slot', 'three')
        check('drop of an absent slot fails too', rc, 2)
        check('nothing was dropped', rev(a, 'refs/worktree/prosper-stash/one'), one)

        rc, out = tool(a, 'pop', '--slot', 'one')
        check('popping the NAMED slot gets that slot', rc, 0, out)
        check('and its content', read(a / 'shared.txt'), 'ONE\n')
        check('the other slot survives', rev(a, 'refs/worktree/prosper-stash/two'), two)


def arm_check_detector():
    print('[arm] check classifies the SHARED stack -- and must not fire on your own entry')
    with tempfile.TemporaryDirectory() as root:
        main, a, b = make_repo(root)

        rc, out = tool(a, 'check')
        check('empty shared stack is clean', rc, 0, out)
        check('and says so', 'is empty' in out, True, out)

        # Only lane A's own entry: the discriminator. A check that always fired would pass the
        # positive case below while being useless.
        write(a / 'shared.txt', 'LANE-A\n')
        git(a, 'stash', 'push', '-q', '-m', 'lane A wip')
        rc, out = tool(a, 'check')
        check('own entry alone does not fire', rc, 0, out)
        check('own entry is labelled yours', 'yours' in out, True, out)

        # Lane B pushes on top: from A's seat the stack is now unsafe to pop.
        write(b / 'shared.txt', 'LANE-B\n')
        git(b, 'stash', 'push', '-q', '-m', 'lane B wip')
        rc, out = tool(a, 'check')
        check('a foreign entry fires', rc, 1)
        check('and names the foreign branch', 'fix/issue-3068-lane-b' in out, True, out)
        check('and still labels the local one yours', 'yours' in out, True, out)

        # Symmetric: from B's seat, A's entry is the foreign one.
        rc, out = tool(b, 'check')
        check('symmetric from the other worktree', rc, 1)
        check('names lane A', 'fix/issue-3051-lane-a' in out, True, out)

        # A hand-made entry that carries no branch attribution must fail CLOSED.
        git(main, 'stash', 'clear')
        write(main / 'shared.txt', 'MAIN\n')
        sha = git(main, 'stash', 'create', 'no attribution here').strip()
        git(main, 'checkout', '-q', '--', 'shared.txt')
        git(main, 'stash', 'store', '-m', 'handmade', sha)
        rc, out = tool(a, 'check')
        check('an unattributable entry fires rather than passing', rc, 1)
        check('and is labelled unattributable', 'UNATTRIBUTABLE' in out, True, out)


def arm_hook():
    print('[arm] opt-in reference-transaction hook: refuses stash, spares everything else')
    with tempfile.TemporaryDirectory() as root:
        main, a, _b = make_repo(root)

        # Installed from a LINKED worktree, because that is where agents work and hooks resolve
        # against the common dir rather than the worktree's own gitdir.
        rc, out = tool(a, 'install-hook')
        check('install exit', rc, 0, out)
        hook = Path(main) / '.git' / 'hooks' / 'reference-transaction'
        check('hook landed in the common dir', hook.exists(), True, str(hook))
        check('and is executable', os.access(hook, os.X_OK), True)
        # The installer proves git honours the hook rather than trusting its own write, and says
        # so. Assert the claim is present: an installer that reports success on the strength of a
        # successful `write()` is what shipped a silently-absent guard to Windows in the first
        # place.
        check('install verified itself against git', 'verified: git refused' in out, True, out)
        check('the sentinel it wrote was cleaned up', rev(main, 'refs/stash'), None)
        # LF endings, asserted on the BYTES. Text-mode writing translates '\n' to os.linesep, so
        # on Windows the hook lands with CRLF and the shell running it sees a stray CR inside every
        # token -- present, unrunnable, silent. This arm is the one that catches that directly, and
        # it is checkable from every platform rather than only from the one that breaks.
        raw = hook.read_bytes()
        check('hook is written with LF endings only', b'\r\n' in raw, False)
        check('and ends in an explicit refusal', raw.rstrip().endswith(b'exit 1'), True)

        write(a / 'shared.txt', 'LANE-A\n')
        rc, out, err = git_rc(a, 'stash', 'push', '-m', 'blocked')
        check('git stash is refused', rc != 0, True, out + err)
        check('with a message naming the replacement', 'wt_stash.py push' in (out + err), True,
              out + err)
        check('the working tree is NOT left half-stashed', read(a / 'shared.txt'), 'LANE-A\n')
        rc, _, _ = git_rc(main, 'rev-parse', '--verify', '--quiet', 'refs/stash')
        check('refs/stash was not created', rc, 1)

        # Blast radius: the hook fires on EVERY ref transaction in the repo, so an ordinary commit
        # and branch creation must still work. This is the arm that would catch an over-broad guard.
        write(a / 'other.txt', 'committed\n')
        git(a, 'add', 'other.txt')
        rc, out, err = git_rc(a, 'commit', '-qm', 'ordinary commit')
        check('ordinary commit still works with the hook installed', rc, 0, out + err)
        rc, out, err = git_rc(a, 'branch', 'some-other-branch')
        check('branch creation still works', rc, 0, out + err)

        # And the replacement path must not be blocked by the guard that redirects to it.
        rc, out = tool(a, 'push', '-m', 'still works')
        check('wt_stash push works with the hook installed', rc, 0, out)
        check('parked', read(a / 'shared.txt'), 'base\n')
        rc, out = tool(a, 'pop')
        check('wt_stash pop works with the hook installed', rc, 0, out)

        # Documented one-off bypass.
        rc, out, err = git_rc(a, '-c', 'core.hooksPath=' + os.devnull, 'stash', 'push',
                              '-m', 'bypassed')
        check('the documented bypass works', rc, 0, out + err)
        git(a, '-c', 'core.hooksPath=' + os.devnull, 'stash', 'pop', '-q')

        rc, out = tool(a, 'install-hook')
        check('a second install is idempotent', rc, 0, out)
        check('and says already installed', 'already installed' in out, True, out)

        rc, out = tool(a, 'uninstall-hook')
        check('uninstall exit', rc, 0, out)
        check('hook file removed', hook.exists(), False)
        rc, out, err = git_rc(a, 'stash', 'push', '-q', '-m', 'allowed again')
        check('plain git stash works again after uninstall', rc, 0, out + err)

        # A foreign hook is somebody else's; refuse it in both directions.
        hook.write_text('#!/bin/sh\nexit 0\n')
        os.chmod(hook, 0o755)
        rc, out = tool(a, 'install-hook')
        check('refuses to overwrite a foreign hook', rc, 2)
        rc, out = tool(a, 'uninstall-hook')
        check('refuses to remove a foreign hook', rc, 2)


def arm_list():
    print('[arm] list sees this worktree only, until asked for --all')
    with tempfile.TemporaryDirectory() as root:
        main, a, b = make_repo(root)
        write(a / 'shared.txt', 'LANE-A\n')
        write(b / 'shared.txt', 'LANE-B\n')
        tool(a, 'push', '-m', 'lane A wip')
        tool(b, 'push', '-m', 'lane B wip')
        rc, out = tool(a, 'list')
        check('list exit', rc, 0, out)
        check('one slot listed', out.count('refs/worktree/prosper-stash/default'), 1, out)
        sha_b = rev(b, 'refs/worktree/prosper-stash/default')
        rc, out = tool(a, 'list', '--all')
        check('list --all exit', rc, 0, out)
        # Assert on the sha rather than on the path string: git prints a Windows worktree path with
        # mixed separators (D:\a/_temp/...), so a containment test against pathlib's spelling fails
        # there for a reason that has nothing to do with the tool.
        check('--all reaches the other worktree\'s slot', sha_b[:12] in out, True, out)
        check('--all names the other worktree',
              os.path.basename(str(b)) in out.replace('\\', '/'), True, out)


def main():
    print('wt_stash self-check')
    arm_plain_git_stash_collides()
    if VOID:
        print('')
        print('VOID: the positive control did not reproduce the #3174 collision, so nothing below')
        print('it can be believed. Fix the fixture (or git changed) before reading any other arm.')
        return 1
    arm_wt_stash_isolates()
    arm_push_refuses_occupied_slot()
    arm_push_refuses_untracked()
    arm_pathspec_park()
    arm_drop_and_missing_slot()
    arm_slots_are_addressed_by_name()
    arm_check_detector()
    arm_list()
    arm_hook()
    print('')
    if FAILURES:
        print('{} failure(s):'.format(len(FAILURES)))
        for f in FAILURES:
            print('  - ' + f)
        return 1
    print('all arms passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
