#!/usr/bin/env python3
"""Park uncommitted changes in a ref that is REALLY private to one worktree.

`git stash` is not worktree-isolated. Every `git worktree` directory shares one `refs/stash` in
the common `.git` directory, so two agents each following CLAUDE.md's "work in your OWN git
worktree" rule still push onto the same LIFO stack. On 2026-09-01 two lanes' pushes interleaved and
each `git stash pop` applied the OTHER lane's diff into its worktree while dropping that lane's
entry from the stack (#3174, instrument trap 247). Nothing was lost only because both agents
happened to recognise foreign content in their own tree.

git's own per-worktree ref namespaces are the fix. `git-worktree(1)` REFS: "refs inside
refs/bisect, refs/worktree and refs/rewritten are not shared". So this tool snapshots with
`git stash create` -- which writes NO ref and touches no working tree -- and stores the result under
`refs/worktree/prosper-stash/<slot>`. Two worktrees can hold that same ref name at two different
values simultaneously; neither can see, pop, or drop the other's.

Three properties make it un-race-able where `git stash` is not:

  * the ref is per-worktree, so there is no shared stack to interleave on;
  * a slot is addressed BY NAME, never by "top of stack", so no pop can pick up a neighbour's entry;
  * `push` REFUSES an occupied slot instead of stacking, so nothing is ever displaced silently.

usage:
  python3 wt_stash.py push [-m MSG] [--slot NAME] [--allow-untracked] [PATH ...]
                                    park changes (all tracked paths, or only PATH ...) and revert
                                    them in the working tree
  python3 wt_stash.py pop   [--slot NAME]   restore the slot into the working tree and delete it
  python3 wt_stash.py apply [--slot NAME]   restore the slot and KEEP it
  python3 wt_stash.py list  [--all]         this worktree's slots (--all: every worktree's)
  python3 wt_stash.py drop  [--slot NAME]   delete a slot, printing the sha so it stays recoverable
  python3 wt_stash.py check                 inspect the SHARED refs/stash for other lanes' entries;
                                            exit 1 if the stack holds anything that is not yours
  python3 wt_stash.py install-hook          opt-in: refuse `git stash` repo-wide (see below)
  python3 wt_stash.py uninstall-hook

install-hook writes a `reference-transaction` hook that aborts any update to `refs/stash` and names
this tool instead. It is OFF by default and it is NOT a safety net you may assume: it lives in
`.git/hooks`, which is untracked, so a fresh clone or a colleague's checkout has no hook at all and
fails open. It is also repo-wide -- once installed, `git stash` stops working for every worktree AND
for the human's own interactive use, including `--autostash`. Bypass a single command with
`git -c core.hooksPath=/dev/null stash ...`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REF_PREFIX = 'refs/worktree/prosper-stash'
MSG_TAG = 'wt_stash'
HOOK_MARKER = '# prosper wt_stash refs/stash guard (tools/wt_stash.py install-hook)'

HOOK_BODY = '''#!/bin/sh
{marker}
#
# Refuses any update to refs/stash. refs/stash is shared by every worktree in this repository, so
# two concurrent agents stashing in their own worktrees collide on one LIFO stack and pop each
# other's work (#3174). Park changes with `python3 prosper/tools/wt_stash.py push` instead, which
# stores them under the per-worktree refs/worktree/ namespace.
#
# Fails OPEN on anything unexpected: only a positively identified refs/stash update in the
# `prepared` phase is refused. Bypass once with `git -c core.hooksPath=/dev/null stash ...`,
# or remove permanently with `python3 prosper/tools/wt_stash.py uninstall-hook`.

state="$1"
hit=""
# Always drain stdin before deciding, so git never writes into a closed pipe.
while read -r _old _new ref; do
    if [ "$ref" = "refs/stash" ]; then
        hit="yes"
    fi
done

[ "$state" = "prepared" ] || exit 0
[ -n "$hit" ] || exit 0

echo "wt_stash guard: refusing to write refs/stash." >&2
echo "  refs/stash is shared by EVERY worktree of this repo, so a concurrent lane's" >&2
echo "  stash pop can take your entry and give you theirs (#3174)." >&2
echo "  Park changes per-worktree instead:" >&2
echo "    python3 prosper/tools/wt_stash.py push -m 'why' [PATH ...]" >&2
echo "    python3 prosper/tools/wt_stash.py pop" >&2
echo "  One-off bypass: git -c core.hooksPath=/dev/null stash ..." >&2
exit 1
'''.format(marker=HOOK_MARKER)


class Fail(Exception):
    """A refusal the caller should see as a message, not a traceback."""


def git(*args, check=True, cwd=None):
    """Run git and return stdout; raise Fail on a non-zero exit when check is set."""
    p = subprocess.run(['git', *args], capture_output=True, text=True, cwd=cwd)
    if check and p.returncode != 0:
        raise Fail('git {}: {}'.format(' '.join(args), (p.stderr or p.stdout).strip()))
    return p.stdout


def git_rc(*args, cwd=None):
    p = subprocess.run(['git', *args], capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stdout, p.stderr


def slot_ref(slot):
    if not slot or slot.startswith('-') or '..' in slot or slot.endswith('/'):
        raise Fail('bad slot name: {!r}'.format(slot))
    ref = '{}/{}'.format(REF_PREFIX, slot)
    rc, _, err = git_rc('check-ref-format', ref)
    if rc != 0:
        raise Fail('bad slot name: {!r} ({})'.format(slot, err.strip()))
    return ref


def current_branch():
    """The branch this worktree has checked out, or None when detached."""
    rc, out, _ = git_rc('symbolic-ref', '--quiet', '--short', 'HEAD')
    return out.strip() if rc == 0 and out.strip() else None


def resolve(ref):
    """The commit a slot ref names, or None.

    Deliberately NOT `<ref>^{commit}`: that peeling syntax reaches git mangled on the Windows/MSYS
    route this project also prescribes, where it resolves nothing and every slot reads as absent --
    push parked fine and pop then said "no such slot". Everything written here is a commit sha this
    tool put there itself, so a plain --verify is both sufficient and portable.
    """
    rc, out, _ = git_rc('rev-parse', '--verify', '--quiet', ref)
    return out.strip() if rc == 0 else None


def encode_message(slot, note, paths):
    """One snapshot commit message that pop() can read back: slot, note, and the parked paths.

    Stored in the commit rather than beside the ref because a ref carries no metadata, and a
    sidecar file would be one more thing that can go missing between push and pop.
    """
    lines = ['{} slot={}'.format(MSG_TAG, slot)]
    if note:
        lines.append('note: ' + note.replace('\n', ' '))
    lines.append('')
    lines.append('paths:')
    lines.extend(paths)
    return '\n'.join(lines)


def decode_paths(sha):
    """The paths recorded by push(), or [] for a whole-worktree snapshot."""
    body = git('log', '-1', '--format=%B', sha)
    lines = body.splitlines()
    try:
        at = lines.index('paths:')
    except ValueError:
        return []
    return [ln for ln in lines[at + 1:] if ln.strip()]


def describe(sha):
    subject = git('log', '-1', '--format=%s', sha).strip()
    when = git('log', '-1', '--format=%ci', sha).strip()
    return '{}  {}  {}'.format(sha[:12], when, subject)


# --- subcommands ---------------------------------------------------------------------------


def cmd_push(args):
    ref = slot_ref(args.slot)
    existing = resolve(ref)
    if existing:
        # Refusing beats stacking. A stack is what makes refs/stash losable, and an overwrite here
        # would reproduce the very failure this tool exists to remove.
        raise Fail('slot {!r} is already occupied by {}\n'
                   '  pop or drop it first, or use --slot with another name'
                   .format(args.slot, describe(existing)))

    untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard').splitlines() if ln]
    if untracked and not args.allow_untracked:
        # git stash create cannot snapshot untracked content at all. Silently leaving it out of a
        # "park everything" operation is the data-shaped failure this repository keeps recording.
        raise Fail('{} untracked file(s) present; `git stash create` cannot snapshot them:\n'
                   '  {}\n'
                   '  they would stay in the working tree, NOT in the slot. Re-run with '
                   '--allow-untracked once you have read that list.'
                   .format(len(untracked), '\n  '.join(untracked[:20])))

    paths = list(args.paths)
    if paths:
        known = git('ls-files', '--', *paths).splitlines()
        if not known:
            raise Fail('no tracked file matches: {}'.format(' '.join(paths)))

    message = encode_message(args.slot, args.message, paths)
    sha = git('stash', 'create', message).strip()
    if not sha:
        print('nothing to park: no tracked changes in this worktree')
        return 0

    # Only now is anything reverted, and only after the snapshot commit exists and is named by a
    # ref -- so an interrupted push loses nothing.
    git('update-ref', '-m', 'wt_stash push', ref, sha)

    if paths:
        git('checkout', 'HEAD', '--', *paths)
    else:
        added = [ln for ln in git('diff', '--cached', '--name-only', '--diff-filter=A').splitlines()
                 if ln]
        git('reset', '--hard', '--quiet', 'HEAD')
        # `git reset --hard` un-stages a newly added file but leaves it in the working tree as
        # untracked, where `git stash push` removes it. The snapshot holds it either way.
        for f in added:
            try:
                os.remove(f)
            except OSError:
                pass

    print('parked {} at {}'.format(sha[:12], ref))
    print('  restore with: python3 prosper/tools/wt_stash.py pop --slot {}'.format(args.slot))
    return 0


def _restore(sha):
    paths = decode_paths(sha)
    if paths:
        git('restore', '--source=' + sha, '--', *paths)
        return paths
    rc, out, err = git_rc('stash', 'apply', sha)
    if rc != 0:
        raise Fail('could not apply {}: {}\n'
                   '  the slot is intact -- resolve the conflict and retry'
                   .format(sha[:12], (err or out).strip()))
    return []


def cmd_apply(args):
    ref = slot_ref(args.slot)
    sha = resolve(ref)
    if not sha:
        raise Fail('no such slot: {!r}'.format(args.slot))
    paths = _restore(sha)
    print('restored {} from {}{}'.format(sha[:12], ref,
                                         ' ({} path(s))'.format(len(paths)) if paths else ''))
    return 0


def cmd_pop(args):
    ref = slot_ref(args.slot)
    sha = resolve(ref)
    if not sha:
        raise Fail('no such slot: {!r}'.format(args.slot))
    paths = _restore(sha)
    git('update-ref', '-d', ref, sha)
    print('restored {} from {}{} and dropped the slot'
          .format(sha[:12], ref, ' ({} path(s))'.format(len(paths)) if paths else ''))
    print('  the snapshot commit {} stays reachable until git gc'.format(sha))
    return 0


def cmd_drop(args):
    ref = slot_ref(args.slot)
    sha = resolve(ref)
    if not sha:
        raise Fail('no such slot: {!r}'.format(args.slot))
    git('update-ref', '-d', ref, sha)
    print('dropped {} -> {}'.format(ref, sha))
    print('  recover with: git update-ref {} {}'.format(ref, sha))
    return 0


def _slots_in(cwd):
    out = git('for-each-ref', '--format=%(refname) %(objectname)', REF_PREFIX + '/**', cwd=cwd)
    return [ln.split(' ', 1) for ln in out.splitlines() if ln]


def cmd_list(args):
    here = _slots_in(None)
    if here:
        for ref, sha in here:
            print('{}  {}'.format(ref, describe(sha)))
    else:
        print('(no slots in this worktree)')
    if not args.all:
        return 0
    print('')
    print('other worktrees:')
    mine = os.path.realpath(git('rev-parse', '--show-toplevel').strip())
    found = False
    for block in git('worktree', 'list', '--porcelain').split('\n\n'):
        first = block.strip().splitlines()
        if not first or not first[0].startswith('worktree '):
            continue
        path = first[0][len('worktree '):]
        if os.path.realpath(path) == mine or not os.path.isdir(path):
            continue
        try:
            slots = _slots_in(path)
        except Fail:
            continue
        for ref, sha in slots:
            found = True
            print('  {}  {}  {}'.format(path, ref, describe(sha)))
    if not found:
        print('  (none)')
    return 0


def stash_entry_branch(subject):
    """The branch a shared-stack entry was created on, or None when it cannot be established.

    `git stash` writes "WIP on <branch>: <sha> <subject>" or "On <branch>: <message>". Anything
    else -- a hand-made `git stash store`, a detached HEAD -- is unattributable, and this returns
    None so the caller can fail closed. "I could not tell whose this is" must never render as
    "it is yours".
    """
    for prefix in ('WIP on ', 'On '):
        if subject.startswith(prefix):
            rest = subject[len(prefix):]
            if ': ' in rest:
                name = rest.split(': ', 1)[0]
                return name if name and name != '(no branch)' else None
    return None


def cmd_check(args):
    rc, out, _ = git_rc('stash', 'list', '--format=%gd|%gs')
    entries = [ln for ln in out.splitlines() if ln]
    if rc != 0:
        raise Fail('could not read the shared stash stack')
    if not entries:
        print('shared refs/stash is empty -- nothing to collide with')
        return 0
    branch = current_branch()
    print('shared refs/stash holds {} entry/entries (visible from EVERY worktree):'
          .format(len(entries)))
    foreign = []
    for line in entries:
        sel, _, subject = line.partition('|')
        owner = stash_entry_branch(subject)
        if owner is None:
            verdict = 'UNATTRIBUTABLE'
        elif branch is not None and owner == branch:
            verdict = 'yours'
        else:
            verdict = 'NOT YOURS'
        if verdict != 'yours':
            foreign.append((sel, owner, subject))
        print('  {:<12} {:<15} on {:<28} {}'
              .format(sel, verdict, owner or '(unknown)', subject))
    if not foreign:
        print('every entry was created on this worktree\'s branch ({})'.format(branch))
        return 0
    print('')
    print('{} entry/entries on the shared stack were not created here.'.format(len(foreign)))
    print('Do NOT `git stash pop`: it takes the TOP of the stack, which may be another lane\'s')
    print('work, and drops their entry in the process (#3174).')
    print('Park your own changes with: python3 prosper/tools/wt_stash.py push')
    return 1


def _hook_path():
    """Where git looks for this repository's reference-transaction hook, as PYTHON spells it.

    Asking git for an absolute path is the obvious way and it is wrong wherever git and the
    interpreter disagree about path spelling -- an MSYS git under native Windows python answers
    `/tmp/...`, which python then resolves against the current drive. The install lands somewhere
    real, git never looks there, and the guard is silently absent. Measured on this project's
    Windows MinGW CI, where install-hook returned 0 and `git stash` went on working. A RELATIVE
    answer has no root spelling to disagree about and survives the crossing; the absolute form
    stays as a fallback, and a candidate that is not a directory here is a refusal rather than a
    guess -- a hook installer that cannot prove where it wrote is worse than none.
    """
    configured = git('config', '--default', '', '--get', 'core.hooksPath').strip()
    if configured:
        raise Fail('core.hooksPath is set to {!r}; this installer only manages the default '
                   'hooks directory. Install the hook there by hand, or unset the config.'
                   .format(configured))
    tried = []
    for fmt in ('relative', 'absolute'):
        rc, out, _ = git_rc('rev-parse', '--path-format=' + fmt, '--git-common-dir')
        if rc != 0 or not out.strip():
            continue
        cand = os.path.abspath(os.path.join(os.getcwd(), out.strip()))
        tried.append(cand)
        if os.path.isdir(cand):
            return os.path.join(cand, 'hooks', 'reference-transaction')
    raise Fail('could not locate this repository\'s common git directory as this interpreter '
               'spells it; git offered {} and none of it exists here. Install the hook by hand.'
               .format(', '.join(repr(t) for t in tried) or '(nothing)'))


def _hook_is_live():
    """Did git actually run the hook we just wrote? True / False / None when it cannot be asked.

    An installer that reports success on the strength of its own `write()` is the shape this
    project keeps a list about: it reads as coverage and cannot fail. So prove it instead, by
    asking git to do the one thing the hook exists to refuse -- a sentinel write to `refs/stash`.

    Skipped, rather than forced, when `refs/stash` already holds something: that entry is somebody's
    work, and racing it to test a guard against losing work would be its own answer. The cleanup
    passes the sentinel as the expected old value, so it can only ever delete the ref this function
    created.
    """
    if resolve('refs/stash') is not None:
        return None
    rc, head, _ = git_rc('rev-parse', '--verify', '--quiet', 'HEAD')
    sentinel = head.strip()
    if rc != 0 or not sentinel:
        return None
    rc, _, _ = git_rc('update-ref', 'refs/stash', sentinel)
    if rc != 0:
        return True
    git_rc('update-ref', '-d', 'refs/stash', sentinel)
    return False


def cmd_install_hook(args):
    path = _hook_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:
            existing = fh.read()
        if HOOK_MARKER in existing:
            print('already installed: {}'.format(path))
            return 0
        if not args.force:
            raise Fail('a different reference-transaction hook already exists at {}\n'
                       '  refusing to overwrite it; merge by hand or pass --force'.format(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline='' -- text mode otherwise translates every '\n' to os.linesep, so on Windows the
    # hook lands with CRLF endings and the shell running it sees a stray CR inside every token
    # ('then\r' is not 'then'). The guard would then be present, unrunnable, and silent.
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(HOOK_BODY)
    os.chmod(path, 0o755)

    live = _hook_is_live()
    if live is False:
        os.remove(path)
        raise Fail('wrote {} but git did not honour it -- a sentinel refs/stash write went '
                   'through anyway. The hook has been removed again rather than left there '
                   'reading as protection. Park changes with `wt_stash.py push` instead.'
                   .format(path))
    print('installed {}'.format(path))
    if live is None:
        print('  NOT verified: refs/stash is already occupied (or HEAD is unborn), so the')
        print('  sentinel check was skipped rather than raced against somebody else\'s entry.')
    else:
        print('  verified: git refused a sentinel refs/stash write through this hook.')
    print('  `git stash` now fails repo-wide, for every worktree AND for interactive use.')
    print('  Bypass once: git -c core.hooksPath=/dev/null stash ...')
    return 0


def cmd_uninstall_hook(args):
    path = _hook_path()
    if not os.path.exists(path):
        print('not installed: {}'.format(path))
        return 0
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:
        existing = fh.read()
    if HOOK_MARKER not in existing:
        raise Fail('{} was not written by wt_stash; refusing to remove it'.format(path))
    os.remove(path)
    print('removed {}'.format(path))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog='wt_stash.py',
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest='cmd')

    p = sub.add_parser('push', help='park changes in this worktree only')
    p.add_argument('paths', nargs='*', metavar='PATH')
    p.add_argument('-m', '--message', default='')
    p.add_argument('--slot', default='default')
    p.add_argument('--allow-untracked', action='store_true')
    p.set_defaults(func=cmd_push)

    p = sub.add_parser('pop', help='restore a slot and delete it')
    p.add_argument('--slot', default='default')
    p.set_defaults(func=cmd_pop)

    p = sub.add_parser('apply', help='restore a slot and keep it')
    p.add_argument('--slot', default='default')
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser('drop', help='delete a slot')
    p.add_argument('--slot', default='default')
    p.set_defaults(func=cmd_drop)

    p = sub.add_parser('list', help='show slots')
    p.add_argument('--all', action='store_true', help='include every other worktree')
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('check', help='is the SHARED refs/stash holding someone else\'s work?')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('install-hook', help='opt-in: make `git stash` fail repo-wide')
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_install_hook)

    p = sub.add_parser('uninstall-hook', help='remove the opt-in hook')
    p.set_defaults(func=cmd_uninstall_hook)
    return ap


def main(argv):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, 'func', None):
        print(__doc__)
        return 2
    try:
        return args.func(args)
    except Fail as e:
        print('wt_stash: {}'.format(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
