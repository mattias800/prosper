#!/usr/bin/env python3
# guest_bt.py — driver for the prosper guest-stack unwinder.
#
# One command turns a live/frozen prosper process into symbolicated GUEST backtraces (managed IL2CPP
# method names included), stepping through prosper's synthesized HLE import stubs that defeat gdb's
# native unwinder. See tools/guest_bt/README.md for the how/why.
#
# Typical use (attach to a running title and dump every guest thread):
#   python3 tools/guest_bt/guest_bt.py --pid $(pgrep -f PPSA25872 | head -1) \
#       --initlog run.log --dump /path/PPSA25872-app0 --all
#
# --initlog is any file containing prosper's `[initlog] module N base=0x.. path=..` lines (run the
# title once with PROSPER_INITLOG=1). The driver flattens+sections each module (cached under --cache)
# and, for the IL2CPP module, reuses an Il2CppDumper script.json for managed symbol names if one is
# found next to the dump or passed via --il2cpp-script.
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRX2ELF = os.path.join(HERE, '..', 'il2cpp', 'prx_to_elf.py')
MKSYM = os.path.join(HERE, 'mk_sym_elf.py')
PLUGIN = os.path.join(HERE, 'guest_bt_gdb.py')
IL2CPP_LO, IL2CPP_HI = 0x440000000, 0x4c0000000   # the il2cpp guest-address window (hle_file.cpp)

def sh(cmd):
    subprocess.run(cmd, check=True)

def parse_initlog(path):
    mods = []
    rx = re.compile(r'\[initlog\] module \d+ base=0x([0-9a-f]+) .*path=(\S+)')
    for ln in open(path, errors='replace'):
        m = rx.search(ln)
        if m:
            mods.append((int(m.group(1), 16), m.group(2)))
    if not mods:
        sys.exit('guest_bt: no [initlog] lines in %s (run the title with PROSPER_INITLOG=1)' % path)
    return mods

def run_gdb(gdb, pid, plugin, command, env):
    """Run the debugger and preserve its status for callers such as hang_probe."""
    return subprocess.run([gdb, '-p', str(pid), '-batch',
                           '-ex', 'set debuginfod enabled off', '-ex', 'set pagination off',
                           '-ex', 'source %s' % plugin, '-ex', command], env=env).returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--initlog', required=True, help='file with prosper [initlog] module lines')
    ap.add_argument('--dump', help='<TITLE>-app0 dir (to locate global-metadata.dat for symbol names)')
    ap.add_argument('--il2cpp-script', help='Il2CppDumper script.json (managed method names)')
    ap.add_argument('--cache', default=os.path.join(HERE, '.cache'))
    ap.add_argument('--stub-base', default='0x600000000')
    ap.add_argument('--stub-size', type=int, default=96)
    ap.add_argument('--thread', help='thread id or name (e.g. GameUpdate); default = all')
    ap.add_argument('--all', action='store_true', help='backtrace every guest thread')
    ap.add_argument('--gdb', default='gdb')
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    mods = parse_initlog(args.initlog)

    # locate an il2cpp script.json for managed symbolication (optional but high-value)
    script = args.il2cpp_script
    if not script and args.dump:
        for cand in (os.path.join(args.cache, 'il2out', 'script.json'),):
            if os.path.exists(cand):
                script = cand
                break

    cfg_mods = []
    for base, path in mods:
        if not os.path.exists(path):
            print('guest_bt: WARNING module missing on disk, skipping: %s' % path)
            continue
        tag = re.sub(r'\W+', '_', os.path.basename(path)) + ('_%x' % base)
        flat = os.path.join(args.cache, tag + '.flat.elf')
        sym = os.path.join(args.cache, tag + '.sym.elf')
        if not os.path.exists(flat):
            sh([sys.executable, PRX2ELF, path, flat])
        if not os.path.exists(sym):
            cmd = [sys.executable, MKSYM, flat, sym]
            if script and IL2CPP_LO <= base < IL2CPP_HI:
                cmd += ['--symbols', script]
            sh(cmd)
        cfg_mods.append({'elf': sym, 'base': '0x%x' % base})

    cfg = {'modules': cfg_mods, 'stub_base': args.stub_base,
           'stub_size': args.stub_size, 'stub_count': 65536}
    cfg_path = os.path.join(args.cache, 'gbt_config.json')
    json.dump(cfg, open(cfg_path, 'w'), indent=2)

    cmd = args.thread or ''
    gdb_cmd = 'guest-bt-all' if (args.all or not cmd) else ('guest-bt %s' % cmd)
    env = dict(os.environ, PROSPER_GBT_CONFIG=cfg_path)
    return run_gdb(args.gdb, args.pid, PLUGIN, gdb_cmd, env)

if __name__ == '__main__':
    sys.exit(main())
