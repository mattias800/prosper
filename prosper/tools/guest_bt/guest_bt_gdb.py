# guest_bt_gdb.py — gdb plugin: symbolicated backtraces of prosper GUEST threads.
#
# The problem it solves: a guest thread blocked in an HLE (e.g. sceKernelWaitOnAddress) has, on its
# stack, guest frames that gdb refuses to unwind — Unity/IL2CPP is built without frame pointers, and
# prosper's synthesized import STUB (BOOT_STUB=0x600000000, no `.eh_frame`) sits between the C++ HLE
# and the guest caller, so gdb's unwind dies at the stub with a bogus saved-rip=0.
#
# This plugin fixes both halves:
#   * loads each guest module's flattened+sectioned ELF (see mk_sym_elf.py) at its runtime base, so
#     gdb's DWARF unwinder has the modules' real CFI + a `.symtab` (managed method names for free), and
#   * registers a custom Unwinder for the stub region that steps THROUGH a stub frame to the guest
#     caller using the fixed swap-stub layout (5 pushes of 0x28 + the `call r10`):
#         caller_sp  = stub_frame_sp + 0x30
#         caller_pc  = *(stub_frame_sp + 0x28)          # the guest return address
#     with all callee-saved registers passed through unchanged (the stub clobbers only rax/r10/r11).
#
# Config comes from the JSON file named by env PROSPER_GBT_CONFIG:
#   { "modules": [ {"elf": "...", "base": "0x410000000"}, ... ],
#     "stub_base": "0x600000000", "stub_size": 96, "stub_ret_off": 61 }
# stub_ret_off is informational; the unwind rule keys only on PC being inside the stub region.
#
# Command:  guest-bt <thread-id-or-name>     e.g.  guest-bt GameUpdate
import gdb, json, os, struct
import gdb.unwinder

_CFG = {}
_STUB_LO = _STUB_HI = 0

def _load_config():
    global _CFG, _STUB_LO, _STUB_HI
    path = os.environ.get('PROSPER_GBT_CONFIG')
    if not path:
        raise gdb.GdbError('PROSPER_GBT_CONFIG not set')
    _CFG = json.load(open(path))
    sb = int(_CFG['stub_base'], 0)
    ss = int(_CFG.get('stub_size', 96))
    n = int(_CFG.get('stub_count', 65536))
    _STUB_LO, _STUB_HI = sb, sb + ss * n
    for m in _CFG['modules']:
        base = int(m['base'], 0)
        try:
            gdb.execute('add-symbol-file %s 0x%x' % (m['elf'], base), to_string=True)
        except gdb.error as e:
            print('guest-bt: add-symbol-file %s failed: %s' % (m['elf'], e))

class _StubUnwinder(gdb.unwinder.Unwinder):
    def __init__(self):
        super().__init__('prosper-hle-stub')

    def __call__(self, pending):
        pc = int(pending.read_register('pc').cast(gdb.lookup_type('unsigned long')))
        if not (_STUB_LO <= pc < _STUB_HI):
            return None
        sp = int(pending.read_register('sp').cast(gdb.lookup_type('unsigned long')))
        caller_sp = sp + 0x30
        try:
            raw = gdb.selected_inferior().read_memory(sp + 0x28, 8)
        except gdb.MemoryError:
            return None
        caller_pc = struct.unpack('<Q', raw)[0]
        if caller_pc == 0:
            return None
        # FrameId(sp, pc): identify THIS (stub) frame by its CFA and pc.
        fid = gdb.unwinder.FrameId(caller_sp, pc)
        info = pending.create_unwind_info(fid)
        ulong = gdb.lookup_type('unsigned long')
        info.add_saved_register('pc', gdb.Value(caller_pc).cast(ulong))
        info.add_saved_register('sp', gdb.Value(caller_sp).cast(ulong))
        # Callee-saved survive the stub unchanged -> pass the inner-frame values straight through so
        # the guest module's CFI can keep unwinding from a correct register file.
        for r in ('rbp', 'rbx', 'r12', 'r13', 'r14', 'r15'):
            try:
                info.add_saved_register(r, pending.read_register(r))
            except gdb.error:
                pass
        return info

def _pick_thread(arg):
    arg = (arg or '').strip()
    inf = gdb.selected_inferior()
    if not arg:
        return gdb.selected_thread()
    for t in inf.threads():
        if str(t.num) == arg:
            return t
        name = t.name or ''
        if name == arg or (t.ptid[1] and str(t.ptid[1]) == arg):
            return t
    # substring match on name (thread names are truncated by the kernel to 15 chars)
    for t in inf.threads():
        if arg[:15] in (t.name or ''):
            return t
    raise gdb.GdbError('no thread matching %r' % arg)

class GuestBt(gdb.Command):
    """guest-bt [thread] — symbolicated backtrace of a prosper guest thread across the HLE stub."""
    def __init__(self):
        super().__init__('guest-bt', gdb.COMMAND_STACK)

    def invoke(self, arg, from_tty):
        t = _pick_thread(arg)
        t.switch()
        print('=== guest-bt thread %s "%s" (LWP %s) ===' %
              (t.num, t.name or '?', t.ptid[1]))
        gdb.execute('bt')

class GuestBtAll(gdb.Command):
    """guest-bt-all — symbolicated backtrace of EVERY prosper guest thread (skips pure-host idlers)."""
    def __init__(self):
        super().__init__('guest-bt-all', gdb.COMMAND_STACK)

    def invoke(self, arg, from_tty):
        for t in gdb.selected_inferior().threads():
            t.switch()
            print('\n=== thread %s "%s" (LWP %s) ===' % (t.num, t.name or '?', t.ptid[1]))
            try:
                gdb.execute('bt')
            except gdb.error as e:
                print('  <bt failed: %s>' % e)

_load_config()
gdb.unwinder.register_unwinder(None, _StubUnwinder(), replace=True)
GuestBt()
GuestBtAll()
print('guest-bt: loaded (%d modules, stub [0x%x,0x%x))' %
      (len(_CFG.get('modules', [])), _STUB_LO, _STUB_HI))
