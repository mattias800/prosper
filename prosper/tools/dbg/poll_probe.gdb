# Probe the issue-#232 GameThread poll wall: break at eboot+0x2cc236a (call *0x20(%rax)),
# dump the polled object + vtable + poll fn, then static context around the loop.
set pagination off
set confirm off
handle SIG34 nostop pass noprint
handle SIG35 nostop pass noprint
handle SIG36 nostop pass noprint
handle SIG37 nostop pass noprint
handle SIG38 nostop pass noprint
handle SIGUSR1 nostop pass noprint
handle SIGUSR2 nostop pass noprint
handle SIGSEGV nostop pass noprint
set $hits = 0
break *0x402cc236a
commands
  silent
  set $hits = $hits + 1
  printf "==== HIT %d ====\n", $hits
  printf "rax=%llx rdi=%llx rsi=%llx rdx=%llx rbx=%llx r12=%llx r13=%llx r14=%llx r15=%llx\n", $rax, $rdi, $rsi, $rdx, $rbx, $r12, $r13, $r14, $r15
  printf "-- [rdi] (candidate object) --\n"
  x/16gx $rdi
  printf "-- [rax] (candidate vtable) --\n"
  x/8gx $rax
  set $fn = *(unsigned long long*)($rax+0x20)
  printf "pollfn=%llx (eboot+0x%llx)\n", $fn, $fn-0x400000000
  if $hits >= 3
    printf "-- disas pollfn --\n"
    x/50i $fn
    printf "-- disas poll caller 0x402cc2340 --\n"
    x/70i 0x402cc2340
    printf "-- disas time-compare fn 0x402324150..  --\n"
    x/60i 0x402324130
    printf "-- backtrace --\n"
    bt 12
    printf "-- stack raw --\n"
    x/48gx $rsp
    printf "==== PROBE DONE ====\n"
    detach
    quit
  end
  continue
end
continue
quit
