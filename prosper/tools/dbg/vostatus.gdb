# Capture the guest caller of sceVideoOutGetOutputStatus and its out-struct usage.
set pagination off
set confirm off
handle SIG34 nostop pass noprint
handle SIG35 nostop pass noprint
handle SIG36 nostop pass noprint
handle SIG37 nostop pass noprint
handle SIG38 nostop pass noprint
handle SIGSEGV nostop pass noprint
handle SIGUSR1 nostop pass noprint
handle SIGUSR2 nostop pass noprint
break _ZN7prosperL22g_vo_get_output_statusEmmmmmm
commands
  silent
  printf "==== GetOutputStatus a0=0x%llx a1=0x%llx ====\n", $rdi, $rsi
  printf "---- guest RAs on stack ----\n"
  set $p = $sp
  set $e = $sp + 1024
  while $p < $e
    set $v = *(unsigned long long*)$p
    if ($v >= 0x400000000 && $v < 0x40a000000)
      printf "sp+0x%04x: eboot+0x%llx\n", (unsigned int)($p - $sp), $v - 0x400000000
    end
    set $p = $p + 8
  end
  printf "---- out struct BEFORE (stack garbage) ----\n"
  x/16gx $rsi
  printf "==== DONE ====\n"
  kill
  quit
end
run
quit
