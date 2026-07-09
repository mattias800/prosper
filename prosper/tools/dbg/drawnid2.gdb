# Capture guest RA chain for k3GhuSNmBLU (per-draw emitter suspect).
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
set $cnt = 0
break prosper::(anonymous namespace)::glog_impl(char const*, void*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long) if ((char*)$rdi)[0]=='k' && ((char*)$rdi)[1]=='3' && ((char*)$rdi)[2]=='G'
commands
  silent
  set $cnt = $cnt + 1
  if $cnt == 3 || $cnt == 40
    printf "==== k3GhuSNmBLU hit #%d ====\n", $cnt
    printf "args a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx\n", $rdx, $rcx, $r8, $r9
    printf "---- guest RAs on stack ----\n"
    set $p = $sp
    set $e = $sp + 2048
    while $p < $e
      set $v = *(unsigned long long*)$p
      if ($v >= 0x400000000 && $v < 0x40a000000)
        printf "sp+0x%04x: eboot+0x%llx\n", (unsigned int)($p - $sp), $v - 0x400000000
      end
      set $p = $p + 8
    end
    printf "---- a0 object ----\n"
    x/32gx $rdx
    printf "==== HIT DONE ====\n"
  end
  if $cnt == 40
    kill
    quit
  end
  continue
end
run
quit
