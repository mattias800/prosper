# RE the DOLL per-draw AGC NIDs: break in glog_impl when nid==fPSCdQxgpSw,
# dump guest RA chain at an early hit and a late (render-burst) hit.
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
break prosper::(anonymous namespace)::glog_impl(char const*, void*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long) if ((char*)$rdi)[0]=='f' && ((char*)$rdi)[1]=='P' && ((char*)$rdi)[2]=='S'
commands
  silent
  set $cnt = $cnt + 1
  if $cnt == 5 || $cnt == 2900
    printf "==== fPSCdQxgpSw hit #%d ====\n", $cnt
    printf "args a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx\n", $rdx, $rcx, $r8, $r9
    printf "---- guest RAs on stack ----\n"
    set $p = $sp
    set $e = $sp + 3072
    while $p < $e
      set $v = *(unsigned long long*)$p
      if ($v >= 0x400000000 && $v < 0x40a000000)
        printf "sp+0x%04x: eboot+0x%llx\n", (unsigned int)($p - $sp), $v - 0x400000000
      end
      set $p = $p + 8
    end
    printf "---- a1 base buffer ----\n"
    x/32wx $rcx
    printf "---- a0 buffer ----\n"
    x/16wx $rdx
    printf "==== HIT DONE ====\n"
  end
  if $cnt == 2900
    kill
    quit
  end
  continue
end
run
quit
