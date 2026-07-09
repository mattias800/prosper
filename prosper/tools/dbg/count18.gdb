# Inspect the swapchain-init object at the [rbx+0x18] count read (eboot+0x2227052).
set pagination off
set confirm off
handle SIG34 nostop pass noprint
handle SIG35 nostop pass noprint
handle SIG36 nostop pass noprint
handle SIG37 nostop pass noprint
handle SIG38 nostop pass noprint
handle SIGSEGV nostop pass noprint
handle SIGILL nostop pass noprint
handle SIGUSR1 nostop pass noprint
handle SIGUSR2 nostop pass noprint
# The guest image is not mapped at gdb start: arm a host-side breakpoint first, then plant the
# guest breakpoint once the image exists (GetOutputStatus runs from the same guest function).
break _ZN7prosperL22g_vo_get_output_statusEmmmmmm
commands
  silent
  break *0x402227052
  commands
    silent
    printf "==== count-read hit: rbx=0x%llx [rbx+0x18]=0x%x ====\n", $rbx, *(unsigned int*)($rbx+0x18)
    printf "---- object dump ----\n"
    x/16gx $rbx
    printf "---- guest RAs ----\n"
    set $p = $sp
    set $e = $sp + 1024
    while $p < $e
      set $v = *(unsigned long long*)$p
      if ($v >= 0x400000000 && $v < 0x40a000000)
        printf "sp+0x%04x: eboot+0x%llx\n", (unsigned int)($p - $sp), $v - 0x400000000
      end
      set $p = $p + 8
    end
    printf "==== HIT DONE ====\n"
    if (*(unsigned int*)($rbx+0x18)) > 0x10000
      printf "==== GARBAGE COUNT — stopping ====\n"
      kill
      quit
    end
    continue
  end
  disable 1
  continue
end
run
quit
