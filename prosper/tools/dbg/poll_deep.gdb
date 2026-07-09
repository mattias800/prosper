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
break *0x402cc236a
commands
  silent
  set $obj = $rdi
  printf "==== POLL obj=%llx ====\n", $obj
  printf "-- obj as UTF-16/bytes (name?) --\n"
  x/64bx $obj
  set $o10 = *(unsigned long long*)($obj+0x10)
  printf "obj+0x10 -> %llx\n", $o10
  x/16gx $o10
  set $o10vt = *(unsigned long long*)$o10
  printf "obj+0x10 vtable=%llx (eboot+0x%llx)\n", $o10vt, $o10vt-0x400000000
  x/8gx $o10vt
  set $m20 = *(unsigned long long*)($o10vt+0x20)
  printf "method[+0x20]=%llx (eboot+0x%llx)\n", $m20, $m20-0x400000000
  x/120i $m20
  printf "-- deadline/start doubles --\n"
  printf "start(-0x78)=%f dead(-0x70)=%f\n", *(double*)($rbp-0x78), *(double*)($rbp-0x70)
  printf "-- 0x402327130 (pump/process) --\n"
  x/80i 0x402327130
  printf "-- the enclosing fn preamble 0x402cc2200 --\n"
  x/60i 0x402cc2200
  printf "-- caller strings --\n"
  x/s 0x4082639c2
  x/s 0x407e3b518
  x/s 0x407fe886a
  printf "==== DEEP DONE ====\n"
  detach
  quit
end
continue
quit
