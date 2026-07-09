# gdb -p PID -batch -x mount232.gdb — issue #232: catch DOLL's savedata mount-wrapper callers.
# eboot base 0x400000000. Wrapper eboot+0x2251610: (this, flags, userId, dirName, blocks, result).
set pagination off
set confirm off

break *0x402251610
commands
  silent
  printf "MOUNTWRAP ra=%lx this=%lx esi=%x edx=%x rcx=%lx r8=%lx r9=%lx\n", *(unsigned long*)$rsp, $rdi, $esi, $edx, $rcx, $r8, $r9
  continue
end

# sceSaveDataDirNameSearch import thunk eboot+0x6698c70: (cond*, result*)
break *0x406698c70
commands
  silent
  printf "DIRSEARCH ra=%lx cond=%lx result=%lx\n", *(unsigned long*)$rsp, $rdi, $rsi
  x/10gx $rdi
  x/10gx $rsi
  continue
end

continue
