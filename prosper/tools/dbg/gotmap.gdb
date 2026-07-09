# Read the GOT slots 0x40948d138/0x40948d140 once the image is linked, then map stubs to NIDs
# by dumping the stub-region addresses. Break late (GetOutputStatus) so relocation is done.
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
break _ZN7prosperL22g_vo_get_output_statusEmmmmmm
commands
  silent
  printf "GOT[0xa8]=0x%llx\n", *(unsigned long long*)0x40948d130
  printf "GOT[0xa9]=0x%llx (the count query)\n", *(unsigned long long*)0x40948d138
  printf "GOT[0xaa]=0x%llx (the fill call)\n", *(unsigned long long*)0x40948d140
  printf "GOT[0xab]=0x%llx\n", *(unsigned long long*)0x40948d148
  kill
  quit
end
run
quit
