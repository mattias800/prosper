# Catch the issue-#222 EUD-ring store fault at eboot+0x59949e4 and dump the object graph.
set pagination off
set confirm off
handle SIG34 nostop pass noprint
handle SIG35 nostop pass noprint
handle SIG36 nostop pass noprint
handle SIG37 nostop pass noprint
handle SIG38 nostop pass noprint
handle SIGUSR1 nostop pass noprint
handle SIGUSR2 nostop pass noprint
catch signal SIGSEGV
commands
  silent
  if $rip == 0x4059949e4
    printf "==== FATAL 222 CAUGHT ====\n"
    info registers rip rax rbx rcx rdx rsi rdi r14 r15
    printf "---- sub object (rbx) ----\n"
    x/16gx $rbx
    printf "---- descriptor ud = [rbx+8] ----\n"
    set $ud = *(unsigned long long*)($rbx+8)
    printf "ud=%llx\n", $ud
    x/10gx $ud
    printf "---- direct_resource_offset table (u16[32]) = [ud+0] ----\n"
    set $dro = *(unsigned long long*)$ud
    printf "dro=%llx\n", $dro
    x/32hx $dro
    printf "---- ring object = [rbx+0x50] ----\n"
    set $ring = *(unsigned long long*)($rbx+0x50)
    printf "ring=%llx\n", $ring
    x/12gx $ring
    printf "---- caller frame: ret addrs on stack ----\n"
    x/16gx $rsp
    printf "---- backtrace ----\n"
    bt 8
    printf "---- thread id ----\n"
    thread
    printf "==== DUMP DONE ====\n"
    kill
    quit
  else
    signal SIGSEGV
  end
end
run
quit
