# Stop before unwinding the std::system_error reported by the Windows PM4 fixtures (#5104).
# A normal exit, missing symbol or unrelated signal must not print the capture marker.
set pagination off
set width 0
set confirm off
# main can be absent from the linked test binary. Arm before launch and resolve
# the runtime's throwing function when its shared library loads.
set breakpoint pending on
break std::__throw_system_error(int)
commands
  silent
  echo PROSPER_SYSTEM_ERROR_STACK\n
  thread apply all bt 32
  quit 42
end
run
echo PROSPER_NO_SYSTEM_ERROR_STACK\n
quit 0
