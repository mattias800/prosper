# Append a flush after a successful executable link. Keep the toolchain's complete link rule.
# The CI profile selects this explicitly; ordinary builds keep their original rules.
function(prosper_flush_link_outputs powershell)
  if(NOT CMAKE_CXX_LINK_EXECUTABLE)
    message(FATAL_ERROR "Executable link rule must be initialized before enabling output flushes")
  endif()
  set(_flush_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/flush_link_output.ps1")
  list(APPEND CMAKE_CXX_LINK_EXECUTABLE
    "\"${powershell}\" -NoLogo -NoProfile -NonInteractive -File \"${_flush_script}\" -OutputFile <TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}" PARENT_SCOPE)
endfunction()
