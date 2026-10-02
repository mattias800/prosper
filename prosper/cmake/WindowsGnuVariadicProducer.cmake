# Clang does not support va_start for a SysV-ABI function on a Windows target.
# Keep only the existing trivial capture frames on the same-sysroot GNU compiler;
# the main compiler owns every host delegate and nontrivial TLS lifetime.
function(prosper_add_gnu_variadic_producer target source name)
  if(NOT PROSPER_NATIVE_CLANG_TLS)
    target_sources(${target} PRIVATE "${source}")
    return()
  endif()
  set(_gcc "${PROSPER_NATIVE_MINGW_ROOT}/bin/g++.exe")
  if(NOT EXISTS "${_gcc}")
    message(FATAL_ERROR "native-Clang profile requires same-sysroot bin/g++.exe for SysV variadic capture")
  endif()
  execute_process(COMMAND "${_gcc}" -dumpmachine OUTPUT_VARIABLE _machine
                  RESULT_VARIABLE _status OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _status EQUAL 0 OR NOT _machine STREQUAL "x86_64-w64-mingw32")
    message(FATAL_ERROR "SysV variadic producer must be an actual x86_64 MinGW GNU compiler")
  endif()
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _configuration)
  separate_arguments(_flags NATIVE_COMMAND
    "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_configuration}}")
  set(_object "${CMAKE_CURRENT_BINARY_DIR}/gnu-variadic/${name}.obj")
  set(_depfile "${_object}.d")
  # Build-local argv, not a shell command: each nonempty line is one actual argument.
  # The compile-case identity scanner can audit the secondary compiler and its headers.
  # This manifest is never exported. Newline-bearing flag/path syntax is unsupported.
  foreach(_argument IN ITEMS "${_gcc}" "${source}" "${PROSPER_NATIVE_MINGW_ROOT}" ${_flags})
    if(_argument MATCHES "[\r\n]")
      message(FATAL_ERROR "SysV variadic producer arguments cannot contain newlines")
    endif()
  endforeach()
  string(JOIN "\n" _flag_lines ${_flags})
  file(GENERATE OUTPUT "${_object}.argv" CONTENT
"${_gcc}
--sysroot=${PROSPER_NATIVE_MINGW_ROOT}
-std=gnu++${CMAKE_CXX_STANDARD}
${_flag_lines}
$<$<BOOL:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>>:-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,\n-I>>
$<$<BOOL:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>>:-D$<JOIN:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,\n-D>>
-MD
-MF
${_depfile}
-MT
${_object}
-c
${source}
-o
${_object}
")
  add_custom_command(OUTPUT "${_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/gnu-variadic"
    COMMAND "${_gcc}" "--sysroot=${PROSPER_NATIVE_MINGW_ROOT}"
      "-std=gnu++${CMAKE_CXX_STANDARD}" ${_flags}
      "$<$<BOOL:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>>:-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,;-I>>"
      "$<$<BOOL:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>>:-D$<JOIN:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,;-D>>"
      -MD -MF "${_depfile}" -MT "${_object}" -c "${source}" -o "${_object}"
    DEPENDS "${source}" "${_gcc}" "${_object}.argv" DEPFILE "${_depfile}"
    COMMAND_EXPAND_LISTS VERBATIM)
  set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  target_sources(${target} PRIVATE "${_object}")
endfunction()
