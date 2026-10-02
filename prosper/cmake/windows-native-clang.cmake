# Opt-in native Windows GNU-ABI Clang profile. Requires existing LLVM and MinGW installations.
# Do not set CMAKE_SYSTEM_NAME: this is a native host build, not a cross-compilation exemption.
if(NOT CMAKE_HOST_WIN32)
  message(FATAL_ERROR "windows-native-clang is a native Windows-only profile")
endif()
set(PROSPER_NATIVE_CLANG_ROOT "" CACHE PATH "Existing LLVM installation root")
set(PROSPER_NATIVE_MINGW_ROOT "" CACHE PATH "Existing x86_64 MinGW GNU-ABI sysroot")
foreach(_tool IN ITEMS clang clang++)
  if(NOT EXISTS "${PROSPER_NATIVE_CLANG_ROOT}/bin/${_tool}.exe")
    message(FATAL_ERROR "PROSPER_NATIVE_CLANG_ROOT must contain bin/${_tool}.exe")
  endif()
endforeach()
if(NOT EXISTS "${PROSPER_NATIVE_MINGW_ROOT}/include/windows.h" OR
   NOT EXISTS "${PROSPER_NATIVE_MINGW_ROOT}/lib/libwinpthread.a")
  message(FATAL_ERROR "PROSPER_NATIVE_MINGW_ROOT must contain MinGW headers and libwinpthread.a")
endif()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  PROSPER_NATIVE_CLANG_ROOT PROSPER_NATIVE_MINGW_ROOT)
set(CMAKE_C_COMPILER "${PROSPER_NATIVE_CLANG_ROOT}/bin/clang.exe")
set(CMAKE_CXX_COMPILER "${PROSPER_NATIVE_CLANG_ROOT}/bin/clang++.exe")
set(CMAKE_C_COMPILER_TARGET x86_64-w64-windows-gnu)
set(CMAKE_CXX_COMPILER_TARGET x86_64-w64-windows-gnu)
set(CMAKE_SYSROOT "${PROSPER_NATIVE_MINGW_ROOT}")
set(PROSPER_NATIVE_CLANG_TLS TRUE)
