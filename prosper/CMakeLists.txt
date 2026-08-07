cmake_minimum_required(VERSION 3.20)
project(prosper LANGUAGES C CXX)   # C for the vendored LibAtrac9 decoder (third_party/libatrac9)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Agentic-first: binaries must run with no PATH/DLL setup so CI (ctest) is hermetic.
# On mingw, statically link the GCC/C++ runtimes.
if(MINGW OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  add_link_options(-static -static-libgcc -static-libstdc++)
endif()

# --- macOS Vulkan (MoltenVK) override ------------------------------------
# prosper on macOS is built x86_64 (to run the guest under Rosetta), but Homebrew's MoltenVK/loader
# are arm64-only, so find_package(Vulkan) would hand back an unlinkable arm64 lib. Point the build at
# a UNIVERSAL (x86_64+arm64) MoltenVK instead — either the LunarG Vulkan SDK, or the prebuilt
# KhronosGroup/MoltenVK release (`MoltenVK-macos.tar`, universal). Pass:
#   -DPROSPER_MACOS_MOLTENVK=/path/to/libMoltenVK.dylib   (universal)
#   -DPROSPER_MACOS_VULKAN_INCLUDE=/path/to/vulkan/headers (dir containing vulkan/vulkan.h)
# When set we pre-create the Vulkan::Vulkan imported target from it, so every later
# find_package(Vulkan) is skipped (guarded on `NOT TARGET Vulkan::Vulkan`). Headers are
# architecture-independent, so Homebrew's vulkan-headers work for the include dir.
if(APPLE AND PROSPER_MACOS_MOLTENVK)
  if(NOT PROSPER_MACOS_VULKAN_INCLUDE)
    set(PROSPER_MACOS_VULKAN_INCLUDE "/opt/homebrew/include")   # Homebrew vulkan-headers (arch-independent)
  endif()
  add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
  set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${PROSPER_MACOS_MOLTENVK}"
    INTERFACE_INCLUDE_DIRECTORIES "${PROSPER_MACOS_VULKAN_INCLUDE}")
  set(Vulkan_FOUND TRUE)
  set(Vulkan_LIBRARY "${PROSPER_MACOS_MOLTENVK}")
  set(Vulkan_INCLUDE_DIR "${PROSPER_MACOS_VULKAN_INCLUDE}")
  # The app dlopen's / links MoltenVK by leaf name; add its dir to the executables' rpath so
  # SDL_Vulkan_LoadLibrary("libMoltenVK.dylib") and the direct link both resolve at runtime.
  get_filename_component(_mvk_dir "${PROSPER_MACOS_MOLTENVK}" DIRECTORY)
  set(CMAKE_BUILD_RPATH "${_mvk_dir}")
  message(STATUS "macOS Vulkan: using universal MoltenVK ${PROSPER_MACOS_MOLTENVK}")
endif()

# --- tools ---------------------------------------------------------------
# self_dump: SELF/ELF inspector and NID import extractor (M0).
add_executable(self_dump tools/self_dump/self_dump.cpp)

# --- third-party: LibAtrac9 (MIT, vendored) - ATRAC9 audio decode -------------------------
# Vendored verbatim (see third_party/libatrac9/README.md). C99; needs libm. Compiled as its own
# static lib so the C sources stay isolated from prosper's C++ and are trivially auditable/updatable.
file(GLOB LIBATRAC9_SRC CONFIGURE_DEPENDS third_party/libatrac9/src/*.c)
add_library(libatrac9 STATIC ${LIBATRAC9_SRC})
target_include_directories(libatrac9 PUBLIC third_party/libatrac9/src)
set_target_properties(libatrac9 PROPERTIES C_STANDARD 99)
if(NOT WIN32)
  target_link_libraries(libatrac9 PUBLIC m)
endif()

# --- core library (M1+) --------------------------------------------------
# Loader / HLE / kernel / gpu sources accrete here per the roadmap.
include(cmake/ProsperBuildRevision.cmake)
prosper_add_build_revision_library(prosper_build_revision WORK_TREE "${CMAKE_SOURCE_DIR}/..")

file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)
if(PROSPER_SRC)
  add_library(prosper_core STATIC ${PROSPER_SRC})
  target_include_directories(prosper_core PUBLIC src)
  target_link_libraries(prosper_core PUBLIC libatrac9 PRIVATE prosper_build_revision)
  target_compile_definitions(prosper_core PRIVATE PROSPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  find_package(Threads REQUIRED)
  target_link_libraries(prosper_core PUBLIC Threads::Threads)
  # Windows: the guest futex HLE (sceKernelWaitOnAddress/WakeByAddress) is backed by the native Win32
  # WaitOnAddress/WakeByAddressSingle/All, which live in the Synchronization import lib (not kernel32).
  if(WIN32)
    target_link_libraries(prosper_core PUBLIC synchronization bcrypt)
  endif()
endif()

# Bounded interactive performance capture (F8): pure host-side state + serialization shared by the
# SDL app and the live graphics/compute backends. It deliberately has no SDL/Vulkan dependency, so
# its ring, bounds, and atomic file contract run in ordinary headless CI.
add_library(prosper_performance_capture STATIC frontends/shared/performance_capture.cpp)
target_include_directories(prosper_performance_capture PUBLIC frontends/shared)
target_link_libraries(prosper_performance_capture PRIVATE prosper_build_revision)

# --- tests (agentic-first: self-checking, exit-code = truth) -------------
enable_testing()

# Pure selection/accounting policy behind the stable compute-timing hash filter. This must remain
# runnable without prosper_core, a game dump, or Vulkan so a broken diagnostic cannot hide behind a
# skipped execution test on a host without graphics support.
add_executable(test_compute_timing_selector tests/test_compute_timing_selector.cpp)
target_include_directories(test_compute_timing_selector PRIVATE frontends/shared)
add_test(NAME compute_timing_selector_policy COMMAND test_compute_timing_selector)

# Pure fail-closed policy for a behavior-neutral census of hypothetical deferred compute results.
# It has no Vulkan dependency and does not change the live backend's synchronous writeback.
add_executable(test_compute_authority_census tests/test_compute_authority_census.cpp)
target_include_directories(test_compute_authority_census PRIVATE frontends/shared)
add_test(NAME compute_authority_census_policy COMMAND test_compute_authority_census)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME re_xref_decoder
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_xref.py")
  add_test(NAME re_edis_mapping
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_edis.py")
  # nid_gate_scan answers "does this title BRANCH on an imported function's return value?", and a
  # wrong answer there is a plausible-looking table rather than a visible failure. Three of its
  # cases are regressions for classifier bugs found only after a table had been published (#2002).
  add_test(NAME re_nid_gate_classifier
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_nid_gate_scan.py")
  # prx_to_elf.py writes an ELF header by hand, four bytes apart from a field that means something
  # else entirely — #2016 put the program-header count in the section-header-count slot, and the
  # result was a malformed image no consumer complained about until binutils refused the file. The
  # test reads the produced bytes at their gABI offsets; nothing else can see this class of bug.
  add_test(NAME il2cpp_prx_to_elf_header
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/il2cpp/test_prx_to_elf.py")
  add_test(NAME snapshot_guard_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/snapshot/test_snapshot.py")
  add_test(NAME gpu_replay_regress_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_regress.py")
  # Half its cases are CORRECT shapes an earlier revision of the checker wrongly rejected. A doc
  # check that fires on valid input gets deleted rather than heeded, so the false-positive cases
  # are what keep it alive.
  add_test(NAME doc_table_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_numbered_table.py")
  # Parsing/gating logic of the Vulkan validation guard (#1704). Pure — it does not create a Vulkan
  # instance, so it runs everywhere. A parser that stops matching the layer's message framing would
  # report a clean suite forever, which is the one way this guard can fail silently.
  add_test(NAME vkval_scan_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/vkval/test_vk_validation_scan.py")
  # PROSPER_ENV_ON/_VALUE sample getenv once and return it forever. Applied to a name a TEST arms at
  # runtime, the arm becomes invisible -- and the test does not fail, it goes vacuous and keeps
  # printing [ok], which is why #2214 surfaced only as one unrelated-looking Vulkan binding error on
  # a single CI job. The scanner carries its own self-tests because its failure mode is silence: a
  # pattern that stops matching reports a clean tree forever. Three of those cases cover the wrapper
  # call forms (set_env/unset_env), which the first revision missed -- it passed a tree that still
  # contained the defect.
  add_test(NAME cached_env_arming_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_cached_env.py"
                   "${CMAKE_SOURCE_DIR}")
  # Most cases here assert on what must NOT appear (negative time, failed dispatches summed in, an
  # image share computed against a filtered denominator, a per-binding mean over aliased folds),
  # because each is a table this tool has already produced once. Note what this gate does NOT do: it
  # feeds the tool SYNTHETIC records, so it pins the tool against its own model and can never notice
  # that model drifting from live_compute.cpp. Only the tool's runtime broken_sum/broken_nest warning
  # detects real drift, and only when someone reads stderr.
  add_test(NAME compute_phase_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_compute_phase_report.py")
  if(GIT_EXECUTABLE)
    # The provenance GATE: refuses to quote a measurement from a build that is not the revision the
    # author thinks it is. Pure argument/exit-status contract, no build required — but it resolves
    # real refs and creates a scratch repository, so it needs git exactly as its sibling below does.
    add_test(NAME check_build_revision
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/revision/test_check_build_revision.py")
    add_test(NAME build_revision_refresh
             COMMAND "${Python3_EXECUTABLE}"
               "${CMAKE_SOURCE_DIR}/tools/revision/test_build_revision.py"
               --cmake "${CMAKE_COMMAND}"
               --git "${GIT_EXECUTABLE}"
               --module "${CMAKE_SOURCE_DIR}/cmake/ProsperBuildRevision.cmake"
               --scratch-root "${CMAKE_CURRENT_BINARY_DIR}/build-revision-regression"
               --generator "${CMAKE_GENERATOR}")
  endif()
endif()
# Path to the (gitignored) local game dump; override with -DGAME_DUMP=...
set(GAME_DUMP "${CMAKE_SOURCE_DIR}/../PPSA24651-app0" CACHE PATH "Local PS5 game dump root")
set(HAVE_GAME_DUMP OFF)
if(EXISTS "${GAME_DUMP}/eboot.bin")
  set(HAVE_GAME_DUMP ON)
else()
  message(STATUS "Game dump not found at ${GAME_DUMP} — skipping dump-backed tests")
endif()

if(TARGET prosper_core)
  # nid_census: every guest import that falls to the dispatcher's return-0 default (#2081).
  # Links prosper_core because both of its inputs must be the real ones — `Module::load` for the
  # imports the loader will actually try to bind, `module_export_nids` for what a sibling module
  # already defines, and the live `Hle` registry for whether a handler exists. A source grep over
  # register_fn() call sites cannot see any of the three.
  add_executable(nid_census tools/nid_census/nid_census.cpp)
  target_link_libraries(nid_census PRIVATE prosper_core)

  add_executable(test_module tests/test_module.cpp)
  target_link_libraries(test_module PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME module_loads_eboot
             COMMAND test_module "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_nid tests/test_nid.cpp)
  target_link_libraries(test_nid PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME nid_hash_matches_imports
             COMMAND test_nid "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_hle_registered tests/test_hle_registered.cpp)
  target_link_libraries(test_hle_registered PRIVATE prosper_core)
  add_test(NAME hle_functions_registered COMMAND test_hle_registered)

  # No NID registered twice with a DIFFERENT handler (the #330 shadow class): a later register_*()
  # silently overwriting a real impl with a stub returns garbage to the guest. Pure, no game dump.
  add_executable(test_hle_no_shadow tests/test_hle_no_shadow.cpp)
  target_link_libraries(test_hle_no_shadow PRIVATE prosper_core)
  add_test(NAME hle_no_shadow COMMAND test_hle_no_shadow)

  # Cooperative stop signal (issue #164 P0): request_stop/stop_requested the frontend uses to wind
  # the guest run-loop down on window-close. Pure, no deps.
  add_executable(test_lifecycle tests/test_lifecycle.cpp)
  target_link_libraries(test_lifecycle PRIVATE prosper_core)
  add_test(NAME lifecycle_stop_signal COMMAND test_lifecycle)

  # Screenshot evidence classification/JSON is pure and must remain testable without Vulkan or a dump.
  add_executable(test_capture_manifest tests/test_capture_manifest.cpp
                                       tools/screenshot/capture_manifest.cpp)
  # `src` and the revision library are for `build_revision.hpp` only: the run header records the
  # revision the binary was compiled from, so the manifest self-describes. Still no Vulkan, no dump.
  target_include_directories(test_capture_manifest PRIVATE tools/screenshot src)
  target_link_libraries(test_capture_manifest PRIVATE prosper_build_revision)
  add_test(NAME capture_manifest COMMAND test_capture_manifest)

  # va2foff resolution order (issue #113): DYNAMIC/PROCPARAM/TLS must never shadow
  # the PT_LOAD that truly maps a VA. Pure unit test — no game dump required.
  add_executable(test_va2foff tests/test_va2foff.cpp)
  target_link_libraries(test_va2foff PRIVATE prosper_core)
  add_test(NAME va2foff_load_priority COMMAND test_va2foff)

  # SELF/ELF/loader parsing must handle a malformed/truncated dump without an out-of-bounds read:
  # overflow-safe rd() bounds (self_read_ok), NUL-terminated str_at, and the linker init-array p+8
  # guard. Pure, no game dump.
  add_executable(test_self_bounds tests/test_self_bounds.cpp)
  target_link_libraries(test_self_bounds PRIVATE prosper_core)
  add_test(NAME self_bounds COMMAND test_self_bounds)

  # SELF data-segment key masks flag bits >= 32 (#339): the SCE segment id is bits [31:20]; without
  # the mask a data segment keys the map wrong and every phdr lookup misses. Pure, no game dump.
  add_executable(test_self_segment_key tests/test_self_segment_key.cpp)
  target_link_libraries(test_self_segment_key PRIVATE prosper_core)
  add_test(NAME self_segment_key COMMAND test_self_segment_key)

  # Guest static-TLS Variant II layout uses each module's real PT_TLS p_align (#143). The layout
  # code is Linux-only (guest %fs), so gate the test on it too.
  if(UNIX)
    add_executable(test_guest_tls_align tests/test_guest_tls_align.cpp)
    target_link_libraries(test_guest_tls_align PRIVATE prosper_core)
    add_test(NAME guest_tls_align COMMAND test_guest_tls_align)
  endif()

  # Cross-module TLS resolution (#136/#338): dynamic and initial-exec relocations resolve through
  # the DEFINING module's id, symbol offset, and shared static layout. Pure synthetic unit test.
  add_executable(test_dtpmod_tls tests/test_dtpmod_tls.cpp)
  target_link_libraries(test_dtpmod_tls PRIVATE prosper_core)
  add_test(NAME dtpmod_tls_cross_module COMMAND test_dtpmod_tls)

  # AMD SSE4a INSERTQ/EXTRQ emulation (the SIGILL handler emulates these Zen2-only ops on Intel
  # hosts). Golden bit-math vectors — header-only, no game dump required.
  add_executable(test_sse4a tests/test_sse4a.cpp)
  add_test(NAME sse4a_insertq_extrq COMMAND test_sse4a)

  # macOS/Rosetta %fs trap-emulation decode core (guest-TLS access redirection). Golden
  # instruction vectors incl. the #727 no-REX AH/CH/DH/BH byte-register rules — header-only,
  # runs on Linux where the __APPLE__ handler itself never compiles.
  add_executable(test_fs_emu tests/test_fs_emu.cpp)
  add_test(NAME fs_emu_decode COMMAND test_fs_emu)

  # Signal-handler-safe raw syscalls (src/host/raw_syscall.hpp): the fault_handler's
  # PROSPER_NULL_PAGE backing must not touch errno/TLS on guest-%fs threads (issue #1071).
  if(UNIX AND NOT APPLE)
    add_executable(test_raw_syscall tests/test_raw_syscall.cpp)
    add_test(NAME raw_syscall COMMAND test_raw_syscall)
  endif()

  # sceAudioOut HLE: format decode, port lifecycle, PCM forwarding, volume, batch, error paths
  # (via a capturing AudioSink — headless, no device).
  add_executable(test_audio tests/test_audio.cpp)
  target_link_libraries(test_audio PRIVATE prosper_core)
  add_test(NAME audio_hle COMMAND test_audio)

  # NGS2 waveform-block streaming (#1059/#1060 part 3): a fed block makes the voice Playing and advances
  # num_decoded_samples across renders (the Dead Cells music stream clock). POSIX-gated (needs a fixed
  # guest-address PCM mapping); the lifecycle assertions run everywhere.
  add_executable(test_ngs2_streaming tests/test_ngs2_streaming.cpp)
  target_link_libraries(test_ngs2_streaming PRIVATE prosper_core)
  add_test(NAME ngs2_streaming COMMAND test_ngs2_streaming)

  # libSceAjm (#187): headless audio-decode lifecycle — handle out-params filled with valid values
  # (context/instance/batch), inputs validated, batch reports complete. Pure, no game dump.
  add_executable(test_ajm tests/test_ajm.cpp)
  target_link_libraries(test_ajm PRIVATE prosper_core)
  add_test(NAME ajm_hle COMMAND test_ajm)

  add_executable(test_avplayer tests/test_avplayer.cpp)
  target_link_libraries(test_avplayer PRIVATE prosper_core)
  add_test(NAME avplayer_hle COMMAND test_avplayer)

  add_executable(test_json2 tests/test_json2.cpp)
  target_link_libraries(test_json2 PRIVATE prosper_core)
  add_test(NAME json2_hle COMMAND test_json2)

  add_executable(test_equeue_events tests/test_equeue_events.cpp)
  target_link_libraries(test_equeue_events PRIVATE prosper_core)
  add_test(NAME equeue_events COMMAND test_equeue_events)

  # Variadic libc formatters (issue #122): real va_list capture so %f/XMM + stack + %s-past-4th
  # args format correctly (was a 3-integer-register forward). Cross-platform (host libc printf).
  add_executable(test_printf tests/test_printf.cpp)
  target_link_libraries(test_printf PRIVATE prosper_core)
  add_test(NAME printf_varargs COMMAND test_printf)

  # Windows' aligned allocator family must remain paired end-to-end. Mixing _aligned_malloc with
  # plain free corrupts the heap under aligned game-buffer churn; ordinary and aligned guest
  # allocations intentionally share one compatible family on that host.
  add_executable(test_libc_alloc tests/test_libc_alloc.cpp)
  target_link_libraries(test_libc_alloc PRIVATE prosper_core)
  add_test(NAME libc_alloc COMMAND test_libc_alloc)

  # EventFlag/Sema delete-under-waiter safety (issue #104): delete wakes parked waiters with EACCES
  # and defers the free to the last one out, instead of free()ing the object under a live wait (UAF).
  add_executable(test_sync_delete tests/test_sync_delete.cpp)
  target_link_libraries(test_sync_delete PRIVATE prosper_core)
  add_test(NAME sync_delete COMMAND test_sync_delete)

  # Anchored wall-clock sources (#92): CLOCK_REALTIME ≈ host now, one shared epoch across
  # time()/gettimeofday/sceRtc*, FreeBSD clockid mapping, valid LocalTime calendar dates.
  add_executable(test_wall_clock tests/test_wall_clock.cpp)
  target_link_libraries(test_wall_clock PRIVATE prosper_core)
  add_test(NAME wall_clock COMMAND test_wall_clock)

  # select/pselect pure-sleep shape honours its timeout; a descriptor-set query stays fail-visible
  # (#1660). Asserts elapsed time, not the return value -- the stub this replaces also returned 0.
  add_executable(test_select_sleep tests/test_select_sleep.cpp)
  target_link_libraries(test_select_sleep PRIVATE prosper_core)
  add_test(NAME select_sleep COMMAND test_select_sleep)

  # Opt-in flip-paced monotonic clock (#240): exact frame steps, thread-safe first-flip anchor,
  # and no coupling into CLOCK_REALTIME while virtual time is paused between flips.
  add_executable(test_det_clock tests/test_det_clock.cpp)
  target_link_libraries(test_det_clock PRIVATE prosper_core)
  add_test(NAME det_clock COMMAND test_det_clock)

  # Synchronous host-GPU work must not become a giant guest frame delta. The clock retains one
  # caller-supplied frame budget, stays monotonic under concurrent reads, and leaves realtime alone.
  add_executable(test_gpu_time_compensation tests/test_gpu_time_compensation.cpp)
  target_link_libraries(test_gpu_time_compensation PRIVATE prosper_core)
  add_test(NAME gpu_time_compensation COMMAND test_gpu_time_compensation)

  # UserService/libkernel startup stubs (HLE audit): getters write deterministic defaults; the
  # libkernel registration/hook calls resolve to benign OK no-ops.
  add_executable(test_service_getters tests/test_service_getters.cpp)
  target_link_libraries(test_service_getters PRIVATE prosper_core)
  add_test(NAME service_getters COMMAND test_service_getters)

  # #2185: reallocalign was unregistered, so the dispatcher's 0 reached the guest as a NULL
  # allocation — a false FAILURE (reads as OOM), the mirror of the #2081 family below. Asserts the
  # extended alignment survives a resize, which plain realloc never promised. Pure, no game dump.
  add_executable(test_reallocalign tests/test_reallocalign.cpp)
  target_link_libraries(test_reallocalign PRIVATE prosper_core)
  add_test(NAME reallocalign COMMAND test_reallocalign)

  add_executable(test_reallocf tests/test_reallocf.cpp)
  target_link_libraries(test_reallocf PRIVATE prosper_core)
  add_test(NAME reallocf COMMAND test_reallocf)

  # #2081 FALSE SUCCESS: an unregistered NID answers the dispatcher's 0, which for an SCE_OK
  # contract is a success report over an out-parameter nothing wrote. Pure, no game dump.
  add_executable(test_false_success_nids tests/test_false_success_nids.cpp)
  target_link_libraries(test_false_success_nids PRIVATE prosper_core)
  add_test(NAME false_success_nids COMMAND test_false_success_nids)

  # #2004: sceAppContentAddcontUnmount releases the claim the mount takes, so a mount/unmount/
  # re-mount cycle works instead of being permanently BUSY. Pure, no game dump.
  add_executable(test_addcontent_unmount tests/test_addcontent_unmount.cpp)
  target_link_libraries(test_addcontent_unmount PRIVATE prosper_core)
  add_test(NAME addcontent_unmount COMMAND test_addcontent_unmount)

  # libSceSysmodule state query (#2002): IsLoaded answers from prosper's own load history, so a
  # title that gates its initialization on UNLOADED stops skipping that initialization.
  add_executable(test_sysmodule_state tests/test_sysmodule_state.cpp)
  target_link_libraries(test_sysmodule_state PRIVATE prosper_core)
  add_test(NAME sysmodule_state COMMAND test_sysmodule_state)

  # libSceHttp local URI parser (#642): two-pass caller pool sizing, public URI layout, and
  # deterministic error paths. Pure and offline; it never contacts a network.
  add_executable(test_http tests/test_http.cpp)
  target_link_libraries(test_http PRIVATE prosper_core)
  add_test(NAME http_hle COMMAND test_http)

  add_executable(test_guard_recursion tests/test_guard_recursion.cpp)
  target_link_libraries(test_guard_recursion PRIVATE prosper_core Threads::Threads)
  add_test(NAME guard_recursion COMMAND test_guard_recursion)
  set_tests_properties(guard_recursion PROPERTIES TIMEOUT 30)

  add_executable(test_font tests/test_font.cpp)
  target_link_libraries(test_font PRIVATE prosper_core)
  add_test(NAME font_hle COMMAND test_font)

  add_executable(test_fiber tests/test_fiber.cpp)
  target_link_libraries(test_fiber PRIVATE prosper_core)
  add_test(NAME fiber_hle COMMAND test_fiber)

  # #1603: libSceUlt has no implementation, so it must fail VISIBLY rather than return 0 = SCE_OK.
  # Pure, no game dump.
  add_executable(test_ult_fail_visible tests/test_ult_fail_visible.cpp)
  target_link_libraries(test_ult_fail_visible PRIVATE prosper_core)
  add_test(NAME ult_fail_visible COMMAND test_ult_fail_visible)

  # #1603 Phase 2: the implemented libSceUlt semantics. Mutual exclusion is DEMONSTRATED with real
  # concurrent threads over a non-atomic counter and an exact expected total — a no-op mutex produces
  # a wrong-but-plausible number rather than a crash, so only the exact total is a real check.
  # Pure, no game dump.
  add_executable(test_ult_semantics tests/test_ult_semantics.cpp)
  target_link_libraries(test_ult_semantics PRIVATE prosper_core Threads::Threads)
  add_test(NAME ult_semantics COMMAND test_ult_semantics)

  add_executable(test_posix_sem tests/test_posix_sem.cpp)
  target_link_libraries(test_posix_sem PRIVATE prosper_core)
  add_test(NAME posix_sem_hle COMMAND test_posix_sem)

  # FreeBSD errno numbering across the libkernel 0x80020000|errno family (#1612). Pins the encoding
  # vocabulary, the host->FreeBSD table, and the guest-observable values on the paths a Linux/macOS
  # host can actually reach (sem trywait, timed mutex, wait-on-address, POSIX setcancelstate).
  add_executable(test_sce_errno tests/test_sce_errno.cpp)
  target_link_libraries(test_sce_errno PRIVATE prosper_core)
  add_test(NAME sce_errno_freebsd COMMAND test_sce_errno)

  # #1659: every diagnostic that prints "<module>+0x<rva>" agrees with the loader's mapped base, and a
  # printed offset round-trips back through PROSPER_BP. Header-only; no dump, no device.
  add_executable(test_guest_module_label tests/test_guest_module_label.cpp)
  target_link_libraries(test_guest_module_label PRIVATE prosper_core)
  add_test(NAME guest_module_label COMMAND test_guest_module_label)

  # PlatformUi hook (#347): the app frontend can service the guest's ImeDialog with real UI; with no
  # backend registered the headless auto-complete default is unchanged. Pure, no game dump.
  add_executable(test_platform_ui tests/test_platform_ui.cpp)
  target_link_libraries(test_platform_ui PRIVATE prosper_core)
  add_test(NAME platform_ui COMMAND test_platform_ui)

  # SDL3 dialog frontend's guest-struct parsing (#347): pure, SDL-free, so it runs in normal CI even
  # though the SDL_ShowMessageBox glue is display-only. Verifies the MsgDialog/ErrorDialog layouts.
  add_executable(test_dialog_helpers frontends/dialog_sdl3/test_dialog_helpers.cpp
                                     frontends/dialog_sdl3/dialog_helpers.cpp)
  add_test(NAME dialog_helpers COMMAND test_dialog_helpers)

  # libSceImeDialog lifecycle + libSceSaveData memory API round-trip (#191): auto-completing dialog
  # (no hang) + a real per-(user,slot) Set->Get memory block with verified shadPS4 struct layouts.
  add_executable(test_savedata_ime tests/test_savedata_ime.cpp)
  target_link_libraries(test_savedata_ime PRIVATE prosper_core)
  add_test(NAME savedata_ime COMMAND test_savedata_ime)

  # libSceIme keyboard API (#186): reports a consistent "no keyboard connected" state (empty resource
  # array + disconnected info, bounds-checked) so the per-frame input poll never reads garbage.
  add_executable(test_ime_keyboard tests/test_ime_keyboard.cpp)
  target_link_libraries(test_ime_keyboard PRIVATE prosper_core)
  add_test(NAME ime_keyboard COMMAND test_ime_keyboard)

  # pthread scheduling getters fill their out-param (not a shadowing no-op): scePthreadGetprio/
  # Getaffinity must WRITE priority/mask, else the caller reads uninitialized stack (harmful-stub class).
  add_executable(test_pthread_sched_getters tests/test_pthread_sched_getters.cpp)
  target_link_libraries(test_pthread_sched_getters PRIVATE prosper_core)
  add_test(NAME pthread_sched_getters COMMAND test_pthread_sched_getters)

  add_executable(test_pthread_names tests/test_pthread_names.cpp)
  target_link_libraries(test_pthread_names PRIVATE prosper_core)
  add_test(NAME pthread_names COMMAND test_pthread_names)

  add_executable(test_cond_clock tests/test_cond_clock.cpp)
  target_link_libraries(test_cond_clock PRIVATE prosper_core)
  add_test(NAME pthread_cond_clock COMMAND test_cond_clock)

  # libScePad controller input: ScePadData layout + HostPadState mapping + HLE fill contract (pure).
  add_executable(test_pad tests/test_pad.cpp)
  target_link_libraries(test_pad PRIVATE prosper_core)
  add_test(NAME pad_input COMMAND test_pad)

  add_executable(test_prosper_app_pad_overlay frontends/prosper-app/test_pad_overlay.cpp)
  target_include_directories(test_prosper_app_pad_overlay PRIVATE src frontends/prosper-app)
  target_link_libraries(test_prosper_app_pad_overlay PRIVATE prosper_core)
  add_test(NAME prosper_app_pad_overlay COMMAND test_prosper_app_pad_overlay)

  # Which key is which pad control (#2234). No SDL here: the frontend keeps the Key -> scancode
  # table and the mapping itself lives in a header, so this runs in the core suite on every host.
  add_executable(test_prosper_app_keyboard_pad_map frontends/prosper-app/test_keyboard_pad_map.cpp)
  target_include_directories(test_prosper_app_keyboard_pad_map PRIVATE src frontends/prosper-app)
  target_link_libraries(test_prosper_app_keyboard_pad_map PRIVATE prosper_core)
  add_test(NAME prosper_app_keyboard_pad_map COMMAND test_prosper_app_keyboard_pad_map)

  add_executable(test_prosper_app_window_controls frontends/prosper-app/test_window_controls.cpp)
  target_include_directories(test_prosper_app_window_controls PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_window_controls COMMAND test_prosper_app_window_controls)

  add_executable(test_prosper_app_capture_schedule frontends/prosper-app/test_capture_schedule.cpp)
  target_include_directories(test_prosper_app_capture_schedule PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_capture_schedule COMMAND test_prosper_app_capture_schedule)

  add_executable(test_prosper_app_performance_capture_schedule
    frontends/prosper-app/test_performance_capture_schedule.cpp)
  target_include_directories(test_prosper_app_performance_capture_schedule
    PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_performance_capture_schedule
    COMMAND test_prosper_app_performance_capture_schedule)

  add_executable(test_prosper_app_present_mode frontends/prosper-app/test_present_mode.cpp)
  target_include_directories(test_prosper_app_present_mode PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_present_mode COMMAND test_prosper_app_present_mode)

  # The interactive open path's path-resolution + action decisions (#1469): pure and SDL-free, so
  # the drag-and-drop / folder-picker behaviour is covered in normal CI without a window.
  add_executable(test_prosper_app_game_path frontends/prosper-app/test_game_path.cpp)
  target_include_directories(test_prosper_app_game_path PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_game_path COMMAND test_prosper_app_game_path)

  # The game library scan, param.json metadata, and settings precedence (#1471): filesystem access is
  # injected, so the whole scan runs in normal CI with no window, no ImGui and no real dump.
  add_executable(test_prosper_app_game_library frontends/prosper-app/test_game_library.cpp)
  target_include_directories(test_prosper_app_game_library PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_game_library COMMAND test_prosper_app_game_library)

  # F9 frame-grab file naming: one stamp and one exclusively claimed name pair per capture, so two
  # titles in one directory cannot overwrite each other and an aborted grab cannot pair with a
  # different capture. Uses a real directory under the build tree and an injected clock — no SDL, no
  # window, no GPU.
  add_executable(test_prosper_app_frame_grab_naming frontends/prosper-app/test_frame_grab_naming.cpp)
  target_include_directories(test_prosper_app_frame_grab_naming PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_frame_grab_naming COMMAND test_prosper_app_frame_grab_naming)

  # F8's pre-trigger ring and post-trigger timing artifact: pure state machine + filesystem test,
  # including bounded overflow accounting and temporary-to-final atomic publication.
  add_executable(test_performance_capture frontends/shared/test_performance_capture.cpp)
  target_link_libraries(test_performance_capture PRIVATE prosper_performance_capture)
  add_test(NAME performance_capture COMMAND test_performance_capture)

  if(Python3_Interpreter_FOUND)
    add_test(NAME performance_capture_report_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_performance_capture_report.py")
  endif()

  # Library grid selection movement (#1471): arithmetic only, so the navigation rules are covered in
  # normal CI rather than by someone pressing arrow keys. No ImGui, no SDL, no window.
  add_executable(test_prosper_app_library_nav frontends/prosper-app/test_library_nav.cpp)
  target_include_directories(test_prosper_app_library_nav PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_library_nav COMMAND test_prosper_app_library_nav)

  # Library background art and focus music (#1630): DDS/BC7 and RIFF/ATRAC9 container validation plus
  # the debounce, two-stage fade and LRU policy. Pure byte and arithmetic work — the containers are
  # synthesized in the test, so no dump, no GPU and no audio device is involved.
  add_executable(test_prosper_app_launcher_media frontends/prosper-app/test_launcher_media.cpp)
  target_include_directories(test_prosper_app_launcher_media PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_launcher_media COMMAND test_prosper_app_launcher_media)

  # Fault-safe guest-pointer read in the PROSPER_AMPRLOG diagnostic (#1154): an unmapped, aligned,
  # > 0xffff pointer (GTA V's 0x302200) must report unreadable, not SIGSEGV. POSIX-only
  # (process_vm_readv / mmap); ampr_arglog lives in the __linux__||__APPLE__ half of hle_kernel_mem.
  if(NOT WIN32)
    add_executable(test_amprlog_safe_read tests/test_amprlog_safe_read.cpp)
    add_test(NAME amprlog_safe_read COMMAND test_amprlog_safe_read)
  endif()

  # Compute-side renderer-owned-RTT scale detection (PROSPER_RENDER_SCALE>1 downscale vs a view alias).
  # Pure integer arithmetic, no deps.
  add_executable(test_rtt_scale frontends/shared/test_rtt_scale.cpp)
  target_include_directories(test_rtt_scale PRIVATE frontends/shared)
  add_test(NAME rtt_scale COMMAND test_rtt_scale)

  # MRT-prefix truncation: which bound colour slot may join a pass, and which extent comparisons are
  # evidence of a real conflict rather than an unmeasured 0x0. Pure integer arithmetic, no deps.
  add_executable(test_mrt_extent frontends/shared/test_mrt_extent.cpp)
  target_include_directories(test_mrt_extent PRIVATE frontends/shared)
  add_test(NAME mrt_extent COMMAND test_mrt_extent)

  add_executable(test_rtt_injection frontends/shared/test_rtt_injection.cpp)
  target_include_directories(test_rtt_injection PRIVATE frontends/shared)
  add_test(NAME rtt_injection COMMAND test_rtt_injection)

  add_executable(test_readback_policy frontends/shared/test_readback_policy.cpp)
  target_include_directories(test_readback_policy PRIVATE frontends/shared)
  add_test(NAME readback_policy COMMAND test_readback_policy)

  add_executable(test_capture_renderer_policy frontends/shared/test_capture_renderer_policy.cpp)
  target_include_directories(test_capture_renderer_policy PRIVATE frontends/shared)
  add_test(NAME capture_renderer_policy COMMAND test_capture_renderer_policy)

  # Which sampled resource is AvPlayer's NV12 chroma plane (#2005). Descriptor-level classification
  # only — no device — but it reads the real guest-linear pitch registry, so it links prosper_core.
  add_executable(test_avplayer_plane_policy frontends/shared/test_avplayer_plane_policy.cpp)
  target_include_directories(test_avplayer_plane_policy PRIVATE frontends/shared src)
  target_link_libraries(test_avplayer_plane_policy PRIVATE prosper_core)
  add_test(NAME avplayer_plane_policy COMMAND test_avplayer_plane_policy)

  # The publish extent contract (#1986): which rendered pass may become the present source, and what
  # the retained-frame net may retain. Pure byte-count policy, so no device and no renderer.
  add_executable(test_present_extent frontends/shared/test_present_extent.cpp)
  target_include_directories(test_present_extent PRIVATE frontends/shared)
  add_test(NAME present_extent COMMAND test_present_extent)

  # When prosper may publish the guest's OWN flipped display buffer (#1968), including the guest
  # authorship precondition whose absence reverted the first attempt (#2044). Pure policy, no device.
  add_executable(test_guest_scanout_present frontends/shared/test_guest_scanout_present.cpp)
  target_include_directories(test_guest_scanout_present PRIVATE frontends/shared)
  add_test(NAME guest_scanout_present COMMAND test_guest_scanout_present)

  # The census window PROSPER_PASS_LOG / PROSPER_DUMP_PERSISTENT open: an ordinal (unchanged) or
  # `ms:<millis>` (#1968). Pure predicate — no renderer, no device, and the clock is an argument.
  add_executable(test_diagnostic_window frontends/shared/test_diagnostic_window.cpp)
  target_include_directories(test_diagnostic_window PRIVATE frontends/shared)
  add_test(NAME diagnostic_window COMMAND test_diagnostic_window)

  # boot_trace periodic BMPs are opt-in, with the legacy disable taking priority. This exercises the
  # exact environment-name lookup and registrar forwarding without constructing Vulkan or writing files.
  add_executable(test_frame_dump_policy frontends/shared/test_frame_dump_policy.cpp)
  target_include_directories(test_frame_dump_policy PRIVATE frontends/shared)
  add_test(NAME frame_dump_policy COMMAND test_frame_dump_policy)

  add_executable(test_capture_compute_policy tests/test_capture_compute_policy.cpp)
  target_include_directories(test_capture_compute_policy PRIVATE src)
  add_test(NAME capture_compute_policy COMMAND test_capture_compute_policy)

  add_executable(test_write_watch_policy frontends/shared/test_write_watch_policy.cpp)
  target_include_directories(test_write_watch_policy PRIVATE frontends/shared)
  add_test(NAME write_watch_policy COMMAND test_write_watch_policy)

  add_executable(test_texture_decode_diagnostic
                 frontends/shared/test_texture_decode_diagnostic.cpp)
  target_include_directories(test_texture_decode_diagnostic PRIVATE frontends/shared src)
  target_link_libraries(test_texture_decode_diagnostic PRIVATE prosper_core)
  add_test(NAME texture_decode_diagnostic COMMAND test_texture_decode_diagnostic)

  # Bounded-acquire present classification (#1182). Needs Vulkan HEADERS only (VkResult enum), no loader
  # call — link Vulkan::Vulkan purely for its include dirs. Guarded so a headers-less host skips it
  # gracefully; every platform that builds prosper-app has Vulkan available.
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(TARGET Vulkan::Vulkan)
    add_executable(test_prosper_app_present_policy frontends/prosper-app/test_present_policy.cpp)
    target_include_directories(test_prosper_app_present_policy PRIVATE frontends/prosper-app)
    target_link_libraries(test_prosper_app_present_policy PRIVATE Vulkan::Vulkan)
    add_test(NAME prosper_app_present_policy COMMAND test_prosper_app_present_policy)

    add_executable(test_present_blit_policy frontends/shared/test_present_blit_policy.cpp)
    target_include_directories(test_present_blit_policy PRIVATE frontends/shared)
    target_link_libraries(test_present_blit_policy PRIVATE Vulkan::Vulkan)
    add_test(NAME present_blit_policy COMMAND test_present_blit_policy)
  endif()

  add_executable(test_sync_on_address tests/test_sync_on_address.cpp)
  target_link_libraries(test_sync_on_address PRIVATE prosper_core)
  add_test(NAME sync_on_address COMMAND test_sync_on_address)

  # PROSPER_SYNC_RING is sized from the environment at static init, so the ring size must be set
  # for the process rather than by the test body. 64 is deliberately tiny: the property under test is
  # what the dump reports once it has DROPPED events.
  add_executable(test_sync_ring tests/test_sync_ring.cpp)
  target_link_libraries(test_sync_ring PRIVATE prosper_core)
  add_test(NAME sync_ring COMMAND test_sync_ring)
  set_tests_properties(sync_ring PROPERTIES ENVIRONMENT "PROSPER_SYNC_RING=64")

  add_executable(test_cond_slot_identity tests/test_cond_slot_identity.cpp)
  target_link_libraries(test_cond_slot_identity PRIVATE prosper_core)
  add_test(NAME cond_slot_identity COMMAND test_cond_slot_identity)

  # sceKernelDlsym honors its module handle (#147): per-module export tables, real handles from
  # LoadStartModule for linked-module paths, global first-wins fallback. Pure, no game dump needed.
  add_executable(test_dlsym_handle tests/test_dlsym_handle.cpp)
  target_link_libraries(test_dlsym_handle PRIVATE prosper_core)
  add_test(NAME dlsym_handle COMMAND test_dlsym_handle)

  # Guest mutex TYPE semantics (#145): settype honored, explicit Sony/FreeBSD ERRORCHECK default,
  # non-recursive static-sentinel self-init, FreeBSD errno values (EDEADLK=11). Pure, no game dump.
  add_executable(test_mutex_types tests/test_mutex_types.cpp)
  target_link_libraries(test_mutex_types PRIVATE prosper_core)
  add_test(NAME mutex_types COMMAND test_mutex_types)

  add_executable(test_agc_shader tests/test_agc_shader.cpp)
  target_link_libraries(test_agc_shader PRIVATE prosper_core)
  add_test(NAME agc_shader_create COMMAND test_agc_shader)

  add_executable(test_tile tests/test_tile.cpp)
  target_link_libraries(test_tile PRIVATE prosper_core)
  add_test(NAME gpu_surface_detile COMMAND test_tile)
  # Run the same byte-identity suite through the row-parallel detile path (#1177), forcing threads so
  # it is exercised even on a low-core CI. Passing here + in gpu_surface_detile (single-threaded control
  # via the < 2-thread gate / PROSPER_DETILE_SINGLE_THREADED) proves threaded == single byte-for-byte.
  add_test(NAME gpu_surface_detile_threaded COMMAND test_tile)
  set_tests_properties(gpu_surface_detile_threaded PROPERTIES
                       ENVIRONMENT "PROSPER_DETILE_THREADS=8")

  # BC1/BC2/BC3 block decompression to RGBA8 (#121): hand-built blocks with known output. Pure.
  add_executable(test_bc_decode tests/test_bc_decode.cpp)
  target_link_libraries(test_bc_decode PRIVATE prosper_core)
  add_test(NAME bc_decode COMMAND test_bc_decode)

  add_executable(test_stat tests/test_stat.cpp)
  target_link_libraries(test_stat PRIVATE prosper_core)
  add_test(NAME sce_stat_layout COMMAND test_stat)

  add_executable(test_file_binary tests/test_file_binary.cpp)
  target_link_libraries(test_file_binary PRIVATE prosper_core)
  add_test(NAME file_binary_roundtrip COMMAND test_file_binary)

  add_executable(test_savedata_dirname tests/test_savedata_dirname.cpp)
  target_link_libraries(test_savedata_dirname PRIVATE prosper_core)
  add_test(NAME savedata_dirname_guard COMMAND test_savedata_dirname)

  add_executable(test_savedata_txres tests/test_savedata_txres.cpp)
  target_link_libraries(test_savedata_txres PRIVATE prosper_core)
  add_test(NAME savedata_tx_resource_ids COMMAND test_savedata_txres)

  add_executable(test_translate_sandbox tests/test_translate_sandbox.cpp)
  target_link_libraries(test_translate_sandbox PRIVATE prosper_core)
  add_test(NAME translate_sandbox_guard COMMAND test_translate_sandbox)

  add_executable(test_service_logging tests/test_service_logging.cpp)
  target_link_libraries(test_service_logging PRIVATE prosper_core)
  add_test(NAME service_logging_unmapped_args COMMAND test_service_logging)

  if(WIN32)
    add_executable(test_win_exception_delivery tests/test_win_exception_delivery.cpp)
    target_link_libraries(test_win_exception_delivery PRIVATE prosper_core)
    add_test(NAME win_exception_delivery COMMAND test_win_exception_delivery)

    add_executable(test_win_pthread_key_destructor tests/test_win_pthread_key_destructor.cpp)
    target_link_libraries(test_win_pthread_key_destructor PRIVATE prosper_core)
    add_test(NAME win_pthread_key_destructor COMMAND test_win_pthread_key_destructor)

    add_executable(test_win_kernel_is_stack tests/test_win_kernel_is_stack.cpp)
    target_link_libraries(test_win_kernel_is_stack PRIVATE prosper_core)
    add_test(NAME win_kernel_is_stack COMMAND test_win_kernel_is_stack)
  endif()

  # Boot preload path case correction (#1006): a module present with different casing than the
  # hard-coded preload path (Blasphemous 2 / Evergate's Il2CppUserAssemblies.prx on a case-sensitive
  # host filesystem) must resolve to the real on-disk entry instead of being dropped as absent.
  # Pure std::filesystem against a temp tree; no game dump needed.
  add_executable(test_module_path_case tests/test_module_path_case.cpp)
  target_link_libraries(test_module_path_case PRIVATE prosper_core)
  add_test(NAME module_path_case COMMAND test_module_path_case)

  # Unity native-plugin auto-link (#1609): a title's own Media/Plugins/*.prx that the fixed preload
  # list never named must still be linked, or its first P/Invoke raises a DllNotFoundException that
  # IL2CPP captures into a Task and never prints. Also guards the export-collision rule that keeps
  # auto-linking from introducing a duplicate NID export (Evergate ships libfmod.prx AND libfmodL.prx,
  # the same library's release and logging builds). The discovery and predicate assertions are pure
  # std::filesystem/struct work and need no dump; the dump, when configured, additionally exercises
  # link_program's honouring of the skip flag.
  add_executable(test_plugin_autolink tests/test_plugin_autolink.cpp)
  target_link_libraries(test_plugin_autolink PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME plugin_autolink COMMAND test_plugin_autolink "${GAME_DUMP}")
    set_tests_properties(plugin_autolink PROPERTIES SKIP_RETURN_CODE 77)
  else()
    add_test(NAME plugin_autolink COMMAND test_plugin_autolink)
    set_tests_properties(plugin_autolink PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # ATRAC9 decode (vendored LibAtrac9 + prosper glue): an embedded 16-superframe slice of a real PS5
  # snd0.at9 decodes bit-exact vs the LibAtrac9 reference (CRC32). Pure, no game dump; runs in CI.
  add_executable(test_atrac9 tests/test_atrac9.cpp)
  target_link_libraries(test_atrac9 PRIVATE prosper_core)
  add_test(NAME atrac9_decode COMMAND test_atrac9)

  # AJM batch-2.0 gapless decode semantics (#1097): skip/total trim, exact EOS landing of
  # uiTotalDecodedSamples, post-EOS silence, and program re-arm. FMOD's codec recycling keys off
  # the exact EOS; without it one-shot SFX exhaust the codec pool. Pure, no game dump; runs in CI.
  add_executable(test_ajm2_gapless tests/test_ajm2_gapless.cpp)
  target_link_libraries(test_ajm2_gapless PRIVATE prosper_core)
  add_test(NAME ajm2_gapless COMMAND test_ajm2_gapless)

  # APR container registry + read-source resolution (#62/#1901): stable ids, path dedup, ID-first
  # reads, unique-size fallback, ambiguous fallback refusal, and method diagnostics. Pure; no dump.
  add_executable(test_apr_registry tests/test_apr_registry.cpp)
  target_link_libraries(test_apr_registry PRIVATE prosper_core)
  add_test(NAME apr_registry_resolution COMMAND test_apr_registry)

  # GTA V (#1139/#1142) net-new HLE regression guards: sceKernelAprGetFileStat fills the real stat
  # (#1133) and sceAudioOut2GetSpeakerArrayMemorySize returns a non-zero allocation size (#1134). The
  # AGC GetSize third piece is covered separately by test_agc_getsize (#1208).
  add_executable(test_apr_stat_audio tests/test_apr_stat_audio.cpp)
  target_link_libraries(test_apr_stat_audio PRIVATE prosper_core)
  add_test(NAME apr_stat_audio_gtav COMMAND test_apr_stat_audio)

  add_executable(test_ampr_measure tests/test_ampr_measure.cpp)
  target_link_libraries(test_ampr_measure PRIVATE prosper_core)
  add_test(NAME ampr_measure COMMAND test_ampr_measure)

  # sceAmprCommandBufferGetSize must not answer for one command buffer with another's recorded
  # capacity: a too-small answer permanently hangs the guest's IoStore append loop, which polls
  # GetSize(cb) - GetCurrentOffset(cb) > 0xff without a bound (The Forgotten City, PPSA03026).
  add_executable(test_ampr_getsize_fallback tests/test_ampr_getsize_fallback.cpp)
  target_link_libraries(test_ampr_getsize_fallback PRIVATE prosper_core)
  add_test(NAME ampr_getsize_fallback COMMAND test_ampr_getsize_fallback)

  # Direct-memory allocate/available (issue #99): AvailableDirectMemorySize reports the largest
  # free aligned block in the search window and writes both out-params (was a total-size stub).
  add_executable(test_dmem tests/test_dmem.cpp)
  target_link_libraries(test_dmem PRIVATE prosper_core)
  add_test(NAME dmem_available COMMAND test_dmem)

  # PROSPER_DMEM_CALLER stack walk (issue #1755): the diagnostic scans up to 160 slots above its
  # frame and used to read straight off the top of a thread stack, dying with a SIGSEGV that reads
  # as a guest fault.
  add_executable(test_dmem_caller_walk tests/test_dmem_caller_walk.cpp)
  target_link_libraries(test_dmem_caller_walk PRIVATE prosper_core)
  add_test(NAME dmem_caller_walk COMMAND test_dmem_caller_walk)

  # PROSPER_DMEM_CALLER correlation (issue #1859): the full stack remains one bounded line per
  # caller identity while every MEMLOG allocation can carry the same stable run-local ID. Pure,
  # including explicit unknown/overflow states and concurrent interning.
  add_executable(test_dmem_caller_chain tests/test_dmem_caller_chain.cpp)
  add_test(NAME dmem_caller_chain COMMAND test_dmem_caller_chain)

  # Capped-diagnostic rate limit (issue #1761): a print budget must never be readable as a
  # population. Pins the sparse tail and the per-key budget; header-only, no GPU state.
  add_executable(test_diag_ratelimit tests/test_diag_ratelimit.cpp)
  add_test(NAME diag_ratelimit COMMAND test_diag_ratelimit)

  # IME keyboard event delivery (issue #1093): sceImeUpdate must invoke the guest handler
  # per queued key event; the old stub delivered none.
  add_executable(test_ime_input tests/test_ime_input.cpp)
  target_link_libraries(test_ime_input PRIVATE prosper_core)
  add_test(NAME ime_input COMMAND test_ime_input)

  # Huge non-fixed sceKernelReserveVirtualRange placement (#312/#946): the 512 GiB UE arena
  # must be steered off its low hint into the guest auto window (see test for the contract).
  add_executable(test_reserve_placement tests/test_reserve_placement.cpp)
  target_link_libraries(test_reserve_placement PRIVATE prosper_core)
  add_test(NAME reserve_placement COMMAND test_reserve_placement)

  # A protection-only change must retain an uncommitted reservation's state (#343), or guest
  # VirtualQuery skips the later commit and the first access faults. Cross-platform HLE contract.
  add_executable(test_retrack_reservation tests/test_retrack_reservation.cpp)
  target_link_libraries(test_retrack_reservation PRIVATE prosper_core)
  add_test(NAME retrack_reservation COMMAND test_retrack_reservation)

  # Hinted map must not clobber a live mapping (#137): MAP_FIXED_NOREPLACE unless the hint is our
  # own uncommitted reservation. Drives the real map HLE handlers. Linux-only (mmap semantics).
  if(UNIX)
    add_executable(test_map_noreplace tests/test_map_noreplace.cpp)
    target_link_libraries(test_map_noreplace PRIVATE prosper_core)
    add_test(NAME map_noreplace COMMAND test_map_noreplace)

    # A guest mprotect(prot=0) makes the page PROT_NONE, not a default RW (#342): guard pages must
    # actually guard. Drives the real sceKernelMprotect + a SIGSEGV probe. Linux-only.
    add_executable(test_prot_none tests/test_prot_none.cpp)
    target_link_libraries(test_prot_none PRIVATE prosper_core)
    add_test(NAME prot_none COMMAND test_prot_none)
  endif()

  # PM4 command-stream decoder (front of the CommandProcessor): decode a Dcb built by the AGC
  # Dcb functions and assert the ops. Pure/cross-platform, no game dump needed.
  add_executable(test_pm4_decode tests/test_pm4_decode.cpp)
  target_link_libraries(test_pm4_decode PRIVATE prosper_core)
  add_test(NAME pm4_decode_stream COMMAND test_pm4_decode)

  # CommandProcessor apply stage: replay a decoded stream into a GpuState (register files + draws).
  add_executable(test_command_processor tests/test_command_processor.cpp)
  target_link_libraries(test_command_processor PRIVATE prosper_core)
  add_test(NAME command_processor_apply COMMAND test_command_processor)

  add_executable(test_command_provenance tests/test_command_provenance.cpp)
  target_link_libraries(test_command_provenance PRIVATE prosper_core)
  add_test(NAME command_provenance COMMAND test_command_provenance)

  # Stage B: CommandProcessor honors RELEASE_MEM/WRITE_DATA memory writes (synchronous end-of-pipe).
  add_executable(test_eop_write tests/test_eop_write.cpp)
  target_link_libraries(test_eop_write PRIVATE prosper_core)
  add_test(NAME eop_write COMMAND test_eop_write)

  # #1945: the interior free-chain poison scan — its structural predicate, and the census that keeps
  # a zero from reading as "the chains are clean" when the walk reached nothing. POSIX only: the test
  # needs a MAP_FIXED anonymous mapping inside the guest VA window to build a pool array with real
  # addresses, and mingw-w64 has no <sys/mman.h>. The code under test is built on every platform.
  if(NOT WIN32)
    add_executable(test_mb3_poison_scan tests/test_mb3_poison_scan.cpp)
    target_link_libraries(test_mb3_poison_scan PRIVATE prosper_core)
    add_test(NAME mb3_poison_scan COMMAND test_mb3_poison_scan)
    # A host that cannot place the mapping must SKIP, not pass: the structural half of the test
    # would otherwise be silently absent from a green run.
    set_tests_properties(mb3_poison_scan PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # #729: the synchronous (PROSPER_EOP_WRITE_SYNC=1) honor_* label writes must probe
  # guest_readable before dereferencing a guest-PM4-supplied address (own executable: the
  # env var latches process-wide on first use).
  add_executable(test_eop_write_sync_guard tests/test_eop_write_sync_guard.cpp)
  target_link_libraries(test_eop_write_sync_guard PRIVATE prosper_core)
  add_test(NAME eop_write_sync_guard COMMAND test_eop_write_sync_guard)

  # #312 per-queue WAIT_REG_MEM barrier model: unsatisfied waits gate downstream memory effects,
  # streams release independently, jumps don't un-gate, timeout backstop preserves liveness.
  add_executable(test_wait_barrier tests/test_wait_barrier.cpp)
  target_link_libraries(test_wait_barrier PRIVATE prosper_core)
  add_test(NAME wait_barrier COMMAND test_wait_barrier)

  # Render-state extraction: fold a stream into a GpuState, then interpret RDNA2 registers.
  add_executable(test_render_state tests/test_render_state.cpp)
  target_link_libraries(test_render_state PRIVATE prosper_core)
  add_test(NAME render_state_extract COMMAND test_render_state)

  # Messenger recompile guard (#228): committed real Messenger VS/PS RDNA2 binaries must keep
  # recompiling — a recompiler change that re-rejects one of their instructions (the #199/#201
  # blue-screen class) fails here instead of shipping. Pure (no Vulkan); needs no game dump.
  add_executable(test_messenger_recompile_guard tests/test_messenger_recompile_guard.cpp)
  target_link_libraries(test_messenger_recompile_guard PRIVATE prosper_core)
  target_compile_definitions(test_messenger_recompile_guard PRIVATE
      PROSPER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME messenger_recompile_guard COMMAND test_messenger_recompile_guard)

  # libkernel/libScePosix signal + time-conversion surface (#190): NID match, registration, and real
  # behavior (UTC<->localtime round-trip fills its out-param; pthread_setcancelstate old-state + EINVAL).
  add_executable(test_kernel_signal_time tests/test_kernel_signal_time.cpp)
  target_link_libraries(test_kernel_signal_time PRIVATE prosper_core)
  add_test(NAME kernel_signal_time COMMAND test_kernel_signal_time)

  # Pipeline-state resolution: RenderState (RDNA2 enums) -> Vulkan-ready pipeline state (pure, no Vulkan).
  add_executable(test_pipeline_state tests/test_pipeline_state.cpp)
  target_link_libraries(test_pipeline_state PRIVATE prosper_core)
  add_test(NAME pipeline_state_resolve COMMAND test_pipeline_state)

  # Recompiler coverage metric (pure): per-instruction support report used by shader_histo.
  add_executable(test_recompile_coverage tests/test_recompile_coverage.cpp)
  target_link_libraries(test_recompile_coverage PRIVATE prosper_core)
  add_test(NAME recompile_coverage COMMAND test_recompile_coverage)

  # Bounded graphics-shader cache: cached modules must match the direct recompiler byte-for-byte.
  add_executable(test_shader_recompile_cache tests/test_shader_recompile_cache.cpp)
  target_link_libraries(test_shader_recompile_cache PRIVATE prosper_core)
  add_test(NAME shader_recompile_cache COMMAND test_shader_recompile_cache)

  # RDNA2 shader instruction-stream walker (first stage of the RDNA2->SPIR-V recompiler).
  add_executable(test_rdna2_decode tests/test_rdna2_decode.cpp)
  target_link_libraries(test_rdna2_decode PRIVATE prosper_core)
  add_test(NAME rdna2_decode_walk COMMAND test_rdna2_decode)

  # Structural RDNA2->SPIR-V validation that does not require Vulkan runtime support.
  add_executable(test_rdna2_spirv_struct tests/test_rdna2_spirv_struct.cpp)
  target_link_libraries(test_rdna2_spirv_struct PRIVATE prosper_core)
  add_test(NAME rdna2_spirv_struct COMMAND test_rdna2_spirv_struct)

  # Scalar const-fold interpreter behind bindless-dynamic vertex-fetch resolution (no Vulkan needed).
  add_executable(test_dynfetch_fold tests/test_dynfetch_fold.cpp)
  target_link_libraries(test_dynfetch_fold PRIVATE prosper_core)
  add_test(NAME dynfetch_fold COMMAND test_dynfetch_fold)

  # Cross-submit readability reuse is valid only while the HLE mapping generation is unchanged.
  add_executable(test_guest_memory_map tests/test_guest_memory_map.cpp)
  target_link_libraries(test_guest_memory_map PRIVATE prosper_core)
  add_test(NAME guest_memory_map COMMAND test_guest_memory_map)

  add_executable(test_guest_read_cache_disabled tests/test_guest_read_cache_disabled.cpp)
  target_link_libraries(test_guest_read_cache_disabled PRIVATE prosper_core)
  add_test(NAME guest_read_cache_disabled COMMAND test_guest_read_cache_disabled)

  if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_guest_write_watch tests/test_guest_write_watch.cpp)
    target_link_libraries(test_guest_write_watch PRIVATE prosper_core)
    add_test(NAME guest_write_watch COMMAND test_guest_write_watch)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_dmem_write_trace tests/test_dmem_write_trace.cpp)
    target_link_libraries(test_dmem_write_trace PRIVATE prosper_core)
    add_test(NAME dmem_write_trace COMMAND test_dmem_write_trace)
  endif()

  # Unannounced-32-bit index-buffer fingerprint (#304 — DOLL's Slate/UMG quads). No Vulkan needed.
  add_executable(test_index_size_detect tests/test_index_size_detect.cpp)
  target_link_libraries(test_index_size_detect PRIVATE prosper_core)
  add_test(NAME index_size_detect COMMAND test_index_size_detect)

  # shader_histo: data-driven recompiler-coverage report over the game's real embedded RDNA2 shaders.
  add_executable(shader_histo tools/shader_histo/shader_histo.cpp)
  target_link_libraries(shader_histo PRIVATE prosper_core)

  # shader_inspect: bounded offline decode/CFG listing for raw PROSPER_SHADER_DUMP binaries.
  add_executable(shader_inspect tools/shader_inspect/shader_inspect.cpp)
  target_link_libraries(shader_inspect PRIVATE prosper_core)

  # Resource-table honesty (#1571): a raw dump carries no descriptors, so a table-less stage
  # rejection must be reported as an UNDETERMINED tool limitation, never as a shader defect.
  if(Python3_Interpreter_FOUND)
    add_test(NAME shader_inspect_resource_table_honesty
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_inspect/test_shader_inspect.py"
                     "$<TARGET_FILE:shader_inspect>")
  endif()

  # spv_validate: emit a module from EVERY SPIR-V-producing entry point and run spirv-val on each —
  # a strict-validation gate (the render tests only prove llvmpipe accepts them; llvmpipe is
  # lenient). No Vulkan needed, but spirv-val (spirv-tools) IS required: it used to report PASS when
  # the validator was absent, which is how #1711 shipped invalid SPIR-V past a gate CLAUDE.md and
  # both READMEs describe as strict. The source root is passed so the tool can read the emitter
  # headers and fail when a new emitter is added without a module here.
  add_executable(spv_validate tools/spv_validate/spv_validate.cpp)
  target_link_libraries(spv_validate PRIVATE prosper_core)
  add_test(NAME spv_validate
           COMMAND spv_validate ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  # AGC→Vulkan resource layer: the front-half↔back-half contract (gpu_resources.hpp). Registry
  # behavior is fixed here so both halves integrate against a stable API. Pure/cross-platform.
  add_executable(test_gpu_resources tests/test_gpu_resources.cpp)
  target_link_libraries(test_gpu_resources PRIVATE prosper_core)
  add_test(NAME gpu_resource_registry COMMAND test_gpu_resources)

  # Resource-binding contract (shader_resources.hpp): format sizing + the provenance/binding lookups
  # the recompiler and pipeline share. Pure/cross-platform. See docs/RESOURCE_BINDING.md.
  add_executable(test_shader_resources tests/test_shader_resources.cpp)
  target_link_libraries(test_shader_resources PRIVATE prosper_core)
  add_test(NAME shader_resource_contract COMMAND test_shader_resources)

  # Versioned realized-draw capture: explicit round-trip, overlapping-resource alias preservation,
  # owned replay backing, and corrupt/truncated input rejection. Pure; no Vulkan or game dump.
  add_executable(test_gpu_capture tests/test_gpu_capture.cpp)
  target_link_libraries(test_gpu_capture PRIVATE prosper_core)
  add_test(NAME gpu_capture_roundtrip COMMAND test_gpu_capture)

  add_executable(test_gpu_capture_bundle tests/test_gpu_capture_bundle.cpp)
  target_link_libraries(test_gpu_capture_bundle PRIVATE prosper_core)
  add_test(NAME gpu_capture_bundle_roundtrip COMMAND test_gpu_capture_bundle)

  # Native-speed append-only submit/present timeline: portable framing, checksums, and truncated-tail
  # recovery are pure and must not depend on Vulkan or a game dump.
  add_executable(test_gpu_timeline tests/test_gpu_timeline.cpp)
  target_link_libraries(test_gpu_timeline PRIVATE prosper_core)
  add_test(NAME gpu_timeline_roundtrip COMMAND test_gpu_timeline)

  add_executable(test_gpu_timeline_capture_exit tests/test_gpu_timeline_capture_exit.cpp)
  target_link_libraries(test_gpu_timeline_capture_exit PRIVATE prosper_core)
  add_test(NAME gpu_timeline_capture_exit COMMAND test_gpu_timeline_capture_exit)

  # Generic guest-log phase gate: exact bounded lines arm the existing seeded whole-frame capture
  # after one completed present. Pure/cross-platform; no game dump or Vulkan device.
  add_executable(test_guest_log_capture tests/test_guest_log_capture.cpp)
  target_link_libraries(test_guest_log_capture PRIVATE prosper_core)
  add_test(NAME guest_log_capture COMMAND test_guest_log_capture)

  # Remote/headless F9 equivalent: an external regular-file witness arms the existing seeded
  # whole-frame capture on the next present. Pure/cross-platform; no game dump or Vulkan device.
  add_executable(test_capture_trigger_file tests/test_capture_trigger_file.cpp)
  target_link_libraries(test_capture_trigger_file PRIVATE prosper_core)
  add_test(NAME capture_trigger_file COMMAND test_capture_trigger_file)

  # gpu_replay output dimensions follow the selected draw/prefix target when its byte count agrees.
  # Pure/cross-platform so focused replay tooling remains available in CI.
  add_executable(test_gpu_replay_extent tests/test_gpu_replay_extent.cpp)
  target_link_libraries(test_gpu_replay_extent PRIVATE prosper_core)
  add_test(NAME gpu_replay_output_extent COMMAND test_gpu_replay_extent)

  add_executable(test_gpu_replay_shader_dump tests/test_gpu_replay_shader_dump.cpp)
  target_link_libraries(test_gpu_replay_shader_dump PRIVATE prosper_core)
  add_test(NAME gpu_replay_shader_dump COMMAND test_gpu_replay_shader_dump)

  # Draw-scoped captured-resource replacement: strict semantic selector, exact span validation,
  # and copy-on-write isolation when draws share tables/backing. Pure/cross-platform; no Vulkan.
  add_executable(test_gpu_replay_resource_override
                 tests/test_gpu_replay_resource_override.cpp)
  target_link_libraries(test_gpu_replay_resource_override PRIVATE prosper_core)
  add_test(NAME gpu_replay_resource_override COMMAND test_gpu_replay_resource_override)

  # Selected-prefix compute output inspection must distinguish captured seed bytes from the
  # descriptor-visible post-dispatch image, including tiled 2D and true 3D storage layouts.
  add_executable(test_gpu_replay_post_compute_resource
                 tests/test_gpu_replay_post_compute_resource.cpp)
  target_link_libraries(test_gpu_replay_post_compute_resource PRIVATE prosper_core)
  add_test(NAME gpu_replay_post_compute_resource COMMAND test_gpu_replay_post_compute_resource)

  # Offline SPI_PS_INPUT_CNTL linkage classification: a pixel-shader input wired to the constant
  # form consumes no producer export, and must never be reported as a PARAM consumer.
  add_executable(test_gpu_replay_pixel_inputs tests/test_gpu_replay_pixel_inputs.cpp)
  target_link_libraries(test_gpu_replay_pixel_inputs PRIVATE prosper_core)
  add_test(NAME gpu_replay_pixel_inputs COMMAND test_gpu_replay_pixel_inputs)

  add_executable(test_gpu_dependency_graph tests/test_gpu_dependency_graph.cpp)
  target_link_libraries(test_gpu_dependency_graph PRIVATE prosper_core)
  add_test(NAME gpu_dependency_graph COMMAND test_gpu_dependency_graph)

  add_executable(gpu_timeline tools/gpu_timeline/gpu_timeline.cpp)
  target_link_libraries(gpu_timeline PRIVATE prosper_core)

  # Front-half resource-table builder + V# descriptor decode (agc_shader_layout.cpp): shader user_data
  # + user-data SGPRs -> ShaderResourceTable (constant buffers, stage 1). Pure/cross-platform.
  add_executable(test_build_shader_resources tests/test_build_shader_resources.cpp)
  target_link_libraries(test_build_shader_resources PRIVATE prosper_core)
  add_test(NAME build_shader_resources COMMAND test_build_shader_resources)

  # GPU descriptor-bound INVARIANT guard (agc_shader_layout.cpp decode_image/buffer_descriptor): the
  # entire guest-input memory-safety argument relies on width/height <= 16384, depth <= 8192, and V#
  # size_bytes clamped to 32 bits. Pins those exact ceilings so a mask-widening / +1-bias refactor fails
  # here first instead of silently un-bounding every downstream size product. Pure decode; no dump/Vulkan.
  add_executable(test_descriptor_clamp tests/test_descriptor_clamp.cpp)
  target_link_libraries(test_descriptor_clamp PRIVATE prosper_core)
  add_test(NAME descriptor_clamp COMMAND test_descriptor_clamp)

  # Descriptor-binding policy (#157): textures/storage images never land on the recompiler's hardwired
  # cbuf bindings 2/3; constant/vertex buffers assigned first. Pure, no game dump / Vulkan needed.
  add_executable(test_convention_bindings tests/test_convention_bindings.cpp)
  target_link_libraries(test_convention_bindings PRIVATE prosper_core)
  add_test(NAME convention_bindings COMMAND test_convention_bindings)

  add_executable(test_writer_provenance tests/test_writer_provenance.cpp)
  target_link_libraries(test_writer_provenance PRIVATE prosper_core)
  add_test(NAME writer_provenance_ranges COMMAND test_writer_provenance)

  # The init-fault report must survive a wild/null faulting RIP (#128). On Windows this also drives
  # the VEH recovery thunk and asserts its compiled-helper call obeys the Microsoft-x64 ABI (#633).
  add_executable(test_initfault_dump tests/test_initfault_dump.cpp)
  target_link_libraries(test_initfault_dump PRIVATE prosper_core)
  add_test(NAME initfault_dump_survives COMMAND test_initfault_dump)

  # Generated guest->host import boundary: preserve all register args, stack args 7-9, return value,
  # and ABI stack alignment. Windows exercises SysV->Microsoft-x64; Linux also drives the FS-swap path.
  add_executable(test_hle_stack_args tests/test_hle_stack_args.cpp)
  target_link_libraries(test_hle_stack_args PRIVATE prosper_core)
  add_test(NAME hle_stack_args COMMAND test_hle_stack_args)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME hle_stack_args_guest_fs COMMAND test_hle_stack_args --guest-fs)
  endif()

  # sceImeUpdate must invoke the guest key handler on the CALLER's guest %fs, not the host %fs the
  # import stub swapped in (#1286: Evergate SIGSEGV at eboot+0xaf8431 on a key press). Drives the real
  # import path on an activated guest TCB; a no-op pass off Linux.
  add_executable(test_ime_fs tests/test_ime_fs.cpp)
  target_link_libraries(test_ime_fs PRIVATE prosper_core)
  add_test(NAME ime_fs COMMAND test_ime_fs)

  # x86-64 low-address READ decoder used by the Windows PROSPER_NULL_PAGE emulation. Header-only, so the
  # decode logic is verified on every platform even though the VEH caller is Windows-only.
  add_executable(test_x86_read_decode tests/test_x86_read_decode.cpp)
  add_test(NAME x86_read_decode COMMAND test_x86_read_decode)

  # #2018: a fatal worker-thread fault report must describe one fault. Header-only snapshot of the
  # handler's own siginfo/ucontext; POSIX signal contexts only (the test self-skips elsewhere).
  add_executable(test_fault_context tests/test_fault_context.cpp)
  # The contention arm spawns real threads. This target links no library, so unlike every other
  # thread-using test here it inherits -pthread from nothing; name Threads::Threads explicitly
  # (found above with prosper_core, whose existence gates this block).
  target_link_libraries(test_fault_context PRIVATE Threads::Threads)
  add_test(NAME fault_context COMMAND test_fault_context)

  # End-to-end AGC submit plus fixed args 7-9 in ReleaseMem/WaitRegMem. This is cross-platform so
  # Windows validates the real fence builders behind the generated import-ABI conformance test.
  add_executable(test_agc_dcb_callback_abi tests/test_agc_dcb_callback_abi.cpp)
  target_link_libraries(test_agc_dcb_callback_abi PRIVATE prosper_core)
  add_test(NAME agc_dcb_callback_abi COMMAND test_agc_dcb_callback_abi)

  add_executable(test_agc_submit tests/test_agc_submit.cpp)
  target_link_libraries(test_agc_submit PRIVATE prosper_core)
  add_test(NAME agc_submit_to_gpustate COMMAND test_agc_submit)

  # Windows guest-execution substrate (exec_image_win.cpp): build boot_trace so the whole boot path
  # links (proves the substrate + SysV-ABI HLE boundary resolve). No evdev/Vulkan here. The guest boot
  # itself is not yet verified on Windows — see docs/PORTING.md "Windows".
  if(WIN32)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    # Live Vulkan renderer on Windows (PROSPER_RENDER=1): with the Vulkan SDK present, wire the same
    # shared DrawItem->Vulkan compositor boot_trace uses on Linux so the guest's SubmitDcb draws
    # execute + present + dump frames. Mirrors the UNIX block below (boot_trace + prosper_live_renderer
    # only; the extra replay/screenshot tools stay Linux-first for now). See docs/PORTING.md "Windows".
    if(NOT TARGET Vulkan::Vulkan)
      find_package(Vulkan QUIET)
    endif()
    if(Vulkan_FOUND)
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)
      add_library(prosper_live_renderer STATIC frontends/shared/live_renderer.cpp
                                               frontends/shared/live_compute.cpp
                                               frontends/shared/present_blit.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan
                                                   prosper_performance_capture)
      target_include_directories(prosper_live_renderer PUBLIC frontends/shared)
      # A LiveTargetPixelFormat mapping that quietly falls back to RGBA8 has now cost two titles a
      # whole render layer (#773 Dead Cells, and Syberia's black 3D menu). frontends/shared/
      # live_target_format.hpp maps the enum with default-less switches; -Werror=switch promotes a
      # member added without updating them from a silent wrong format to a build failure.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(prosper_live_renderer PRIVATE -Werror=switch)
      endif()
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # Keep the production compute backend covered on native Windows too. This catches host- and
      # driver-specific image upload/writeback failures in the same backend prosper-app uses.
      add_executable(test_game_compute tests/test_game_compute.cpp)
      target_link_libraries(test_game_compute PRIVATE prosper_live_renderer)
      add_test(NAME game_compute_exec COMMAND test_game_compute)
      add_test(NAME game_compute_exec_no_adaptive_storage_result COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_adaptive_storage_result PROPERTIES
        ENVIRONMENT "PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1")
    else()
      message(STATUS "Vulkan not found — Windows boot_trace built without the live renderer")
    endif()
  endif()

  # M2+: real memory mapping, trap, boot, and the boot-trace debug tool (Linux + macOS/Rosetta).
  if(UNIX)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    if(NOT APPLE)
      # Zero-dependency Linux evdev gamepad frontend (outside prosper_core). Always available on Linux;
      # boot_trace installs it (gated by PROSPER_PAD) when the SDL3 gamepad frontend isn't built.
      add_library(prosper_pad_evdev STATIC frontends/pad_evdev/pad_evdev.cpp)
      target_include_directories(prosper_pad_evdev PUBLIC frontends/pad_evdev src)
      target_link_libraries(prosper_pad_evdev PUBLIC prosper_core)
      target_link_libraries(boot_trace PRIVATE prosper_pad_evdev)
      target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_EVDEV=1)
    endif()

    add_executable(imgdump tools/imgdump/imgdump.cpp)
    target_link_libraries(imgdump PRIVATE prosper_core)

    add_executable(test_trap_linux tests/test_trap_linux.cpp)
    target_link_libraries(test_trap_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME trap_identifies_imports
               COMMAND test_trap_linux "${GAME_DUMP}")
    endif()

    add_executable(test_boot_linux tests/test_boot_linux.cpp)
    target_link_libraries(test_boot_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME boot_reaches_first_syscall
               COMMAND test_boot_linux "${GAME_DUMP}")
    endif()

    if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
      # Real runtime PRX loading (#639). Hermetic: the test writes its own synthetic PRX outside
      # Media/Plugins, so it needs no game dump — but it maps that module at the fixed runtime-module
      # base and EXECUTES its module_start, whose bytes the test emits by hand, so it is host-x86-64
      # only. Its scratch root goes in the build tree, never /tmp.
      add_executable(test_runtime_prx_load tests/test_runtime_prx_load.cpp)
      target_link_libraries(test_runtime_prx_load PRIVATE prosper_core)
      add_test(NAME runtime_prx_load
               COMMAND test_runtime_prx_load "${CMAKE_CURRENT_BINARY_DIR}/runtime-prx-test-root")
    endif()

    # Regression test for the real setjmp/longjmp impl (x86-64 asm; used by the GC's root scan).
    add_executable(test_setjmp tests/test_setjmp.cpp)
    target_link_libraries(test_setjmp PRIVATE prosper_core)
    add_test(NAME setjmp_saves_and_restores COMMAND test_setjmp)

    add_executable(test_stack_registry tests/test_stack_registry.cpp)
    target_link_libraries(test_stack_registry PRIVATE prosper_core)
    add_test(NAME thread_stack_registry COMMAND test_stack_registry)

    add_executable(test_rwlock_once tests/test_rwlock_once.cpp)
    target_link_libraries(test_rwlock_once PRIVATE prosper_core)
    add_test(NAME rwlock_once_semantics COMMAND test_rwlock_once)

    # #2042: a destroyed guest sync object's storage is quarantined, not returned to the allocator,
    # so a Destroy racing a handler cannot corrupt prosper's own heap. The assertions hold on every
    # allocator, so the test itself runs everywhere.
    add_executable(test_sync_destroy_lifetime tests/test_sync_destroy_lifetime.cpp)
    target_link_libraries(test_sync_destroy_lifetime PRIVATE prosper_core)
    add_test(NAME sync_destroy_lifetime COMMAND test_sync_destroy_lifetime)
    set_tests_properties(sync_destroy_lifetime PROPERTIES TIMEOUT 120)

    # The window switch must FAIL SAFE. `atof` returns 0.0 on no conversion, so an early version of
    # retire_window_seconds() turned the guard off for `=on`, `=true` or any typo — the exact
    # use-after-free this exists to prevent, reached by misspelling a tuning knob. This arm passes an
    # unparseable value and requires the suite to still PASS, i.e. the default window is in force.
    # It asserts the green outcome, so unlike the counter-arm it is allocator-independent.
    add_test(NAME sync_destroy_lifetime_bad_window COMMAND test_sync_destroy_lifetime)
    set_tests_properties(sync_destroy_lifetime_bad_window PROPERTIES
        ENVIRONMENT "PROSPER_SYNC_RETIRE_SECONDS=on" TIMEOUT 120)

    # The counter-arm runs the SAME binary with the quarantine switched off
    # (PROSPER_SYNC_RETIRE_SECONDS=0 is the pre-#2042 immediate free) and requires it to FAIL, so CI
    # keeps proving the assertions can move without anyone rebuilding against an older master.
    #
    # LINUX ONLY, and the restriction is the honest one rather than a convenience: the arm is red
    # because glibc's tcache hands a just-freed block of the same size back to the very next
    # allocation, deterministically and LIFO. That reuse is the mechanism being demonstrated, not an
    # incidental detail — on an allocator that does not reuse promptly the assertions would pass and
    # WILL_FAIL would turn a correct result into a red build. Asserting an allocator's internals as
    # a merge gate on hosts where they were never measured is how a guard starts failing for reasons
    # unrelated to what it guards.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      add_test(NAME sync_destroy_lifetime_counter_arm COMMAND test_sync_destroy_lifetime)
      set_tests_properties(sync_destroy_lifetime_counter_arm PROPERTIES
          ENVIRONMENT "PROSPER_SYNC_RETIRE_SECONDS=0" WILL_FAIL TRUE TIMEOUT 120)
    endif()
    # #2178: every fallible scePthread* spelling reports the libkernel-encoded error while the POSIX
    # spelling on the same body keeps the bare FreeBSD errno. Asserts both halves per entry point —
    # a single-spelling arm cannot see a missing (or a wrongly added) SCE_PTHREAD_ALIAS.
    add_executable(test_pthread_error_encoding tests/test_pthread_error_encoding.cpp)
    target_link_libraries(test_pthread_error_encoding PRIVATE prosper_core)
    add_test(NAME pthread_error_encoding COMMAND test_pthread_error_encoding)

    # __tls_get_addr per-thread DTV is purged (blocks freed) on both thread-exit paths (#68), and
    # normal return leaves guest %fs before glibc resumes host pthread cleanup (#644).
    add_executable(test_tls_dtv_purge tests/test_tls_dtv_purge.cpp)
    target_link_libraries(test_tls_dtv_purge PRIVATE prosper_core)
    add_test(NAME tls_dtv_purge_on_thread_exit COMMAND test_tls_dtv_purge)

    add_executable(test_agc_dcb tests/test_agc_dcb.cpp)
    target_link_libraries(test_agc_dcb PRIVATE prosper_core)
    add_test(NAME agc_dcb_pm4 COMMAND test_agc_dcb)

    # #1143: each sceAgc*GetSize must equal its builder's emitted dword count — drift guard.
    add_executable(test_agc_getsize tests/test_agc_getsize.cpp)
    target_link_libraries(test_agc_getsize PRIVATE prosper_core)
    add_test(NAME agc_getsize_drift COMMAND test_agc_getsize)

    # #1748: a builder must not emit MORE dwords than the hardware packet it stands for — the guest
    # reserves its command buffer from the real AGC sizes, so an oversized packet overruns it.
    add_executable(test_agc_epilogue_window tests/test_agc_epilogue_window.cpp)
    target_link_libraries(test_agc_epilogue_window PRIVATE prosper_core)
    add_test(NAME agc_epilogue_window COMMAND test_agc_epilogue_window)

    # libSceVideoOut: query/config functions return real 1080p60 values + swapchain-scaffolding
    # display-buffer registry records the game's registered framebuffers.
    add_executable(test_videoout tests/test_videoout.cpp)
    target_link_libraries(test_videoout PRIVATE prosper_core)
    add_test(NAME videoout_queries COMMAND test_videoout)

    # libSceVideoOut flip->present: SubmitFlip scans out the registered buffer; present_readback
    # returns the presented frame's real pixels. The surface the renderer presents frames to.
    add_executable(test_present tests/test_present.cpp)
    target_link_libraries(test_present PRIVATE prosper_core)
    add_test(NAME videoout_present COMMAND test_present)

    # Graphics-phase foundation + agentic-first verification harness: headless offscreen Vulkan
    # (render/clear -> readback -> pixel/CRC assert). Only built if Vulkan dev files are present.
    if(NOT TARGET Vulkan::Vulkan)
      find_package(Vulkan QUIET)
    endif()
    if(Vulkan_FOUND)
      # Live game rendering: with Vulkan available, boot_trace can register the live submit renderer
      # (PROSPER_RENDER=1) so the game's SubmitDcb draws execute + present + dump frames to disk.
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)

      # Shared live renderer (frontends/shared): the DrawItem->Vulkan compositor, factored out of
      # boot_trace so BOTH boot_trace and prosper-app register the SAME renderer instead of
      # duplicating it. Needs render_runner.h (tests/) and refvs.inc (tools/boot_trace/).
      add_library(prosper_live_renderer STATIC frontends/shared/live_renderer.cpp
                                               frontends/shared/live_compute.cpp
                                               frontends/shared/present_blit.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan
                                                   prosper_performance_capture)
      target_include_directories(prosper_live_renderer PUBLIC frontends/shared)
      # A LiveTargetPixelFormat mapping that quietly falls back to RGBA8 has now cost two titles a
      # whole render layer (#773 Dead Cells, and Syberia's black 3D menu). frontends/shared/
      # live_target_format.hpp maps the enum with default-less switches; -Werror=switch promotes a
      # member added without updating them from a silent wrong format to a build failure.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(prosper_live_renderer PRIVATE -Werror=switch)
      endif()
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # Local-only realized-submit replayer (#514): no guest boot, reads a versioned capture whose game
      # data remains gitignored, renders it through the same live backend, and checks the captured hash.
      add_executable(gpu_replay tools/gpu_replay/gpu_replay.cpp)
      target_include_directories(gpu_replay PRIVATE src tests)
      target_link_libraries(gpu_replay PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      # gpu_replay maps LiveTargetPixelFormat itself (inspect_live_target), and it is the tool an
      # investigation reaches for FIRST. An unhandled member there falls through to the
      # GpuCaptureRttSeed::format default (Rgba8Unorm) and makes --inspect-only mislabel the seed,
      # sending the next investigation down the wrong path using its own instrument. Same gate as
      # prosper_live_renderer so the exhaustiveness contract holds at every mapping, not just the
      # ones inside frontends/shared/live_target_format.hpp.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gpu_replay PRIVATE -Werror=switch)
      endif()

      # Seed flags are renderer-registration inputs. Exercise the real gpu_replay process so an
      # early --recompile-raw registration cannot pass by proving substitution but failing restore.
      add_executable(test_gpu_replay_seed_order tests/test_gpu_replay_seed_order.cpp)
      target_include_directories(test_gpu_replay_seed_order PRIVATE src tests)
      target_link_libraries(test_gpu_replay_seed_order PRIVATE prosper_core)
      add_test(NAME gpu_replay_seeded_raw_execution
        COMMAND test_gpu_replay_seed_order $<TARGET_FILE:gpu_replay>)

      # Exercise the real metadata-only CLI path for v46's selected raw resource-input witness.
      # Unit tests cover verdict policy; this process test keeps inspect formatting from silently
      # disappearing or collapsing full-match/mismatch/unavailable into one output state.
      add_test(NAME gpu_replay_resource_input_inspect
        COMMAND test_gpu_capture --inspect-resource-input $<TARGET_FILE:gpu_replay>)

      # Capture -> serialize -> owned replay backing -> SAME live renderer -> pixels. This specifically
      # proves replay textures do not dereference the captured guest VA and still sample exact bytes.
      add_executable(test_gpu_capture_render tests/test_gpu_capture_render.cpp)
      target_include_directories(test_gpu_capture_render PRIVATE src tests tools/boot_trace)
      target_link_libraries(test_gpu_capture_render PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME gpu_capture_render_replay COMMAND test_gpu_capture_render)

      add_executable(test_vulkan_offscreen tests/test_vulkan_offscreen.cpp)
      target_link_libraries(test_vulkan_offscreen PRIVATE Vulkan::Vulkan)
      add_test(NAME vulkan_offscreen_clear COMMAND test_vulkan_offscreen)

      # First real rendered output: a VkPipeline whose topology/format come from a RenderState via
      # vk_translate, drawing a triangle (embedded placeholder SPIR-V), verified by pixel readback.
      add_executable(test_vulkan_triangle tests/test_vulkan_triangle.cpp)
      target_link_libraries(test_vulkan_triangle PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vulkan_triangle_render COMMAND test_vulkan_triangle)

      # Execution-differential harness (verification layer 4): run a compute shader with known
      # inputs, assert numeric outputs. The RDNA2->SPIR-V recompiler's output will plug in here.
      add_executable(test_compute_exec tests/test_compute_exec.cpp)
      target_link_libraries(test_compute_exec PRIVATE Vulkan::Vulkan)
      add_test(NAME compute_exec_differential COMMAND test_compute_exec)

      # Execute a real game kernel with its user/system-register ABI and resource table.
      add_executable(test_game_compute tests/test_game_compute.cpp)
      target_link_libraries(test_game_compute PRIVATE prosper_live_renderer)
      add_test(NAME game_compute_exec COMMAND test_game_compute)
      add_test(NAME game_compute_exec_no_adaptive_storage_result COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_adaptive_storage_result PROPERTIES
        ENVIRONMENT "PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1")

      # Prove our own SPIR-V emitter produces valid, correct, runnable SPIR-V (execution-differential).
      add_executable(test_spirv_builder tests/test_spirv_builder.cpp)
      target_link_libraries(test_spirv_builder PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME spirv_builder_exec COMMAND test_spirv_builder)

      # End-to-end: recompile real RDNA2 (llvm-mc) -> SPIR-V -> run -> assert numeric result.
      add_executable(test_rdna2_to_spirv tests/test_rdna2_to_spirv.cpp)
      target_link_libraries(test_rdna2_to_spirv PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME rdna2_to_spirv_exec COMMAND test_rdna2_to_spirv)

      # Render a triangle whose PIXEL shader is recompiled from RDNA2 (EXP MRT0 -> color output).
      add_executable(test_recompiled_fragment tests/test_recompiled_fragment.cpp)
      target_link_libraries(test_recompiled_fragment PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME recompiled_fragment_render COMMAND test_recompiled_fragment)

      # Render a frame where BOTH vertex + pixel shaders are recompiled from RDNA2.
      add_executable(test_recompiled_shaders tests/test_recompiled_shaders.cpp)
      target_link_libraries(test_recompiled_shaders PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME recompiled_shaders_render COMMAND test_recompiled_shaders)

      # Vertex fetch: a VS that pulls positions from a bound vertex buffer via buffer_load_format,
      # keyed by a ShaderResourceTable — real game vertex shaders work this way.
      add_executable(test_vertex_fetch_render tests/test_vertex_fetch_render.cpp)
      target_link_libraries(test_vertex_fetch_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vertex_fetch_render COMMAND test_vertex_fetch_render)

      # Texture sampling: a recompiled pixel shader does MIMG image_sample against a bound 2x2 texture.
      add_executable(test_texture_sample_render tests/test_texture_sample_render.cpp)
      target_link_libraries(test_texture_sample_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME texture_sample_render COMMAND test_texture_sample_render)

      # Multi-draw backend: N draws composited into ONE cleared-once framebuffer (red opaque + green
      # additive -> yellow center proves accumulation into a single target).
      add_executable(test_multidraw_render tests/test_multidraw_render.cpp)
      target_link_libraries(test_multidraw_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME multidraw_render COMMAND test_multidraw_render)

      # Indexed draws (#64): vkCmdDrawIndexed from a BackendDraw's index data — the [0,1,2,2,3,0] quad
      # must match the retired perimeter-fan rendering pixel-exact, and indices must really be applied.
      add_executable(test_indexed_render tests/test_indexed_render.cpp)
      target_link_libraries(test_indexed_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME indexed_render COMMAND test_indexed_render)

      # Storage images: a recompiled compute kernel does MIMG image_load->image_store (no sampler) to
      # copy a 1D storage image bit-exact (the game's blit/copy compute-shader class).
      add_executable(test_storage_image_copy tests/test_storage_image_copy.cpp)
      target_link_libraries(test_storage_image_copy PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME storage_image_copy COMMAND test_storage_image_copy)

      # Interpolation: a recompiled VS exports EXP PARAM0, a recompiled PS reads it via v_interp.
      add_executable(test_interp_render tests/test_interp_render.cpp)
      target_link_libraries(test_interp_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME interp_render COMMAND test_interp_render)

      # Full textured-draw path: VS exports interpolated UVs, PS samples a texture at them (+screenshot).
      add_executable(test_textured_interp_render tests/test_textured_interp_render.cpp)
      target_link_libraries(test_textured_interp_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME textured_interp_render COMMAND test_textured_interp_render)

      # IMAGE_SAMPLE_C_LZ 2D (shadow-map compare, #1271): manual-dref lowering, end-to-end.
      add_executable(test_shadow_compare_render tests/test_shadow_compare_render.cpp)
      target_link_libraries(test_shadow_compare_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME shadow_compare_render COMMAND test_shadow_compare_render)

      # Generated mip chains bounded by the T#'s declared levels (#1272), end-to-end.
      add_executable(test_texture_mip_render tests/test_texture_mip_render.cpp)
      target_link_libraries(test_texture_mip_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME texture_mip_render COMMAND test_texture_mip_render)

      # Pipeline realization: build a VkPipeline whose fixed-function state comes from
      # resolve_pipeline_state(RenderState) and prove the resolved state drives rendering.
      add_executable(test_pipeline_render tests/test_pipeline_render.cpp)
      target_link_libraries(test_pipeline_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME pipeline_render COMMAND test_pipeline_render)

      # Full back-half spine: a GpuState -> recompile shaders from their PGM addresses -> resolve
      # pipeline state -> rendered frame (the path a real submitted Dcb will drive).
      add_executable(test_gpustate_render tests/test_gpustate_render.cpp)
      target_link_libraries(test_gpustate_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpustate_render COMMAND test_gpustate_render)

      # GPU-executor core (Stage A): GpuState -> execute_gpustate() -> render -> present round-trip.
      add_executable(test_gpu_execute tests/test_gpu_execute.cpp)
      target_link_libraries(test_gpu_execute PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpu_execute COMMAND test_gpu_execute)

      # Payoff demo: recompile the GAME'S OWN embedded shaders and render them on real Vulkan.
      # Needs both Vulkan and the (gitignored) game dump, so it's gated on HAVE_GAME_DUMP.
      if(HAVE_GAME_DUMP)
        add_executable(test_real_shader_render tests/test_real_shader_render.cpp)
        target_link_libraries(test_real_shader_render PRIVATE prosper_core Vulkan::Vulkan)
        add_test(NAME real_shader_render COMMAND test_real_shader_render "${GAME_DUMP}/eboot.bin")
      endif()
      message(STATUS "Vulkan found (${Vulkan_LIBRARY}) — offscreen graphics harness enabled")
    else()
      message(STATUS "Vulkan not found — skipping offscreen graphics harness")
    endif()
  endif()
endif()

# Normal sampled-render capture on every host with the live renderer. Unix writes PNG through zlib;
# Windows uses its built-in WIC PNG encoder so the native tool needs no extra package.
if(TARGET prosper_live_renderer AND TARGET Vulkan::Vulkan)
  if(WIN32)
    add_executable(screenshot tools/screenshot/screenshot.cpp
                              tools/screenshot/capture_manifest.cpp)
    target_include_directories(screenshot PRIVATE src tools/screenshot)
    target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan
                                              windowscodecs ole32)
  else()
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
      add_executable(screenshot tools/screenshot/screenshot.cpp
                                tools/screenshot/capture_manifest.cpp)
      target_include_directories(screenshot PRIVATE src tools/screenshot)
      target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan ZLIB::ZLIB)
    else()
      message(STATUS "zlib not found — skipping the screenshot tool")
    endif()
  endif()
endif()

# Optional FFmpeg AJM codec backend (#1347). Keep it outside prosper_core so core-only/headless
# builds retain no host codec dependency. The core owns the AJM ABI and copies; this frontend only
# accepts host byte spans and returns interleaved S16 PCM. MP3 is the first implemented codec.
option(PROSPER_AUDIO_FFMPEG "Build the FFmpeg AJM compressed-audio decoder backend" ON)
if(UNIX AND PROSPER_AUDIO_FFMPEG AND TARGET prosper_core)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PROSPER_AJM_FFMPEG QUIET IMPORTED_TARGET
                      libavcodec libavutil libswresample)
  endif()
  if(PROSPER_AJM_FFMPEG_FOUND)
    add_library(prosper_ajm_ffmpeg STATIC
                frontends/audio_ffmpeg/ajm_ffmpeg.cpp)
    target_include_directories(prosper_ajm_ffmpeg PUBLIC frontends/audio_ffmpeg PRIVATE src)
    target_link_libraries(prosper_ajm_ffmpeg PUBLIC prosper_core
                          PkgConfig::PROSPER_AJM_FFMPEG)

    add_executable(test_ajm_mp3 tests/test_ajm_mp3.cpp)
    target_link_libraries(test_ajm_mp3 PRIVATE prosper_ajm_ffmpeg)
    add_test(NAME ajm_mp3_decode COMMAND test_ajm_mp3)

    if(TARGET boot_trace)
      target_link_libraries(boot_trace PRIVATE prosper_ajm_ffmpeg)
      target_compile_definitions(boot_trace PRIVATE PROSPER_AUDIO_FFMPEG=1)
    endif()
    if(TARGET screenshot)
      target_link_libraries(screenshot PRIVATE prosper_ajm_ffmpeg)
      target_compile_definitions(screenshot PRIVATE PROSPER_AUDIO_FFMPEG=1)
    endif()
  else()
    message(WARNING "FFmpeg audio development packages not found - AJM MP3 decoding disabled")
  endif()
endif()

# Windows native AvPlayer backend (#841): Media Foundation performs MP4 demux and video/audio
# decode on a worker thread. Keep it outside prosper_core so core-only/headless builds retain no OS
# media dependency and so software fallback can never be selected merely because no frontend exists.
if(WIN32 AND TARGET prosper_core)
  add_library(prosper_video_mf STATIC
              frontends/video_mf/media_foundation_backend.cpp)
  target_include_directories(prosper_video_mf PUBLIC frontends/video_mf PRIVATE src)
  target_link_libraries(prosper_video_mf PUBLIC prosper_core
                         mfplat mfreadwrite mfuuid ole32 propsys d3d11 dxgi strmiids)

  add_executable(test_video_mf frontends/video_mf/test_video_mf.cpp)
  target_link_libraries(test_video_mf PRIVATE prosper_video_mf)
  add_test(NAME video_media_foundation COMMAND test_video_mf)

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_video_mf)
    target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_MF=1)
  endif()

  # The capture frontend must boot with the same real decoder as boot_trace and prosper-app;
  # otherwise video-gated titles never reach the frames that the tool is meant to preserve.
  if(TARGET screenshot)
    target_link_libraries(screenshot PRIVATE prosper_video_mf)
    target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_MF=1)
  endif()
endif()

# Linux native AvPlayer backend (#841): FFmpeg provides MP4 demux/audio conversion and drives a
# VA-API hardware decoder for video. Keep the dependency outside prosper_core; systems without the
# development packages retain the headless core but do not advertise a native video backend.
option(PROSPER_VIDEO_VAAPI "Build the Linux FFmpeg/VA-API AvPlayer backend" ON)
if(UNIX AND NOT APPLE AND PROSPER_VIDEO_VAAPI AND TARGET prosper_core)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PROSPER_FFMPEG QUIET IMPORTED_TARGET
                      libavformat libavcodec libavutil libswresample libswscale libva)
  endif()
  if(PROSPER_FFMPEG_FOUND)
    add_library(prosper_video_vaapi STATIC
                frontends/video_vaapi/vaapi_backend.cpp)
    target_include_directories(prosper_video_vaapi PUBLIC frontends/video_vaapi PRIVATE src)
    target_link_libraries(prosper_video_vaapi PUBLIC prosper_core PkgConfig::PROSPER_FFMPEG)

    add_executable(test_video_vaapi frontends/video_vaapi/test_video_vaapi.cpp)
    target_link_libraries(test_video_vaapi PRIVATE prosper_video_vaapi)
    # Point the test at a committed test-pattern clip so it decodes a real stream. On headless CI
    # (no VA-API hardware) a successful decode exercises the default software-decode fallback (#320).
    target_compile_definitions(test_video_vaapi PRIVATE
      PROSPER_TEST_VIDEO_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/testpattern.mp4")
    add_test(NAME video_vaapi COMMAND test_video_vaapi)

    if(TARGET boot_trace)
      target_link_libraries(boot_trace PRIVATE prosper_video_vaapi)
      target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_VAAPI=1)
    endif()

    # Match boot_trace and prosper-app: headless captures must install the real Linux decoder too,
    # otherwise video-gated titles only expose an empty scanout to the screenshot frontend.
    if(TARGET screenshot)
      target_link_libraries(screenshot PRIVATE prosper_video_vaapi)
      target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_VAAPI=1)
    endif()
  else()
    message(WARNING "FFmpeg/libva development packages not found - Linux AvPlayer video disabled")
  endif()
endif()

# --- SDL3 frontends (optional; headless core stays dependency-free) --------------------------
# The sceAudioOut sink (frontends/audio_sdl3) and the libScePad gamepad backend (frontends/pad_sdl3)
# can both use SDL3. They must share ONE SDL3 build: if each fetched independently, the first fetch's
# option set would win — e.g. an AUDIO+PAD build would get audio's joystick-less SDL and the gamepad
# backend would silently have no device. So acquire SDL3 ONCE here, with the UNION of the subsystems
# the enabled frontends need (JOYSTICK/HIDAPI only when the pad frontend is on). Both consumer blocks
# below just use the resulting SDL3::SDL3 target.
option(PROSPER_AUDIO_SDL3 "Build the optional SDL3 audio frontend"   OFF)
option(PROSPER_PAD_SDL3   "Build the optional SDL3 gamepad frontend" OFF)
option(PROSPER_APP        "Build the prosper-app frontend (SDL3 window + Vulkan present)" OFF)
if((PROSPER_AUDIO_SDL3 OR PROSPER_PAD_SDL3 OR PROSPER_APP) AND TARGET prosper_core AND NOT TARGET SDL3::SDL3)
  # On macOS the build is x86_64 (Rosetta), but a system SDL3 (e.g. Homebrew) is arm64-only and won't
  # link — so skip discovery there and always FetchContent+build SDL3 for the requested architecture.
  if(NOT APPLE)
    find_package(SDL3 QUIET CONFIG)
  endif()
  if(NOT SDL3_FOUND)
    message(STATUS "SDL3 not found — fetching release-3.2.30 (audio=${PROSPER_AUDIO_SDL3} pad=${PROSPER_PAD_SDL3} app=${PROSPER_APP})")
    include(FetchContent)
    set(SDL_TEST OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    # The window frontend (PROSPER_APP) needs a windowing system (Wayland/X11) for the Vulkan
    # surface; audio/pad-only builds do not — use SDL's console (no-video) escape hatch there. This
    # is the same "union of subsystems" logic as the joystick block below: enable VIDEO iff any
    # enabled frontend needs a window.
    if(PROSPER_APP)
      set(SDL_UNIX_CONSOLE_BUILD OFF CACHE BOOL "" FORCE)
      set(SDL_VIDEO   ON CACHE BOOL "" FORCE)
      # Wayland/X11 are the Linux window backends; on macOS SDL uses Cocoa (forcing X11/Wayland ON
      # there makes SDL hunt for X11/Wayland dev headers that don't exist). Only request them on Linux.
      if(NOT APPLE)
        set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
        set(SDL_X11     ON CACHE BOOL "" FORCE)
      endif()
    else()
      set(SDL_UNIX_CONSOLE_BUILD ON CACHE BOOL "" FORCE)
      set(SDL_VIDEO   OFF CACHE BOOL "" FORCE)
      set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
      set(SDL_X11     OFF CACHE BOOL "" FORCE)
    endif()
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_POWER OFF CACHE BOOL "" FORCE)
    # The app frontend opens a title with the host's native folder picker (#1469); the audio- and
    # pad-only builds have no window to parent a dialog to, so they keep SDL's dialog code out.
    if(PROSPER_APP)
      set(SDL_DIALOG ON CACHE BOOL "" FORCE)
    else()
      set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    endif()
    # Gamepad support is needed ONLY by the pad frontend — enable JOYSTICK + HIDAPI whenever it's on,
    # regardless of which frontend triggers the (single) fetch. This is the union that fixes the
    # combined-build trap: an AUDIO+PAD build now gets a joystick-capable SDL.
    if(PROSPER_PAD_SDL3)
      set(SDL_JOYSTICK ON CACHE BOOL "" FORCE)
      set(SDL_HIDAPI   ON CACHE BOOL "" FORCE)
    else()
      set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
      set(SDL_HIDAPI   OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(SDL3
      GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
      GIT_TAG release-3.2.30
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
  endif()
endif()

# --- prosper-app: SDL3 window + Vulkan swapchain present frontend (issue #164 P0) -----------
# Presents the frame prosper's renderer hands to the present layer (present_readback) in a real
# window. A SEPARATE Vulkan context from the core's headless render device (docs/FRONTEND_APP.md);
# frames cross as CPU pixels. Entirely outside prosper_core. Default OFF; needs SDL3 (video) + Vulkan.
if(PROSPER_APP AND TARGET SDL3::SDL3)
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(Vulkan_FOUND)
    # Dear ImGui for the game library view (#1471), vendored under third_party/imgui. FRONTEND ONLY:
    # prosper_core neither links nor sees it, so the one-way dependency arrow in docs/FRONTEND_APP.md
    # is unchanged. Built as its own static lib so the vendored sources compile with their own
    # warning expectations rather than prosper's.
    add_library(prosper_imgui STATIC
                third_party/imgui/src/imgui.cpp
                third_party/imgui/src/imgui_draw.cpp
                third_party/imgui/src/imgui_tables.cpp
                third_party/imgui/src/imgui_widgets.cpp
                third_party/imgui/backends/imgui_impl_sdl3.cpp
                third_party/imgui/backends/imgui_impl_vulkan.cpp)
    target_include_directories(prosper_imgui PUBLIC third_party/imgui/src third_party/imgui/backends)
    target_link_libraries(prosper_imgui PUBLIC SDL3::SDL3 Vulkan::Vulkan)
    # Vendored third-party code is not held to prosper's warning settings; keep its noise out of our
    # build output so a real warning in prosper's own code stays visible.
    if(NOT MSVC)
      target_compile_options(prosper_imgui PRIVATE -w)
    endif()

    add_executable(prosper-app frontends/prosper-app/main.cpp frontends/prosper-app/library_ui.cpp
                               frontends/prosper-app/library_media.cpp)
    target_include_directories(prosper-app PRIVATE src third_party/stb)
    target_link_libraries(prosper-app PRIVATE prosper_core SDL3::SDL3 Vulkan::Vulkan prosper_imgui
                                              prosper_performance_capture)
    target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_LIBRARY_UI=1)
    # Share the live renderer with boot_trace so the window shows composited game frames (not blank).
    if(TARGET prosper_live_renderer)
      target_link_libraries(prosper-app PRIVATE prosper_live_renderer)
      target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_LIVE_RENDERER=1)
    endif()
    # SDL3 dialog frontend (#347): real SDL_ShowMessageBox for the guest's MsgDialog/ErrorDialog via
    # the PlatformUi hook. The layout parsing (dialog_helpers) is pure + unit-tested (see below); this
    # module adds the SDL glue and needs SDL3 video (already required by prosper-app).
    add_library(prosper_dialog_sdl3 STATIC
                frontends/dialog_sdl3/platform_ui_sdl3.cpp frontends/dialog_sdl3/dialog_helpers.cpp)
    target_include_directories(prosper_dialog_sdl3 PUBLIC frontends/dialog_sdl3 src)
    target_link_libraries(prosper_dialog_sdl3 PUBLIC prosper_core SDL3::SDL3)
    target_link_libraries(prosper-app PRIVATE prosper_dialog_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_DIALOG_SDL3=1)
    message(STATUS "prosper-app frontend enabled (SDL3 window + Vulkan present)")
  else()
    message(WARNING "PROSPER_APP=ON but Vulkan not found — skipping prosper-app")
  endif()
endif()

# --- optional SDL3 audio frontend (headless core stays dependency-free) ---------------------
# The sceAudioOut HLE is backend-agnostic (src/hle/audio.hpp). This builds an SDL3-backed sink,
# entirely OUTSIDE prosper_core, so the guest's audio plays on the host.
if(PROSPER_AUDIO_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_audio_sdl3 STATIC frontends/audio_sdl3/audio_sink_sdl3.cpp)
  target_include_directories(prosper_audio_sdl3 PUBLIC frontends/audio_sdl3 src)
  target_link_libraries(prosper_audio_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_audio_sdl3 PUBLIC PROSPER_AUDIO_SDL3=1)

  # Headless self-test of the full SDL3 path (run under the SDL "dummy" audio driver in CI):
  # install the sink, drive the sceAudioOut HLE, assert grains flow through SDL with no device.
  add_executable(test_audio_sdl3 frontends/audio_sdl3/test_audio_sdl3.cpp)
  target_link_libraries(test_audio_sdl3 PRIVATE prosper_audio_sdl3)
  add_test(NAME audio_sdl3 COMMAND test_audio_sdl3)
  set_tests_properties(audio_sdl3 PROPERTIES ENVIRONMENT "SDL_AUDIO_DRIVER=dummy")

  # Route the live game's audio to the host when the runner is built (guarded by the macro).
  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_audio_sdl3)
    target_compile_definitions(boot_trace PRIVATE PROSPER_AUDIO_SDL3=1)
  endif()

  message(STATUS "SDL3 audio frontend enabled (prosper_audio_sdl3)")
endif()

# --- optional SDL3 gamepad frontend (headless core stays dependency-free) --------------------
# The libScePad HLE is backend-agnostic (src/input/pad.hpp). This builds an SDL_Gamepad-backed
# PadBackend, entirely OUTSIDE prosper_core, so a real controller drives the guest cross-platform.
# Off by default.  Enable with:  -DPROSPER_PAD_SDL3=ON  (SDL3 acquired by the shared step above,
# with JOYSTICK/HIDAPI ON because this option is set).
if(PROSPER_PAD_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_pad_sdl3 STATIC frontends/pad_sdl3/pad_sink_sdl3.cpp)
  target_include_directories(prosper_pad_sdl3 PUBLIC frontends/pad_sdl3 src)
  target_link_libraries(prosper_pad_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_pad_sdl3 PUBLIC PROSPER_PAD_SDL3=1)

  # Headless self-test of the full SDL3 gamepad path (no controller attached in CI): install the
  # backend, poll -> disconnected/neutral, drive the pure mapping. Uses the SDL dummy video driver.
  add_executable(test_pad_sdl3 frontends/pad_sdl3/test_pad_sdl3.cpp)
  target_link_libraries(test_pad_sdl3 PRIVATE prosper_pad_sdl3)
  add_test(NAME pad_sdl3 COMMAND test_pad_sdl3)
  set_tests_properties(pad_sdl3 PROPERTIES ENVIRONMENT "SDL_VIDEO_DRIVER=dummy;SDL_AUDIO_DRIVER=dummy")

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_pad_sdl3)
    target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_SDL3=1)
  endif()

  message(STATUS "SDL3 gamepad frontend enabled (prosper_pad_sdl3)")
endif()

# --- wire the SDL3 audio/pad frontends into prosper-app (after their targets exist) ---------
# prosper-app is defined above (before these libs), so link them here once they're built. Enable
# both alongside the app for a full experience:  -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
if(TARGET prosper-app)
  if(TARGET prosper_ajm_ffmpeg)
    target_link_libraries(prosper-app PRIVATE prosper_ajm_ffmpeg)
    target_compile_definitions(prosper-app PRIVATE PROSPER_AUDIO_FFMPEG=1)
    message(STATUS "prosper-app: FFmpeg AJM audio decoding enabled")
  endif()
  if(TARGET prosper_audio_sdl3)
    target_link_libraries(prosper-app PRIVATE prosper_audio_sdl3)
    target_include_directories(prosper-app PRIVATE frontends/audio_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_AUDIO_SDL3=1)
    message(STATUS "prosper-app: audio out enabled")
  endif()
  if(TARGET prosper_pad_sdl3)
    target_link_libraries(prosper-app PRIVATE prosper_pad_sdl3)
    target_include_directories(prosper-app PRIVATE frontends/pad_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_PAD_SDL3=1)
    message(STATUS "prosper-app: controller input enabled")
  endif()
endif()

if(TARGET prosper-app AND TARGET prosper_video_mf)
  target_link_libraries(prosper-app PRIVATE prosper_video_mf)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_MF=1)
  message(STATUS "prosper-app: Media Foundation video enabled")
endif()

if(TARGET prosper-app AND TARGET prosper_video_vaapi)
  target_link_libraries(prosper-app PRIVATE prosper_video_vaapi)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_VAAPI=1)
  message(STATUS "prosper-app: FFmpeg/VA-API video enabled")
endif()

# Shared-device adoption guard (#1091). prosper_live_renderer only exists in frontend-enabled
# configurations and is defined later in this file than the core test block, so register this
# after the fact and only when the target is actually present.
if(TARGET prosper_live_renderer AND TARGET prosper_core)
  add_executable(test_shared_vulkan_device tests/test_shared_vulkan_device.cpp)
  target_link_libraries(test_shared_vulkan_device PRIVATE prosper_live_renderer prosper_core)
  add_test(NAME shared_vulkan_device COMMAND test_shared_vulkan_device)

  # Live-target format round trip: every LiveTargetPixelFormat a compute dispatch can import must
  # survive the write-back notifier. An unnamed format is reported as RGBA8, fails the mirror
  # identity check, and silently leaves the renderer's target invalidated (#773's family).
  add_executable(test_live_target_format tests/test_live_target_format.cpp)
  target_include_directories(test_live_target_format PRIVATE src tests)
  target_link_libraries(test_live_target_format PRIVATE prosper_live_renderer prosper_core
                                                        Vulkan::Vulkan)
  add_test(NAME live_target_format COMMAND test_live_target_format)

  # GPU scanout handoff (#1270): the front-buffer image blits into a scanout slot byte-exactly, and the
  # producer/consumer slot handoff never tears or reuses an image in flight. Headless (no surface).
  add_executable(test_present_blit tests/test_present_blit.cpp)
  target_include_directories(test_present_blit PRIVATE src tests)
  target_link_libraries(test_present_blit PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
  add_test(NAME present_blit COMMAND test_present_blit)
endif()

# ---------------------------------------------------------------------------
# Per-case scratch directory for every registered test (#1613).
#
# Tests that must touch the real filesystem take their paths from
# tests/test_scratch.h, which reads PROSPER_TEST_SCRATCH_DIR. Giving each ctest
# *case* its own root is what makes `ctest -j` safe: one test binary is
# sometimes registered as several cases (game_compute_exec and
# game_compute_exec_no_adaptive_storage_result are the same executable), so
# per-binary uniqueness is not enough. The helper adds a pid component
# underneath, which separates concurrent ctest invocations from different
# worktrees. The root is inside the build directory, never /tmp — see the
# header for why that matters on this machine.
#
# This must stay at the very end of the file: it snapshots the DIRECTORY TESTS
# property, so any add_test() below it would not be covered. A test that did get
# missed still fails safe rather than silently — test_scratch.h falls back to
# <cwd>/prosper-test-scratch/, and ctest's working directory is this same build
# directory, so it lands off the tmpfs either way; it merely loses the per-case
# split (the pid component still separates concurrent processes).
get_property(prosper_registered_tests DIRECTORY PROPERTY TESTS)
foreach(prosper_test_case IN LISTS prosper_registered_tests)
  set_property(TEST ${prosper_test_case} APPEND PROPERTY ENVIRONMENT
    "PROSPER_TEST_SCRATCH_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-scratch/${prosper_test_case}")
endforeach()
