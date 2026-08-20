// immortal.hpp — process-lifetime state that must remain usable during and after static destruction.
//
// #2613. prosper never joins guest threads: k_pthread_create spawns them and nothing quiesces them
// at shutdown, so every prosper process reaches exit() with live guest threads. thread_trampoline's
// tail (tls_dtv_purge_current_thread / unregister_thread_stack / retire_guest_thread_name) then runs
// concurrently with __run_exit_handlers, and it touches namespace-scope registries whose static
// destructors that handler chain has already executed. Observed as a real core dump: the main thread
// inside _dl_fini, a guest thread inside std::unordered_map lookup on a destroyed map.
//
// Immortal<T> removes the ordering question rather than narrowing the window. The contained object is
// constructed during its translation unit's dynamic initialization exactly where the plain object was,
// and is NEVER destroyed: the holder's only member is a byte array, so the holder is trivially
// destructible and the compiler emits no __cxa_atexit registration for it. Storage is
// process-lifetime, so a thread that reaches the registry at any point — including after every exit
// handler has run — finds a live object. There is no window to widen and no ordering to get right.
//
// The cost is that T's destructor never runs. For a process-lifetime registry that is the intent:
// the OS reclaims the memory at exit, and declining to destroy it is what makes the shutdown race
// unrepresentable. Do NOT use Immortal for anything whose destructor has a side effect the process
// actually needs (flushing a stream, releasing a lock visible outside this process, joining a
// thread) — those still need an explicit, ordered shutdown.
//
// Pair every declaration with
//     static_assert(std::is_trivially_destructible_v<decltype(the_object)>, "...");
// at the declaration site. That assert is the guard against the regression this file exists to
// prevent: it fires if the holder is ever simplified back to a bare container.
//
// TWO PROPERTIES THAT ARE NOT VISIBLE AT A CALL SITE, both measured on this tree's toolchain
// (Fedora gcc 16, libstdc++). Neither is a problem for anything converted in #2613; both are the
// kind of thing that bites whoever applies this header somewhere new.
//
//  1. THE static_assert IS VACUOUS FOR SOME T. libstdc++'s bare std::mutex already measures
//     trivially-destructible = 1, so an assert naming only mutex operands proves nothing. Every
//     assert in this tree is a conjunction that also names the CONTAINER beside the mutex, and the
//     containers measure 0 -- that operand is what makes the guard fire. Keep the container in the
//     conjunction; do not write an Immortal<std::mutex> assert on its own and read it as coverage.
//
//  2. WRAPPING CONVERTS CONSTANT INITIALIZATION INTO DYNAMIC INITIALIZATION. A bare namespace-scope
//     std::mutex has a constexpr constructor and emits no _GLOBAL__sub_I_ at all; wrapped in
//     Immortal it acquires one, because this class has a user-provided constructor. That trades a
//     zero-cost object for one with static-initialization-order exposure -- a net loss if the
//     object had no destructor to begin with. It is harmless as applied here only because every
//     converted mutex is taken alongside a container in the same TU that already required dynamic
//     initialization. Before wrapping a constant-initialized T, check that it is not the only
//     reason its TU gains a static initializer.
#pragma once

#include <new>
#include <type_traits>
#include <utility>

namespace prosper {

template <class T>
class Immortal {
public:
    template <class... Args>
    explicit Immortal(Args&&... args) {
        ::new (static_cast<void*>(storage_)) T(std::forward<Args>(args)...);
    }
    Immortal(const Immortal&) = delete;
    Immortal& operator=(const Immortal&) = delete;
    // No destructor is declared, on purpose. Declaring one — even `= default` — would still be
    // trivial here, but writing it invites someone to "complete" it later with a call to T's.

    // std::launder is REQUIRED, not decoration. An array and its first element are not
    // pointer-interconvertible, so the reinterpret_cast alone yields a pointer to the byte array
    // rather than to the T constructed into it -- formally UB since C++17. No shipping compiler
    // exploits it for this pattern today, and with the launder the generated assembly for this
    // tree differs only in .file directives and local label numbering; it is here so the next T
    // this header is applied to (a type with a const or reference member, where the aliasing
    // question has teeth) does not inherit a latent defect.
    T& get() noexcept { return *std::launder(reinterpret_cast<T*>(storage_)); }
    const T& get() const noexcept { return *std::launder(reinterpret_cast<const T*>(storage_)); }
    T& operator*() noexcept { return get(); }
    const T& operator*() const noexcept { return get(); }
    T* operator->() noexcept { return &get(); }
    const T* operator->() const noexcept { return &get(); }

private:
    alignas(T) unsigned char storage_[sizeof(T)];
};

}   // namespace prosper
