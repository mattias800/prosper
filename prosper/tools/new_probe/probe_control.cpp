#include <pthread.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <new>

constexpr int kCalls = 500;

extern "C" __attribute__((noinline)) void new_probe_scalar_site() {
    void* p = ::operator new(37);
    asm volatile("" : : "g"(p) : "memory");
    ::operator delete(p);
}

extern "C" __attribute__((noinline)) void new_probe_array_site() {
    void* p = ::operator new[](59);
    asm volatile("" : : "g"(p) : "memory");
    ::operator delete[](p);
}

static void run_sites() {
    for (int i = 0; i < kCalls; ++i) {
        new_probe_scalar_site();
        new_probe_array_site();
    }
}

static void* worker(void* arg) {
    *static_cast<unsigned*>(arg) = static_cast<unsigned>(syscall(SYS_gettid));
    run_sites();
    return nullptr;
}

int main(int argc, char** argv) {
    const unsigned main_tid = static_cast<unsigned>(syscall(SYS_gettid));
    unsigned worker_tid = 0;
    pthread_t thread{};
    if (pthread_create(&thread, nullptr, worker, &worker_tid) != 0) return 2;
    run_sites();
    if (pthread_join(thread, nullptr) != 0) return 3;
    std::atomic<size_t> impossible{SIZE_MAX};
    bool failed_as_expected = false;
    try {
        void* p = ::operator new(impossible.load());
        ::operator delete(p);
    } catch (const std::bad_alloc&) {
        failed_as_expected = true;
    }
    if (!failed_as_expected) return 4;
    printf("main_tid=%u worker_tid=%u calls_per_site_per_tid=%d\n",
           main_tid, worker_tid, kCalls);
    fflush(stdout);
    if (argc > 1 && argv[1][0] == 's') sleep(3);
    if (argc > 1 && argv[1][0] == 'f') {
        const pid_t child = fork();
        if (child < 0) return 5;
        if (child == 0) {
            new_probe_scalar_site();
            return 0;
        }
        int status = 0;
        if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status))
            return 6;
        printf("fork_child_pid=%d\n", static_cast<int>(child));
    }
    return 0;
}
