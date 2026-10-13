#include "host/image/process_entry.hpp"

#include "host/platform/immortal.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace prosper {
namespace {
constexpr size_t kStackBytes = size_t{16} * 1024 * 1024;
std::atomic<const ProcessEntryContext*> published{nullptr};
struct ProcessEntryOwner {
    std::mutex mutex;
    std::unique_ptr<ProcessEntryContext> context;
};
struct EntryStackDeleter {
    void operator()(void* stack) const noexcept {
#ifdef _WIN32
        VirtualFree(stack, 0, MEM_RELEASE);
#else
        munmap(stack, kStackBytes);
#endif
    }
};

std::vector<std::string> environment_strings(const char* configured) {
    std::vector<std::string> result;
    if (!configured) return result;
    const std::string text(configured);
    for (size_t begin = 0; begin < text.size();) {
        const size_t end = text.find(';', begin);
        const size_t size = (end == std::string::npos ? text.size() : end) - begin;
        if (size) result.push_back(text.substr(begin, size));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}
} // namespace

bool prepare_process_entry(const std::vector<std::string>& arguments,
                           const char* configured_environment, std::string* error) {
    // The owning pointer would be destroyed with a bare holder, invalidating getters reached by
    // late guest workers. Immortal retains both the owner and its mutex through static teardown.
    static Immortal<ProcessEntryOwner> owner;
    static_assert(std::is_trivially_destructible_v<decltype(owner)>);
    std::lock_guard lock(owner->mutex);
    if (published.load(std::memory_order_acquire)) return true;
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    const auto environment = environment_strings(configured_environment);
    if (arguments.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        return fail("guest argument count exceeds the kernel count width");
    // Budget strings, vector words, the existing 16-byte alignment and entry bias before writing.
    // Keep usable stack below the crt0 data; an oversized configuration must not underflow it.
    // CONFIDENCE: LOW on this reserve, an implementation capacity bound rather than an SDK limit.
    constexpr size_t kDataBudget = kStackBytes - size_t{64} * 1024;
    size_t bytes = 64;
    const auto add = [&](size_t size) {
        if (size > kDataBudget - bytes) return false;
        bytes += size;
        return true;
    };
    for (const auto& value : arguments)
        if (value.size() >= kDataBudget || !add(value.size() + 1))
            return fail("guest arguments exceed the entry stack");
    for (const auto& value : environment)
        if (value.size() >= kDataBudget || !add(value.size() + 1))
            return fail("guest environment exceeds the entry stack");
    if (arguments.size() > kDataBudget / sizeof(uint64_t) ||
        environment.size() > kDataBudget / sizeof(uint64_t) ||
        !add((arguments.size() + environment.size() + 6) * sizeof(uint64_t)))
        return fail("guest startup vector exceeds the entry stack");

    void* stack = nullptr;
#ifdef _WIN32
    stack = VirtualAlloc(nullptr, kStackBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    stack = mmap(nullptr, kStackBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (stack == MAP_FAILED) stack = nullptr;
#endif
    if (!stack) return fail("guest entry stack allocation failed");
    // Allocation of the vector/context below can throw. Until publication, this mapping must be
    // reclaimed on unwinding rather than leaked as if guest workers could already retain it.
    std::unique_ptr<void, EntryStackDeleter> pending_stack(stack);
    uint64_t top = (reinterpret_cast<uint64_t>(stack) + kStackBytes) & ~uint64_t{15};
    std::vector<uint64_t> vector;
    vector.push_back(arguments.size());
    const auto append_strings = [&](const std::vector<std::string>& strings) {
        for (const auto& value : strings) {
            top -= value.size() + 1;
            std::memcpy(reinterpret_cast<void*>(top), value.c_str(), value.size() + 1);
            vector.push_back(top);
        }
        vector.push_back(0);
    };
    append_strings(arguments);
    append_strings(environment);
    vector.push_back(0);
    vector.push_back(0);   // AT_NULL
    if (vector.size() & 1) vector.push_back(0);
    top = (top & ~uint64_t{15}) - vector.size() * sizeof(uint64_t);
    top = (top & ~uint64_t{15}) - 8;   // Sony crt0 expects entry RSP == 8 (mod 16).
    std::memcpy(reinterpret_cast<void*>(top), vector.data(), vector.size() * sizeof(uint64_t));
    // CONFIDENCE: HIGH — native libkernel startup saves argc=[crt0], argv=crt0+8, and the getters
    // return those cached cells (libkernel+0x2ef08/0x2ef1f, +0x2f0a7/+0x2f0ae and +0x2fc30/+0x2fc40).
    owner->context = std::make_unique<ProcessEntryContext>(ProcessEntryContext{
        nullptr, kStackBytes, top, static_cast<uint32_t>(arguments.size()), top + 8});
    // Transfer the mapping only after the context allocation succeeds. Published guest pointers
    // remain valid for the process lifetime.
    owner->context->stack = pending_stack.release();
    published.store(owner->context.get(), std::memory_order_release);
    return true;
}

ProcessEntryContext process_entry_context() {
    const auto* context = published.load(std::memory_order_acquire);
    return context ? *context : ProcessEntryContext{};
}
}   // namespace prosper
