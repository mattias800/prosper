// libSceJson2 parsing and value access. Astro Bot parses configuration JSON during level startup
// and asserts that values reported as booleans have the matching Json2 type.
#include "../src/hle/dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

namespace {
int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [ok]   %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

struct JsonValue {
    void* parent;
    void* root_parameter;
    uint64_t payload;
    uint8_t reserved[4];
    uint32_t type;
};
static_assert(sizeof(JsonValue) == 32 && offsetof(JsonValue, type) == 28);

uint64_t call(HleFn fn, const void* a0 = nullptr, const void* a1 = nullptr,
              uint64_t a2 = 0, uint64_t a3 = 0) {
    return fn(reinterpret_cast<uintptr_t>(a0), reinterpret_cast<uintptr_t>(a1), a2, a3, 0, 0);
}
} // namespace

// A two-page reservation whose FIRST page is readable and whose second is not. Used to build a
// string that runs flush into unmapped memory -- the case that separates "refuse" from "hand
// back a truncated prefix". Portable because the behaviour under test is not platform-specific;
// guarding the arm to one OS would leave the other silently uncovered (the #1970 shape).
size_t probe_page_size() {
#ifdef _WIN32
    SYSTEM_INFO si{}; GetSystemInfo(&si); return si.dwPageSize;
#else
    return (size_t)sysconf(_SC_PAGESIZE);
#endif
}

char* reserve_two_pages_first_readable() {
    const size_t page = probe_page_size();
#ifdef _WIN32
    auto* base = static_cast<char*>(VirtualAlloc(nullptr, page * 2, MEM_RESERVE, PAGE_NOACCESS));
    if (!base) return nullptr;
    if (!VirtualAlloc(base, page, MEM_COMMIT, PAGE_READWRITE)) {
        VirtualFree(base, 0, MEM_RELEASE); return nullptr;
    }
    return base;
#else
    auto* base = static_cast<char*>(
        mmap(nullptr, page * 2, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (base == MAP_FAILED) return nullptr;
    if (mprotect(base, page, PROT_READ | PROT_WRITE) != 0) {
        munmap(base, page * 2); return nullptr;
    }
    return base;
#endif
}

void release_two_pages(char* base) {
    const size_t page = probe_page_size();
#ifdef _WIN32
    (void)page; VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, page * 2);
#endif
}

int main() {
    std::puts("== test_json2 ==");
    register_builtin_hle();

    HleFn ctor = Hle::lookup("qBMjqyBn3OM");
    HleFn dtor = Hle::lookup("WTtYf+cNnXI");
    HleFn parse = Hle::lookup("S5JxQnoGF3E");
    HleFn index_string = Hle::lookup("HwDt5lD9Bfo");
    HleFn index_uint = Hle::lookup("XlWbvieLj2M");
    HleFn get_type = Hle::lookup("SHtAad20YYM");
    HleFn get_boolean = Hle::lookup("zTwZdI8AZ5Y");
    HleFn get_integer = Hle::lookup("DIxvoy7Ngvk");
    HleFn get_unsigned = Hle::lookup("sn4HNCtNRzY");
    HleFn get_string = Hle::lookup("epJ6x2LV0kU");
    HleFn string_cstr = Hle::lookup("L1KAkYWml-M");
    HleFn count = Hle::lookup("RBw+4NukeGQ");
    HleFn assign = Hle::lookup("4zrm6VrgIAw");
    HleFn string_ctor = Hle::lookup("qSmqLXXCPas");
    HleFn string_dtor = Hle::lookup("cG1VE2HMl6c");
    HleFn to_string = Hle::lookup("Ncel8t2Rrpc");
    HleFn refer_string = Hle::lookup("wLsJlmgEIaI");
    HleFn set_bool = Hle::lookup("5yHuiWXo2gg");
    CHECK(ctor && dtor && parse && index_string && index_uint && get_type && get_boolean &&
          get_integer && get_unsigned && get_string && string_cstr && count && assign && string_ctor && string_dtor &&
          to_string && refer_string && set_bool, "Json2 entry points are registered");
    if (failures) return 1;

    constexpr char source[] =
        R"({"enabled":true,"nested":{"off":false},"items":[-7,2],"emoji":"\uD83D\uDE80"})";
    JsonValue root{};
    call(ctor, &root);
    CHECK(call(parse, &root, source, std::strlen(source)) == 0, "object JSON parses successfully");
    CHECK(call(get_type, &root) == 7, "root is reported as an object");
    CHECK(call(count, &root) == 4, "object member count is preserved");

    auto* enabled = reinterpret_cast<JsonValue*>(call(index_string, &root, "enabled"));
    CHECK(enabled && call(get_type, enabled) == 1, "boolean member has boolean type");
    auto* enabled_value = reinterpret_cast<const bool*>(call(get_boolean, enabled));
    CHECK(enabled_value && *enabled_value, "boolean member has true value");

    auto* nested = reinterpret_cast<JsonValue*>(call(index_string, &root, "nested"));
    auto* off = reinterpret_cast<JsonValue*>(call(index_string, nested, "off"));
    auto* off_value = reinterpret_cast<const bool*>(call(get_boolean, off));
    CHECK(off && call(get_type, off) == 1 && off_value && !*off_value,
          "nested false boolean is preserved");

    auto* items = reinterpret_cast<JsonValue*>(call(index_string, &root, "items"));
    CHECK(items && call(get_type, items) == 6 && call(count, items) == 2,
          "array type and element count are preserved");
    auto* first = reinterpret_cast<JsonValue*>(call(index_uint, items, nullptr, 0));
    auto* integer = reinterpret_cast<const int64_t*>(call(get_integer, first));
    CHECK(first && call(get_type, first) == 2 && integer && *integer == -7,
          "signed array element is preserved");
    auto* second = reinterpret_cast<JsonValue*>(
        index_uint(reinterpret_cast<uintptr_t>(items), 1, 0, 0, 0, 0));
    auto* unsigned_integer = reinterpret_cast<const uint64_t*>(call(get_unsigned, second));
    CHECK(second && call(get_type, second) == 3 && unsigned_integer && *unsigned_integer == 2,
          "unsigned array element is preserved");

    auto* emoji = reinterpret_cast<JsonValue*>(call(index_string, &root, "emoji"));
    auto* json_string = reinterpret_cast<void*>(call(get_string, emoji));
    auto* utf8 = reinterpret_cast<const char*>(call(string_cstr, json_string));
    CHECK(utf8 && std::strcmp(utf8, "\xF0\x9F\x9A\x80") == 0,
          "UTF-16 surrogate pair decodes to UTF-8");

    JsonValue copy{};
    call(ctor, &copy);
    CHECK(call(assign, &copy, &root) == reinterpret_cast<uintptr_t>(&copy),
          "value assignment returns its destination");
    call(dtor, &root);
    auto* copied_enabled = reinterpret_cast<JsonValue*>(call(index_string, &copy, "enabled"));
    auto* copied_boolean = reinterpret_cast<const bool*>(call(get_boolean, copied_enabled));
    CHECK(copied_boolean && *copied_boolean, "value assignment owns an independent deep copy");

    struct JsonString { void* implementation; } serialized{};
    call(string_ctor, &serialized);
    CHECK(call(to_string, &copy, &serialized) == 0, "value serialization succeeds");
    const char* serialized_text = reinterpret_cast<const char*>(call(string_cstr, &serialized));
    CHECK(serialized_text && std::strstr(serialized_text, "\"enabled\":true"),
          "serialized object retains boolean members");
    call(string_dtor, &serialized);

    auto* missing = reinterpret_cast<JsonValue*>(call(index_string, &copy, "missing"));
    CHECK(missing && call(get_type, missing) == 0, "missing member yields stable null value");
    call(string_ctor, &serialized);
    CHECK(call(refer_string, &copy, &serialized) == 0,
          "missing pointer-style lookup returns null");
    call(string_dtor, &serialized);
    call(dtor, &copy);
    CHECK(copy.type == 0, "value destructor releases the parsed tree");

    constexpr char invalid[] = R"({"broken":])";
    call(ctor, &root);
    CHECK(call(parse, &root, invalid, std::strlen(invalid)) == 0x80848101u,
          "invalid JSON returns the Json2 parse error");
    CHECK(root.type == 0, "failed parse leaves a null value");
    call(dtor, &root);

    // Sonic Origins links the PS5 SDK 3.20 spellings for its startup JSON objects. Exercise the
    // exact imported aliases so a generic success stub cannot leave caller-owned storage stale.
    HleFn ps5_parameter_ctor = Hle::lookup("WSOuge5IsCg");
    HleFn ps5_set_allocator = Hle::lookup("I2QC8PYhJWY");
    HleFn ps5_set_file_buffer_size = Hle::lookup("Eu95jmqn5Rw");
    HleFn ps5_initialize = Hle::lookup("IXW-z8pggfg");
    HleFn ps5_object_ctor = Hle::lookup("OJPTonqdg0I");
    HleFn ps5_object_dtor = Hle::lookup("5JmzZt8twAo");
    HleFn ps5_object_index = Hle::lookup("ERuf9y0DY84");
    HleFn ps5_object_empty = Hle::lookup("i2l3IYvQ9UE");
    HleFn ps5_string_value_ctor = Hle::lookup("sZIoMRGO+jk");
    HleFn ps5_object_value_ctor = Hle::lookup("3xUXnmUkXfo");
    HleFn ps5_set_object = Hle::lookup("dFCphqnd+a4");
    CHECK(ps5_parameter_ctor && ps5_set_allocator && ps5_set_file_buffer_size && ps5_initialize &&
          ps5_object_ctor && ps5_object_dtor && ps5_object_index && ps5_object_empty &&
          ps5_string_value_ctor && ps5_object_value_ctor && ps5_set_object,
          "Sonic's PS5 Json2 aliases are registered");

    struct JsonInitParameter2 {
        void* allocator;
        void* user_data;
        uint64_t file_buffer_size;
        uint32_t special_float_format_type;
        uint32_t reserved[3];
    } parameter;
    std::memset(&parameter, 0xa5, sizeof(parameter));
    CHECK(call(ps5_parameter_ctor, &parameter) == reinterpret_cast<uintptr_t>(&parameter) &&
          !parameter.allocator && !parameter.user_data && parameter.file_buffer_size == 0 &&
          parameter.special_float_format_type == 0,
          "PS5 InitParameter2 constructor clears its complete ABI object");
    ps5_set_allocator(reinterpret_cast<uintptr_t>(&parameter), 0x12340000u, 0x56780000u, 0, 0, 0);
    ps5_set_file_buffer_size(reinterpret_cast<uintptr_t>(&parameter), 0x9000u, 0, 0, 0, 0);
    CHECK(parameter.allocator == reinterpret_cast<void*>(0x12340000u) &&
          parameter.user_data == reinterpret_cast<void*>(0x56780000u) &&
          parameter.file_buffer_size == 0x9000u && call(ps5_initialize, &parameter) == 0,
          "PS5 Json2 initialization parameters retain allocator and file-buffer state");

    struct JsonObject { void* implementation; } object{};
    JsonString key{};
    call(ps5_object_ctor, &object);
    CHECK(call(ps5_object_empty, &object) == 1, "new PS5 JsonObject is empty");
    call(string_ctor, &key);
    auto* inserted = reinterpret_cast<JsonValue*>(call(ps5_object_index, &object, &key));
    call(set_bool, inserted, reinterpret_cast<const void*>(1));
    CHECK(inserted && call(ps5_object_empty, &object) == 0,
          "PS5 JsonObject index creates a writable value");

    JsonValue object_value{};
    call(ps5_object_value_ctor, &object_value, &object);
    call(ps5_object_dtor, &object);
    auto* copied_empty_key = reinterpret_cast<JsonValue*>(call(index_string, &object_value, ""));
    auto* copied_true = reinterpret_cast<const bool*>(call(get_boolean, copied_empty_key));
    CHECK(object_value.type == 7 && copied_true && *copied_true,
          "PS5 JsonValue object constructor owns a deep copy");

    JsonValue string_value{};
    call(ps5_string_value_ctor, &string_value, &key);
    CHECK(string_value.type == 5, "PS5 JsonValue string constructor preserves the string type");
    call(ps5_set_object, &string_value, reinterpret_cast<void*>(object_value.payload));
    auto* set_empty_key = reinterpret_cast<JsonValue*>(call(index_string, &string_value, ""));
    auto* set_true = reinterpret_cast<const bool*>(call(get_boolean, set_empty_key));
    CHECK(string_value.type == 7 && set_true && *set_true,
          "PS5 JsonValue set(Object) replaces and deep-copies the prior value");
    call(dtor, &string_value);
    call(string_dtor, &key);
    call(dtor, &object_value);

    // --- #1967: sce::Json::Value::Value(const char*) ------------------------------------------
    // The dispatcher's `return 0` default is safe for a status-returning function and unsafe for a
    // CONSTRUCTOR: the guest reserves the storage and expects the callee to write the object into
    // it. An unimplemented ctor therefore leaves the guest using stack residue as a live Value, and
    // no return value can fix that -- which is why this had to be implemented rather than stubbed.
    {
        HleFn ctor_cstr = Hle::lookup("b9V6fmppLXY");
        CHECK(ctor_cstr != nullptr, "Value(const char*) is registered");
        if (ctor_cstr) {
            // POISONED storage, not zeroed. A zeroed slot would let an unimplemented ctor pass this
            // arm by accident -- type 0 is Null, which is a legal value -- so the arm has to start
            // from something no correct constructor could leave behind.
            JsonValue v;
            std::memset(&v, 0xA5, sizeof v);
            call(ctor_cstr, &v, const_cast<char*>("hello"));
            CHECK(v.type == 5, "Value(const char*) constructs a String value (type 5)");
            auto* js = reinterpret_cast<void*>(call(get_string, &v));
            auto* cs = js ? reinterpret_cast<const char*>(call(string_cstr, js)) : nullptr;
            CHECK(cs && std::strcmp(cs, "hello") == 0,
                  "Value(const char*) stores the text it was given");
            call(dtor, &v);
        }
        if (ctor_cstr) {
            // Value("") is a String whose text is empty, NOT Null. Returning Null here would be a
            // silent wrong answer rather than a refusal: getType() would report 0 where the guest
            // constructed a string, and every later getString() would disagree with the ctor.
            JsonValue v;
            std::memset(&v, 0xA5, sizeof v);
            call(ctor_cstr, &v, const_cast<char*>(""));
            CHECK(v.type == 5, "Value(\"\") constructs an empty String (type 5), not Null");
            auto* js = reinterpret_cast<void*>(call(get_string, &v));
            auto* cs = js ? reinterpret_cast<const char*>(call(string_cstr, js)) : nullptr;
            CHECK(cs && cs[0] == 0, "Value(\"\") stores empty text");
            call(dtor, &v);
        }
        if (ctor_cstr) {
            // A null argument is a null Value, not a crash -- and the storage must still be left
            // well-formed, because the guest will destroy it either way.
            JsonValue v;
            std::memset(&v, 0xA5, sizeof v);
            call(ctor_cstr, &v, nullptr);
            CHECK(v.type == 0, "Value(nullptr) constructs a Null value rather than faulting");
            call(dtor, &v);
        }
        if (ctor_cstr) {
            // An unmapped-but-non-null pointer. 0x80 is the shape that actually occurs (a null base
            // plus a field offset) and the one that walked into #1963 in libkernel. Reaching the
            // next line at all is the assertion: an unvalidated read here does not fail, it faults.
            JsonValue v;
            std::memset(&v, 0xA5, sizeof v);
            call(ctor_cstr, &v, reinterpret_cast<void*>(static_cast<uintptr_t>(0x80)));
            CHECK(v.type == 0, "Value(bad pointer) yields a well-formed Null value, not a fault");
            call(dtor, &v);
        }
        if (ctor_cstr) {
            // UNTERMINATED: readable text running flush into an unmapped page with no NUL.
            // Built by reserving two pages and making only the first accessible, so the bytes
            // are genuinely readable up to a hard boundary. A truncated construction here is a
            // wrong ANSWER rather than a refusal, and the earlier code produced exactly that.
            char* base = reserve_two_pages_first_readable();
            if (base) {
                const size_t page = probe_page_size();
                std::memset(base, 0x41, page);          // a page of "A", no terminator
                JsonValue v;
                std::memset(&v, 0xA5, sizeof v);
                call(ctor_cstr, &v, base + page - 8);   // 8 readable bytes, then nothing
                CHECK(v.type == 0,
                      "Value(unterminated) refuses rather than constructing a truncated String");
                call(dtor, &v);
                release_two_pages(base);
            }
        }
        // The two destructors this issue also names. They own nothing prosper allocated, so a no-op
        // is the correct body -- the point is that they RESOLVE rather than falling to the
        // unimplemented path, where each costs a log line and reads as a gap.
        CHECK(Hle::lookup("RujUxbr3haM") != nullptr, "Initializer::~Initializer is registered");
        CHECK(Hle::lookup("OcAgPxcq5Vk") != nullptr, "MemAllocator::~MemAllocator is registered");
    }

    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
