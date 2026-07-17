// libSceJson2 parsing and value access. Astro Bot parses configuration JSON during level startup
// and asserts that values reported as booleans have the matching Json2 type.
#include "../src/hle/dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

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
    CHECK(ctor && dtor && parse && index_string && index_uint && get_type && get_boolean &&
          get_integer && get_unsigned && get_string && string_cstr && count && assign && string_ctor && string_dtor &&
          to_string && refer_string, "Json2 entry points are registered");
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

    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
