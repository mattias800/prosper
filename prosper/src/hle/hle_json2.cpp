#include "hle_json2.hpp"
#include "dispatch.hpp"

#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace prosper {
namespace {

#define JHLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                                         uint64_t a3, uint64_t a4, uint64_t a5)
template <typename T> T* ptr(uint64_t value) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(value));
}

enum class ValueType : uint32_t { Null, Boolean, Integer, Unsigned, Real, String, Array, Object };
struct Value;
struct String { std::string* host = nullptr; };
struct Array { std::vector<Value*>* host = nullptr; };
struct Object { std::map<std::string, Value*>* host = nullptr; };
struct InitParameter2 {
    void* allocator = nullptr;
    void* user_data = nullptr;
    size_t file_buffer_size = 0;
    uint32_t special_float_format_type = 0;
    uint32_t reserved[3]{};
};
static_assert(sizeof(InitParameter2) == 0x28, "libSceJson2 InitParameter2 ABI");
struct Value {
    Value* parent = nullptr;
    void* root_parameter = nullptr;
    union Data {
        bool boolean;
        int64_t integer;
        uint64_t unsigned_integer;
        double real;
        String* string;
        Array* array;
        Object* object;
        Data() : unsigned_integer(0) {}
    } data;
    uint8_t reserved[4]{};
    ValueType type = ValueType::Null;
};
static_assert(sizeof(Value) == 32 && offsetof(Value, data) == 16 && offsetof(Value, type) == 28,
              "libSceJson2 Value ABI");

constexpr uint64_t JSON_PARSE_ERROR = 0x80848101u;
constexpr uint64_t JSON_ARGUMENT_ERROR = 0x80848120u;

bool logging() {
    static const bool enabled = std::getenv("PROSPER_JSONLOG") != nullptr;
    return enabled;
}

void clear(Value* value);
void destroy(Value* value) { if (value) { clear(value); delete value; } }

void clear(Value* value) {
    if (!value) return;
    if (value->type == ValueType::String && value->data.string) {
        delete value->data.string->host; delete value->data.string;
    } else if (value->type == ValueType::Array && value->data.array) {
        if (value->data.array->host) for (Value* child : *value->data.array->host) destroy(child);
        delete value->data.array->host; delete value->data.array;
    } else if (value->type == ValueType::Object && value->data.object) {
        if (value->data.object->host)
            for (auto& entry : *value->data.object->host) destroy(entry.second);
        delete value->data.object->host; delete value->data.object;
    }
    value->parent = nullptr; value->root_parameter = nullptr; value->data.unsigned_integer = 0;
    std::memset(value->reserved, 0, sizeof(value->reserved)); value->type = ValueType::Null;
}

bool clone(Value* destination, const Value* source) {
    if (!destination || !source) return false;
    destination->type = source->type;
    switch (source->type) {
        case ValueType::Null: break;
        case ValueType::Boolean: destination->data.boolean = source->data.boolean; break;
        case ValueType::Integer: destination->data.integer = source->data.integer; break;
        case ValueType::Unsigned: destination->data.unsigned_integer = source->data.unsigned_integer; break;
        case ValueType::Real: destination->data.real = source->data.real; break;
        case ValueType::String:
            destination->data.string = new String{new std::string(
                source->data.string && source->data.string->host ? *source->data.string->host : "")};
            break;
        case ValueType::Array:
            destination->data.array = new Array{new std::vector<Value*>};
            if (source->data.array && source->data.array->host) {
                for (const Value* source_child : *source->data.array->host) {
                    auto* child = new Value;
                    if (!clone(child, source_child)) { destroy(child); return false; }
                    child->parent = destination;
                    destination->data.array->host->push_back(child);
                }
            }
            break;
        case ValueType::Object:
            destination->data.object = new Object{new std::map<std::string, Value*>};
            if (source->data.object && source->data.object->host) {
                for (const auto& [key, source_child] : *source->data.object->host) {
                    auto* child = new Value;
                    if (!clone(child, source_child)) { destroy(child); return false; }
                    child->parent = destination;
                    destination->data.object->host->emplace(key, child);
                }
            }
            break;
    }
    return true;
}

void escape_json_string(std::string& output, std::string_view input) {
    static constexpr char hex[] = "0123456789abcdef";
    output.push_back('"');
    for (unsigned char c : input) {
        switch (c) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c < 0x20) {
                    output += "\\u00";
                    output.push_back(hex[c >> 4]); output.push_back(hex[c & 15]);
                } else output.push_back(static_cast<char>(c));
                break;
        }
    }
    output.push_back('"');
}

void serialize(std::string& output, const Value* value) {
    if (!value) { output += "null"; return; }
    char number[64]{};
    switch (value->type) {
        case ValueType::Null: output += "null"; break;
        case ValueType::Boolean: output += value->data.boolean ? "true" : "false"; break;
        case ValueType::Integer: {
            const auto result = std::to_chars(number, number + sizeof(number), value->data.integer);
            output.append(number, result.ptr); break;
        }
        case ValueType::Unsigned: {
            const auto result = std::to_chars(number, number + sizeof(number), value->data.unsigned_integer);
            output.append(number, result.ptr); break;
        }
        case ValueType::Real: {
            const auto result = std::to_chars(number, number + sizeof(number), value->data.real,
                                              std::chars_format::general,
                                              std::numeric_limits<double>::max_digits10);
            if (result.ec == std::errc{}) output.append(number, result.ptr); else output += "null";
            break;
        }
        case ValueType::String:
            escape_json_string(output, value->data.string && value->data.string->host
                                           ? *value->data.string->host : std::string_view{});
            break;
        case ValueType::Array: {
            output.push_back('['); bool first = true;
            if (value->data.array && value->data.array->host) {
                for (const Value* child : *value->data.array->host) {
                    if (!first) output.push_back(','); first = false; serialize(output, child);
                }
            }
            output.push_back(']'); break;
        }
        case ValueType::Object: {
            output.push_back('{'); bool first = true;
            if (value->data.object && value->data.object->host) {
                for (const auto& [key, child] : *value->data.object->host) {
                    if (!first) output.push_back(','); first = false;
                    escape_json_string(output, key); output.push_back(':'); serialize(output, child);
                }
            }
            output.push_back('}'); break;
        }
    }
}

Value& null_value() { static Value value; return value; }
String& empty_string() { static String value{new std::string}; return value; }
Array& empty_array() { static Array value{new std::vector<Value*>}; return value; }
Object& empty_object() { static Object value{new std::map<std::string, Value*>}; return value; }

class Parser {
public:
    Parser(const char* data, size_t size): cursor_(data), end_(data + size) {}
    bool parse(Value* output) { space(); return value(output) && (space(), cursor_ == end_); }

private:
    const char* cursor_;
    const char* end_;

    void space() {
        while (cursor_ != end_ && (*cursor_ == ' ' || *cursor_ == '\t' ||
               *cursor_ == '\r' || *cursor_ == '\n')) ++cursor_;
    }
    bool take(char wanted) {
        if (cursor_ == end_ || *cursor_ != wanted) return false;
        ++cursor_; return true;
    }
    bool literal(std::string_view text) {
        if (static_cast<size_t>(end_ - cursor_) < text.size() ||
            std::memcmp(cursor_, text.data(), text.size()) != 0) return false;
        cursor_ += text.size(); return true;
    }
    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }
    bool code_unit(uint32_t& result) {
        if (end_ - cursor_ < 4) return false;
        result = 0;
        for (unsigned i = 0; i < 4; ++i) {
            int digit = hex(*cursor_++); if (digit < 0) return false;
            result = (result << 4) | static_cast<uint32_t>(digit);
        }
        return true;
    }
    static void utf8(std::string& output, uint32_t codepoint) {
        if (codepoint <= 0x7f) output.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }
    bool string(std::string& output) {
        if (!take('"')) return false;
        output.clear();
        while (cursor_ != end_) {
            unsigned char c = static_cast<unsigned char>(*cursor_++);
            if (c == '"') return true;
            if (c < 0x20) return false;
            if (c != '\\') { output.push_back(static_cast<char>(c)); continue; }
            if (cursor_ == end_) return false;
            switch (*cursor_++) {
                case '"': output.push_back('"'); break; case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break; case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break; case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break; case 't': output.push_back('\t'); break;
                case 'u': {
                    uint32_t high = 0; if (!code_unit(high)) return false;
                    uint32_t point = high;
                    if (high >= 0xd800 && high <= 0xdbff) {
                        if (end_ - cursor_ < 6 || cursor_[0] != '\\' || cursor_[1] != 'u') return false;
                        cursor_ += 2; uint32_t low = 0;
                        if (!code_unit(low) || low < 0xdc00 || low > 0xdfff) return false;
                        point = 0x10000 + ((high - 0xd800) << 10) + low - 0xdc00;
                    } else if (high >= 0xdc00 && high <= 0xdfff) return false;
                    utf8(output, point); break;
                }
                default: return false;
            }
        }
        return false;
    }
    bool number(Value* output) {
        const char* begin = cursor_;
        if (cursor_ != end_ && *cursor_ == '-') ++cursor_;
        if (cursor_ == end_) return false;
        if (*cursor_ == '0') ++cursor_;
        else {
            if (*cursor_ < '1' || *cursor_ > '9') return false;
            while (cursor_ != end_ && std::isdigit(static_cast<unsigned char>(*cursor_))) ++cursor_;
        }
        bool is_real = false;
        if (cursor_ != end_ && *cursor_ == '.') {
            is_real = true; ++cursor_; const char* digits = cursor_;
            while (cursor_ != end_ && std::isdigit(static_cast<unsigned char>(*cursor_))) ++cursor_;
            if (digits == cursor_) return false;
        }
        if (cursor_ != end_ && (*cursor_ == 'e' || *cursor_ == 'E')) {
            is_real = true; ++cursor_;
            if (cursor_ != end_ && (*cursor_ == '+' || *cursor_ == '-')) ++cursor_;
            const char* digits = cursor_;
            while (cursor_ != end_ && std::isdigit(static_cast<unsigned char>(*cursor_))) ++cursor_;
            if (digits == cursor_) return false;
        }
        std::string_view token(begin, static_cast<size_t>(cursor_ - begin));
        if (is_real) {
            std::string temporary(token); char* parsed_end = nullptr;
            output->data.real = std::strtod(temporary.c_str(), &parsed_end);
            if (parsed_end != temporary.c_str() + temporary.size()) return false;
            output->type = ValueType::Real; return true;
        }
        if (!token.empty() && token.front() == '-') {
            auto result = std::from_chars(token.data(), token.data() + token.size(), output->data.integer);
            if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) return false;
            output->type = ValueType::Integer; return true;
        }
        auto result = std::from_chars(token.data(), token.data() + token.size(), output->data.unsigned_integer);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) return false;
        output->type = ValueType::Unsigned; return true;
    }
    bool array(Value* output) {
        if (!take('[')) return false;
        output->type = ValueType::Array; output->data.array = new Array{new std::vector<Value*>};
        space(); if (take(']')) return true;
        for (;;) {
            Value* child = new Value; if (!value(child)) { destroy(child); return false; }
            child->parent = output; output->data.array->host->push_back(child); space();
            if (take(']')) return true; if (!take(',')) return false; space();
        }
    }
    bool object(Value* output) {
        if (!take('{')) return false;
        output->type = ValueType::Object; output->data.object = new Object{new std::map<std::string, Value*>};
        space(); if (take('}')) return true;
        for (;;) {
            std::string key; if (!string(key)) return false; space();
            if (!take(':')) return false; space();
            Value* child = new Value; if (!value(child)) { destroy(child); return false; }
            child->parent = output;
            auto [position, inserted] = output->data.object->host->emplace(std::move(key), child);
            if (!inserted) { destroy(position->second); position->second = child; }
            space(); if (take('}')) return true; if (!take(',')) return false; space();
        }
    }
    bool value(Value* output) {
        if (!output || cursor_ == end_) return false;
        if (*cursor_ == '{') return object(output); if (*cursor_ == '[') return array(output);
        if (*cursor_ == '"') {
            std::string text; if (!string(text)) return false; output->type = ValueType::String;
            output->data.string = new String{new std::string(std::move(text))}; return true;
        }
        if (*cursor_ == 't' && literal("true")) {
            output->type = ValueType::Boolean; output->data.boolean = true; return true;
        }
        if (*cursor_ == 'f' && literal("false")) {
            output->type = ValueType::Boolean; output->data.boolean = false; return true;
        }
        if (*cursor_ == 'n' && literal("null")) return true;
        return number(output);
    }
};

std::string& text(String* string) {
    static std::string fallback;
    if (!string) return fallback;
    if (!string->host) string->host = new std::string;
    return *string->host;
}
const std::string& text(const String* string) {
    static const std::string fallback;
    return string && string->host ? *string->host : fallback;
}

JHLE(noop_self) { return a0; }
JHLE(noop_zero) { return 0; }
JHLE(init_parameter2_ctor) {
    if (auto* parameter = ptr<InitParameter2>(a0)) *parameter = InitParameter2{};
    return a0;
}
JHLE(init_parameter2_set_allocator) {
    if (auto* parameter = ptr<InitParameter2>(a0)) {
        parameter->allocator = ptr<void>(a1);
        parameter->user_data = ptr<void>(a2);
    }
    return 0;
}
JHLE(init_parameter2_set_file_buffer_size) {
    if (auto* parameter = ptr<InitParameter2>(a0)) parameter->file_buffer_size = a1;
    return 0;
}
JHLE(value_ctor) { if (auto* value = ptr<Value>(a0)) *value = Value{}; return a0; }
JHLE(value_dtor) { clear(ptr<Value>(a0)); return 0; }
JHLE(value_assign) {
    auto* destination = ptr<Value>(a0); auto* source = ptr<const Value>(a1);
    if (!destination || !source || destination == source) return a0;
    Value replacement;
    if (!clone(&replacement, source)) { clear(&replacement); return a0; }
    clear(destination); *destination = replacement;
    if (destination->type == ValueType::Array && destination->data.array && destination->data.array->host)
        for (Value* child : *destination->data.array->host) child->parent = destination;
    if (destination->type == ValueType::Object && destination->data.object && destination->data.object->host)
        for (auto& entry : *destination->data.object->host) entry.second->parent = destination;
    return a0;
}
JHLE(set_bool) {
    if (auto* value = ptr<Value>(a0)) { clear(value); value->type = ValueType::Boolean; value->data.boolean = a1 != 0; }
    return 0;
}
JHLE(set_int) {
    if (auto* value = ptr<Value>(a0)) { clear(value); value->type = ValueType::Integer; value->data.integer = static_cast<int64_t>(a1); }
    return 0;
}
JHLE(set_uint) {
    if (auto* value = ptr<Value>(a0)) { clear(value); value->type = ValueType::Unsigned; value->data.unsigned_integer = a1; }
    return 0;
}
JHLE(set_double) {
    // The current integer-only HLE entry ABI does not expose XMM0. Preserve the correct type; parsing
    // (the path used by Astro) supplies real double payloads directly.
    if (auto* value = ptr<Value>(a0)) { clear(value); value->type = ValueType::Real; value->data.real = 0.0; }
    return 0;
}
JHLE(set_string) {
    if (auto* value = ptr<Value>(a0)) {
        clear(value); value->type = ValueType::String;
        value->data.string = new String{new std::string(text(ptr<const String>(a1)))};
    }
    return 0;
}
JHLE(set_type) {
    auto* value = ptr<Value>(a0); if (!value) return 0; clear(value);
    const uint32_t raw = static_cast<uint32_t>(a1);
    if (raw > static_cast<uint32_t>(ValueType::Object)) return 0;
    value->type = static_cast<ValueType>(raw);
    if (value->type == ValueType::String) value->data.string = new String{new std::string};
    if (value->type == ValueType::Array) value->data.array = new Array{new std::vector<Value*>};
    if (value->type == ValueType::Object) value->data.object = new Object{new std::map<std::string, Value*>};
    return 0;
}
JHLE(string_cstring_ctor) {
    auto* string = ptr<String>(a0); const char* source = ptr<const char>(a1);
    if (string) string->host = new std::string(source ? source : ""); return a0;
}
JHLE(string_ctor) { if (auto* string = ptr<String>(a0)) string->host = new std::string; return a0; }
JHLE(string_dtor) {
    if (auto* string = ptr<String>(a0)) { delete string->host; string->host = nullptr; } return 0;
}
JHLE(object_ctor) {
    if (auto* object = ptr<Object>(a0)) object->host = new std::map<std::string, Value*>;
    return a0;
}
JHLE(object_dtor) {
    if (auto* object = ptr<Object>(a0)) {
        if (object->host) for (auto& [key, child] : *object->host) destroy(child);
        delete object->host;
        object->host = nullptr;
    }
    return 0;
}
JHLE(object_index_string) {
    auto* object = ptr<Object>(a0);
    const auto* key = ptr<const String>(a1);
    if (!object) return reinterpret_cast<uintptr_t>(&null_value());
    if (!object->host) object->host = new std::map<std::string, Value*>;
    auto [position, inserted] = object->host->try_emplace(text(key), nullptr);
    if (inserted || !position->second) position->second = new Value;
    return reinterpret_cast<uintptr_t>(position->second);
}
JHLE(object_empty) {
    const auto* object = ptr<const Object>(a0);
    return !object || !object->host || object->host->empty();
}
JHLE(value_string_ctor) {
    auto* value = ptr<Value>(a0);
    if (value) {
        *value = Value{};
        value->type = ValueType::String;
        value->data.string = new String{new std::string(text(ptr<const String>(a1)))};
    }
    return a0;
}
bool set_value_from_object(Value* value, const Object* object, bool clear_first) {
    if (!value) return false;
    Value source;
    source.type = ValueType::Object;
    source.data.object = const_cast<Object*>(object ? object : &empty_object());
    Value replacement;
    if (!clone(&replacement, &source)) return false;
    if (clear_first) clear(value);
    *value = replacement;
    if (value->data.object && value->data.object->host)
        for (auto& [key, child] : *value->data.object->host) child->parent = value;
    return true;
}
JHLE(value_object_ctor) {
    auto* value = ptr<Value>(a0);
    if (value) {
        *value = Value{};
        set_value_from_object(value, ptr<const Object>(a1), false);
    }
    return a0;
}
JHLE(value_set_object) {
    set_value_from_object(ptr<Value>(a0), ptr<const Object>(a1), true);
    return 0;
}
JHLE(parse) {
    auto* output = ptr<Value>(a0); const char* source = ptr<const char>(a1);
    if (!output || !source) return JSON_ARGUMENT_ERROR;
    clear(output); Parser parser(source, static_cast<size_t>(a2)); const bool ok = parser.parse(output);
    if (!ok) clear(output);
    if (logging()) std::fprintf(stderr, "[json2] parse bytes=%llu ok=%d type=%u\n",
        (unsigned long long)a2, (int)ok, static_cast<unsigned>(output->type));
    return ok ? 0 : JSON_PARSE_ERROR;
}
JHLE(index_string) {
    auto* value = ptr<Value>(a0); const char* key = ptr<const char>(a1); Value* result = &null_value();
    if (value && value->type == ValueType::Object && value->data.object && value->data.object->host) {
        const auto found = value->data.object->host->find(key ? key : "");
        if (found != value->data.object->host->end()) result = found->second;
    }
    if (logging()) std::fprintf(stderr, "[json2] index key='%s' -> type=%u\n", key ? key : "",
                                static_cast<unsigned>(result->type));
    return reinterpret_cast<uintptr_t>(result);
}
JHLE(index_wrapped_string) {
    auto* value = ptr<Value>(a0); auto* key = ptr<const String>(a1); Value* result = &null_value();
    if (value && value->type == ValueType::Object && value->data.object && value->data.object->host) {
        const auto found = value->data.object->host->find(text(key));
        if (found != value->data.object->host->end()) result = found->second;
    }
    return reinterpret_cast<uintptr_t>(result);
}
JHLE(refer_wrapped_string) {
    auto* value = ptr<Value>(a0); auto* key = ptr<const String>(a1);
    if (!value || value->type != ValueType::Object || !value->data.object ||
        !value->data.object->host) return 0;
    const auto found = value->data.object->host->find(text(key));
    return found == value->data.object->host->end() ? 0
        : reinterpret_cast<uintptr_t>(found->second);
}
JHLE(index_uint) {
    auto* value = ptr<Value>(a0); Value* result = &null_value();
    if (value && value->type == ValueType::Array && value->data.array && value->data.array->host &&
        a1 < value->data.array->host->size()) result = (*value->data.array->host)[static_cast<size_t>(a1)];
    return reinterpret_cast<uintptr_t>(result);
}
JHLE(refer_uint) {
    auto* value = ptr<Value>(a0);
    if (!value || value->type != ValueType::Array || !value->data.array ||
        !value->data.array->host || a1 >= value->data.array->host->size()) return 0;
    return reinterpret_cast<uintptr_t>((*value->data.array->host)[static_cast<size_t>(a1)]);
}
JHLE(get_type) { auto* value = ptr<const Value>(a0); return value ? static_cast<uint32_t>(value->type) : 0; }
JHLE(get_boolean) {
    static const bool fallback = false; auto* value = ptr<const Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::Boolean ? &value->data.boolean : &fallback);
}
JHLE(get_integer) {
    static const int64_t fallback = 0; auto* value = ptr<const Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::Integer ? &value->data.integer : &fallback);
}
JHLE(get_unsigned) {
    static const uint64_t fallback = 0; auto* value = ptr<const Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::Unsigned
                                           ? &value->data.unsigned_integer : &fallback);
}
JHLE(get_real) {
    static const double fallback = 0; auto* value = ptr<const Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::Real ? &value->data.real : &fallback);
}
JHLE(get_string) {
    auto* value = ptr<Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::String ? value->data.string : &empty_string());
}
JHLE(get_array) {
    auto* value = ptr<Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::Array ? value->data.array : &empty_array());
}
JHLE(get_object) {
    auto* value = ptr<Value>(a0);
    return reinterpret_cast<uintptr_t>(value && value->type == ValueType::Object ? value->data.object : &empty_object());
}
JHLE(count) {
    auto* value = ptr<const Value>(a0);
    if (value && value->type == ValueType::Array && value->data.array && value->data.array->host)
        return value->data.array->host->size();
    if (value && value->type == ValueType::Object && value->data.object && value->data.object->host)
        return value->data.object->host->size();
    return 0;
}
JHLE(string_cstr) { return reinterpret_cast<uintptr_t>(text(ptr<const String>(a0)).c_str()); }
JHLE(string_length) { return text(ptr<const String>(a0)).size(); }
JHLE(value_to_string) {
    auto* output = ptr<String>(a1); if (!output) return JSON_ARGUMENT_ERROR;
    std::string& destination = text(output); destination.clear(); serialize(destination, ptr<const Value>(a0));
    return 0;
}
JHLE(array_size) {
    auto* array = ptr<const Array>(a0); return array && array->host ? array->host->size() : 0;
}

#undef JHLE
} // namespace

void register_json2_hle() {
    auto reg = [](const char* nid, HleFn fn, const char* name) { Hle::register_fn(nid, fn, name); };
    reg("-hJRce8wn1U", noop_self, "sceJsonMemAllocator::ctor");
    reg("qBMjqyBn3OM", value_ctor, "sceJsonValue::ctor");
    reg("-wa17B7TGnw", value_ctor, "sceJsonValue::ctor");
    reg("WTtYf+cNnXI", value_dtor, "sceJsonValue::dtor");
    reg("0eUrW9JAxM0", value_dtor, "sceJsonValue::dtor");
    reg("4zrm6VrgIAw", value_assign, "sceJsonValue::assign");
    reg("5yHuiWXo2gg", set_bool, "sceJsonValue::setBool");
    reg("QxVVYhP-mvg", set_int, "sceJsonValue::setInt");
    reg("SIe1ZmW7e7s", set_uint, "sceJsonValue::setUInt");
    reg("BSmWDIkV4w4", set_double, "sceJsonValue::setDouble");
    reg("6l3Bv2gysNc", set_string, "sceJsonValue::setString");
    reg("IKQimvG9Wqs", set_type, "sceJsonValue::setType");
    reg("9KUZFjI1IxA", string_cstring_ctor, "sceJsonString::cstringCtor");
    reg("qSmqLXXCPas", string_ctor, "sceJsonString::ctor");
    reg("cG1VE2HMl6c", string_dtor, "sceJsonString::dtor");
    reg("cK6bYHf-Q5E", noop_self, "sceJsonInitializer::ctor");
    reg("Cxwy7wHq4J0", noop_zero, "sceJsonInitializer::initializeV1");
    reg("00oCq0RwSAY", noop_zero, "sceJsonInitializer::setGlobalNullAccessCallback");
    reg("S5JxQnoGF3E", parse, "sceJsonParser::parse");
    reg("HwDt5lD9Bfo", index_string, "sceJsonValue::indexString");
    reg("clF7J7N9xXE", index_wrapped_string, "sceJsonValue::indexStringObject");
    reg("MsMOdxWfbwQ", refer_wrapped_string, "sceJsonValue::getValueString");
    reg("wLsJlmgEIaI", refer_wrapped_string, "sceJsonValue::referValueString");
    reg("XlWbvieLj2M", index_uint, "sceJsonValue::indexUInt");
    reg("0YqYAoO-+Uo", refer_uint, "sceJsonValue::getValueUInt");
    reg("gLzCc67aTbw", refer_uint, "sceJsonValue::referValueUInt");
    reg("SHtAad20YYM", get_type, "sceJsonValue::getType");
    reg("zTwZdI8AZ5Y", get_boolean, "sceJsonValue::getBoolean");
    reg("DIxvoy7Ngvk", get_integer, "sceJsonValue::getInteger");
    reg("sn4HNCtNRzY", get_unsigned, "sceJsonValue::getUInteger");
    reg("3qrge7L-AU4", get_real, "sceJsonValue::getReal");
    reg("epJ6x2LV0kU", get_string, "sceJsonValue::getString");
    reg("ONT8As5R1ug", get_array, "sceJsonValue::getArray");
    reg("IlsmvBtMkak", get_object, "sceJsonValue::getObject");
    reg("RBw+4NukeGQ", count, "sceJsonValue::count");
    reg("L1KAkYWml-M", string_cstr, "sceJsonString::c_str");
    reg("EUH+EmT-v9E", string_length, "sceJsonString::length");
    reg("Ncel8t2Rrpc", value_to_string, "sceJsonValue::toString");
    reg("R7FDWtcN6f8", value_to_string, "sceJsonValue::serialize");
    reg("rQGJeNjOuUk", array_size, "sceJsonArray::size");

    // PS5 SDK 3.20 aliases exercised by Sonic Origins. The existing implementation originally
    // registered the older PS4 spellings only, leaving these constructors/accessors as generic
    // success stubs: caller-owned Object and Value storage then remained uninitialized. Names and
    // ABIs are verified against the local PS5 3.20 libSceJson2 stub corpus and Kyty's Json2 model.
    reg("WSOuge5IsCg", init_parameter2_ctor, "sceJsonInitParameter2::ctor");
    reg("I2QC8PYhJWY", init_parameter2_set_allocator, "sceJsonInitParameter2::setAllocator");
    reg("Eu95jmqn5Rw", init_parameter2_set_file_buffer_size,
        "sceJsonInitParameter2::setFileBufferSize");
    reg("IXW-z8pggfg", noop_zero, "sceJsonInitializer::initializeV2");
    reg("OJPTonqdg0I", object_ctor, "sceJsonObject::ctor");
    reg("5JmzZt8twAo", object_dtor, "sceJsonObject::dtor");
    reg("ERuf9y0DY84", object_index_string, "sceJsonObject::indexString");
    reg("i2l3IYvQ9UE", object_empty, "sceJsonObject::empty");
    reg("sZIoMRGO+jk", value_string_ctor, "sceJsonValue::stringCtor");
    reg("3xUXnmUkXfo", value_object_ctor, "sceJsonValue::objectCtor");
    reg("dFCphqnd+a4", value_set_object, "sceJsonValue::setObject");
}
} // namespace prosper
