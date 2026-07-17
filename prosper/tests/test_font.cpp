#include "../src/hle/dispatch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    register_builtin_hle();
    auto create = Hle::lookup("n590hj5Oe-k");
    auto open_set = Hle::lookup("cKYtVmeSTcw");
    auto metrics = Hle::lookup("IQtleGLL5pQ");
    auto text_init = Hle::lookup("oaJ1BpN2FQk");
    auto create_string = Hle::lookup("MO24vDhmS4E");
    auto chars = Hle::lookup("Avv7OApgCJk");
    CHECK(create && open_set && metrics && text_init && create_string && chars,
          "Astro Font HLE surface is registered");

    uint8_t mem[64]{};
    void* library = nullptr;
    CHECK(create((uint64_t)(uintptr_t)mem, 0, 0, (uint64_t)(uintptr_t)&library, 0, 0) == 0 && library,
          "CreateLibraryWithEdition writes a non-null library handle");
    void* font = nullptr;
    CHECK(open_set((uint64_t)(uintptr_t)library, 0, 0, 0,
                   (uint64_t)(uintptr_t)&font, 0) == 0 && font,
          "OpenFontSet writes a non-null font handle");
    float glyph[8]{};
    CHECK(metrics((uint64_t)(uintptr_t)font, 'A', (uint64_t)(uintptr_t)glyph, 0, 0, 0) == 0 &&
          glyph[0] > 0.0f && glyph[1] > 0.0f && glyph[4] > 0.0f,
          "glyph metrics are deterministic and non-zero");

    uint8_t source[0x60]; std::memset(source, 0xcc, sizeof(source));
    const char text[] = "ASTRO";
    CHECK(text_init((uint64_t)(uintptr_t)source, (uint64_t)(uintptr_t)text, sizeof(text), 0, 0, 0) == 0,
          "TextSourceInit initializes the public source layout");
    void* string = nullptr;
    CHECK(create_string((uint64_t)(uintptr_t)mem, (uint64_t)(uintptr_t)source, 0,
                        (uint64_t)(uintptr_t)&string, 0, 0) == 0 && string,
          "CreateString writes an opaque string handle");
    uint32_t count = 0xdeadbeef;
    CHECK(chars((uint64_t)(uintptr_t)string, (uint64_t)(uintptr_t)&count, 0, 0, 0, 0) == 0 && count == 0,
          "fallback string exposes a safe empty character range");

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
