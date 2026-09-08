#include "shared/compute/storage_write_mask_spirv.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>

using namespace prosper::frontend;
namespace {
using Words = std::vector<uint32_t>;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
void put(Words& words, uint32_t op, std::initializer_list<uint32_t> args) {
    words.push_back((uint32_t(args.size() + 1) << 16) | op);
    words.insert(words.end(), args.begin(), args.end());
}
struct Fixture {
    Words words;
    StorageWriteMaskTarget target;
    uint32_t image_type = 0, image_variable = 0, load = 0, coordinate = 0;
};
Fixture fixture(uint32_t dim, bool arrayed, std::array<uint32_t, 3> xyz,
                bool signed_coord = true, bool alias = false, bool write = true,
                uint32_t version = 0x00010300u) {
    Fixture f;
    f.target = {0, 5, 20, 7, dim ? 3u : 1u, (arrayed || dim == 2) ? 2u : 1u, arrayed};
    Words& w = f.words;
    w = {0x07230203u, version, 0, 100, 0};
    put(w, 17, {1}); // Shader
    put(w, 17, {56}); // StorageImageWriteWithoutFormat
    if (dim == 0) put(w, 17, {44}); // Image1D
    put(w, 14, {0, 1});
    if (version >= 0x00010400u) {
        if (alias) put(w, 15, {5, 40, 0x6e69616du, 0, 12, 13});
        else put(w, 15, {5, 40, 0x6e69616du, 0, 12});
    } else put(w, 15, {5, 40, 0x6e69616du, 0});
    put(w, 16, {40, 17, 1, 1, 1});
    put(w, 71, {12, 34, 0}); put(w, 71, {12, 33, 5});
    if (alias) { put(w, 71, {13, 34, 0}); put(w, 71, {13, 33, 6}); }
    put(w, 19, {1});
    put(w, 33, {2, 1});
    put(w, 21, {3, 32, 0});
    put(w, 21, {4, 32, 1});
    put(w, 23, {5, 3, 4});
    const uint32_t components = dim + 1 + uint32_t(arrayed);
    const uint32_t scalar = signed_coord ? 4 : 3;
    if (components > 1) put(w, 23, {6, scalar, components});
    put(w, 25, {10, 3, dim, 0, uint32_t(arrayed), 0, 2, 0});
    put(w, 32, {11, 0, 10});
    put(w, 43, {3, 14, 0x3f000000u});
    put(w, 44, {5, 15, 14, 14, 14, 14});
    for (uint32_t c = 0; c < components; ++c) put(w, 43, {scalar, 20 + c, xyz[c]});
    if (components == 2) put(w, 44, {6, 25, 20, 21});
    if (components == 3) put(w, 44, {6, 25, 20, 21, 22});
    put(w, 59, {11, 12, 0});
    if (alias) put(w, 59, {11, 13, 0});
    put(w, 54, {1, 40, 0, 2});
    put(w, 248, {41});
    put(w, 61, {10, 42, 12});
    const uint32_t coordinate = components == 1 ? 20 : 25;
    if (write) put(w, 99, {42, coordinate, 15});
    if (alias) { put(w, 61, {10, 43, 13}); put(w, 99, {43, coordinate, 15}); }
    put(w, 253, {}); put(w, 56, {});
    f.image_type = 10; f.image_variable = 12; f.load = 42; f.coordinate = coordinate;
    return f;
}
struct Atomic { uint32_t variable, index, scope, semantics, value; };
// Interpret the actual emitted integer SSA, not a separate copy of the intended
// coordinate formula. The source fixtures use constants, so no GPU is needed to
// establish which cell/value each generated atomic reaches, including OOB.
std::vector<Atomic> evaluate(const Words& words) {
    std::map<uint32_t, Words> values;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> pointers;
    std::vector<Atomic> atomics;
    for (size_t p = 5; p < words.size(); p += words[p] >> 16) {
        const uint32_t op = words[p] & 0xffffu, n = words[p] >> 16;
        auto v = [&](uint32_t offset) { return values.at(words[p + offset]).at(0); };
        if (op == 43) values[words[p + 2]] = {words[p + 3]};
        if (op == 44) {
            Words value;
            for (uint32_t i = 3; i < n; ++i) value.push_back(v(i));
            values[words[p + 2]] = value;
        }
        if (op == 81) values[words[p + 2]] = {values.at(words[p + 3]).at(words[p + 4])};
        if (op == 124) values[words[p + 2]] = {v(3)};
        if (op == 128) values[words[p + 2]] = {v(3) + v(4)};
        if (op == 132) values[words[p + 2]] = {v(3) * v(4)};
        if (op == 176) values[words[p + 2]] = {uint32_t(v(3) < v(4))};
        if (op == 167) values[words[p + 2]] = {uint32_t(v(3) && v(4))};
        if (op == 169) values[words[p + 2]] = {v(3) ? v(4) : v(5)};
        if (op == 65) {
            check(v(4) == 0, "mask accesses struct member zero");
            pointers[words[p + 2]] = {words[p + 3], v(5)};
        }
        if (op == 241) {
            const auto pointer = pointers.at(words[p + 3]);
            atomics.push_back({pointer.first, pointer.second, v(4), v(5), v(6)});
        }
    }
    return atomics;
}
std::vector<Words> instructions(const Words& words, uint32_t opcode) {
    std::vector<Words> found;
    for (size_t p = 5; p < words.size(); p += words[p] >> 16) {
        const uint32_t n = words[p] >> 16;
        if ((words[p] & 0xffffu) == opcode)
            found.emplace_back(words.begin() + p, words.begin() + p + n);
    }
    return found;
}
void reject(const Fixture& f, const char* label) {
    const auto result = instrument_storage_image_writes(f.words, std::span(&f.target, 1));
    check(!result && !result.error.empty() && result.words.empty(), label);
}
} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path output = argc == 2 ? argv[1] : "";
    if (!output.empty()) std::filesystem::create_directories(output);
    unsigned serial = 0;
    for (uint32_t dim = 0; dim <= 2; ++dim) for (bool arrayed : {false, true}) {
        if (dim == 2 && arrayed) continue;
        const uint32_t axes = dim + 1 + uint32_t(arrayed);
        const std::array<uint32_t, 3> limits{7, dim ? 3u : 2u, 2};
        for (bool signed_coord : {false, true}) for (uint32_t axis = 0; axis < axes; ++axis)
            for (uint32_t kind = 0; kind < 4; ++kind) {
                std::array<uint32_t, 3> coordinate{6, dim ? 2u : 1u, 1};
                coordinate[axis] = kind == 0 ? 0 : kind == 1 ? limits[axis] - 1 :
                                   kind == 2 ? limits[axis] : 0xffffffffu;
                auto f = fixture(dim, arrayed, coordinate, signed_coord);
                const auto result = instrument_storage_image_writes(f.words, std::span(&f.target, 1));
                check(bool(result), "supported coordinate fixture instruments");
                if (!result) { std::fprintf(stderr, "%s\n", result.error.c_str()); continue; }
                const auto atomics = evaluate(result.words);
                const uint32_t y = dim ? coordinate[1] : 0;
                const uint32_t z = dim == 2 ? coordinate[2] : arrayed ? coordinate[dim + 1] : 0;
                const uint32_t expected = coordinate[0] + 7 * (y + (dim ? 3 : 1) * z);
                check(atomics.size() == 1 && atomics[0].value == uint32_t(kind < 2) &&
                          atomics[0].index == (kind < 2 ? expected : 0),
                      "actual generated SSA marks exact valid texel, or safe zero for OOB");
                check(atomics[0].scope == 1 && atomics[0].semantics == 0,
                      "mask atomics use device scope with relaxed ordering");
                check(instructions(f.words, 99) == instructions(result.words, 99),
                      "original image stores and their same-value payloads are unchanged");
                for (uint32_t op : {245u, 246u, 247u, 248u, 249u, 250u})
                    check(instructions(f.words, op) == instructions(result.words, op),
                          "no phi, loop, merge, block or branch is rewritten");
                if (!output.empty()) {
                    std::ofstream file(output / ("mask-" + std::to_string(serial++) + ".spv"), std::ios::binary);
                    file.write(reinterpret_cast<const char*>(result.words.data()), result.words.size() * 4);
                }
            }
    }
    for (uint32_t version : {0x00010300u, 0x00010400u, 0x00010600u}) {
        auto alias = fixture(1, true, {2, 1, 1}, true, true, true, version);
        auto second = alias.target; second.image_binding = 6;
        const std::array targets{alias.target, second};
        const auto result = instrument_storage_image_writes(alias.words, targets);
        check(bool(result) && result.instrumented_writes == 2, "both aliases instrument");
        if (!result) continue;
        const auto atomics = evaluate(result.words);
        check(atomics.size() == 2 && atomics[0].variable == atomics[1].variable &&
                  atomics[0].index == 30 && atomics[1].index == 30,
              "alias stores share one exact-coordinate mask buffer");
        const auto before = instructions(alias.words, 15), after = instructions(result.words, 15);
        check(after[0].size() == before[0].size() + (version >= 0x00010400u ? 1 : 0),
              "SPIR-V 1.4+ entry interface includes precisely one shared mask");
        if (!output.empty()) {
            std::ofstream file(output / ("alias-" + std::to_string(version) + ".spv"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(result.words.data()), result.words.size() * 4);
        }
        second.width = 8;
        const std::array conflicting{alias.target, second};
        const auto bad = instrument_storage_image_writes(alias.words, conflicting);
        check(!bad && !bad.error.empty(), "alias extent disagreement rejects");
    }
    auto f = fixture(1, false, {1, 1, 0}, true, false, false);
    auto none = instrument_storage_image_writes(f.words, std::span(&f.target, 1));
    check(bool(none) && none.instrumented_writes == 0 && evaluate(none.words).empty(),
          "target with no static stores is valid and leaves its zero mask untouched");
    f = fixture(1, false, {1, 1, 0}); f.words[5] = 0; reject(f, "zero instruction length rejects");
    f = fixture(1, false, {1, 1, 0}); f.words.pop_back(); reject(f, "truncated instruction rejects");
    f = fixture(1, false, {1, 1, 0}); f.words[3] = 0xfffffff0u; reject(f, "fresh ID overflow rejects");
    f = fixture(1, false, {1, 1, 0}); f.target.mask_binding = 5; reject(f, "descriptor collision rejects");
    f = fixture(1, false, {1, 1, 0}); f.target.image_binding = 17; reject(f, "absent target rejects");
    f = fixture(1, false, {1, 1, 0}); f.target.arrayed = true; reject(f, "array view mismatch rejects");
    f = fixture(1, false, {1, 1, 0}); f.target.width = 0; reject(f, "empty extent rejects");
    f = fixture(2, false, {1, 1, 0}); f.target.width = 0xffffffffu; reject(f, "mask extent overflow rejects");
    f = fixture(1, false, {1, 1, 0});
    for (size_t p = 5; p < f.words.size(); p += f.words[p] >> 16)
        if ((f.words[p] & 0xffffu) == 99) f.words[p + 1] = 91;
    reject(f, "unresolved store image rejects instead of omitting a mark");
    f = fixture(1, false, {1, 1, 0});
    f.words.insert(f.words.end() - 2, {(6u << 16) | 60u, 11, 90, 12, 25, 20});
    reject(f, "image texel pointer atomics reject instead of leaving false untouched bits");
    std::printf("storage-write-mask SPIR-V: %d failures\n", failures);
    return failures ? 1 : 0;
}
