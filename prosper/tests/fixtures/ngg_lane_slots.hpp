#pragma once
// Project-owned NGG lane-slot programs shared by CPU production-compiler tests and strict SPIR-V
// representatives. Scalar slots have distinct raw values; poison launch VGPRs cannot be the oracle.
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace prosper::test::ngg_slots {
inline constexpr uint32_t kFirst = 0x12345670u;
inline constexpr uint32_t kSecond = 0x76543210u;
inline constexpr uint32_t kExecFull = 0xbefe04c1u;

inline void append(std::vector<uint32_t>& code, std::initializer_list<uint32_t> words) {
    code.insert(code.end(), words);
}
inline std::vector<uint32_t> begin() {
    return {kExecFull, 0x7e120280u};
}
inline void scalar(std::vector<uint32_t>& code, uint32_t reg, uint32_t value) {
    append(code, {0xbe8003ffu | (reg << 16), value});
}
inline void write(std::vector<uint32_t>& code, uint32_t vgpr = 4, uint32_t lane = 7,
                  uint32_t source = 20) {
    append(code, {0xd7610000u | vgpr, source | ((128u + lane) << 9)});
}
inline void read(std::vector<uint32_t>& code, uint32_t vgpr = 4, uint32_t lane = 7,
                 uint32_t dest = 21) {
    append(code, {0xd7600000u | dest, (256u + vgpr) | ((128u + lane) << 9)});
}
inline size_t branch(std::vector<uint32_t>& code, uint32_t opcode) {
    const size_t pc = code.size();
    code.push_back(0xbf800000u | (opcode << 16));
    return pc;
}
inline void target(std::vector<uint32_t>& code, size_t pc, size_t destination) {
    code[pc] |=
        static_cast<uint16_t>(static_cast<int32_t>(destination) - static_cast<int32_t>(pc) - 1);
}
inline void finish(std::vector<uint32_t>& code) {
    append(code, {0x7e120215u, 0xb07c3005u, 0xbf900009u, 0xf8000941u, 0x00000009u, 0xf80000cfu,
                  0x03020100u, 0xbf810000u});
}
inline std::vector<uint32_t> straight(uint32_t vgpr = 4, uint32_t lane = 7) {
    auto code = begin();
    scalar(code, 20, kFirst);
    write(code, vgpr, lane);
    read(code, vgpr, lane);
    finish(code);
    return code;
}
inline std::vector<uint32_t> joined(bool both_arms = true) {
    auto code = begin();
    // Different physical waves take different scalar arms; s3's GS-wave-id field is not read.
    append(code, {0x90169803u, 0x87168116u, 0xbf068016u});   // s22=(s3>>24)&1; cmp s22,0
    const size_t to_else = branch(code, 0x04);   // SCC0
    scalar(code, 20, kFirst);
    write(code);
    const size_t to_join = branch(code, 0x02);
    target(code, to_else, code.size());
    scalar(code, 20, kSecond);
    if (both_arms) write(code);
    target(code, to_join, code.size());
    read(code);
    finish(code);
    return code;
}
inline std::vector<uint32_t> counted(bool preheader_slot = true) {
    auto code = begin();
    scalar(code, 20, kFirst);
    if (preheader_slot) write(code);
    scalar(code, 22, 3);
    const size_t header = code.size();
    read(code);
    code.push_back(0x80148115u);   // s_add_u32 s20,s21,1
    write(code);
    append(code, {0x80968116u, 0xbf068016u});   // s_sub_u32 s22,s22,1; cmp s22,0
    const size_t again = branch(code, 0x04);
    target(code, again, header);
    read(code);
    finish(code);
    return code;
}
}   // namespace prosper::test::ngg_slots
