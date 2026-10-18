// Scalar-uniform ifs choose independent per-lane words. The hand-written source selection
// below is separate from the branch encoder, so a bad offset or arm order cannot self-confirm.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include <cstdint>
#include <functional>
#include <vector>
namespace prosper::test::ir::cfg_structured {
struct Case {
    uint32_t shape, choices;
};
class Stream {
public:
    std::vector<uint32_t> words;
    void move(uint32_t vgpr) { words.push_back(vop1(1, 1, 256u + vgpr)); }
    void choose(bool condition, const std::function<void()>& yes,
                const std::function<void()>& no = {}) {
        words.insert(words.end(), {0xbe800380u | uint32_t(!condition), 0xbf008000u});
        const auto branch = static_cast<uint32_t>(words.size());
        words.push_back(0);
        yes();
        if (no) {
            const auto jump = static_cast<uint32_t>(words.size());
            words.push_back(0);
            patch(branch, cfg_streams::kScc0);
            no();
            patch(jump, cfg_streams::kBranch);
        } else {
            patch(branch, cfg_streams::kScc0);
        }
    }

private:
    void patch(uint32_t at, uint32_t opcode) {
        words[at] = cfg_streams::sopp(opcode, static_cast<int>(words.size() - at - 1u));
    }
};
inline std::vector<uint32_t> words(Case c) {
    Stream s;
    const bool a = (c.choices & 1u) != 0, b = (c.choices & 2u) != 0;
    switch (c.shape) {
        case 0: s.choose(a, [&] { s.move(2); }); break;
        case 1: s.choose(a, [&] { s.move(2); }, [&] { s.move(3); }); break;
        case 2:
            s.choose(
                a, [&] { s.choose(b, [&] { s.move(2); }, [&] { s.move(3); }); },
                [&] { s.move(5); });
            break;
        case 3:
            s.choose(
                a, [&] { s.move(2); },
                [&] { s.choose(b, [&] { s.move(3); }, [&] { s.move(5); }); });
            break;
        case 4:
            s.choose(a, [&] { s.move(2); }, [&] { s.move(3); });
            s.choose(b, [&] { s.move(6); });
            break;
    }
    return s.words;
}
inline uint32_t source(Case c) {
    const bool a = (c.choices & 1u) != 0, b = (c.choices & 2u) != 0;
    switch (c.shape) {
        case 0: return a ? 0u : 4u;
        case 1: return a ? 0u : 1u;
        case 2: return a ? (b ? 0u : 1u) : 2u;
        case 3: return a ? 0u : (b ? 1u : 2u);
        case 4: return b ? 3u : (a ? 0u : 1u);
    }
    return 4;
}
inline LaneInputs inputs(uint32_t mask) {
    auto in = lane_inputs(0xc0620000u);
    in.s2.reserve(in.lanes());
    in.s3.reserve(in.lanes());
    for (uint32_t lane = 0; lane < in.lanes(); ++lane) {
        in.s2.push_back(in.s0[lane] ^ 0x5ac39e17u);
        in.s3.push_back(in.s1[lane] + 0x1739bde5u);
        if (mask == 0) in.active[lane] = true;
        if (mask == 2) in.active[lane] = false;
    }
    return in;
}
inline uint32_t expected(Case c, const LaneInputs& in, uint32_t lane) {
    if (!in.active[lane]) return in.old_d[lane];
    switch (source(c)) {
        case 0: return in.s0[lane];
        case 1: return in.s1[lane];
        case 2: return in.s2[lane];
        case 3: return in.s3[lane];
        default: return in.old_d[lane];
    }
}
} // namespace prosper::test::ir::cfg_structured
