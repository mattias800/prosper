#pragma once
// Independent, fail-closed integer SPIR-V interpreter for this test's ACTUAL
// recompile_compute modules. This is not a Vulkan interpreter or a device
// oracle. It executes the entrypoint, dispatcher/phis, typed memory and common
// barriers; unsupported operations, poison reads, invalid pointers and split
// barriers fail.
#include <array>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace bpermute_oracle {
struct Value {
  uint32_t type = 0;
  std::vector<uint32_t> words;
  bool defined = true;
};
struct Type {
  uint32_t op = 0;
  std::vector<uint32_t> a;
};
struct Pointer {
  uint32_t root = 0, offset = 0, type = 0;
};
struct Memory {
  uint32_t type = 0;
  std::vector<uint32_t> words;
  std::vector<bool> valid;
};
struct Inst {
  uint32_t op = 0;
  std::vector<uint32_t> a;
};
struct Lane {
  std::map<uint32_t, Value> v;
  std::map<uint32_t, Pointer> p;
  std::map<uint32_t, Memory> memory;
  size_t pc = 0, steps = 0;
  uint32_t label = 0, from = 0;
  bool done = false;
};
struct Interpreter {
  std::vector<Inst> ins;
  std::map<uint32_t, Type> types;
  std::map<uint32_t, Value> constants;
  std::map<uint32_t, uint32_t> builtins, bindings;
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> decorations;
  std::map<std::array<uint32_t, 3>, std::vector<uint32_t>> member_decorations;
  std::map<uint32_t, size_t> labels;
  std::map<uint32_t, Memory> shared;
  std::map<uint32_t, bool> readonly_storage;
  std::map<uint32_t, uint32_t> extra_writable_roots;   // binding -> storage variable
  std::vector<Inst> globals;
  // Optional fixture-owned words for the closed uint32 array PushConstant ABI. The default
  // supplies nothing: a declared block cannot read invented zeros or infer a launch value.
  std::vector<uint32_t> supplied_push_constants;
  uint32_t main = 0, storage = 0;
  // Default compute fixtures stay strict set 0. Actual NGG modules explicitly select their
  // separate set-2 launch/export ABI; no module words or resource ownership are rewritten.
  uint32_t storage_descriptor_set = 0;
  // Explicit legacy shell fixture ABI: binding 0/1 may carry raw bits in scalar float32 slots.
  // This permits loads, stores and bitcasts only; unsupported float arithmetic still fails.
  bool legacy_float_word_storage = false;
  void use_ngg_shell_storage() { storage_descriptor_set = 2; }
  size_t entry = 0;
  std::string error;
  // An explicit fixture contract, never inferred from a declared native builtin or vote.
  uint32_t native_subgroup_width = 0;
  // Fixture-supplied dispatch extent. Nonzero opts into exact (32,1,1) workgroups;
  // the module's own entrypoint LocalSize must agree before any builtin is supplied.
  uint32_t native32_workgroup_count = 0;
  std::map<uint32_t, std::vector<uint32_t>> local_sizes;
  std::set<uint32_t> specialized_local_sizes;
  std::vector<uint32_t> push_constants;
  struct SampledMip {
    uint32_t width = 0, height = 0;
    std::vector<std::array<uint32_t, 4>> texels;
  };
  uint32_t glsl = 0;
  // Independent fixture-owned texels for ONLY exact-center, integral-LOD nearest sampling.
  std::map<uint32_t, std::vector<SampledMip>> sampled_images;
  [[noreturn]] void fail(const std::string &what) {
    throw std::runtime_error(what);
  }
  const Type &ty(uint32_t id) {
    auto it = types.find(id);
    if (it == types.end())
      fail("unknown type");
    return it->second;
  }
  uint32_t scalar(uint32_t id) {
    const auto &t = ty(id);
    if (t.op == 20)
      return 1;
    if ((t.op == 21 || t.op == 22) && !t.a.empty() && t.a[0] == 32)
      return 1;
    if (t.op == 21 && t.a == std::vector<uint32_t>{64, 0}) return 2;
    if (t.op == 27 && t.a.size() == 1) return 1; // closed opaque sampled-image handle
    fail("not a 32-bit scalar");
  }
  uint32_t size(uint32_t id) {
    const auto &t = ty(id);
    if (t.op == 23 && t.a.size() == 2)
      return scalar(t.a[0]) * t.a[1];
    if (t.op == 28 && t.a.size() == 2)
      return size(t.a[0]) * constant(t.a[1]);
    if (t.op == 30 && t.a.size() == 1 && ty(t.a[0]).op != 29)
      return size(t.a[0]);
    if (t.op == 30 && t.a.size() == 2 && uint_type(t.a[0]) && t.a[0] == t.a[1])
        return 2; // closed scalar-u32 low/high product pair
    return scalar(id);
  }
  uint32_t constant(uint32_t id) {
    auto it = constants.find(id);
    if (it == constants.end() || it->second.words.size() != 1)
      fail("not a scalar constant");
    return it->second.words[0];
  }
  uint32_t pointee(uint32_t id) {
    const auto &t = ty(id);
    if (t.op != 32 || t.a.size() != 2)
      fail("not a pointer type");
    return t.a[1];
  }
  Memory make_memory(uint32_t type, uint32_t count) {
    return {type, std::vector<uint32_t>(count),
            std::vector<bool>(count, false)};
  }
  Value value(Lane &l, uint32_t id) {
    auto it = l.v.find(id);
    if (it != l.v.end())
      return it->second;
    auto c = constants.find(id);
    if (c != constants.end())
      return c->second;
    fail("unknown or poison value " + std::to_string(id));
  }
  uint32_t word(Lane &l, uint32_t id) {
    auto v = value(l, id);
    if (v.words.size() != 1 || !v.defined)
      fail("not defined scalar value");
    return v.words[0];
  }
  bool boolean(Lane &l, uint32_t id) {
    auto v = value(l, id);
    if (ty(v.type).op != 20 || v.words.size() != 1 || v.words[0] > 1 ||
        !v.defined)
      fail("not defined bool value");
    return v.words[0] != 0;
  }
  bool uint_type(uint32_t id) {
    const auto &t = ty(id);
    return t.op == 21 && t.a == std::vector<uint32_t>{32, 0};
  }
  bool unsigned_type(uint32_t id) {
    const auto &t = ty(id);
    return t.op == 21 && (t.a == std::vector<uint32_t>{32, 0} ||
                         t.a == std::vector<uint32_t>{64, 0});
  }
  uint64_t unsigned_value(const Value &v) {
    if (!unsigned_type(v.type) || v.words.size() != size(v.type) || !v.defined)
      fail("not defined unsigned integer");
    return v.words[0] | (v.words.size() == 2 ? uint64_t(v.words[1]) << 32 : 0);
  }
  void set_unsigned(Lane &l, uint32_t id, uint32_t type, uint64_t value, bool defined = true) {
    if (!unsigned_type(type)) fail("unsigned result type mismatch");
    std::vector<uint32_t> words{static_cast<uint32_t>(value)};
    if (size(type) == 2) words.push_back(static_cast<uint32_t>(value >> 32));
    set(l, id, type, std::move(words), defined);
  }
  void set(Lane &l, uint32_t id, uint32_t type, std::vector<uint32_t> words,
           bool defined = true) {
    if (size(type) != words.size())
      fail("result shape mismatch");
    l.v[id] = {type, std::move(words), defined};
  }
  Memory &memory(Lane &l, uint32_t root) {
    auto local = l.memory.find(root);
    if (local != l.memory.end())
      return local->second;
    auto global = shared.find(root);
    if (global != shared.end())
      return global->second;
    fail("unknown memory");
  }
  Pointer pointer(Lane &l, uint32_t id) {
    auto it = l.p.find(id);
    if (it == l.p.end())
      fail("unknown pointer");
    return it->second;
  }
  Value read(Lane &l, Pointer p) {
    auto &m = memory(l, p.root);
    const uint32_t n = size(p.type);
    if (p.offset > m.words.size() || n > m.words.size() - p.offset)
      fail("load out of bounds");
    std::vector<uint32_t> v;
    bool defined = true;
    for (uint32_t i = 0; i < n; ++i) {
      defined &= m.valid[p.offset + i];
      v.push_back(m.words[p.offset + i]);
    }
    // A bounded OpLoad may produce an undefined value. Track it explicitly;
    // a known OpSelect can discard it, but it cannot steer control/indexing
    // or reach the live storage sink. Never invent a zero for poison.
    return {p.type, std::move(v), defined};
  }
  void write(Lane &l, Pointer p, Value v) {
    if (p.type != v.type)
      fail("store type mismatch");
    if (p.root == storage && !v.defined)
      fail("poison live storage write");
    if (readonly_storage.contains(p.root))
      fail("write to readonly supplied storage");
    auto &m = memory(l, p.root);
    if (p.offset > m.words.size() || v.words.size() > m.words.size() - p.offset)
      fail("store out of bounds");
    for (size_t i = 0; i < v.words.size(); ++i) {
      m.words[p.offset + i] = v.words[i];
      m.valid[p.offset + i] = v.defined;
    }
  }
  explicit Interpreter(const std::vector<uint32_t> &module) {
    try {
      if (module.size() < 5 || module[0] != 0x07230203u || module[3] > 200000)
        fail("invalid module header");
      bool function = false;
      for (size_t pc = 5; pc < module.size();) {
        const uint32_t count = module[pc] >> 16, op = module[pc] & 0xffff;
        if (!count || count > module.size() - pc)
          fail("invalid instruction length");
        Inst i{op, {module.begin() + pc + 1, module.begin() + pc + count}};
        pc += count;
        const auto &a = i.a;
        if (op == 11) {
          const std::vector<uint32_t> expected{0x4c534c47u, 0x6474732eu, 0x3035342eu, 0};
          if (glsl || a.size() != 5 || std::vector<uint32_t>(a.begin() + 1, a.end()) != expected)
            fail("unsupported extension import");
          glsl = a[0];
        }
        if (op == 15 && a.size() >= 2 && a[0] == 5)
          main = a[1];
        if (op == 16 && a.size() >= 2 && a[1] == 17) {
            if (a.size() != 5 ||
                !local_sizes.emplace(a[0], std::vector<uint32_t>{a.begin() + 2, a.end()}).second)
                fail("invalid or duplicate LocalSize");
        }
        if (op == 331 && a.size() >= 2 && a[1] == 38) specialized_local_sizes.insert(a[0]);
        if (op >= 19 && op <= 33 && !a.empty())
          types[a[0]] = {op, {a.begin() + 1, a.end()}};
        if (op == 71 && a.size() >= 2) {
          if (!decorations
                   .emplace(std::make_pair(a[0], a[1]),
                            std::vector<uint32_t>{a.begin() + 2, a.end()})
                   .second)
            fail("duplicate decoration outside oracle profile");
          if (a.size() == 3 && a[1] == 11)
            builtins[a[0]] = a[2];
          if (a.size() == 3 && a[1] == 33)
            bindings[a[0]] = a[2];
        }
        if (op == 72 && a.size() >= 3) {
          if (!member_decorations
                   .emplace(std::array<uint32_t, 3>{a[0], a[1], a[2]},
                            std::vector<uint32_t>{a.begin() + 3, a.end()})
                   .second)
            fail("duplicate member decoration outside oracle profile");
        }
        if (op == 41 || op == 42 || op == 43 || op == 44) {
          if (a.size() < 2)
            fail("malformed constant");
          std::vector<uint32_t> v;
          if (op == 41 || op == 42)
            v.push_back(op == 41);
          else if (op == 43)
            v = {a.begin() + 2, a.end()};
          else
            for (size_t j = 2; j < a.size(); ++j) {
              auto c = constants.find(a[j]);
              if (c == constants.end())
                fail("unknown composite constant");
              v.insert(v.end(), c->second.words.begin(), c->second.words.end());
            }
          if (size(a[0]) != v.size())
            fail("constant type mismatch");
          constants[a[1]] = {a[0], std::move(v)};
        }
        if (op == 54) {
          if (a.size() != 4 || a[1] != main)
            fail("unsupported function");
          function = true;
        } else if (op == 56)
          function = false;
        else if (function) {
          if (op == 248) {
            if (a.size() != 1)
              fail("invalid label");
            labels[a[0]] = ins.size();
            if (!entry)
              entry = ins.size() + 1;
          }
          ins.push_back(std::move(i));
        } else if (op == 59)
          globals.push_back(std::move(i));
      }
      if (!main || !entry || ins.empty())
        fail("missing main");
    } catch (const std::exception &e) {
      error = e.what();
    }
  }
  void jump(Lane &l, uint32_t label) {
    auto it = labels.find(label);
    if (it == labels.end())
      fail("unknown branch label");
    l.from = l.label;
    l.pc = it->second;
  }
  void atomic_header_increment(Lane& l, const std::vector<uint32_t>& a) {
    // Only the NGG shell's commutative Device/Relaxed uint32 header increment. Sequential
    // CPU execution checks values/bounds; it supplies no device concurrency evidence.
      if (storage_descriptor_set != 2 || a.size() != 6 || !uint_type(a[0]) || word(l, a[3]) != 1 ||
          word(l, a[4]) != 0 || word(l, a[5]) != 1)
          fail("unsupported AtomicIAdd profile");
      const auto p = pointer(l, a[2]);
      const auto old = read(l, p);
      const auto increment = value(l, a[5]);
      if (p.root != storage || p.type != a[0] || increment.type != a[0] || !old.defined ||
          old.words.size() != 1)
          fail("unsupported AtomicIAdd storage");
      write(l, p, {a[0], {old.words[0] + 1u}, true});
      set(l, a[1], a[0], old.words);
  }
  // Advance one invocation to the next structurally common barrier or Return.
  void advance(Lane &l) {
    while (!l.done) {
      if (++l.steps > 1000000 || l.pc >= ins.size())
        fail("nonterminating or invalid program");
      const auto &i = ins[l.pc];
      const auto &a = i.a;
      const uint32_t op = i.op;
      if (op == 334 || op == 335 || op == 339 || op == 345) return;
      if (op == 224) {
        if (a.size() != 3 || word(l, a[0]) != 2 || word(l, a[1]) != 2 ||
            (word(l, a[2]) & 0x108) != 0x108)
          fail("not Workgroup AcquireRelease memory barrier");
        return;
      }
      if (op == 234) {
          ++l.pc;
          atomic_header_increment(l, a);
          continue;
      }
      ++l.pc;
      if (op == 248) {
        l.label = a.at(0);
        std::vector<std::pair<uint32_t, Value>> phis;
        while (l.pc < ins.size() && ins[l.pc].op == 245) {
          const auto &p = ins[l.pc++].a;
          bool selected = false;
          if (p.size() < 4 || p.size() % 2)
            fail("malformed Phi");
          for (size_t j = 2; j < p.size(); j += 2)
            if (p[j + 1] == l.from) {
              auto v = value(l, p[j]);
              if (v.type != p[0])
                fail("Phi type mismatch");
              phis.emplace_back(p[1], std::move(v));
              selected = true;
              break;
            }
          if (!selected)
            fail("missing Phi predecessor");
        }
        for (auto &p : phis)
          l.v[p.first] = std::move(p.second);
      } else if (op == 59) {
        if (a.size() < 3 || a[2] != 7)
          fail("not Function variable");
        const auto t = pointee(a[0]);
        l.p[a[1]] = {a[1], 0, t};
        l.memory[a[1]] = make_memory(t, size(t));
        if (a.size() == 4)
          write(l, l.p[a[1]], value(l, a[3]));
      } else if (op == 61) {
        if (a.size() < 3)
          fail("malformed Load");
        auto v = read(l, pointer(l, a[2]));
        if (v.type != a[0])
          fail("Load type mismatch");
        l.v[a[1]] = std::move(v);
      } else if (op == 62) {
        if (a.size() < 2)
          fail("malformed Store");
        write(l, pointer(l, a[0]), value(l, a[1]));
      } else if (op == 65 || op == 66) {
        if (a.size() < 4)
          fail("malformed AccessChain");
        auto p = pointer(l, a[2]);
        for (size_t j = 3; j < a.size(); ++j) {
          const auto &t = ty(p.type);
          const uint32_t index = word(l, a[j]);
          if (t.op == 30 && t.a.size() == 1 && index == 0)
            p.type = t.a[0];
          else if (t.op == 28 || t.op == 29 || t.op == 23) {
            if (t.a.empty())
              fail("malformed array");
            const uint32_t width = size(t.a[0]);
            const uint32_t count =
                t.op == 28   ? constant(t.a.at(1))
                : t.op == 23 ? t.a.at(1)
                             : static_cast<uint32_t>(
                                   memory(l, p.root).words.size() / width);
            if (index >= count)
              fail("AccessChain out of bounds");
            p.offset += index * width;
            p.type = t.a[0];
          } else
            fail("unsupported pointer aggregate");
        }
        if (pointee(a[0]) != p.type)
          fail("AccessChain result type mismatch");
        l.p[a[1]] = p;
      } else if (op == 68) {
        if (a.size() != 4 || a[3] != 0 || !uint_type(a[0]))
          fail("unsupported ArrayLength");
        auto p = pointer(l, a[2]);
        const auto &t = ty(p.type);
        if (t.op != 30 || t.a.size() != 1 || ty(t.a[0]).op != 29)
          fail("not runtime array block");
        set(l, a[1], a[0],
            {static_cast<uint32_t>(memory(l, p.root).words.size() /
                                   size(ty(t.a[0]).a[0]))});
      } else if (op == 80) {
        std::vector<uint32_t> v;
        bool defined = true;
        for (size_t j = 2; j < a.size(); ++j) {
          auto x = value(l, a[j]);
          defined &= x.defined;
          v.insert(v.end(), x.words.begin(), x.words.end());
        }
        set(l, a.at(1), a.at(0), std::move(v), defined);
      } else if (op == 81) {
        if (a.size() != 4)
          fail("unsupported CompositeExtract");
        auto v = value(l, a[2]);
        const auto& aggregate = ty(v.type);
        const bool vector = aggregate.op == 23 && aggregate.a[0] == a[0];
        const bool pair = aggregate.op == 30 && aggregate.a.size() == 2 &&
                          uint_type(aggregate.a[0]) && aggregate.a[0] == aggregate.a[1] &&
                          aggregate.a[0] == a[0];
        if ((!vector && !pair) || a[3] >= v.words.size()) fail("extract type/index mismatch");
        set(l, a[1], a[0], {v.words[a[3]]}, v.defined);
      } else if (op == 151) { // OpUMulExtended, scalar u32 only: first low, then high.
          if (a.size() != 4) fail("malformed UMulExtended");
          const auto x = value(l, a[2]), y = value(l, a[3]);
          const auto& result = ty(a[0]);
          if (!uint_type(x.type) || y.type != x.type || x.words.size() != 1 ||
              y.words.size() != 1 || result.op != 30 ||
              result.a != std::vector<uint32_t>{x.type, x.type})
              fail("unsupported UMulExtended types");
          const uint64_t product = uint64_t(x.words[0]) * y.words[0];
          set(l, a[1], a[0], {static_cast<uint32_t>(product), static_cast<uint32_t>(product >> 32)},
              x.defined && y.defined);
      } else if (op == 113) {
          if (a.size() != 3 || !unsigned_type(a[0])) fail("unsupported UConvert");
          const auto v = value(l, a[2]);
          if (!unsigned_type(v.type)) fail("UConvert source type mismatch");
          set_unsigned(l, a[1], a[0], v.defined ? unsigned_value(v) : 0, v.defined);
      } else if (op == 12) {
          if (a.size() < 5 || !glsl || a[2] != glsl) fail("unsupported ExtInst set");
          const auto x = value(l, a[4]);
          if (!uint_type(x.type) || x.words.size() != 1) fail("ExtInst operand type");
          if (a[3] == 38 || a[3] == 41) {
              if (a.size() != 6 || a[0] != x.type) fail("min/max type mismatch");
              const auto y = value(l, a[5]);
              if (y.type != x.type || y.words.size() != 1) fail("min/max operand type");
              set(l, a[1], a[0],
                  {a[3] == 38 ? std::min(x.words[0], y.words[0])
                              : std::max(x.words[0], y.words[0])},
                  x.defined && y.defined);
          } else if (a[3] == 75) {
              if (a.size() != 5 || ty(a[0]).op != 21 || ty(a[0]).a != std::vector<uint32_t>{32, 1})
                  fail("FindUMsb result type mismatch");
              uint32_t bit = UINT32_MAX;
              for (uint32_t n = 0; n < 32; ++n)
                  if (x.words[0] & (uint32_t{1} << n)) bit = n;
              set(l, a[1], a[0], {bit}, x.defined && x.words[0] != 0);
          } else
              fail("unsupported live ExtInst");
      } else if (op == 88) {
          if (a.size() != 6 || a[4] != 2 || ty(a[0]).op != 23 || ty(a[0]).a.size() != 2 ||
              ty(a[0]).a[1] != 4 || ty(ty(a[0]).a[0]).op != 22 ||
              ty(ty(a[0]).a[0]).a != std::vector<uint32_t>{32})
              fail("unsupported sampled result/profile");
          const auto handle = value(l, a[2]), coord = value(l, a[3]), lod = value(l, a[5]);
          if (ty(handle.type).op != 27 || ty(handle.type).a.size() != 1 ||
              ty(ty(handle.type).a[0]).op != 25 || ty(ty(handle.type).a[0]).a.empty() ||
              ty(ty(handle.type).a[0]).a[0] != ty(a[0]).a[0] || handle.words.size() != 1 ||
              !handle.defined || ty(coord.type).op != 23 ||
              ty(coord.type).a != std::vector<uint32_t>{ty(a[0]).a[0], 2} || !coord.defined ||
              lod.type != ty(a[0]).a[0] || lod.words.size() != 1 || !lod.defined)
              fail("sample operand type/definition");
          const auto found = sampled_images.find(handle.words[0]);
          if (found == sampled_images.end()) fail("missing owned sampled image");
          uint32_t level = UINT32_MAX;
          for (uint32_t n = 0; n < found->second.size(); ++n)
              if (lod.words[0] == std::bit_cast<uint32_t>(static_cast<float>(n))) level = n;
          if (level == UINT32_MAX) fail("sample LOD outside closed oracle domain");
          const auto& mip = found->second[level];
          uint32_t px = UINT32_MAX, py = UINT32_MAX;
          for (uint32_t x = 0; x < mip.width; ++x)
              if (coord.words[0] == std::bit_cast<uint32_t>((float(x) + 0.5f) / float(mip.width)))
                  px = x;
          for (uint32_t y = 0; y < mip.height; ++y)
              if (coord.words[1] == std::bit_cast<uint32_t>((float(y) + 0.5f) / float(mip.height)))
                  py = y;
          if (px == UINT32_MAX || py == UINT32_MAX ||
              mip.texels.size() != size_t(mip.width) * mip.height)
              fail("sample coordinate outside closed oracle domain");
          const auto& texel = mip.texels.at(size_t(py) * mip.width + px);
          set(l, a[1], a[0], {texel.begin(), texel.end()});
      } else if (op == 83 || op == 124) {
          if (a.size() != 3) fail("malformed copy/bitcast");
          auto v = value(l, a[2]);
          if (op == 83 && v.type != a[0]) fail("CopyObject type mismatch");
          set(l, a[1], a[0], std::move(v.words), v.defined);
      } else if (op == 112) {
          if (a.size() != 3 || !uint_type(value(l, a[2]).type) || ty(a[0]).op != 22)
              fail("unsupported unsigned-to-float conversion");
          set(l, a[1], a[0], {std::bit_cast<uint32_t>(static_cast<float>(word(l, a[2])))});
      } else if (op == 184) {
          if (a.size() != 4 || ty(value(l, a[2]).type).op != 22 || ty(value(l, a[3]).type).op != 22)
              fail("unsupported float comparison");
          set(l, a[1], a[0],
              {uint32_t(std::bit_cast<float>(word(l, a[2])) <
                        std::bit_cast<float>(word(l, a[3])))});
      } else if (op == 169) {
          if (a.size() != 5) fail("malformed Select");
        // Both arms must be defined and typed even when only one is selected.
          const auto left = value(l, a[3]), right = value(l, a[4]);
          if (left.type != a[0] || right.type != a[0]) fail("Select type mismatch");
          const auto condition = value(l, a[2]);
          if (ty(condition.type).op != 20 || condition.words.size() != 1)
              fail("Select condition type mismatch");
        // A dead Select with an undefined condition produces poison, not a control-flow
          // decision. A later known Select may discard it; branches and live sinks still refuse.
          if (!condition.defined)
              set(l, a[1], a[0], left.words, false);
          else
              l.v[a[1]] = boolean(l, a[2]) ? left : right;
      } else if (op == 166 || op == 167 || op == 164 || op == 165) {
          if (a.size() != 4 || ty(a[0]).op != 20) fail("logical result type mismatch");
          const bool x = boolean(l, a[2]), y = boolean(l, a[3]);
          set(l, a[1], a[0],
              {uint32_t(op == 166   ? x || y
                        : op == 167 ? x && y
                        : op == 164 ? x == y
                                    : x != y)});
      } else if (op == 168) {
          if (a.size() != 3 || ty(a[0]).op != 20) fail("LogicalNot type mismatch");
          set(l, a[1], a[0], {uint32_t(!boolean(l, a[2]))});
      } else if (op == 230) {
          // Narrow SOURCE-only uint32 Device/Relaxed compare-exchange. The evaluator serializes
          // workers; this checks actual instructions, bounds and readonly ownership, NOT Vulkan
          // atomic ordering or inter-workgroup synchronization (those need real GPU controls).
          if (a.size() != 8 || !uint_type(a[0]) || word(l, a[3]) != 1 || word(l, a[4]) != 0 ||
              word(l, a[5]) != 0)
              fail("unsupported compare-exchange profile");
          const auto p = pointer(l, a[2]);
          if (p.type != a[0] || readonly_storage.contains(p.root))
              fail("compare-exchange type/readonly mismatch");
          const auto previous = read(l, p), desired = value(l, a[6]), compare = value(l, a[7]);
          if (previous.type != a[0] || desired.type != a[0] || compare.type != a[0] ||
              !previous.defined || !desired.defined || !compare.defined)
              fail("compare-exchange undefined/type mismatch");
          set(l, a[1], a[0], previous.words);
          if (previous.words == compare.words) write(l, p, desired);
      } else if (op == 205) {
          if (a.size() != 3 || !uint_type(a[0])) fail("unsupported BitCount result");
          const auto x = value(l, a[2]);
          if (x.type != a[0] || x.words.size() != 1) fail("BitCount operand type mismatch");
          uint32_t bits = 0;
          for (uint32_t position = 0; position < 32; ++position)
              bits += (x.words[0] >> position) & 1u;
          set(l, a[1], a[0], {bits}, x.defined);
      } else if (op == 200) {
          if (a.size() != 3 || !unsigned_type(a[0])) fail("unsupported Not result");
          const auto x = value(l, a[2]);
          if (x.type != a[0] || x.words.size() != size(x.type)) fail("Not operand type mismatch");
          set_unsigned(l, a[1], a[0], x.defined ? ~unsigned_value(x) : 0, x.defined);
      } else if (op ==
                 177) {   // OpSLessThan: the CFG dispatcher bounds its signed program counter.
          if (a.size() != 4 || ty(a[0]).op != 20) fail("malformed signed comparison");
          const auto x = value(l, a[2]), y = value(l, a[3]);
          if (ty(x.type).op != 21 || ty(x.type).a != std::vector<uint32_t>{32, 1} ||
              x.type != y.type || x.words.size() != 1 || y.words.size() != 1)
              fail("signed comparison operand types");
          set(l, a[1], a[0],
              {std::bit_cast<int32_t>(x.words[0]) < std::bit_cast<int32_t>(y.words[0])},
              x.defined && y.defined);
      } else if (op == 170 || op == 171 || op == 172 || op == 174 || op == 176 || op == 178 ||
                 op == 128 || op == 130 || op == 132 || op == 134 || op == 137 || op == 194 ||
                 op == 196 || op == 197 || op == 198 || op == 199) {
          if (a.size() != 4) fail("malformed integer instruction");
          const auto x = value(l, a[2]), y = value(l, a[3]);
          if (!unsigned_type(x.type) || x.type != y.type || x.words.size() != size(x.type) ||
              y.words.size() != size(y.type))
              fail("integer operand types");
          const bool comparison = op >= 170 && op <= 178;
          if (comparison ? ty(a[0]).op != 20 : a[0] != x.type) fail("integer result type");
          if (!x.defined || !y.defined) {
              set(l, a[1], a[0], std::vector<uint32_t>(size(a[0]), 0), false);
              continue;
          }
          const uint64_t u = unsigned_value(x), v = unsigned_value(y);
          uint64_t r = 0;
          if (op == 170)
              r = u == v;
          else if (op == 171)
              r = u != v;
          else if (op == 172)
              r = u > v;
          else if (op == 174)
              r = u >= v;
          else if (op == 176)
              r = u < v;
          else if (op == 178)
              r = u <= v;
          else if (op == 128)
              r = u + v;
          else if (op == 130)
              r = u - v;
          else if (op == 132)
              r = u * v;
          else if (op == 134 || op == 137) {
              if (!v) fail("division by zero");
              r = op == 134 ? u / v : u % v;
          } else if (op == 194 || op == 196) {
              if (v >= 32 * size(x.type)) fail("poison shift");
              r = op == 194 ? u >> v : u << v;
          } else if (op == 197)
              r = u | v;
          else if (op == 198)
              r = u ^ v;
          else
              r = u & v;
          if (comparison)
              set(l, a[1], a[0], {static_cast<uint32_t>(r)});
          else
              set_unsigned(l, a[1], a[0], r);
      } else if (op == 250) {
          if (a.size() < 3) fail("malformed BranchConditional");
          jump(l, boolean(l, a[0]) ? a[1] : a[2]);
      } else if (op == 249) {
          if (a.size() != 1) fail("malformed Branch");
          jump(l, a[0]);
      } else if (op == 251) {
          if (a.size() < 2 || a.size() % 2) fail("malformed Switch");
          const auto selector = word(l, a[0]);
          uint32_t target = a[1];
          for (size_t j = 2; j < a.size(); j += 2)
              if (a[j] == selector) {
                  target = a[j + 1];
                  break;
              }
          jump(l, target);
      } else if (op == 253)
          l.done = true;
      else if (op != 246 && op != 247 && op != 8 && op != 317 && op != 0)
        fail("unsupported live opcode " + std::to_string(op));
    }
  }
  std::vector<uint32_t> run(uint32_t count,
                            const std::vector<uint32_t> &buffer) {
    return run_buffers(count, {{3, buffer}}, 3);
  }
  // Same typed evaluator, with two independently pinned raw-u32 blocks for packet execution.
  // The input is read-only and the output is the sole observable sink. All supplied ABI layouts
  // retain the exact Set0/Block/Offset0/Stride4 checks; this is not general descriptor emulation.
  std::vector<uint32_t> run_packet(const std::vector<uint32_t>& input,
                                   const std::vector<uint32_t>& output, uint32_t workgroup_x = 0) {
      return run_buffers(64, {{0, input}, {1, output}}, 1, workgroup_x);
  }
  // Bindings besides the sink that the shader may legitimately write (an in-place pass). Their
  // final contents are read back with writable_result() after the run.
  std::set<uint32_t> extra_writable_bindings;
  std::vector<uint32_t> writable_result(uint32_t binding) const {
      const auto root = extra_writable_roots.find(binding);
      if (root == extra_writable_roots.end()) return {};
      const auto value = shared.find(root->second);
      return value == shared.end() ? std::vector<uint32_t>{} : value->second.words;
  }
  std::vector<uint32_t> run_buffers(uint32_t count,
                                    const std::map<uint32_t, std::vector<uint32_t>>& buffers,
                                    uint32_t sink_binding, uint32_t workgroup_x = 0) {
      try {
          if (!error.empty()) fail(error);
          if (!count || count > 1024) fail("invalid invocation count");
          if (native_subgroup_width == 32) {
              if (count != 32) fail("unproved native32 instance");
              if (native32_workgroup_count) {
                  const auto local = local_sizes.find(main);
                  if (local == local_sizes.end() ||
                      local->second != std::vector<uint32_t>{32, 1, 1} ||
                      specialized_local_sizes.contains(main) ||
                      std::any_of(builtins.begin(), builtins.end(),
                                  [](const auto& builtin) { return builtin.second == 25; }) ||
                      workgroup_x >= native32_workgroup_count || workgroup_x > UINT32_MAX / 32)
                      fail("unproved native32 instance");
              } else if (workgroup_x != 0) {
                  fail("unproved native32 instance");
              }
          } else if (native32_workgroup_count) {
              fail("unproved native32 instance");
          }
          storage = 0;
          shared.clear();
          readonly_storage.clear();
          extra_writable_roots.clear();
          std::map<uint32_t, bool> seen_bindings;
          std::map<uint32_t, bool> seen_sampled_bindings;
          std::vector<Lane> lanes(count);
          for (const auto& g : globals) {
              const auto& a = g.a;
              if (a.size() < 3) fail("malformed global Variable");
              const auto t = pointee(a[0]);
              if (a[2] == 4)
                  shared[a[1]] = make_memory(t, size(t));
              else if (a[2] == 9) {
                  if (supplied_push_constants.empty() && !push_constants.empty()) {
                      shared[a[1]] = make_memory(t, size(t));
                      if (push_constants.size() != size(t)) fail("push constant shape mismatch");
                      shared[a[1]] = {t, push_constants,
                                      std::vector<bool>(push_constants.size(), true)};
                  } else {
                      const auto& structure = ty(t);
                      const auto block = decorations.find({t, 2});
                      const auto offset = member_decorations.find({t, 0, 35});
                      if (supplied_push_constants.empty()) fail("missing supplied push constants");
                      if (ty(a[0]).a[0] != 9 || structure.op != 30 || structure.a.size() != 1 ||
                          block == decorations.end() || !block->second.empty() ||
                          offset == member_decorations.end() ||
                          offset->second != std::vector<uint32_t>{0})
                          fail("unsupported push constant ABI");
                      const auto& array = ty(structure.a[0]);
                      const auto stride = decorations.find({structure.a[0], 6});
                      if (array.op != 28 || array.a.size() != 2 || !uint_type(array.a[0]) ||
                          stride == decorations.end() ||
                          stride->second != std::vector<uint32_t>{4} ||
                          constant(array.a[1]) != supplied_push_constants.size())
                          fail("unsupported push constant ABI");
                      shared[a[1]] = {t, supplied_push_constants,
                                      std::vector<bool>(supplied_push_constants.size(), true)};
                      readonly_storage.emplace(a[1], true);
                  }
              } else if (a[2] == 2 || a[2] == 12) {
                  const auto binding = bindings.find(a[1]);
                  const auto supplied =
                      binding == bindings.end() ? buffers.end() : buffers.find(binding->second);
                  if (supplied != buffers.end()) {
                      if (!seen_bindings.emplace(binding->second, true).second)
                          fail("duplicate live buffer binding");
                      // This oracle supports exactly the fixture's externally observable
                      // ABI. Never silently flatten a differently laid-out Vulkan block
                      // into uint words.
                      const auto set = decorations.find({a[1], 34});
                      const auto block = decorations.find({t, 2});
                      const auto offset = member_decorations.find({t, 0, 35});
                      const auto& structure = ty(t);
                      if (a[2] != 12 || ty(a[0]).a[0] != 12 || set == decorations.end() ||
                          set->second != std::vector<uint32_t>{storage_descriptor_set} ||
                          block == decorations.end() || !block->second.empty() ||
                          offset == member_decorations.end() ||
                          offset->second != std::vector<uint32_t>{0} || structure.op != 30 ||
                          structure.a.size() != 1)
                          fail("unsupported live storage ABI");
                      const auto array = structure.a[0];
                      const auto stride = decorations.find({array, 6});
                      if (ty(array).op != 29 || ty(array).a.size() != 1)
                          fail("unsupported live storage ABI");
                      const auto& element = ty(ty(array).a.at(0));
                      const bool float_word = legacy_float_word_storage && binding->second <= 1 &&
                                              element.op == 22 &&
                                              element.a == std::vector<uint32_t>{32};
                      if ((!uint_type(ty(array).a[0]) && !float_word) ||
                          stride == decorations.end() || stride->second != std::vector<uint32_t>{4})
                          fail("unsupported live storage ABI");
                      if (binding->second == sink_binding)
                          storage = a[1];
                      else if (extra_writable_bindings.contains(binding->second))
                          extra_writable_roots[binding->second] = a[1];
                      else
                          readonly_storage.emplace(a[1], true);
                      shared[a[1]] = {t, supplied->second,
                                      std::vector<bool>(supplied->second.size(), true)};
                  } else {
                      // The production shell declares unused synthetic buffers too.
                      // Give them no initialized bytes: any actual read/write fails.
                      shared[a[1]] = {t, {}, {}};
                  }
              } else if (a[2] == 0) {
                  const auto binding = bindings.find(a[1]);
                  const auto set = decorations.find({a[1], 34});
                  const auto& sampled = ty(t);
                  if (binding == bindings.end() || !sampled_images.contains(binding->second) ||
                      set == decorations.end() || set->second != std::vector<uint32_t>{0} ||
                      ty(a[0]).a[0] != 0 || sampled.op != 27 || sampled.a.size() != 1)
                      fail("unsupported sampled image ABI");
                  if (!seen_sampled_bindings.emplace(binding->second, true).second)
                      fail("duplicate sampled image binding");
                  const auto& image = ty(sampled.a[0]);
                  if (image.op != 25 || image.a.size() != 7 || ty(image.a[0]).op != 22 ||
                      ty(image.a[0]).a != std::vector<uint32_t>{32} ||
                      std::vector<uint32_t>(image.a.begin() + 1, image.a.end()) !=
                          std::vector<uint32_t>{1, 0, 0, 0, 1, 0})
                      fail("unsupported sampled image type");
                  shared[a[1]] = {t, {binding->second}, {true}};
              } else if (a[2] != 1)
                  fail("unsupported global storage class");
              for (uint32_t lane = 0; lane < count; ++lane) {
                  auto& l = lanes[lane];
                  l.p[a[1]] = {a[1], 0, t};
                  if (a[2] == 1) {
                      auto b = builtins.find(a[1]);
                      if (b == builtins.end()) fail("Input without builtin");
                      std::vector<uint32_t> v;
                      if (b->second == 26)
                          v = {workgroup_x, 0, 0};
                      else if (b->second == 28)
                          v = {workgroup_x * (native_subgroup_width == 32 ? 32 : 64) + lane, 0, 0};
                      else if (b->second == 27)
                          v = {lane, 0, 0};
                      else if (b->second == 29)
                          v = {lane};
                      else if ((b->second == 40 || b->second == 41) &&
                               (native_subgroup_width == 32 || native_subgroup_width == 64) &&
                               count == native_subgroup_width)
                          v = {b->second == 40 ? 0u : lane};
                      else
                          fail("unsupported builtin");
                      if (size(t) != v.size()) fail("Input shape mismatch");
                      l.memory[a[1]] = {t, v, std::vector<bool>(v.size(), true)};
                  }
              }
          }
          if (!storage) fail("missing live storage sink");
          if (seen_bindings.size() != buffers.size()) fail("missing supplied storage binding");
          if (seen_sampled_bindings.size() != sampled_images.size())
              fail("missing supplied sampled image binding");
          // The first function label is kept in the instruction stream, including
          // any Phi.
          for (auto& l : lanes) l.pc = entry - 1;
          for (size_t phase = 0; phase < 10000; ++phase) {
              for (auto& l : lanes) advance(l);
              bool all_done = true;
              for (const auto& l : lanes) all_done &= l.done;
              if (all_done) return shared.at(storage).words;
              const size_t site = lanes.front().pc;
              const auto& collective = ins.at(site);
              if (collective.op ==
                  345) {   // OpGroupNonUniformShuffle: explicit exact-native profile
                  const auto& a = collective.a;
                  if ((native_subgroup_width != 32 && native_subgroup_width != 64) ||
                      count != native_subgroup_width || a.size() != 5 || !uint_type(a[0]))
                      fail("unproved native shuffle");
                  std::vector<Value> inputs, results;
                  for (auto& l : lanes) {
                      if (l.done || l.pc != site || word(l, a[2]) != 3)
                          fail("nonuniform native shuffle");
                      auto input = value(l, a[3]);
                      if (input.type != a[0]) fail("shuffle input type mismatch");
                      inputs.push_back(std::move(input));
                  }
                  for (auto& l : lanes) {
                      if (!uint_type(value(l, a[4]).type)) fail("shuffle index type mismatch");
                      const auto index = word(l, a[4]);
                      if (index >= count) fail("shuffle index outside supplied subgroup");
                      results.push_back(inputs[index]);
                  }
                  for (uint32_t lane = 0; lane < count; ++lane) {
                      const auto& result = results[lane];
                      set(lanes[lane], a[1], a[0], result.words, result.defined);
                      ++lanes[lane].pc;
                  }
                  continue;
              }
              if (collective.op == 334 || collective.op == 335 || collective.op == 339) {
                  if ((native_subgroup_width != 32 && native_subgroup_width != 64) ||
                      count != native_subgroup_width || collective.a.size() != 4)
                      fail("unproved native subgroup");
                  const auto& a = collective.a;
                  std::vector<uint32_t> result(collective.op == 339 ? 4 : 1);
                  if (collective.op == 334) result[0] = 1;
                  for (uint32_t lane = 0; lane < count; ++lane) {
                      auto& l = lanes[lane];
                      if (l.done || l.pc != site || word(l, a[2]) != 3)
                          fail("nonuniform native collective");
                      const bool predicate = boolean(l, a[3]);
                      if (collective.op == 339) {
                          if (predicate) result[lane / 32] |= 1u << (lane % 32);
                      } else if (collective.op == 334)
                          result[0] &= predicate;
                      else
                          result[0] |= predicate;
                  }
                  for (auto& l : lanes) {
                      set(l, a[1], a[0], result);
                      ++l.pc;
                  }
                  continue;
              }
              for (auto& l : lanes) {
                  if (l.done || l.pc != site || ins[l.pc].op != 224)
                      fail("nonuniform Workgroup barrier");
                  ++l.pc;
              }
          }
          fail("too many barrier phases");
      } catch (const std::exception& e) {
          error = e.what();
          return {};
      }
  }
};
} // namespace bpermute_oracle
