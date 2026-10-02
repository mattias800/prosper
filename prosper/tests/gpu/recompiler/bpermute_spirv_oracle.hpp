#pragma once
// Independent, fail-closed integer SPIR-V interpreter for this test's ACTUAL
// recompile_compute modules. This is not a Vulkan interpreter or a device
// oracle. It executes the entrypoint, dispatcher/phis, typed memory and common
// barriers; unsupported operations, poison reads, invalid pointers and split
// barriers fail.
#include <array>
#include <cstdint>
#include <map>
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
  std::map<uint32_t, size_t> labels;
  std::map<uint32_t, Memory> shared;
  std::vector<Inst> globals;
  uint32_t main = 0, storage = 0;
  size_t entry = 0;
  std::string error;
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
        if (op == 15 && a.size() >= 2 && a[0] == 5)
          main = a[1];
        if (op >= 19 && op <= 33 && !a.empty())
          types[a[0]] = {op, {a.begin() + 1, a.end()}};
        if (op == 71 && a.size() == 3) {
          if (a[1] == 11)
            builtins[a[0]] = a[2];
          if (a[1] == 33)
            bindings[a[0]] = a[2];
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
  // Advance one invocation to the next structurally common barrier or Return.
  void advance(Lane &l) {
    while (!l.done) {
      if (++l.steps > 1000000 || l.pc >= ins.size())
        fail("nonterminating or invalid program");
      const auto &i = ins[l.pc];
      const auto &a = i.a;
      const uint32_t op = i.op;
      if (op == 224) {
        if (a.size() != 3 || word(l, a[0]) != 2 || word(l, a[1]) != 2 ||
            (word(l, a[2]) & 0x108) != 0x108)
          fail("not Workgroup AcquireRelease memory barrier");
        return;
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
        if (ty(v.type).op != 23 || a[3] >= v.words.size() ||
            ty(v.type).a[0] != a[0])
          fail("extract type/index mismatch");
        set(l, a[1], a[0], {v.words[a[3]]}, v.defined);
      } else if (op == 83 || op == 124) {
        if (a.size() != 3)
          fail("malformed copy/bitcast");
        auto v = value(l, a[2]);
        if (op == 83 && v.type != a[0])
          fail("CopyObject type mismatch");
        set(l, a[1], a[0], std::move(v.words), v.defined);
      } else if (op == 169) {
        if (a.size() != 5)
          fail("malformed Select");
        // Both arms must be defined and typed even when only one is selected.
        const auto left = value(l, a[3]), right = value(l, a[4]);
        if (left.type != a[0] || right.type != a[0])
          fail("Select type mismatch");
        l.v[a[1]] = boolean(l, a[2]) ? left : right;
      } else if (op == 166 || op == 167 || op == 164 || op == 165) {
        if (a.size() != 4 || ty(a[0]).op != 20)
          fail("logical result type mismatch");
        const bool x = boolean(l, a[2]), y = boolean(l, a[3]);
        set(l, a[1], a[0],
            {uint32_t(op == 166   ? x || y
                      : op == 167 ? x && y
                      : op == 164 ? x == y
                                  : x != y)});
      } else if (op == 168) {
        if (a.size() != 3 || ty(a[0]).op != 20)
          fail("LogicalNot type mismatch");
        set(l, a[1], a[0], {uint32_t(!boolean(l, a[2]))});
      } else if (op == 170 || op == 171 || op == 172 || op == 174 ||
                 op == 176 || op == 178 || op == 128 || op == 130 ||
                 op == 132 || op == 134 || op == 137 || op == 194 ||
                 op == 196 || op == 197 || op == 198 || op == 199) {
        if (a.size() != 4)
          fail("malformed integer instruction");
        const auto x = value(l, a[2]), y = value(l, a[3]);
        if (!uint_type(x.type) || x.type != y.type || x.words.size() != 1 ||
            y.words.size() != 1)
          fail("integer operand types");
        const bool comparison = op >= 170 && op <= 178;
        if (comparison ? ty(a[0]).op != 20 : a[0] != x.type)
          fail("integer result type");
        if (!x.defined || !y.defined) {
          set(l, a[1], a[0], {0}, false);
          continue;
        }
        const uint32_t u = x.words[0], v = y.words[0];
        uint32_t r = 0;
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
          if (!v)
            fail("division by zero");
          r = op == 134 ? u / v : u % v;
        } else if (op == 194 || op == 196) {
          if (v >= 32)
            fail("poison shift");
          r = op == 194 ? u >> v : u << v;
        } else if (op == 197)
          r = u | v;
        else if (op == 198)
          r = u ^ v;
        else
          r = u & v;
        set(l, a[1], a[0], {r});
      } else if (op == 250) {
        if (a.size() < 3)
          fail("malformed BranchConditional");
        jump(l, boolean(l, a[0]) ? a[1] : a[2]);
      } else if (op == 249) {
        if (a.size() != 1)
          fail("malformed Branch");
        jump(l, a[0]);
      } else if (op == 251) {
        if (a.size() < 2 || a.size() % 2)
          fail("malformed Switch");
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
    try {
      if (!error.empty())
        fail(error);
      if (!count || count > 1024)
        fail("invalid invocation count");
      std::vector<Lane> lanes(count);
      for (const auto &g : globals) {
        const auto &a = g.a;
        if (a.size() < 3)
          fail("malformed global Variable");
        const auto t = pointee(a[0]);
        if (a[2] == 4)
          shared[a[1]] = make_memory(t, size(t));
        else if (a[2] == 2 || a[2] == 12) {
          if (bindings[a[1]] == 3) {
            if (storage)
              fail("duplicate live buffer binding");
            storage = a[1];
            shared[a[1]] = {t, buffer, std::vector<bool>(buffer.size(), true)};
          } else {
            // The production shell declares unused synthetic buffers too.
            // Give them no initialized bytes: any actual read/write fails.
            shared[a[1]] = {t, {}, {}};
          }
        } else if (a[2] != 1)
          fail("unsupported global storage class");
        for (uint32_t lane = 0; lane < count; ++lane) {
          auto &l = lanes[lane];
          l.p[a[1]] = {a[1], 0, t};
          if (a[2] == 1) {
            auto b = builtins.find(a[1]);
            if (b == builtins.end())
              fail("Input without builtin");
            std::vector<uint32_t> v;
            if (b->second == 26)
              v = {0, 0, 0};
            else if (b->second == 27 || b->second == 28)
              v = {lane, 0, 0};
            else if (b->second == 29)
              v = {lane};
            else
              fail("unsupported builtin");
            if (size(t) != v.size())
              fail("Input shape mismatch");
            l.memory[a[1]] = {t, v, std::vector<bool>(v.size(), true)};
          }
        }
      }
      if (!storage)
        fail("missing live storage sink");
      // The first function label is kept in the instruction stream, including
      // any Phi.
      for (auto &l : lanes)
        l.pc = entry - 1;
      for (size_t phase = 0; phase < 10000; ++phase) {
        for (auto &l : lanes)
          advance(l);
        bool all_done = true;
        for (const auto &l : lanes)
          all_done &= l.done;
        if (all_done)
          return shared.at(storage).words;
        const size_t site = lanes.front().pc;
        for (auto &l : lanes) {
          if (l.done || l.pc != site || ins[l.pc].op != 224)
            fail("nonuniform Workgroup barrier");
          ++l.pc;
        }
      }
      fail("too many barrier phases");
    } catch (const std::exception &e) {
      error = e.what();
      return {};
    }
  }
};
} // namespace bpermute_oracle
