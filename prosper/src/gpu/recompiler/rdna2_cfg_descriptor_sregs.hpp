// Entry descriptor identity is separate from scalar storage allocated by the dispatcher.
#pragma once
#include <set>
namespace prosper::gpu {
struct ShaderResourceTable;
std::set<int> cfg_direct_descriptor_sregs(const ShaderResourceTable* rt);
} // namespace prosper::gpu
