// nid.hpp — Sony NID (symbol name hash) computation + name<->NID database.
//
// A PS5 dynamic symbol is imported/exported by NID, an 11-char Sony-base64 encoding
// of the first 8 bytes of SHA1(name + fixed-salt). We can't invert a hash, so to show
// real names in logs we hash a dictionary of known symbol names forward and match.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper {

// Compute the 11-char NID for a symbol name.
std::string nid_hash(const std::string& name);

// name <-> NID dictionary. Seed with known names; look up by NID.
class NidDb {
public:
    NidDb();                                   // seeds built-ins + known_names.txt if found
    void add(const std::string& name);         // hashes and registers one name
    void add_many(const std::vector<std::string>& names);
    // Hash every newline-separated symbol name in `path` into the DB. Returns count added.
    size_t load_names_file(const std::string& path);
    // Resolve a NID to a name, or "" if unknown.
    const std::string& resolve(const std::string& nid) const;
    size_t size() const { return by_nid_.size(); }
private:
    std::unordered_map<std::string, std::string> by_nid_; // nid -> name
    std::string empty_;
};

// The built-in seed list of common libc / libkernel / pthread symbol names.
const std::vector<std::string>& builtin_symbol_names();

} // namespace prosper
