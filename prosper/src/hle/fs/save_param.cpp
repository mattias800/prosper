#include "hle/fs/save_param.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>

#ifdef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/utime.h>
#else
#include <sys/stat.h>
#include <utime.h>
#endif

namespace prosper {
namespace {

// param.sfo container. Header 20 bytes, then one 16-byte index entry per key, then the key table
// (NUL-terminated names) padded to the data table, then the values.
constexpr uint32_t kSfoMagic = 0x46535000u;    // "\0PSF"
constexpr uint32_t kSfoVersion = 0x00000101u;
constexpr uint16_t kFmtUtf8 = 0x0204u;         // NUL-terminated UTF-8
constexpr uint16_t kFmtInt32 = 0x0404u;

// The keys a save's parameter block carries. Written in ascending key order, which is how the
// format's readers expect to find them.
constexpr const char* kKeyCategory = "CATEGORY";
constexpr const char* kKeyDetail = "DETAIL";
constexpr const char* kKeyMainTitle = "MAINTITLE";
constexpr const char* kKeyDirectory = "SAVEDATA_DIRECTORY";
constexpr const char* kKeyListParam = "SAVEDATA_LIST_PARAM";
constexpr const char* kKeySubTitle = "SUBTITLE";
constexpr const char* kKeyTitleId = "TITLE_ID";

struct Entry {
    const char* key;
    uint16_t format;
    std::string text;       // kFmtUtf8: the value without its NUL
    uint32_t integer = 0;   // kFmtInt32
    uint32_t max_len;       // the ABI maximum for this key, which the file records
};

void put32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t)(v & 0xff));
    out.push_back((uint8_t)((v >> 8) & 0xff));
    out.push_back((uint8_t)((v >> 16) & 0xff));
    out.push_back((uint8_t)((v >> 24) & 0xff));
}
void put16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back((uint8_t)(v & 0xff));
    out.push_back((uint8_t)((v >> 8) & 0xff));
}
uint32_t get32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint16_t get16(const uint8_t* p) { return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8)); }

std::string clamp_text(const std::string& value, size_t max_with_nul) {
    // The ABI maxima include the terminator, so the longest storable text is one byte shorter.
    return value.size() >= max_with_nul ? value.substr(0, max_with_nul - 1) : value;
}

}   // namespace

std::vector<uint8_t> encode_param_sfo(const SaveDataParam& param, const std::string& title_id,
                                      const std::string& directory) {
    std::vector<Entry> entries = {
        // "sd" is the save-data category, the same string a console's save param.sfo carries. It is
        // a constant of the container, not an answer to a question about local content.
        {kKeyCategory, kFmtUtf8, "sd", 0, 4},
        {kKeyDetail, kFmtUtf8, clamp_text(param.detail, 1024), 0, 1024},
        {kKeyMainTitle, kFmtUtf8, clamp_text(param.title, 128), 0, 128},
        {kKeyDirectory, kFmtUtf8, clamp_text(directory, 32), 0, 32},
        {kKeyListParam, kFmtInt32, {}, param.user_param, 4},
        {kKeySubTitle, kFmtUtf8, clamp_text(param.sub_title, 128), 0, 128},
        {kKeyTitleId, kFmtUtf8, clamp_text(title_id, 16), 0, 16},
    };

    std::vector<uint8_t> keys;
    std::vector<uint8_t> data;
    struct Index { uint16_t key_offset, format; uint32_t len, max_len, data_offset; };
    std::vector<Index> index;
    index.reserve(entries.size());
    for (const Entry& e : entries) {
        Index ix{};
        ix.key_offset = (uint16_t)keys.size();
        keys.insert(keys.end(), e.key, e.key + std::strlen(e.key) + 1);
        ix.format = e.format;
        ix.max_len = e.max_len;
        ix.data_offset = (uint32_t)data.size();
        if (e.format == kFmtInt32) {
            ix.len = 4;
            put32(data, e.integer);
        } else {
            ix.len = (uint32_t)e.text.size() + 1;
            data.insert(data.end(), e.text.begin(), e.text.end());
            data.push_back(0);
            // A value occupies its declared maximum in the file, so a reader that trusts max_len
            // sees exactly the bytes the entry claims.
            data.resize(data.size() + (e.max_len - ix.len), 0);
        }
        index.push_back(ix);
    }
    while (keys.size() % 4) keys.push_back(0);

    const uint32_t key_table_offset = (uint32_t)(20 + index.size() * 16);
    const uint32_t data_table_offset = key_table_offset + (uint32_t)keys.size();

    std::vector<uint8_t> out;
    out.reserve(data_table_offset + data.size());
    put32(out, kSfoMagic);
    put32(out, kSfoVersion);
    put32(out, key_table_offset);
    put32(out, data_table_offset);
    put32(out, (uint32_t)index.size());
    for (const Index& ix : index) {
        put16(out, ix.key_offset);
        put16(out, ix.format);
        put32(out, ix.len);
        put32(out, ix.max_len);
        put32(out, ix.data_offset);
    }
    out.insert(out.end(), keys.begin(), keys.end());
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

bool decode_param_sfo(const uint8_t* bytes, size_t size, SaveDataParam& out) {
    if (!bytes || size < 20) return false;
    if (get32(bytes) != kSfoMagic) return false;
    const uint32_t key_table = get32(bytes + 8);
    const uint32_t data_table = get32(bytes + 12);
    const uint32_t count = get32(bytes + 16);
    // Bound every table against the real file length before reading through it: this file is on
    // disk, so a truncated or corrupted one must read as absent rather than walk off the buffer.
    if (count > 4096) return false;
    if ((uint64_t)20 + (uint64_t)count * 16 > size) return false;
    if (key_table > size || data_table > size) return false;

    SaveDataParam parsed;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t* ix = bytes + 20 + i * 16;
        const uint32_t key_offset = key_table + get16(ix);
        const uint16_t format = get16(ix + 2);
        const uint32_t len = get32(ix + 4);
        const uint32_t data_offset = data_table + get32(ix + 12);
        if (key_offset >= size) return false;
        if ((uint64_t)data_offset + len > size) return false;
        const char* key = (const char*)bytes + key_offset;
        const size_t key_room = size - key_offset;
        if (!memchr(key, 0, key_room)) return false;   // unterminated key name
        const uint8_t* value = bytes + data_offset;
        if (format == kFmtInt32) {
            if (len < 4) return false;
            if (!std::strcmp(key, kKeyListParam)) parsed.user_param = get32(value);
        } else {
            // Trust the NUL, not the declared length: the declared length is the used size within a
            // fixed-maximum field, and a writer that padded differently must still read back equal.
            const size_t text_room = std::min<size_t>(len, size - data_offset);
            size_t text_len = 0;
            while (text_len < text_room && value[text_len]) text_len++;
            const std::string text((const char*)value, text_len);
            if (!std::strcmp(key, kKeyMainTitle)) parsed.title = text;
            else if (!std::strcmp(key, kKeySubTitle)) parsed.sub_title = text;
            else if (!std::strcmp(key, kKeyDetail)) parsed.detail = text;
        }
    }
    out = parsed;
    return true;
}

std::string save_param_path(const std::string& save_dir) {
    if (save_dir.empty()) return {};
    return save_dir + "/sce_sys/param.sfo";
}

bool save_param_store(const std::string& save_dir, const SaveDataParam& param,
                      const std::string& title_id, const std::string& directory) {
    const std::string path = save_param_path(save_dir);
    if (path.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(save_dir) / "sce_sys", ec);
    const std::vector<uint8_t> bytes = encode_param_sfo(param, title_id, directory);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool wrote = fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    fclose(f);
    if (!wrote) return false;
    if (param.mtime > 0) {
        // The console's save timestamp IS this file's timestamp -- savedata0_dir_mtime() reads it
        // for the save dialog's date ordering -- so a guest that sets MTIME moves the file rather
        // than having its value stored somewhere only we would look.
#ifdef _WIN32
        struct _utimbuf times{};
        times.actime = (time_t)param.mtime;
        times.modtime = (time_t)param.mtime;
        _utime(path.c_str(), &times);
#else
        struct utimbuf times{};
        times.actime = (time_t)param.mtime;
        times.modtime = (time_t)param.mtime;
        utime(path.c_str(), &times);
#endif
    }
    return true;
}

bool save_param_load(const std::string& save_dir, SaveDataParam& out) {
    const std::string path = save_param_path(save_dir);
    if (path.empty()) return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> bytes;
    uint8_t chunk[4096];
    size_t n = 0;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) bytes.insert(bytes.end(), chunk, chunk + n);
    fclose(f);
    SaveDataParam parsed;
    if (!decode_param_sfo(bytes.data(), bytes.size(), parsed)) return false;
    // The timestamp is the file's, not a field inside it.
#ifdef _WIN32
    struct _stat64 st{};
    if (_stat64(path.c_str(), &st) == 0) parsed.mtime = (int64_t)st.st_mtime;
#else
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) parsed.mtime = (int64_t)st.st_mtime;
#endif
    out = parsed;
    return true;
}

}   // namespace prosper
