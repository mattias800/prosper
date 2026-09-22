// save_capacity.cpp — see save_capacity.hpp for the accounting contract and its confidence (#3654).
#include "hle/fs/save_capacity.hpp"

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#define prosper_getpid _getpid
#else
#include <unistd.h>
#define prosper_getpid getpid
#endif

namespace prosper {
namespace {

namespace fs = std::filesystem;

// Serializes record read-modify-write inside one process. Across processes the rename is what keeps
// a reader from ever seeing a torn record.
std::mutex g_capacity_mx;

constexpr const char* kRecordMagic = "prosper-savedata-capacity 1";

std::string record_path(const std::string& title_save_dir, const std::string& dirname) {
    return title_save_dir + "/" + kSaveCapacityDirName + "/" + dirname;
}

// Strict parse of "<magic>\nblocks <decimal>\n". Anything else is Corrupt.
bool parse_record(const std::string& text, uint64_t& blocks) {
    const std::string head = std::string(kRecordMagic) + "\nblocks ";
    if (text.compare(0, head.size(), head) != 0) return false;
    size_t i = head.size();
    if (i >= text.size() || text[i] < '0' || text[i] > '9') return false;
    uint64_t v = 0;
    for (; i < text.size() && text[i] >= '0' && text[i] <= '9'; ++i) {
        const uint64_t d = (uint64_t)(text[i] - '0');
        if (v > (UINT64_MAX - d) / 10) return false;   // overflow is corruption, not a huge save
        v = v * 10 + d;
    }
    if (i + 1 != text.size() || text[i] != '\n') return false;
    blocks = v;
    return true;
}

SaveAllocationState read_locked(const std::string& title_save_dir, const std::string& dirname,
                                uint64_t& blocks) {
    const std::string path = record_path(title_save_dir, dirname);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return errno == ENOENT ? SaveAllocationState::Absent : SaveAllocationState::Unreadable;
    std::string text;
    char buf[128];
    size_t n = 0;
    bool too_long = false;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        text.append(buf, n);
        if (text.size() > 256) { too_long = true; break; }   // a record is ~40 bytes
    }
    const bool read_error = ferror(f) != 0;
    fclose(f);
    if (read_error) return SaveAllocationState::Unreadable;
    if (too_long || !parse_record(text, blocks)) return SaveAllocationState::Corrupt;
    return SaveAllocationState::Present;
}

bool write_locked(const std::string& title_save_dir, const std::string& dirname, uint64_t blocks) {
    std::error_code ec;
    const fs::path dir = fs::path(title_save_dir) / kSaveCapacityDirName;
    fs::create_directories(dir, ec);
    if (ec) return false;
    static std::atomic<uint64_t> seq{0};
    char suffix[64];
    snprintf(suffix, sizeof suffix, ".tmp.%ld.%" PRIu64, (long)prosper_getpid(), seq.fetch_add(1));
    const std::string final_path = record_path(title_save_dir, dirname);
    const std::string tmp_path = final_path + suffix;
    FILE* f = fopen(tmp_path.c_str(), "wb");
    if (!f) return false;
    char text[96];
    const int len = snprintf(text, sizeof text, "%s\nblocks %" PRIu64 "\n", kRecordMagic, blocks);
    bool ok = len > 0 && fwrite(text, 1, (size_t)len, f) == (size_t)len;
    ok = (fflush(f) == 0) && ok;
    ok = (fclose(f) == 0) && ok;
    if (ok) {
        fs::rename(tmp_path, final_path, ec);   // replaces an existing record atomically
        ok = !ec;
    }
    if (!ok) fs::remove(tmp_path, ec);
    return ok;
}

}   // namespace

SaveAllocationState save_allocation_read(const std::string& title_save_dir, const std::string& dirname,
                                         uint64_t& blocks) {
    std::lock_guard<std::mutex> lk(g_capacity_mx);
    return read_locked(title_save_dir, dirname, blocks);
}

bool save_allocation_write(const std::string& title_save_dir, const std::string& dirname,
                           uint64_t blocks) {
    std::lock_guard<std::mutex> lk(g_capacity_mx);
    return write_locked(title_save_dir, dirname, blocks);
}

bool save_allocation_note_mount(const std::string& title_save_dir, const std::string& dirname,
                                bool created, uint64_t requested_blocks) {
    if (requested_blocks == 0) return true;   // nothing asked for, nothing to record
    std::lock_guard<std::mutex> lk(g_capacity_mx);
    uint64_t existing = 0;
    const SaveAllocationState state = read_locked(title_save_dir, dirname, existing);
    // A creating mount records its request. So does the first requesting mount of a save with no
    // record (a legacy save). A creating mount over a stale record -- a save directory deleted by
    // hand while its record survived -- is a new save, so its request replaces the record. A
    // corrupt or unreadable record is never overwritten by an OPEN, so the evidence stays.
    const bool write = created ? state != SaveAllocationState::Unreadable
                               : state == SaveAllocationState::Absent;
    if (!write) {
        // Nothing to write: an OPEN of a recorded (or corrupt, or unreadable) save keeps what is
        // there. The one failure is a CREATE that could not even read what was there.
        return !(created && state == SaveAllocationState::Unreadable);
    }
    const bool ok = write_locked(title_save_dir, dirname, requested_blocks);
    if (!ok)
        fprintf(stderr, "[savedata] could not record the %" PRIu64 "-block allocation of save '%s'; "
                        "its capacity will be reported as unknown\n", requested_blocks, dirname.c_str());
    return ok;
}

bool save_usage_blocks(const std::string& save_dir, uint64_t& used) {
    used = 0;
    std::error_code ec;
    if (!fs::is_directory(save_dir, ec) || ec) return false;
    fs::recursive_directory_iterator it(save_dir, fs::directory_options::none, ec);
    if (ec) return false;
    for (const fs::recursive_directory_iterator end; it != end; it.increment(ec)) {
        if (ec) return false;
        const fs::file_status st = it->symlink_status(ec);
        if (ec) return false;
        if (!fs::is_regular_file(st)) continue;   // directories and anything exotic charge nothing
        const uintmax_t size = it->file_size(ec);
        if (ec) return false;
        const uint64_t blocks = save_blocks_for_bytes((uint64_t)size);
        used = (used > UINT64_MAX - blocks) ? UINT64_MAX : used + blocks;
    }
    return !ec;
}

SaveCapacity save_capacity_from(bool allocation_known, uint64_t allocation_blocks, uint64_t used_blocks) {
    SaveCapacity c;
    if (!allocation_known) {
        // Unknown size: the legacy answer, so a save that predates allocation records still has room
        // (see save_capacity.hpp). Usage still counts against it.
        c.blocks = used_blocks > kSaveUnknownAllocationBlocks ? used_blocks : kSaveUnknownAllocationBlocks;
        c.free_blocks = c.blocks - used_blocks;
        return c;
    }
    c.blocks = allocation_blocks;
    c.free_blocks = used_blocks >= allocation_blocks ? 0 : allocation_blocks - used_blocks;
    return c;
}

}   // namespace prosper
