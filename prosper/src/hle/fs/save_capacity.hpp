// save_capacity.hpp — a /savedata0 save's allocation, and how much of it is used (#3654).
//
// Every sceSaveDataMount* call carries the save's requested size in blocks, and
// sceSaveDataGetMountInfo answers "how big is this save and how much is free". prosper used to
// discard the request and answer a fixed 0x40000 / 0x40000 whatever was mounted or written, so a
// title asking how much room remained got a number derived from nothing. This is the one backend
// both halves go through.
//
// ## The accounting contract, and how sure each part is
//
//   * One block is 32 KiB. CONFIDENCE: MED -- the PS4-inherited libSceSaveData unit
//     (SCE_SAVE_DATA_BLOCK_SIZE), and the NIDs carrying it are unchanged in the PS5 3.20 export
//     table. Corroborated, not proven, by DOLL's live Mount3 requests of 0x60 and 0x105 blocks:
//     0x60 = 96 is exactly the PS4 SDK's documented minimum allocation (SCE_SAVE_DATA_BLOCKS_MIN2).
//   * The allocation is fixed when the save is CREATED and is not changed by a later mount's
//     request. CONFIDENCE: MED -- the published contract describes `blocks` as the size to create;
//     no title has been observed resizing through a mount.
//   * Usage is the sum over regular files of each file's size rounded UP to whole blocks, and
//     directories and prosper's bookkeeping are charged nothing. CONFIDENCE: LOW -- the console
//     stores a save as a filesystem image with its own metadata overhead, which prosper cannot know.
//     This is the simplest rule that is monotonic in what the guest writes; it is not a claim about
//     the console's exact figure.
//   * free = allocation - usage, and 0 when usage exceeds the allocation. The advertised allocation
//     is NEVER silently enlarged to cover usage (the host directory is not size-limited, so a title
//     can write past what it asked for; reporting that honestly is 0 free, not a bigger save).
//   * Nothing here ENFORCES the allocation on file writes. That is a separate contract.
//
// ## Where the allocation is kept
//
// Outside the guest-visible save contents, in `<title save dir>/.prosper-capacity/<dirName>`: one
// small text file per save, written to a temporary name and renamed into place, so an interrupted
// write leaves either the old record or the new one and never a torn one. The leading '.' keeps it
// out of sceSaveDataDirNameSearch (savedata0_list_dirs skips dot entries), and savedata_dirname_ok
// refuses the one guest dirName that would mount it. Per title because the directory it sits in is
// already per title (#2734), so two titles with the same dirName keep separate records.
//
// ## Saves with no record, and records that do not parse
//
// A save written before this existed has no record. Its size is unknown -- prosper never saw the
// request that created it -- so the first mount that DOES carry a request adopts that request as
// its allocation and records it: the title's own statement of how big its save is, which is the
// only primary evidence there is. Nothing is moved, deleted or rewritten inside the save.
// A record that does not parse is left exactly as it is (never overwritten, never deleted) and the
// save is reported as allocation-unknown.
//
// Allocation-unknown keeps the answer prosper gave before it tracked allocations at all: blocks =
// max(kSaveUnknownAllocationBlocks, used), free = blocks - used. That is deliberately NOT "full":
// every save a user already has was created before these records existed, and reporting those as
// full would make a title that checks free space before writing refuse to save over them -- a
// regression from what already worked. The figure is not derived from the save; it is the legacy
// constant, kept so an unknown allocation behaves exactly as every allocation did before #3654, and
// usage is still subtracted so free space never exceeds what the save could plausibly hold.
// CONFIDENCE: LOW on the number itself (0x40000 blocks = 8 GiB, never derived from anything).
#pragma once

#include <cstdint>
#include <string>

namespace prosper {

inline constexpr uint64_t kSaveDataBlockBytes = 32768;   // CONFIDENCE: MED, see above

// What an allocation prosper never saw is reported as: the pre-#3654 fixed answer. CONFIDENCE: LOW.
inline constexpr uint64_t kSaveUnknownAllocationBlocks = 0x40000;

// ceil(bytes / block), overflow-safe for every uint64_t input.
constexpr uint64_t save_blocks_for_bytes(uint64_t bytes) {
    return bytes / kSaveDataBlockBytes + (bytes % kSaveDataBlockBytes ? 1 : 0);
}

// The bookkeeping directory's name inside a title's save root. Reserved: no guest save may use it.
inline constexpr const char* kSaveCapacityDirName = ".prosper-capacity";

enum class SaveAllocationState {
    Present,      // a well-formed record; `blocks` holds it
    Absent,       // no record (a save that predates this, or one created without a request)
    Corrupt,      // a record exists and does not parse -- left untouched
    Unreadable,   // the record exists but could not be read (host I/O error)
};

// Read / write one save's allocation record under `title_save_dir` (savedata0_dir()).
SaveAllocationState save_allocation_read(const std::string& title_save_dir, const std::string& dirname,
                                         uint64_t& blocks);
bool save_allocation_write(const std::string& title_save_dir, const std::string& dirname,
                           uint64_t blocks);

// The mount path's bookkeeping, called once a mount has succeeded. `created` says whether this mount
// created the save; `requested_blocks` is what the mount descriptor asked for (0 = no request).
// Records the request on creation, adopts it for a save with no record, and otherwise leaves the
// existing record alone. Returns false only when a record should have been written and could not be.
bool save_allocation_note_mount(const std::string& title_save_dir, const std::string& dirname,
                                bool created, uint64_t requested_blocks);

// Blocks used by the files under `save_dir`, by the rule above. False when the directory cannot be
// walked (vanished, unreadable); `used` is then unspecified. Saturates rather than wrapping.
bool save_usage_blocks(const std::string& save_dir, uint64_t& used);

// The pair sceSaveDataGetMountInfo reports, from an allocation (if known) and a usage.
struct SaveCapacity {
    uint64_t blocks = 0;
    uint64_t free_blocks = 0;
};
SaveCapacity save_capacity_from(bool allocation_known, uint64_t allocation_blocks, uint64_t used_blocks);

}   // namespace prosper
