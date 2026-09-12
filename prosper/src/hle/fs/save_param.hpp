// save_param.hpp — the save's parameter block: what a save browser shows for it, and the file it
// lives in (#2786).
//
// `sceSaveDataSetParam` / `sceSaveDataGetParam` carry a save's title, sub-title, detail line and a
// title-defined user param. Both were unregistered, so both reached the dispatcher's `return 0` --
// and 0 is SCE_OK for this contract. The guest was told its metadata had been written when nothing
// was written, and read back zeros it could not distinguish from what it had set. That is the
// false-success class tracked by #2081, in the direction that looks safe: nothing crashes, the save
// merely loses its identity.
//
// WHERE IT IS STORED. Inside the save's own directory, at `sce_sys/param.sfo`, which is where the
// console keeps it and where the mounted save exposes it to the guest. prosper already assumes that
// path: `savedata0_dir_mtime()` (used to order the save dialog by date) stats
// `<save dir>/sce_sys/param.sfo` and falls back to the directory when it is absent -- so before this
// it could never find one. Writing the real container rather than a prosper-private sidecar keeps
// those two agreeing and leaves the data readable by anything that knows the format.
// CONFIDENCE: MED that a PS5 save's parameter block is an SFO at this path (it is the PS4-inherited
// shape, and prosper's own code already encodes that assumption); HIGH on the container format
// itself, which is published and round-tripped by tests/hle/test_savedata_param.cpp.
//
// The file is created only when the guest actually calls `sceSaveDataSetParam`. A save whose title
// never sets a param block gets no `sce_sys` directory from us, so we never add a file to a save
// directory the guest did not ask for.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prosper {

// One save's parameter block, in host form. The string fields are stored without their trailing NUL
// and are clamped to the ABI's maxima when they cross the guest boundary.
struct SaveDataParam {
    std::string title;        // SCE_SAVE_DATA_TITLE_MAXSIZE 128, including the NUL
    std::string sub_title;    // SCE_SAVE_DATA_SUBTITLE_MAXSIZE 128
    std::string detail;       // SCE_SAVE_DATA_DETAIL_MAXSIZE 1024
    uint32_t user_param = 0;
    // Modification time, in seconds since the epoch. NOT carried in the SFO: on the console this is
    // the file's own timestamp, which is also what savedata0_dir_mtime() reads, so it is derived
    // from the stored file rather than duplicated inside it. A guest that sets it explicitly moves
    // the file's timestamp.
    int64_t mtime = 0;

    bool operator==(const SaveDataParam&) const = default;
};

// SFO container, encode and decode. Pure: no filesystem, so the format can be tested on its own.
// `title_id` and `directory` are the locally derivable identity keys a real param.sfo also carries;
// they are written for fidelity and ignored on read (prosper answers identity questions from the
// running application and the mount, never from a file a save directory happens to contain).
std::vector<uint8_t> encode_param_sfo(const SaveDataParam& param, const std::string& title_id,
                                      const std::string& directory);
// Returns false and leaves `out` untouched when the bytes are not a param.sfo this can read. A
// malformed or truncated file is never partially applied: an unreadable block must look absent, not
// look like a block whose fields happen to be empty.
bool decode_param_sfo(const uint8_t* bytes, size_t size, SaveDataParam& out);

// Persistence against one save directory (the host path of a mounted save). `store` creates
// `sce_sys/` as needed and applies `param.mtime` to the file when it is non-zero. `load` returns
// false when no readable block exists, in which case `out` is untouched -- the caller decides what
// an absent block means rather than being handed a zeroed one that looks stored.
bool save_param_store(const std::string& save_dir, const SaveDataParam& param,
                      const std::string& title_id, const std::string& directory);
bool save_param_load(const std::string& save_dir, SaveDataParam& out);

// The path `store`/`load` operate on, exposed so a test and a diagnostic can name the same file the
// implementation uses instead of re-spelling the layout.
std::string save_param_path(const std::string& save_dir);

}   // namespace prosper
