#pragma once
#include <cstdio>
#include <string>

namespace prosper {

// POSIX guest filesystem barrier. Retain one private descriptor per encountered filesystem
// until process exit, so closing a file, unmounting a save or switching app0 cannot lose dirty
// data. A barrier flushes tracked writable guest stdio, then each retained filesystem. This
// includes unrelated host files on those filesystems; it is NOT directory-level isolation.
// Windows keeps its existing process-stdio fallback; the note functions are no-ops there.
void guest_sync_note_fd(int fd);
void guest_sync_note_root(const std::string& root);
void guest_sync_note_path(const std::string& path);
void guest_sync_note_stream(FILE* stream, bool writable);
int guest_sync_close_stream(FILE* stream);
void guest_sync();

} // namespace prosper
