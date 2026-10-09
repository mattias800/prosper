// null_device.hpp — the host's null device as a read-only descriptor, for HLE objects that need a
// real descriptor number (close/dup/fstat/lseek behave natively) but answer their reads themselves.
// The guest's served /dev/urandom is the first (hle/fs/guest_devices.hpp).
#pragma once

namespace prosper::host {

// Open the host null device (/dev/null, or NUL on Windows) read-only at a descriptor >= 3, so it can
// never be mistaken for stdio. Returns -1 with errno on failure.
int open_null_device_readonly();

// Close a descriptor this header's open returned.
void close_null_device(int fd);

}   // namespace prosper::host
