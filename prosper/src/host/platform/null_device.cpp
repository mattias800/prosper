// null_device.cpp — see null_device.hpp.
#include "host/platform/null_device.hpp"

#include <cerrno>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace prosper::host {

int open_null_device_readonly() {
#ifdef _WIN32
    int fd = ::_open("NUL", _O_RDONLY | _O_BINARY);
    // _dup hands out the lowest free number, so hold 0-2 until a higher one comes back.
    int low[3] = {-1, -1, -1};
    int held = 0;
    while (fd >= 0 && fd < 3 && held < 3) {
        low[held++] = fd;
        fd = ::_dup(fd);
    }
    const int error = errno;
    for (int i = 0; i < held; ++i) ::_close(low[i]);
    errno = error;
    return fd;
#else
    const int fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (fd < 0 || fd >= 3) return fd;
    const int high = ::fcntl(fd, F_DUPFD_CLOEXEC, 3);
    const int error = errno;
    ::close(fd);
    errno = error;
    return high;
#endif
}

void close_null_device(int fd) {
#ifdef _WIN32
    ::_close(fd);
#else
    ::close(fd);
#endif
}

}   // namespace prosper::host
