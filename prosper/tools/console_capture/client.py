"""ps5debug-NG wire protocol client for optional console capture.

Implements a deliberately small subset of the ps5debug-NG TCP protocol, for observing a title on
the developer's own unlocked PS5: read-only queries (branding, firmware version, foreground app,
process list and maps, bounded process memory reads) plus one software breakpoint for capturing a
GPU submit, with attach/detach/continue around it.

The one operation that changes console state is the breakpoint: the debugger writes a trap
instruction into the target process at the breakpoint address. It is therefore off unless the
caller opts in (``PS5DebugClient(..., allow_breakpoint=True)``), and only slot 0 is usable, so a
session can never hold more than one patched address. Nothing here writes other console memory, and
nothing writes captured bytes to disk: callers decide where captures go (never into the repository).

Console evidence is optional human evidence. No test, CI job or agent workflow may depend on a
console; the tests drive this client against a local fake server only.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

PACKET_MAGIC = 0xFFAABBCC
CMD_PACKET_SIZE = 12

# The only commands this client sends.
CMD_VERSION = 0xBD000001
CMD_FW_VERSION = 0xBD000500
CMD_BRANDING = 0xBD000501
CMD_PLATFORM_ID = 0xBD000502

CMD_PROC_LIST = 0xBDAA0001
CMD_PROC_READ = 0xBDAA0002
CMD_PROC_MAPS = 0xBDAA0004
CMD_FOREGROUND_APP = 0xBDDD0006

CMD_DEBUG_ATTACH = 0xBDBB0001
CMD_DEBUG_DETACH = 0xBDBB0002
CMD_DEBUG_SET_BREAKPOINT = 0xBDBB0003
CMD_DEBUG_GETREGS = 0xBDBB0008
CMD_DEBUG_CONTINUE = 0xBDBB0010

# Status codes (unswapped / server macro representation)
CMD_SUCCESS = 0x40000000
CMD_ERROR = 0xF0000002
CMD_DATA_NULL = 0xF0000003
CMD_ALREADY_DEBUG = 0xF0000008
CMD_INVALID_INDEX = 0xF000000A

# Wire representations (after bitswap32)
WIRE_CMD_SUCCESS = 0x80000000
WIRE_CMD_ERROR = 0xF0000001
WIRE_CMD_DATA_NULL = 0xF0000003
WIRE_CMD_ALREADY_DEBUG = 0xF0000004
WIRE_CMD_INVALID_INDEX = 0xF0000005


# Upper bounds on what one reply may make the client read. A corrupt or hostile reply cannot make it
# allocate or wait for more than this.
MAX_READ_BYTES = 16 * 1024 * 1024
MAX_STRING_BYTES = 4096
MAX_LIST_ENTRIES = 65536
BREAKPOINT_SLOT = 0


def _bounded(value: int, limit: int, what: str) -> int:
    if value < 0 or value > limit:
        raise ValueError(f"{what} {value} outside 0..{limit}")
    return value


def bitswap32(val: int) -> int:
    """Bit-swap adjacent even/odd bit pairs (involution)."""
    val &= 0xFFFFFFFF
    return (((val >> 1) & 0x55555555) | ((val << 1) & 0xAAAAAAAA)) & 0xFFFFFFFF


@dataclass
class ProcessEntry:
    name: str
    pid: int


@dataclass
class MemoryMapEntry:
    name: str
    start: int
    end: int
    offset: int
    prot: int


@dataclass
class ForegroundAppInfo:
    pid: int
    titleid: str
    contentid: str
    name: str
    app_ver: str


@dataclass
class SysVRegs:
    r15: int
    r14: int
    r13: int
    r12: int
    r11: int
    r10: int
    r9: int
    r8: int
    rdi: int
    rsi: int
    rbp: int
    rbx: int
    rdx: int
    rcx: int
    rax: int
    trapno: int
    fs: int
    gs: int
    err: int
    rip: int
    cs: int
    rflags: int
    rsp: int
    ss: int


@dataclass
class InterruptEvent:
    lwpid: int
    status: int
    tdname: str
    regs: SysVRegs


class PS5DebugClient:
    """Client for talking to ps5debug-NG server over TCP."""

    def __init__(
        self, host: str, port: int = 744, timeout: float = 10.0, allow_breakpoint: bool = False
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.allow_breakpoint = allow_breakpoint
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def __enter__(self) -> PS5DebugClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _send_all(self, data: bytes) -> None:
        if not self.sock:
            raise RuntimeError("Client not connected")
        self.sock.sendall(data)

    def _recv_all(self, n: int) -> bytes:
        if not self.sock:
            raise RuntimeError("Client not connected")
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(f"Connection closed after {len(buf)}/{n} bytes")
            buf.extend(chunk)
        return bytes(buf)

    def _send_packet(self, cmd: int, body: bytes = b"") -> None:
        header = struct.pack("<III", PACKET_MAGIC, cmd, len(body))
        self._send_all(header + body)

    def _recv_status(self) -> int:
        raw = self._recv_all(4)
        wire_val = struct.unpack("<I", raw)[0]
        # Unswap to recover server-side macro representation
        return bitswap32(wire_val)

    def _check_status(self) -> None:
        status = self._recv_status()
        if status != CMD_SUCCESS:
            raise RuntimeError(f"Command returned error status: 0x{status:08X}")

    def get_version(self) -> str:
        self._send_packet(CMD_VERSION)
        length_bytes = self._recv_all(4)
        length = _bounded(struct.unpack("<I", length_bytes)[0], MAX_STRING_BYTES, "version length")
        ver_bytes = self._recv_all(length)
        return ver_bytes.decode("ascii", errors="replace").rstrip("\x00")

    def get_fw_version(self) -> int:
        self._send_packet(CMD_FW_VERSION)
        raw = self._recv_all(2)
        return struct.unpack("<H", raw)[0]

    def get_branding(self) -> str:
        self._send_packet(CMD_BRANDING)
        length_bytes = self._recv_all(4)
        length = _bounded(struct.unpack("<I", length_bytes)[0], MAX_STRING_BYTES, "branding length")
        raw = self._recv_all(length)
        # Split on NUL if capability string is present
        parts = raw.split(b"\x00", 1)
        return parts[0].decode("ascii", errors="replace")

    def get_foreground_app(self) -> ForegroundAppInfo:
        self._send_packet(CMD_FOREGROUND_APP)
        self._check_status()
        resp_data = self._recv_all(140)
        pid = struct.unpack("<I", resp_data[:4])[0]
        titleid = resp_data[4:20].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        contentid = resp_data[20:84].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        name = resp_data[84:124].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        app_ver = resp_data[124:140].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        return ForegroundAppInfo(
            pid=pid, titleid=titleid, contentid=contentid, name=name, app_ver=app_ver
        )

    def get_process_list(self) -> list[ProcessEntry]:
        self._send_packet(CMD_PROC_LIST)
        self._check_status()
        num = _bounded(struct.unpack("<I", self._recv_all(4))[0], MAX_LIST_ENTRIES, "process count")
        entries: list[ProcessEntry] = []
        for _ in range(num):
            entry_raw = self._recv_all(36)
            name = entry_raw[:32].split(b"\x00", 1)[0].decode("latin1", errors="replace")
            pid = struct.unpack("<i", entry_raw[32:36])[0]
            entries.append(ProcessEntry(name=name, pid=pid))
        return entries

    def get_memory_maps(self, pid: int) -> list[MemoryMapEntry]:
        body = struct.pack("<I", pid)
        self._send_packet(CMD_PROC_MAPS, body)
        self._check_status()
        num = _bounded(struct.unpack("<I", self._recv_all(4))[0], MAX_LIST_ENTRIES, "map count")
        entries: list[MemoryMapEntry] = []
        for _ in range(num):
            entry_raw = self._recv_all(58)
            name = entry_raw[:32].split(b"\x00", 1)[0].decode("latin1", errors="replace")
            start, end, offset, prot = struct.unpack("<QQQH", entry_raw[32:58])
            entries.append(
                MemoryMapEntry(name=name, start=start, end=end, offset=offset, prot=prot)
            )
        return entries

    def read_memory(self, pid: int, address: int, length: int) -> bytes:
        if length <= 0:
            raise ValueError(f"read length {length} must be positive")
        _bounded(length, MAX_READ_BYTES, "read length")
        body = struct.pack("<IQI", pid, address, length)
        self._send_packet(CMD_PROC_READ, body)
        self._check_status()
        return self._recv_all(length)

    def debug_attach(self, pid: int) -> None:
        body = struct.pack("<I", pid)
        self._send_packet(CMD_DEBUG_ATTACH, body)
        self._check_status()

    def debug_detach(self) -> None:
        self._send_packet(CMD_DEBUG_DETACH)
        self._check_status()

    def set_breakpoint(self, index: int, enabled: bool, address: int) -> None:
        """Arm or clear the one software breakpoint (writes a trap into the target process)."""
        if not self.allow_breakpoint:
            raise PermissionError(
                "set_breakpoint writes into the console process; construct the client with "
                "allow_breakpoint=True to opt in"
            )
        if index != BREAKPOINT_SLOT:
            raise ValueError(f"Breakpoint index {index}: only slot {BREAKPOINT_SLOT} is allowed")
        body = struct.pack("<IIQ", index, 1 if enabled else 0, address)
        self._send_packet(CMD_DEBUG_SET_BREAKPOINT, body)
        self._check_status()

    def get_registers(self, lwpid: int) -> SysVRegs:
        body = struct.pack("<I", lwpid)
        self._send_packet(CMD_DEBUG_GETREGS, body)
        self._check_status()
        raw = self._recv_all(176)
        fields = struct.unpack("<15Q4I5Q", raw)
        return SysVRegs(
            r15=fields[0],
            r14=fields[1],
            r13=fields[2],
            r12=fields[3],
            r11=fields[4],
            r10=fields[5],
            r9=fields[6],
            r8=fields[7],
            rdi=fields[8],
            rsi=fields[9],
            rbp=fields[10],
            rbx=fields[11],
            rdx=fields[12],
            rcx=fields[13],
            rax=fields[14],
            trapno=fields[15],
            fs=fields[16],
            gs=fields[17],
            err=fields[18],
            rip=fields[19],
            cs=fields[20],
            rflags=fields[21],
            rsp=fields[22],
            ss=fields[23],
        )

    def continue_execution(self) -> None:
        # action 0 = resume
        body = struct.pack("<I", 0)
        self._send_packet(CMD_DEBUG_CONTINUE, body)
        self._check_status()


class AsyncInterruptReceiver:
    """Listens on TCP port 755 for debug events sent by the console upon breakpoint hit."""

    def __init__(self, port: int = 755, timeout: float = 30.0, bind_host: str = "0.0.0.0"):
        # The console connects back to this host, so the default listens on every interface;
        # pass the LAN address (or 127.0.0.1 in tests) to narrow it.
        self.port = port
        self.timeout = timeout
        self.bind_host = bind_host
        self.server_sock: socket.socket | None = None
        self.client_sock: socket.socket | None = None

    def start(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.bind_host, self.port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(self.timeout)

    def wait_for_connection(self) -> None:
        if not self.server_sock:
            raise RuntimeError("Interrupt receiver not started")
        self.client_sock, _ = self.server_sock.accept()
        self.client_sock.settimeout(self.timeout)

    def receive_event(self) -> InterruptEvent:
        if not self.client_sock:
            raise RuntimeError("No active interrupt connection from console")
        buf = bytearray()
        while len(buf) < 1184:
            chunk = self.client_sock.recv(1184 - len(buf))
            if not chunk:
                raise ConnectionError("Interrupt channel closed unexpectedly")
            buf.extend(chunk)

        lwpid, status = struct.unpack("<II", buf[0:8])
        tdname = buf[8:48].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        reg_raw = buf[48:224]
        fields = struct.unpack("<15Q4I5Q", reg_raw)
        regs = SysVRegs(
            r15=fields[0],
            r14=fields[1],
            r13=fields[2],
            r12=fields[3],
            r11=fields[4],
            r10=fields[5],
            r9=fields[6],
            r8=fields[7],
            rdi=fields[8],
            rsi=fields[9],
            rbp=fields[10],
            rbx=fields[11],
            rdx=fields[12],
            rcx=fields[13],
            rax=fields[14],
            trapno=fields[15],
            fs=fields[16],
            gs=fields[17],
            err=fields[18],
            rip=fields[19],
            cs=fields[20],
            rflags=fields[21],
            rsp=fields[22],
            ss=fields[23],
        )
        return InterruptEvent(lwpid=lwpid, status=status, tdname=tdname, regs=regs)

    def close(self) -> None:
        if self.client_sock:
            try:
                self.client_sock.close()
            except OSError:
                pass
            self.client_sock = None
        if self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass
            self.server_sock = None

    def __enter__(self) -> AsyncInterruptReceiver:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
