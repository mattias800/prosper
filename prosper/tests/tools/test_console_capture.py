"""Unit tests for the console_capture client against a fake in-memory server."""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from prosper.tools.console_capture.client import (
    CMD_BRANDING,
    CMD_ERROR,
    CMD_FOREGROUND_APP,
    CMD_FW_VERSION,
    CMD_PROC_MAPS,
    CMD_PROC_READ,
    CMD_SUCCESS,
    MAX_LIST_ENTRIES,
    MAX_READ_BYTES,
    PACKET_MAGIC,
    WIRE_CMD_ERROR,
    WIRE_CMD_SUCCESS,
    AsyncInterruptReceiver,
    PS5DebugClient,
    bitswap32,
)


def test_bitswap32_involution():
    """bitswap32 must be an involution: f(f(x)) == x."""
    assert bitswap32(bitswap32(0x12345678)) == 0x12345678
    assert bitswap32(CMD_SUCCESS) == WIRE_CMD_SUCCESS
    assert bitswap32(CMD_ERROR) == WIRE_CMD_ERROR
    assert bitswap32(WIRE_CMD_SUCCESS) == CMD_SUCCESS
    assert bitswap32(WIRE_CMD_ERROR) == CMD_ERROR


def test_bitswap32_mutation_redness():
    """Omitting bitswap32 causes status checks to fail."""
    # Wire success is 0x80000000. Without swap, it is not equal to CMD_SUCCESS (0x40000000).
    assert WIRE_CMD_SUCCESS != CMD_SUCCESS
    assert bitswap32(WIRE_CMD_SUCCESS) == CMD_SUCCESS


class FakeServer:
    def __init__(self, port: int = 0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.thread: threading.Thread | None = None
        self.running = True

    def start(self, handler):
        def loop():
            conn, _ = self.sock.accept()
            try:
                handler(conn)
            finally:
                conn.close()

        self.thread = threading.Thread(target=loop)
        self.thread.daemon = True
        self.thread.start()

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass
        if self.thread:
            self.thread.join(timeout=1.0)


def test_client_branding_and_fw():
    server = FakeServer()

    def handle(conn: socket.socket):
        # 1. Expect CMD_BRANDING
        hdr = conn.recv(12)
        magic, cmd, datalen = struct.unpack("<III", hdr)
        assert magic == PACKET_MAGIC
        assert cmd == CMD_BRANDING
        brand_bytes = b"ps5debug-NG by OSR v1.3.2 [938d1a7350180738]\x001.1"
        conn.sendall(struct.pack("<I", len(brand_bytes)) + brand_bytes)

        # 2. Expect CMD_FW_VERSION
        hdr = conn.recv(12)
        magic, cmd, datalen = struct.unpack("<III", hdr)
        assert magic == PACKET_MAGIC
        assert cmd == CMD_FW_VERSION
        conn.sendall(struct.pack("<H", 1340))

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port) as client:
            brand = client.get_branding()
            assert brand == "ps5debug-NG by OSR v1.3.2 [938d1a7350180738]"
            fw = client.get_fw_version()
            assert fw == 1340
    finally:
        server.close()


def test_client_status_bitswap_and_error():
    server = FakeServer()

    def handle(conn: socket.socket):
        # Read packet
        hdr = conn.recv(12)
        magic, cmd, datalen = struct.unpack("<III", hdr)
        assert magic == PACKET_MAGIC
        # Send back WIRE_CMD_ERROR (bit-swapped CMD_ERROR)
        conn.sendall(struct.pack("<I", WIRE_CMD_ERROR))

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port) as client:
            with pytest.raises(RuntimeError) as exc_info:
                client.debug_detach()
            assert f"0x{CMD_ERROR:08X}" in str(exc_info.value)
    finally:
        server.close()


def test_client_foreground_app():
    server = FakeServer()

    def handle(conn: socket.socket):
        hdr = conn.recv(12)
        magic, cmd, datalen = struct.unpack("<III", hdr)
        assert magic == PACKET_MAGIC
        assert cmd == CMD_FOREGROUND_APP
        # Reply with WIRE_CMD_SUCCESS
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))
        # 140 bytes of response
        resp = bytearray(140)
        struct.pack_into("<I", resp, 0, 1234)  # pid
        resp[4:14] = b"PPSA28183\x00"  # titleid
        resp[20:30] = b"EP0001-PPSA"  # contentid
        resp[84:93] = b"eboot.bin"  # name
        resp[124:129] = b"01.00"  # app_ver
        conn.sendall(resp)

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port) as client:
            app = client.get_foreground_app()
            assert app.pid == 1234
            assert app.titleid == "PPSA28183"
            assert app.name == "eboot.bin"
    finally:
        server.close()


def test_client_memory_maps_and_read():
    server = FakeServer()

    def handle(conn: socket.socket):
        # 1. Maps
        hdr = conn.recv(12)
        magic, cmd, datalen = struct.unpack("<III", hdr)
        assert cmd == CMD_PROC_MAPS
        body = conn.recv(datalen)
        assert struct.unpack("<I", body)[0] == 100
        # Success status
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))
        # 1 entry
        conn.sendall(struct.pack("<I", 1))
        entry = bytearray(58)
        entry[:16] = b"libSceAgcDriver\x00"
        struct.pack_into("<QQQH", entry, 32, 0x10000, 0x20000, 0, 5)  # r-x
        conn.sendall(entry)

        # 2. Read memory
        hdr = conn.recv(12)
        magic, cmd, datalen = struct.unpack("<III", hdr)
        assert cmd == CMD_PROC_READ
        body = conn.recv(datalen)
        pid, addr, length = struct.unpack("<IQI", body)
        assert pid == 100
        assert addr == 0x10000
        assert length == 16
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))
        conn.sendall(bytes([0x48, 0x89, 0xFE, 0x48, 0x8D, 0x3D] + [0] * 10))

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port) as client:
            maps = client.get_memory_maps(100)
            assert len(maps) == 1
            assert maps[0].name == "libSceAgcDriver"
            assert maps[0].start == 0x10000
            assert maps[0].prot == 5

            mem = client.read_memory(100, 0x10000, 16)
            assert len(mem) == 16
            assert mem[:3] == b"\x48\x89\xfe"
    finally:
        server.close()


def test_client_breakpoint_and_registers():
    server = FakeServer()

    def handle(conn: socket.socket):
        # Attach
        hdr = conn.recv(12)
        conn.recv(struct.unpack("<III", hdr)[2])
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))

        # Set breakpoint
        hdr = conn.recv(12)
        conn.recv(struct.unpack("<III", hdr)[2])
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))

        # Get regs
        hdr = conn.recv(12)
        conn.recv(struct.unpack("<III", hdr)[2])
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))
        regs_blob = bytearray(176)
        # rdi is at offset 64 (index 8 of uint64)
        struct.pack_into("<Q", regs_blob, 64, 0x40001000)
        # rsi is at offset 72 (index 9 of uint64)
        struct.pack_into("<Q", regs_blob, 72, 0x100)
        conn.sendall(regs_blob)

        # Continue
        hdr = conn.recv(12)
        conn.recv(struct.unpack("<III", hdr)[2])
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port, allow_breakpoint=True) as client:
            client.debug_attach(100)
            client.set_breakpoint(0, True, 0x123456)
            regs = client.get_registers(1)
            assert regs.rdi == 0x40001000
            assert regs.rsi == 0x100
            client.continue_execution()
    finally:
        server.close()


def test_short_read_failure():
    server = FakeServer()

    def handle(conn: socket.socket):
        conn.recv(12)
        # Only send 2 bytes of 4-byte status then close
        conn.sendall(b"\x80\x00")
        conn.close()

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port) as client:
            with pytest.raises(ConnectionError):
                client.get_foreground_app()
    finally:
        server.close()


def test_breakpoint_needs_opt_in_and_slot_zero():
    """The breakpoint writes into the console process: refused before anything is sent."""
    client = PS5DebugClient("127.0.0.1")  # never connected: a refusal must not reach the wire
    with pytest.raises(PermissionError):
        client.set_breakpoint(0, True, 0x123456)
    client = PS5DebugClient("127.0.0.1", allow_breakpoint=True)
    with pytest.raises(ValueError):
        client.set_breakpoint(1, True, 0x123456)


def test_read_memory_length_is_bounded():
    client = PS5DebugClient("127.0.0.1")  # refused before any I/O
    for length in (0, -1, MAX_READ_BYTES + 1):
        with pytest.raises(ValueError):
            client.read_memory(100, 0x10000, length)


def test_reply_count_is_bounded():
    """A corrupt entry count fails fast instead of waiting for gigabytes."""
    server = FakeServer()

    def handle(conn: socket.socket):
        hdr = conn.recv(12)
        conn.recv(struct.unpack("<III", hdr)[2])
        conn.sendall(struct.pack("<I", WIRE_CMD_SUCCESS))
        conn.sendall(struct.pack("<I", MAX_LIST_ENTRIES + 1))

    server.start(handle)
    try:
        with PS5DebugClient("127.0.0.1", port=server.port) as client:
            with pytest.raises(ValueError):
                client.get_memory_maps(100)
    finally:
        server.close()


def test_interrupt_receiver_on_loopback():
    """The receiver parses one 1184-byte event; bound to loopback, on an ephemeral port."""
    with AsyncInterruptReceiver(port=0, timeout=5.0, bind_host="127.0.0.1") as receiver:
        port = receiver.server_sock.getsockname()[1]
        packet = bytearray(1184)
        struct.pack_into("<II", packet, 0, 77, 5)
        packet[8:14] = b"render"
        struct.pack_into("<Q", packet, 48 + 8 * 8, 0x40001000)  # rdi
        struct.pack_into("<Q", packet, 48 + 15 * 8 + 4 * 4, 0x804F6CCC0)  # rip

        def send():
            with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
                sock.sendall(bytes(packet))

        sender = threading.Thread(target=send)
        sender.start()
        receiver.wait_for_connection()
        event = receiver.receive_event()
        sender.join()
    assert event.lwpid == 77
    assert event.tdname == "render"
    assert event.regs.rdi == 0x40001000
    assert event.regs.rip == 0x804F6CCC0
