"""A WebSocket server small enough to read, with no dependencies.

WHY NOT A LIBRARY. This exists to put numbers on a screen while a control loop
runs; adding `websockets` or `aiohttp` to a conda env that already fights over
numpy, eigenpy and a hand-built crocoddyl is a worse trade than eighty lines of
RFC 6455. Only what a telemetry panel needs is implemented: the handshake, text
frames both ways, close, ping/pong. No extensions, no fragmentation on send, no
TLS. Frames arriving fragmented are reassembled because browsers may send them;
frames are never sent fragmented because there is no reason to.

THREADING, AND THE GIL. Every connection gets a thread, and those threads are
subject to the same starvation that broke the DDS receive path (see
croco/plant/dds_plant.py): while crocoddyl holds the GIL nothing here runs. That
is FINE, and it is why the panel PUSHES from the control thread rather than
being polled: telemetry is written inside `on_step`, which already holds the
GIL, and the socket threads only need to be scheduled between periods to drain
what the browser sent. A GUI may lag a period. A controller may not.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_OP_TEXT, _OP_BIN, _OP_CLOSE, _OP_PING, _OP_PONG = 0x1, 0x2, 0x8, 0x9, 0xA


def _frame(payload: bytes, opcode=_OP_TEXT) -> bytes:
    """One unmasked server->client frame. Servers never mask (RFC 6455 5.1)."""
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < (1 << 16):
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    return head + payload


class _Conn:
    """One browser. `send` is safe to call from the control thread."""

    def __init__(self, sock, on_message):
        self.sock = sock
        self.on_message = on_message
        self.alive = True
        self._lock = threading.Lock()

    def send(self, obj):
        if not self.alive:
            return
        data = _frame(json.dumps(obj).encode())
        try:
            with self._lock:
                self.sock.sendall(data)
        except OSError:
            self.alive = False          # the browser went away; not an error

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf

    def pump(self):
        """Read frames until the socket closes. Runs on its own thread."""
        parts, part_op = [], None
        try:
            while self.alive:
                b0, b1 = self._recv_exact(2)
                fin, op = b0 & 0x80, b0 & 0x0F
                masked, n = b1 & 0x80, b1 & 0x7F
                if n == 126:
                    n = struct.unpack("!H", self._recv_exact(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", self._recv_exact(8))[0]
                mask = self._recv_exact(4) if masked else None
                data = self._recv_exact(n) if n else b""
                if mask:
                    data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
                if op == _OP_CLOSE:
                    break
                if op == _OP_PING:
                    with self._lock:
                        self.sock.sendall(_frame(data, _OP_PONG))
                    continue
                if op == _OP_PONG:
                    continue
                if op == 0x0:                       # continuation
                    parts.append(data)
                else:
                    parts, part_op = [data], op
                if not fin:
                    continue
                payload, parts = b"".join(parts), []
                if part_op == _OP_TEXT and self.on_message:
                    try:
                        self.on_message(json.loads(payload.decode()))
                    except Exception:               # noqa: BLE001
                        pass                        # a bad message is the
                        # browser's problem, never the control loop's
        except (OSError, ConnectionError, struct.error):
            pass
        finally:
            self.alive = False
            try:
                self.sock.close()
            except OSError:
                pass


class Server:
    """Serve one HTML page and accept WebSocket upgrades on the same port.

    `broadcast` is the only method the control loop touches, and it never
    blocks on a slow client for longer than the socket buffer allows -- a dead
    or wedged browser marks itself not-alive and is dropped on the next sweep.
    """

    def __init__(self, page_path, on_message=None, host="127.0.0.1", port=8770):
        self.page_path = page_path
        self.on_message = on_message
        self.host, self.port = host, port
        self.conns = []
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(8)
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def url(self):
        return "http://%s:%d/" % (self.host, self.port)

    def broadcast(self, obj):
        with self._lock:
            conns = list(self.conns)
        for c in conns:
            c.send(obj)
        if any(not c.alive for c in conns):
            with self._lock:
                self.conns = [c for c in self.conns if c.alive]

    def close(self):
        self._stop = True
        try:
            self._srv.close()
        except OSError:
            pass

    # -- internals ---------------------------------------------------------

    def _accept(self):
        while not self._stop:
            try:
                sock, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _serve(self, sock):
        try:
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = sock.recv(4096)
                if not chunk:
                    return
                req += chunk
            head = req.split(b"\r\n\r\n", 1)[0].decode("latin1")
            lines = head.split("\r\n")
            hdr = {}
            for line in lines[1:]:
                if ": " in line:
                    k, v = line.split(": ", 1)
                    hdr[k.lower()] = v
            if "sec-websocket-key" in hdr:
                accept = base64.b64encode(hashlib.sha1(
                    hdr["sec-websocket-key"].encode() + GUID).digest()).decode()
                sock.sendall(("HTTP/1.1 101 Switching Protocols\r\n"
                              "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                              "Sec-WebSocket-Accept: %s\r\n\r\n" % accept).encode())
                conn = _Conn(sock, self.on_message)
                with self._lock:
                    self.conns.append(conn)
                conn.pump()
                return
            body = open(self.page_path, "rb").read() \
                if os.path.exists(self.page_path) else b"panel page missing"
            sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; "
                         b"charset=utf-8\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\n"
                         b"Cache-Control: no-store\r\nConnection: close\r\n\r\n"
                         + body)
        except OSError:
            pass
        finally:
            if "sec-websocket-key" not in locals().get("hdr", {}):
                try:
                    sock.close()
                except OSError:
                    pass
