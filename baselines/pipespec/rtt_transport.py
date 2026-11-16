
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

MAGIC = b"PSX1"
KIND_NEW_REQUEST = 0
KIND_VALUE = 1
KIND_CLOSE = 2
FLAG_COMPLETED = 1 << 0
FLAG_EOS = 1 << 1

HDR = struct.Struct("<4sBBIQI")
HEADER_SIZE = HDR.size


class Frame:
    __slots__ = ("kind", "flags", "req_id", "ver", "ids")

    def __init__(self, kind, flags, req_id, ver, ids):
        self.kind = kind
        self.flags = flags
        self.req_id = req_id
        self.ver = ver
        self.ids = ids

    @property
    def completed(self):
        return bool(self.flags & FLAG_COMPLETED)

    @property
    def eos(self):
        return bool(self.flags & FLAG_EOS)

    def encode(self):
        arr = np.asarray(self.ids, dtype="<i8").reshape(-1)
        return HDR.pack(MAGIC, self.kind, self.flags, self.req_id, self.ver,
                        int(arr.size)) + arr.tobytes()

    def __repr__(self):
        return (f"Frame(kind={self.kind} req={self.req_id} ver={self.ver} "
                f"n={len(self.ids)} completed={self.completed})")


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return bytes(buf)


def recv_frame(sock):
    magic, kind, flags, req_id, ver, n = HDR.unpack(recv_exact(sock, HEADER_SIZE))
    if magic != MAGIC:
        raise ConnectionError("bad magic %r" % (magic,))
    if n:
        ids = np.frombuffer(recv_exact(sock, int(n) * 8), dtype="<i8").copy()
    else:
        ids = np.empty(0, dtype="<i8")
    return Frame(kind, flags, req_id, ver, ids)


def as_i8(ids):
    """Accept torch tensors on any device, numpy arrays or lists; return int64 numpy."""
    if hasattr(ids, "detach"):
        ids = ids.detach().to("cpu").numpy()
    return np.asarray(ids, dtype="<i8").reshape(-1)


class LatestSlot:
    """Latest-value buffer shared by a stage and its transport threads."""

    def __init__(self, name="slot"):
        self.name = name
        self._cv = threading.Condition()
        self._latest = None
        self._ver = 0
        self._closed = False


    def publish(self, kind, req_id, ids, completed=False, eos=False):
        flags = (FLAG_COMPLETED if completed else 0) | (FLAG_EOS if eos else 0)
        with self._cv:
            self._ver += 1
            self._latest = Frame(kind, flags, int(req_id), self._ver, as_i8(ids))
            self._cv.notify_all()
            return self._ver

    def close(self):
        with self._cv:
            self._ver += 1
            self._latest = Frame(KIND_CLOSE, 0, 0, self._ver, np.empty(0, dtype="<i8"))
            self._closed = True
            self._cv.notify_all()

    def mark_closed(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self):
        with self._cv:
            return self._closed


    def latest(self):
        with self._cv:
            return self._latest

    def wait_newer(self, ver, timeout=None):
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while True:
                if self._latest is not None and self._latest.ver > ver:
                    return self._latest
                if self._closed:
                    return self._latest
                wait = 0.2 if deadline is None else min(0.2, deadline - time.time())
                if wait <= 0:
                    return self._latest
                self._cv.wait(wait)

    def wait_request(self, req_id, timeout=None):
        """Wait until some frame for `req_id` (or a later request) exists."""
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while True:
                if self._latest is not None and self._latest.req_id >= req_id:
                    return self._latest
                if self._closed:
                    return self._latest
                wait = 0.2 if deadline is None else min(0.2, deadline - time.time())
                if wait <= 0:
                    return self._latest
                self._cv.wait(wait)

    def put(self, frame):
        """Used by a receiving transport thread (latest wins)."""
        with self._cv:
            if self._latest is None or frame.req_id != self._latest.req_id or frame.ver >= self._latest.ver:
                self._latest = frame
            if frame.kind == KIND_CLOSE:
                self._closed = True
            self._cv.notify_all()


class _Base:
    def __init__(self, slot, name):
        self.slot = slot
        self.name = name
        self._stop = False
        self._sock = None
        self._peer = None

    def close(self):
        self._stop = True
        for s in (self._peer, self._sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


class PusherServer(_Base):
    """Listen; push the slot's newest frame to one peer."""

    def __init__(self, slot, host, port, name="push", send_buffer=8 << 20, accept_timeout=1800.0):
        super().__init__(slot, name)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._srv.settimeout(0.5)
        self.port = self._srv.getsockname()[1]
        self.accept_timeout = accept_timeout
        threading.Thread(target=self._accept, name=f"{name}-accept", daemon=True).start()
        threading.Thread(target=self._push, name=f"{name}-push", daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                peer, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
            self._peer = peer
            self._last_ver = 0


    def _push(self):
        while not self._stop:
            if self._peer is None:
                time.sleep(0.05)
                continue
            frame = self.slot.wait_newer(getattr(self, "_last_ver", 0), timeout=0.5)
            if frame is None or frame.ver <= getattr(self, "_last_ver", 0):
                continue
            self._last_ver = frame.ver
            try:
                self._peer.sendall(frame.encode())
            except OSError:
                self._peer = None
                self._last_ver = 0


class SenderClient(_Base):

    def __init__(self, slot, host, port, name="send", retry_s=0.2, timeout_s=1800.0):
        super().__init__(slot, name)
        self.host, self.port, self.retry_s = host, port, retry_s
        self._connect(timeout_s)
        self._last_ver = 0
        self.reconnects = 0
        threading.Thread(target=self._push, name=f"{name}-push", daemon=True).start()

    def _connect(self, timeout_s=1800.0):
        deadline = time.time() + timeout_s
        while True:
            try:
                self._peer = socket.create_connection((self.host, self.port), timeout=5)
                self._peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
                return
            except OSError:
                if self._stop or time.time() > deadline:
                    raise
                time.sleep(self.retry_s)

    def _push(self):
        while not self._stop:
            frame = self.slot.wait_newer(self._last_ver, timeout=0.5)
            if frame is None or frame.ver <= self._last_ver:
                continue
            try:
                self._peer.sendall(frame.encode())
                self._last_ver = frame.ver
            except OSError:

                try:
                    self._peer.close()
                except OSError:
                    pass
                try:
                    self._connect()
                    self.reconnects += 1
                except OSError:
                    return


class SinkServer(_Base):
    """Listen; drain one peer into the slot."""

    def __init__(self, slot, host, port, name="sink", accept_timeout=1800.0):
        super().__init__(slot, name)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._srv.settimeout(0.5)
        self.port = self._srv.getsockname()[1]
        self.accept_timeout = accept_timeout
        threading.Thread(target=self._run, name=f"{name}-run", daemon=True).start()

    def _run(self):
        while not self._stop:
            try:
                peer, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._peer = peer
            try:
                while not self._stop:
                    self.slot.put(recv_frame(peer))
            except (OSError, ConnectionError):
                continue


class DrainClient(_Base):
    """Connect out; drain into the slot."""

    def __init__(self, slot, host, port, name="drain", retry_s=0.2, timeout_s=1800.0):
        super().__init__(slot, name)
        self.host, self.port, self.retry_s = host, port, retry_s
        self.reconnects = 0
        self._connect(timeout_s)
        threading.Thread(target=self._run, name=f"{name}-run", daemon=True).start()

    def _connect(self, timeout_s=1800.0):
        deadline = time.time() + timeout_s
        while True:
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=5)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return
            except OSError:
                if self._stop or time.time() > deadline:
                    raise
                time.sleep(self.retry_s)

    def _run(self):
        while not self._stop:
            try:
                self.slot.put(recv_frame(self._sock))
            except (OSError, ConnectionError):
                if self._stop:
                    return
                try:
                    self._sock.close()
                except OSError:
                    pass
                while not self._stop:
                    try:
                        self._connect(timeout_s=1.0)
                        self.reconnects += 1
                        break
                    except OSError:
                        time.sleep(min(2.0, self.retry_s + 0.2))


Publisher = PusherServer
Subscriber = DrainClient
