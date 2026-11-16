from collections import OrderedDict
import json
import os
from pathlib import Path
import stat
import threading

_CACHE = OrderedDict()
_LOCK = threading.RLock()
_PID = os.getpid()
_LIMIT = 8


def signature(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def read_snapshot(path, *, prefix='EXPERIMENT'):
    global _PID
    path = Path(path)
    with _LOCK:
        if _PID != os.getpid():
            _CACHE.clear()
            _PID = os.getpid()
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(prefix+'_CONTROL_SYMLINK')
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(prefix+'_CONTROL_NOT_REGULAR')
        key = str(path)
        stamp = signature(st)
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            _CACHE.move_to_end(key)
            return stamp, dict(cached[1])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            actual = os.fstat(stream.fileno())
            if not stat.S_ISREG(actual.st_mode):
                raise ValueError(prefix+'_CONTROL_NOT_REGULAR')
            value = json.load(stream)
            if signature(os.fstat(stream.fileno())) != signature(actual):
                raise ValueError(prefix+'_CONTROL_CHANGED_DURING_READ')
        if type(value) is not dict:
            raise ValueError(prefix+'_CONTROL_NOT_OBJECT')
        stamp = signature(actual)
        _CACHE[key] = stamp, value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _LIMIT:
            _CACHE.popitem(last=False)
        return stamp, dict(value)
