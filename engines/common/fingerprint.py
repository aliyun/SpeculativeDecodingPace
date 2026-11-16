from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
import threading


@lru_cache(maxsize=128)
def _snapshot_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, separators=(',', ':')).encode()).hexdigest()


class PrefixFingerprintCache:
    """Object identity is only a hint; immutable full content decides reuse."""
    def __init__(self):
        self.entries, self.lock = OrderedDict(), threading.Lock()
        self.hits = self.increments = self.rebuilds = 0

    def digest(self, hint, snapshot):
        with self.lock:
            previous = self.entries.get(hint)
            if previous is not None and snapshot == previous[0]:
                self.hits += 1
                self.entries.move_to_end(hint)
                return previous[2]
            size = len(previous[0]) if previous is not None else 0
            if previous is not None and len(snapshot) > size and snapshot[:size] == previous[0]:
                suffix = json.dumps(snapshot[size:], separators=(',', ':')).encode()[1:-1]
                opened = previous[1].copy()
                if size:
                    opened.update(b',')
                opened.update(suffix)
                self.increments += 1
            else:
                encoded = json.dumps(snapshot, separators=(',', ':')).encode()
                opened = hashlib.sha256(encoded[:-1])
                self.rebuilds += 1
            closed = opened.copy()
            closed.update(b']')
            digest = closed.hexdigest()
            self.entries[hint] = (snapshot, opened, digest)
            self.entries.move_to_end(hint)
            if len(self.entries) > 128:
                self.entries.popitem(last=False)
            return digest

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.hits = self.increments = self.rebuilds = 0


_incremental = PrefixFingerprintCache()


def fingerprint(tokens):
    mode = os.environ.get('TRIAGE_FINGERPRINT_CACHE', 'off')
    if mode not in ('off', 'snapshot', 'incremental'):
        raise ValueError('invalid prefix fingerprint cache mode')
    if mode != 'off' and isinstance(tokens, (list, tuple)) and len(tokens) <= 1024:
        snapshot = tuple(tokens)
        # bool / float equality must never alias an integer's different JSON.
        if all(type(t) is int for t in snapshot):
            if mode == 'snapshot':
                return _snapshot_digest(snapshot)
            return _incremental.digest(id(tokens), snapshot)
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()
