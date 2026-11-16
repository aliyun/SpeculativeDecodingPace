import os
import time

SLOW_NS = 80_000_000
REPORT_EVERY = 512
MAX_GROUPS = 8192


def enabled():
    """Whether in-memory round timing is requested."""
    value = os.environ.get('TRIAGE_ROUND_TIMING', '0')
    if value not in ('0', '1'):
        raise ValueError('ROUND_TIMING')
    return value == '1'


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(n=len(ordered), p50=_quantile(ordered, .5), p90=_quantile(ordered, .9),
                mean=sum(ordered) / len(ordered), total=sum(ordered))


class RoundTiming:
    """Three-segment round timing with an aggregate report, or a no-op when off."""

    def __init__(self, emit=None, clock=time.monotonic_ns, slow_ns=SLOW_NS,
                 report_every=REPORT_EVERY):
        self.enabled = enabled()
        self.emit = emit
        self.clock = clock
        self.slow_ns = slow_ns
        self.report_every = report_every
        self._issued = {}
        self._verified = {}
        self._last_done = {}
        self._closed = 0
        self._parts = []

    # --- recording -------------------------------------------------------
    def issue(self, ids):
        """A verify RPC is being issued for these rows."""
        if not self.enabled:
            return
        now = self.clock()
        for rid in ids:
            previous = self._last_done.get(rid)
            self._issued[rid] = (now, None if previous is None else now - previous)

    def verified(self):
        """The verify RPC returned; every issued row has its rpc segment closed."""
        if not self.enabled:
            return
        now = self.clock()
        for rid, (issued_at, _) in self._issued.items():
            entry = self._verified.get(rid)
            if entry is None or entry[0] < issued_at:
                self._verified[rid] = (issued_at, now - issued_at, None)

    def done(self, ids):
        """The round has been committed; close and keep the three segments."""
        if not self.enabled:
            return
        now = self.clock()
        for rid in ids:
            issued = self._issued.pop(rid, None)
            verified = self._verified.pop(rid, None)
            if issued is None:
                continue
            wait = issued[1]
            rpc = verified[1] if verified is not None else None
            commit = None if verified is None else now - verified[0] - verified[1]
            self._last_done[rid] = now
            if wait is None or rpc is None or commit is None:
                # The first round of a row has no previous commit, so it has no wait
                # segment; it is counted but never attributed.
                continue
            self._parts.append((wait, rpc, commit))
            self._closed += 1
            if len(self._parts) > MAX_GROUPS:
                del self._parts[0]
        if self._closed and self._closed % self.report_every == 0:
            self.report()

    # --- reporting -------------------------------------------------------
    def summary(self):
        """Segment statistics for all closed rounds, split by slow and fast."""
        if not self._parts:
            return None
        slow = [p for p in self._parts if sum(p) > self.slow_ns]
        fast = [p for p in self._parts if sum(p) <= self.slow_ns]
        def group(rows):
            if not rows:
                return None
            return dict(rounds=len(rows),
                        wait_ns=_stats([r[0] for r in rows]),
                        rpc_ns=_stats([r[1] for r in rows]),
                        commit_ns=_stats([r[2] for r in rows]),
                        total_ns=_stats([sum(r) for r in rows]))
        return dict(rounds=len(self._parts), slow_threshold_ns=self.slow_ns,
                    slow=group(slow), fast=group(fast),
                    slow_share=len(slow) / len(self._parts))

    def report(self, final=False):
        """Emit the aggregate. Called periodically, and once more at shutdown."""
        if not self.enabled or self.emit is None:
            return None
        summary = self.summary()
        if summary is None:
            return None
        self.emit('growth_round_timing', final=bool(final), **summary)
        return summary
