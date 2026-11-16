from collections import deque
import math
import os

CAP_MS = 400
MIN_SAMPLES = 24
FACTOR = 2.0
CENSORED_SHARE = 0.5


def _env_int(name, default, low, high):
    raw = os.environ.get(name, '')
    if raw == '':
        return default
    if not raw.isdigit() or not low <= int(raw) <= high:
        raise ValueError('%s must be %d..%d' % (name, low, high))
    return int(raw)


def _env_factor(name, default):
    raw = os.environ.get(name, '')
    if raw == '':
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError('%s must be a number' % name)
    if not 1.0 <= value <= 10.0:
        raise ValueError('%s must be 1..10' % name)
    return value


def probe_enabled():
    value = os.environ.get('EC_WAIT_PROBE', '0')
    if value not in ('0', '1'):
        raise ValueError('EC_WAIT_PROBE must be 0 or 1')
    return value == '1'


class WaitProbe:
    """Sample draft arrivals once, freeze one window, then stay immutable."""

    def __init__(self, *, floor_ms, cap_ms=None, min_samples=None, factor=None):
        self.floor_ms = int(floor_ms)
        self.cap_ms = int(cap_ms if cap_ms is not None else
                          _env_int('EC_WAIT_PROBE_CAP_MS', CAP_MS, self.floor_ms, 5000))
        self.min_samples = int(min_samples if min_samples is not None else
                               _env_int('EC_WAIT_PROBE_SAMPLES', MIN_SAMPLES, 1, 4096))
        self.factor = float(factor if factor is not None else _env_factor('EC_WAIT_PROBE_FACTOR', FACTOR))
        if self.cap_ms < self.floor_ms or self.min_samples < 1 or not 1.0 <= self.factor <= 10.0:
            raise ValueError('invalid wait probe bounds')
        self.samples = deque(maxlen=self.min_samples)
        self.censored = 0
        self.value_ms = None
        self.reason = 'sampling'

    @property
    def frozen(self):
        return self.value_ms is not None

    def deadline_ns(self):
        """Budget for a row that has no consistent draft yet."""
        if self.frozen:
            return self.value_ms * 1_000_000
        return self.cap_ms * 1_000_000

    def observe(self, wait_ns):
        """Record one realised arrival; returns the frozen window or None."""
        if self.frozen or type(wait_ns) is not int or wait_ns < 0:
            return None
        milli = wait_ns / 1_000_000
        self.samples.append(milli)
        # A sample that reaches the sampling cap only proves "not within it".
        if milli >= self.cap_ms - 1:
            self.censored += 1
        if len(self.samples) < self.min_samples:
            return None
        return self.freeze()

    def freeze(self):
        if self.frozen:
            return self.value_ms
        ordered = sorted(self.samples)
        if not ordered:
            return None
        q90 = ordered[math.ceil(.9 * len(ordered)) - 1]
        if q90 >= self.cap_ms - 1 or self.censored / len(ordered) > CENSORED_SHARE:
            self.value_ms, self.reason = self.floor_ms, 'censored'
        else:
            wanted = math.ceil(self.factor * q90)
            self.value_ms, self.reason = min(max(wanted, self.floor_ms), self.cap_ms), 'measured'
        return self.value_ms

    def snapshot(self):
        ordered = sorted(self.samples)
        result = dict(wait_probe_samples=len(ordered), wait_probe_censored=self.censored,
                      wait_probe_reason=self.reason, wait_probe_value_ms=self.value_ms or 0,
                      wait_probe_floor_ms=self.floor_ms, wait_probe_cap_ms=self.cap_ms)
        if ordered:
            result.update(wait_probe_p50_ms=round(ordered[len(ordered) // 2], 3),
                          wait_probe_q90_ms=round(ordered[math.ceil(.9 * len(ordered)) - 1], 3),
                          wait_probe_max_ms=round(ordered[-1], 3))
        return result
