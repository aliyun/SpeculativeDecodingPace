from collections import OrderedDict, deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Offer:
    rid: str
    prefix: tuple
    k: int
    source: str
    reward: float | None
    cache: object = None


@dataclass(frozen=True)
class Batch:
    mode: str
    offers: tuple
    reason: str
    predicted_ms: float | None = None
    progress: float | None = None

    @property
    def ids(self):
        return tuple(o.rid for o in self.offers)

    @property
    def k(self):
        return self.offers[0].k if self.mode == 'spec' else 0


class BatchRate:
    MAX_BATCH = 8
    MAX_CANDIDATES = 40
    MAX_SHAPES = 256
    MIN_SAMPLES = 2

    def __init__(self):
        self.geometry = None
        self.samples = OrderedDict()
        self.choices = 0
        self.last_candidates = 0

    @staticmethod
    def geometry_key(shape):
        if shape is None:
            return None
        if (type(shape) is not dict or shape.get('engine') != 'hf'
                or shape.get('layout') not in ('exact_rows', 'fixed_bank', 'resident_rows')
                or shape.get('execution') not in ('graph', 'compiled', 'eager')
                or type(shape.get('physical_rows')) is not int
                or not 0 <= shape['physical_rows'] <= 16
                or type(shape.get('context')) is not int or shape['context'] < 1):
            raise ValueError('STREAM_BATCH_NATIVE_GEOMETRY')
        return (shape['engine'], shape['layout'], shape['execution'],
                shape['physical_rows'], shape['context'])

    def observe_geometry(self, shape):
        key = self.geometry_key(shape)
        if key is not None:
            self.geometry = key

    def key(self, b, k, length, *, geometry=None):
        g = self.geometry if geometry is None else self.geometry_key(geometry)
        if g is None:
            return None
        engine, layout, execution, physical, context = g
        physical = b if layout == 'exact_rows' else physical
        return engine, layout, execution, physical, context, b, k, (length+127)//128

    def observe(self, b, k, length, elapsed_ms, shape):
        if (type(b) is not int or not 1 <= b <= 16 or type(k) is not int
                or k not in (0, 3, 4, 5, 6, 7) or type(length) is not int or length < 1
                or not math.isfinite(elapsed_ms) or elapsed_ms <= 0):
            raise ValueError('STREAM_BATCH_NATIVE_SAMPLE')
        if shape is None:
            return  # Old/fixture feedback cannot invent physical-bank evidence.
        key = self.key(b, k, length, geometry=shape)
        if key not in self.samples:
            self.samples[key] = deque(maxlen=16)
        self.samples[key].append(elapsed_ms)
        self.samples.move_to_end(key)
        while len(self.samples) > self.MAX_SHAPES:
            self.samples.popitem(last=False)

    def cost(self, b, k, length):
        values = self.samples.get(self.key(b, k, length), ())
        if len(values) < self.MIN_SAMPLES:
            return None
        return sorted(values)[len(values)//2]

    def candidates(self, rows, ks, anchor):
        """At most five K x eight FIFO-prefix batches; all contain the anchor."""
        result = []
        for k in ks:
            available = [by_k[k] for by_k in rows if k in by_k]
            if not available or anchor not in {o.rid for o in available}:
                continue
            # Callers put the oldest serviceable row first, not the highest reward.
            if available[0].rid != anchor:
                raise ValueError('STREAM_BATCH_ANCHOR_ORDER')
            for b in range(1, min(self.MAX_BATCH, len(available))+1):
                result.append(tuple(available[:b]))
        if len(result) > self.MAX_CANDIDATES:
            raise ValueError('STREAM_BATCH_ENUMERATION_BOUND')
        self.last_candidates = len(result)
        return result

    def choose(self, rows, ks, anchor, *, length, ar_rows=(), allow_ar=False,
               elapsed_ms=0., probe_ms=0., forced=None):
        batches = self.candidates(rows, ks, anchor)
        if forced is not None:
            batches = [o for o in batches if o[0].k == forced]
        if not batches:
            return None
        bootstrap = min(batches, key=lambda o: (o[0].k, -len(o)))
        if forced is not None:
            return Batch('spec', max(batches, key=len), 'stream_batch_forced')
        measured, pending = [], []
        for offers in batches:
            cost = self.cost(len(offers), offers[0].k, length)
            if cost is not None and all(o.reward is not None for o in offers):
                progress = sum(o.reward for o in offers)
                measured.append((progress/cost, len(offers), -offers[0].k,
                                 Batch('spec', offers, 'stream_batch_rate', cost, progress)))
            else:
                pending.append(offers)
        base = self.cost(len(bootstrap), bootstrap[0].k, length)
        if base is None:
            # Identify the actual concurrent baseline before old C1 samples can
            # monopolize dispatch and prevent ever observing this batch size.
            return Batch('spec', bootstrap, 'stream_batch_bootstrap')
        if pending and base is not None and (self.choices+1) % 8 == 0:
            # Real same-B probes only; an unknown cost is never a mature winner.
            possible = [o for o in pending if len(o) == len(bootstrap)
                        and o[0].k > bootstrap[0].k]
            if possible:
                offers = max(possible, key=lambda o: o[0].k)
                estimate = self.cost(len(offers), offers[0].k, length) or 1.1*base
                if 2*base+.10*elapsed_ms-probe_ms >= estimate:
                    return Batch('spec', offers, 'stream_batch_probe')
        if allow_ar:
            for b in range(1, min(self.MAX_BATCH, len(ar_rows))+1):
                offers = tuple(ar_rows[:b])
                if offers[0].rid != anchor:
                    continue
                cost = self.cost(b, 0, length)
                if cost is not None:
                    measured.append((b/cost, b, 0,
                                     Batch('ar', offers, 'stream_batch_ar_rate', cost, float(b))))
        if measured:
            return max(measured, key=lambda item: item[:3])[3]
        return Batch('spec', bootstrap, 'stream_batch_bootstrap')

    def snapshot(self):
        return dict(objective='FIFO_CONSTRAINED_BATCH_RATE_NOT_JOINT_BELLMAN',
                    choices=self.choices, candidates=self.last_candidates,
                    max_candidates=self.MAX_CANDIDATES, geometry=self.geometry,
                    measured_shapes=len(self.samples),
                    costs=[dict(key=list(k), n=len(v), median_ms=sorted(v)[len(v)//2])
                           for k, v in self.samples.items()])
