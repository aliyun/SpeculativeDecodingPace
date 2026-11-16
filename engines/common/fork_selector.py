from collections import deque
import json
import math


WILSON_Z = 1.96


def wilson_lower(successes, trials, z=WILSON_Z):
    """One-sided Wilson lower bound; 0 when there is no evidence."""
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt(max(0.0, p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)))
    return max(0.0, (centre - spread) / denom)


def quantile(values, fraction):
    """Upper quantile (ceil), so cost bounds stay conservative."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * fraction))]


class ForkAction:
    __slots__ = ('at', 'length')

    def __init__(self, at, length):
        self.at, self.length = int(at), int(length)

    def key(self):
        return (self.at, self.length)

    def __eq__(self, other):
        return isinstance(other, ForkAction) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return 'fork(j=%d,b=%d)' % self.key()


class ForkStats:
    __slots__ = ('trials', 'rescues', 'g', 't', 'latency', 'ewma_g', 'ewma_t', 'last')

    def __init__(self, window=32):
        self.trials = 0
        self.rescues = 0
        self.g = deque(maxlen=window)
        self.t = deque(maxlen=window)
        self.latency = deque(maxlen=window)
        self.ewma_g = None
        self.ewma_t = None
        self.last = None

    def observe(self, delivered, round_ms, rescued, alpha=0.2):
        self.trials += 1
        self.rescues += 1 if rescued else 0
        self.g.append(float(delivered))
        self.t.append(float(round_ms))
        if round_ms > 0:
            self.latency.append(float(round_ms))
        self.ewma_g = delivered if self.ewma_g is None else (1 - alpha) * self.ewma_g + alpha * delivered
        self.ewma_t = round_ms if self.ewma_t is None else (1 - alpha) * self.ewma_t + alpha * round_ms
        self.last = dict(delivered=delivered, round_ms=round_ms, rescued=bool(rescued))

    def snapshot(self):
        return dict(trials=self.trials, rescues=self.rescues,
                    rescue_rate=(self.rescues / self.trials) if self.trials else None,
                    rescue_lower=wilson_lower(self.rescues, self.trials),
                    ewma_g=self.ewma_g, ewma_t=self.ewma_t,
                    cost_p90=quantile(self.latency, 0.9))


class OnlineForkSelector:
    """Bounded, auditable online chooser of (j, b) or no fork."""

    def __init__(self, *, window=32, n_min=12, explore_fraction=0.10,
                 max_consecutive=4, max_length=3, max_fork_at_slack=1,
                 baseline_floor=0.99, prior=None,
                 shapes=((1, 1), (1, 2), (3, 1), (3, 2)),
                 d_ms=1.3, mask_ms=0.5, shape_miss_limit=1):
        if not 0.0 < explore_fraction <= 1.0:
            raise ValueError('FORK_EXPLORE_FRACTION')
        if type(n_min) is not int or n_min < 1 or type(max_consecutive) is not int or max_consecutive < 1:
            raise ValueError('FORK_BOUNDS')
        self.window, self.n_min = window, n_min
        self.explore_fraction, self.max_consecutive = explore_fraction, max_consecutive
        self.max_length = max_length
        self.max_fork_at_slack = max_fork_at_slack
        self.baseline_floor = baseline_floor
        # Only pre-captured tree shapes are legal: one graph per (K, j, b, batch)
        # and a post-seal miss is GROWTH_FUSED_UNPREPARED. Root top-2 is the free
        # special case and is not in this list (it keeps the same graph shape).
        self.shapes = tuple(sorted({(int(j), int(b)) for j, b in shapes}))
        if any(j < 1 or b < 1 for j, b in self.shapes):
            raise ValueError('FORK_SHAPES')
        if d_ms < 0 or mask_ms < 0:
            raise ValueError('FORK_SHAPE_COSTS')
        self.d_ms, self.mask_ms = float(d_ms), float(mask_ms)
        self.shape_miss_limit = shape_miss_limit
        self.shape_misses = 0
        self.paused_reason = None
        # Weak prior over positions from the first-rejection histogram; used only
        # to order exploration, never as evidence about r_j.
        self.prior = dict(prior or {})
        self.actions = {}
        self.baseline = {}
        self.rounds = 0
        self.forked = 0
        self.consecutive = 0
        self.decisions = []

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def bucket(batch, prefix_length, traffic='default'):
        return (int(batch), int(prefix_length) // 128, str(traffic))

    def _action_stats(self, bucket, action):
        key = (bucket, action.key())
        stats = self.actions.get(key)
        if stats is None:
            stats = self.actions[key] = ForkStats(self.window)
        return stats

    def _baseline(self, bucket):
        return self.baseline.get(bucket)

    # ---- observation -----------------------------------------------------
    def observe(self, bucket, action, delivered, round_ms, rescued=False):
        self.rounds += 1
        if action is None:
            stats = self.baseline.get(bucket)
            if stats is None:
                stats = self.baseline[bucket] = ForkStats(self.window)
            stats.observe(delivered, round_ms, False)
            self.consecutive = 0
        else:
            self._action_stats(bucket, action).observe(delivered, round_ms, rescued)
            self.forked += 1

    def pause(self, reason='paused'):
        """Explicit stop signal (drift or an unprepared shape)."""
        self.consecutive = self.max_consecutive
        self.paused_reason = reason

    def note_shape_miss(self):
        """A requested shape was not captured; fail closed after the limit."""
        self.shape_misses += 1
        if self.shape_misses >= self.shape_miss_limit:
            self.pause('shape_miss_limit')

    # ---- decision --------------------------------------------------------
    def budget_left(self):
        if self.rounds == 0:
            return True
        return (self.forked / self.rounds) < self.explore_fraction

    def candidates(self, k):
        """Action space = pre-captured shapes that also fit this K.

        j <= k-1 keeps at least one position after the branch; b <= max_length
        bounds the extra nodes. Anything not pre-captured would raise
        GROWTH_FUSED_UNPREPARED at the edge, so it is not a candidate.
        """
        return [ForkAction(j, b) for j, b in self.shapes
                if 1 <= j <= k - self.max_fork_at_slack and 1 <= b <= self.max_length]

    def decide(self, bucket, k):
        """Return (action|None, reason, inputs) and log it."""
        base = self._baseline(bucket)
        inputs = dict(bucket=list(bucket), k=k, rounds=self.rounds, forked=self.forked,
                      explore_budget=self.budget_left(), consecutive=self.consecutive,
                      n_min=self.n_min, action=None, lcb=None, baseline=None,
                      shape_misses=self.shape_misses, paused=self.paused_reason)
        if self.paused_reason is not None:
            return self._log(bucket, None, 'paused_%s' % self.paused_reason, inputs)
        if base is None or base.trials < self.n_min or base.ewma_t is None:
            return self._log(bucket, None, 'cold_start_no_fork', inputs)
        base_rate = base.ewma_g / base.ewma_t if base.ewma_t else None
        if base_rate is None:
            return self._log(bucket, None, 'cold_start_no_fork', inputs)
        best = None
        for action in self.candidates(k):
            stats = self._action_stats(bucket, action)
            if stats.trials < self.n_min or not stats.ewma_t:
                continue
            lower = wilson_lower(stats.rescues, stats.trials)
            # Rescue adds at most (b + 1) tokens; cost upper bound uses p90.
            gain = lower * (action.length + 1)
            measured = (quantile(stats.latency, 0.9) or stats.ewma_t) - base.ewma_t
            # The mechanism floor keeps a lucky cheap window from claiming the
            # branch is free: it costs b edge steps plus b tree rows.
            mechanistic = action.length * (self.d_ms + self.mask_ms)
            delta_cost = max(0.0, measured, mechanistic)
            lcb = (base.ewma_g + gain) / (base.ewma_t + delta_cost)
            if lcb > base_rate * self.baseline_floor and (best is None or lcb > best[0]):
                best = (lcb, action)
        if best is None:
            # Exploration is what the caps constrain; a justified LCB winner is
            # utilisation, not exploration, and is never blocked by them.
            if self.consecutive >= self.max_consecutive:
                return self._log(bucket, None, 'explore_cap_consecutive', inputs)
            if not self.budget_left():
                return self._log(bucket, None, 'explore_budget_exhausted', inputs)
            unmeasured = [a for a in self.candidates(k)
                          if self._action_stats(bucket, a).trials < self.n_min]
            if unmeasured:
                pick = max(unmeasured, key=lambda a: (self.prior.get(a.at, 0.0), -a.at, -a.length))
                inputs['prior'] = self.prior.get(pick.at, 0.0)
                return self._log(bucket, pick, 'targeted_exploration', inputs)
            return self._log(bucket, None, 'insufficient_evidence', inputs)
        inputs.update(lcb=best[0], baseline=base_rate)
        return self._log(bucket, best[1], 'lcb_beats_baseline', inputs)

    def _log(self, bucket, action, reason, inputs):
        # Consecutive count tracks decisions, not observations: a fork observed
        # out of band (tests, replays) must not spend the exploration cap.
        self.consecutive = self.consecutive + 1 if action is not None else 0
        record = dict(bucket=list(bucket), action=(action.key() if action else None),
                      reason=reason, inputs=inputs)
        self.decisions.append(record)
        if len(self.decisions) > 256:
            del self.decisions[:128]
        return action, reason, inputs

    def snapshot(self):
        return dict(rounds=self.rounds, forked=self.forked,
                    fork_fraction=(self.forked / self.rounds) if self.rounds else 0.0,
                    consecutive=self.consecutive,
                    explore_fraction=self.explore_fraction, n_min=self.n_min,
                    actions={str(k): v.snapshot() for k, v in list(self.actions.items())[:16]},
                    baseline={str(k): v.snapshot() for k, v in list(self.baseline.items())[:8]},
                    last_decisions=self.decisions[-4:])
