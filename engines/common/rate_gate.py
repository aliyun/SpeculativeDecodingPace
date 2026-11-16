import math
import os

CACHE_SOURCES = ('exact_prefix', 'suffix', 'similar_context')
SIDES = ('cache', 'alt')


def seed_check_from_env(default=8):
    """How many seed-origin drafts to observe before judging the seed.

    A seed whose drafts never earn an accepted token is not merely useless: it
    keeps the workload's reuse verdict and the cache's capacity, so it is dropped
    instead of priced. Zero disables the check.
    """
    raw = os.environ.get('TRIAGE_UNIFIED_SEED_CHECK')
    if raw is None:
        return default
    if not raw.isdigit() or not 0 <= int(raw) <= 256:
        raise ValueError('TRIAGE_UNIFIED_SEED_CHECK must be 0..256')
    return int(raw)


def rate_gate_from_env(environ=None):
    """Disabled unless a launch asks for it; every bound is finite."""
    environ = os.environ if environ is None else environ
    raw = environ.get('TRIAGE_UNIFIED_RATE_GATE')
    if raw is None or raw == '0':
        return RateGate(enabled=False)
    if raw != '1':
        raise ValueError('TRIAGE_UNIFIED_RATE_GATE must be 0 or 1')
    minimum = environ.get('TRIAGE_UNIFIED_RATE_GATE_MIN', '4')
    optimism = environ.get('TRIAGE_UNIFIED_RATE_GATE_OPTIMISM', '1.0')
    if not minimum.isdigit() or not 1 <= int(minimum) <= 256:
        raise ValueError('TRIAGE_UNIFIED_RATE_GATE_MIN must be 1..256')
    value = float(optimism)
    if not math.isfinite(value) or not 0 <= value <= 8:
        raise ValueError('TRIAGE_UNIFIED_RATE_GATE_OPTIMISM must be 0..8')
    probation = environ.get('TRIAGE_UNIFIED_GATE_PROBATION', '4')
    if not probation.isdigit() or not 0 <= int(probation) <= 256:
        raise ValueError('TRIAGE_UNIFIED_GATE_PROBATION must be 0..256')
    return RateGate(enabled=True, min_samples=int(minimum), optimism=value,
                    probation=int(probation))


class RateGate:
    """Committed-token rate per side, compared per scope.

    ``note`` records committed tokens and elapsed time once per round.
    ``allow`` decides whether a cache scope may supply the next proposal.
    """

    def __init__(self, enabled=False, min_samples=4, optimism=1.0, decay=0.9,
                 probe_every=16, probation=4):
        if type(enabled) is not bool:
            raise ValueError('rate gate switch')
        if type(min_samples) is not int or not 1 <= min_samples <= 256:
            raise ValueError('rate gate minimum samples')
        if not math.isfinite(optimism) or not 0 <= optimism <= 8:
            raise ValueError('rate gate optimism')
        if not 0 < decay <= 1:
            raise ValueError('rate gate decay')
        if type(probe_every) is not int or not 1 <= probe_every <= 4096:
            raise ValueError('rate gate probe interval')
        if type(probation) is not int or not 0 <= probation <= 256:
            raise ValueError('rate gate probation')
        self.enabled = enabled
        self.min_samples = min_samples
        self.optimism = optimism
        self.decay = decay
        self.probe_every = probe_every
        self.probation = probation
        # side -> scope or None -> [tokens, elapsed_ns, samples, earned]
        self.state = {}
        self.vetoes = self.allows = self.explores = 0
        # Why a round was not charged: a gate that never records is inert, and
        # that has to be visible in /status rather than inferred from silence.
        self.counters = {}
        self.probes = {}
        # What a scope would have earned without serving: see note_shadow.
        self.shadow = {}

    # -- measurement -----------------------------------------------------
    def note(self, side, scope, tokens, elapsed_ns):
        if not self.enabled:
            return
        if side not in SIDES:
            raise ValueError('rate gate side')
        if type(tokens) is not int or tokens < 0:
            raise ValueError('rate gate tokens')
        if type(elapsed_ns) is not int or elapsed_ns <= 0:
            return
        row = self.state.get((side, scope))
        if row is None:
            row = self.state[(side, scope)] = [0.0, 0.0, 0, 0.0]
        row[0] = self.decay*row[0] + tokens
        row[1] = self.decay*row[1] + elapsed_ns
        row[2] = min(row[2] + 1, 1 << 20)
        # Progress beyond the one correction token every round gets anyway.
        # A scope that only ever commits that single token has earned nothing.
        row[3] = self.decay*row[3] + max(0, tokens - 1)

    def count(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1

    def note_shadow(self, scope, tokens, elapsed_ns):
        """Score a proposal that was built but served by the alternative.

        The draft is compared against what the target actually committed, so this
        costs one lookup and one comparison and needs no round of its own. The
        cost side is the alternative's measured round, which is pessimistic for a
        local round (W=0), so a scope has to win on progress alone.
        """
        if not self.enabled:
            return
        if type(tokens) is not int or tokens < 0 or type(elapsed_ns) is not int or elapsed_ns <= 0:
            return
        row = self.shadow.get(scope)
        if row is None:
            row = self.shadow[scope] = [0.0, 0.0, 0]
        row[0] = self.decay*row[0] + tokens
        row[1] = self.decay*row[1] + elapsed_ns
        row[2] = min(row[2] + 1, 1 << 20)

    def _shadow_rate(self, scope):
        row = self.shadow.get(scope)
        if row is None or row[2] == 0 or row[1] <= 0:
            return None, 0
        return row[0]/row[1], row[2]

    def _rate(self, side, scope):
        row = self.state.get((side, scope))
        if row is None or row[2] == 0 or row[1] <= 0:
            return None, 0
        return row[0]/row[1], row[2]

    # -- decision --------------------------------------------------------
    def allow(self, scope):
        """True when this scope may serve the round.

        Deny scopes that have supplied no accepted draft tokens. Probe every
        ``probe_every`` lookups so a scope can recover when its proposals improve.
        """
        if not self.enabled:
            return True
        row = self.state.get(('cache', scope))
        if row is not None and row[2] >= 1 and row[3] <= 0:
            probe = self.probes.get(scope, 0) + 1
            self.probes[scope] = probe
            if probe % self.probe_every:
                self.vetoes += 1
                return False
            self.explores += 1
            return True
        rate, samples = self._rate('cache', scope)
        alternative, alt_samples = self._rate('alt', None)
        if (rate is None or samples == 0) and self.probation:
            # During probation, compare the shadow rate before serving the scope.
            shadow_rate, shadow_samples = self._shadow_rate(scope)
            if shadow_samples:
                if shadow_samples < self.probation or alternative is None or alt_samples < 1:
                    self.counters['probation'] = self.counters.get('probation', 0) + 1
                    self.vetoes += 1
                    return False
                if shadow_rate is not None and shadow_rate*(1.0 + self.optimism/math.sqrt(shadow_samples)) >= alternative:
                    self.allows += 1
                    return True
                self.counters['probation_lost'] = self.counters.get('probation_lost', 0) + 1
                self.vetoes += 1
                return False
        if rate is None or samples < self.min_samples or alternative is None or alt_samples < self.min_samples:
            # Cold start is exploration, not evidence: try the scope, but never
            # call a scope allowed when it already has losing evidence.
            self.explores += 1
            return True
        # Discount the local side while its sample is small: the same optimism
        # the selector uses, applied to progress rather than to latency.
        effective = rate*(1.0 + self.optimism/math.sqrt(samples))
        if effective >= alternative:
            self.allows += 1
            return True
        self.vetoes += 1
        return False

    def snapshot(self):
        rows = {}
        for (side, scope), (tokens, elapsed, samples, earned) in self.state.items():
            key = side if scope is None else '%s:%s' % (side, ':'.join(scope))
            rows[key] = dict(tokens_per_ms=round(1e6*tokens/elapsed, 3) if elapsed > 0 else None,
                             samples=samples, earned=round(earned, 2))
        shadows = {}
        for scope, (tokens, elapsed, samples) in self.shadow.items():
            shadows[':'.join(scope)] = dict(
                tokens_per_ms=round(1e6*tokens/elapsed, 3) if elapsed > 0 else None, samples=samples)
        return dict(enabled=self.enabled, min_samples=self.min_samples,
                    optimism=self.optimism, vetoes=self.vetoes, allows=self.allows,
                    explores=self.explores, counters=dict(self.counters), scopes=rows,
                    probation=self.probation, shadow=shadows)
