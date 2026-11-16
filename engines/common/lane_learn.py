import math

RESCUE_TOKENS = 1.0  # a rescued round contributes the alternate root's token

PRICE_STATE_VERSION = 2


def wilson_lower(hits, n, z=1.0):
    """Lower bound of a binomial rate: the pessimism the learner acts on."""
    if n <= 0:
        return 0.0
    p = hits / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return max(0.0, (centre - spread) / denom)


class PricePosterior:
    """Online Bayesian linear regression for `dCost = a + b*g`, per width bucket.

    Each hedged round contributes its current tokens-per-round estimate g and
    extra elapsed time relative to the stream baseline. The posterior updates
    the configurable prior in closed form. The prediction is clamped to
    [0, cap], and every observation must use the same definition of g.
    """

    def __init__(self, a0=-14.47, b0=5.50, var_a=400.0, var_b=25.0, noise=25.0, cap=200.0):
        self.theta = [float(a0), float(b0)]
        self.cov = [[float(var_a), 0.0], [0.0, float(var_b)]]
        self.noise = max(1e-6, float(noise))
        self.cap = float(cap)
        self.n = 0

    def update(self, g, y):
        """One observation: at g tokens/round the hedged round cost y extra ms."""
        phi = (1.0, float(g))
        c = self.cov
        s = (phi[0] * (c[0][0] * phi[0] + c[0][1] * phi[1])
             + phi[1] * (c[1][0] * phi[0] + c[1][1] * phi[1])) + self.noise
        gain = ((c[0][0] * phi[0] + c[0][1] * phi[1]) / s,
                (c[1][0] * phi[0] + c[1][1] * phi[1]) / s)
        err = float(y) - (self.theta[0] + self.theta[1] * phi[1])
        self.theta[0] += gain[0] * err
        self.theta[1] += gain[1] * err
        new = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            gi = gain[i]
            for j in range(2):
                new[i][j] = c[i][j] - gi * (phi[0] * c[0][j] + phi[1] * c[1][j])
        self.cov = new
        self.n += 1
        return self

    def predict(self, g):
        """(mean, sd) of the extra cost at this g, clamped into [0, cap]."""
        g = float(g)
        mean = self.theta[0] + self.theta[1] * g
        phi = (1.0, g)
        c = self.cov
        var = (phi[0] * (c[0][0] * phi[0] + c[0][1] * phi[1])
               + phi[1] * (c[1][0] * phi[0] + c[1][1] * phi[1])) + self.noise
        mean = min(self.cap, max(0.0, mean))
        return mean, math.sqrt(max(0.0, var))

    def state(self):
        return dict(a=self.theta[0], b=self.theta[1], n=self.n,
                    var_a=self.cov[0][0], var_b=self.cov[1][1])


class LaneLearner:
    """Per-cell cadence, rescue rate and the transport decision they imply."""

    def __init__(self, *, min_rounds=2, explore_rounds=1, rescue_tokens=RESCUE_TOKENS,
                 z=1.0, shrink=2.0, request_margin=1.0, request_stride=0,
                 min_request_samples=1, price_mode=False, price_a=-14.47, price_b=5.50,
                 price_p=None, price_margin=1.0, price_cap_ms=200.0,
                 price_prior_var_a=400.0, price_prior_var_b=25.0, price_noise=25.0,
                 price_every=3, price_floor=8, price_min_labels=6, price_max_sd=6.0,
                 lane_rule=False, lane_g=3.8, lane_p=0.13,
                 min_round_tokens=2, min_stream_depth=3, degenerate_ratio=3.0,
                 noop=False):
        self.min_rounds = max(1, int(min_rounds))
        self.explore_rounds = max(0, int(explore_rounds))
        self.rescue_tokens = float(rescue_tokens)
        self.z = float(z)
        self.shrink = max(0.0, float(shrink))
        self.request_margin = max(0.0, float(request_margin))
        self.request_stride = max(0, int(request_stride))
        self.min_request_samples = max(1, int(min_request_samples))
        self.price_mode = bool(price_mode)
        self.price_a = float(price_a)
        self.price_b = float(price_b)
        self.price_p = None if price_p is None else float(price_p)
        self.price_margin = max(1.0, float(price_margin))
        self.price_cap_ms = max(1.0, float(price_cap_ms))
        # Estimate dCost with Bayesian regression and rescue probability from
        # both label sources. Bounded sync exploration supplies observations
        # at admission boundaries, with a decaying sampling frequency.
        self.price_prior_a = float(price_a)
        self.price_prior_b = float(price_b)
        self.price_prior_var_a = float(price_prior_var_a)
        self.price_prior_var_b = float(price_prior_var_b)
        self.price_noise = float(price_noise)
        self.price_every = max(1, int(price_every))
        self.price_floor = max(self.price_every, int(price_floor))
        self.price_min_labels = max(1, int(price_min_labels))
        self.price_max_sd = float(price_max_sd)
        self.lane_rule = bool(lane_rule)
        self.lane_g = float(lane_g)
        self.lane_p = float(lane_p)
        self.price = {}       # k -> PricePosterior
        self.labels = {}      # k -> dict(trials, rescued, cf_trials, cf_rescued)
        self.admits = {}      # k -> admissions seen (exploration counter)
        self.min_round_tokens = max(1, int(min_round_tokens))
        self.min_stream_depth = max(1, int(min_stream_depth))
        self.degenerate_ratio = max(0.0, float(degenerate_ratio))
        self.noop = bool(noop)
        self.cells = {}

    # ---- cells -----------------------------------------------------------
    @staticmethod
    def cell(k, gap=None, delivery=None):
        """Observable key: the shape being run and its gap bucket.

        The request's changing pace is stored inside the cell rather than in
        its key, so observations accumulate across successive rounds.
        """
        gap_bucket = None if gap is None else int(round(float(gap)))
        return (int(k), gap_bucket)

    def _cell(self, cell):
        return self.cells.setdefault(cell, dict(n=0, tokens=0.0, ms=0.0,
                                                s_n=0, s_tokens=0.0, s_ms=0.0,
                                                trials=0, rescued=0, explorations=0))

    # ---- observation -----------------------------------------------------
    def record_stream(self, cell, tokens, ms):
        if self.noop:
            return
        """One stream round: the baseline this policy is supposed to beat.

        The main cadence estimate excludes rounds below min_round_tokens.
        The all-round accumulator retains them for comparisons that require
        matching filters across lanes.
        """
        row = self._cell(cell)
        if tokens < self.min_round_tokens or ms <= 0:
            row['degenerate'] = row.get('degenerate', 0) + 1
            if ms > 0:
                row['an'] = row.get('an', 0) + 1
                row['a_tokens'] = row.get('a_tokens', 0.0) + float(tokens)
                row['a_ms'] = row.get('a_ms', 0.0) + float(ms)
            return
        row['n'] += 1
        row['tokens'] += float(tokens)
        row['ms'] += float(ms)
        row['an'] = row.get('an', 0) + 1
        row['a_tokens'] = row.get('a_tokens', 0.0) + float(tokens)
        row['a_ms'] = row.get('a_ms', 0.0) + float(ms)

    def record_sync(self, cell, tokens, ms):
        if self.noop:
            return
        """One sync round: its own cost, measured rather than assumed."""
        row = self._cell(cell)
        if tokens < self.min_round_tokens or ms <= 0:
            row['degenerate'] = row.get('degenerate', 0) + 1
            return
        row['s_n'] += 1
        row['s_tokens'] += float(tokens)
        row['s_ms'] += float(ms)

    def record_rescue(self, cell, rescued):
        if self.noop:
            return
        row = self._cell(cell)
        row['trials'] += 1
        row['rescued'] += 1 if rescued else 0
        self.record_accept(cell[0] if isinstance(cell, tuple) else None, rescued,
                           counterfactual=False)

    # ---- online price model and acceptance labels -------------------------
    def price_model(self, k):
        key = int(k) if k is not None else min(self.cells and [c[0] for c in self.cells] or [3])
        model = self.price.get(key)
        if model is None:
            model = PricePosterior(self.price_prior_a, self.price_prior_b,
                                   self.price_prior_var_a, self.price_prior_var_b,
                                   self.price_noise, self.price_cap_ms)
            self.price[key] = model
        return model

    def record_price(self, k, g, d_ms):
        """One hedged round that actually ran: at g tok/round it cost d_ms extra."""
        if self.noop or k is None:
            return
        self.price_model(k).update(g, d_ms)

    def record_accept(self, k, rescued, counterfactual=False):
        """Acceptance label for P, from a real hedge or from a stream-side counterfactual."""
        if self.noop or k is None:
            return
        row = self.labels.setdefault(int(k), dict(trials=0, rescued=0,
                                                  cf_trials=0, cf_rescued=0))
        if counterfactual:
            row['cf_trials'] += 1
            row['cf_rescued'] += 1 if rescued else 0
        else:
            row['trials'] += 1
            row['rescued'] += 1 if rescued else 0

    def accept_lower(self, k):
        """Wilson lower bound of P over both label sources."""
        row = self.labels.get(int(k))
        if not row:
            return 0.0, 0
        trials = row['trials'] + row['cf_trials']
        rescued = row['rescued'] + row['cf_rescued']
        return wilson_lower(rescued, trials, self.z), trials

    def wants_explore(self, k):
        """Forced hedge every `price_every`-th admission, decaying to a floor.

        Count admissions rather than scheduler ticks. Both cost and rescue
        probability need continued observations, so the sampling rate decays
        from 1/price_every to 1/price_floor as evidence accumulates.
        """
        if not self.price_mode:
            return False
        key = int(k)
        self.admits[key] = self.admits.get(key, 0) + 1
        n = self.admits[key]
        _p, trials = self.accept_lower(key)
        stride = self.price_every if trials < self.price_min_labels else self.price_floor
        return n % stride == 0

    # ---- what the cells know ---------------------------------------------
    def aggregate(self, k):
        """A merged row over every gap bucket of this width.

        At admission the edge has not reported the root gap. Aggregate over
        gap buckets until the first frame identifies the request's own cell.
        """
        merged = dict(n=0, tokens=0.0, ms=0.0, s_n=0, s_tokens=0.0, s_ms=0.0,
                      trials=0, rescued=0, explorations=0,
                      req_stream=[], req_sync=[], degenerate=0, st_tps=None, sy_tps=None,
                      explore=0)
        for (cell_k, _gap), row in self.cells.items():
            if cell_k != int(k):
                continue
            for key in merged:
                if key in ('req_stream', 'req_sync', 'st_tps', 'sy_tps'):
                    continue          # merged/recomputed below
                merged[key] += row.get(key, 0 if key not in
                                       ('tokens', 'ms', 's_tokens', 's_ms') else 0.0)
        for (cell_k, _gap), row in self.cells.items():
            if cell_k != int(k):
                continue
            merged['req_stream'] = merged['req_stream'] + list(row.get('req_stream', []))
            merged['req_sync'] = merged['req_sync'] + list(row.get('req_sync', []))
        merged['st_tps'] = (sum(merged['req_stream']) / len(merged['req_stream'])) if merged['req_stream'] else None
        merged['sy_tps'] = (sum(merged['req_sync']) / len(merged['req_sync'])) if merged['req_sync'] else None
        return merged

    def ready(self, k):
        """Can this width answer yet? Price mode only needs the stream cadence."""
        merged = self.aggregate(k)
        if self.price_mode:
            return merged['n'] >= self.min_rounds and merged['ms'] > 0
        return merged['s_n'] >= 1

    def price_verdict(self, cell):
        """Price the hedged round with ONLINE estimates and confidence bounds.

        A hedged round costs dCost extra ms and buys P extra tokens (the alternate
        root is accepted at rate P). It pays iff P / dCost > rate (tok/ms), i.e.
        P > dCost * rate. Both sides are now estimated:
          * dCost: Bayesian regression posterior at the current g, taken at its UPPER
            bound (pessimistic about buying) and clamped into [0, cap];
          * P: Wilson lower bound over real-hedge labels plus stream-side
            counterfactual labels.
        While either interval is too wide the verdict is 'price_uncertain' and the
        exploration rule (wants_explore) decides; a point estimate never commits.
        """
        row = self.cells.get(cell)
        if row is None or row['n'] < self.min_rounds or row['ms'] <= 0:
            return None
        g = row['tokens'] / row['n']
        c = row['ms'] / row['n']
        if g <= 0 or c <= 0:
            return None
        k = cell[0] if isinstance(cell, tuple) else None
        model = self.price_model(k)
        mean_d, sd_d = model.predict(g)
        d_hi = min(self.price_cap_ms, mean_d + self.z * sd_d)
        st = self.request_rates(cell)[0]
        rate = (st / 1000.0) if st else (g / c)
        need = d_hi * rate
        p_lcb, trials = self.accept_lower(k)
        if self.price_p is not None:
            # explicit override (calibration runs only): still bounded by labels when
            # they exist, never allowed to exceed the measured lower bound.
            p_lcb = max(p_lcb, min(float(self.price_p), 1.0)) if not trials else p_lcb
        self.last_price = dict(g=g, rate=rate, d_mean=mean_d, d_sd=sd_d, d_hi=d_hi,
                               need=need, p_lcb=p_lcb, trials=trials, a=model.theta[0],
                               b=model.theta[1], n_model=model.n)
        if model.n < 2 or trials < self.price_min_labels or sd_d > self.price_max_sd:
            return 'stream', 'price_uncertain'
        if p_lcb > need * self.price_margin:
            return 'sync', 'price_buys'
        return 'stream', 'price_declines'

    def decide_k(self, k, sync_cost_ms):
        """Admission-time decision for a width whose gap is not known yet."""
        if self.request_stride:
            row = self.cells.setdefault((int(k), None), dict(
                n=0, tokens=0.0, ms=0.0, s_n=0, s_tokens=0.0, s_ms=0.0, trials=0,
                rescued=0, explorations=0, req_stream=[], req_sync=[]))
            row['admits'] = row.get('admits', 0) + 1
            if row['admits'] % self.request_stride == 0:
                return 'stream', 'explore_request_stream'
        merged = self.aggregate(k)
        merged.setdefault('degenerate', 0)
        saved = self.cells.get((int(k), None))
        self.cells[(int(k), None)] = merged
        try:
            return self.decide((int(k), None), sync_cost_ms, admission=True)
        finally:
            if saved is None:
                self.cells.pop((int(k), None), None)
            else:
                self.cells[(int(k), None)] = saved

    def stream_mean(self, cell):
        """(tokens per round, ms per round) the stream lane achieves here."""
        row = self.cells.get(cell)
        if row is None or row['n'] < self.min_rounds or row['ms'] <= 0:
            return None
        return row['tokens'] / row['n'], row['ms'] / row['n']

    def stream_mean_all(self, cell):
        """(tokens per round, ms per round) over EVERY round, thin ones included.

        Apply the same round filtering to both lanes. Excluding one-token
        rounds from only the stream baseline would bias the extra-cost estimate.
        """
        row = self.cells.get(cell)
        if row is None:
            return None
        n = row.get('an', 0)
        if n < self.min_rounds or row.get('a_ms', 0.0) <= 0:
            return None
        return row['a_tokens'] / n, row['a_ms'] / n

    def stream_rate(self, cell):
        """Tokens per millisecond the stream lane actually achieves here."""
        row = self.cells.get(cell)
        if row is None or row['n'] < self.min_rounds or row['ms'] <= 0:
            return None
        return (row['tokens'] / row['n']) / (row['ms'] / row['n'])

    def sync_rate(self, cell, sync_cost_ms):
        """Tokens per millisecond a sync round achieves in this cell.

        A sync round has its own delivered-token distribution, with its draft
        width as a cold prior. Shrink sparse measurements toward that prior so
        one observation cannot determine the lane.
        """
        row = self.cells.get(cell)
        if row is None or not row['n']:
            return None
        base = row['tokens'] / row['n']
        p_lo = wilson_lower(row['rescued'], row['trials'], self.z) if row['trials'] else 0.0
        prior_g = base + p_lo * self.rescue_tokens
        prior_cost = float(sync_cost_ms)
        n = row['s_n']
        if n and row['s_ms'] > 0:
            w = n / (n + self.shrink)
            delivered = w * (row['s_tokens'] / n) + (1.0 - w) * prior_g
            cost = w * (row['s_ms'] / n) + (1.0 - w) * prior_cost
        else:
            delivered, cost = prior_g, prior_cost
        if cost <= 0:
            return None
        return delivered / cost

    def record_request(self, cell, lane, tokens, seconds, *, min_tokens=64):
        if self.noop:
            return
        """One finished request's end-to-end throughput on the lane it ran.

        This includes the request's full elapsed time. Per-round cadence alone
        does not account for every cost paid across a complete request.
        """
        if tokens <= 0 or seconds <= 0:
            return
        if tokens < int(min_tokens):
            return
        row = self._cell(cell)
        key = 'req_sync' if lane == 'sync' else 'req_stream'
        series = row.setdefault(key, [])
        series.append(tokens / seconds)
        if len(series) > 32:
            del series[0]
        row['st_tps'] = sum(row.get('req_stream') or []) / len(row['req_stream']) if row.get('req_stream') else None
        row['sy_tps'] = sum(row.get('req_sync') or []) / len(row['req_sync']) if row.get('req_sync') else None

    def request_rates(self, cell):
        row = self.cells.get(cell) or {}
        return (row.get('st_tps'), row.get('sy_tps'))

    def decide(self, cell, sync_cost_ms, admission=False):
        if self.noop:
            return 'fallback', 'noop'
        """Override the score gate only when whole requests have measured a lane.

        Compare whole-request throughput once both lanes have sufficient
        observations. Before then, preserve the gate's choice. The selector's
        row-level rule and online price mode have their own decision paths.
        """
        row = self.cells.get(cell)
        if self.lane_rule:
            # The selector owns the row-level rule because it sees every stream
            # round and commits the request to a lane. Defer to that owner rather
            # than substituting the per-cell price criterion.
            return 'fallback', 'rule_elsewhere'
        if self.price_mode:
            # Price mode never defers to the gate: with no cadence yet the answer is
            # simply the default lane (stream), which is what the gate would also
            # choose here, but without handing the decision to the per-round scorer.
            verdict = self.price_verdict(cell) if row is not None else None
            if verdict is None:
                return 'stream', 'price_cold'
            if verdict[1] == 'price_uncertain':
                k = cell[0] if isinstance(cell, tuple) else None
                if admission and k is not None and self.wants_explore(k):
                    return 'sync', 'explore_hedge'
            return verdict
        if row is None:
            return 'fallback', 'no_request_samples'
        if self.request_stride:
            row['admits'] = row.get('admits', 0) + 1
            if row['admits'] % self.request_stride == 0 and not row.get('req_stream'):
                return 'stream', 'explore_request_stream'
        st_tps, sy_tps = self.request_rates(cell)
        n_st = len(row.get('req_stream') or [])
        n_sy = len(row.get('req_sync') or [])
        if n_st < self.min_request_samples or n_sy < self.min_request_samples:
            if n_st == 0 and n_sy == 0:
                # A cold cell has nothing to compare and nothing to alternate with:
                # defer to the gate instead of inventing a lane.
                return 'fallback', 'no_request_samples'
            if n_st <= n_sy:
                return 'stream', 'explore_request_stream'
            return 'sync', 'explore_request_sync'
        if sy_tps > st_tps * self.request_margin:
            return 'sync', 'request_rate_beats_stream'
        return 'stream', 'request_rate_lower'
        return 'fallback', 'no_request_samples'

    def to_state(self):
        """Serialise the cells: a server keeps them, and a second window can too."""
        out = {}
        for (k, gap), row in self.cells.items():
            key = '%s|%s' % (k, '' if gap is None else gap)
            out[key] = dict(n=row.get('n', 0), tokens=row.get('tokens', 0.0),
                            ms=row.get('ms', 0.0), s_n=row.get('s_n', 0),
                            s_tokens=row.get('s_tokens', 0.0), s_ms=row.get('s_ms', 0.0),
                            trials=row.get('trials', 0), rescued=row.get('rescued', 0),
                            explorations=row.get('explorations', 0), admits=row.get('admits', 0),
                            req_stream=list(row.get('req_stream', [])),
                            req_sync=list(row.get('req_sync', [])),
                            st_tps=row.get('st_tps'), sy_tps=row.get('sy_tps'))
        out['@price_version'] = PRICE_STATE_VERSION
        out['@price'] = {str(k): m.state() for k, m in self.price.items()}
        out['@price_cov'] = {str(k): [row for row in m.cov] for k, m in self.price.items()}
        out['@labels'] = {str(k): dict(row) for k, row in self.labels.items()}
        out['@admits'] = {str(k): v for k, v in self.admits.items()}
        return out

    def load_state(self, state):
        if not isinstance(state, dict):
            return 0
        version = state.get('@price_version')
        stale = version != PRICE_STATE_VERSION
        for key, model in ({} if stale else (state.get('@price') or {})).items():
            try:
                k = int(key)
            except Exception:
                continue
            post = self.price_model(k)
            post.theta = [float(model.get('a', post.theta[0])), float(model.get('b', post.theta[1]))]
            post.n = int(model.get('n', 0))
            cov = (state.get('@price_cov') or {}).get(key)
            if isinstance(cov, list) and len(cov) == 2:
                post.cov = [[float(cov[0][0]), float(cov[0][1])],
                            [float(cov[1][0]), float(cov[1][1])]]
        for key, row in ({} if stale else (state.get('@labels') or {})).items():
            try:
                self.labels[int(key)] = dict(trials=int(row.get('trials', 0)),
                                             rescued=int(row.get('rescued', 0)),
                                             cf_trials=int(row.get('cf_trials', 0)),
                                             cf_rescued=int(row.get('cf_rescued', 0)))
            except Exception:
                continue
        self.stale_state = bool(stale and state.get('@price'))
        for key, value in ({} if stale else (state.get('@admits') or {})).items():
            try:
                self.admits[int(key)] = int(value)
            except Exception:
                continue
        loaded = 0
        for key, row in state.items():
            if isinstance(key, str) and key.startswith('@'):
                continue
            try:
                k_text, gap_text = str(key).split('|', 1)
                cell = (int(k_text), None if gap_text == '' else int(gap_text))
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            self.cells[cell] = dict(
                n=int(row.get('n', 0)), tokens=float(row.get('tokens', 0.0)),
                ms=float(row.get('ms', 0.0)), s_n=int(row.get('s_n', 0)),
                s_tokens=float(row.get('s_tokens', 0.0)), s_ms=float(row.get('s_ms', 0.0)),
                trials=int(row.get('trials', 0)), rescued=int(row.get('rescued', 0)),
                explorations=int(row.get('explorations', 0)), admits=int(row.get('admits', 0)),
                req_stream=list(row.get('req_stream', [])),
                req_sync=list(row.get('req_sync', [])),
                st_tps=row.get('st_tps'), sy_tps=row.get('sy_tps'))
            loaded += 1
        return loaded

    def snapshot(self, cell):
        row = self.cells.get(cell) or {}
        return dict(n=row.get('n', 0), s_n=row.get('s_n', 0),
                    trials=row.get('trials', 0), rescued=row.get('rescued', 0),
                    stream_rate=self.stream_rate(cell))
