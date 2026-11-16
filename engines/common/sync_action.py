import math
import os


def _num(value, default=0.0):
    if value is None:
        return float(default)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('SYNC_ACTION_COST')
    return float(value)


class SyncDraftAction:
    """One `sync_draft(K, fork)` acquisition, priced in milliseconds."""

    name = 'sync_draft'

    def __init__(self, k, *, rtt_ms, edge_ms_per_token, c_ms, fork_at=0, fork_len=0,
                 c_node_ms=0.0, c_mask_ms=0.0, c_shape_ms=0.0, split_loss_ms=0.0,
                 switch_ms=0.0, remaining_rounds=1, rescue_gain=1.0):
        if type(k) is not int or k < 1:
            raise ValueError('SYNC_ACTION_K')
        if type(fork_at) is not int or type(fork_len) is not int:
            raise ValueError('SYNC_ACTION_FORK')
        if fork_at and not 1 <= fork_at <= k:
            raise ValueError('SYNC_ACTION_FORK')
        if fork_at and not 1 <= fork_len:
            raise ValueError('SYNC_ACTION_FORK')
        self.k = k
        self.fork_at, self.fork_len = fork_at, fork_len
        self.rtt_ms = _num(rtt_ms)
        self.edge_ms_per_token = _num(edge_ms_per_token)
        # Target compute covers both verification and commit.
        self.c_ms = _num(c_ms)
        if self.c_ms <= 0:
            raise ValueError('SYNC_ACTION_C')
        self.c_node_ms = _num(c_node_ms)
        self.c_mask_ms = _num(c_mask_ms)
        self.c_shape_ms = _num(c_shape_ms)
        self.split_loss_ms = _num(split_loss_ms)
        self.switch_ms = _num(switch_ms)
        self.remaining_rounds = max(1, int(remaining_rounds))
        self.rescue_gain = _num(rescue_gain, 1.0)

    # ---- costs -----------------------------------------------------------
    def w_sync_ms(self):
        """RTT plus the edge's sequential K-step draft."""
        return self.rtt_ms + self.k * self.edge_ms_per_token

    def node_ms(self):
        """Extra verification rows for the branch (root top-2 costs one)."""
        nodes = self.fork_len if self.fork_at else 1  # root top-2: one side node
        return nodes * self.c_node_ms

    def amortised_switch_ms(self):
        return self.switch_ms / self.remaining_rounds

    def cost_ms(self):
        """Every term the decision must charge, in one number."""
        return (self.w_sync_ms() + self.node_ms() + self.c_mask_ms
                + self.c_shape_ms + self.split_loss_ms + self.amortised_switch_ms())

    # ---- score -----------------------------------------------------------
    def delivered(self, base_g):
        """Expected committed progress: the stream's g plus the rescue gain.

        The action includes an alternate root, so the rescue term applies.
        """
        return base_g + self.rescue_gain

    def score(self, base_g, base_c_ms):
        """E[N_a]/E[T_a] for this action: C + W_sync + charged extras."""
        denominator = self.c_ms + self.cost_ms()
        if denominator <= 0:
            raise ValueError('SYNC_ACTION_TIME')
        return self.delivered(base_g) / denominator

    def stream_score(self, base_g, base_c_ms):
        return base_g / base_c_ms if base_c_ms > 0 else 0.0

    def wins(self, base_g, base_c_ms, *, margin=1.10):
        """Switch only when the conservative score clears the stream by `margin`.

        The margin is the LCB stand-in for one round: without per-action variance
        we refuse to switch on a tie, and B is already inside cost_ms, so a
        marginal win is not enough to justify a drain/reopen.
        """
        return self.score(base_g, base_c_ms) > margin * self.stream_score(base_g, base_c_ms)

    def snapshot(self):
        return dict(name=self.name, k=self.k, fork=(self.fork_at, self.fork_len),
                    c_ms=self.c_ms, w_sync_ms=self.w_sync_ms(), node_ms=self.node_ms(),
                    c_mask_ms=self.c_mask_ms, c_shape_ms=self.c_shape_ms,
                    split_loss_ms=self.split_loss_ms,
                    switch_ms=self.switch_ms, remaining_rounds=self.remaining_rounds,
                    amortised_switch_ms=self.amortised_switch_ms(),
                    cost_ms=self.cost_ms())


class GapCurve:
    """Assumed exponential mapping from root gap to rescue probability.

    The curve and its bounds are configurable priors. A hedge must clear the
    minimum rescue probability and the tolerated extra round cost. The
    throughput break-even is reported separately from this capability decision.
    """

    def __init__(self, *, pmax=0.149, tau=4.0, rescue_tokens=2, gap_max=2.0,
                 p_min=0.09, price_cap=1.5):
        if pmax < 0 or tau <= 0 or rescue_tokens < 0 or gap_max < 0:
            raise ValueError('GAP_CURVE')
        if not 0.0 <= p_min <= 1.0 or price_cap < 1.0:
            raise ValueError('GAP_CURVE_POLICY')
        self.pmax, self.tau = pmax, tau
        self.rescue_tokens, self.gap_max = rescue_tokens, gap_max
        # Both bounds are prior assumptions until per-round rescue labels
        # provide direct evidence for the current workload.
        self.p_min, self.price_cap = p_min, price_cap

    def p_rescue(self, gap):
        if gap is None or not _num_ok(gap) or gap > self.gap_max:
            return 0.0
        return self.pmax * math.exp(-float(gap) / self.tau)

    def eta(self, gap):
        """Expected extra tokens from hedging at this gap."""
        return self.p_rescue(gap) * self.rescue_tokens

    def break_even_eta(self, base_g, base_c_ms, sync_round_ms):
        """Rescued tokens per round needed for the hedge to pay for itself."""
        if base_c_ms <= 0 or sync_round_ms <= 0:
            return None
        return base_g * (sync_round_ms / base_c_ms - 1.0)

    def worth(self, gap, base_g, base_c_ms, action, *, sync_round_ms=None,
              p_min=None, price_cap=None):
        """Buy the second answer when rescue is likely and its price is bounded.

        `sync_round_ms` is the *measured* cadence of a hedged round when the
        executor has one; otherwise the action's terms price it (C + W_sync +
        extras). The measured form matters because the edge overlaps the round
        trip with its own drafting, so a hedged round does not cost a full RTT.
        """
        if gap is None or not _num_ok(gap) or gap > self.gap_max:
            return False, 'gap_too_large_or_unknown'
        p = self.p_rescue(gap)
        if p <= 0:
            return False, 'gap_eta_zero'
        if p < (self.p_min if p_min is None else p_min):
            return False, 'rescue_probability_too_low'
        price = (action.c_ms + action.cost_ms()) if sync_round_ms is None else sync_round_ms
        if price <= 0:
            return False, 'gap_sync_time'
        cap = self.price_cap if price_cap is None else price_cap
        if base_c_ms > 0 and price > cap * base_c_ms:
            return False, 'price_above_cap'
        return True, 'second_answer_worth_the_price'


def _num_ok(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def decide_transport(action, base_g, base_c_ms, *, previous='stream',
                     capability_required=False, margin=1.10):
    """One round's transport choice; pure so it is testable off the GPU.

    Returns (transport, reason, inputs). Capability forces sync; otherwise the
    action must clear `margin` times the stream score, with B already inside the
    action's cost. A switch to/from sync charges the same B once.
    """
    stream_score = action.stream_score(base_g, base_c_ms)
    sync_score = action.score(base_g, base_c_ms)
    inputs = dict(previous=previous, stream_score=stream_score, sync_score=sync_score,
                  margin=margin, capability_required=bool(capability_required),
                  cost_ms=action.cost_ms(), switch_ms=action.switch_ms)
    if capability_required:
        return 'sync', 'capability_required', inputs
    if sync_score > margin * stream_score:
        return 'sync', 'score_beats_stream_margin', inputs
    if previous == 'sync' and sync_score > stream_score:
        # Staying is cheaper than paying B again; only leave on a clear loss.
        return 'sync', 'hold_no_margin_to_leave', inputs
    return 'stream', 'stream_score_higher', inputs


def second_answer_requested():
    """Capability gate: sync exists only for a request that wants a second answer."""
    value = os.environ.get('TRIAGE_SECOND_ANSWER', '0')
    if value not in ('0', '1'):
        raise ValueError('SECOND_ANSWER_FLAG')
    return value == '1'
