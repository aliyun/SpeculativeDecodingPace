from collections import deque
from dataclasses import replace
import os
import time
from .policy import JOIN_MS, join_ms_from_env
from .wait_probe import WaitProbe, probe_enabled


def ar_candidate_enabled():
    """AR competes on its own rate instead of only waiting out a deadline."""
    value = os.environ.get('EC_AR_CANDIDATE', '0')
    if value not in ('0', '1'):
        raise ValueError('EC_AR_CANDIDATE must be 0 or 1')
    return value == '1'


class CohortPolicy:
    adaptive = cohort = never_wait = True

    def prepare_policy(self, clock=time.monotonic_ns):
        self.clock = clock
        self.local_only = False
        self.waits = {}
        self.target_step_ns = None
        self.grace_ns = JOIN_MS*1_000_000
        # Opt-in: one startup measurement may raise the window in use, never lower
        # it. The floor is therefore the configured window, not the shipped one.
        self.probe = (WaitProbe(floor_ms=max(JOIN_MS, join_ms_from_env()))
                      if probe_enabled() else None)
        self.stats.update(adaptive_waits=0, adaptive_step_ns=0)
        # AR as a candidate: price an AR step and a speculative round separately
        # and let the slower one lose. Off by default; while any estimate is
        # missing the budget is exactly the shipped/probed window.
        self.ar_candidate = ar_candidate_enabled()
        # Rates, not analytic costs. A round is priced by what it actually
        # delivered over what it actually took, and an AR step by its own
        # measured duration. Medians over a bounded window, and nothing is
        # enabled until both sides have enough samples: the first rounds of a
        # process are graph capture, not steady state.
        self.ar_samples = deque(maxlen=64)
        self.spec_samples = deque(maxlen=64)
        self.rounds_priced = 0
        self.stats.update(ar_candidate=self.ar_candidate, ar_action=0,
                          ar_ready=False, ar_rate_tok_s=0, spec_rate_tok_s=0)
        if self.probe is not None:
            self.stats.update(self.probe.snapshot())

    MIN_SAMPLES = 8
    WARMUP_ROUNDS = 32

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        return ordered[len(ordered)//2]

    def observe_round(self, mode, progress, duration_ns):
        """Price one committed round by what it delivered over what it took.

        An AR round prices AR (its own step). A speculative round prices the
        speculator (committed tokens per elapsed round). Both are medians of a
        bounded window so a cold first round cannot dominate, and the decision
        stays off until each side has MIN_SAMPLES and the process is past its
        first WARMUP_ROUNDS.
        """
        if (not getattr(self, 'ar_candidate', False) or duration_ns is None
                or type(progress) is not int or not 0 < progress <= 64
                or type(duration_ns) is not int or duration_ns <= 0):
            return
        self.rounds_priced += 1
        if mode == 'ar':
            self.ar_samples.append(duration_ns)
            return
        # Tokens per second for this round, kept as an integer.
        self.spec_samples.append(progress*10**9//duration_ns)

    def ar_rate_tok_s(self):
        if len(self.ar_samples) < self.MIN_SAMPLES:
            return None
        ns = self._median(self.ar_samples)
        return 10**9 // ns if ns else None

    def spec_rate_tok_s(self):
        if len(self.spec_samples) < self.MIN_SAMPLES:
            return None
        return self._median(self.spec_samples)

    def ar_wins(self):
        """True when a plain AR step is faster than the rounds being run now."""
        if (not getattr(self, 'ar_candidate', False)
                or self.rounds_priced < self.WARMUP_ROUNDS):
            return None
        ar, spec = self.ar_rate_tok_s(), self.spec_rate_tok_s()
        if ar is None or spec is None:
            return None
        self.stats.update(ar_ready=True, ar_rate_tok_s=ar, spec_rate_tok_s=spec)
        return ar > spec

    def _wait_budget_ns(self):
        """Budget for a row whose consistent edge draft has not arrived yet.

        Without the probe this is the shipped window, unchanged. With it, the
        budget is the sampling cap until the one-shot freeze and the frozen
        estimate afterwards -- both decided by the probe, never by grace_ns.

        AR competes here as the alternative to waiting: when a measured AR step
        is faster than the rounds this deployment is actually running, the row
        gets no budget at all and takes AR this round. The window stays the
        feasibility bound; AR can only shorten a wait, never extend it.
        """
        if self.probe is not None:
            budget = self.probe.deadline_ns()
        else:
            # grace_ns is the configured window: JOIN_MS by default (clamped nowhere)
            # and whatever --join-window-ms or the peer policy set otherwise.
            budget = self.grace_ns
        wins = self.ar_wins()
        if not wins:
            return budget
        first = self.stats.get('ar_action', 0) == 0
        self.stats['ar_action'] = self.stats.get('ar_action', 0) + 1
        if first:
            self.event('ar_action', budget_ns=budget,
                       ar_rate_tok_s=self.stats['ar_rate_tok_s'],
                       spec_rate_tok_s=self.stats['spec_rate_tok_s'],
                       ar_samples=len(self.ar_samples), spec_samples=len(self.spec_samples),
                       rounds_priced=self.rounds_priced)
        return 0

    def _sample(self, rid, wait_ns):
        """Fold one observed wait into the startup probe, at most once per wait."""
        if self.probe is None:
            return
        old = self.waits.get(rid)
        if old is None or old[2] is None:
            return
        self.waits[rid] = (old[0], old[1], None)
        value = self.probe.observe(wait_ns)
        self.stats.update(self.probe.snapshot())
        if value is not None:
            self.grace_ns = value*1_000_000
            self.event('wait_probe', value_ms=value, previous_ms=self.probe.floor_ms,
                       **self.probe.snapshot())

    def _observe_arrival(self, rid, now_ns):
        old = self.waits.get(rid)
        if old is not None and old[2] is not None:
            self._sample(rid, max(0, now_ns-old[2]))

    def _observe_expiry(self, rid):
        # No draft inside the sampling cap: the only fact measured is "not within it".
        self._sample(rid, self.probe.cap_ms*1_000_000 if self.probe is not None else 0)

    def probe_window_ms(self):
        """Frozen startup estimate, or None while sampling/disabled."""
        return self.probe.value_ms if self.probe is not None else None

    def observe_target_step(self, duration_ns):
        if duration_ns > 0:
            sample = min(duration_ns, JOIN_MS*1_000_000)
            self.target_step_ns = (sample if self.target_step_ns is None else
                                   (3*self.target_step_ns+sample)//4)
            self.stats['adaptive_step_ns'] = self.target_step_ns

    def select(self, requests, decision):
        """Split real ready drafts from due AR work; never extend a prefix deadline."""
        if self.local_only:
            return decision
        with self.lock:
            selected = set(decision.selected)
            ready, waiting, expired = set(), set(), set()
            deadlines = []
            for req in requests:
                if req.rid not in selected or not req.rid.startswith(self.ec_prefix):
                    continue
                if getattr(self.entries.get(req.rid), 'source', None) == 'cancelled':
                    continue
                if self.has_draft(req.rid, req.prefix):
                    ready.add(req.rid)
                    self._observe_arrival(req.rid, decision.now_ns)
                    continue
                old = self.waits.get(req.rid)
                if old is None or old[0] != req.prefix:
                    old = (req.prefix, decision.now_ns+self._wait_budget_ns(),
                           decision.now_ns)
                    self.waits[req.rid] = old
                    self.stats['adaptive_waits'] += 1
                waiting.add(req.rid)
                deadlines.append(old[1])
                if decision.now_ns >= old[1]:
                    expired.add(req.rid)
                    self._observe_expiry(req.rid)
            ordinary = selected-ready-waiting
            chosen = ordinary | (expired if expired else ready)
            reason = 'adaptive_ar_due' if expired else 'adaptive_ready' if ready else 'adaptive_wait'
            if ready and waiting:
                ready_deadlines = []
                for req in requests:
                    if req.rid not in ready:
                        continue
                    old = self.waits.get(req.rid)
                    if old is None or old[0] != req.prefix:
                        old = (req.prefix, min(deadlines), None)
                        self.waits[req.rid] = old
                    ready_deadlines.append(old[1])
                if not expired:
                    due = decision.now_ns >= min(ready_deadlines)
                    chosen = ordinary | (ready if due else set())
                    reason = 'cohort_ready_due' if due else 'cohort_grace'
                deadlines.extend(ready_deadlines)
            return replace(decision, selected=tuple(rid for rid in decision.selected if rid in chosen),
                           held_ready=tuple(req.rid for req in requests if req.rid in ready-chosen),
                           pending=tuple(req.rid for req in requests if req.rid in waiting-chosen),
                           deadline_ns=min(deadlines) if deadlines else None, reason=reason)


class FutureCohortPolicy(CohortPolicy):
    """Native-plan providers retain RPC ownership after a target deadline."""
    uses_stream_readers = False
    def prepare_future_policy(self):
        self.prepare_policy()
        self.pending_native = {}

    def ready(self, rid, prefix):
        with self.lock:
            entry = self.entries.get(str(rid))
            if entry is None or entry.prefix != tuple(prefix):
                raise ValueError('missing or stale committed-prefix proposal')
            return True

    def has_draft(self, rid, prefix):
        with self.lock:
            entry = self.entries.get(str(rid))
            if entry is None or entry.prefix != tuple(prefix):
                raise ValueError('missing or stale committed-prefix proposal')
            if entry.source in ('ar', 'cohort_pending'):
                return False
            return self.draft_ready(rid, prefix)

    def submit_committed(self, requests, stride=4, phase='commit', mode=None):
        from concurrent.futures import Future
        from .prefetch import Entry
        requests = list(requests)
        forwarded = []
        with self.lock:
            for rid, future in list(self.pending_native.items()):
                if future.done():
                    future.result()  # Never hide a failed native operation.
                    self.pending_native.pop(rid)
                    self.root_top2.pop(rid, None)
                    entry = self.entries.get(rid)
                    if entry is not None and entry.source == 'cohort_pending':
                        self.entries.pop(rid)
            for request in requests:
                rid = str(request.rid)
                if rid in self.pending_native and not request.finished() and request.output_ids:
                    prefix = tuple(request.prefix)
                    entry = self.entries.get(rid)
                    if entry is None or entry.prefix != prefix:
                        self.entries[rid] = Entry(prefix, stride, Future(), 0, 'cohort_pending')
                else:
                    forwarded.append(request)
        if forwarded:return self.submit_native(forwarded, stride, phase, mode)

    def take_ar(self, rid, prefix):
        rid = str(rid)
        with self.lock:
            entry = self.entries.get(rid)
            if entry is None or entry.prefix != tuple(prefix):
                raise ValueError('AR mode / prefix binding mismatch')
            if entry.source == 'edge' and not entry.future.done():
                self.pending_native[rid] = entry.future
            plan = getattr(entry, 'plan', None)
            if plan is not None:self.growth.issued.pop((rid, plan.prefix_hash), None)
            self.entries.pop(rid)
            self.root_top2.pop(rid, None)
            self.waits.pop(rid, None)
            self.stats['ar_consumed'] += 1

    def release(self, ids, retry_failed=False):
        ids = tuple(ids)
        with self.lock:
            for rid in ids:
                self.waits.pop(str(rid), None)
                self.pending_native.pop(str(rid), None)
        futures = super().release(ids, retry_failed)
        def retired(done):
            with self.lock:
                for rid in ids:self.root_top2.pop(str(rid), None)
        for future in futures:future.add_done_callback(retired)
        return futures
