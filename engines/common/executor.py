import time

from .contracts import Dispatch, Proposal
from .policy import CohortJoin, LoadDecision, Row, MAX_REQUESTS, PREFILL_REQUESTS, WIDTH, JOIN_MS, execution_mode
from .stream import make_prefetch


class Executor:
    ec_prefix = 'ec-'
    def __init__(self, client=None, *, prefetch=None, clock=time.monotonic_ns, join_ms=JOIN_MS):
        self.clock, self.join = clock, CohortJoin(join_ms)
        self.prefetch = prefetch if prefetch is not None else make_prefetch(client)
        self.weak_edge_multiedge = bool(getattr(getattr(self.prefetch, 'client', None), 'weak_edge_multiedge', False))
        self.peer_join_ms = join_ms
        if getattr(self.prefetch, 'cohort', False):
            self.prefetch.grace_ns = self.join.window_ns
        self.requests, self.executing, self.released = {}, {}, set()
        self.growth_plans = {}
        self.executing_started = {}
        self.executing_mode = {}

        self.last_decision, self.last_signature = None, None
        self.ar_rounds = self.spec_rounds = 0

    def _update_peer_join_window(self, active):
        if self.weak_edge_multiedge and not self.join.members:
            self.join.window_ns = self.prefetch.client.ready_join_ms(
                self.peer_join_ms, active=active) * 1_000_000
            self.prefetch.grace_ns = self.join.window_ns
        # A frozen startup estimate applies to cohorts that have not started yet;
        # a live deadline is immutable and keeps the window it was opened with.
        probe = getattr(self.prefetch, 'probe_window_ms', None)
        if probe is not None and not self.join.members and self.join.started_ns is None:
            value = probe()
            if value is not None and self.join.window_ns != value*1_000_000:
                self.join.window_ns = value*1_000_000
                self.prefetch.grace_ns = self.join.window_ns

    @staticmethod
    def admission_budget(active):
        if not 0 <= active <= MAX_REQUESTS:
            raise ValueError('invalid active request count')
        return min(PREFILL_REQUESTS, MAX_REQUESTS-active)

    def _observe(self, request):
        if not request.rid.startswith('ec-'):
            return
        previous = self.requests.get(request.rid)
        if previous is not None and (request.prompt != previous.prompt
                or request.maximum != previous.maximum
                or request.generated[:len(previous.generated)] != previous.generated):
            raise ValueError('native committed prefix was rewritten')
        self.requests[request.rid] = request

    def committed(self, requests, phase='commit'):
        """Native has committed once; only now learn and schedule next drafts."""
        self.prefetch.poll_cleanup()
        finished, durations = [], []
        for request in requests:
            if not request.rid.startswith('ec-'):
                continue
            previous = self.executing.pop(request.rid, None)
            growth_plan = self.growth_plans.pop(request.rid, None)
            started = self.executing_started.pop(request.rid, None)
            round_mode = self.executing_mode.pop(request.rid, 'spec')
            elapsed = None if started is None else self.clock()-started
            if elapsed is not None:
                durations.append(elapsed)

            observed = self.requests.get(request.rid)
            progress = len(request.generated) - (len(observed.generated) if observed else 0)
            if previous is not None:
                if request.prefix[:len(previous.prefix)] != previous.prefix:
                    raise ValueError('verification result has a different committed prefix')
                progress = len(request.generated)-len(previous.generated)
                if not 0 <= progress <= (growth_plan.k+1 if growth_plan else WIDTH) or not progress and not (request.aborted or request.done):
                    raise ValueError('invalid native verification progress')
            if progress:
                self.prefetch.event('commit', rid=request.rid, phase=phase,
                    prefix_length=len(request.prefix), emitted=progress,
                    token_ids=list(request.generated[-progress:]))
            if elapsed is not None and getattr(self.prefetch, 'ar_candidate', False):
                self.prefetch.observe_round(round_mode, progress, elapsed)
            self._observe(request)
            if (getattr(self.prefetch, 'exact_cache', None) is not None
                    or getattr(self.prefetch, 'similar_cache', None) is not None):
                self.prefetch.observe_committed(request)
            if request.done:
                finished.append(request.rid)
        if durations and getattr(self.prefetch, 'adaptive', False):
            self.prefetch.observe_target_step(max(durations))
        self.finish(finished)
        active = list(self.requests.values())
        self.prefetch.submit_committed(active, WIDTH, phase)

    def plan(self, requests, now_ns=None):
        """Prepare proposals with the context guard, then release one ready cohort."""
        self.prefetch.poll_cleanup()
        live = [request for request in requests if not request.done]
        for request in live:
            self._observe(request)
        eligible = [request for request in live if request.rid.startswith('ec-')]
        mode = (self.prefetch.mode(eligible) if self.weak_edge_multiedge else
                execution_mode(len(request.prefix) for request in eligible))
        self.prefetch.submit_committed(eligible, WIDTH, 'plan', mode=mode.mode)
        rows = [Row(request.rid, request.rid.startswith('ec-'),
                    self.prefetch.ready(request.rid, request.prefix)
                    if request.rid.startswith('ec-') else True) for request in live]
        self.released.intersection_update(row.rid for row in rows if row.ec and row.ready)
        self._update_peer_join_window(len(eligible))
        decision = self.join.decide(rows, self.clock() if now_ns is None else now_ns,
                                    released=self.released)
        if mode.mode == 'spec' and getattr(self.prefetch, 'adaptive', False):
            decision = self.prefetch.select(live, decision)
        if self.weak_edge_multiedge:
            ec_rows = [row for row in rows if row.ec]
            self.prefetch.client.observe_draft_wait(
                mode.mode == 'spec' and bool(ec_rows) and not decision.selected
                and not any(self.prefetch.has_draft(req.rid, req.prefix) for req in eligible), len(ec_rows))
        selected = set(decision.selected)
        actual = execution_mode(len(request.prefix) for request in eligible if request.rid in selected)
        if selected and actual.mode != mode.mode and not self.weak_edge_multiedge:
            raise ValueError('execution mode changed after cohort selection')
        signature = (decision.selected, decision.reason, decision.cohort, decision.deadline_ns, mode.mode)
        if signature != self.last_signature or decision.selected:
            self.prefetch.event('cohort', selected=decision.selected, reason=decision.reason,
                mode=mode.mode, deadline_ns=decision.deadline_ns)
            self.last_signature = signature
        self.last_decision = decision
        return decision

    def native_selection(self, actual_ids, phase):
        """A native prefill may defer a selected decode without consuming it."""
        if self.last_decision is not None:
            self.released.update(rid for rid in self.last_decision.selected if rid in self.requests)
        if phase == 'decode':
            self.released.difference_update(actual_ids)

    def begin(self, requests):
        """Bind and consume proposals against the actual native decode batch."""
        self.prefetch.poll_cleanup()
        mode = (self.prefetch.mode(requests) if self.weak_edge_multiedge else
                execution_mode(len(request.prefix) for request in requests))
        if self.weak_edge_multiedge:
            self.prefetch.submit_committed(requests, WIDTH, 'begin', mode=mode.mode)
        if mode.mode == 'spec' and getattr(self.prefetch, 'never_wait', False):
            available = [self.prefetch.has_draft(request.rid, request.prefix)
                         for request in requests if request.rid.startswith('ec-')]
            drafted = (bool(available) and all(available) if getattr(self.prefetch, 'adaptive', False)
                       else any(available))
            mode = LoadDecision('spec', 'stream_draft') if drafted else LoadDecision('ar', 'no_stream_draft')
        proposals = []
        for request in requests:
            self._observe(request)
            if request.rid in self.executing:
                raise ValueError('request already has uncommitted native work')
            candidates = []
            growth_plan = None
            if request.rid.startswith('ec-'):
                if mode.mode == 'ar':
                    self.prefetch.take_ar(request.rid, request.prefix)
                else:
                    if hasattr(self.prefetch,'plan_for'):
                        growth_plan=self.prefetch.plan_for(request.rid,request.prefix)
                    candidates = self.prefetch.take(request.rid, request.prefix)
                if request.aborted:
                    candidates = []
                self.executing[request.rid] = request
                self.executing_mode[request.rid] = mode.mode
                if growth_plan is not None:self.growth_plans[request.rid]=growth_plan
                if getattr(self.prefetch, 'adaptive', False):
                    self.executing_started[request.rid] = self.clock()
            proposals.append(Proposal(request.rid, request.prefix, tuple(candidates), request.aborted, growth_plan))

        self.released.difference_update(request.rid for request in requests)
        if mode.mode == 'ar':
            self.ar_rounds += 1
        else:
            self.spec_rounds += 1
        self.prefetch.event('execute', mode=mode.mode, reason=mode.reason,
            ids=[request.rid for request in requests], prefix_lengths=[len(request.prefix) for request in requests])
        return Dispatch(mode.mode, mode.reason, tuple(proposals))

    def cancel(self, rid, prefix):
        self.prefetch.cancel(rid, prefix)

    def finish(self, ids):
        for rid in ids:
            self.requests.pop(rid, None)
            self.executing.pop(rid, None)
            self.growth_plans.pop(rid, None)
            self.executing_started.pop(rid, None)

            self.released.discard(rid)
        return self.prefetch.release(ids)

    def close(self):
        self.prefetch.close()
