from dataclasses import dataclass
from collections import OrderedDict
import os
from .fingerprint import fingerprint

_growth = os.environ.get('TRIAGE_GROWTH_EXPERIMENT', '0')
if _growth not in ('0', '1'):
    raise ValueError('GROWTH_EXPERIMENT_MODE')
if _growth == '1' and (os.environ.get('TRIAGE_OFFLOAD_EXPERIMENT') != '1'
        or os.environ.get('TRIAGE_DRAFT_TOKENS') not in ('3', '5', '7')):
    raise ValueError('GROWTH_EXPERIMENT_HORIZON')
_k_env = os.environ.get('TRIAGE_DRAFT_TOKENS')
if _k_env is not None and _k_env not in ('3', '5', '7'):
    raise ValueError('TRIAGE_DRAFT_TOKENS must be one of 3/5/7')
DRAFT_TOKENS = int(_k_env) if _k_env is not None else 3
WIDTH = DRAFT_TOKENS + 1
_ahead_env = os.environ.get('TRIAGE_STREAM_AHEAD')
if _ahead_env is not None:
    # Depth of the edge's draft buffer. Deeper lets production jitter be absorbed
    # without changing what is drafted; unset keeps the declared 32.
    if not _ahead_env.isdigit() or not 8 <= int(_ahead_env) <= 256:
        raise ValueError('TRIAGE_STREAM_AHEAD must be 8..256')
    STREAM_AHEAD = int(_ahead_env)
else:
    STREAM_AHEAD = 32
MAX_REQUESTS = 16
PREFILL_REQUESTS = 4
JOIN_MS = 100
JOIN_WINDOWS = (10, 20, JOIN_MS)
JOIN_RANGE = (10, 1000)


def join_ms_from_env(default=JOIN_MS):
    """Cloud-side window; a startup probe may only raise it inside the range."""
    raw = os.environ.get('EC_JOIN_MS', '')
    if raw == '':
        return default
    low, high = JOIN_RANGE
    if not raw.isdigit() or not low <= int(raw) <= high:
        raise ValueError('EC_JOIN_MS must be %d..%d' % JOIN_RANGE)
    return int(raw)
from .footprint_experiment import context as _experiment_context
CONTEXT = _experiment_context(32768)
MAX_PROMPT = 8193
MAX_OUTPUT = 1024
PARENTS = tuple(range(-1, DRAFT_TOKENS))


@dataclass(frozen=True)
class LoadDecision:
    mode: str
    reason: str


def execution_mode(prefix_lengths, context=CONTEXT):
    """Fixed K3 for valid cohorts, with the native context-boundary AR guard."""
    lengths = list(prefix_lengths)
    if not lengths:
        return LoadDecision('spec', 'empty')
    if len(lengths) > MAX_REQUESTS or any(type(n) is not int or not 1 <= n <= context for n in lengths):
        raise ValueError('invalid execution cohort')
    if any(n-1+WIDTH > context for n in lengths):
        return LoadDecision('ar', 'context_boundary')
    from .offload_experiment import strategy
    if strategy() == 'ar':
        return LoadDecision('ar', 'explicit_same_executor_ar_ablation')
    if os.environ.get('TRIAGE_EDGE_BACKEND') == 'cpu':
        cpu_context = int(os.environ.get('TRIAGE_CPU_CONTEXT', '12288'))
        if not DRAFT_TOKENS < cpu_context <= context:
            raise ValueError('invalid CPU draft context')
        if any(n+DRAFT_TOKENS > cpu_context for n in lengths):
            return LoadDecision('ar', 'cpu_context_boundary')
    if os.environ.get('TRIAGE_EDGE_BACKEND') == 'webgpu' and any(n+DRAFT_TOKENS > 1024 for n in lengths):
        return LoadDecision('ar', 'webgpu_context_boundary')
    return LoadDecision('spec', 'fixed_k%d' % DRAFT_TOKENS)


def token_ids(tokens):
    if not isinstance(tokens, (list, tuple)) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('nonempty token IDs required')
    return list(tokens)


def common_prefix(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def validate_budget(length, maximum):
    if type(length) is not int or not 1 <= length <= MAX_PROMPT:
        raise ValueError('prompt must contain 1..8193 tokens')
    if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT:
        raise ValueError('max_tokens must be 1..1024')
    if length + maximum + WIDTH > CONTEXT:
        raise ValueError('context limit')


def validate_sampling(body):
    if not isinstance(body, dict):
        raise ValueError('request must be an object')
    if body.get('stream') is not True or body.get('temperature') != 0 or body.get('n', 1) != 1:
        raise ValueError('single-edge requires greedy, single-completion SSE')
    maximum = body.get('max_tokens')
    validate_budget(1, maximum)
    ignore = body.get('ignore_eos', False)
    minimum = body.get('min_tokens', 0)
    if type(ignore) is not bool or type(minimum) is not int or not 0 <= minimum <= maximum:
        raise ValueError('invalid min_tokens / ignore_eos')
    if minimum and not ignore:
        raise ValueError('positive min_tokens requires ignore_eos=true')
    if body.get('logprobs') is not None:
        raise ValueError('logprobs unsupported')
    for key in ('rid', 'session_id', 'session_params'):
        if body.get(key) is not None:
            raise ValueError('unsupported request field: '+key)
    for key in ('stop', 'stop_token_ids', 'stop_regex', 'logprobs', 'regex', 'json_schema',
                'ebnf', 'logit_bias', 'custom_logit_processor', 'rid', 'session_id',
                'session_params', 'lora_path', 'response_format', 'return_hidden_states',
                'return_routed_experts', 'custom_params', 'images_config', 'bootstrap_host',
                'bootstrap_port', 'bootstrap_room', 'extra_key', 'cache_salt', 'priority',
                'routed_dp_rank', 'disagg_prefill_dp_rank', 'data_parallel_rank'):
        if body.get(key) is not None and body.get(key) not in (False, [], {}, ''):
            raise ValueError('unsupported request field: ' + key)
    for key, default in (('presence_penalty', 0), ('frequency_penalty', 0), ('repetition_penalty', 1)):
        if body.get(key, default) != default:
            raise ValueError('unsupported penalty: ' + key)
    if body.get('echo', False) or body.get('suffix'):
        raise ValueError('echo / suffix unsupported')
    if body.get('best_of') not in (None, 1):
        raise ValueError('best_of unsupported')
    return maximum, ignore


class SuffixCache:
    """Bounded LRU learned only from target-committed generated IDs.

    Insertions and lookups use only generated tokens committed by the target.
    """
    def __init__(self, max_entries=262144, min_suffix=4, max_suffix=12, horizon=max(6, DRAFT_TOKENS)):
        if not 0 < min_suffix <= max_suffix or max_entries < 1 or horizon < DRAFT_TOKENS:
            raise ValueError('invalid cache bounds')
        self.max_entries, self.min_suffix, self.max_suffix, self.horizon = max_entries, min_suffix, max_suffix, horizon
        self.entries, self.cursor = OrderedDict(), OrderedDict()

    def observe(self, rid, generated):
        if generated:
            token_ids(generated)
        self._observe_snapshot(rid, generated)

    def observe_request(self, request):
        from .contracts import Request
        if type(request) is Request and type(request.generated) is tuple:
            self._observe_snapshot(request.rid, request.generated)
        else:
            self.observe(request.rid, request.generated)

    def _observe_snapshot(self, rid, generated):
        end = max(0, len(generated) - self.horizon + 1)
        start = self.cursor.get(rid, self.min_suffix)
        for position in range(start, end):
            value = tuple(generated[position:position + self.horizon])
            for width in range(self.min_suffix, min(self.max_suffix, position) + 1):
                key = tuple(generated[position-width:position])
                self.entries[key] = value
                self.entries.move_to_end(key)
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
        self.cursor[rid] = max(start, end)
        self.cursor.move_to_end(rid)
        while len(self.cursor) > 512:
            self.cursor.popitem(last=False)

    def propose(self, generated, want=DRAFT_TOKENS):
        if not 0 < want <= self.horizon:
            return []
        for width in range(min(self.max_suffix, len(generated)), self.min_suffix-1, -1):
            key = tuple(generated[-width:])
            if key in self.entries:
                self.entries.move_to_end(key)
                return list(self.entries[key][:want])
        return []

    def release(self, rid):
        self.cursor.pop(rid, None)


@dataclass(frozen=True)
class Row:
    rid: str
    ec: bool
    ready: bool


@dataclass(frozen=True)
class Decision:
    selected: tuple
    held_ready: tuple
    pending: tuple
    cohort: tuple
    started_ns: int | None
    deadline_ns: int | None
    reason: str
    now_ns: int
    released_ready: tuple = ()


class CohortJoin:
    """Ready-cohort deadline is immutable; native prefill releases stay eligible."""
    def __init__(self, window_ms=JOIN_MS, max_requests=MAX_REQUESTS):
        low, high = JOIN_RANGE
        if (type(window_ms) is not int or not low <= window_ms <= high
                or type(max_requests) is not int or max_requests != MAX_REQUESTS):
            raise ValueError('cohort requires 10..1000ms and 16 requests')
        self.window_ns = window_ms * 1_000_000
        self.max_requests = max_requests
        self.members, self.started_ns, self.previous_ns = set(), None, None

    def decide(self, rows, now_ns, released=()):
        ids = [r.rid for r in rows]
        if type(now_ns) is not int or now_ns < 0 or self.previous_ns is not None and now_ns < self.previous_ns:
            raise ValueError('monotonic timestamp required')
        if len(ids) > self.max_requests or len(ids) != len(set(ids)):
            raise ValueError('unique bounded rows required')
        if any(not r.ec and not r.ready for r in rows):
            raise ValueError('bootstrap rows must be ready')
        self.previous_ns = now_ns
        protected = set(released)
        if len(protected) != len(released) or not protected <= {r.rid for r in rows if r.ec and r.ready}:
            raise ValueError('invalid previously released rows')
        ec = {r.rid for r in rows if r.ec} - protected
        ready = {r.rid for r in rows if r.ec and r.ready} - protected
        pending = ec - ready
        ordinary = {r.rid for r in rows if not r.ec} | protected
        self.members.intersection_update(ec)
        if not self.members:
            self.started_ns = None
        if not pending:
            reason = 'all_ready'
        elif self.members and self.members <= ready:
            reason = 'cohort_ready'
        elif self.members and now_ns >= self.started_ns + self.window_ns:
            reason = 'deadline'
        elif self.members:
            reason = 'joining'
        elif ready:
            self.members, self.started_ns = set(ec), now_ns
            reason = 'join_started'
        else:
            reason = 'waiting_for_any_draft'
        selected = ordinary | ready if reason in ('all_ready', 'cohort_ready', 'deadline') else ordinary
        ordered = lambda group: tuple(rid for rid in ids if rid in group)
        result = Decision(ordered(selected), ordered(ready-selected), ordered(pending),
                          tuple(sorted(self.members)), self.started_ns,
                          self.started_ns+self.window_ns if self.started_ns is not None else None,
                          reason, now_ns, ordered(protected))
        if reason in ('all_ready', 'cohort_ready', 'deadline'):
            self.members, self.started_ns = set(), None
        return result
