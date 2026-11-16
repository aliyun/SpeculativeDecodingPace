from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from itertools import islice
import math
import os
from collections import OrderedDict
from .fingerprint import fingerprint

from .growth import KS
from .policy import DRAFT_TOKENS, token_ids
from .contracts import Request


@dataclass(slots=True)
class ContextEntry:
    vector: dict | None
    continuation: tuple | None
    first_rid: str | None = None
    shared: bool = False


class SimilarContextCache:
    def __init__(self, max_entries=4096, max_token_slots=262144, window=32,
                 min_context=8, position_bucket=32, max_candidates=64,
                 min_similarity=0.9, cross_request_only=False):
        if DRAFT_TOKENS != 3:
            raise ValueError('similar-context drafts require fixed K3')
        dimensions = (max_entries, max_token_slots, window, min_context,
                      position_bucket, max_candidates)
        if (any(type(n) is not int or n < 1 for n in dimensions)
                or min_context > window or window > 256 or max_candidates > 256):
            raise ValueError('positive bounded similar-cache dimensions required')
        if (type(min_similarity) not in (int, float)
                or not math.isfinite(min_similarity) or not 0 < min_similarity <= 1):
            raise ValueError('similarity threshold must be finite and in (0, 1]')
        if type(cross_request_only) is not bool:
            raise ValueError('cross-request admission requires a boolean')
        self.cross_request_only = cross_request_only
        self.self_filtered = self.no_cross_candidate = 0
        self.max_entries, self.max_token_slots = max_entries, max_token_slots
        self.window, self.min_context = window, min_context
        self.position_bucket, self.max_candidates = position_bucket, max_candidates
        self.min_similarity = min_similarity
        self.entries, self.buckets, self.cursor = OrderedDict(), {}, OrderedDict()
        self.token_slots = self.conflicts = self.evictions = 0
        self.queries = self.hits = self.comparisons = 0

    @staticmethod
    def _vector(context):
        # Disjoint tuple lengths avoid unigram/bigram hash-feature collisions.
        counts = Counter((t,) for t in context)
        for left, right in zip(context, context[1:]):
            counts[(left, right)] += 2
        norm = math.sqrt(sum(value*value for value in counts.values()))
        return {key: value/norm for key, value in counts.items()}

    def _key(self, prompt, generated, position):
        if position >= self.window:
            context = generated[position-self.window:position]
        else:
            context = prompt[-(self.window-position):] + generated[:position]
        if len(context) < self.min_context:
            return None
        return position//self.position_bucket, context

    @staticmethod
    def _bucket(key):
        return key[0], key[1][-1]

    def _unindex(self, key):
        bucket_key = self._bucket(key)
        bucket = self.buckets.get(bucket_key)
        if bucket is not None:
            bucket.pop(key, None)
            if not bucket:
                self.buckets.pop(bucket_key)

    def _put(self, key, continuation, owner=None):
        size = len(key[1]) + DRAFT_TOKENS
        if size > self.max_token_slots:
            return
        row = self.entries.get(key)
        if row is not None:
            if row.continuation is not None and row.continuation != continuation:
                # Local contexts can be ambiguous even with exact local IDs.
                # Quarantine conflicting keys rather than selecting a winner.
                row.continuation = None
                self.conflicts += 1
                self._unindex(key)
            elif (self.cross_request_only and row.continuation is not None
                  and not row.shared and row.first_rid != owner):
                # One RID plus this bit proves whether any query has an
                # independent donor; never retain an unbounded owner set.
                row.shared = True
            self.entries.move_to_end(key)
            if row.continuation is not None:
                self.buckets[self._bucket(key)].move_to_end(key)
            return
        # Most learned contexts are evicted without ever being compared.
        # Keep IDs only until a lookup actually needs this vector.
        row = ContextEntry(None, continuation,
                           owner if self.cross_request_only else None)
        self.entries[key] = row
        self.buckets.setdefault(self._bucket(key), OrderedDict())[key] = row
        self.token_slots += size
        while len(self.entries) > self.max_entries or self.token_slots > self.max_token_slots:
            old, _ = self.entries.popitem(last=False)
            self.token_slots -= len(old[1]) + DRAFT_TOKENS
            self._unindex(old)
            self.evictions += 1

    def observe(self, rid, prompt, generated):
        prompt = tuple(token_ids(prompt))
        generated = tuple(token_ids(generated)) if generated else ()
        self._observe(rid, prompt, generated)

    def observe_committed(self, request):
        """Reuse only an actual Request's already-validated immutable tuples."""
        if (type(request) is Request and type(request.prompt) is tuple
                and type(request.generated) is tuple):
            self._observe(request.rid, request.prompt, request.generated)
        else:
            # Frozen dataclasses can still contain mutable lists. Do not
            # bypass validation for those, or for arbitrary native objects.
            self.observe(request.rid, request.prompt, request.generated)

    @staticmethod
    def _check_rid(rid):
        if type(rid) is not str or not rid:
            raise ValueError('cross-request admission requires a request id')

    def _observe(self, rid, prompt, generated):
        if self.cross_request_only:
            self._check_rid(rid)
        start = self.cursor.get(rid, 1)
        end = max(1, len(generated)-DRAFT_TOKENS+1)
        for position in range(start, end):
            key = self._key(prompt, generated, position)
            if key is not None:
                continuation = generated[position:position+DRAFT_TOKENS]
                if self.cross_request_only:
                    self._put(key, continuation, rid)
                else:
                    self._put(key, continuation)
        self.cursor[rid] = max(start, end)
        self.cursor.move_to_end(rid)
        while len(self.cursor) > 512:
            self.cursor.popitem(last=False)

    def propose(self, prompt, generated, *, rid=None):
        prompt = tuple(token_ids(prompt))
        generated = tuple(token_ids(generated)) if generated else ()
        if self.cross_request_only:
            return self._propose(prompt, generated, rid)
        return self._propose(prompt, generated)

    def propose_committed(self, request):
        if (type(request) is Request and type(request.prompt) is tuple
                and type(request.generated) is tuple):
            if self.cross_request_only:
                return self._propose(request.prompt, request.generated, request.rid)
            return self._propose(request.prompt, request.generated)
        if self.cross_request_only:
            return self.propose(request.origin_input_ids, request.output_ids, rid=request.rid)
        return self.propose(request.origin_input_ids, request.output_ids)

    def _propose(self, prompt, generated, rid=None):
        if self.cross_request_only:
            self._check_rid(rid)
        self.queries += 1
        if not generated:
            return []
        if not self.entries:
            return []
        key = self._key(prompt, generated, len(generated))
        if key is None:
            return []
        previous = self.entries.get(key)
        if previous is not None and previous.continuation is None:
            # A known ambiguous query must not sidestep quarantine by using
            # its nearest neighbor. This lasts until the bounded LRU evicts it.
            return []
        bucket = self.buckets.get(self._bucket(key))
        if not bucket:
            return []
        vector = None
        best, best_score = None, -1.0
        # Bound inspected entries, not just eligible/scored entries. A bucket
        # holds the same row objects as entries, avoiding another context hash.
        for candidate, row in islice(reversed(bucket.items()), self.max_candidates):
            if (self.cross_request_only and not row.shared
                    and row.first_rid == rid):
                self.self_filtered += 1
                continue
            if vector is None:
                vector = self._vector(key[1])
            if row.vector is None:
                row.vector = self._vector(candidate[1])
            get = row.vector.get
            score = math.fsum(value*get(feature, 0.0)
                              for feature, value in vector.items())
            self.comparisons += 1
            if score+1e-12 >= self.min_similarity and score > best_score:
                best, best_score = candidate, score
        if best is None:
            if self.cross_request_only and vector is None:
                self.no_cross_candidate += 1
            return []
        self.entries.move_to_end(best)
        bucket.move_to_end(best)
        self.hits += 1
        return list(self.entries[best].continuation)

    def release(self, rid):
        self.cursor.pop(rid, None)

    def clear(self):
        self.entries.clear()
        self.buckets.clear()
        self.cursor.clear()
        self.token_slots = 0

    def snapshot(self):
        result = dict(similar_cache_entries=len(self.entries),
                    similar_cache_token_slots=self.token_slots,
                    similar_cache_buckets=len(self.buckets),
                    similar_cache_cursors=len(self.cursor),
                    similar_cache_conflicts=self.conflicts,
                    similar_cache_evictions=self.evictions,
                    similar_cache_queries=self.queries,
                    similar_cache_hits=self.hits,
                    similar_cache_comparisons=self.comparisons,
                    similar_cache_materialized_vectors=sum(row.vector is not None
                                                           for row in self.entries.values()))
        if self.cross_request_only:
            result.update(similar_cache_cross_request_only=True,
                          similar_cache_self_filtered=self.self_filtered,
                          similar_cache_no_cross_candidate=self.no_cross_candidate)
        return result


def auto_gate_from_env(default=False):
    """Layer/width acceptance gate; off unless a launch asks for it.

    Disable a layer or suffix width after ``gate_window`` drafts whose mean
    acceptance is below ``gate_ratio``, with periodic probes to recover.
    """
    raw = os.environ.get('TRIAGE_UNIFIED_AUTO_GATE')
    if raw is None:
        return default
    if raw not in ('0', '1'):
        raise ValueError('TRIAGE_UNIFIED_AUTO_GATE must be 0 or 1')
    return raw == '1'


def gate_ratio_from_env(default=0.5):
    raw = os.environ.get('TRIAGE_UNIFIED_GATE_RATIO')
    if raw is None:
        return default
    value = float(raw)
    if not 0 < value <= 1:
        raise ValueError('TRIAGE_UNIFIED_GATE_RATIO must be in (0, 1]')
    return value


def gate_window_from_env(default=16):
    raw = os.environ.get('TRIAGE_UNIFIED_GATE_WINDOW')
    if raw is None:
        return default
    if not raw.isdigit() or not 4 <= int(raw) <= 256:
        raise ValueError('TRIAGE_UNIFIED_GATE_WINDOW must be 4..256')
    return int(raw)


def fuzzy_mode_from_env(default='current'):
    """Experiment selector, read before the service builds its shared cache."""
    mode = os.environ.get('TRIAGE_UNIFIED_FUZZY', default)
    if mode not in UnifiedContextCache.FUZZY_MODES:
        raise ValueError('TRIAGE_UNIFIED_FUZZY must be one of %s'
                         % (UnifiedContextCache.FUZZY_MODES,))
    return mode


def per_request_gate_from_env(default=True):
    raw = os.environ.get('TRIAGE_UNIFIED_PER_REQUEST_GATE')
    if raw is None:
        return default
    if raw not in ('0', '1'):
        raise ValueError('TRIAGE_UNIFIED_PER_REQUEST_GATE must be 0 or 1')
    return raw == '1'


def multi_suffix_from_env(default=True):
    raw = os.environ.get('TRIAGE_CACHE_MULTI_SUFFIX')
    if raw is None:
        return default
    if raw not in ('0', '1'):
        raise ValueError('TRIAGE_CACHE_MULTI_SUFFIX must be 0 or 1')
    return raw == '1'


def progress_from_env(default=True):
    raw = os.environ.get('TRIAGE_CACHE_PROGRESS')
    if raw is None:
        return default
    if raw not in ('0', '1'):
        raise ValueError('TRIAGE_CACHE_PROGRESS must be 0 or 1')
    return raw == '1'


def unique_off_from_env(default=True):
    raw = os.environ.get('TRIAGE_UNIFIED_UNIQUE_OFF')
    if raw is None:
        return default
    if raw not in ('0', '1'):
        raise ValueError('TRIAGE_UNIFIED_UNIQUE_OFF must be 0 or 1')
    return raw == '1'


def repeat_min_from_env(default=16):
    raw = os.environ.get('TRIAGE_UNIFIED_REPEAT_MIN')
    if raw is None:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError('TRIAGE_UNIFIED_REPEAT_MIN must be >= 0')
    return value


def repeat_gate_from_env(default=True):
    raw = os.environ.get('TRIAGE_UNIFIED_REPEAT_GATE')
    if raw is None:
        return default
    if raw not in ('0', '1'):
        raise ValueError('TRIAGE_UNIFIED_REPEAT_GATE must be 0 or 1')
    return raw == '1'


def min_width_from_env(default=None):
    raw = os.environ.get('TRIAGE_UNIFIED_MIN_WIDTH')
    if raw is None:
        return default
    value = int(raw)
    if not 2 <= value <= 12:
        raise ValueError('TRIAGE_UNIFIED_MIN_WIDTH must be in [2, 12]')
    return value


def prompt_width_from_env(default=8):
    raw = os.environ.get('TRIAGE_UNIFIED_FUZZY_PROMPT_WIDTH')
    if raw is None:
        return default
    value = int(raw)
    if not 2 <= value <= 16:
        raise ValueError('TRIAGE_UNIFIED_FUZZY_PROMPT_WIDTH must be in [2, 16]')
    return value


def fuzzy_threshold_from_env(default=0.75):
    raw = os.environ.get('TRIAGE_UNIFIED_FUZZY_THRESHOLD')
    if raw is None:
        return default
    value = float(raw)
    if not 0 < value <= 1:
        raise ValueError('TRIAGE_UNIFIED_FUZZY_THRESHOLD must be in (0, 1]')
    return value


class UnifiedContextCache:
    """Committed-token proposals for both fixed K3 and immutable growth Plans.

    One service-local owner for prompt replay, short suffixes and lexical bands.
    These are token suggestions only; target verification and KV remain private.
    """
    PROMPT_CAPACITY = 4096
    WINDOW_CAPACITY = 65536
    BAND_CAPACITY = 32768
    CURSOR_CAPACITY = 512
    GRAM_CAPACITY = 1 << 16

    FUZZY_MODES = ('none', 'current', 'content', 'suffix1', 'both', 'prompt',
                   'prompt+content', 'full', 'full-copy-first')
    # Ordered draft sources per mode; the first layer that fires wins.
    PIPELINES = {
        'none': (),
        'current': ('fuzzy',),
        'content': ('content',),
        'suffix1': ('suffix1',),
        'both': ('suffix1', 'content'),
        'prompt': ('lookup',),
        'prompt+content': ('lookup', 'content'),
        'full': ('suffix1', 'lookup', 'content'),
        'full-copy-first': ('lookup', 'suffix1', 'content'),
    }

    def __init__(self, *, window=32, min_similarity=0.25, cross_request_only=False,
                 capacity=4096, max_token_slots=262144,
                 min_suffix=4, max_suffix=12, horizon=7,
                 bands=4, suffix_widths=(4, 8, 12),
                 fuzzy_mode='current', fuzzy_threshold=0.75, fuzzy_capacity=16384,
                 fuzzy_bucket=64, fuzzy_tail_gate=True, fuzzy_rank='suffix',
                 prompt_width=8, suffix_tail=8, suffix_mismatches=1,
                 auto_gate=False, gate_min=0.75, gate_window=16, gate_probe=8,
                 gate_ratio=0.5, min_width=None, repeat_gate=False,
                 repeat_ratio=0.3, repeat_min=16, repeat_gram=12, repeat_probes=32,
                 unique_off=False, per_request_gate=False, per_request_ratio=0.6,
                 multi_suffix=False, suffix_candidates=8,
                 progress=False, progress_eta=0.5, progress_prior=1.0):
        if (type(horizon) is not int or horizon < 7
                or not 0 < min_suffix <= max_suffix):
            raise ValueError('invalid suffix bounds')
        if type(bands) is not int or bands < 1 or window % bands:
            raise ValueError('window must be divisible by bands')
        if type(min_similarity) not in (int, float) or not 0 < min_similarity <= 1:
            raise ValueError('invalid band similarity')
        if fuzzy_mode not in self.FUZZY_MODES:
            raise ValueError('invalid fuzzy mode')
        if (type(fuzzy_threshold) not in (int, float)
                or not 0 < fuzzy_threshold <= 1):
            raise ValueError('invalid fuzzy threshold')
        if (type(fuzzy_capacity) is not int or fuzzy_capacity < 1
                or type(fuzzy_bucket) is not int or fuzzy_bucket < 1):
            raise ValueError('invalid fuzzy capacity')
        if type(fuzzy_tail_gate) is not bool:
            raise ValueError('invalid fuzzy tail gate')
        if fuzzy_rank not in ('suffix', 'recent'):
            raise ValueError('invalid fuzzy rank rule')
        if type(prompt_width) is not int or not 2 <= prompt_width <= 16:
            raise ValueError('invalid prompt lookup width')
        if (type(suffix_tail) is not int or not 4 <= suffix_tail <= window
                or type(suffix_mismatches) is not int
                or not 0 <= suffix_mismatches < suffix_tail):
            raise ValueError('invalid tolerant suffix bounds')
        self.window, self.bands = window, bands
        self.position_bucket, self.band_stride = 32, 64
        self.band_width = window // bands
        self.min_similarity = min_similarity
        self.min_suffix, self.max_suffix, self.horizon = min_suffix, max_suffix, horizon
        self.suffix_widths = tuple(sorted(set(suffix_widths)))
        if (not self.suffix_widths or any(type(w) is not int or not min_suffix <= w <= max_suffix
                                          for w in self.suffix_widths)):
            raise ValueError('invalid suffix width set')
        self.cross_request_only = cross_request_only
        # Retain the legacy diagnostic surface, not its write/lookup workload.
        self.fuzzy = SimilarContextCache(window=window, min_similarity=0.9,
                                         cross_request_only=cross_request_only,
                                         max_entries=capacity, max_token_slots=max_token_slots)
        self.prompt_index, self.window_index, self.band_index = OrderedDict(), OrderedDict(), OrderedDict()
        if (type(multi_suffix) is not bool or type(suffix_candidates) is not int
                or not 1 <= suffix_candidates <= 8
                or type(max_token_slots) is not int or max_token_slots < 7):
            raise ValueError('invalid multi suffix bounds')
        self.multi_suffix, self.suffix_candidates = multi_suffix, suffix_candidates
        self.suffix_token_budget = max_token_slots
        self.suffix_token_slots = self.suffix_candidate_count = 0
        self.suffix_sequence = self.suffix_evictions = 0
        if (type(progress) is not bool or type(progress_eta) not in (int, float)
                or not 0 < progress_eta <= 1
                or type(progress_prior) not in (int, float) or progress_prior < 0):
            raise ValueError('invalid progress bounds')
        # A progress record lives inside the per-candidate bucket, so the legacy
        # single-tuple suffix index has nowhere to keep one.
        self.progress = progress and multi_suffix
        self.progress_eta = float(progress_eta)
        self.progress_prior = float(progress_prior)
        # Incremental mean for the first 1/eta samples, fixed-weight EWMA after.
        self.progress_hold = max(1, round(1/self.progress_eta))
        self.progress_updates = self.progress_skips = self.progress_misses = 0
        self.progress_total = 0.0
        self.progress_samples = 0
        self.source = dict(prompt=0, window=0, fuzzy=0, content=0, suffix1=0, lookup=0)
        self.width_hits = {w: 0 for w in self.suffix_widths}
        self.band_hits = {b: 0 for b in range(bands)}
        self.fuzzy_comparisons = 0
        self._seen, self._band_seen = OrderedDict(), OrderedDict()
        # Fuzzy v2: real context comparison over an index built at every commit.
        self.fuzzy_mode = fuzzy_mode
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_capacity, self.fuzzy_bucket = fuzzy_capacity, fuzzy_bucket
        self.fuzzy_tail_gate = fuzzy_tail_gate
        self.fuzzy_rank = fuzzy_rank
        self.prompt_width = prompt_width
        self.suffix_tail, self.suffix_mismatches = suffix_tail, suffix_mismatches
        self.min_width = self.suffix_widths[0] if min_width is None else min_width
        if type(repeat_gate) is not bool:
            raise ValueError('invalid repeat gate flag')
        if type(repeat_ratio) not in (int, float) or not 0 <= repeat_ratio <= 1:
            raise ValueError('invalid repeat ratio')
        if type(repeat_min) is not int or repeat_min < 0:
            raise ValueError('invalid repeat sample size')
        self.repeat_gate = repeat_gate
        if type(repeat_gram) is not int or not 4 <= repeat_gram <= 64:
            raise ValueError('invalid repeat gram length')
        if type(repeat_probes) is not int or repeat_probes < 1:
            raise ValueError('invalid repeat probe count')
        if type(unique_off) is not bool:
            raise ValueError('invalid unique-traffic switch')
        if type(per_request_gate) is not bool:
            raise ValueError('invalid per-request gate switch')
        if type(per_request_ratio) not in (int, float) or not 0 < per_request_ratio <= 1:
            raise ValueError('invalid per-request ratio')
        self.repeat_ratio, self.repeat_min = repeat_ratio, repeat_min
        self.repeat_gram, self.repeat_probes = repeat_gram, repeat_probes
        self.unique_off = unique_off
        self.per_request_gate = per_request_gate
        self.per_request_ratio = per_request_ratio
        self.reuse_rid = OrderedDict()
        self.gram_index = OrderedDict()
        self.prompt_seen = 0
        self.prompt_repeats = 0
        self.seeded_records = self.seeded_tokens = 0
        self.seeded_prompts = OrderedDict()
        # Acceptance counters for drafts proposed from a seed file.
        self.seed_drafts = self.seed_accepted = 0
        self._counted = OrderedDict()
        self._prompt_known = OrderedDict()
        if self.min_width not in self.suffix_widths:
            raise ValueError('min width must be one of the suffix widths')
        if type(auto_gate) is not bool:
            raise ValueError('invalid auto gate flag')
        if type(gate_window) is not int or gate_window < 4:
            raise ValueError('invalid gate window')
        if type(gate_min) not in (int, float) or gate_min < 0:
            raise ValueError('invalid gate threshold')
        if type(gate_probe) is not int or gate_probe < 1:
            raise ValueError('invalid gate probe interval')
        if type(gate_ratio) not in (int, float) or not 0 <= gate_ratio <= 1:
            raise ValueError('invalid gate ratio')
        self.auto_gate, self.gate_min, self.gate_window = auto_gate, gate_min, gate_window
        self._repeat_gate_flag = repeat_gate
        self.gate_ratio = gate_ratio
        self.donors = OrderedDict()
        self.donor_keys = {}
        self.content_index, self.suffix1_keys = {}, {}
        self.fuzzy_candidates = 0
        self.donor_seq = 0
        # Per-layer acceptance: a served draft is settled on the next commit.
        self.reserved = {}
        self._offer_issuer = object()
        self.layer_stats = {name: dict(drafts=0, accepted=0, rejected=0)
                            for name in ('prompt', 'window', 'content', 'suffix1', 'fuzzy', 'lookup')}
        self.width_stats = {width: dict(drafts=0, accepted=0) for width in self.suffix_widths}
        self.width_history = {width: deque(maxlen=self.gate_window) for width in self.suffix_widths}
        self.width_enabled = {width: True for width in self.suffix_widths}
        self.width_probe = {width: 0 for width in self.suffix_widths}
        self.gate_history = {name: deque(maxlen=self.gate_window) for name in self.layer_stats}
        self.gate_enabled = {name: True for name in self.layer_stats}
        # A disabled layer must keep probing, otherwise it can never recover.
        self.gate_probe = {name: 0 for name in self.layer_stats}
        self.gate_probe_every = gate_probe
        self._fuzzy_seen = OrderedDict()

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'fuzzy'), name)

    @staticmethod
    def _want(want):
        # Cache entries carry at most horizon=7 tokens, so K>7 is a miss and
        # falls back to the edge. A wider declared space may include 4..7.
        if type(want) is not int or want not in KS:
            raise ValueError('cache draft length must be in the declared K space')

    @staticmethod
    def _remember(index, key, value, capacity):
        index[key] = value
        index.move_to_end(key)
        while len(index) > capacity:
            index.popitem(last=False)

    def _remember_prompt(self, prompt, generated):
        old = self.prompt_index.get(prompt)
        # A sibling just starting must not erase a matching longer donor replay.
        if old is not None and len(old[1]) >= len(generated) and old[1][:len(generated)] == generated:
            return
        self._remember(self.prompt_index, prompt, (prompt, generated), self.PROMPT_CAPACITY)

    def seed_health(self):
        """(drafts, accepted tokens) for drafts that came from the seed file."""
        return self.seed_drafts, self.seed_accepted

    def prompt_origin(self, prompt):
        """'seed' when this prompt's replay came from a file, else 'live'.

        The tag is sticky: a live commit that later overwrites the replay does
        not erase the fact that the prompt was seeded, because the gate needs to
        price the origin a round was *proposed* from, and a seeded prompt keeps
        being proposed before its first live confirmation.
        """
        return 'seed' if prompt in self.seeded_prompts else 'live'

    def observe_progress(self, prompt, generated, rid=None):
        generated = tuple(token_ids(generated)) if generated else ()
        start = self._seen.get(rid, self.min_suffix)
        end = max(0, len(generated) - self.horizon + 1)
        if self.multi_suffix:
            prompt = tuple(prompt)
            for position in range(start, end):
                value = generated[position:position + self.horizon]
                context = self._suffix_context(prompt, generated, position)
                for width in self.suffix_widths:
                    if width <= position:
                        self._remember_suffix(generated[position-width:position], value, context)
        else:
            batch = {}
            for position in range(start, end):
                value = generated[position:position + self.horizon]
                for width in self.suffix_widths:
                    if width <= position:
                        batch[generated[position - width:position]] = value
            self.window_index.update(batch)
            for key in batch:
                self.window_index.move_to_end(key)
            while len(self.window_index) > self.WINDOW_CAPACITY:
                self.window_index.popitem(last=False)
        if rid is not None:
            self._remember(self._seen, rid, max(start, end), self.CURSOR_CAPACITY)

    def _suffix_context(self, prompt, generated, position):
        # Copy at most window tokens, even for a long input document.
        if position >= self.window:
            return generated[position-self.window:position]
        return prompt[-(self.window-position):] + generated[:position]

    @staticmethod
    def _suffix_entry_slots(value, metadata):
        return len(value) + sum(map(len, metadata[2]))

    def _remember_suffix(self, key, value, context):
        bucket = self.window_index.get(key)
        if bucket is None:
            bucket = self.window_index[key] = OrderedDict()
            self.suffix_token_slots += len(key)
        previous = bucket.get(value)
        if previous is None:
            support, contexts, progress = 0, (), None
            self.suffix_candidate_count += 1
        else:
            support, _, contexts, progress = previous
            self.suffix_token_slots -= self._suffix_entry_slots(value, previous)
        contexts = tuple(c for c in contexts if c != context) + (context,)
        self.suffix_sequence += 1
        metadata = (support+1, self.suffix_sequence, contexts[-2:],
                    {} if progress is None else progress)
        bucket[value] = metadata
        bucket.move_to_end(value)
        self.suffix_token_slots += self._suffix_entry_slots(value, metadata)
        if len(bucket) > self.suffix_candidates:
            evicted, meta = bucket.popitem(last=False)
            self.suffix_token_slots -= self._suffix_entry_slots(evicted, meta)
            self.suffix_candidate_count -= 1
            self.suffix_evictions += 1
        self.window_index.move_to_end(key)
        while (len(self.window_index) > self.WINDOW_CAPACITY
               or self.suffix_token_slots > self.suffix_token_budget):
            evicted, bucket = self.window_index.popitem(last=False)
            self.suffix_token_slots -= len(evicted) + sum(
                self._suffix_entry_slots(v, m) for v, m in bucket.items())
            self.suffix_candidate_count -= len(bucket)
            self.suffix_evictions += len(bucket)

    def _best_suffix(self, bucket, context, want):
        """Best continuation for this suffix, with its own progress estimate.

        Identity comes first: how much extra context the donor shares decides
        which state this really is. The measured committed progress follows,
        so two arms of the *same* state are ranked by what they actually paid
        out instead of by how often they were merely observed.
        """
        def matched_of(contexts):
            matched = 0
            for donor in contexts:
                count = 0
                for left, right in zip(reversed(context), reversed(donor)):
                    if left != right:
                        break
                    count += 1
                matched = max(matched, count)
            return matched

        def score(item):
            _, (support, recent, contexts, progress) = item
            matched = matched_of(contexts)
            if not self.progress:
                return matched, support, recent
            estimate = progress.get(want, (self.progress_prior, 0))[0]
            return matched, estimate, support, recent

        eligible = ((v, m) for v, m in bucket.items() if len(v) >= want)
        best = max(eligible, key=score, default=None)
        if best is None:
            return None, self.progress_prior, 0
        if not self.progress:
            return best[0], self.progress_prior, 0
        estimate, samples = best[1][3].get(want, (self.progress_prior, 0))
        return best[0], estimate, samples

    def _progress_update(self, key, value, want, committed):
        """One EWMA step on the (suffix, continuation) record.

        ``committed`` is what the round actually committed: the verified draft
        prefix plus the one token every round delivers anyway. It is keyed by
        ``want`` because a 3-token draft and a 7-token draft are different
        actions with different ranges, and one shared average would bias both.
        """
        bucket = self.window_index.get(key)
        entry = None if bucket is None else bucket.get(value)
        if entry is None:
            # Evicted between the serve and the commit. There is nowhere to put
            # the evidence, and inventing a record would score a candidate that
            # this cache no longer retains.
            self.progress_misses += 1
            return
        progress = entry[3]
        estimate, samples = progress.get(want, (self.progress_prior, 0))
        samples += 1
        estimate += (committed - estimate)/min(samples, self.progress_hold)
        progress[want] = (estimate, samples)
        self.progress_updates += 1
        self.progress_total += estimate
        self.progress_samples += 1

    # tools/replay_cache_cpu.py calibrated a verify round at 22.75 ms plus
    # 4.51 ms per verified token. A want-token draft therefore pays only when
    # the round commits more than (R+C*(want+1))/(R+C) tokens, which is 2 for
    # want<=6 and 3 for want=7.
    PROGRESS_ROUND_MS = 22.75
    PROGRESS_TOKEN_MS = 4.51

    def _progress_floor(self, want):
        ratio = ((self.PROGRESS_ROUND_MS + self.PROGRESS_TOKEN_MS*(want+1))
                 / (self.PROGRESS_ROUND_MS + self.PROGRESS_TOKEN_MS))
        return math.floor(ratio) + 1

    def _context(self, prompt, generated, position):
        if position >= self.window:
            context = generated[position-self.window:position]
        else:
            context = prompt[-(self.window-position):] + generated[:position]
        return context if len(context) == self.window else None

    def _index_bands(self, rid, prompt, generated):
        end = max(1, len(generated) - 3 + 1)
        # Revisit a trailing short continuation only until it reaches horizon.
        start = max(1, self._band_seen.get(rid, 1) - self.horizon + 1)
        start = 1 + ((start - 1 + self.band_stride - 1) // self.band_stride) * self.band_stride
        for position in range(start, end, self.band_stride):
            context = self._context(prompt, generated, position)
            if context is None:
                continue
            following = generated[position:position+self.horizon]
            identity = (rid, position, context)
            for band in range(self.bands):
                key = (position//self.position_bucket, context[-1], band,
                       context[band*self.band_width:(band+1)*self.band_width])
                bucket = self.band_index.setdefault(key, OrderedDict())
                self._remember(bucket, identity, following, 8)
                self.band_index.move_to_end(key)
        while len(self.band_index) > self.BAND_CAPACITY:
            self.band_index.popitem(last=False)
        if rid is not None:
            self._remember(self._band_seen, rid, end, self.CURSOR_CAPACITY)

    def observe(self, rid, prompt, generated):
        prompt = tuple(token_ids(prompt))
        generated = tuple(token_ids(generated)) if generated else ()
        self._settle(str(rid), generated)
        if not generated:
            return
        # Count each request once and judge repetition by how much of its
        # prompt was already seen, measured with the same n-gram the long
        # suffix layer uses.
        key = str(rid)
        if key not in self._counted:
            self._remember(self._counted, key, True, self.CURSOR_CAPACITY)
            self.prompt_seen += 1
            grams = self._prompt_grams(prompt)
            known = sum(1 for gram in grams if gram in self.gram_index)
            reuse = bool(grams) and known >= 0.5*len(grams)
            if reuse:
                self.prompt_repeats += 1
            self._remember(self.reuse_rid, key, reuse, self.CURSOR_CAPACITY)
            for gram in grams:
                self._remember(self.gram_index, gram, True, self.GRAM_CAPACITY)
        self._remember_prompt(prompt, generated)
        # A live commit is the first evidence that this prompt's continuation is
        # real, so the seed tag ends here: from now on the replay is ours.
        self.seeded_prompts.pop(prompt, None)
        self.observe_progress(prompt, generated, rid)
        if self.fuzzy_mode == 'current':
            self._index_bands(rid, prompt, generated)
        else:
            self._index_fuzzy(rid, prompt, generated)

    def _prompt_grams(self, prompt):
        width, stride = self.repeat_gram, 1
        span = len(prompt) - width + 1
        if span < 1:
            return ()
        stride = max(1, span//self.repeat_probes)
        return [hash(prompt[pos:pos+width]) for pos in range(0, span, stride)]

    def short_suffixes_harmful(self):
        """Restrict short suffixes until observed prompt repetition supports them.

        No observations or no repeated prompts enable the restriction immediately.
        After repeat_min requests, compare the repetition rate with repeat_ratio.
        """
        if self.prompt_seen == 0 or self.prompt_repeats == 0:
            return True
        if self.prompt_seen < self.repeat_min:
            return False
        return self.prompt_repeats/self.prompt_seen < self.repeat_ratio

    def _workload_repetitive(self):
        if self.prompt_seen < self.repeat_min:
            return False
        return self.prompt_repeats/self.prompt_seen >= self.per_request_ratio

    def no_reuse_observed(self):
        """True once enough requests have arrived with no repeated prompt."""
        return self.prompt_seen >= self.repeat_min and self.prompt_repeats == 0

    def _gate_ok(self, layer):
        if not self.auto_gate or self.gate_enabled.get(layer, True):
            return True
        # A disabled layer keeps probing, otherwise it can never recover.
        self.gate_probe[layer] += 1
        return self.gate_probe[layer] % self.gate_probe_every == 0

    def _settle(self, rid, generated):
        pending = self.reserved.pop(rid, None)
        if pending is None:
            return
        layer, position, want, width, origin, served = pending
        delivered = max(0, len(generated) - position)
        accepted = delivered - 1 if delivered > 1 else 0
        accepted = min(accepted, want)
        if (self.progress and served is not None and accepted
                and tuple(generated[position:position+accepted]) != served[1][:accepted]):
            # Something else served this round, so the accepted count is not
            # this candidate's evidence. Drop the sample instead of grading it.
            served = None
            self.progress_misses += 1
        stats = self.layer_stats[layer]
        stats['drafts'] += 1
        stats['accepted'] += accepted
        if origin == 'seed':
            self.seed_drafts += 1
            self.seed_accepted += accepted
        if not accepted:
            stats['rejected'] += 1
        if self.auto_gate and width is not None and want:
            self.width_stats[width]['drafts'] += 1
            self.width_stats[width]['accepted'] += accepted
            history = self.width_history[width]
            history.append(accepted/want)
            if len(history) >= self.gate_window:
                mean = sum(history)/len(history)
                if mean < self.gate_ratio:
                    self.width_enabled[width] = False
                elif mean >= min(1.0, 1.25*self.gate_ratio):
                    self.width_enabled[width] = True
        if self.auto_gate and layer in self.gate_history and want:
            history = self.gate_history[layer]
            history.append(accepted/want)
            if len(history) >= self.gate_window:
                mean = sum(history)/len(history)
                if mean < self.gate_ratio:
                    self.gate_enabled[layer] = False
                elif mean >= min(1.0, 1.25*self.gate_ratio):
                    self.gate_enabled[layer] = True
        if self.progress and served is not None:
            self._progress_update(served[0], served[1], want, accepted + 1)

    def _reserve(self, rid, layer, position, want, width=None, origin='live', served=None):
        if rid is None:
            return
        self.reserved[str(rid)] = (layer, position, want, width, origin, served)

    def _prompt_lookup(self, prompt, generated, want, limit=4096):
        """LLMA-style copy: draft what followed this n-gram earlier in the prompt."""
        window = prompt + generated
        width = self.prompt_width
        if len(window) < width + 1:
            return []
        tail = window[-width:]
        floor = max(0, len(window) - limit)
        for start in range(len(window) - width - 1, floor - 1, -1):
            if window[start:start+width] != tail:
                continue
            following = window[start+width:start+width+want]
            if len(following) == want:
                return list(following)
        return []

    def _fuzzy_erase(self, donor):
        for index, keys in ((self.content_index, self.donor_keys.pop(donor, ())),
                            (self.content_index, self.suffix1_keys.pop(donor, ()))):
            for key in keys:
                bucket = index.get(key)
                if bucket is None:
                    continue
                bucket.pop(donor, None)
                if not bucket:
                    index.pop(key, None)
        self.donors.pop(donor, None)

    def _fuzzy_shingles(self, context):
        # Offset-tagged keys: an 8-token block must sit at the same distance
        # from the tail in both contexts, otherwise shifted contexts vote.
        width = self.band_width
        return tuple((i, context[i*width:(i+1)*width]) for i in range(self.bands))

    def _fuzzy_suffix_keys(self, tail):
        # Pigeonhole: with at most m mismatches, splitting the tail into m+1
        # disjoint chunks guarantees at least one chunk is exact.
        pieces = self.suffix_mismatches + 1
        size = len(tail)//pieces
        keys, start = [], 0
        for index in range(pieces):
            end = len(tail) if index == pieces-1 else start+size
            keys.append(('s%d' % index, tail[start:end]))
            start = end
        return tuple(keys)

    def _fuzzy_store(self, donor, context, following):
        if len(following) < 3:
            return
        old = self.donors.get(donor)
        if old is not None:
            self.donors[donor] = (old[0], following, old[2])
            return
        self.donor_seq += 1
        self.donors[donor] = (context, following, self.donor_seq)
        keys = []
        if self.fuzzy_mode in ('content', 'both', 'prompt+content',
                               'full', 'full-copy-first'):
            for key in self._fuzzy_shingles(context):
                bucket = self.content_index.setdefault(key, OrderedDict())
                bucket[donor] = None
                while len(bucket) > self.fuzzy_bucket:
                    bucket.popitem(last=False)
                keys.append(key)
            self.donor_keys[donor] = tuple(keys)
        if self.fuzzy_mode in ('suffix1', 'both', 'full', 'full-copy-first'):
            keys = self._fuzzy_suffix_keys(context[-self.suffix_tail:])
            for key in keys:
                bucket = self.content_index.setdefault(key, OrderedDict())
                bucket[donor] = None
                while len(bucket) > self.fuzzy_bucket:
                    bucket.popitem(last=False)
            self.suffix1_keys[donor] = keys
        while len(self.donors) > self.fuzzy_capacity:
            self._fuzzy_erase(next(iter(self.donors)))

    def _index_fuzzy(self, rid, prompt, generated):
        start = max(1, self._fuzzy_seen.get(rid, 1))
        end = len(generated) - 2
        for position in range(start, max(start, end)):
            context = self._context(prompt, generated, position)
            if context is None:
                continue
            self._fuzzy_store((rid, position), context,
                              generated[position:position+self.horizon])
        # A donor indexed near the tail gains following tokens as they commit.
        for position in range(max(1, end-self.horizon+1), max(1, end)):
            donor = (rid, position)
            if donor in self.donors:
                old = self.donors[donor]
                self.donors[donor] = (old[0],
                                      generated[position:position+self.horizon],
                                      old[2])
        if rid is not None:
            self._remember(self._fuzzy_seen, rid, max(start, end), self.CURSOR_CAPACITY)

    @staticmethod
    def _common_suffix(left, right):
        matched = 0
        for first, second in zip(reversed(left), reversed(right)):
            if first != second:
                break
            matched += 1
        return matched

    def _fuzzy_rank_of(self, stored, context):
        suffix = self._common_suffix(stored[0], context)
        return (suffix, stored[2]) if self.fuzzy_rank == 'suffix' else (stored[2], suffix)

    def _fuzzy_votes(self, keys):
        votes = {}
        for key in keys:
            bucket = self.content_index.get(key)
            if not bucket:
                continue
            recent = list(islice(reversed(bucket), self.fuzzy_bucket))
            for donor in recent:
                votes[donor] = votes.get(donor, 0) + 1
        return votes

    def _content_proposal(self, prompt, generated, rid, want, *, record=True):
        context = self._context(prompt, generated, len(generated))
        if context is None:
            return []
        votes = self._fuzzy_votes(self._fuzzy_shingles(context))
        best, best_rank = None, None
        for donor, matched in votes.items():
            if record:
                self.fuzzy_candidates += 1
            score = matched/self.bands
            if score+1e-12 < self.fuzzy_threshold:
                continue
            if self.cross_request_only and donor[0] == rid:
                continue
            stored = self.donors[donor]
            if (self.fuzzy_tail_gate
                    and sum(left != right for left, right in
                            zip(stored[0][-8:], context[-8:])) > 1):
                continue
            if len(stored[1]) < want:
                continue
            # Explicit tie rule; never dict iteration order.
            rank = self._fuzzy_rank_of(stored, context)
            if best_rank is None or rank > best_rank:
                best, best_rank = stored[1], rank
        return list(best[:want]) if best is not None else []

    def _suffix1_proposal(self, prompt, generated, rid, want, *, record=True):
        context = self._context(prompt, generated, len(generated))
        if context is None:
            return []
        tail = context[-self.suffix_tail:]
        votes = self._fuzzy_votes(self._fuzzy_suffix_keys(tail))
        best, best_rank = None, None
        for donor in votes:
            if record:
                self.fuzzy_candidates += 1
            if self.cross_request_only and donor[0] == rid:
                continue
            stored = self.donors[donor]
            if len(stored[1]) < want:
                continue
            if sum(left != right for left, right in
                   zip(stored[0][-self.suffix_tail:], tail)) > self.suffix_mismatches:
                continue
            rank = self._fuzzy_rank_of(stored, context)
            if best_rank is None or rank > best_rank:
                best, best_rank = stored[1], rank
        return list(best[:want]) if best is not None else []

    def observe_committed(self, request):
        self.observe(request.rid, request.prompt, request.generated)

    def observe_request(self, request):
        # The argument is already target-committed; never feed draft callbacks here.
        self.observe_committed(request)

    def seed(self, prompt, generated):
        """Fill all three tiers from committed tokens; never counts a request.

        Only the startup loader (``engines.common.cache_seed``) calls this. The
        n-gram index is filled too, so a later request whose prompt was seeded
        is recognised as reuse by the same evidence live traffic would leave.
        Traffic counters stay untouched: a seed is data, not a served request.
        """
        prompt = tuple(token_ids(prompt))
        generated = tuple(token_ids(generated))
        rid = 'seed:%d' % self.seeded_records
        self._remember_prompt(prompt, generated)
        self.observe_progress(prompt, generated, rid)
        if self.fuzzy_mode == 'current':
            self._index_bands(rid, prompt, generated)
        else:
            self._index_fuzzy(rid, prompt, generated)
        for gram in self._prompt_grams(prompt):
            self._remember(self.gram_index, gram, True, self.GRAM_CAPACITY)
        self._remember(self.seeded_prompts, prompt, True, self.PROMPT_CAPACITY)
        self.seeded_records += 1
        self.seeded_tokens += len(prompt)+len(generated)

    def warm(self):
        return bool(self.prompt_index or self.window_index or self.band_index)

    def _suffix_proposal(self, generated, want=DRAFT_TOKENS, *, record=True, min_width=None, prompt=()):
        floor = self.min_width if min_width is None else max(self.min_width, min_width)
        for width in reversed(self.suffix_widths):
            if width < floor:
                continue
            if self.auto_gate and not self.width_enabled[width]:
                self.width_probe[width] += 1
                if self.width_probe[width] % self.gate_probe_every:
                    continue
            if len(generated) < width:
                continue
            key = generated[-width:]
            bucket = self.window_index.get(key)
            if not bucket:
                continue
            if self.multi_suffix:
                value, estimate, samples = self._best_suffix(
                    bucket, self._suffix_context(prompt, generated, len(generated)), want)
            else:
                value = tuple(bucket) if len(bucket) >= want else None
                estimate, samples = self.progress_prior, 0
            if value is None or len(value) < want:
                continue
            if self.progress and samples and estimate < self._progress_floor(want):
                # Tried, measured and still a net loss: fall through to a
                # shorter suffix. Only evidence may silence a key, so an
                # untried suffix always keeps its first serve.
                if record:
                    self.progress_skips += 1
                continue
            if record:
                self.window_index.move_to_end(key)
                self.width_hits[width] += 1
            return list(value[:want]), width, (key, value)
        return [], None, None

    def _band_proposal(self, prompt, generated, rid, want=DRAFT_TOKENS, *, record=True):
        if not generated:
            return []
        context = self._context(prompt, generated, len(generated))
        if context is None:
            return []
        candidates = {}
        for band in range(self.bands):
            key = (len(generated)//self.position_bucket, context[-1], band,
                   context[band*self.band_width:(band+1)*self.band_width])
            for identity, following in self.band_index.get(key, {}).items():
                if len(following) < want or self.cross_request_only and identity[0] == rid:
                    continue
                # Each donor-context can earn at most one vote from each band.
                pair = (identity, following)
                candidates[pair] = candidates.get(pair, 0) + 1
                if record:
                    self.band_hits[band] += 1
        best, best_score = None, -1.0
        for (_, following), matched in candidates.items():
            if record:
                self.fuzzy_comparisons += 1
            score = matched / self.bands
            if score+1e-12 >= self.min_similarity and score > best_score:
                best, best_score = following, score
        return list(best[:want]) if best is not None else []

    def _lookup(self, prompt, generated, rid, want, *, record, detail=None):
        if (self.per_request_gate and self.fuzzy_mode != 'none'
                and self.prompt_seen >= self.repeat_min
                and not self._workload_repetitive()
                and not self.reuse_rid.get(str(rid))):
            # A mostly-repeating workload keeps the global verdict, and during
            # the sampling window the cache still serves long suffixes: only a
            # request that is judged unique after enough evidence steps aside.
            return [], None
        elif (self.unique_off and self.fuzzy_mode != 'none'
                and self.no_reuse_observed()):
            return [], None
        entry = (self.prompt_index.get(prompt)
                 if self.fuzzy_mode != 'none' and self._gate_ok('prompt') else None)
        if entry is not None:
            stored_prompt, replay = entry
            if (stored_prompt == prompt and replay[:len(generated)] == generated
                    and len(replay)-len(generated) >= want):
                if record:
                    self.prompt_index.move_to_end(prompt)
                    self.source['prompt'] += 1
                if detail is not None:
                    detail.update(layer='prompt', width=None)
                return list(replay[len(generated):len(generated)+want]), 'prompt'
        # A short suffix is only informative when this prompt was already known
        # before the request started: otherwise it matches template scaffolding.
        # The verdict is fixed on the first lookup of a request, since observe()
        # registers the prompt itself right after.
        floor = None
        if self.per_request_gate and self.prompt_seen >= self.repeat_min:
            if not self._workload_repetitive() and not self.reuse_rid.get(str(rid)):
                floor = 4
        elif self.repeat_gate and self.short_suffixes_harmful():
            floor = 4
        tokens, width, served = ((self._suffix_proposal(generated, want, record=record, min_width=floor, prompt=prompt)
                          if self.fuzzy_mode != 'none' and self._gate_ok('window')
                          else ([], None, None)))
        if tokens:
            if detail is not None:
                detail.update(layer='window', width=width)
            if record:
                self.source['window'] += 1
                self._reserve(rid, 'window', len(generated), want, width=width,
                              served=served)
                return tokens, 'window'
            return tokens, 'window'
        for layer in self.PIPELINES[self.fuzzy_mode]:
            if not self._gate_ok(layer):
                continue
            if layer == 'lookup':
                tokens = self._prompt_lookup(prompt, generated, want)
            elif layer == 'suffix1':
                tokens = self._suffix1_proposal(prompt, generated, rid, want, record=record)
            elif layer == 'content':
                tokens = self._content_proposal(prompt, generated, rid, want, record=record)
            else:
                tokens = self._band_proposal(prompt, generated, rid, want, record=record)
            if tokens:
                if detail is not None:
                    detail.update(layer=layer, width=None)
                if record:
                    self.source[layer] += 1
                return tokens, layer
        return [], None

    def propose_committed(self, request, want=DRAFT_TOKENS):
        return self.propose_committed_with_source(request, want)[0]

    def preview_committed(self, request, want=DRAFT_TOKENS):
        """Copy a real offer without reserving, recording a hit or moving LRU.

        Auto-gate's legacy periodic probes mutate on lookup; that optional mode
        requires a separate probe clock and is not silently emulated here.
        """
        from .cache_offer import CacheOffer
        self._want(want)
        if self.auto_gate:raise ValueError('CACHE_OFFER_AUTO_GATE_UNSUPPORTED')
        if not self.warm():return None
        prompt=tuple(token_ids(request.prompt))
        generated=tuple(token_ids(request.generated)) if request.generated else ()
        detail={}
        tokens,kind=self._lookup(prompt,generated,request.rid,want,record=False,detail=detail)
        if not tokens:return None
        if len(tokens)!=want:raise ValueError('CACHE_OFFER_INCOMPLETE')
        origin='seed' if kind=='prompt' and prompt in self.seeded_prompts else 'live'
        return CacheOffer(str(request.rid),prompt,generated,tuple(tokens),
                          'exact_prefix' if kind=='prompt' else 'suffix',
                          detail['layer'],detail.get('width'),origin,self._offer_issuer)

    def claim_offer(self, request, offer, *, count):
        """Select the held tokens once; never look them up again after planning."""
        from .cache_offer import CacheOffer
        self._want(count)
        if (type(offer) is not CacheOffer or offer.issuer is not self._offer_issuer
                or str(request.rid)!=offer.rid or tuple(request.prompt)!=offer.prompt
                or tuple(request.generated)!=offer.generated or count>len(offer.tokens)):
            raise ValueError('CACHE_OFFER_BINDING')
        if offer.rid in self.reserved:raise ValueError('CACHE_OFFER_ALREADY_CLAIMED')
        # A cache index may have been evicted since preview. The copied proposal
        # is still target-verifiable; only touch LRU when its old entry survives.
        served=None
        if offer.layer=='prompt':
            entry=self.prompt_index.get(offer.prompt)
            position=len(offer.generated)
            if entry is not None and entry[1][position:position+len(offer.tokens)]==offer.tokens:
                self.prompt_index.move_to_end(offer.prompt)
        elif offer.layer=='window':
            key=offer.generated[-offer.width:]
            entry=self.window_index.get(key)
            if entry is not None:
                if self.multi_suffix:
                    served=next((value for value in entry
                                 if value[:len(offer.tokens)]==offer.tokens),None)
                elif tuple(entry[:len(offer.tokens)])==offer.tokens:
                    served=tuple(entry)
            if served is not None:
                self.window_index.move_to_end(key)
            self.width_hits[offer.width]+=1
        self.source[offer.layer]+=1
        self._reserve(offer.rid,offer.layer,len(offer.generated),count,
                      width=offer.width,origin=offer.origin,
                      served=(key,served) if served is not None else None)
        return list(offer.tokens[:count])

    def propose_committed_with_source(self, request, want=DRAFT_TOKENS, detail=None):
        """Return (tokens, source), where source is the native packer taxonomy.

        A complete prompt replay keeps its historical exact_prefix identity;
        suffix and band matches remain suffix proposals. The caller still binds
        these tokens to the complete committed prefix and the native verifier.
        """
        self._want(want)
        if not self.warm():
            return [], None
        prompt = tuple(token_ids(request.prompt))
        generated = tuple(token_ids(request.generated)) if request.generated else ()
        tokens, kind = self._lookup(prompt, generated, request.rid, want, record=True,
                                    detail=detail)
        if tokens and kind != 'window':      # window reserved itself with its width
            layer = kind if kind in self.layer_stats else 'fuzzy'
            origin = 'seed' if kind == 'prompt' and prompt in self.seeded_prompts else 'live'
            self._reserve(request.rid, layer, len(generated), want, origin=origin)
        source = 'exact_prefix' if kind == 'prompt' else 'suffix' if kind else None
        return tokens, source

    def propose(self, prompt, generated, *, rid=None, want=DRAFT_TOKENS):
        from types import SimpleNamespace
        return self.propose_committed(
            SimpleNamespace(prompt=prompt, generated=generated, rid=rid), want=want)

    def peek_source(self, prefix, generated, want=7, detail=None):
        """Read-only source forecast; no observation, LRU movement or counters."""
        self._want(want)
        generated = tuple(generated)
        prefix = tuple(prefix)
        prompt = prefix[:-len(generated)] if generated else prefix
        _, source = self._lookup(prompt, generated, None, want, record=False, detail=detail)
        return 'exact_prefix' if source == 'prompt' else 'suffix' if source else 'edge'

    def release(self, rid):
        # An aborted request never verified its draft, so releasing a
        # reservation is not progress evidence: only the layer counters move.
        pending = self.reserved.pop(str(rid), None)
        if pending is not None and pending[0] in self.layer_stats:
            stats = self.layer_stats[pending[0]]
            stats['drafts'] += 1
            stats['rejected'] += 1
        self._seen.pop(rid, None)
        self._band_seen.pop(rid, None)
        self._fuzzy_seen.pop(rid, None)
        self._prompt_known.pop(str(rid), None)
        self._counted.pop(str(rid), None)
        self.reuse_rid.pop(str(rid), None)
        self.fuzzy.release(rid)

    def snapshot(self):
        result = dict(self.fuzzy.snapshot())
        if self.multi_suffix:
            result.update(unified_multi_suffix=True,
                          unified_suffix_candidates=self.suffix_candidate_count,
                          unified_suffix_token_slots=self.suffix_token_slots,
                          unified_suffix_evictions=self.suffix_evictions,
                          unified_suffix_sequence=self.suffix_sequence,
                          unified_progress=self.progress,
                          unified_progress_eta=self.progress_eta,
                          unified_progress_hold=self.progress_hold,
                          unified_progress_updates=self.progress_updates,
                          unified_progress_skips=self.progress_skips,
                          unified_progress_misses=self.progress_misses,
                          unified_progress_mean=(self.progress_total/self.progress_samples
                                                 if self.progress_samples else None))
        result.update(unified_prompt_entries=len(self.prompt_index),
                      unified_window_entries=len(self.window_index),
                      unified_band_entries=len(self.band_index),
                      unified_prompt_hits=self.source['prompt'],
                      unified_window_hits=self.source['window'],
                      unified_fuzzy_hits=self.source['fuzzy'],
                      unified_width_hits=dict(self.width_hits),
                      unified_band_hits=dict(self.band_hits),
                      unified_fuzzy_comparisons=self.fuzzy_comparisons,
                      unified_cursors=len(self._seen), unified_band_cursors=len(self._band_seen),
                      unified_fuzzy_mode=self.fuzzy_mode,
                      unified_fuzzy_threshold=self.fuzzy_threshold,
                      unified_fuzzy_tail_gate=self.fuzzy_tail_gate,
                      unified_fuzzy_rank=self.fuzzy_rank,
                      unified_suffix_tail=self.suffix_tail,
                      unified_suffix_mismatches=self.suffix_mismatches,
                      unified_auto_gate=self.auto_gate,
                      unified_gate_enabled=dict(self.gate_enabled),
                      unified_gate_ratio=self.gate_ratio,
                      unified_unique_off=self.unique_off,
                      unified_per_request_gate=self.per_request_gate,
                      unified_per_request_ratio=self.per_request_ratio,
                      unified_reuse_requests=sum(1 for v in self.reuse_rid.values() if v),
                      unified_seed_records=self.seeded_records,
                      unified_seed_tokens=self.seeded_tokens,
                      unified_seed_drafts=self.seed_drafts,
                      unified_seed_accepted=self.seed_accepted,
                      unified_prompt_seen=self.prompt_seen,
                      unified_repeat_gram=self.repeat_gram,
                      unified_prompt_repeats=self.prompt_repeats,
                      unified_short_suffix_floor=4 if (self.repeat_gate and self.short_suffixes_harmful()) else self.min_width,
                      unified_min_width=self.min_width,
                      unified_donor_entries=len(self.donors),
                      unified_content_hits=self.source['content'],
                      unified_suffix1_hits=self.source['suffix1'],
                      unified_fuzzy_candidates=self.fuzzy_candidates,
                      unified_lookup_hits=self.source['lookup'],
                      unified_layer_stats={name: dict(values)
                                           for name, values in self.layer_stats.items()})
        return result

    def clear(self):
        self.prompt_index.clear()
        self.window_index.clear()
        self.suffix_token_slots = self.suffix_candidate_count = 0
        self.band_index.clear()
        self._seen.clear()
        self._band_seen.clear()
        self.donors.clear()
        self.donor_keys.clear()
        self.suffix1_keys.clear()
        self.content_index.clear()
        self._fuzzy_seen.clear()
        self._prompt_known.clear()
        self._counted.clear()
        self.reuse_rid.clear()
        self.gram_index.clear()
        self.seeded_records = self.seeded_tokens = 0
        self.seed_drafts = self.seed_accepted = 0
        self.progress_updates = self.progress_skips = self.progress_misses = 0
        self.progress_total = 0.0
        self.progress_samples = 0
        self.seeded_prompts.clear()
        self.fuzzy.clear()
