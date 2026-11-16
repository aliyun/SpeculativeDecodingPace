from collections import OrderedDict

from .policy import DRAFT_TOKENS, token_ids


class ExactPrefixCache:
    def __init__(self, max_entries=8192, max_token_slots=4_194_304, horizon=DRAFT_TOKENS):
        if (type(max_entries) is not int or max_entries < 1
                or type(max_token_slots) is not int or max_token_slots < 1):
            raise ValueError('positive exact-cache bounds required')
        if type(horizon) is not int or horizon not in (3,5,7):raise ValueError('invalid exact-cache horizon')
        self.horizon = horizon
        self.max_entries, self.max_token_slots = max_entries, max_token_slots
        self.entries, self.cursor = OrderedDict(), {}
        self.token_slots = self.conflicts = self.evictions = 0

    def _put(self, prefix, continuation):
        size = len(prefix) + self.horizon
        if size > self.max_token_slots:
            return
        if prefix in self.entries:
            previous = self.entries[prefix]
            if previous is not None and previous != continuation:
                # Observed numerical/shape differences are not an accuracy
                # proof. Stop proposing this prefix rather than choosing one.
                self.entries[prefix] = None
                self.conflicts += 1
            self.entries.move_to_end(prefix)
            return
        self.entries[prefix] = continuation
        self.token_slots += size
        while len(self.entries) > self.max_entries or self.token_slots > self.max_token_slots:
            key, _ = self.entries.popitem(last=False)
            self.token_slots -= len(key) + self.horizon
            self.evictions += 1

    def observe(self, rid, prompt, generated):
        prompt = tuple(token_ids(prompt))
        generated = tuple(token_ids(generated)) if generated else ()
        self._observe_snapshot(rid, prompt, generated)

    def observe_request(self, request):
        # Only this exact frozen type has already validated both integer tuples.
        # Mutable adapters and subclasses must keep the ordinary validation.
        from .contracts import Request
        if (type(request) is Request and type(request.prompt) is tuple
                and type(request.generated) is tuple):
            self._observe_snapshot(request.rid, request.prompt, request.generated)
        else:
            self.observe(request.rid, request.prompt, request.generated)

    def observe_committed(self, request):
        # Compatibility entry for horizon-aware proposals.
        self.observe_request(request)

    def _observe_snapshot(self, rid, prompt, generated):
        # Position 1 contains the prefill's first generated root. Only learn
        # when all three following IDs are already target-committed.
        start = self.cursor.get(rid, 1)
        end = max(1, len(generated) - self.horizon + 1)
        for position in range(start, end):
            self._put(prompt + generated[:position],
                      generated[position:position + self.horizon])
        self.cursor[rid] = max(start, end)

    def propose(self, prefix, want=None):
        key = tuple(prefix)
        if key not in self.entries:
            return []
        self.entries.move_to_end(key)
        value = self.entries[key]
        return list(value[:self.horizon if want is None else want]) if value is not None and (want is None or type(want) is int and 0 < want <= len(value)) else []

    def release(self, rid):
        self.cursor.pop(rid, None)

    def clear(self):
        self.entries.clear()
        self.cursor.clear()
        self.token_slots = 0

    def snapshot(self):
        return dict(exact_cache_entries=len(self.entries),
                    exact_cache_token_slots=self.token_slots,
                    exact_cache_cursors=len(self.cursor),
                    exact_cache_conflicts=self.conflicts,
                    exact_cache_evictions=self.evictions)
