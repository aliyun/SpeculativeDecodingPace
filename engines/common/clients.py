import http.client
from itertools import groupby
import json
import logging
import os
import threading
import time
import uuid

from .policy import DRAFT_TOKENS, CONTEXT, MAX_REQUESTS
from .root_shadow import validate_top2
from .urls import http_url


class EdgeHTTPError(RuntimeError):
    def __init__(self,status,body):
        self.status=status
        super().__init__(f'edge HTTP status {status}: {body[:200]!r}')


class JsonTransport:
    """One ordered keep-alive connection. Mutating requests are never retried."""
    def __init__(self, *, slot_erase=False):
        self.slot_erase = slot_erase
        self.connection, self.origin, self.last_end = None, None, None
        self.lock = threading.RLock()

    def close(self):
        with self.lock:
            if self.connection is not None:
                self.connection.close()
            self.connection = None

    _timing_path = os.environ.get('TRIAGE_HTTP_TIMING', '')

    def __call__(self, url, payload):
        if self._timing_path:
            _t0 = time.perf_counter()
            try:
                value = self._call_inner(url, payload)
            finally:
                # Record operation, origin and duration for each HTTP call.
                try:
                    with open(self._timing_path, 'a') as fh:
                        fh.write('%s %s %.1f\n' % (str((payload or {}).get('op')), url,
                                                   (time.perf_counter() - _t0) * 1000.0))
                except OSError:
                    pass
            return value
        return self._call_inner(url, payload)

    def _call_inner(self, url, payload):
        parsed = http_url(url, allow_query=self.slot_erase)
        erase = (self.slot_erase and parsed.query == 'action=erase'
                 and parsed.path.startswith('/slots/') and parsed.path[7:].isascii()
                 and parsed.path[7:].isdigit())
        if parsed.query and not erase:
            raise ValueError('unsupported edge HTTP query')
        origin = parsed.hostname, parsed.port or 80
        with self.lock:
            # Native keep-alive expires at 5s. Open proactively after 1s idle,
            # including benchmark boundaries, without replaying any request.
            if origin != self.origin or self.last_end is None or time.monotonic()-self.last_end >= 1:
                self.close()
            if self.connection is None:
                self.connection = http.client.HTTPConnection(*origin, timeout=60)
                self.origin = origin
            try:
                target = (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
                self.connection.request('POST', target, json.dumps(payload).encode(),
                                        {'Content-Type': 'application/json'})
                response = self.connection.getresponse()
                data = response.read()
                if response.status != 200:
                    raise EdgeHTTPError(response.status,data)
                if response.will_close:
                    self.close()
                self.last_end = time.monotonic()
                return json.loads(data) if data else None
            except BaseException:
                self.close()
                raise


class _StreamConnections:
    """Bounded idle keep-alive leases; never replay a failed mutating request."""
    def __init__(self, limit=32):
        self.lock, self.idle, self.closed, self.limit = threading.Lock(), [], False, limit
    def take(self, host, port):
        with self.lock:
            if self.closed:
                raise RuntimeError('stream connections closed')
            while self.idle:
                origin, ended, connection = self.idle.pop()
                if origin == (host, port) and time.monotonic()-ended < 1:
                    return connection
                connection.close()
        return http.client.HTTPConnection(host, port, timeout=60)
    def put(self, connection, reusable):
        with self.lock:
            if reusable and not self.closed and len(self.idle) < self.limit:
                self.idle.append(((connection.host, connection.port), time.monotonic(), connection))
            else:
                connection.close()
    def close(self):
        with self.lock:
            self.closed = True
            for _, _, connection in self.idle:
                connection.close()
            self.idle.clear()


class EdgeStream:
    """One streamed edge path on its own connection; tokens are delivered from a reader thread.

    on_token(i, token) for each streamed token, on_end(sent, error) exactly once. close() may be
    called from any thread and ends the reader without an error report.
    """
    def _configure_connection(self, url, connections=None):
        parsed = http_url(url)
        self.connection_lock, self.connections = threading.Lock(), connections
        self.connection = (connections.take(parsed.hostname, parsed.port or 80) if connections is not None
                           else http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=60))

    def _release_connection(self, reusable=False):
        with self.connection_lock:
            connection, self.connection = self.connection, None
            if connection is not None:
                if self.connections is None:
                    connection.close()
                else:
                    self.connections.put(connection, reusable and not self.closed.is_set())

    def __init__(self, url, payload, on_token, on_end, *, connections=None, readers=None, validate_tokens=None,
                 on_gap=None):
        self.timings = dict(created_ns=time.monotonic_ns())
        self._configure_connection(url, connections)
        self.timings['connection_ready_ns'] = time.monotonic_ns()
        self.payload, self.on_token, self.on_end = payload, on_token, on_end
        # Optional root top-2 sink: explicit, or a `gap` attribute on the token
        # callback (the same idiom `chunk` uses for framed batches). Without a
        # sink the reader never inspects or validates the field.
        self.on_gap = on_gap if on_gap is not None else getattr(on_token, 'gap', None)
        self.want_gap = self.on_gap is not None
        self.validate_tokens = validate_tokens
        self.closed, self.readers = threading.Event(), readers
        try:
            if readers is None:
                self.thread = threading.Thread(target=self._run, name='edge-stream', daemon=True)
                self.thread.start()
            else:
                self.thread = readers.submit(self._run, self.close)
        except BaseException:
            self._release_connection()
            raise

    def _run(self):
        self.timings['reader_start_ns'] = time.monotonic_ns()
        if getattr(self, 'readers', None) is not None and self.closed.is_set():
            self._release_connection()
            self.timings['released_ns'] = time.monotonic_ns()
            self.on_end(None, None)
            return
        sent, error, reusable = None, None, False
        try:
            self.connection.request('POST', '/stream', json.dumps(self.payload).encode(),
                                    {'Content-Type': 'application/json'})
            self.timings['request_sent_ns'] = time.monotonic_ns()
            response = self.connection.getresponse()
            self.timings['headers_ns'] = time.monotonic_ns()
            if response.status != 200:
                raise RuntimeError(f'edge stream HTTP status {response.status}')
            batch = self.payload.get('requests')
            expected = {item['id']: 0 for item in batch} if batch is not None else 0
            limits = {item['id']: min(item['ahead'], item['max_tokens']) for item in batch} if batch is not None else {}
            while not self.closed.is_set():
                line = response.readline()
                if not line:
                    raise RuntimeError('edge stream ended without completion')
                if 'first_frame_ns' not in self.timings:
                    self.timings['first_frame_ns'] = time.monotonic_ns()
                row = json.loads(line)
                if row.get('done'):
                    sent = row['sent']
                    if batch is not None and (not isinstance(sent, dict) or sent != expected
                            or any(type(n) is not int for n in sent.values())):
                        raise ValueError('edge stream batch completion binding')
                    break
                if batch is not None and 'chunks' in row:
                    chunks, proposed = row['chunks'], dict(expected)
                    if not isinstance(chunks, list) or not 1 <= len(chunks) <= 16:
                        raise ValueError('edge stream chunk frame binding')
                    gaps = []
                    for chunk in chunks:
                        rid, index, tokens = chunk.get('rid'), chunk.get('i'), chunk.get('tokens')
                        if (rid not in proposed or type(index) is not int or index != proposed[rid]
                                or not isinstance(tokens, list) or not 1 <= len(tokens) <= 3
                                or any(type(t) is not int or t < 0 for t in tokens)
                                or proposed[rid]+len(tokens) > limits[rid]):
                            raise ValueError('edge stream chunk row binding')
                        top2 = chunk.get('root_top2') if self.want_gap else None
                        if top2 is not None:
                            # Optional root metadata: only a request's first
                            # streamed position can be a root.
                            if index != 0:
                                raise ValueError('edge stream root top2 binding')
                            gaps.append((rid, index, validate_top2(top2, tokens[0])))
                        proposed[rid] += len(tokens)
                    if self.validate_tokens is not None:
                        for chunk in chunks:self.validate_tokens(chunk['tokens'])
                    expected = proposed
                    for rid, index, meta in gaps:
                        self._root_top2(rid, index, meta)
                    for chunk in chunks:
                        self.on_token(chunk['rid'], chunk['i'], chunk['tokens'])
                    continue
                if batch is not None:
                    rid = row.get('rid')
                    if (rid not in expected or row.get('i') != expected[rid]
                            or type(row.get('token')) is not int or row['token'] < 0):
                        raise ValueError('edge stream batch row binding')
                    top2 = row.get('root_top2') if self.want_gap else None
                    if top2 is not None:
                        if row['i'] != 0:
                            raise ValueError('edge stream root top2 binding')
                        self._root_top2(rid, row['i'], validate_top2(top2, row['token']))
                    expected[rid] += 1
                    self.on_token(rid, row['i'], row['token'])
                    continue
                if row.get('i') != expected or type(row.get('token')) is not int or row['token'] < 0:
                    raise ValueError('edge stream order / token binding')
                top2 = row.get('root_top2') if self.want_gap else None
                if top2 is not None:
                    if row['i'] != 0:
                        raise ValueError('edge stream root top2 binding')
                    self._root_top2(self.payload.get('id'), row['i'], validate_top2(top2, row['token']))
                expected += 1
                self.on_token(row['i'], row['token'])
            if sent is not None and not self.closed.is_set():
                if response.read():
                    raise ValueError('edge stream data after completion')
                reusable = not response.will_close
        except BaseException as exc:
            if not self.closed.is_set():
                error = exc
        finally:
            self.timings['read_done_ns'] = time.monotonic_ns()
            self._release_connection(reusable and error is None)
            self.timings['released_ns'] = time.monotonic_ns()
            self.on_end(sent, error)

    def _root_top2(self, rid, index, meta):
        """Optional per-frame root metadata; never required by older readers."""
        if self.on_gap is not None:
            self.on_gap(rid, index, meta)

    def close(self):
        self.closed.set()
        with self.connection_lock:
            sock = self.connection.sock if self.connection is not None else None
            if sock is not None:
                try:
                    sock.shutdown(2)
                except OSError:
                    pass


class _BatchHandle:
    def __init__(self, group, rid):
        self.group, self.rid = group, rid
    @property
    def timings(self):
        return self.group.reader.timings
    @property
    def thread(self):
        return self.group.reader.thread
    def close(self):
        self.group.close(self.rid)


class _BatchPaths:
    """One framed reader; a row may close without cancelling live siblings.

    Never invoke user callbacks under this lock: StreamPrefetch closes handles
    while holding its own lock, and reader callbacks acquire that same lock.
    """
    def __init__(self, url, items, connections=None, readers=None, validate_tokens=None):
        if not 1 <= len(items) <= 16 or len({item[0]['id'] for item in items}) != len(items):
            raise ValueError('unique bounded stream batch required')
        self.lock = threading.Lock()
        self.validate_tokens = validate_tokens
        self.rows = {p['id']: dict(token=token, end=end, done=False, closed=False)
                     for p, token, end in items}
        # Only a row whose callback carries a gap sink makes the reader inspect
        # the field at all.
        asked = any(getattr(token, 'gap', None) is not None for _, token, _ in items)
        self.reader = EdgeStream(url, dict(id='ec-batch-'+uuid.uuid4().hex, framing='chunks',
            requests=[p for p, _, _ in items]), self.token, self.end, connections=connections, readers=readers,
            validate_tokens=validate_tokens, on_gap=self.gap if asked else None)
    def gap(self, rid, index, meta):
        if self.validate_tokens is not None:
            self.validate_tokens([meta['alternate']])
        with self.lock:
            row = self.rows.get(rid)
            callback = (None if row is None or row['done'] or row['closed'] else row['token'])
        sink = getattr(callback, 'gap', None) if callback is not None else None
        if sink is not None:
            sink(rid, index, meta)
    def token(self, rid, index, token):
        with self.lock:
            row = self.rows[rid]
            callback = None if row['done'] or row['closed'] else row['token']
        if callback is not None:
            if isinstance(token, list):
                chunk = getattr(callback, 'chunk', None)
                if chunk is not None:
                    chunk(index, token)
                else:
                    for offset, value in enumerate(token):
                        callback(index+offset, value)
            else:
                callback(index, token)
    def finish(self, rid, sent, error):
        with self.lock:
            row = self.rows[rid]
            if row['done']:
                return
            row['done'] = True
            callback = row['end']
        callback(sent, error)
    def end(self, sent, error):
        for rid in self.rows:
            self.finish(rid, sent.get(rid) if isinstance(sent, dict) else None, error)
    def close(self, rid):
        with self.lock:
            self.rows[rid]['closed'] = True
            all_closed = all(row['closed'] for row in self.rows.values())
        self.finish(rid, None, None)
        if all_closed:
            self.reader.close()


def make_edge_client(url, *, bounded_capacity=False, expected=None, service_context=None):
    if os.environ.get('TRIAGE_EDGE_BACKEND', 'native') not in ('native', 'hf'):
        raise ValueError('only native and hf edge backends are included')
    model = os.environ.get('TRIAGE_MODEL_PATH')
    if model is None and expected is None:
        return EdgeClient(url)
    from .models import vocabulary
    from .policy import CONTEXT
    context = CONTEXT
    if service_context is None:
        service_context = context
        if bounded_capacity and os.environ.get('TRIAGE_EDGE_BACKEND') == 'hf':
            if os.environ.get('TRIAGE_HF_CLOUD_STATIC') == '1':
                service_context = int(os.environ.get('TRIAGE_HF_CLOUD_CACHE', '512'))
            elif os.environ.get('TRIAGE_HF_FIXED_SLAB') == '1':
                service_context = int(os.environ.get('TRIAGE_HF_FIXED_CONTEXT', '1024'))
    return EdgeClient(url, expected=vocabulary(model).identity() if expected is None else expected, context=context,
                      bounded_capacity=bounded_capacity, service_context=service_context)


class EdgeClient:
    def __init__(self, url, transport=None, *, expected=None, context=None, bounded_capacity=False,
                 service_context=None):
        self.url = url.rstrip('/')
        self.transport = JsonTransport() if transport is None else transport
        self.stream_multiplexed = os.environ.get('TRIAGE_EDGE_BACKEND', 'native') == 'native'
        self.stream_connections = _StreamConnections()
        self.stream_readers, self.closed, self.readers_lock = None, False, threading.Lock()
        self.vocabulary = None
        self.context = context if context is not None else getattr(self, 'context', None)
        if expected is not None:
            from .models import check_identity
            try:
                actual = self.transport(self.url+'/batch', dict(op='identity',
                    tokenizer_count=expected['tokenizer_count']))
                check_identity(expected, actual, context)
                from .growth import check_capability
                check_capability(actual)
                self.growth_capability=actual.get('growth')
                if (os.environ.get('TRIAGE_EDGE_BACKEND') == 'hf' or
                        isinstance(actual.get('service'), dict) and actual['service'].get('engine') == 'hf'):
                    # Model identity context is an EXACT protocol field, while
                    # service_context is the physical request budget. Never
                    # substitute the latter into check_identity above.
                    self._service_contract(actual.get('service'),
                        context if service_context is None else service_context, bounded_capacity)
            except BaseException:
                self.close()
                raise
            self.vocabulary = expected['tokenizer_count']
            self.identity = dict(target=expected, draft=actual)
            logging.getLogger(__name__).info('MODEL_PAIR %s', json.dumps(self.identity, sort_keys=True))

    def negotiate_growth(self):
        """Read and validate the edge's growth-protocol capability."""
        from .growth import check_capability
        if not hasattr(self,'growth_capability'):
            actual=self.transport(self.url+'/batch',dict(op='status'))
            self.growth_capability=check_capability(actual)
        else:
            check_capability(dict(growth=self.growth_capability))
        return self.growth_capability

    def _service_contract(self, service, required_context, bounded_capacity):
        if not isinstance(service, dict) or service.get('engine') != 'hf':
            raise ValueError('HF edge must advertise effective service bounds')
        context, capacity = service.get('context'), service.get('capacity')
        if type(context) is not int or not DRAFT_TOKENS < context <= CONTEXT:
            raise ValueError('invalid HF edge service context')
        if type(capacity) is not int or not 1 <= capacity <= MAX_REQUESTS:
            raise ValueError('invalid HF edge session capacity')
        if required_context is not None and context < required_context:
            raise ValueError(f'edge service context {context} below target {required_context}; align cache bounds')
        if capacity < MAX_REQUESTS and not bounded_capacity:
            raise ValueError('edge session capacity below native resident limit; enlarge edge pool or use bounded HF cloud')
        self.context = min(context, required_context) if required_context is not None else context
        self.session_capacity = capacity
        self.service = dict(service)

    def _tokens(self, tokens):
        if self.vocabulary is None:
            return
        if not isinstance(tokens, (list, tuple)) or not tokens:
            raise ValueError('token outside negotiated target / draft vocabulary')
        if len(tokens) > 32:
            # C-level runs for homogeneous integer prefixes, without weakening type/range checks.
            invalid = (any(kind is not int for kind, _ in groupby(tokens, type))
                       or min(tokens) < 0 or max(tokens) >= self.vocabulary)
        else:
            invalid = any(type(t) is not int or not 0 <= t < self.vocabulary for t in tokens)
        if invalid:
            raise ValueError('token outside negotiated target / draft vocabulary')

    def _prefix(self, tokens, ahead):
        self._tokens(tokens)
        if self.vocabulary is not None and (type(ahead) is not int or ahead < 0
                                            or len(tokens)+ahead > self.context):
            raise ValueError('prefix exceeds negotiated edge context')

    def prepare_stream_readers(self):
        from .stream_readers import StreamReaders
        with self.readers_lock:
            if self.closed:
                raise RuntimeError('edge client closed')
            if self.stream_readers is None:
                self.stream_readers = StreamReaders()
            return self.stream_readers.prepare()

    def stream(self, item, on_token, on_end):
        self._prefix(item['tokens'], item['ahead'])
        def checked_token(index, token):
            self._tokens([token])
            on_token(index, token)
        sink = getattr(on_token, 'gap', None)
        def checked_gap(rid, index, meta):
            # The alternate is a draft-vocabulary token even when nobody reads it.
            self._tokens([meta['alternate']])
            sink(rid, index, meta)
        return EdgeStream(self.url+'/stream', item, checked_token, on_end,
                          connections=self.stream_connections, readers=self.stream_readers,
                          on_gap=checked_gap if sink is not None else None)

    def stream_batch(self, items):
        if not self.stream_multiplexed:
            return [self.stream(payload, on_token, on_end) for payload, on_token, on_end in items]
        checked = []
        for payload, on_token, on_end in items:
            self._prefix(payload['tokens'], payload['ahead'])
            def checked_token(index, token, callback=on_token):
                self._tokens([token])
                callback(index, token)
            chunk = getattr(on_token, 'chunk', None)
            if callable(chunk):
                def checked_chunk(index, tokens, callback=chunk):
                    self._tokens(tokens)
                    callback(index, tokens)
                checked_token.chunk = checked_chunk
            checked_token.gap = getattr(on_token, 'gap', None)
            checked.append((payload, checked_token, on_end))
        group = _BatchPaths(self.url+'/stream', checked, self.stream_connections, self.stream_readers,
                            validate_tokens=self._tokens)
        return [_BatchHandle(group, payload['id']) for payload, _, _ in checked]

    def propose(self, items):
        from .growth import Plan
        plans=[]
        for item in items:
            p=(Plan.bind(item['growth_plan'],item['id'],item['tokens']) if 'growth_plan' in item else None)
            plans.append(p)
            if p is not None and self.negotiate_growth() is None:raise ValueError('GROWTH_NOT_NEGOTIATED')
            self._prefix(item['tokens'],p.k if p else DRAFT_TOKENS)
        if os.environ.get('TRIAGE_PLAN_TIMING') == '1':
            self._propose_pre = time.perf_counter()
        rows = self.transport(self.url+'/batch', dict(op='sync_draft', requests=items))
        if os.environ.get('TRIAGE_PLAN_TIMING') == '1':
            self._propose_post = time.perf_counter()
        if not isinstance(rows,list) or len(rows)!=len(items) or any(type(row) is not dict for row in rows):
            raise ValueError('invalid negotiated draft batch')
        for item,row,p in zip(items,rows,plans):
            self._tokens(row.get('tokens'))
            if p is not None:
                from .policy import token_ids
                from .root_shadow import validate_top2
                if (row.get('id')!=p.rid or row.get('prefix_hash')!=p.prefix_hash
                        or Plan.bind(row.get('growth_plan'),p.rid,item['tokens'])!=p or len(token_ids(row.get('tokens')))!=p.k):
                    raise ValueError('GROWTH_EDGE_BINDING')
                if p.branches==1:
                    if 'root_top2' not in row or row['root_top2'] is not None:raise ValueError('GROWTH_ROOT1_CAPABILITY')
                else:
                    meta=validate_top2(row.get('root_top2'),row['tokens'][0])
                    self._tokens([meta['primary'],meta['alternate']])
        return rows

    def warm(self, items):
        for item in items:
            self._prefix(item['tokens'], 0)
        result = self.transport(self.url+'/batch', dict(op='warm', requests=items))
        ids = [item['id'] for item in items]
        if (not isinstance(result, dict) or not isinstance(result.get('warmed'), list)
                or set(result['warmed'])-set(ids) or len(result['warmed']) != len(set(result['warmed']))):
            raise ValueError('invalid edge warm response')
        return result['warmed']

    def free(self, ids):
        result = self.transport(self.url+'/batch', dict(op='free', ids=ids))
        if (not isinstance(result, dict) or not isinstance(result.get('freed'), list)
                or set(result['freed'])-set(ids) or len(result['freed']) != len(set(result['freed']))):
            raise ValueError('invalid edge free response')

    def close(self):
        with self.readers_lock:
            self.closed = True
            readers = self.stream_readers
        if readers is None:
            close = getattr(self.transport, 'close', None)
            if close is not None:
                close()
            self.stream_connections.close()
            return
        try:
            readers.close()
        finally:
            close = getattr(self.transport, 'close', None)
            try:
                if close is not None:
                    close()
            finally:
                self.stream_connections.close()
