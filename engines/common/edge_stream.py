import asyncio
import itertools
import json
import logging
import time

from .policy import CONTEXT, DRAFT_TOKENS, MAX_REQUESTS, STREAM_AHEAD, token_ids
from .contracts import EdgeAdmissionError
from .root_shadow import validate_top2


class StreamDrafts:
    def __init__(self, dispatch, *, context=CONTEXT, capacity=MAX_REQUESTS):
        if type(context) is not int or not DRAFT_TOKENS < context <= CONTEXT:
            raise ValueError('invalid edge service context')
        if type(capacity) is not int or not 1 <= capacity <= MAX_REQUESTS:
            raise ValueError('invalid edge service capacity')
        self.context, self.capacity = context, capacity
        self.dispatch = dispatch
        self.lock = asyncio.Lock()
        self.epochs, self.counter = {}, itertools.count(1)
        self.pending, self.worker, self.failure = {}, None, None
        self.stats = dict(streams=0, stream_rounds=0, stream_tokens=0, superseded=0,
                          stream_batches=0, stream_batch_rows=0, stream_batch_max=0)

    async def _locked(self, work):
        # A cancelled reader must not release the lock while native work still runs.
        task = asyncio.ensure_future(work)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _round(self, rid, epoch, prefix, maximum, want_gap=False):
        if self.failure is not None:
            raise RuntimeError('native stream batch failed') from self.failure
        if self.epochs.get(rid) != epoch:
            return None
        future = asyncio.get_running_loop().create_future()
        previous = self.pending.pop(rid, None)
        if previous is not None and not previous[-1].done():
            previous[-1].set_result(None)
        self.pending[rid] = (rid, epoch, prefix, maximum, want_gap, future)
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._flush())
        return await asyncio.shield(future)

    async def _flush(self):
        """Coalesce already-ready rows, without a fixed batching delay."""
        selected = []
        try:
            while self.pending:
                selected = [self.pending.pop(rid) for rid in list(self.pending)[:MAX_REQUESTS]]
                async with self.lock:
                    live = []
                    for item in selected:
                        rid, epoch, prefix, maximum, want_gap, future = item
                        if self.epochs.get(rid) == epoch:
                            live.append(item)
                        elif not future.done():
                            future.set_result(None)
                    if not live:
                        continue
                    try:
                        rows = await self.dispatch(dict(op='sync_draft', requests=[
                            dict(id=rid, tokens=prefix, max_tokens=maximum, rewind=True,
                                 **({'root_top2': True} if want_gap else {}))
                            for rid, epoch, prefix, maximum, want_gap, future in live]))
                    except EdgeAdmissionError as exc:
                        # A declared capacity/context rejection happens before
                        # model mutation. Do NOT swallow internal ValueError,
                        # nonfinite or CUDA failures: those remain fail-closed.
                        for item in live:
                            if not item[-1].done(): item[-1].set_exception(exc)
                        continue
                    if not isinstance(rows, list) or len(rows) != len(live):
                        raise ValueError('native stream round binding')
                    outputs = []
                    for row, item in zip(rows, live):
                        if row.get('id') != item[0]:
                            raise ValueError('native stream round binding')
                        tokens = token_ids(row.get('tokens'))
                        if len(tokens) != DRAFT_TOKENS:
                            raise ValueError('incomplete native stream round')
                        # Root top-2 is only read and validated for a request
                        # that asked for it (payload gap=True, default off), so
                        # the default path neither inspects nor validates rows.
                        top2 = row.get('root_top2') if item[4] else None
                        outputs.append((tokens, None if top2 is None else validate_top2(top2, tokens[0])))
                    self.stats['stream_batches'] += 1
                    self.stats['stream_batch_rows'] += len(live)
                    self.stats['stream_batch_max'] = max(self.stats['stream_batch_max'], len(live))
                    for item, result in zip(live, outputs):
                        if not item[-1].done():
                            item[-1].set_result(result)
        except BaseException as exc:
            logging.getLogger(__name__).exception('EDGE_STREAM_FAILURE')
            self.failure = exc
            for item in selected + list(self.pending.values()):
                if not item[-1].done():
                    item[-1].set_exception(exc)
            self.pending.clear()
        finally:
            self.worker = None

    async def stream(self, payload, emit):
        """Emit up to payload['ahead'] greedy tokens after payload['tokens']; returns the count sent."""
        rid, prefix = payload.get('id'), token_ids(payload.get('tokens'))
        ahead, maximum = payload.get('ahead'), payload.get('max_tokens')
        # Root metadata is only ever used for this path's first streamed
        # position, so only that round reads and validates it. `payload['gap']`
        # lets a client opt out entirely (default: forward it once per request).
        want_gap = payload.get('gap') is not False
        if not isinstance(rid, str) or not rid.startswith('ec-') or not prefix:
            raise ValueError('invalid edge stream request')
        if type(ahead) is not int or not 1 <= ahead <= STREAM_AHEAD:
            raise ValueError('invalid edge stream horizon')
        if type(maximum) is not int or maximum < 1:
            raise ValueError('edge stream requires the remaining token budget')
        if self.context != CONTEXT and len(prefix)+DRAFT_TOKENS > self.context:
            raise EdgeAdmissionError('prefix exceeds edge service context')
        if rid not in self.epochs and len(self.epochs) >= self.capacity:
            raise EdgeAdmissionError('edge live stream limit')
        epoch = next(self.counter)
        if rid in self.epochs:
            self.stats['superseded'] += 1
        self.epochs[rid] = epoch
        self.stats['streams'] += 1
        ahead, sent, rounds, start = min(ahead, maximum), 0, [], time.monotonic_ns()
        try:
            while sent < ahead and len(prefix)+DRAFT_TOKENS <= self.context:
                begin = time.monotonic_ns()
                result = await self._locked(self._round(rid, epoch, prefix, maximum-sent,
                                                       want_gap and sent == 0))
                if result is None:
                    break
                tokens, top2 = result
                rounds.append(time.monotonic_ns()-begin)
                self.stats['stream_rounds'] += 1
                for token in tokens[:ahead-sent]:
                    # Only the request's first streamed position is a root: a
                    # later round starts at a speculative prefix. The third
                    # argument is omitted when absent so older emitters (two
                    # positional parameters) keep working unchanged.
                    if sent == 0 and top2 is not None:
                        await emit(sent, token, top2)
                    else:
                        await emit(sent, token)
                    sent += 1
                    self.stats['stream_tokens'] += 1
                prefix = prefix+tokens
            return sent
        except EdgeAdmissionError:
            if self.epochs.get(rid) == epoch: self.epochs.pop(rid, None)
            raise
        finally:
            if self.epochs.get(rid) == epoch:
                self.epochs[rid] = -epoch
            logging.getLogger(__name__).info('EDGE_STREAM %s', json.dumps(dict(
                id=rid, epoch=epoch, open_length=len(token_ids(payload['tokens'])), ahead=ahead, sent=sent,
                round_ns=rounds, start_ns=start, end_ns=time.monotonic_ns()), sort_keys=True))

    async def sync_draft(self, payload, dispatch):
        """Give a planned round exclusive ownership of each requested edge session."""
        async def work():
            async with self.lock:
                epochs = {row['id']: self.epochs.get(row['id']) for row in payload['requests']
                          if 'growth_plan' in row}
                result = await dispatch(payload)
                for rid, epoch in epochs.items():
                    if epoch is not None and epoch > 0 and self.epochs.get(rid) == epoch:
                        self.epochs[rid] = -epoch
                return result
        return await self._locked(work())

    async def free(self, payload, dispatch):
        """Stop the request's paths, then free its native state under the same order."""
        if not isinstance(payload, dict) or payload.get('op') != 'free':
            raise ValueError('invalid edge free payload')
        ids = payload.get('ids')
        if (not isinstance(ids, list) or any(not isinstance(rid, str) or not rid for rid in ids)
                or len(ids) != len(set(ids))):
            raise ValueError('invalid edge free ids')
        # Validate before ANY epoch mutation; bind the same immutable ID set for
        # both bookkeeping and native dispatch, even while awaiting this lock.
        ids = tuple(ids)
        payload = dict(payload, ids=list(ids))
        async def work():
            async with self.lock:
                for rid in ids:
                    self.epochs.pop(rid, None)
                result = await dispatch(payload)
                # Free dominates even a replacement stream queued while native free was in flight.
                for rid in ids:
                    self.epochs.pop(rid, None)
                    pending = self.pending.pop(rid, None)
                    if pending is not None and not pending[-1].done():
                        pending[-1].set_result(None)
                return result
        return await self._locked(work())

    def batch_response(self, items, chunked=False):
        """One HTTP submission admits all sibling producers before their first flush."""
        from fastapi.responses import StreamingResponse
        if (not isinstance(items, list) or not 1 <= len(items) <= MAX_REQUESTS
                or any(not isinstance(item, dict) for item in items)):
            raise ValueError('bounded stream batch required')
        ids = [item.get('id') for item in items]
        if (any(not isinstance(rid, str) or not rid.startswith('ec-') for rid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('unique stream batch binding required')
        for item in items:
            token_ids(item.get('tokens'))
            if (type(item.get('ahead')) is not int or not 1 <= item['ahead'] <= STREAM_AHEAD
                    or type(item.get('max_tokens')) is not int or item['max_tokens'] < 1):
                raise ValueError('invalid stream batch budget')
        rows = asyncio.Queue()
        async def one(item):
            pending, root = [], []
            async def emit(i, token, top2=None):
                if top2 is not None:
                    # The root frame of this path; a chunked row buffers it.
                    root.append(dict(root_top2=top2))
                if not chunked:
                    rows.put_nowait(dict(rid=item['id'], i=i, token=token,
                                         **(root[0] if top2 is not None else {})))
                    return
                pending.append(token)
                if len(pending) == DRAFT_TOKENS:
                    index = i+1-len(pending)
                    rows.put_nowait(dict(rid=item['id'], i=index, tokens=list(pending),
                                         **(root[0] if index == 0 and root else {})))
                    pending.clear()
            sent = await self.stream(item, emit)
            if pending:
                index = sent-len(pending)
                rows.put_nowait(dict(rid=item['id'], i=index, tokens=list(pending),
                                     **(root[0] if index == 0 and root else {})))
            return sent
        async def produce():
            tasks = [asyncio.create_task(one(item)) for item in items]
            try:
                counts = await asyncio.gather(*tasks)
                return dict(zip(ids, counts))
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        async def body():
            task = asyncio.create_task(produce())
            task.add_done_callback(lambda _: rows.put_nowait(None))
            try:
                ended = False
                while not ended:
                    row = await rows.get()
                    if row is None:
                        break
                    if not chunked:
                        yield json.dumps(row)+'\n'
                        continue
                    ready = [row]
                    while not rows.empty() and len(ready) < MAX_REQUESTS:
                        row = rows.get_nowait()
                        if row is None:
                            ended = True
                            break
                        ready.append(row)
                    yield json.dumps(dict(chunks=ready))+'\n'
                yield json.dumps(dict(done=True, sent=task.result()))+'\n'
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return StreamingResponse(body(), media_type='application/x-ndjson')

    def response(self, payload):
        """Newline-delimited rows; an error ends the body without completion."""
        if 'requests' in payload:
            return self.batch_response(payload['requests'], chunked=payload.get('framing') == 'chunks')
        from fastapi.responses import StreamingResponse
        rows = asyncio.Queue()

        async def emit(i, token, top2=None):
            rows.put_nowait(dict(i=i, token=token,
                                 **({'root_top2': top2} if top2 is not None else {})))

        async def body():
            task = asyncio.ensure_future(self.stream(payload, emit))
            task.add_done_callback(lambda _: rows.put_nowait(None))
            try:
                while (row := await rows.get()) is not None:
                    yield json.dumps(row)+'\n'
                yield json.dumps(dict(done=True, sent=task.result()))+'\n'
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        return StreamingResponse(body(), media_type='application/x-ndjson')
