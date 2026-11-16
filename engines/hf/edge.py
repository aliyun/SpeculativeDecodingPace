import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

from engines.common.edge_stream import StreamDrafts
from engines.common.models import vocabulary
from engines.common.policy import CONTEXT, DRAFT_TOKENS, MAX_REQUESTS, token_ids
from engines.common.proposal_cache import install_cache


def validate_batch_payload(payload):
    """Reject malformed batches before any stream epoch or model state changes."""
    if not isinstance(payload, dict):
        raise ValueError('edge payload must be an object')
    op = payload.get('op')
    if op not in ('identity', 'status', 'warm', 'sync_draft', 'free'):
        raise ValueError('invalid edge operation')
    if op == 'free':
        ids = payload.get('ids')
    elif op in ('warm', 'sync_draft'):
        rows = payload.get('requests')
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_REQUESTS:
            raise ValueError('invalid edge request batch')
        ids = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('edge request must be an object')
            rid = row.get('id')
            if not isinstance(rid, str) or not rid.startswith('ec-'):
                raise ValueError('invalid edge request id')
            token_ids(row.get('tokens'))
            ids.append(rid)
    else:
        return
    # Check every type BEFORE hashing or deleting any epoch. Empty free remains
    # an idempotent no-op; do not add a new limit to legacy cleanup batches.
    if (not isinstance(ids, list) or any(not isinstance(rid, str) or not rid for rid in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('invalid edge request id batch')


def make_app(args):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    from engines.hf.backend import HFBackend
    from engines.hf.runtime import ChainRuntime

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = vocabulary(args.model)
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        backend = HFBackend(args.model)
        runtime = ChainRuntime(backend, role='edge', context=CONTEXT)
        if backend.context and backend.context < CONTEXT:
            raise RuntimeError('HF draft context is below the fixed service context')
        state['runtime'] = runtime
        state['lock'] = asyncio.Lock()

        async def dispatch(payload):
            # One device, one serialized forward batch; health stays responsive.
            async with state['lock']:
                return await asyncio.to_thread(runtime.dispatch, payload)

        state['dispatch'] = dispatch
        state['cache'] = install_cache(dispatch, 'transformers', args.model)
        state['drafts'] = StreamDrafts(dispatch, context=runtime.request_context,
                                       capacity=runtime.edge_capacity)
        if runtime.static or runtime.fused:
            # Exercise the same warm/draft/free path once before advertising ready.
            runtime.dispatch(dict(op='warm', requests=[
                dict(id='ec-__warm__', tokens=[1, 2, 3, 4])]))
            runtime.dispatch(dict(op='sync_draft', requests=[
                dict(id='ec-__warm__', tokens=[1, 2, 3, 4], max_tokens=8)]))
            runtime.dispatch(dict(op='free', ids=['ec-__warm__']))
        config = dict(engine='transformers', role='edge', model=args.model,
                      context=CONTEXT, draft_tokens=DRAFT_TOKENS, max_requests=runtime.edge_capacity,
                      kv=runtime.status()['kv'], batching=runtime.status()['batching'],
                      effective_context=runtime.request_context,
                      cuda_graph=runtime.fused, fused_steps=runtime.fused, paged_kv=False)
        (out/'ENGINE_CONFIG.json').write_text(json.dumps(config, indent=2)+'\n')
        (out/'ENGINE_READY.json').write_text(json.dumps(runtime.status(), indent=2)+'\n')
        yield
        state['drafts'].epochs.clear()
        (out/'SERVICE_COUNTS.json').write_text(json.dumps(dict(
            runtime=state['runtime'].status(),
            exact_proposals=state['cache'].status() if state['cache'] is not None else None),
            indent=2)+'\n')

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health():
        if 'runtime' not in state:
            raise HTTPException(503, 'not ready')
        if state['drafts'].failure is not None:
            raise HTTPException(503, 'HF edge stream failed')
        return {'status': 'ok', 'role': 'edge', 'engine': 'transformers'}

    @app.get('/status')
    async def status():
        cache = state['cache']
        return dict(runtime=state['runtime'].status(), stream_failed=state['drafts'].failure is not None,
                    exact_proposals=cache.status() if cache is not None else None)

    @app.post('/batch')
    async def batch(request: Request):
        try:
            payload = await request.json()
            validate_batch_payload(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if payload.get('op') == 'identity':
            try:
                result = identity.identity(payload.get('tokenizer_count'))
                if state['runtime'].growth:
                    from engines.common.growth import capability
                    result['growth'] = capability()
                result['service'] = dict(engine='hf', context=state['runtime'].request_context,
                                         capacity=state['runtime'].edge_capacity)
                return result
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        if payload.get('op') not in ('sync_draft', 'warm', 'free', 'status'):
            raise HTTPException(400, 'invalid edge operation')
        dispatch = state['cache'].dispatch if state['cache'] is not None else state['dispatch']
        try:
            if payload['op'] == 'free':
                return await state['drafts'].free(payload, dispatch)
            if payload['op'] == 'warm':
                return await state['dispatch'](payload)
            if payload['op'] == 'sync_draft' and any('growth_plan' in row for row in payload['requests']):
                return await state['drafts'].sync_draft(payload, dispatch)
            return await dispatch(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post('/stream')
    async def stream(request: Request):
        return state['drafts'].response(await request.json())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    os.environ.setdefault('TRIAGE_ADAPTIVE_GROWTH', '1')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    import uvicorn
    uvicorn.run(make_app(args), host=args.host, port=args.port, log_level='warning',
                access_log=False)


if __name__ == '__main__':
    main()
