import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
import uuid


def experiments():
    """Read cloud proposal and scheduling options from the environment."""
    def flag(name, default):
        value = os.environ.get(name, default)
        if value not in ('0', '1'):
            raise ValueError('invalid HF experiment switch: '+name)
        return value == '1'
    if os.environ.get('EC_WEAK_EDGE_MULTIEDGE', '0') != '0':
        raise ValueError('HF cloud has no authenticated peer pool')
    if os.environ.get('EC_STREAM_POLICY', 'cohort') != 'cohort':
        raise ValueError('cohort-pipeline is the only proposal policy')
    if os.environ.get('TRIAGE_EXACT_PROPOSALS', 'off') != 'off':
        raise ValueError('exact proposal cache is edge-only')
    exact, latency = flag('EC_EXACT_PREFIX_DRAFTS', '1'), flag('EC_LATENCY_AWARE_PREFILL', '0')
    warm = flag('EC_CPU_PROMPT_WARM', '0')
    if latency and not exact:
        raise ValueError('latency-aware prefill requires exact-prefix drafts')
    if warm and os.environ.get('TRIAGE_EDGE_BACKEND') != 'cpu':
        raise ValueError('CPU prompt warm requires the CPU edge backend')
    result = dict(exact_prefix_drafts=exact, latency_aware_prefill=latency,
                  cpu_prompt_warm=warm, cpu_stream_drafts=True, stream_policy='cohort')
    rows = os.environ.get('EC_CPU_STREAM_ROWS')
    if rows is not None:
        from engines.common.policy import MAX_REQUESTS
        if not rows.isdigit() or not 1 <= int(rows) <= MAX_REQUESTS:
            raise ValueError('stream rows requires a bound of 1..%d' % MAX_REQUESTS)
        result['cpu_stream_rows'] = int(rows)
    return result


def latency_status(ctl):
    return ({'latency_prefill': ctl.latency_snapshot()}
            if getattr(ctl, 'latency_aware_prefill', False) else {})


class RuntimeTransport:
    """The controller's only engine touch point: one synchronous dispatch."""

    def __init__(self, runtime):
        self.runtime = runtime

    def collective_rpc(self, method, *, args, timeout=None):
        if method != 'tree_service_call' or not isinstance(args, tuple) or len(args) != 1:
            raise ValueError('invalid HF transport call')
        return [self.runtime.dispatch(args[0])]


def make_app(args):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from transformers import AutoTokenizer

    from engines.common.completion import parse_completion
    from engines.common.controller import Controller
    from engines.common.detokenization import (byte_fallback_plan,
                                               validate_byte_fallback_decoder)
    from engines.common.models import model_info, served_name, vocabulary
    from engines.common.policy import CONTEXT, DRAFT_TOKENS, WIDTH
    from engines.hf.backend import HFBackend
    from engines.hf.runtime import ChainRuntime

    settings = experiments()
    cloud_static = os.environ.get('TRIAGE_HF_CLOUD_STATIC', '0') == '1'
    fixed_slab = os.environ.get('TRIAGE_HF_FIXED_SLAB', '0') == '1'
    if cloud_static and fixed_slab:
        raise ValueError('fixed batched slab and single-row static mode are exclusive')
    try:
        fixed_context = int(os.environ.get('TRIAGE_HF_FIXED_CONTEXT', '1024'))
        cloud_cache_len = int(os.environ.get('TRIAGE_HF_CLOUD_CACHE', '512'))
    except ValueError as exc:
        raise ValueError('invalid TRIAGE_HF_CLOUD_CACHE') from exc
    if cloud_static and not 256 <= cloud_cache_len <= 32768:
        raise ValueError('cloud static cache length out of range')
    if fixed_slab and not 256 <= fixed_context <= 32768:
        raise ValueError('fixed slab context length out of range')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = served_name(args.model, getattr(args, 'served_model_name', None), engine='hf')
    metadata = model_info(args.model)
    model_vocabulary = vocabulary(args.model)
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        backend = HFBackend(args.model)
        budget = os.environ.get('TRIAGE_HF_KV_TOKENS')
        if budget is not None and (not budget.isdigit() or int(budget) < 1024):
            raise ValueError('TRIAGE_HF_KV_TOKENS must be an integer of at least 1024')
        runtime = ChainRuntime(backend, role='cloud',
                               kv_budget_tokens=int(budget) if budget is not None else None)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        text_plan = None
        if metadata.gemma3:
            decoder_document = json.loads((Path(args.model)/'tokenizer.json').read_text())
            validate_byte_fallback_decoder(decoder_document)
            text_plan = byte_fallback_plan(model_vocabulary.tokens, decoder_document,
                                           tokenizer, excluded_token_ids=model_vocabulary.excluded_token_ids)
        eos_ids = set(metadata.eos_ids)
        if type(tokenizer.eos_token_id) is int:
            eos_ids.add(tokenizer.eos_token_id)
        config = dict(engine='transformers', role='cloud', model=args.model,
                      served_model_name=name, context=CONTEXT,
                      draft_tokens=DRAFT_TOKENS, width=WIDTH, verify=runtime.status()['verify'],
                      kv=runtime.status()['kv'], batching=runtime.status()['batching'],
                      cuda_graph=runtime.status()['cuda_graph'],
                      tree=runtime.growth, growth=runtime.status().get('growth'), paged_kv=False,
                      cloud_static=cloud_static, cloud_cache_len=cloud_cache_len,
                      fixed_slab=fixed_slab, effective_context=runtime.request_context,
                      decode_attention=runtime.status()['decode_attention'],
                      prefill_attention=runtime.status()['prefill_attention'],
                      experiments=settings)
        (out/'ENGINE_CONFIG.json').write_text(json.dumps(config, indent=2)+'\n')
        from engines.common.clients import make_edge_client
        # Explicit model identity/budget also covers the runnable module CLI and
        # embedding entry, which need not inherit launcher environment variables.
        client = make_edge_client(args.edge_url, expected=model_vocabulary.identity(),
                                  service_context=runtime.request_context, bounded_capacity=True)
        ctl = Controller(RuntimeTransport(runtime), tokenizer, args.edge_url, out,
                         root_fork5=runtime.growth,
                         bounded_edge_admission=True, edge_client=client, **settings)
        ctl.token_limit = len(model_vocabulary.tokens)
        ctl.text_plan = text_plan
        state['ctl'], state['eos_ids'] = ctl, eos_ids
        state['verify_width'] = runtime.verify_width
        await ctl.rpc({'op': 'status'})
        await ctl.rpc({'op': 'validate', 'enabled': False})
        await ctl.rpc({'op': 'configure_compaction', 'mode': 'batched'})
        await ctl.rpc({'op': 'configure_tree', 'mode': 'fork5' if runtime.growth else 'chain4'})
        (out/'ENGINE_READY.json').write_text(json.dumps(
            dict(runtime=await ctl.rpc({'op': 'status'}), eos_ids=sorted(eos_ids)), indent=2)+'\n')
        task = asyncio.create_task(ctl.loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            try:
                await ctl.free(list(dict.fromkeys([*ctl.active, *ctl.admitting])))
            finally:
                await asyncio.to_thread(ctl.prefetch.close)
        (out/'SERVICE_COUNTS.json').write_text(json.dumps(dict(
            submitted=ctl.submitted, completed=ctl.completed, peak_active=ctl.peak_active,
            failure=ctl.failure, admit_calls=ctl.admit_calls, max_admit_batch=ctl.max_admit_batch,
            admission_deferrals=ctl.admission_deferrals,
            ar_rounds=getattr(ctl, 'ar_rounds', 0), spec_rounds=getattr(ctl, 'spec_rounds', 0),
            worker=await ctl.rpc({'op': 'status'}), **latency_status(ctl)), indent=2)+'\n')

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health():
        if 'ctl' not in state or state['ctl'].failure:
            raise HTTPException(503, 'not ready')
        return {'status': 'ok', 'role': 'cloud', 'engine': 'transformers'}

    @app.get('/status')
    async def status():
        ctl = state['ctl']
        return dict(worker=await ctl.rpc({'op': 'status'}), active=len(ctl.active),
                    waiting=ctl.waiting.qsize(), proposals=ctl.prefetch.snapshot(),
                    ar_rounds=getattr(ctl, 'ar_rounds', 0), spec_rounds=getattr(ctl, 'spec_rounds', 0),
                    submitted=ctl.submitted, completed=ctl.completed, peak_active=ctl.peak_active,
                    failure=ctl.failure, admit_calls=ctl.admit_calls,
                    max_admit_batch=ctl.max_admit_batch,
                    admission_deferrals=ctl.admission_deferrals, **latency_status(ctl))

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': name, 'object': 'model'}]}

    @app.post('/v1/completions')
    @app.post('/v1/chat/completions')
    async def completions(request: Request):
        ctl = state['ctl']
        body = await request.json()
        if ctl.failure:
            raise HTTPException(503, ctl.failure)
        is_chat = request.url.path.endswith('/chat/completions')
        try:
            tokens, max_tokens, ignore_eos = parse_completion(
                body, ctl.ingress_tokenizer, is_chat=is_chat, token_limit=ctl.token_limit)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if cloud_static and len(tokens)+max_tokens+4 > cloud_cache_len:
            raise HTTPException(400, 'request exceeds the cloud static cache')
        if fixed_slab and len(tokens)+max_tokens+state['verify_width'] > fixed_context:
            raise HTTPException(400, 'request exceeds the fixed batched KV context')
        rid = 'ec-'+uuid.uuid4().hex
        from engines.common.cpu_request_accounting import begin as cpu_begin, finish as cpu_finish
        cpu_meter = cpu_begin(out, rid)
        r = dict(id=rid, tokens=tokens, max_tokens=max_tokens, generated=[], text='',
                 queue=asyncio.Queue(), input_length=len(tokens),
                 eos_ids=set() if ignore_eos else state['eos_ids'], done=False)
        try:
            ctl.submit(r)
        except asyncio.QueueFull:
            cpu_finish(cpu_meter, 'rejected', 0)
            raise HTTPException(503, 'bounded request queue full')

        async def stream():
            def frame(delta, finish=None, usage=None):
                obj = dict(id=rid, object='chat.completion.chunk', created=int(time.time()),
                           model=name, choices=[])
                if not usage:
                    obj['choices'] = [dict(index=0, delta=delta, finish_reason=finish)]
                if not is_chat:
                    obj['object'] = 'text_completion'
                    if not usage:
                        obj['choices'] = [dict(index=0, text=delta.get('content', ''),
                                               token_ids=delta.get('token_ids', []),
                                               finish_reason=finish)]
                if usage:
                    obj['usage'] = usage
                return 'data: '+json.dumps(obj, ensure_ascii=False)+'\n\n'
            try:
                yield frame({'role': 'assistant', 'content': ''})
                while True:
                    event = await r['queue'].get()
                    if 'error' in event:
                        cpu_finish(cpu_meter, 'error', len(r['generated']))
                        yield 'data: '+json.dumps({'error': event['error']})+'\n\n'
                        break
                    yield frame({'content': event['text'], 'token_ids': event['token_ids']})
                    if event['finish_reason']:
                        cpu_finish(cpu_meter, 'completed', len(r['generated']))
                        yield frame({}, event['finish_reason'])
                        yield frame({}, usage=dict(prompt_tokens=r['input_length'],
                                                   completion_tokens=len(r['generated']),
                                                   total_tokens=r['input_length']+len(r['generated'])))
                        yield 'data: [DONE]\n\n'
                        break
            finally:
                cpu_finish(cpu_meter, 'cancelled', len(r['generated']))
                if not r['done']:
                    ctl.cancelled.add(rid)
                    ctl.trace('cancel', r)
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--served-model-name')
    parser.add_argument('--edge-url', default='http://127.0.0.1:38101')
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    os.environ.setdefault('TRIAGE_ADAPTIVE_GROWTH', '1')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    import uvicorn
    uvicorn.run(make_app(args), host=args.host, port=args.port, log_level='warning',
                access_log=False)


if __name__ == '__main__':
    main()
