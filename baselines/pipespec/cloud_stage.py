#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import os

import numpy as np
import torch


def eos_id_set(tokenizer):
    """Every token id that can end a turn, not just the scalar tokenizer.eos_token_id."""
    import json as _json
    import os as _os
    ids = set()
    single = getattr(tokenizer, 'eos_token_id', None)
    if single is not None:
        ids.add(int(single))
    for x in (getattr(tokenizer, 'additional_special_tokens_ids', None) or []):
        try:
            ids.add(int(x))
        except (TypeError, ValueError):
            pass
    root = getattr(tokenizer, 'name_or_path', '') or ''
    for name in ('generation_config.json', 'config.json'):
        try:
            cfg = _json.load(open(_os.path.join(root, name)))
        except Exception:
            continue
        e = cfg.get('eos_token_id')
        for x in ([e] if isinstance(e, int) else (e or [])):
            try:
                ids.add(int(x))
            except (TypeError, ValueError):
                pass
    return ids or {0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--pub-port', type=int, required=True, help='cloud -> edge stream (we listen)')
    ap.add_argument('--sub-port', type=int, required=True, help='edge -> cloud stream (we listen)')
    ap.add_argument('--lookahead', type=int, default=5)
    ap.add_argument('--connect-timeout', type=float, default=1800.0)
    ap.add_argument('--ignore-eos', action='store_true',
                    help='forced-output loads (S01-S04): never stop on EOS, run to the cap')
    ap.add_argument('--stops', default='', help='JSON list of stop strings (HumanEval+)')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import decoding  # noqa: E402
    from rtt_transport import (LatestSlot, PusherServer, SinkServer,  # noqa: E402
                               KIND_NEW_REQUEST, KIND_CLOSE)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    device = os.environ.get('PIPESPEC_DEVICE', 'cuda:0')
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    model, tokenizer = decoding.init_model(
        decoding.ModelConfig(model_path=args.target, devices=device, dtype=dtype))
    model.eval()
    eos = tokenizer.eos_token_id
    eos_ids = eos_id_set(tokenizer)
    stops = [str(x) for x in json.loads(args.stops)] if args.stops.strip() else []
    ignore_eos = bool(args.ignore_eos)
    (out / 'CLOUD_READY.json').write_text(json.dumps({
        'ready_s': time.time(), 'load_s': time.perf_counter() - t0, 'eos_token_id': eos,
        'torch': torch.__version__,
        'device': device if device == 'cpu' else torch.cuda.get_device_name(0),
    }, indent=2) + '\n')
    print(json.dumps({'cloud_ready': True, 'load_s': time.perf_counter() - t0}), flush=True)

    verified = LatestSlot('cloud-verified')
    drafts = LatestSlot('cloud-drafts')
    pub = PusherServer(verified, args.host, args.pub_port, name='cloud-pub',
                       accept_timeout=args.connect_timeout)
    sink = SinkServer(drafts, args.host, args.sub_port, name='cloud-sink',
                      accept_timeout=args.connect_timeout)
    print(json.dumps({'boundary': {'pub_port': pub.port, 'sub_port': sink.port}}), flush=True)

    state = None
    cur_req = -1
    rows = []
    log = (out / 'cloud-requests.jsonl').open('w')

    def publish(req_id, ids, completed=False, eos_flag=False):
        verified.publish(1, req_id, ids, completed=completed, eos=eos_flag)

    while True:
        frame = drafts.wait_request(cur_req + 1, timeout=args.connect_timeout)
        if frame is None:
            print(json.dumps({'error': 'no request from edge'}), flush=True)
            break
        if frame.kind == KIND_CLOSE:
            print(json.dumps({'close': 'edge closed'}), flush=True)
            break
        if frame.req_id == cur_req:

            pass
        while frame.req_id > cur_req:
            cur_req = int(frame.req_id)
            if frame.ids.size < 2:
                print(json.dumps({'skip': 'short frame', 'req': cur_req}), flush=True)
                frame = drafts.wait_request(cur_req + 1, timeout=args.connect_timeout)
                if frame is None:
                    break
                continue
            cap = int(frame.ids[0])
            plen = int(frame.ids[1])
            prompt = torch.tensor([frame.ids[2:2 + plen].tolist()], dtype=torch.long,
                                  device=model.device)
            rec = {'req_id': cur_req, 'cap': cap, 'prompt_tokens': int(prompt.size(1)),
                   't_start': time.time()}
            state = decoding.step(model, prompt)
            publish(cur_req, state['output_ids'][0])
            n_verify = 0
            accepted = []
            done = False
            while not done:
                draft = drafts.wait_request(cur_req)
                if draft is None:
                    rec['error'] = 'draft timeout'
                    done = True
                    break
                if draft.req_id != cur_req or draft.kind == KIND_CLOSE:
                    break
                curr_len = state['output_ids'].size(1)
                prev_len = int(draft.ids.size) - 2 - plen
                if curr_len <= prev_len - args.lookahead + 1:
                    draft_ids = draft.ids[2 + plen:]
                    verify_inputs = decoding.build_verify_inputs(
                        torch.tensor([draft_ids.tolist()], dtype=torch.long),
                        state['input_ids'], state['output_ids'],
                        state['past_key_values'], state['num_input_tokens'])
                    state = decoding.verify_step(model, **verify_inputs)
                    n_verify += 1
                    accepted.append(int(state['num_verified_tokens']))
                    publish(cur_req, state['output_ids'][0])
                    nv = int(state['num_verified_tokens'])
                    tail = state['output_ids'][0, -nv:]
                    if stops:


                        gen = state['output_ids'][0]
                        text = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
                        hit = min((text.find(x) for x in stops if x in text), default=-1)
                        if hit >= 0:
                            keep = 0
                            for n in range(len(gen)):
                                if len(tokenizer.decode(
                                        gen[:n].tolist(), skip_special_tokens=True)) >= hit:
                                    keep = n
                                    break
                            state['output_ids'] = state['output_ids'][:, :keep]
                            publish(cur_req, state['output_ids'][0], completed=True)
                            rec['finish'] = 'stop'
                            done = True
                            break


                    if (not ignore_eos) and any(t in eos_ids for t in tail.tolist()):
                        _full = state['output_ids'][0].tolist()
                        eos_index = max(i for i, t in enumerate(_full) if t in eos_ids)
                        state['output_ids'] = state['output_ids'][:, :eos_index + 1]
                        publish(cur_req, state['output_ids'][0], completed=True, eos_flag=True)
                        rec['finish'] = 'eos'
                        done = True
                        break
                    if state['output_ids'].size(1) >= cap:


                        over = int(state['output_ids'].size(1)) - cap
                        if over > 0:
                            state['output_ids'] = state['output_ids'][:, :cap]
                            rec['overshoot_tokens'] = over
                        publish(cur_req, state['output_ids'][0], completed=True)
                        rec['finish'] = 'length'
                        done = True
                        break
                else:
                    newer = drafts.wait_newer(draft.ver, timeout=120.0)
                    if newer is None:
                        rec['error'] = 'stalled waiting for draft'
                        done = True
                        break
            rec.update({'n_verify': n_verify, 'accepted': accepted,
                        'gen_tokens': int(state['output_ids'].size(1)) if state else 0,
                        'wall_s': time.time() - rec.pop('t_start')})
            rows.append(rec)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(json.dumps({k: rec[k] for k in ('req_id', 'gen_tokens', 'n_verify', 'wall_s')
                              if k in rec}), flush=True)
            frame = drafts.latest()
            if frame is None or frame.req_id <= cur_req:
                frame = drafts.wait_request(cur_req + 1, timeout=args.connect_timeout)
                if frame is None:
                    break
    summary = {
        'requests': len(rows),
        'mean_gen_tokens': (sum(r['gen_tokens'] for r in rows) / len(rows)) if rows else None,
        'mean_verify_per_request': (sum(r['n_verify'] for r in rows) / len(rows)) if rows else None,
        'mean_accepted_per_verify': (
            sum(sum(r['accepted']) for r in rows) / max(1, sum(r['n_verify'] for r in rows))
            if rows else None),
        'errors': [r for r in rows if 'error' in r][:5],
    }
    (out / 'cloud-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('CLOUD_DONE', json.dumps(summary), flush=True)
    log.close()
    pub.close()
    sink.close()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
