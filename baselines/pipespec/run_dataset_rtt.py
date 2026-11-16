#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def eos_id_set(tokenizer):
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
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='supplement paper root (has dataset/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--draft', required=True)
    ap.add_argument('--target', required=True, help='target tokenizer (vocab identity only)')
    ap.add_argument('--cloud-host', required=True)
    ap.add_argument('--cloud-port', type=int, required=True, help='cloud -> edge stream')
    ap.add_argument('--draft-port', type=int, required=True, help='edge -> cloud stream')
    ap.add_argument('--limit', type=int, default=512)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--lookahead', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--warmup-tokens', type=int, default=32)
    ap.add_argument('--warmup-id-max', type=int, default=20000,
                    help='upper bound of the synthetic warmup ids (CPU smoke uses a tiny vocab)')
    ap.add_argument('--min-free-gib', type=float, default=8.0)
    ap.add_argument('--max-consecutive-errors', type=int, default=3)
    ap.add_argument('--load', required=True, help='R01..R10 or S01..S04')
    ap.add_argument('--pools', required=True, help='frozen pools.json (R loads)')
    ap.add_argument('--ignore-eos', action='store_true', help='forced-output loads')
    ap.add_argument('--arm', default='async-chain-rtt')
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE))

    import decoding  # noqa: E402
    from rtt_transport import (DrainClient, LatestSlot, SenderClient,  # noqa: E402
                               KIND_NEW_REQUEST, KIND_VALUE, KIND_CLOSE, as_i8)
    import helpers as U  # noqa: E402
    import dataset_pools as DP  # noqa: E402

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.target), local_files_only=True)
    load = DP.load_pool(args.pools, args.load, generations=512, micro_generations=128,
                        tokenizer=tokenizer)
    rows, order = load.rows, load.order
    forced = load.forced
    if forced and not args.ignore_eos:
        raise SystemExit('forced-output load %s needs --ignore-eos' % args.load)
    golds = [U.gsm8k_gold(r) for r in rows] if args.load == 'R05' else [None] * len(rows)

    eos = tokenizer.eos_token_id
    eos_set = eos_id_set(tokenizer)

    device = os.environ.get('PIPESPEC_DEVICE', 'cuda:0')
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    draft_model, _ = decoding.init_model(
        decoding.ModelConfig(model_path=args.draft, devices=device, dtype=dtype))
    draft_model.eval()

    cloud_verified = LatestSlot('edge-cloud-verified')
    our_drafts = LatestSlot('edge-drafts')
    sub = DrainClient(cloud_verified, args.cloud_host, args.cloud_port, name='edge-drain',
                      timeout_s=1800.0)
    pub = SenderClient(our_drafts, args.cloud_host, args.draft_port, name='edge-send',
                       timeout_s=1800.0)
    print(json.dumps({'edge_wired': True, 'cloud': f'{args.cloud_host}:{args.cloud_port}',
                      'draft_port': args.draft_port}), flush=True)

    env = {
        'arm': args.arm,
        'transport': 'tcp latest-value streams, one container per stage',
        'upstream': 'BradMcDanel/PipeSpec @ 9a64774 decoding.py unmodified (helpers only)',
        'torch': torch.__version__,
        'transformers': __import__('transformers').__version__,
        'vendor_decoding_sha256': U.sha256(Path(decoding.__file__).resolve()),
        'runner_sha256': U.sha256(Path(__file__).resolve()),
        'transport_sha256': U.sha256(HERE / 'rtt_transport.py'),
        'cloud_stage_sha256': U.sha256(HERE / 'cloud_stage.py'),
        'load': args.load, 'forced_output': forced,
        'pools_sha256': U.sha256(Path(args.pools)),
        'replay_rule': 'random.Random(20261020) shuffle(range(80)); sample = rows[order[i % 80]]',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_names': ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                         if torch.cuda.is_available() else ['cpu']),
        'draft': args.draft, 'target': args.target,
        'limit': args.limit, 'max_new_tokens': args.max_new_tokens,
        'lookahead': args.lookahead, 'warmup': args.warmup,
        'cloud': f'{args.cloud_host}:{args.cloud_port}', 'draft_port': args.draft_port,
    }
    U.save_json(out / 'ENV.json', env)
    U.save_json(out / 'READY.json', {'eos_token_id': eos, 'ready_s': time.time()})
    print(json.dumps({'ready': True, 'eos': eos}), flush=True)

    req_id = [0]

    def generate(prompt_ids, cap, timeout_s=3600.0):


        rid = req_id[0]
        req_id[0] += 1
        prompt_np = np.asarray(prompt_ids[0].tolist(), dtype='<i8')

        def payload_of(draft_ids):


            return np.concatenate([np.asarray([cap, prompt_np.size], dtype='<i8'),
                                   prompt_np, as_i8(draft_ids)])

        our_drafts.publish(KIND_NEW_REQUEST, rid, payload_of(np.empty(0, dtype='<i8')))
        state = decoding.step(draft_model, prompt_ids.to(draft_model.device))
        our_drafts.publish(KIND_VALUE, rid, payload_of(state['output_ids'][0]))
        acked = 0
        deadline = time.time() + timeout_s
        while True:


            f = cloud_verified.latest()
            if f is not None and f.req_id > rid:
                raise RuntimeError('cloud moved past request %s to %s' % (rid, f.req_id))
            if f is not None and f.req_id == rid:
                if f.completed:
                    return [int(x) for x in f.ids]
                if f.ids.size > acked:


                    acked = int(f.ids.size)
                    state = decoding.build_draft_inputs(
                        torch.tensor([f.ids.tolist()], dtype=torch.long),
                        state['input_ids'], state['output_ids'],
                        state['past_key_values'], state['num_input_tokens'])
                    our_drafts.publish(KIND_VALUE, rid, payload_of(state['output_ids'][0]))
            elif cloud_verified.closed:
                raise RuntimeError('boundary closed while waiting for request %s' % rid)
            elif time.time() > deadline:
                raise RuntimeError('cloud silent for request %s' % rid)
            state = decoding.step(draft_model, state['input_ids'], state['output_ids'],
                                  state['past_key_values'], state['num_input_tokens'])
            our_drafts.publish(KIND_VALUE, rid, payload_of(state['output_ids'][0]))
            if time.time() > deadline:
                raise RuntimeError('request timeout after %.0fs' % timeout_s)
            if time.time() > deadline:
                raise RuntimeError('request timeout after %.0fs' % timeout_s)

    warm_ok = 0
    for w in range(args.warmup):
        ids = torch.randint(100, args.warmup_id_max, (1, 256), dtype=torch.long)
        try:
            generate(ids, args.warmup_tokens)
            warm_ok += 1
        except Exception as exc:
            print(json.dumps({'warmup_error': repr(exc)[:300]}), flush=True)
    env['warmup'] = args.warmup
    env['warmup_tokens'] = args.warmup_tokens
    env['warmup_ok'] = warm_ok
    U.save_json(out / 'ENV.json', env)
    print(json.dumps({'warmup_ok': warm_ok, 'of': args.warmup, 'cap': args.warmup_tokens}), flush=True)

    records = []
    consecutive = 0
    aborted = None
    t_arm = time.perf_counter()
    with (out / 'requests.jsonl').open('w') as fh:
        for i, row_index in enumerate(order[:args.limit]):
            free_b = (torch.cuda.mem_get_info(0)[0] if torch.cuda.is_available() else 1 << 40)
            if free_b < args.min_free_gib * (1 << 30):
                aborted = 'low-free-memory before i=%d' % i
                break
            row = rows[row_index]
            cap = int(row.get('max_tokens') or args.max_new_tokens)
            first = DP.first_turn_ids(row)
            ids = torch.tensor([first], dtype=torch.long)
            rec = {'i': i, 'load': args.load, 'pool_index': row_index,
                   'sample_id': row['id'], 'input_tokens': ids.shape[1], 'cap': cap}
            try:
                t0 = time.perf_counter()
                new_ids = generate(ids, cap)
                if args.load == 'R01':


                    prev_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    second = DP.second_turn_ids(tokenizer, row, prev_text)
                    new_ids2 = generate(torch.tensor([second], dtype=torch.long), cap)
                    new_ids = list(new_ids) + list(new_ids2)
                    rec['input_tokens'] += len(second)
                    rec['turn_tokens'] = [len(new_ids) - len(new_ids2), len(new_ids2)]
                wall = time.perf_counter() - t0

                budget = cap * (2 if args.load == 'R01' else 1)
                overshoot = max(0, len(new_ids) - budget)
                if overshoot:
                    new_ids = new_ids[:budget]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                pred, gold = (U.gsm8k_pred(text), golds[row_index]) if args.load == 'R05' else (None, None)
                if forced:
                    finish = 'length'
                elif args.load == 'R07':


                    if new_ids and new_ids[-1] in eos_set:
                        finish = 'eos'
                    elif len(new_ids) < cap:
                        finish = 'stop'
                    else:
                        finish = 'length'
                else:
                    finish = 'eos' if new_ids and new_ids[-1] in eos_set else (
                        'length' if len(new_ids) >= cap else 'other')
                rec.update({
                    'status': 'OK', 'new_tokens': len(new_ids), 'wall_s': wall,
                    'tok_per_s': len(new_ids) / wall if wall else None,
                    'finish': finish, 'overshoot_tokens': overshoot,
                    'output_ids_sha256': U.sha256(
                        json.dumps(new_ids, separators=(',', ':')).encode()),
                    'text': text, 'gsm8k_pred': pred, 'gsm8k_gold': gold,
                    'gsm8k_correct': (None if gold is None
                                      else (pred is not None and pred == gold)),
                })
            except Exception as exc:
                rec.update({'status': 'ERROR', 'error': repr(exc)[:800]})
                consecutive += 1
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
                if consecutive >= args.max_consecutive_errors:
                    aborted = 'consecutive-errors %d at i=%d' % (consecutive, i)
                    break
                continue
            consecutive = 0
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
            if (i + 1) % 8 == 0 or i + 1 == args.limit:
                ok = [r for r in records if r['status'] == 'OK']
                spent = sum(r['wall_s'] for r in ok)
                toks = sum(r['new_tokens'] for r in ok)
                print(json.dumps({'arm': args.arm, 'done': i + 1, 'ok': len(ok),
                                  'tok_per_s': (toks / spent) if spent else None,
                                  'elapsed_s': time.perf_counter() - t_arm}), flush=True)

    summary = U.summarize(args.arm, records, args.max_new_tokens)
    summary['load'] = args.load
    summary['arm_elapsed_s'] = time.perf_counter() - t_arm
    summary['aborted_reason'] = aborted
    summary['complete'] = (len([r for r in records if r.get('status') == 'OK']) == args.limit)
    summary['planned_generations'] = args.limit
    summary['warmup_ok'] = warm_ok
    summary['warmup_tokens'] = args.warmup_tokens
    U.save_json(out / 'summary.json', summary)
    print('ARM_DONE', json.dumps(summary), flush=True)
    try:
        our_drafts.publish(KIND_CLOSE, req_id[0], np.empty(0, dtype='<i8'))
        time.sleep(0.5)
    finally:
        pub.close()
        sub.close()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
