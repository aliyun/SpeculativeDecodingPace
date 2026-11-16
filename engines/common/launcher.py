import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from .urls import http_url, loopback_host
from .growth import CHOICES, KS
from .policy import DRAFT_TOKENS, WIDTH, JOIN_MS, JOIN_WINDOWS, JOIN_RANGE, MAX_REQUESTS, PREFILL_REQUESTS
from .models import model_info, served_name, vocabulary
ROOT = Path(__file__).resolve().parents[2]

def plan(args):
    if args.engine != 'hf' or getattr(args, 'edge_backend', 'hf') != 'hf':
        raise ValueError('only HF cloud and HF edge are included')
    if os.environ.get('EC_PEER_CONFIG') or os.environ.get('EC_WEAK_EDGE_MULTIEDGE', '0') != '0':
        raise ValueError('only single-edge execution is included')
    if os.environ.get('TRIAGE_EDGE_BACKEND', 'hf') != 'hf':
        raise ValueError('only the HF edge backend is included')
    for key in ('TRIAGE_BATCH_GROWTH', 'TRIAGE_GROWTH_PACKED_TARGET', 'TRIAGE_GROWTH_TABLE_REUSE', 'TRIAGE_GROWTH_REPAIR_HOST', 'TRIAGE_GROWTH_REPAIR_LAYOUTS', 'TRIAGE_GROWTH_CLOUD_PREFILL', 'TRIAGE_GROWTH_READY_WAKE', 'TRIAGE_GROWTH_IMMUTABLE_VIEWS', 'TRIAGE_FORK_SELECTOR'):
        if os.environ.get(key, '0') != '0':
            raise ValueError('unsupported HF option: ' + key)
    name = served_name(args.model, getattr(args, 'served_model_name', None), engine='hf')
    metadata = model_info(args.model) if (Path(args.model) / 'config.json').is_file() else None
    hf_fused = getattr(args, 'hf_fused_draft', False)
    hf_draft_context = getattr(args, 'hf_draft_context', None)
    hf_draft_pool = getattr(args, 'hf_draft_pool', None)
    if type(hf_fused) is not bool:
        raise ValueError('invalid HF fused-draft switch')
    if hf_fused:
        if not args.role == 'edge' or metadata is None or metadata.family not in ('qwen3', 'qwen2'):
            raise ValueError('HF fused draft requires a local Qwen3 or Qwen2 HF edge')
        hf_draft_context = 1024 if hf_draft_context is None else hf_draft_context
        hf_draft_pool = 4 if hf_draft_pool is None else hf_draft_pool
        if type(hf_draft_context) is not int or not 256 <= hf_draft_context <= 32768:
            raise ValueError('HF draft context must be 256..32768')
        if type(hf_draft_pool) is not int or not 1 <= hf_draft_pool <= MAX_REQUESTS:
            raise ValueError('HF draft pool must be 1..16')
    elif hf_draft_context is not None or hf_draft_pool is not None:
        raise ValueError('HF draft bounds require --hf-fused-draft')
    hf_fixed = getattr(args, 'hf_fixed_slab', False)
    hf_fixed_context = getattr(args, 'hf_fixed_context', None)
    hf_fixed_graphs = getattr(args, 'hf_fixed_graphs', False)
    hf_prefill_first = getattr(args, 'hf_prefill_first_token', False)
    hf_prefill_buckets = getattr(args, 'hf_prefill_buckets', None)
    if hf_prefill_buckets is not None:
        from engines.hf.prefill_graph import parse_buckets
        parse_buckets(hf_prefill_buckets)
        if not hf_prefill_first:
            raise ValueError('HF prefill buckets require --hf-prefill-first-token')
    if type(hf_prefill_first) is not bool or (hf_prefill_first and (not hf_fixed)):
        raise ValueError('HF prefill-first requires --hf-fixed-slab')
    hf_fixed_attention = getattr(args, 'hf_fixed_attention', 'sdpa')
    if hf_fixed_attention not in ('sdpa', 'flash-kv'):
        raise ValueError('invalid HF fixed attention')
    if hf_fixed_attention != 'sdpa' and (not hf_fixed):
        raise ValueError('HF fixed attention requires --hf-fixed-slab')
    if type(hf_fixed_graphs) is not bool or (hf_fixed_graphs and (not hf_fixed)):
        raise ValueError('HF fixed graphs require --hf-fixed-slab')
    if type(hf_fixed) is not bool:
        raise ValueError('invalid HF fixed-slab switch')
    if hf_fixed:
        if args.role != 'cloud' or metadata is None or metadata.family not in ('qwen3', 'qwen2', 'gemma3', 'gemma3_text'):
            raise ValueError('HF fixed slab requires a local Qwen3, Qwen2 or Gemma3 HF cloud')
        hf_fixed_context = 1024 if hf_fixed_context is None else hf_fixed_context
        if type(hf_fixed_context) is not int or not 256 <= hf_fixed_context <= 32768:
            raise ValueError('HF fixed context must be 256..32768')
    elif hf_fixed_context is not None:
        raise ValueError('HF fixed context requires --hf-fixed-slab')
    join_ms = getattr(args, 'join_window_ms', JOIN_MS)
    wait_probe = getattr(args, 'wait_probe', True)
    if type(wait_probe) is not bool:
        raise ValueError('wait probe requires a boolean')
    asked_candidate = getattr(args, 'ar_candidate', None)
    if asked_candidate is not None and type(asked_candidate) is not bool:
        raise ValueError('AR candidate requires a boolean')
    if asked_candidate and args.role != 'cloud':
        raise ValueError('AR candidate is a cloud-side policy only')
    ar_candidate = True if asked_candidate is None else asked_candidate
    exact = getattr(args, 'exact_prefix_drafts', None)
    if exact is None:
        exact = args.role == 'cloud' and True and True and True
    elif type(exact) is not bool:
        raise ValueError('invalid exact-prefix draft switch')
    latency = getattr(args, 'latency_aware_prefill', False)
    if type(latency) is not bool or (latency and (not exact)):
        raise ValueError('latency-aware prefill requires exact-prefix drafts')
    if type(join_ms) is not int or not JOIN_RANGE[0] <= join_ms <= JOIN_RANGE[1]:
        raise ValueError('join window must be 10..1000 milliseconds')
    if join_ms != JOIN_MS and args.role != 'cloud':
        raise ValueError('join window changes require the cloud role')
    if args.role != 'cloud' and join_ms != JOIN_MS:
        raise ValueError('join experiments require cloud; overlap requires native-ar')
    if exact and (args.role != 'cloud' or False):
        raise ValueError('exact-prefix drafts require unified cloud')
    cache_seed = getattr(args, 'cache_seed', None)
    if cache_seed is not None:
        if not isinstance(cache_seed, str) or not cache_seed or args.role != 'cloud':
            raise ValueError('cache seed requires the cloud proposal cache')
        seed_path = Path(cache_seed).resolve()
        if not seed_path.is_file():
            raise ValueError('cache seed file does not exist')
        from .cache_seed import verify as verify_seed
        seed_summary = verify_seed(str(seed_path), args.model)
        cache_seed = str(seed_path)
    if os.environ.get('TRIAGE_OFFLOAD_EXPERIMENT') == '1':
        raise ValueError('HF engine has no native offload experiment')
    out = Path(args.out).resolve()
    if out.is_relative_to(ROOT) or out.exists():
        raise ValueError('--out must be a new directory outside source')
    port = args.port or (38100 if args.role == 'cloud' else 38101)
    if not 1024 <= port <= 65535:
        raise ValueError('nonprivileged port required')
    parsed = http_url(args.edge_url, origin=True)
    edge_port = parsed.port
    if edge_port is None or not 1024 <= edge_port <= 65535:
        raise ValueError('edge URL requires an explicit nonprivileged port')
    if args.role == 'cloud' and loopback_host(parsed.hostname) and edge_port == port:
        raise ValueError('loopback edge and cloud must use distinct ports')
    proposal_mode = getattr(args, 'proposal_cache', 'off')
    fingerprint_mode = getattr(args, 'fingerprint_cache', 'off')
    if proposal_mode not in ('off', 'exact') or (proposal_mode != 'off' and args.role != 'edge'):
        raise ValueError('--proposal-cache exact is edge-only')
    if fingerprint_mode not in ('off', 'snapshot', 'incremental'):
        raise ValueError('invalid fingerprint cache mode')
    if os.environ.get('TRIAGE_GROWTH_EXPERIMENT') == '1' and True:
        raise ValueError('GROWTH_EXPERIMENT_NOT_INCLUDED')
    adaptive = getattr(args, 'adaptive_growth', None)
    if adaptive is None:
        value = os.environ.get('TRIAGE_ADAPTIVE_GROWTH', '1')
        if value not in ('0', '1'):
            raise ValueError('ADAPTIVE_GROWTH_MODE')
        adaptive = value == '1'
    if type(adaptive) is not bool or (adaptive and (DRAFT_TOKENS != 3 or proposal_mode != 'off')):
        raise ValueError('ADAPTIVE_GROWTH_REQUIRES_BASE_K3_AND_NO_LEGACY_SHADOW_CACHE')
    legacy_selector = any((key in os.environ for key in ('TRIAGE_GROWTH_RULE', 'TRIAGE_GROWTH_RULE_FORK'))) or any((os.environ.get(key) == '1' for key in ('TRIAGE_GROWTH_EXPERIMENT', 'TRIAGE_BATCH_GROWTH', 'TRIAGE_UNIFIED_AUTO_GATE', 'TRIAGE_UNIFIED_RATE_GATE')))
    stream_default = '1' if adaptive and True and True and (not legacy_selector) and True else '0'
    stream_selector = os.environ.get('TRIAGE_STREAM_SELECTOR', stream_default)
    if stream_selector not in ('0', '1') or (stream_selector == '1' and (not adaptive or False or False or False or (os.environ.get('TRIAGE_GROWTH_EXPERIMENT') == '1'))):
        raise ValueError('STREAM_SELECTOR_REQUIRES_ADAPTIVE_HF')
    if stream_selector == '1' and any((os.environ.get(key) == '1' for key in ('TRIAGE_UNIFIED_AUTO_GATE', 'TRIAGE_UNIFIED_RATE_GATE'))):
        raise ValueError('STREAM_CACHE_LEGACY_GATE_UNSUPPORTED')
    rule_default = '1' if adaptive and False and True and True and False and (stream_selector != '1') else '0'
    growth_space = os.environ.get('TRIAGE_GROWTH_SPACE')
    if (os.environ.get('TRIAGE_GROWTH_KS') is not None or os.environ.get('TRIAGE_GROWTH_FORCE_K') is not None) and growth_space is None:
        raise ValueError('GROWTH_KS_REQUIRES_EXPLICIT_GROWTH_SPACE')
    if growth_space is None and (rule_default == '1' or stream_selector == '1'):
        growth_space = '3,4,5,6,7'
    if growth_space is not None:
        try:
            parsed = tuple(sorted({int(x) for x in str(growth_space).split(',') if x.strip()}))
        except (TypeError, ValueError):
            raise ValueError('GROWTH_SPACE') from None
        if not parsed or any((k < 1 or k > 9 for k in parsed)) or (not {3, 5, 7} <= set(parsed)) or (not adaptive) or False or False or (not (stream_selector == '1' and True and (max(parsed) <= 7))):
            raise ValueError('GROWTH_SPACE_REQUIRES_ADAPTIVE_HF_STREAM_SELECTOR')
    if stream_selector == '1' and any((not 3 <= k <= 7 for k in parsed)):
        raise ValueError('STREAM_SELECTOR_SPACE_REQUIRES_K3_TO_K7')
    transport = os.environ.get('TRIAGE_DRAFT_TRANSPORT', 'auto' if stream_selector == '1' else 'stream')
    if transport not in ('stream', 'sync', 'auto'):
        raise ValueError('DRAFT_TRANSPORT_MODE')
    if transport == 'auto' and stream_selector != '1':
        raise ValueError('AUTO_TRANSPORT_REQUIRES_STREAM_SELECTOR')
    transport_rule = 'COMMITTED_PROGRESS_MODE_SELECTION' if transport == 'auto' else None
    forced_k = os.environ.get('TRIAGE_GROWTH_FORCE_K')
    forced = None
    if forced_k is not None:
        try:
            forced = int(str(forced_k).strip())
        except (TypeError, ValueError):
            raise ValueError('GROWTH_FORCE_K') from None
        if not adaptive or forced not in parsed:
            raise ValueError('GROWTH_FORCE_K_REQUIRES_DECLARED_ADAPTIVE_SPACE')
    second_answer = os.environ.get('TRIAGE_SECOND_ANSWER', '1' if transport == 'auto' else '0')
    if second_answer not in ('0', '1'):
        raise ValueError('SECOND_ANSWER_FLAG')
    if second_answer == '1' and transport != 'auto':
        raise ValueError('SECOND_ANSWER_REQUIRES_AUTO_TRANSPORT')
    fork_selector = os.environ.get('TRIAGE_FORK_SELECTOR', '0')
    if fork_selector not in ('0', '1'):
        raise ValueError('FORK_SELECTOR_FLAG')
    if fork_selector == '1' and transport != 'sync':
        raise ValueError('FORK_SELECTOR_REQUIRES_SYNC')
    forced_roots = os.environ.get('TRIAGE_GROWTH_FORCE_ROOTS', '1')
    if forced_roots not in ('1', '2'):
        raise ValueError('GROWTH_FORCE_ROOTS')
    hedge_gate = os.environ.get('TRIAGE_HEDGE_GATE', '0')
    if hedge_gate not in ('0', '1'):
        raise ValueError('HEDGE_GATE')
    if forced_roots == '2':
        if args.role == 'cloud' and transport != 'sync' and not (transport == 'auto' and second_answer == '1') and hedge_gate != '1':
            raise ValueError('FORCED_ROOT_2_REQUIRES_SYNC')
        if hedge_gate == '1' and stream_selector != '1':
            raise ValueError('HEDGE_GATE_REQUIRES_STREAM_SELECTOR')
        if forced is None:
            raise ValueError('FORCED_ROOT_2_REQUIRES_FORCED_K')
    growth_ks = os.environ.get('TRIAGE_GROWTH_KS')
    chosen = None
    if growth_ks is not None:
        try:
            chosen = tuple(sorted({int(x) for x in str(growth_ks).split(',') if x.strip()}))
        except (TypeError, ValueError):
            raise ValueError('GROWTH_KS') from None
        if not chosen or not adaptive or (not set(chosen) <= set(parsed)):
            raise ValueError('GROWTH_KS_REQUIRES_DECLARED_ADAPTIVE_SPACE')
    fork_gap = os.environ.get('TRIAGE_GROWTH_FORK_GAP')
    if fork_gap is not None:
        try:
            gap_value = float(fork_gap)
        except (TypeError, ValueError):
            raise ValueError('GROWTH_FORK_GAP') from None
        if not math.isfinite(gap_value) or gap_value < 0 or (not adaptive):
            raise ValueError('GROWTH_FORK_GAP')
    fork_topk = os.environ.get('TRIAGE_GROWTH_FORK_TOPK_MS')
    if fork_topk is not None:
        try:
            topk_value = float(fork_topk)
        except (TypeError, ValueError):
            raise ValueError('GROWTH_FORK_TOPK_MS') from None
        if not math.isfinite(topk_value) or topk_value < 0 or (not adaptive):
            raise ValueError('GROWTH_FORK_TOPK_MS')
    fork_rpc = os.environ.get('TRIAGE_GROWTH_FORK_RPC_MS')
    if fork_rpc is not None:
        try:
            rpc_value = float(fork_rpc)
        except (TypeError, ValueError):
            raise ValueError('GROWTH_FORK_RPC_MS') from None
        if not math.isfinite(rpc_value) or rpc_value < 0 or (not adaptive):
            raise ValueError('GROWTH_FORK_RPC_MS')
    growth_rule = os.environ.get('TRIAGE_GROWTH_RULE', rule_default)
    if growth_rule not in ('0', '1') or (growth_rule == '1' and (not adaptive)):
        raise ValueError('GROWTH_RULE_REQUIRES_ADAPTIVE_GROWTH')
    rule_fork_default = '1' if growth_rule == '1' else '0'
    growth_rule_fork = os.environ.get('TRIAGE_GROWTH_RULE_FORK', rule_fork_default)
    if growth_rule_fork not in ('0', '1') or (growth_rule_fork == '1' and growth_rule != '1'):
        raise ValueError('GROWTH_RULE_FORK_REQUIRES_RULE')
    if cache_seed is not None and (not adaptive):
        raise ValueError('cache seed requires the adaptive unified draft cache')
    engine = ROOT / 'engines' / 'hf'
    env = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONSAFEPATH='1', PYTHONPATH=str(ROOT), TRIAGE_ENGINE_ROLE=args.role, TRIAGE_MODEL_PATH=args.model, TRIAGE_EXACT_PROPOSALS=proposal_mode, TRIAGE_FINGERPRINT_CACHE=fingerprint_mode)
    env['TRIAGE_ADAPTIVE_GROWTH'] = '1' if adaptive else '0'
    if stream_selector == '1':
        env['TRIAGE_STREAM_SELECTOR'] = '1'
    if growth_rule == '1':
        env['TRIAGE_GROWTH_RULE'] = '1'
    if growth_rule_fork == '1':
        env['TRIAGE_GROWTH_RULE_FORK'] = '1'
    if growth_space is not None:
        env['TRIAGE_GROWTH_SPACE'] = growth_space
    if growth_ks is not None:
        env['TRIAGE_GROWTH_KS'] = growth_ks
    draft_eager = os.environ.get('TRIAGE_DRAFT_EAGER', '0')
    draft_temp = os.environ.get('TRIAGE_DRAFT_TEMPERATURE', '0')
    if draft_eager not in ('0', '1'):
        raise ValueError('DRAFT_EAGER_FLAG')
    try:
        draft_temperature = float(draft_temp or 0)
    except (TypeError, ValueError):
        raise ValueError('DRAFT_TEMPERATURE') from None
    if draft_eager == '1' or draft_temperature > 0:
        raise ValueError('EAGER_DRAFT_NOT_INCLUDED')
    sample_graph = os.environ.get('TRIAGE_DRAFT_SAMPLE_GRAPH', '0')
    sample_seed = os.environ.get('TRIAGE_DRAFT_SAMPLE_SEED', '0')
    if sample_graph not in ('0', ''):
        try:
            sample_temperature = float(sample_graph)
        except (TypeError, ValueError):
            raise ValueError('DRAFT_SAMPLE_GRAPH') from None
        if not 0.0 < sample_temperature <= 2.0:
            raise ValueError('DRAFT_SAMPLE_GRAPH')
        raise ValueError('DRAFT_SAMPLE_GRAPH_NOT_INCLUDED')
    if transport == 'sync' and not adaptive:
        raise ValueError('SYNC_TRANSPORT_REQUIRES_ADAPTIVE_NATIVE_EDGE')
    env['TRIAGE_DRAFT_TRANSPORT'] = transport
    env['TRIAGE_SECOND_ANSWER'] = second_answer
    if forced is not None:
        env['TRIAGE_GROWTH_FORCE_K'] = str(forced)
    if fork_selector == '1':
        env['TRIAGE_FORK_SELECTOR'] = '1'
    for env_name in ('TRIAGE_SYNC_ACTION_RTT_MS', 'TRIAGE_SYNC_ACTION_C_MS', 'TRIAGE_SYNC_ACTION_EDGE_MS', 'TRIAGE_SYNC_ACTION_SWITCH_MS'):
        value = os.environ.get(env_name)
        if value is not None:
            try:
                float(value)
            except (TypeError, ValueError):
                raise ValueError('SYNC_ACTION_COST') from None
            env[env_name] = value
    for env_name in ('TRIAGE_HEDGE_GATE', 'TRIAGE_HEDGE_CLASS', 'TRIAGE_HEDGE_STATE', 'TRIAGE_HEDGE_GAP_SAMPLES', 'TRIAGE_HEDGE_ROUND_BUDGET', 'TRIAGE_HEDGE_BURNIN', 'TRIAGE_HEDGE_REWIND', 'TRIAGE_HEDGE_RATE_MIN', 'TRIAGE_HEDGE_DELTA_MIN', 'TRIAGE_HEDGE_LANE_FLOOR', 'TRIAGE_HEDGE_MIN_OBS', 'TRIAGE_ADMIT_SYNC', 'TRIAGE_ADMIT_SYNC_FORCE', 'TRIAGE_ADMIT_MARGIN', 'TRIAGE_ADMIT_MIN_TOKENS', 'TRIAGE_PLAN_TIMING', 'TRIAGE_HEDGE_DELTA_ONLINE', 'TRIAGE_HEDGE_ONCE', 'TRIAGE_HEDGE_BETA_MIN', 'TRIAGE_HEDGE_EXPLORE'):
        value = os.environ.get(env_name)
        if value is not None:
            env[env_name] = value
    if forced_roots == '2':
        env['TRIAGE_GROWTH_FORCE_ROOTS'] = '2'
    if fork_gap is not None:
        env['TRIAGE_GROWTH_FORK_GAP'] = fork_gap
    if fork_topk is not None:
        env['TRIAGE_GROWTH_FORK_TOPK_MS'] = fork_topk
    if fork_rpc is not None:
        env['TRIAGE_GROWTH_FORK_RPC_MS'] = fork_rpc
    if hf_fused:
        env.update(TRIAGE_HF_FUSED_DRAFT='1', TRIAGE_HF_DRAFT_CONTEXT=str(hf_draft_context), TRIAGE_HF_DRAFT_POOL=str(hf_draft_pool), TRIAGE_HF_STATIC='0', TRIAGE_HF_COMPILE='0')
    if hf_fixed:
        env.update(TRIAGE_HF_FIXED_SLAB='1', TRIAGE_HF_FIXED_CONTEXT=str(hf_fixed_context), TRIAGE_HF_FIXED_ATTENTION=hf_fixed_attention)
        if hf_fixed_graphs:
            env['TRIAGE_HF_FIXED_GRAPHS'] = '1'
        if hf_prefill_first:
            env['TRIAGE_HF_PREFILL_FIRST'] = '1'
        if hf_prefill_buckets is not None:
            env['TRIAGE_HF_PREFILL_BUCKETS'] = hf_prefill_buckets
    if cache_seed is not None:
        env['TRIAGE_CACHE_SEED'] = cache_seed
    required_backend_files, completion_file = ({}, None)
    stream_drafts = args.role == 'cloud'
    stream_policy = getattr(args, 'stream_policy', 'cohort')
    if stream_policy != 'cohort':
        raise ValueError('cohort-pipeline is the only proposal policy')
    stream_rows = getattr(args, 'stream_rows', None)
    if stream_rows is not None and (type(stream_rows) is not int or not 1 <= stream_rows <= MAX_REQUESTS or (not stream_drafts)):
        raise ValueError('CPU stream rows requires stream drafts and a bound of 1..%d' % MAX_REQUESTS)
    env.update(TRIAGE_EDGE_BACKEND='hf')
    if args.role == 'edge':
        command = [sys.executable, '-B', '-m', 'engines.hf.edge', '--model', args.model, '--out', str(out), '--port', str(port), '--host', args.host]
    else:
        if exact:
            env['EC_EXACT_PREFIX_DRAFTS'] = '1'
        if latency:
            env['EC_LATENCY_AWARE_PREFILL'] = '1'
        env.update(EC_CPU_STREAM_DRAFTS='1', EC_STREAM_POLICY='cohort')
        if stream_rows is not None:
            env['EC_CPU_STREAM_ROWS'] = str(stream_rows)
        command = [sys.executable, '-B', '-m', 'engines.hf.cloud', '--model', args.model, '--out', str(out), '--port', str(port), '--edge-url', args.edge_url, '--served-model-name', name, '--host', args.host]
    policy = dict(proposal_strategy='cohort-pipeline', draft_tokens=DRAFT_TOKENS, width=WIDTH, join_ms=join_ms, wait_probe=wait_probe, max_requests=MAX_REQUESTS, prefill_requests=PREFILL_REQUESTS, local_draft=False, load_aware=False, ar_min_batch=None, proposal_cache=proposal_mode, fingerprint_cache=fingerprint_mode)
    if cache_seed is not None:
        policy['cache_seed'] = dict(path=cache_seed, **seed_summary)
    if ar_candidate:
        policy['ar_candidate'] = True
    if exact:
        policy.update(exact_prefix_drafts=True, exact_cache_max_entries=8192, exact_cache_max_token_slots=4194304)
    if latency:
        policy.update(latency_aware_prefill=True, ready_prefill_requests=1, decode_opportunity_after_prefill=True)
    if stream_drafts and True:
        policy['stream_drafts'] = True if adaptive and True else not adaptive
    if stream_policy != 'never-wait':
        policy['stream_policy'] = stream_policy
    if stream_rows is not None:
        policy['stream_rows'] = stream_rows
    unset = []
    if cache_seed is None:
        unset.append('TRIAGE_CACHE_SEED')
    if args.role == 'cloud' and True and True:
        env['EC_JOIN_MS'] = str(join_ms)
        env['EC_WAIT_PROBE'] = '1' if wait_probe else '0'
        env['EC_AR_CANDIDATE'] = '1' if ar_candidate else '0'
    if args.role == 'cloud':
        policy.update(target_engine='hf', target_verify='GREEDY_CHAIN_CAUSAL_FORWARD', target_kv='PADDED_SHARED_SLAB', target_batching='PADDED_SLAB_BATCH', target_cuda_graph=False, target_tree=adaptive, target_batch_invariant=False)
    if hf_fixed:
        policy.update(target_kv='FIXED_SCATTER_BANK', target_batching='FIXED_SCATTER_BUCKETS_1_4_16', target_torch_compile=True, effective_context=hf_fixed_context, warm_query_widths=[1, 4, 5, 6, 7, 8, 9] if adaptive else [1, WIDTH], fixed_batch_buckets=[1, 4, 16], target_cuda_graph=hf_fixed_graphs, target_attention=hf_fixed_attention, target_prefill_first_token=hf_prefill_first, prefill_buckets=hf_prefill_buckets)
    policy.update(edge_backend='hf', edge_execution='TRANSFORMERS_PER_ROW_FORWARD' if args.role == 'edge' else 'EXTERNAL_HF_EDGE', edge_cuda_graph=False, edge_fused_steps=False)
    if hf_fused:
        policy.update(edge_execution='TRANSFORMERS_DECLARED_K_ROOT_TOP2_CUDA_GRAPH' if adaptive else 'TRANSFORMERS_K3_CUDA_GRAPH', edge_cuda_graph=True, edge_fused_steps=True, effective_edge_context=hf_draft_context, edge_cache_pool=hf_draft_pool)
    result = dict(command=command, env=env, unset_env=unset, out=str(out), engine='hf', process_engine='hf', role=args.role, fused_draft=False, required_backend_files=required_backend_files, required_completion_file=completion_file, policy=policy)
    if hedge_gate == '1':
        result['policy']['hedge_gate'] = dict(enabled=True, force_roots=forced_roots, decided='per_request_from_root_gap', stream_lane_branches=1)
    if adaptive:
        result['policy'].update(adaptive_growth=True, growth=dict(protocol=1, ks=list(KS), roots=[1, 2]), candidate_K=[k for k in KS if k in CHOICES], root_candidates=[1, 2], selector='live_source_cost', declared_ks=list(KS), growth_rule=growth_rule == '1', growth_rule_fork=growth_rule_fork == '1', fork_gap=fork_gap, fork_topk_ms=fork_topk, fork_rpc_ms=fork_rpc, context_fallback='AR_BEFORE_NATIVE_SUBMISSION', plan_required=True, draft_transport='SYNC_BATCH_PER_ROUND' if transport == 'sync' else 'BOUNDED_PLAN_RPC_WITH_BASELINE_STREAM', draft_transport_rule=transport_rule, draft_sample_graph=env.get('TRIAGE_DRAFT_SAMPLE_GRAPH'), choice_ks=list(chosen) if chosen is not None else None, forced_k=forced, forced_roots=int(forced_roots), sync_action=dict(enabled=env.get('TRIAGE_SECOND_ANSWER') == '1', scored_by='E[N_a|s]/E[T_a|s]', terms=['C', 'W_sync', 'c_node*b', 'c_mask', 'c_shape', 'split_loss', 'B/remaining'], c_ms=env.get('TRIAGE_SYNC_ACTION_C_MS'), rtt_ms=env.get('TRIAGE_SYNC_ACTION_RTT_MS')), fork_selector=dict(enabled=env.get('TRIAGE_FORK_SELECTOR') == '1', n_min=12, explore_fraction=0.1, max_consecutive=4, max_length=3, max_fork_at='K-1', cold_start='no_fork', fallback='no_fork'), qualification='EXPERIMENTAL_NOT_HARDWARE_QUALIFIED')
        from engines.hf.fixed_slab import bank_rows
        hf_ks = parsed if stream_selector == '1' else KS
        result['policy'].update(target_verify='HF_DECLARED_CHAIN_AND_ANCESTOR_TREE', target_verify_batch_max=8, resident_requests=MAX_REQUESTS, declared_query_widths=[1, 4, 5, 6, 7, 8, 9], chain_attention=hf_fixed_attention if hf_fixed else 'sdpa', real_side_attention='DECLARED_SDPA_NOT_ERROR_FALLBACK', edge_graphs=4 * len(hf_ks) * hf_draft_pool if hf_fused else None, target_graphs=len(bank_rows()) * (1 + 2 * len(hf_ks)) if hf_fixed_graphs else None)
    else:
        result['policy']['adaptive_growth'] = False
    if stream_selector == '1':
        transport_labels = dict(sync='SYNC_BATCH_PER_ROUND', stream='BOUNDED_CONTINUOUS_STREAM',
                                auto='ADAPTIVE_STREAM_OR_SYNC')
        sync_roots = [2] if second_answer == '1' else [1]
        result['policy'].update(selector='stream_event_rollout_v2', streaming_selector=True, growth=dict(protocol=1, ks=list(parsed), roots=[1, 2]), candidate_K=list(parsed), declared_ks=list(parsed), root_candidates=[1, 2] if transport == 'sync' or second_answer == '1' else [1], draft_transport=transport_labels[transport], plan_binding='READY_BEFORE_NATIVE_DISPATCH', draft_transport_rule=transport_rule, choice_ks=list(chosen) if chosen is not None else None, planner='BOUNDED_H2_RATE_ROOT' if env.get('EC_AR_CANDIDATE') == '1' else 'READY_RATE_ONLY', ar_bootstrap_max_decode_steps=16 if env.get('EC_AR_CANDIDATE') == '1' else 0)
        if transport == 'auto':
            result['policy']['sync_action'].update(enabled=True, root_candidates=sync_roots)
    if stream_selector == '1':
        result['policy'].update(concurrent_K=list(parsed), concurrent_verify_batches=list(range(1, 9)), concurrent_planner='FIFO_CONSTRAINED_BATCH_RATE', concurrent_wait='EVENT_OR_EXISTING_HARD_DEADLINE_NOT_IDENTIFIED_JOINT_H2', concurrent_max_candidates=40)
    if metadata is not None:
        result['model_profile'] = dict(family=metadata.family, context=metadata.context, eos_ids=list(metadata.eos_ids), unified_kv=metadata.gemma3)
    return result

def validate_native(plan):
    torch_version = importlib.metadata.version('torch').split('+')[0]
    if torch_version != '2.11.0':
        raise RuntimeError(f'expected torch 2.11.0, found {torch_version}')
    found = importlib.metadata.version('transformers').split('+')[0]
    try:
        major, minor = (int(part) for part in found.split('.')[:2])
    except ValueError as exc:
        raise RuntimeError('unreadable transformers version: ' + found) from exc
    if major != 5 or minor < 12:
        raise RuntimeError(f'expected transformers >=5.12,<6, found {found}')
    if importlib.util.find_spec('transformers') is None:
        raise RuntimeError('cannot locate transformers')
    return

def execute(plan):
    env = os.environ.copy()
    env.pop('EC_EXACT_PREFIX_DRAFTS', None)
    env.pop('EC_LATENCY_AWARE_PREFILL', None)
    env.pop('EC_EDGE_GEMM', None)
    env.pop('EC_CPU_PROMPT_WARM', None)
    env.pop('EC_CPU_STREAM_DRAFTS', None)
    env.pop('EC_CPU_STREAM_ROWS', None)
    env.pop('EC_STREAM_POLICY', None)
    env.pop('TRIAGE_WEBGPU_TOKENIZER', None)
    env.pop('TRIAGE_HF_FIXED_SLAB', None)
    env.pop('TRIAGE_HF_FIXED_CONTEXT', None)
    env.pop('TRIAGE_HF_FIXED_GRAPHS', None)
    env.pop('TRIAGE_HF_FIXED_ATTENTION', None)
    env.pop('TRIAGE_HF_FUSED_DRAFT', None)
    env.pop('TRIAGE_HF_DRAFT_CONTEXT', None)
    env.pop('TRIAGE_HF_DRAFT_POOL', None)
    env.pop('TRIAGE_HF_PREFILL_FIRST', None)
    env.pop('TRIAGE_HF_PREFILL_BUCKETS', None)
    for key in ('EC_WEAK_EDGE_MULTIEDGE', 'EC_PEER_CONFIG', 'EC_PEER_BACKEND', 'EC_PEER_CONTEXT'):
        env.pop(key, None)
    env.pop('TRIAGE_ADAPTIVE_GROWTH', None)
    env.pop('TRIAGE_GROWTH_PACKED_TARGET', None)
    env.pop('TRIAGE_GROWTH_READY_WAKE', None)
    env.pop('TRIAGE_GROWTH_IMMUTABLE_VIEWS', None)
    env.update(plan['env'])
    model_identity = None
    model = plan['env']['TRIAGE_MODEL_PATH']
    context = 32768
    model_identity = vocabulary(model, context).identity()
    for key in plan['unset_env']:
        env.pop(key, None)
    out = Path(plan['out'])
    out.mkdir(parents=True)
    if model_identity is not None:
        (out / 'MODEL_IDENTITY.json').write_text(json.dumps(dict(scope='STARTUP_TOKEN_IDENTITY_NOT_WEIGHT_OR_GPU_QUALIFICATION', model=plan['env']['TRIAGE_MODEL_PATH'], **model_identity), indent=2) + '\n')
    subprocess.run([plan['command'][0], '-B', '-c', 'import json,sys; from engines.common.launcher import validate_native; validate_native(json.load(sys.stdin))'], input=json.dumps(plan), text=True, env=env, cwd=ROOT, check=True)
    (out / 'LAUNCH.json').write_text(json.dumps(plan, indent=2) + '\n')
    with (out / 'service.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.execvpe(plan['command'][0], plan['command'], env)

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--engine', choices=('hf',), required=True)
    parser.add_argument('--role', choices=('cloud', 'edge'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--served-model-name', help='public model alias; defaults to the model directory name')
    parser.add_argument('--host', choices=('0.0.0.0',), required=True)
    parser.add_argument('--port', type=int)
    parser.add_argument('--edge-url', default='http://127.0.0.1:38101')
    parser.add_argument('--join-window-ms', type=int, default=JOIN_MS, help='cloud cohort wait window in milliseconds (10..1000; default 100)')
    parser.add_argument('--wait-probe', action=argparse.BooleanOptionalAction, default=True, help='cloud: raise the window once from the measured draft-arrival tail (default on; --no-wait-probe keeps the fixed window)')
    parser.add_argument('--ar-candidate', action=argparse.BooleanOptionalAction, default=None, help='cloud: let cloud AR compete on rate instead of only on a deadline; the row takes AR this round when a measured AR step is faster than the rounds this deployment is running (default on; --no-ar-candidate keeps AR on the deadline only)')
    parser.add_argument('--exact-prefix-drafts', action=argparse.BooleanOptionalAction, default=None, help='default on for HF cloud; --no-exact-prefix-drafts disables only this proposal cache')
    parser.add_argument('--cache-seed', help='cloud: load one committed-token seed file into the unified cache before ready')
    parser.add_argument('--latency-aware-prefill', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--proposal-cache', choices=('off', 'exact'), default='off')
    parser.add_argument('--fingerprint-cache', choices=('off', 'snapshot', 'incremental'), default='off')
    parser.add_argument('--adaptive-growth', action=argparse.BooleanOptionalAction, default=None, help='default on for the HF draft backend; --no-adaptive-growth restores fixed K3')
    parser.add_argument('--hf-fused-draft', action='store_true', help='experimental BF16 Qwen3/Qwen2 HF edge: declared K and root-top2 CUDA graph pool')
    parser.add_argument('--hf-draft-context', type=int, help='fused draft cache bound (default 1024); requires --hf-fused-draft')
    parser.add_argument('--hf-draft-pool', type=int, help='fused draft live cache pool (default 4); requires --hf-fused-draft')
    parser.add_argument('--hf-fixed-slab', action='store_true', help='experimental Qwen3/Qwen2/Gemma3 HF cloud: fixed batched KV bank with compiled decode/verify')
    parser.add_argument('--hf-fixed-graphs', action='store_true', help='experimental fixed-slab CUDA graph replay; requires --hf-fixed-slab')
    parser.add_argument('--hf-fixed-attention', choices=('sdpa', 'flash-kv'), default='sdpa', help='experimental BF16 fixed-bank attention; ordinary prefill stays SDPA')
    parser.add_argument('--hf-prefill-buckets', help='experimental bounded prefill graphs, e.g. 16,32,48,64,80,96,112,128; above max uses counted eager')
    parser.add_argument('--hf-prefill-first-token', action='store_true', help='experimental fixed HF cloud: keep full prompt KV and reuse prefill logits')
    parser.add_argument('--hf-fixed-context', type=int, help='fixed KV context bound (default 1024); requires --hf-fixed-slab')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--edge-backend', choices=('hf',), default='hf', help='explicit draft backend; target remains on GPU')
    parser.add_argument('--stream-policy', choices=('cohort',), default='cohort', help='compatibility alias: cohort-pipeline is always the sole cloud policy')
    parser.add_argument('--stream-rows', type=int, help='cloud: maximum concurrent draft streams (1..16)')
    args = parser.parse_args()
    try:
        result = plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        execute(result)
