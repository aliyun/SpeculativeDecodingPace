import time
import torch
from baselines.fsd import generate_fsd
ARM = {'name': 'sd-g4', 'kind': 'sd', 'do_sample': False, 'fsd_threshold': 0.0, 'fsd_div_type': 'exact', 'gamma': 4}

def configure(target, draft):
    draft.generation_config.do_sample = False
    for model in (draft, target):
        model.generation_config.num_assistant_tokens = ARM['gamma']
        model.generation_config.num_assistant_tokens_schedule = 'constant'

def vocab_width(model):
    getter = getattr(model.config, 'get_text_config', None)
    cfg = getter() if callable(getter) else model.config
    return getattr(cfg, 'vocab_size', None)

def assist_kwargs(target, draft, tokenizer):
    if vocab_width(target) == vocab_width(draft):
        return {}
    return {'tokenizer': tokenizer, 'assistant_tokenizer': tokenizer}

def run_one(arm, target, draft, tokenizer, ids, eos, max_new_tokens, seed, stopping_criteria=None):
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=arm['do_sample'], eos_token_id=eos, pad_token_id=eos[0])
    if arm['do_sample']:
        gen_kwargs['temperature'] = arm['temperature']
        gen_kwargs['top_p'] = 1.0
        gen_kwargs['top_k'] = 0
    if stopping_criteria is not None:
        gen_kwargs['stopping_criteria'] = stopping_criteria
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = generate_fsd.generate(target, inputs=ids, attention_mask=torch.ones_like(ids), assistant_model=draft, fsd_threshold=arm['fsd_threshold'], fsd_div_type=arm['fsd_div_type'], track_acceptance_metrics=True, return_dict_in_generate=True, **gen_kwargs, **assist_kwargs(target, draft, tokenizer))
        rate = getattr(out, 'draft_token_acceptance_rate', None)
        draft_total = getattr(out, 'total_draft_tokens', None)
        accepted_total = getattr(out, 'total_accepted_tokens', None)
        acceptance = {'draft_token_acceptance_rate': float(rate) if rate is not None else None, 'total_draft_tokens': int(draft_total) if draft_total is not None else None, 'total_accepted_tokens': int(accepted_total) if accepted_total is not None else None}
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    seq = out.sequences if hasattr(out, 'sequences') else out
    new_ids = seq[0][ids.shape[1]:].tolist()
    del out
    torch.cuda.empty_cache()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return (new_ids, text, wall, acceptance)
