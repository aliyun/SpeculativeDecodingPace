import time
import torch
ARM = {'name': 'ar-greedy', 'kind': 'ar', 'do_sample': False, 'fsd_threshold': 0.0, 'fsd_div_type': 'js'}

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
        out = target.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **gen_kwargs)
        acceptance = {}
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    seq = out.sequences if hasattr(out, 'sequences') else out
    new_ids = seq[0][ids.shape[1]:].tolist()
    del out
    torch.cuda.empty_cache()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return (new_ids, text, wall, acceptance)
