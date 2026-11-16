import torch
import util
from config import SpecEdgeClientConfig as config
from specedge.client.specexec import SpecExecClient
from specedge.tree import Tree

class DatasetClient(SpecExecClient):

    def set_prompt_ids(self, ids, max_len, prompt_text=''):
        if len(ids) > max_len:
            raise ValueError('prompt %d ids exceeds max_len %d' % (len(ids), max_len))
        self._prompt = prompt_text
        self._prefix_tokens = torch.tensor([ids], dtype=torch.long, device=self._device)
        self._num_original_tokens = int(self._prefix_tokens.numel())
        self._tree = Tree(prefix_tokens=self._prefix_tokens, device=self._device, dtype=self._dtype, max_len=self._engine.max_len)
        if self._proactive_type != 'disabled':
            from specedge.client.proactive import SpecExecProactiveDraft
            self._proactive_client = SpecExecProactiveDraft(tree=self._tree, engine=self._engine, max_len=self._max_len)

    async def generate(self, req_idx, cap, ignore_eos=False, stops=()):
        util.set_seed(config.seed)
        step_idx = 0
        warmup_tokens = await self._cycle(req_idx, step_idx, prefill=True)
        self._prefix_tokens = torch.cat([self._prefix_tokens, warmup_tokens], dim=-1)
        step_idx = 1
        finished = False
        stop_hit = False
        eos_ids = [] if ignore_eos else _eos_id_set(self._tokenizer)
        eos = eos_ids[0] if eos_ids else -1
        while not finished and (not stop_hit):
            produced = int(self._prefix_tokens.numel() - self._num_original_tokens - warmup_tokens.numel())
            if produced >= cap:
                break
            fresh_tokens = await self._cycle(req_idx, step_idx)
            if eos_ids:
                hits = (fresh_tokens.unsqueeze(-1) == torch.tensor(eos_ids, dtype=fresh_tokens.dtype, device=fresh_tokens.device)).any(dim=-1).nonzero()
                if hits.numel() > 0:
                    fresh_tokens = fresh_tokens[:hits[0, 0].item() + 1]
                    finished = True
            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
            if stops:
                text = self._tokenizer.decode(self._prefix_tokens[0, self._num_original_tokens:].tolist(), skip_special_tokens=True)
                stop_hit = any((s in text for s in stops))
            step_idx += 1
        out = self._prefix_tokens[0, self._num_original_tokens:].tolist()
        self._overshoot = max(0, len(out) - cap)
        return out[:cap]

def _eos_id_set(tokenizer):
    import json as _json
    import os as _os
    ids = set()
    single = getattr(tokenizer, 'eos_token_id', None)
    if single is not None:
        ids.add(int(single))
    for x in getattr(tokenizer, 'additional_special_tokens_ids', None) or []:
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
        for x in [e] if isinstance(e, int) else e or []:
            try:
                ids.add(int(x))
            except (TypeError, ValueError):
                pass
    return sorted(ids) or [0]

def _stop_truncate(tokenizer, ids, stops):
    if not stops:
        return (ids, False)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    cut = min((text.find(s) for s in stops if s in text), default=-1)
    if cut < 0:
        return (ids, False)
    keep = 0
    for n in range(len(ids)):
        if len(tokenizer.decode(ids[:n], skip_special_tokens=True)) >= cut:
            keep = n
            break
    return (ids[:keep], True)
