from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261020


REPLAY_SEED = SEED + 1
CONTEXT = 32768
MICRO = {"S01": (256, 256), "S02": (256, 1024), "S03": (4096, 256), "S04": (4096, 1024)}


def micro_text(n_in):
    return "the" + " the" * (int(n_in) - 1)
REAL_LOADS = ("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10")


class Load:
    def __init__(self, name, rows, order, source):
        self.name = name
        self.rows = rows
        self.order = order
        self.source = source

    @property
    def forced(self):
        return self.name.startswith("S")

    def plan(self, limit):
        return self.order[:limit]


def _replay(n_pool, generations, seed=REPLAY_SEED):
    order = list(range(n_pool))
    random.Random(seed).shuffle(order)
    return [order[i % n_pool] for i in range(generations)]


def load_pool(pools_path, name, generations=512, micro_generations=128, tokenizer=None):
    if tokenizer is None:
        raise ValueError("load_pool needs a tokenizer to derive prompt_text")
    pools_path = Path(pools_path)
    if name in MICRO:
        n_in, cap = MICRO[name]
        text = micro_text(n_in)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        rows = [{"id": f"{name}/synthetic", "input_ids": list(ids), "max_tokens": cap,
                 "prompt_text": text}]
        order = [0] * micro_generations
        rows[0]["prefix_exact"] = (tokenizer(text, add_special_tokens=False)["input_ids"] == ids)
        if not rows[0]["prefix_exact"]:
            raise ValueError("%s: prompt_text does not re-tokenise to its own ids" % name)
        return Load(name, rows, order, "synthetic")
    pools = json.loads(pools_path.read_text())
    if name not in pools:
        raise KeyError(f"{name} not in {pools_path} (have {sorted(pools)})")
    rows = pools[name]
    for r in rows:
        r.setdefault("prompt_text", tokenizer.decode(
            r["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return Load(name, rows, _replay(len(rows), generations), "pools.json")


def first_turn_ids(row):
    """Rings the frozen template ids for turn 1 (already fully templated)."""
    return list(row["input_ids"])


def second_turn_ids(tokenizer, row, previous_text):
    """Rebuild turn-2 ids exactly like the AR client did."""
    turns = row.get("turns") or []
    if len(turns) < 2:
        raise ValueError("row has fewer than two turns: %r" % (row.get("id"),))
    messages = [{"role": "user", "content": turns[0]},
                {"role": "assistant", "content": previous_text},
                {"role": "user", "content": turns[1]}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
    cap = int(row["max_tokens"])
    if len(ids) + cap > CONTEXT:
        raise ValueError("MT history exceeded context for %s" % (row.get("id"),))
    return list(ids)


def stop_strings(row):
    return list(row.get("stop") or [])


def truncate_at_stop(tokenizer, ids, stops):
    """Drop everything from the first stop-string hit onward (server-side `stop`)."""
    if not stops or not ids:
        return ids, False
    text = tokenizer.decode(ids, skip_special_tokens=True)
    cut = min((text.find(s) for s in stops if s in text), default=-1)
    if cut < 0:
        return ids, False

    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) < cut:
            lo = mid + 1
        else:
            hi = mid
    return ids[:lo], True
