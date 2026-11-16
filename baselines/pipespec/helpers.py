import hashlib
import json
import os
import re
from pathlib import Path

def sha256(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return hashlib.sha256(path_or_bytes).hexdigest()
    h = hashlib.sha256()
    with Path(path_or_bytes).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(str(q), str(p))

def gsm8k_gold(row):
    text = ' '.join(row['references'])
    nums = re.findall('####\\s*(-?[\\d,]+(?:\\.\\d+)?)', text)
    if not nums:
        nums = re.findall('(-?[\\d,]+(?:\\.\\d+)?)', text.split('####')[-1])
    return nums[-1].replace(',', '') if nums else None

def gsm8k_pred(text):
    nums = re.findall('(-?[\\d,]+(?:\\.\\d+)?)', text.split('####')[-1])
    return nums[-1].replace(',', '') if nums else None

def summarize(arm, records, max_new_tokens):
    ok = [r for r in records if r.get('status') == 'OK']
    spent = sum((r['wall_s'] for r in ok))
    toks = sum((r['new_tokens'] for r in ok))
    correct = [r for r in ok if r.get('gsm8k_correct')]
    rates = sorted((r['tok_per_s'] for r in ok), reverse=True)
    pct = lambda q: rates[min(len(rates) - 1, int(q * (len(rates) - 1)))] if rates else None
    return {'arm': arm, 'generations': len(records), 'ok': len(ok), 'errors': len(records) - len(ok), 'output_tokens': toks, 'elapsed_s': spent, 'output_tokens_per_s': toks / spent if spent else None, 'request_p50_tokens_per_s': pct(0.5), 'request_p05_tokens_per_s': pct(0.95), 'mean_new_tokens': toks / len(ok) if ok else None, 'eos_finish': sum((1 for r in ok if r.get('finish') == 'eos')), 'length_capped': sum((1 for r in ok if r.get('finish') == 'length')), 'gsm8k_accuracy': len(correct) / len(ok) if ok else None, 'max_new_tokens': max_new_tokens}
