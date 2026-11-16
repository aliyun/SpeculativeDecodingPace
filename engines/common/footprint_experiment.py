import hashlib,json,os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def profile():
    flag=os.environ.get('TRIAGE_REDUCED_FOOTPRINT','0')
    path=os.environ.get('TRIAGE_FOOTPRINT_TEMPLATE')
    if flag not in ('0','1') or (flag=='0' and path):raise ValueError('REDUCED_FOOTPRINT_EXPLICIT_GATE')
    if flag=='0':return None
    if not path or not os.environ.get('TRIAGE_FOOTPRINT_SHA256'):raise ValueError('REDUCED_FOOTPRINT_SOURCE_REQUIRED')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=os.environ['TRIAGE_FOOTPRINT_SHA256']:raise ValueError('REDUCED_FOOTPRINT_HASH')
    p=json.loads(raw)
    if (p['schema']!=1 or p['context']!=2048 or p['max_num_seqs']!=16 or p['num_gpu_blocks_override']!=1024 or p['graph_limit']!=64
            or p['role_settings'] not in (
                {'cloud':{'gpu_memory_utilization':.35,'cpu_offload_gb':40},'edge':{'gpu_memory_utilization':.10,'cpu_offload_gb':0}},
                {'cloud':{'gpu_memory_utilization':.90,'cpu_offload_gb':0},'edge':{'gpu_memory_utilization':.10,'cpu_offload_gb':0}})):raise ValueError('REDUCED_FOOTPRINT_FROZEN_PROFILE')
    return p

def context(default=32768):
    p=profile();return p['context'] if p else default
