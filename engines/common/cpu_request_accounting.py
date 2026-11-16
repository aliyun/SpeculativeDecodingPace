import os,json,time,hashlib,re
from pathlib import Path

def _read_counter(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:raw=stream.read(128).strip()
    if not raw.isdigit():raise RuntimeError('CPU_ACCOUNTING_COUNTER_INVALID')
    return int(raw)

class Meter:
    def __init__(self,out,rid,mapfile=None,mount=None,identity=None):
        self.out=Path(out);self.rid=rid;self.mapfile=Path(mapfile or os.environ['TRIAGE_CPU_MAP']);self.mount=Path(mount or '/host-cpuacct');self.closed=False
        raw=self.mapfile.read_bytes();self.mapsha=hashlib.sha256(raw).hexdigest();self.doc=json.loads(raw);wanted=identity or os.environ['TRIAGE_CPU_IDENTITY']
        if self.doc['identity']!=wanted or set(self.doc['roles'])!={'cloud','edge'}:raise RuntimeError('CPU_ACCOUNTING_OWNERSHIP_MAP')
        self.paths={}
        for role,row in self.doc['roles'].items():
            cid=row['cid'];relative=row['relative_counter_path'];parts=Path(relative).parts
            if not re.fullmatch('[0-9a-f]{64}',cid) or Path(relative).is_absolute() or '..' in parts or parts[-2:]!=(cid,'cpuacct.usage'):raise RuntimeError('CPU_ACCOUNTING_FOREIGN_OR_INVALID_PATH')
            self.paths[role]=self.mount/relative
        self.start_ns=time.monotonic_ns();self.before={r:_read_counter(p) for r,p in self.paths.items()}
    def finish(self,status,output_tokens):
        if self.closed:return
        if hashlib.sha256(self.mapfile.read_bytes()).hexdigest()!=self.mapsha:raise RuntimeError('CPU_ACCOUNTING_MAP_CHANGED')
        after={r:_read_counter(p) for r,p in self.paths.items()}
        if any(after[r]<self.before[r] for r in after):raise RuntimeError('CPU_ACCOUNTING_COUNTER_RESET')
        z=dict(rid=self.rid,status=status,output_tokens=output_tokens,map_sha256=self.mapsha,roles=self.doc['roles'],before_cpu_ns=self.before,after_cpu_ns=after,cpu_seconds={r:(after[r]-self.before[r])/1e9 for r in after},wall_elapsed_seconds=(time.monotonic_ns()-self.start_ns)/1e9,scope='OWN_ROLE_CGROUP_CPU_POSTPARSE_ADMISSION_TO_FINAL_GENERATION;CLOUD_INCLUDES_CLIENT_TOKENIZER;NOT_GPU_COMPUTE_OR_ALLOCATED_CORES')
        with (self.out/'CPU_REQUESTS.jsonl').open('a') as stream:stream.write(json.dumps(z,separators=(',',':'))+'\n');stream.flush()
        self.closed=True

def begin(out,rid):
    if os.environ.get('TRIAGE_CPU_ACCOUNTING','0')!='1':return None
    return Meter(out,rid)

def finish(meter,status,output_tokens):
    if meter is not None:meter.finish(status,output_tokens)
