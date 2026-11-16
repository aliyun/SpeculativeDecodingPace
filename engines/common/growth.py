from collections import defaultdict,deque
from dataclasses import dataclass
import json,math,os,time
from pathlib import Path
from .policy import CONTEXT,MAX_REQUESTS,fingerprint,token_ids
def declared_space():
    """Declared draft lengths, defaulting to K3/K5/K7 when unset.

    TRIAGE_GROWTH_SPACE selects the catalogue before model initialization.
    The launcher supplies K3/K4/K5/K6/K7 for the default adaptive path.
    """
    value=os.environ.get('TRIAGE_GROWTH_SPACE')
    if value is None:return (3,5,7)
    items=[item.strip() for item in str(value).split(',') if item.strip()]
    try:ks=tuple(sorted({int(item) for item in items}))
    except (TypeError,ValueError):raise ValueError('GROWTH_SPACE') from None
    if not ks or any(k<1 or k>9 for k in ks) or not {3,5,7}<=set(ks):raise ValueError('GROWTH_SPACE')
    return ks

KS=declared_space()  # protocol/catalogue space

def choice_ks():
    """Draft lengths this side may select; a subset of the declared space."""
    value=os.environ.get('TRIAGE_GROWTH_KS')
    if value is None:return KS
    items=[item.strip() for item in str(value).split(',') if item.strip()]
    try:ks=tuple(sorted({int(item) for item in items}))
    except (TypeError,ValueError):raise ValueError('GROWTH_KS') from None
    if not ks or any(k not in KS for k in ks):raise ValueError('GROWTH_KS')
    return ks

def fork_gap_limit():
    """Optional root-alternate gate in top-1/top-2 logit gap units.

    branches=2 becomes an option: the edge still returns the real top2, while
    pack() appends the alternate only when the measured gap is at or below the
    limit. Unset preserves the historical forced-fork behavior.
    """
    value=os.environ.get('TRIAGE_GROWTH_FORK_GAP')
    if value is None:return None
    try:limit=float(value)
    except (TypeError,ValueError):raise ValueError('GROWTH_FORK_GAP') from None
    if not math.isfinite(limit) or limit<0:raise ValueError('GROWTH_FORK_GAP')
    return limit

def forced_action():
    """Force one action past every guard, for isolating an action's own cost.

    TRIAGE_GROWTH_FORCE_K overrides the decision after the pressure, static and
    context guards, and labels the plan so the reason is auditable afterwards.
    Unset changes nothing.
    """
    value=os.environ.get('TRIAGE_GROWTH_FORCE_K')
    if value is None:return None
    try:k=int(str(value).strip())
    except (TypeError,ValueError):raise ValueError('GROWTH_FORCE_K') from None
    if k not in KS:raise ValueError('GROWTH_FORCE_K')
    roots=os.environ.get('TRIAGE_GROWTH_FORCE_ROOTS','1')
    if roots not in ('1','2'):raise ValueError('GROWTH_FORCE_ROOTS')
    return (k,int(roots))

def draft_transport():
    """How cloud obtains drafts: live `/stream` frames or one `/batch` per round.

    `stream` (default) keeps the bounded live producer and binds a Plan at ready
    time. `sync` never opens a stream: each round issues one `sync_draft` batch
    and waits for the whole K, which the historical catalogue already supported.
    `auto` lets the streaming selector choose between those paths.
    """
    value=os.environ.get('TRIAGE_DRAFT_TRANSPORT','stream')
    if value not in ('stream','sync','auto'):raise ValueError('DRAFT_TRANSPORT_MODE')
    return value

CHOICES=choice_ks() # what this side is allowed to select
FORCED=forced_action()

def enabled():
    value=os.environ.get('TRIAGE_ADAPTIVE_GROWTH','0')
    if value not in ('0','1'):raise ValueError('ADAPTIVE_GROWTH_MODE')
    return value=='1'

def capability():
    """Advertise only on adapters which implement the complete growth protocol."""
    return dict(protocol=1,ks=list(KS),roots=[1,2]) if enabled() else None

def check_capability(record, active=None):
    if active is None:active=enabled()
    value=record.get('growth') if type(record) is dict else None
    expected=dict(protocol=1,ks=list(KS),roots=[1,2]) if active else None
    # Exact types as well as values: bool is not a protocol/version integer.
    if value!=expected or value is not None and (
            type(value) is not dict or set(value)!=set(expected)
            or type(value['protocol']) is not int
            or type(value['ks']) is not list or any(type(x) is not int for x in value['ks'])
            or type(value['roots']) is not list or any(type(x) is not int for x in value['roots'])):
        raise ValueError('GROWTH_CAPABILITY_MISMATCH_RESTART_BOTH_ENDPOINTS')
    return value

def create_policy(event, clock=time.monotonic_ns):
    """Production policy: fresh source-aware measurements, no seed/control files.

    The historical Policy/A-B selector remains available only to explicit
    experiment callers. This factory does not dispatch through those selectors.
    """
    stream_selector=os.environ.get('TRIAGE_STREAM_SELECTOR','0')
    if stream_selector not in ('0','1'):raise ValueError('STREAM_SELECTOR_FLAG')
    if stream_selector=='1':
        from .stream_selector import StreamingPolicy
        policy=StreamingPolicy(event,clock)
        policy.refresh()
        return policy
    if os.environ.get('TRIAGE_GROWTH_RULE')=='1':
        from .growth_rule import RulePolicy
        policy=RulePolicy(event,clock)
        policy.refresh()
        return policy
    from .growth_source_cost import SourceCostPolicy
    class LiveSourceCostPolicy(SourceCostPolicy):
        live_runtime=True
        def __init__(self,event,clock=time.monotonic_ns):
            super().__init__(event,clock)
            # Historical experiments intentionally stop at 20000 exported rows.
            # Production retains a stricter bounded ring, while every original
            # event is still emitted and the factual pricing windows stay intact.
            self.observations=deque(maxlen=self.WINDOW)
            self.cost_observations=deque(maxlen=self.COST_WINDOW)
        def commit_batch(self,rows,*,rpc_ms=0.):
            super().commit_batch(rows,rpc_ms=rpc_ms)
            if self.cycle_id%self.WINDOW==0:
                retained={(r['generation'],r['cycle_id']) for values in self.cost_rows.values() for r in values}
                self.sample_tick={key:tick for key,tick in self.sample_tick.items() if key in retained}
        def snapshot(self):
            return dict(super().snapshot(),observed_cycles=self.cycle_id,
                        telemetry_retention=self.WINDOW,live_bounded_feedback=True)
        def refresh(self):
            if self.control is None:
                self.epoch=0
                self.control=dict(epoch=0,mode='adaptive',learn=True)
                self.event('growth_live_policy',selector=self.selector_name,generation=self.generation)
        def load_seed(self, rows):
            raise ValueError('GROWTH_LIVE_POLICY_REJECTS_SEED')
    policy=LiveSourceCostPolicy(event,clock)
    policy.refresh()
    return policy

def parents(k,side=False):
    if type(k) is not int or k not in KS or type(side) is not bool:raise ValueError('GROWTH_TREE')
    return tuple(range(-1,k))+((0,) if side else ())

def fork_parents(k,at,length=1):
    """Chain of k drafts plus a branch of `length` nodes off draft position `at`.

    Node 0 is the committed-prefix root; draft position j is node j. A branch at
    at=1 is therefore the same layout as the root top-2 side node, while at>1
    hangs the branch off an interior draft. This is a targeted rescue, not a
    second chain.
    """
    if type(k) is not int or k not in KS:raise ValueError('GROWTH_TREE')
    if type(at) is not int or not 1<=at<=k:raise ValueError('GROWTH_FORK_AT')
    if type(length) is not int or not 1<=length<=k:raise ValueError('GROWTH_FORK_LEN')
    nodes=list(range(-1,k));parent=at-1
    for _ in range(length):
        nodes.append(parent);parent=len(nodes)-1
    return tuple(nodes)

def fork_plan_fields(value):
    """(fork_at, fork_len) as validated by Plan; (0,0) means no branch."""
    if type(value) is not tuple or len(value)!=2:raise ValueError('GROWTH_FORK')
    at,length=value
    return (0,0) if at==0 and length==0 else (at,length)

@dataclass(frozen=True)
class Plan:
    rid:str
    prefix_hash:str
    k:int
    branches:int
    round_id:int
    generation:str
    reason:str
    # Position-selective fork: a branch of fork_len nodes hanging off draft
    # position fork_at. (0,0) is the default and means no branch at all.
    fork_at:int=0
    fork_len:int=0
    def __post_init__(self):
        if (not isinstance(self.rid,str) or not self.rid or not isinstance(self.prefix_hash,str)
                or len(self.prefix_hash)!=64 or any(c not in '0123456789abcdef' for c in self.prefix_hash)
                or type(self.k) is not int or self.k not in KS
                or type(self.branches) is not int or self.branches not in (1,2)
                or type(self.round_id) is not int or self.round_id<1
                or not isinstance(self.generation,str) or not self.generation
                or not isinstance(self.reason,str) or not self.reason
                or type(self.fork_at) is not int or type(self.fork_len) is not int
                or (self.fork_at,self.fork_len)!=(0,0) and (
                    not 1<=self.fork_at<=self.k or not 1<=self.fork_len<=self.k
                    or self.branches!=1)):raise ValueError('GROWTH_PLAN')
    def wire(self):
        # Every field is a validated immutable scalar; return a detached mapping
        # in the original dataclass field order without recursive deepcopy.
        return dict(rid=self.rid,prefix_hash=self.prefix_hash,k=self.k,branches=self.branches,
                    round_id=self.round_id,generation=self.generation,reason=self.reason,
                    fork_at=self.fork_at,fork_len=self.fork_len)
    @classmethod
    def bind(cls,value,rid,prefix):
        if type(value) is not dict or set(value)!=set(cls.__dataclass_fields__):raise ValueError('GROWTH_PLAN_SCHEMA')
        p=cls(**value)
        if p.rid!=rid or p.prefix_hash!=fingerprint(prefix):raise ValueError('GROWTH_PLAN_PREFIX')
        return p

def pack(proposals,contexts):
    if not 1<=len(proposals)<=MAX_REQUESTS or len(proposals)!=len(contexts):raise ValueError('GROWTH_PACK_BATCH')
    from .root_shadow import validate_top2
    from .growth_batch_profile import enabled as batch_growth_enabled,BATCH_MAX
    if batch_growth_enabled() and (len(proposals)>BATCH_MAX or len({(r['growth_plan']['k'],r['growth_plan']['branches']) for r in proposals})!=1):
        raise ValueError('BATCH_GROWTH_PACK_COHORT')
    result=[]
    for row,c in zip(proposals,contexts):
        p=Plan(**row['growth_plan']);nodes=token_ids(row['nodes'])
        if (row['id']!=p.rid or row['prefix_hash']!=p.prefix_hash or c['id']!=p.rid
                or c['prefix_hash']!=p.prefix_hash or len(nodes)!=p.k+1
                or type(c['remaining']) is not int or c['remaining']<0
                or type(c['prefix_length']) is not int or c['prefix_length']<1
                or type(c['ignore_eos']) is not bool):raise ValueError('GROWTH_PACK_PREFIX')
        if c['source']=='edge':
            if p.branches==1:
                if c['top2'] is not None:raise ValueError('GROWTH_ROOT1_CAPABILITY')
            else:
                meta=validate_top2(c['top2'],nodes[1])
                limit=c.get('fork_gap')
                if limit is None:limit=fork_gap_limit()
                if ((p.reason=='auto_sync_dual_root' or
                     ((len(proposals)==1 or batch_growth_enabled()) and c['ignore_eos']))
                        and c['remaining']>=p.k+1 and c['prefix_length']+p.k+1<=CONTEXT
                        and (limit is None or meta['gap']<=limit)):nodes.append(meta['alternate'])
        elif c['source'] not in ('suffix','exact_prefix') or c['top2'] is not None:raise ValueError('GROWTH_PACK_SOURCE')
        result.append(dict(id=p.rid,prefix_hash=p.prefix_hash,nodes=nodes,growth_plan=p.wire()))
    return result

class Policy:
    """Windowed actual delivery/cycle, finite global ranking and bounded refresh."""
    WINDOW=128
    MIN_SAMPLES=32
    PERIOD=16
    MARGIN=1.02
    ACTIONS=tuple((k,b) for k in CHOICES for b in (1,2))
    def __init__(self,event,clock=time.monotonic_ns):
        self.event,self.clock=event,clock;self.generation=os.urandom(12).hex();self.round_id=0
        self.epoch=None;self.control=None;self.current=(3,1);self.choices=0
        self.state=dict(active=0,waiting=0,free_blocks=None,capacity=None)
        self.stats=defaultdict(lambda:dict(n=0,tokens=0,seconds=0.,ewma_delivery=0.,ewma_ms=0.,
            window_n=0,window_tokens=0.,window_seconds=0.))
        self.windows={};self.buckets={}
        self.last_commit={};self.issued={};self.last_used={};self.observations=[]
    def refresh(self):
        path=os.environ.get('TRIAGE_GROWTH_CONTROL')
        if path and os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH')=='1':
            from .control_snapshot import read_snapshot
            stamp,v=read_snapshot(path,prefix='SHAPE');signature=(path,stamp)
            if self.control is not None and getattr(self,'control_signature',None)==signature:return
        else:
            signature=(path,Path(path).stat().st_mtime_ns if path and Path(path).exists() else None)
            if self.control is not None and getattr(self,'control_signature',None)==signature:return
            v=json.loads(Path(path).read_text()) if path and Path(path).exists() else dict(epoch=0,mode='adaptive')
        if (type(v) is not dict or v.get('mode') not in ('static','adaptive')
                or type(v.get('epoch')) is not int or v['epoch']<0 or type(v.get('learn',True)) is not bool
                or type(v.get('skip_first_request',False)) is not bool
                or v.get('phase_key') is not None and not isinstance(v['phase_key'],str)
                or v['mode']=='static' and (type(v.get('k')) is not int or v['k'] not in KS
                    or type(v.get('branches')) is not int or v['branches'] not in (1,2))):raise ValueError('GROWTH_CONTROL')
        if self.epoch!=v['epoch']:
            if self.issued or self.last_used:raise ValueError('GROWTH_EPOCH_REQUIRES_DRAIN')
            self.epoch=v['epoch'];self.current=(3,1);self.choices=0;self.buckets.clear()
            self.last_commit.clear();self.last_used.clear()
            self.event('growth_epoch',control=v,generation=self.generation,calibration_buckets=len(self.stats),
                selector='window128_global_refresh_v2')
        if getattr(self,'phase_key',None)!=v.get('phase_key'):
            if self.issued or self.last_used:raise ValueError('GROWTH_PHASE_REQUIRES_DRAIN')
            self.phase_key=v.get('phase_key');self.preflight_rid=None
        self.control=v;self.control_signature=signature
    @staticmethod
    def bucket(batch,length):return batch,((length+127)//128)*128
    def resources(self,*,active=None,waiting=None,free_blocks=None,capacity=None):
        for key,value in dict(active=active,waiting=waiting,free_blocks=free_blocks,capacity=capacity).items():
            if value is not None:
                if type(value) is not int or value<0:raise ValueError('GROWTH_RESOURCE_SAMPLE')
                self.state[key]=value
    def score(self,bucket,action):
        s=self.stats.get((*bucket,*action))
        if not s:return None
        # The lifetime fallback supports explicit seeded calibration fixtures.
        n=s.get('window_n',0);tokens=s.get('window_tokens',0.);seconds=s.get('window_seconds',0.)
        if not n:n,tokens,seconds=s['n'],s['tokens'],s['seconds']
        return tokens/seconds if n>=self.MIN_SAMPLES and seconds>0 else None
    def _bucket(self,key):
        if key not in self.buckets:
            self.buckets[key]=dict(current=(3,1),target=None,choices=0,last_switch=-self.PERIOD,
                tried={a:0 for a in self.ACTIONS})
        return self.buckets[key]
    @staticmethod
    def _step(current,target):
        i,j=KS.index(current[0]),KS.index(target[0])
        return (KS[i+(1 if j>i else -1)],target[1]) if i!=j else target
    def _adaptive(self,key,st):
        current=st['current']
        # A scheduled bridge cannot be blocked by a worse intermediate score.
        if st['target'] is not None:
            current=self._step(current,st['target']);st['current']=current;st['last_switch']=st['choices']
            if current==st['target']:st['target']=None
            return current,'global_target_bridge'
        # Refresh known as well as unknown alternatives. This is a real timed
        # proposal, not a counterfactual target-result selection.
        if st['choices']%self.PERIOD==0:
            other=[a for a in self.ACTIONS if a!=current]
            action=min(other,key=lambda a:(st['tried'][a],a))
            return action,('bounded_known_probe' if self.score(key,action) is not None else 'bounded_cold_probe')
        if (st['choices']-1)%self.PERIOD==0 and st['choices']-st['last_switch']>=self.PERIOD:
            scored=[(self.score(key,a),a) for a in self.ACTIONS]
            scored=[v for v in scored if v[0] is not None]
            if scored:
                value,best=max(scored);old=self.score(key,current)
                if best!=current and (old is None or value>old*self.MARGIN):
                    st['target']=best;current=self._step(current,best)
                    st['current']=current;st['last_switch']=st['choices']
                    if current==best:st['target']=None
                    return current,'window_delivery_cycle'
        return current,'hold'
    def choose(self,rid,prefix,batch):
        self.refresh();key=self.bucket(batch,len(prefix))
        if (rid,fingerprint(prefix)) in self.issued:raise ValueError('GROWTH_PLAN_ALREADY_PENDING')
        preflight=rid==getattr(self,'preflight_rid',None)
        if preflight:a=(3,1);reason='preflight_baseline';st=None
        else:
            self.choices+=1;st=self._bucket(key);st['choices']+=1
            pressure=(batch!=1 or self.state['waiting']>0 or (self.state['capacity']
                and self.state['free_blocks'] is not None and self.state['free_blocks']<self.state['capacity']//10))
            if pressure:
                a=(3,1);st.update(current=a,target=None,last_switch=st['choices']);reason='cloud_pressure'
            elif self.control['mode']=='static':
                a=(self.control['k'],self.control['branches']);reason='static_calibration'
            else:a,reason=self._adaptive(key,st)
            if len(prefix)+a[0]+1>CONTEXT:
                a=(3,1);st.update(current=a,target=None,last_switch=st['choices']);reason='context_floor'
            if FORCED is not None and len(prefix)+FORCED[0]+1<=CONTEXT:
                # Applied last so no guard can silently undo an isolated arm; the
                # reason records both what was chosen and what was overridden.
                reason='forced_action_over_%s'%reason;a=FORCED
            st['tried'][a]=st['choices'];self.current=st['current']
        self.round_id+=1;p=Plan(rid,fingerprint(prefix),*a,self.round_id,self.generation,reason)
        self.issued[(rid,p.prefix_hash)]=p
        self.event('growth_plan',plan=p.wire(),state=dict(self.state),produce_batch=batch,bucket=key,
            selector='window128_global_refresh_v2',bucket_choice=st['choices'] if st else None,
            bucket_current=list(st['current']) if st else None,
            scores=[dict(k=a[0],branches=a[1],score=self.score(key,a)) for a in self.ACTIONS])
        return p
    def consumed(self,p):
        if self.issued.pop((p.rid,p.prefix_hash),None)!=p:raise ValueError('GROWTH_PLAN_CONSUMPTION')
        self.last_used[p.rid]=p
    def commit(self,rid,*,delivered,remaining,raw_emitted,batch,prefix_length,rpc_ms=0.,plan=None,source=None,side=False):
        now=self.clock();before=self.last_commit.get(rid);self.last_commit[rid]=now
        if plan is None and self.control and self.control.get('skip_first_request',False) and getattr(self,'preflight_rid',None) is None:
            self.preflight_rid=rid;self.event('growth_preflight_excluded',rid=rid,phase_key=self.control.get('phase_key'))
        if plan is None or before is None:return
        if self.last_used.pop(rid,None)!=plan:raise ValueError('GROWTH_COMMIT_PLAN')
        elapsed=(now-before)/1e9
        if not math.isfinite(elapsed) or elapsed<=0:raise ValueError('GROWTH_CLOCK')
        censored=remaining<=raw_emitted;key=(*self.bucket(batch,prefix_length),plan.k,plan.branches)
        learning=self.control.get('learn',True) and rid!=getattr(self,'preflight_rid',None)
        if not censored and learning:
            s=self.stats[key];s['n']+=1;s['tokens']+=delivered;s['seconds']+=elapsed
            # Keep v1 telemetry fields for diagnostics, but never rank with them.
            alpha=.2;s['ewma_delivery']=delivered if s['n']==1 else (1-alpha)*s['ewma_delivery']+alpha*delivered
            s['ewma_ms']=elapsed*1000 if s['n']==1 else (1-alpha)*s['ewma_ms']+alpha*elapsed*1000
            window=self.windows.setdefault(key,deque())
            if len(window)==self.WINDOW:
                old_d,old_t=window.popleft();s['window_tokens']-=old_d;s['window_seconds']-=old_t
            window.append((delivered,elapsed));s['window_n']=len(window)
            s['window_tokens']+=delivered;s['window_seconds']+=elapsed
        record=dict(rid=rid,plan=plan.wire(),actual_batch=batch,prefix_length=prefix_length,source=source,
            delivered=delivered,raw_emitted=raw_emitted,censored=censored,learning=learning,cycle_ms=elapsed*1000,
            target_rpc_host_ms=rpc_ms,non_target_cycle_proxy_ms=max(0.,elapsed*1000-rpc_ms),side=side)
        if len(self.observations)>=20000:raise RuntimeError('GROWTH_OBSERVATION_LIMIT')
        self.observations.append(record);self.event('growth_cycle',**record)
    def forget(self,rid):
        self.last_commit.pop(rid,None);self.last_used.pop(rid,None)
        for key in [key for key in self.issued if key[0]==rid]:self.issued.pop(key)
    def snapshot(self):
        return dict(enabled=True,generation=self.generation,control=self.control,current=list(self.current),
            issued=len(self.issued),last_used=len(self.last_used),observed=len(self.observations),
            selector='window128_global_refresh_v2',score_window=self.WINDOW,probe_period=self.PERIOD,
            state_buckets=[dict(batch=k[0],prefix_bucket=k[1],current=list(v['current']),
                choices=v['choices'],target=list(v['target']) if v['target'] else None) for k,v in self.buckets.items()],
            stats=[dict(batch=k[0],prefix_bucket=k[1],k=k[2],branches=k[3],**v) for k,v in self.stats.items()],
            limits=dict(target_graphs=57,edge_graphs=58,max_graphs=64))

if os.environ.get('TRIAGE_SELECTOR_PAIR')=='1' and os.environ.get('TRIAGE_GROWTH_EXPERIMENT')=='1':
    raise ValueError('historical selector-pair experiments are not included')
