from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math

WAIT=-1
AR=0
INVALID=-10**9


@dataclass(frozen=True)
class Supply:
    lead:int
    remaining:int
    arrival_ms:float


@dataclass(frozen=True)
class Costs:
    alpha:float
    ar_ms:float
    verify_ms:tuple
    frame_ms:float
    repair_ms:float
    continuation_ms:float
    ahead:int=12
    deadline_ms:float=100.

    def __post_init__(self):
        if not math.isfinite(self.alpha) or not 0<=self.alpha<=1:raise ValueError('STREAM_MODEL_ALPHA')
        positive=(self.ar_ms,self.frame_ms,self.repair_ms,self.continuation_ms,self.deadline_ms)
        if any(not math.isfinite(v) or v<=0 for v in positive):raise ValueError('STREAM_MODEL_TIME')
        ks=[k for k,v in self.verify_ms]
        if (not ks or len(set(ks))!=len(ks) or any(type(k) is not int or k<1 for k in ks)
                or any(not math.isfinite(v) or v<=0 for k,v in self.verify_ms)
                or type(self.ahead) is not int or self.ahead<max(ks)):
            raise ValueError('STREAM_MODEL_CATALOGUE')


@dataclass(frozen=True)
class Choice:
    action:int
    value:float
    evaluations:int
    progress:float=0.
    elapsed_ms:float=0.


class ModelBudgetExceeded(ValueError):
    pass


class StreamValue:
    MAX_EVALUATIONS=8192

    def __init__(self,costs):
        self.c=costs
        self.verify=dict(costs.verify_ms)
        self.expected=[1.]
        for k in range(1,max(self.verify)+1):self.expected.append(self.expected[-1]+costs.alpha**k)
        self._advance_cache={}
        self._outcomes_cache={}

    def check(self,s):
        if s.lead==INVALID and s.remaining==0 and s.arrival_ms==0:return
        if (type(s.lead) is not int or not -128<=s.lead<=2*self.c.ahead
                or type(s.remaining) is not int or not 0<=s.remaining<=self.c.ahead
                or s.lead+s.remaining>2*self.c.ahead
                or not math.isfinite(s.arrival_ms) or s.arrival_ms<0
                or (s.remaining==0)!=(s.arrival_ms==0)):
            raise ValueError('STREAM_MODEL_SUPPLY')

    def reset(self,limit=None):
        count=self.c.ahead if limit is None else min(self.c.ahead,limit)
        return Supply(0,count,self.c.repair_ms if count else 0.)

    def normalize(self,s,limit=None):
        self.check(s)
        if limit==0:return Supply(0,0,0.)
        if s.lead==INVALID:return self.reset(limit)
        if not s.remaining and s.lead<self.c.ahead:
            # Continue from the actual prefix plus any aligned buffered suffix.
            # Never invent new production beyond the known generation budget.
            lead=max(0,s.lead)
            count=self.c.ahead if limit is None else min(self.c.ahead,max(0,limit-lead))
            return Supply(lead,count,self.c.continuation_ms if count else 0.)
        return s

    def advance(self,s,elapsed_ms):
        """Advance only current in-flight work; do not reopen a finished path."""
        self.check(s)
        if not math.isfinite(elapsed_ms) or elapsed_ms<0:raise ValueError('STREAM_MODEL_ELAPSED')
        key=(s,elapsed_ms)
        if key in self._advance_cache:return self._advance_cache[key]
        if s.lead==INVALID or not s.remaining or elapsed_ms+1e-12<s.arrival_ms:
            values=((1.,Supply(s.lead,s.remaining,max(0.,s.arrival_ms-elapsed_ms))),)
        else:
            count=min(3,s.remaining)
            left=s.remaining-count
            arrived=Supply(s.lead+count,left,self.c.frame_ms if left else 0.)
            aligned=(self.c.alpha**min(count,-s.lead) if s.lead<0 else 1.)
            rest=max(0.,elapsed_ms-s.arrival_ms)
            merged=defaultdict(float)
            if aligned:
                for probability,next_state in self.advance(arrived,rest):merged[next_state]+=aligned*probability
            if aligned<1.:merged[Supply(INVALID,0,0.)]+=1.-aligned
            values=tuple((p,state) for state,p in merged.items() if p)
        if len(self._advance_cache)>=4096:self._advance_cache.clear()
        self._advance_cache[key]=values
        return values

    def outcomes(self,s,action,remaining=None):
        """(elapsed, ((probability, committed progress, boundary state), ...))."""
        self.check(s)
        if s.lead==INVALID:raise ValueError('STREAM_MODEL_NOT_A_BOUNDARY')
        key=(s,action,remaining)
        if key in self._outcomes_cache:return self._outcomes_cache[key]
        out=defaultdict(float)
        def after(state,progress):
            return self.normalize(state,None if remaining is None else max(0,remaining-progress))
        if action==WAIT:
            if not s.remaining:raise ValueError('STREAM_MODEL_WAIT_WITHOUT_PRODUCER')
            elapsed=s.arrival_ms
            for mass,ns in self.advance(s,elapsed):out[0,after(ns,0)]+=mass
        else:
            if action!=AR and (action not in self.verify or s.lead<action):
                raise ValueError('STREAM_MODEL_PHANTOM_DEPTH')
            elapsed=self.c.ar_ms if action==AR else self.verify[action]
            for mass,ns in self.advance(s,elapsed):
                b,r,q=ns.lead,ns.remaining,ns.arrival_ms
                if action==AR:
                    if b==INVALID:out[1,after(Supply(INVALID,0,0.),1)]+=mass
                    elif b>0:
                        out[1,after(Supply(b-1,r,q),1)]+=mass*self.c.alpha
                        out[1,after(Supply(INVALID,0,0.),1)]+=mass*(1-self.c.alpha)
                    else:out[1,after(Supply(b-1,r,q),1)]+=mass
                    continue
                if b==INVALID:raise AssertionError('a ready prefix cannot become target-ahead during verify')
                for accepted in range(action):
                    out[accepted+1,after(Supply(INVALID,0,0.),accepted+1)]+=mass*self.c.alpha**accepted*(1-self.c.alpha)
                full=mass*self.c.alpha**action
                if b>=action+1:
                    out[action+1,after(Supply(b-action-1,r,q),action+1)]+=full*self.c.alpha
                    out[action+1,after(Supply(INVALID,0,0.),action+1)]+=full*(1-self.c.alpha)
                else:
                    # The unreceived bonus is unknown, not automatically aligned.
                    out[action+1,after(Supply(b-action-1,r,q),action+1)]+=full
        values=tuple((p,progress,ns) for (progress,ns),p in out.items() if p)
        if abs(sum(p for p,n,ns in values)-1)>1e-9:raise ValueError('STREAM_MODEL_MASS')
        if len(self._outcomes_cache)>=4096:self._outcomes_cache.clear()
        self._outcomes_cache[key]=(elapsed,values)
        return elapsed,values

    def choose(self,state,remaining,*,rho,wait_ms=None,horizon=2,room=32768):
        """Maximize E[sum(progress-rho*elapsed)] over a bounded rollout.

        WAIT has zero output and does not consume the native horizon; at most
        three frame waits are examined per prefix and the hard wait budget is
        never extended. Native commit creates a new prefix/deadline.
        """
        if (type(remaining) is not int or remaining<1 or type(room) is not int or room<remaining
                or not math.isfinite(rho) or rho<=0 or horizon not in (1,2)):
            raise ValueError('STREAM_MODEL_OBJECTIVE')
        state=self.normalize(state,remaining)
        budget=self.c.deadline_ms if wait_ms is None else wait_ms
        if not math.isfinite(budget) or budget<0:raise ValueError('STREAM_MODEL_WAIT_BUDGET')
        evaluations=0

        @lru_cache(maxsize=None)
        def value(s,left,physical,h,budget,waits):
            nonlocal evaluations
            evaluations+=1
            if evaluations>self.MAX_EVALUATIONS:raise ModelBudgetExceeded('STREAM_MODEL_ROLLOUT_BUDGET')
            if not h or not left:return 0.,AR,0.,0.
            actions=[AR]+[k for k in sorted(self.verify) if k<=s.lead and k<=physical]
            if waits and s.remaining and s.arrival_ms<=budget:actions.append(WAIT)
            scored=[]
            for action in actions:
                if action!=WAIT and (h==1 or left==1):
                    # No continuation value after the terminal native action:
                    # integrate the censored reward analytically, without
                    # building producer/repair outcomes that cannot be used.
                    elapsed=self.c.ar_ms if action==AR else self.verify[action]
                    progress=self.expected[min(action,left-1)]
                    priority=0 if action==AR else 1
                    scored.append((progress-rho*elapsed,-priority,-action,action,progress,elapsed))
                    continue
                elapsed,transitions=self.outcomes(s,action,remaining=left)
                score=-rho*elapsed
                total_progress=0.
                total_elapsed=elapsed
                for probability,progress,ns in transitions:
                    actual=min(left,progress)
                    future=(value(ns,left,physical,h,max(0.,budget-elapsed),waits-1)
                            if action==WAIT else
                            value(ns,left-actual,physical-actual,h-1,self.c.deadline_ms,3))
                    score+=probability*(actual+future[0])
                    total_progress+=probability*(actual+future[2])
                    total_elapsed+=probability*future[3]
                # Ties favor AR, then the smaller ready K, never an extra wait.
                priority=0 if action==AR else 2 if action==WAIT else 1
                scored.append((score,-priority,-action,action,total_progress,total_elapsed))
            best=max(scored)
            return best[0],best[3],best[4],best[5]

        score,action,progress,elapsed=value(state,remaining,room,horizon,budget,3)
        return Choice(action,score,evaluations,progress,elapsed)

    def choose_rate(self,state,remaining,*,wait_ms=None,horizon=2,room=32768):
        """Dinkelbach root for the bounded rollout's E[progress]/E[time].

        No hand-picked opportunity-price constant is used by the runtime. This
        solves only the stated finite rollout, not the infinite-horizon SMDP.
        """
        rho=1./self.c.ar_ms
        evaluations=0
        for _ in range(8):
            result=self.choose(state,remaining,rho=rho,wait_ms=wait_ms,horizon=horizon,room=room)
            evaluations+=result.evaluations
            rate=result.progress/result.elapsed_ms
            if abs(result.value)<=1e-7 or abs(rate-rho)<=1e-9:
                return Choice(result.action,result.value,evaluations,result.progress,result.elapsed_ms)
            rho=rate
        raise ModelBudgetExceeded('STREAM_MODEL_RATE_ROOT_BUDGET')
