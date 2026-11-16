import os,json,time
from pathlib import Path
def strategy():
 path=os.environ.get('TRIAGE_STRATEGY_FILE')
 if path and os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH')=='1':
  from .control_snapshot import read_snapshot
  mode=read_snapshot(path)[1]['mode']
 else:mode=json.loads(Path(path).read_text())['mode'] if path and Path(path).exists() else os.environ.get('TRIAGE_STRATEGY_CONTROL','a2')
 if mode not in ('ar','k3','a2','master'):raise ValueError('OFFLOAD_EXPERIMENT_STRATEGY_FILE')
 return mode
def reset_for_epoch(prefetch):
 if os.environ.get('TRIAGE_OFFLOAD_EXPERIMENT')!='1' or os.environ.get('TRIAGE_ENGINE_ROLE')!='cloud':return
 path=os.environ.get('TRIAGE_STRATEGY_FILE')
 if not path:return
 if os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH')=='1':
  from .control_snapshot import read_snapshot
  record=read_snapshot(path)[1]
 else:
  if not Path(path).exists():return
  record=json.loads(Path(path).read_text())
 epoch=record['epoch']
 with prefetch.lock:
  if getattr(prefetch,'experiment_epoch',None)==epoch:return
  if prefetch.entries or prefetch.live or prefetch.active_rids or prefetch.releasing or any(not f.done() for f in prefetch.inflight) or any(not f.done() for f in getattr(prefetch,'warming',())):raise ValueError('EXPERIMENT_SUFFIX_RESET_REQUIRES_DRAIN')
  previous=len(prefetch.cache.entries);prefetch.cache=type(prefetch.cache)();prefetch.experiment_epoch=epoch
  exact=getattr(prefetch,'exact_cache',None)
  if exact is not None:exact.clear()
  prefetch.event('experiment_policy_epoch',epoch=epoch,mode=record['mode'],suffix_entries_cleared=previous)
