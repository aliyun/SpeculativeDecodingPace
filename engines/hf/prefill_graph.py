from types import SimpleNamespace
from .attention import install
from .fixed_slab import FixedKV


def parse_buckets(value):
    try:
        if not isinstance(value,str):raise ValueError()
        sizes=[int(n) for n in value.split(',')]
    except (ValueError,TypeError):
        raise ValueError('prefill buckets must be comma-separated integers') from None
    if (not 1<=len(sizes)<=8 or sizes!=sorted(set(sizes)) or
            any(n<16 or n>256 or n%16 for n in sizes)):
        raise ValueError('prefill buckets need 1..8 sorted unique multiples of16 in16..256')
    return sizes


def choose_bucket(length,sizes):
    if type(length) is not int or length<1:raise ValueError('nonempty prefill required')
    return next((n for n in sizes if n>=length),None)


class PrefixView:
    def __init__(self,cache,length):
        self.length=length
        self.layers=[SimpleNamespace(keys=l.keys[:,:,:length],values=l.values[:,:,:length])
                     for l in cache.layers]
    def get_seq_length(self,layer_idx=0):return self.length


class PrefillGraph:
    def __init__(self,owner,bucket):
        self.owner,self.bucket=owner,bucket
        backend=owner.backend;torch=backend.torch;device=backend.device
        self.inputs=torch.ones(1,bucket+1,dtype=torch.long,device=device)
        self.inputs[0,bucket]=bucket
        self.positions=torch.arange(bucket,device=device)[None,:]
        self.mask=torch.arange(owner.context,device=device)[None,None,None,:]<=self.positions[:,None,:,None]
        self.lengths=torch.tensor([bucket],dtype=torch.int32,device=device)
        # Separate root code frame from decode compilation; at most8 buckets.
        self.module=torch.compile(self._model_call,mode='max-autotune-no-cudagraphs',dynamic=False)
        stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):self._forward()
        stream.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream),torch.inference_mode():
            self.outputs=self._forward()
        torch.cuda.current_stream(device).wait_stream(stream)

    def _model_call(self,ids,keep):
        return self.owner.backend.model(input_ids=ids,past_key_values=self.owner.cache,
            position_ids=self.positions,attention_mask={'full_attention':self.mask},
            triage_fixed_kv_lengths=self.lengths,logits_to_keep=keep,
            use_cache=True,return_dict=True).logits

    def _forward(self):
        torch=self.owner.backend.torch
        self.owner.cache.positions=self.positions
        keep=self.inputs[0,self.bucket:self.bucket+1]-1
        logits=self.module(self.inputs[:,:self.bucket],keep)
        prediction=logits[0,-1].argmax()
        finite=torch.isfinite(logits).all().long()
        self.retained=(keep,logits,prediction,finite)
        return torch.stack((prediction,finite))

    def replay(self,tokens,pad=0):
        torch=self.owner.backend.torch
        values=list(tokens)+[pad]*(self.bucket-len(tokens))+[len(tokens)]
        with torch.inference_mode():
            self.inputs.copy_(torch.tensor([values],dtype=torch.long))
            self.graph.replay()
        token,finite=self.outputs.tolist()
        if finite!=1:raise RuntimeError('nonfinite prefill graph logits')
        return token


class PrefillGraphs:
    def __init__(self,backend,buckets,context_limit):
        if (getattr(backend.model.config,'model_type',None)!='qwen3' or any(
                t!='full_attention' for t in (getattr(backend.model.config,'layer_types',None) or []))):
            raise ValueError('prefill graphs require Qwen3 full attention')
        self.buckets=parse_buckets(buckets)
        self.context=max(self.buckets)
        if self.context>min(backend.context,context_limit):raise ValueError('prefill bucket exceeds context')
        install(backend)
        if self.context>backend.kv_budget_tokens(fraction=.75):raise ValueError('prefill scratch exceeds budget')
        self.backend=backend;self.cache=FixedKV(backend,1,self.context)
        self.graphs={};self.ready=None;self.in_use=False
        self.stats=dict(graph_prefills=0,eager_fallbacks=0,real_tokens=0,padded_tokens=0)
        import logging
        logger=logging.getLogger(__name__)
        for n in self.buckets:
            logger.info('HF_PREFILL_GRAPH_WARM_START q=%d',n)
            self.graphs[n]=PrefillGraph(self,n)
            logger.info('HF_PREFILL_GRAPH_WARM_DONE q=%d',n)
        self._record()

    def _wait(self):
        if self.ready is not None:self.backend.torch.cuda.current_stream(self.backend.device).wait_event(self.ready)
    def _record(self):
        self.ready=self.backend.torch.cuda.Event()
        self.ready.record(self.backend.torch.cuda.current_stream(self.backend.device))

    def fill_slot(self,slab,slot,tokens):
        if self.in_use:raise RuntimeError('prefill scratch already leased')
        bucket=choose_bucket(len(tokens),self.buckets)
        self.in_use=True
        try:
            self._wait()
            if bucket is None:
                cache=self.backend.new_cache();token=self.backend.prefill_first(cache,tokens)
                slab.install(slot,cache,tokens)
                self.stats['eager_fallbacks']+=1
            else:
                token=self.graphs[bucket].replay(tokens)
                # Never expose the scratch view beyond its copy/lease scope.
                slab.install(slot,PrefixView(self.cache,len(tokens)),tokens)
                self.stats['graph_prefills']+=1
                self.stats['padded_tokens']+=bucket-len(tokens)
            self.stats['real_tokens']+=len(tokens)
            return token
        finally:
            # Includes destination KV copies even when the next owner/thread uses
            # a different CUDA stream. Recording immediately after replay is wrong.
            self._record();self.in_use=False

    def status(self):
        return dict(buckets=list(self.buckets),scratch_context=self.context,cuda_graph=True,
                    attention='flash-kv',
                    fallback='EXPLICIT_EAGER_ABOVE_MAX_BUCKET',stats=dict(self.stats))
