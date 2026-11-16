from collections import defaultdict


class CacheAgreement:
    MAX_PENDING=64

    def __init__(self):
        self.evidence=defaultdict(lambda:[0,0])
        self.pending=defaultdict(list)
        self.seen={}
        self.boundaries={}
        self.registered={}

    @staticmethod
    def scope(offer):
        return offer.source,offer.layer,offer.width,offer.origin

    def quality(self,offer):
        good,bad=self.evidence.get((offer.rid,self.scope(offer)),(0,0))
        return (good+1.)/(good+bad+2.),good+bad

    def record(self,request,offers,*,register=True):
        rid=str(request.rid);prompt=tuple(request.prompt);generated=tuple(request.generated)
        previous=self.boundaries.get(rid)
        if previous is not None:
            old_prompt,old_generated=previous
            if prompt!=old_prompt or generated[:len(old_generated)]!=old_generated:
                raise ValueError('CACHE_SHADOW_NONMONOTONE_COMMIT')
        kept=[]
        for item in self.pending[rid]:
            offer,checked=item
            scope=self.scope(offer);key=rid,scope;offset=len(offer.generated)
            limit=min(len(offer.tokens),len(generated)-offset)
            failed=False
            while checked<limit:
                position=offset+checked
                matches=offer.tokens[checked]==generated[position]
                if position>self.seen.get(key,-1):
                    self.evidence[key][0 if matches else 1]+=1
                    self.seen[key]=position
                checked+=1
                if not matches:failed=True;break
            if not failed and checked<len(offer.tokens):kept.append((offer,checked))
        if register and self.registered.get(rid)!=(prompt,generated):
            # One longest actual proposal per scope at this boundary. Different
            # retrieval tiers stay distinct; repeated readiness does not add one.
            longest={}
            for offer in offers:
                if offer.rid!=rid or offer.prompt!=prompt or offer.generated!=generated:
                    raise ValueError('CACHE_SHADOW_BINDING')
                scope=self.scope(offer)
                if scope not in longest or len(offer.tokens)>len(longest[scope].tokens):
                    longest[scope]=offer
            kept.extend((offer,0) for offer in longest.values())
            self.registered[rid]=prompt,generated
        self.pending[rid]=kept[-self.MAX_PENDING:]
        self.boundaries[rid]=prompt,generated

    def forget(self,rid):
        rid=str(rid)
        self.pending.pop(rid,None);self.boundaries.pop(rid,None);self.registered.pop(rid,None)
        for values in (self.evidence,self.seen):
            for key in list(values):
                if key[0]==rid:values.pop(key)
