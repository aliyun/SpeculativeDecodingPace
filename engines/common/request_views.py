from .contracts import Request
from .policy import MAX_REQUESTS,token_ids


class RequestViews:
    def __init__(self):self.entries={}

    def get(self,owner,factory):
        # Other adapters, mutable buffers and scalar subclasses retain the
        # original complete construction/validation path, never an identity hit.
        if type(owner) is not dict:return factory(owner)
        rid=owner['id'];tokens=owner['tokens'];generated=owner['generated']
        length=owner['input_length'];maximum=owner['max_tokens'];done=owner['done']
        if (type(rid) is not str or type(tokens) is not tuple or type(generated) is not tuple
                or type(length) is not int or type(maximum) is not int or type(done) is not bool
                or not 0<length<=len(tokens)):
            return factory(owner)
        controls=(length,maximum,done)
        old=self.entries.get(rid)
        if (old is not None and old[0] is owner and old[1] is tokens
                and old[2] is generated and old[3]==controls):return old[4]
        # Validate the full native token tuple as well as factory's prompt and
        # generated fields before treating either tuple as an immutable proof.
        token_ids(tokens)
        view=factory(owner)
        if type(view) is not Request:return view
        if (owner['tokens'] is not tokens or owner['generated'] is not generated
                or type(owner['id']) is not str or type(owner['input_length']) is not int
                or type(owner['max_tokens']) is not int or type(owner['done']) is not bool
                or owner['id']!=rid or (owner['input_length'],owner['max_tokens'],owner['done'])!=controls
                or view.rid!=rid or view.prompt!=tokens[:length] or view.generated is not generated
                or view.maximum!=maximum or view.done is not done):
            raise ValueError('IMMUTABLE_VIEW_CHANGED_DURING_VALIDATION')
        if rid not in self.entries and len(self.entries)>=MAX_REQUESTS:
            raise ValueError('IMMUTABLE_VIEW_LIVE_LIMIT')
        self.entries[rid]=(owner,tokens,generated,controls,view)
        return view

    def forget(self,ids):
        for rid in ids:self.entries.pop(rid,None)
