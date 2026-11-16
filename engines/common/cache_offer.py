from dataclasses import dataclass, field
from .policy import token_ids


@dataclass(frozen=True)
class CacheOffer:
    rid:str
    prompt:tuple
    generated:tuple
    tokens:tuple
    source:str
    layer:str
    width:int|None
    origin:str
    issuer:object=field(repr=False,compare=False)

    def __post_init__(self):
        if not isinstance(self.rid,str) or not self.rid:raise ValueError('CACHE_OFFER_ID')
        if any(type(v) is not tuple for v in (self.prompt,self.generated,self.tokens)):
            raise ValueError('CACHE_OFFER_IMMUTABLE_TOKENS')
        token_ids(self.prompt)
        if self.generated:token_ids(self.generated)
        token_ids(self.tokens)
        if (self.layer not in ('prompt','window','content','suffix1','fuzzy','lookup')
                or self.source!=('exact_prefix' if self.layer=='prompt' else 'suffix')
                or self.origin not in ('seed','live')
                or self.layer!='prompt' and self.origin!='live'
                or self.layer=='window' and (type(self.width) is not int or self.width<1)
                or self.layer!='window' and self.width is not None):
            raise ValueError('CACHE_OFFER_SOURCE')

    @property
    def detail(self):
        return dict(layer=self.layer,width=self.width,origin=self.origin)
