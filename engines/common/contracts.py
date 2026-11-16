from dataclasses import dataclass
from types import SimpleNamespace

from .policy import token_ids, CONTEXT, MAX_OUTPUT


class EdgeAdmissionError(ValueError):
    """Rejected edge capacity/context before model or session mutation.

    Unlike an internal model/CUDA error, this must not poison shared streams.
    """


@dataclass(frozen=True)
class Request:
    rid: str
    prompt: tuple
    generated: tuple
    maximum: int
    done: bool = False
    aborted: bool = False

    def __post_init__(self):
        if not isinstance(self.rid, str) or not self.rid:
            raise ValueError('request id required')
        token_ids(self.prompt)
        if self.generated:
            token_ids(self.generated)
        if self.rid.startswith('ec-'):
            # Public serving limits are checked at ingress. This internal
            # snapshot also represents the native context-boundary AR guard.
            if (type(self.maximum) is not int or not 1 <= self.maximum <= MAX_OUTPUT
                    or len(self.prompt)+self.maximum > CONTEXT):
                raise ValueError('invalid native request budget')
            if len(self.generated) > self.maximum:
                raise ValueError('committed output exceeds request budget')

    @property
    def prefix(self):
        return self.prompt+self.generated

    @property
    def origin_input_ids(self):
        return self.prompt

    @property
    def output_ids(self):
        return self.generated

    @property
    def sampling_params(self):
        return SimpleNamespace(max_new_tokens=self.maximum)

    def finished(self):
        return self.done


@dataclass(frozen=True)
class Proposal:
    rid: str
    prefix: tuple
    candidates: tuple
    aborted: bool = False
    growth_plan: object = None

    @property
    def nodes(self):
        return self.prefix[-1:]+self.candidates


@dataclass(frozen=True)
class Dispatch:
    mode: str
    reason: str
    proposals: tuple
