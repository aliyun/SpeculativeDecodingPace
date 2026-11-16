from .policy import validate_budget, validate_sampling

def _native_body(body, tokenizer, *, is_chat=False, context_length=32768):
    if body.get("stream") is not True or body.get("temperature", 0) != 0 or body.get("n", 1) != 1:
        raise ValueError("Only greedy, single completion SSE is supported")
    maximum = body.get("max_tokens", 512)
    if type(maximum) is not int or not 1 <= maximum <= 1024:
        raise ValueError("max_tokens must be an integer from 1 to 1024")
    ignore_eos = body.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be boolean")
    minimum = body.get("min_tokens", 0)
    if type(minimum) is not int or minimum < 0 or minimum > maximum:
        raise ValueError("Invalid min_tokens")
    if minimum and not ignore_eos:
        raise ValueError("Positive min_tokens requires ignore_eos in this experiment")
    if body.get("stop") or body.get("stop_token_ids") or body.get("logprobs"):
        raise ValueError("Custom stops and logprobs are unsupported")
    if is_chat:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages required")
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=False)
    else:
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
        elif isinstance(prompt, list):
            vocabulary_size = len(tokenizer)
            if not all(type(t) is int and 0 <= t < vocabulary_size for t in prompt):
                raise ValueError("prompt must be raw text or a single list of token IDs")
            tokens = list(prompt)
        else:
            raise ValueError("prompt must be raw text or a single list of token IDs")
    if not tokens:
        raise ValueError("prompt must contain at least one token")
    if len(tokens) + maximum + 7 > context_length:
        raise ValueError("context limit")
    return tokens, maximum, ignore_eos


def parse_completion(body, tokenizer, *, is_chat=False, token_limit=None):
    if is_chat:
        raise ValueError('single-edge parity uses /v1/completions')
    validate_sampling(body)
    tokens, maximum, ignore = _native_body(body, tokenizer)
    if token_limit is not None and (type(token_limit) is not int or token_limit < 1
                                     or max(tokens) >= token_limit):
        raise ValueError('token outside negotiated text vocabulary')
    validate_budget(len(tokens), maximum)
    return tokens, maximum, ignore
