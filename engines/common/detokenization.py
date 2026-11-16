from dataclasses import dataclass
import re

_BYTE = re.compile(r'<0x([0-9a-fA-F]{2})>\Z')
_DECODER = {'type': 'Sequence', 'decoders': [
    {'type': 'Replace', 'pattern': {'String': '\u2581'}, 'content': ' '},
    {'type': 'ByteFallback'}, {'type': 'Fuse'}]}


@dataclass(frozen=True)
class ByteFallbackPlan:
    byte_ids: frozenset
    filtered_ids: frozenset


def validate_byte_fallback_decoder(document):
    if not isinstance(document, dict) or document.get('decoder') != _DECODER:
        raise ValueError('unsupported byte-fallback decoder composition')


def byte_fallback_plan(tokens, document, tokenizer, *, excluded_token_ids=()):
    """Bind immutable token IDs and the initialized HF decoder before allocation."""
    validate_byte_fallback_decoder(document)
    mapping = {}
    for index, token in enumerate(tokens):
        match = _BYTE.fullmatch(token)
        if match:
            value = int(match[1], 16)
            if value in mapping:
                raise ValueError('ambiguous byte fallback vocabulary')
            mapping[value] = index
    if len(mapping) != 256:
        raise ValueError('incomplete byte fallback vocabulary')
    added = document.get('added_tokens', [])
    if not isinstance(added, list):
        raise ValueError('invalid added token metadata')
    seen = set()
    for item in added:
        if not isinstance(item, dict):
            raise ValueError('invalid added token metadata')
        index, content, special = item.get('id'), item.get('content'), item.get('special')
        if (type(index) is not int or index < 0 or type(special) is not bool
                or not isinstance(content, str) or index in seen):
            raise ValueError('invalid added token metadata')
        seen.add(index)
        if index in excluded_token_ids:
            continue
        if index >= len(tokens) or tokens[index] != content:
            raise ValueError('added token does not match negotiated vocabulary')
    # HF initialization can change special flags from tokenizer.json. Its
    # actual backend (including flags absent from all_special_ids) is authoritative.
    runtime_filtered = set(tokenizer.all_special_ids)
    runtime_filtered.update(index for index, token in tokenizer.added_tokens_decoder.items()
                            if token.special)
    filtered = set()
    for index in runtime_filtered:
        if type(index) is not int or index < 0:
            raise ValueError('invalid runtime filtered token ID')
        if index in excluded_token_ids:
            continue
        if index >= len(tokens):
            raise ValueError('runtime filtered token outside negotiated vocabulary')
        filtered.add(index)
    for index in mapping.values():
        if tokenizer.convert_ids_to_tokens(index) != tokens[index]:
            raise ValueError('runtime byte token does not match negotiated vocabulary')
    return ByteFallbackPlan(frozenset(mapping.values()), frozenset(filtered))


def stable_text(tokenizer, generated, plan=None, *, final):
    end = len(generated)
    if plan is not None and not final:
        # ByteFallback replaces an entire byte run when any byte is invalid.
        # Filtered special IDs do not terminate that run in the HF decoder.
        while end and (generated[end-1] in plan.byte_ids or generated[end-1] in plan.filtered_ids):
            end -= 1
    text = tokenizer.decode(generated[:end], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    return text.rstrip('\ufffd') if plan is None and not final else text
