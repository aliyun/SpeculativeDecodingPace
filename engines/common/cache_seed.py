import json
import logging
import os
import time
from pathlib import Path

from .models import vocabulary
from .policy import CONTEXT, DRAFT_TOKENS, MAX_OUTPUT

logger = logging.getLogger(__name__)

FORMAT = 'triage-unified-cache-seed'
VERSION = 1
HORIZON = 7
DEFAULT_MAX_TOKENS = 4_194_304
MAX_RECORDS = 65536
MAX_TOKENS_LIMIT = 4 * DEFAULT_MAX_TOKENS
# Only the token space decides whether a stored token means the same thing.
MODEL_KEYS = ('token_id_sha256', 'tokenizer_count', 'model_vocab_size')


def model_identity(model):
    """Token-space identity of the served target, from local metadata only."""
    known = vocabulary(str(model)).identity()
    return dict(token_id_sha256=known['token_id_sha256'],
                tokenizer_count=known['tokenizer_count'],
                model_vocab_size=known['model_vocab_size'],
                draft_tokens=DRAFT_TOKENS, horizon=HORIZON)


def _identity(value):
    if not isinstance(value, dict):
        raise ValueError('CACHE_SEED_IDENTITY')
    digest = value.get('token_id_sha256')
    count = value.get('tokenizer_count')
    model_vocab = value.get('model_vocab_size')
    draft = value.get('draft_tokens')
    horizon = value.get('horizon')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)):
        raise ValueError('CACHE_SEED_IDENTITY_TOKENIZER')
    if type(count) is not int or count < 2:
        raise ValueError('CACHE_SEED_IDENTITY_COUNT')
    if type(model_vocab) is not int or model_vocab < count:
        raise ValueError('CACHE_SEED_IDENTITY_VOCAB')
    if type(draft) is not int or draft not in (3, 5, 7):
        raise ValueError('CACHE_SEED_IDENTITY_DRAFT')
    if type(horizon) is not int or horizon < HORIZON:
        raise ValueError('CACHE_SEED_IDENTITY_HORIZON')
    return dict(token_id_sha256=digest, tokenizer_count=count,
                model_vocab_size=model_vocab, draft_tokens=draft, horizon=horizon)


def _ids(value, name, vocabulary_size, low, high):
    if (not isinstance(value, list) or not low <= len(value) <= high
            or any(type(token) is not int or not 0 <= token < vocabulary_size
                   for token in value)):
        raise ValueError('CACHE_SEED_%s_IDS' % name)
    return tuple(value)


def read_document(document, max_tokens=None):
    """Validate one parsed seed document; nothing is applied here."""
    if not isinstance(document, dict):
        raise ValueError('CACHE_SEED_DOCUMENT')
    if document.get('format') != FORMAT:
        raise ValueError('CACHE_SEED_FORMAT')
    if document.get('version') != VERSION:
        raise ValueError('CACHE_SEED_VERSION')
    identity = _identity(document.get('identity'))
    records = document.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('CACHE_SEED_RECORDS')
    if len(records) > MAX_RECORDS:
        raise ValueError('CACHE_SEED_RECORD_LIMIT')
    budget = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    rows, total = [], 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('CACHE_SEED_RECORD')
        prompt = _ids(record.get('prompt_ids'), 'PROMPT', identity['model_vocab_size'], 1, CONTEXT)
        output = _ids(record.get('output_ids'), 'OUTPUT', identity['model_vocab_size'], 1, MAX_OUTPUT)
        total += len(prompt)+len(output)
        if total > budget:
            raise ValueError('CACHE_SEED_TOKEN_BUDGET')
        rows.append((prompt, output))
    return identity, rows


def read(path, max_tokens=None):
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('CACHE_SEED_UNREADABLE') from error
    return read_document(document, max_tokens)


def document(identity, records):
    return dict(format=FORMAT, version=VERSION, identity=dict(identity),
                records=[dict(prompt_ids=list(prompt), output_ids=list(output))
                         for prompt, output in records])


def write(path, identity, records):
    """Write a seed that the loader accepts; refused documents never land."""
    value = document(identity, records)
    read_document(value)
    Path(path).write_text(json.dumps(value, separators=(',', ':'))+'\n')
    return value


def _require(identity, cache, model=None, expected=None):
    if identity['horizon'] != getattr(cache, 'horizon', HORIZON):
        raise ValueError('CACHE_SEED_HORIZON')
    known = expected if expected is not None else (
        model_identity(model) if model is not None else None)
    if known is None:
        return
    for key in MODEL_KEYS:
        if identity[key] != known[key]:
            raise ValueError('CACHE_SEED_MODEL_IDENTITY')


def load(cache, path, *, model=None, expected=None, max_tokens=None):
    """Seed ``cache`` from one file. Every refusal precedes the first write."""
    identity, rows = read(path, max_tokens)
    _require(identity, cache, model=model, expected=expected)
    started = time.monotonic()
    for prompt, output in rows:
        cache.seed(prompt, output)
    elapsed = time.monotonic()-started
    values = cache.snapshot()
    result = dict(records=len(rows), tokens=sum(len(p)+len(o) for p, o in rows),
                  elapsed_s=round(elapsed, 3),
                  prompt_entries=values.get('unified_prompt_entries'),
                  window_entries=values.get('unified_window_entries'),
                  band_entries=values.get('unified_band_entries'))
    logger.info('CACHE_SEED_LOADED records=%d tokens=%d elapsed_s=%.3f prompt=%s window=%s band=%s',
                result['records'], result['tokens'], result['elapsed_s'],
                result['prompt_entries'], result['window_entries'], result['band_entries'])
    return result


def verify(path, model, *, horizon=HORIZON, max_tokens=None):
    """Validate one seed against a target before any process is spawned."""
    identity, rows = read(path, max_tokens)
    if identity['horizon'] != horizon:
        raise ValueError('CACHE_SEED_HORIZON')
    known = model_identity(model)
    for key in MODEL_KEYS:
        if identity[key] != known[key]:
            raise ValueError('CACHE_SEED_MODEL_IDENTITY')
    return dict(records=len(rows), tokens=sum(len(prompt)+len(output) for prompt, output in rows))


def load_configured(prefetch, environ=None):
    """Startup hook: no-op unless ``TRIAGE_CACHE_SEED`` is set.

    The cloud is the only role that proposes, so a seed on an edge (or on a
    launch without the unified cache) is a configuration error, not a no-op.
    """
    environ = os.environ if environ is None else environ
    path = environ.get('TRIAGE_CACHE_SEED')
    if not path:
        return None
    cache = getattr(prefetch, 'similar_cache', None)
    if cache is None:
        raise ValueError('CACHE_SEED_REQUIRES_UNIFIED_CACHE')
    model = environ.get('TRIAGE_MODEL_PATH')
    if not model:
        raise ValueError('CACHE_SEED_REQUIRES_MODEL_PATH')
    raw = environ.get('TRIAGE_CACHE_SEED_MAX_TOKENS')
    max_tokens = None
    if raw is not None:
        if not raw.isdigit() or not 1 <= int(raw) <= MAX_TOKENS_LIMIT:
            raise ValueError('CACHE_SEED_MAX_TOKENS')
        max_tokens = int(raw)
    return load(cache, path, model=model, max_tokens=max_tokens)
