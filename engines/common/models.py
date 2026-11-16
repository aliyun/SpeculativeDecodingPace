import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct

from .policy import CONTEXT

MAX_VOCAB = 1_048_576
FAMILIES = {'qwen3', 'qwen2', 'llama', 'gemma3', 'gemma3_text'}


def _integer(value, name, maximum=None):
    if type(value) is not int or value < 1 or maximum is not None and value > maximum:
        raise ValueError('invalid '+name)
    return value


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('JSON object required: '+str(path))
    return value


def _stop_ids(value):
    if value is None:
        return set()
    values = value if isinstance(value, list) else [value]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('invalid model EOS token IDs')
    return set(values)


@dataclass(frozen=True)
class ModelInfo:
    family: str
    vocab_size: int
    context: int
    eos_ids: tuple

    @property
    def gemma3(self):
        return self.family in ('gemma3', 'gemma3_text')


def model_info(model):
    """The supported text paths retain the engine's own model implementation."""
    root = Path(model)
    config = _json(root/'config.json')
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        raise ValueError('invalid text model config')
    family = text.get('model_type', config.get('model_type'))
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError('unsupported native model family: '+str(family))
    # Gemma3TextConfig omits default-valued fields in composite 27B config.json.
    # These are upstream config defaults, not guesses from the directory name.
    gemma3 = family in ('gemma3', 'gemma3_text')
    vocabulary = _integer(text.get('vocab_size', 262208 if gemma3 else None),
                          'model vocabulary', MAX_VOCAB)
    context = _integer(text.get('max_position_embeddings', 131072 if gemma3 else None),
                       'model context')
    if context < CONTEXT:
        raise ValueError('native model must support the fixed 32768-token service context')
    eos = _stop_ids(text.get('eos_token_id', 1 if gemma3 else None))
    eos.update(_stop_ids(config.get('eos_token_id')))
    generation = root/'generation_config.json'
    if generation.is_file():
        eos.update(_stop_ids(_json(generation).get('eos_token_id')))
    if any(t >= vocabulary for t in eos):
        raise ValueError('model EOS outside vocabulary')
    return ModelInfo(family, vocabulary, CONTEXT, tuple(sorted(eos)))


def served_name(model, explicit=None, *, engine='hf'):
    if explicit is None:
        name = Path(model).name
    else:
        name = explicit
    if not isinstance(name, str) or not name.strip() or any(ord(c) < 32 for c in name):
        raise ValueError('nonempty served model name required')
    return name


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple
    model_vocab_size: int
    context: int
    excluded_token_ids: tuple = ()

    def identity(self, count=None):
        count = len(self.tokens) if count is None else _integer(count, 'shared vocabulary', MAX_VOCAB)
        if not 1 <= count <= min(len(self.tokens), self.model_vocab_size):
            raise ValueError('edge does not contain the target token-ID space')
        digest = hashlib.sha256(json.dumps(self.tokens[:count], ensure_ascii=True,
            separators=(',', ':')).encode()).hexdigest()
        result = dict(protocol=1, tokenizer_count=count, token_id_sha256=digest,
                      model_vocab_size=self.model_vocab_size, context=self.context)
        if self.excluded_token_ids:
            result['excluded_nontext_token_ids'] = list(self.excluded_token_ids)
        return result


def _hf_vocabulary(root):
    info = model_info(root)
    data = _json(Path(root)/'tokenizer.json')
    tokenizer_model = data.get('model')
    if not isinstance(tokenizer_model, dict):
        raise ValueError('tokenizer model object required')
    vocab = tokenizer_model.get('vocab')
    if isinstance(vocab, dict):
        pairs = list(vocab.items())
    elif isinstance(vocab, list):
        # HF Unigram (Gemma/SentencePiece) stores [token, score] in ID order.
        if any(not isinstance(row, list) or len(row) != 2 for row in vocab):
            raise ValueError('invalid Unigram vocabulary')
        pairs = [(row[0], index) for index, row in enumerate(vocab)]
    else:
        raise ValueError('BPE or Unigram tokenizer.json vocabulary required')
    mapping = {}
    for word, index in pairs:
        if not isinstance(word, str) or type(index) is not int or not 0 <= index < info.vocab_size:
            raise ValueError('invalid token-ID mapping')
        if index in mapping:
            raise ValueError('duplicate base token ID')
        mapping[index] = word
    added = data.get('added_tokens', [])
    if not isinstance(added, list):
        raise ValueError('invalid added tokens')
    excluded = []
    for item in added:
        if not isinstance(item, dict):
            raise ValueError('invalid added token entry')
        index, word = item.get('id'), item.get('content')
        if (info.gemma3 and type(index) is int and index == 262144
                and word == '<image_soft_token>' and item.get('special') is True
                and len(mapping) == index and index not in mapping):
            # Shared Gemma3 tokenizer appends an image-only input marker. The
            # 1B text model has no embedding/output row for it. Keep every
            # text ID unchanged and expose this explicit protocol exclusion.
            if index in excluded:
                raise ValueError('duplicate excluded nontext token ID')
            excluded.append(index)
            continue
        if (type(index) is not int or not 0 <= index < info.vocab_size
                or not isinstance(word, str) or index in mapping and mapping[index] != word):
            raise ValueError('conflicting added token ID')
        mapping[index] = word
    if not mapping or set(mapping) != set(range(len(mapping))):
        raise ValueError('contiguous token-ID mapping required')
    return Vocabulary(tuple(mapping[i] for i in range(len(mapping))), info.vocab_size,
                      info.context, tuple(excluded))


class _GGUF:
    """Bounded metadata-only GGUF v2/v3 reader, never visiting tensor data."""
    formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
               6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

    def __init__(self, file):
        self.file = file
        self.limit = min(file.seek(0, 2), 128*1024*1024)
        file.seek(0)

    def take(self, count):
        if count < 0 or self.file.tell()+count > self.limit:
            raise ValueError('truncated or oversized GGUF metadata')
        value = self.file.read(count)
        if len(value) != count:
            raise ValueError('truncated GGUF metadata')
        return value

    def number(self, kind):
        fmt = '<'+self.formats[kind]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self, keep=True):
        size = self.number(10)
        if size > 16*1024*1024:
            raise ValueError('oversized GGUF string')
        raw = self.take(size)
        return raw.decode('utf-8') if keep else None

    def value(self, kind, keep=True):
        if kind in self.formats:
            value = self.number(kind)
            return value if keep else None
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            element, count = self.number(4), self.number(10)
            if element not in (*self.formats, 8) or count > MAX_VOCAB:
                raise ValueError('unsupported or oversized GGUF metadata array')
            values = [] if keep else None
            for _ in range(count):
                value = self.value(element, keep)
                if keep:
                    values.append(value)
            return values
        raise ValueError('unsupported GGUF metadata type')


def _gguf_vocabulary(path, context):
    context = _integer(context, 'GGUF slot context', CONTEXT)
    with Path(path).open('rb') as file:
        reader = _GGUF(file)
        if reader.take(4) != b'GGUF' or reader.number(4) not in (2, 3):
            raise ValueError('GGUF v2 or v3 required')
        reader.number(10)  # Tensor count; tensors and weights are not read.
        count = reader.number(10)
        if count > 100_000:
            raise ValueError('oversized GGUF metadata table')
        metadata = {}
        for _ in range(count):
            key, kind = reader.string(), reader.number(4)
            keep = key in ('general.architecture', 'tokenizer.ggml.tokens') or key.endswith('.context_length')
            value = reader.value(kind, keep)
            if keep:
                if key in metadata:
                    raise ValueError('duplicate GGUF identity metadata')
                metadata[key] = value
    words = metadata.get('tokenizer.ggml.tokens')
    if not isinstance(words, list) or not words or any(not isinstance(w, str) for w in words):
        raise ValueError('GGUF token-ID mapping required')
    architecture = metadata.get('general.architecture')
    trained_context = _integer(metadata.get(str(architecture)+'.context_length'), 'GGUF trained context')
    if context > trained_context:
        raise ValueError('CPU slot context exceeds GGUF trained context')
    return Vocabulary(tuple(words), len(words), context)


@lru_cache(maxsize=2)
def vocabulary(model, context=CONTEXT):
    """Models are immutable for a serving process; restart after changing assets."""
    path = Path(model)
    return (_gguf_vocabulary(path, context) if path.suffix.lower() == '.gguf'
            else _hf_vocabulary(path))


def check_identity(expected, actual, context):
    if not isinstance(actual, dict):
        raise ValueError('missing edge model identity')
    for key in ('protocol', 'tokenizer_count', 'model_vocab_size', 'context'):
        if type(actual.get(key)) is not int:
            raise ValueError('invalid edge model identity: '+key)
    if (actual['protocol'] != 1 or actual['tokenizer_count'] != expected['tokenizer_count']
            or actual.get('token_id_sha256') != expected['token_id_sha256']
            or actual['model_vocab_size'] < expected['tokenizer_count']
            or actual['context'] != context):
        raise ValueError('target / draft token-ID mapping or context mismatch')


def compare_pair(target, draft, cpu_context=12288):
    target_vocabulary = vocabulary(str(Path(target).resolve()))
    draft_path = Path(draft).resolve()
    context = cpu_context if draft_path.suffix.lower() == '.gguf' else CONTEXT
    draft_vocabulary = vocabulary(str(draft_path), context)
    expected = target_vocabulary.identity()
    actual = draft_vocabulary.identity(expected['tokenizer_count'])
    check_identity(expected, actual, context)
    return dict(scope='LOCAL_METADATA_ONLY_NOT_GPU_QUALIFICATION', target=str(target), draft=str(draft),
                target_identity=expected, draft_identity=actual,
                caveat='Exact token-ID mapping is not an acceptance-rate, quality or performance result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--target', required=True, help='local HF target directory')
    parser.add_argument('--draft', required=True, help='local HF draft directory or CPU GGUF')
    parser.add_argument('--cpu-context', type=int, default=12288)
    args = parser.parse_args()
    try:
        result = compare_pair(args.target, args.draft, args.cpu_context)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
