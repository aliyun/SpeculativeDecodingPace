import json
import random
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Union
import numpy as np
import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from model.gemma3 import Gemma3ForCausalLM
from model.llama import LlamaForCausalLM
from model.qwen2 import Qwen2ForCausalLM
from model.qwen3 import Qwen3ForCausalLM

def convert_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case 'bf16':
            return torch.bfloat16
        case 'fp16':
            return torch.float16
        case 'fp32':
            return torch.float32
        case _:
            raise ValueError(f'Unsupported dtype: {dtype}')

def graph_model_class(name: str):
    """Select the tree-mask model by checkpoint model_type, with a name fallback."""
    model_type = None
    try:
        model_type = AutoConfig.from_pretrained(name).model_type
    except Exception:
        pass
    if model_type == 'llama':
        return LlamaForCausalLM
    if model_type in ('qwen3', 'qwen3_moe'):
        return Qwen3ForCausalLM
    if model_type in ('qwen2', 'qwen2_moe'):
        return Qwen2ForCausalLM
    if model_type in ('gemma3', 'gemma3_text'):
        return Gemma3ForCausalLM
    lowered = name.lower()
    if 'llama' in lowered or 'vicuna' in lowered:
        return LlamaForCausalLM
    if 'qwen3' in lowered:
        return Qwen3ForCausalLM
    raise ValueError(f'Unsupported model: {name} (config model_type={model_type!r}). Add a tree-mask implementation under src/model/ for this architecture.')

def load_graph_model(name: str, device: torch.device, dtype: torch.dtype):
    cls = graph_model_class(name)
    if hasattr(cls, 'from_checkpoint'):
        return cls.from_checkpoint(name, dtype, device)
    model = cls.from_pretrained(name, torch_dtype=dtype, device_map=device)
    model.eval()
    return model

def load_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_dataset(name: str, model_name: str | None=None, reasoning: bool | None=False):
    dataset_name = f'{name.lower()}_prompts'
    try:
        if dataset_name == 'specbench_prompts':
            if model_name is None:
                raise ValueError('Model name is required for specbench dataset.')
            file_path = f'data/{dataset_name}.jsonl'
            questions = []
            _tokenizer = load_tokenizer(model_name)
            with open(file_path, 'r') as question_file:
                for line in question_file:
                    if line:
                        raw_question = json.loads(line)
                        chat = [{'role': 'user', 'content': raw_question['turns'][0]}]
                        conv = _tokenizer.apply_chat_template(chat, tokenize=False, enable_thinking=reasoning, add_generation_prompt=True)
                        questions.append(conv)
            return questions
        file_path = f'data/{dataset_name}.json'
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        return [x[1] for x in dataset]
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Missing `data/{dataset_name}.json` file.') from e

def invert_mask(mask: torch.Tensor):
    if mask.min() == 0:
        min_dtype = torch.finfo(mask.dtype).min
        mask = mask.eq(0.0).to(dtype=mask.dtype) * min_dtype
    return mask

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class Timing:

    def __init__(self, device: Optional[Union[torch.device, str]]=None, mode='no-sync', enabled: bool=True):
        if mode not in ['no-sync', 'sync', 'event']:
            raise ValueError(f'Unsupported mode: {mode}')
        if mode in ['sync', 'event'] and device is None:
            raise ValueError('Device must be specified for sync and event modes')
        self.device = device
        self.mode = mode
        self._elapsed = 0.0
        self._enabled = enabled

    def __enter__(self):
        if not self._enabled:
            return self
        if self.mode == 'event':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.stream = torch.cuda.default_stream(device=self.device)
            self.start_event.record(self.stream)
        else:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled:
            return
        if self.mode == 'sync':
            torch.cuda.synchronize(self.device)
            self._elapsed = (time.perf_counter() - self.start) * 1000
        elif self.mode == 'no-sync':
            self._elapsed = (time.perf_counter() - self.start) * 1000
        elif self.mode == 'event':
            self.end_event.record(self.stream)
            torch.cuda.synchronize(self.device)
            self._elapsed = self.start_event.elapsed_time(self.end_event)

    @property
    def elapsed(self):
        return self._elapsed

def encode(target: torch.Tensor):
    return target.contiguous().cpu().numpy().tobytes()

def decode(target: bytes, device: torch.device, dtype: torch.dtype, shape):
    return torch.frombuffer(target, dtype=dtype).reshape(shape).to(device=device)

@torch.inference_mode()
def sampler_from_logits(logits: torch.Tensor, temperature, top_p=0.8, min_tokens_to_keep=1):
    if temperature > 0:
        if temperature != 1:
            scores = logits / temperature
        else:
            scores = logits
        if top_p != 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= 1 - top_p
            sorted_indices_to_remove[..., -min_tokens_to_keep:] = False
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            scores.masked_fill_(indices_to_remove, -float('inf'))
        probs = torch.softmax(scores, dim=-1)
        logits_prefix_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        probs_flat = probs.view(-1, vocab_size)
        _ = torch.multinomial(probs_flat, 1)
        selection_flat = torch.multinomial(probs_flat, 1)
        selection = selection_flat.view(logits_prefix_shape)
    else:
        selection = torch.argmax(logits, dim=-1)
    return selection
