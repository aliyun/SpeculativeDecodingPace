# SD/FSD core adapted from maxholsman/fuzzy-spec-dec and Transformers.
# Local greedy acceptance and padded-vocabulary support; Apache-2.0 (LICENSE).
import copy
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.functional import kl_div, log_softmax
from transformers.cache_utils import Cache
from transformers.generation.candidate_generator import AssistedCandidateGenerator, _prepare_attention_mask, _prepare_token_type_ids
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import ModelOutput, is_sklearn_available
if is_sklearn_available():
    from sklearn.metrics import roc_curve
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.generation.streamers import BaseStreamer
ALL_CACHE_NAMES = ['past_key_values', 'cache_params', 'state', 'mems', 'past_buckets_states']
GENERATION_MODES_MAPPING = {GenerationMode.SAMPLE: '_sample', GenerationMode.GREEDY_SEARCH: '_sample', GenerationMode.BEAM_SEARCH: '_beam_search', GenerationMode.BEAM_SAMPLE: '_beam_search', GenerationMode.ASSISTED_GENERATION: '_assisted_decoding'}

@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: tuple[torch.FloatTensor] | None = None
    logits: tuple[torch.FloatTensor] | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None
    past_key_values: Cache | None = None
    draft_token_acceptance_rate: float | None = None
    total_draft_tokens: int | None = None
    total_accepted_tokens: int | None = None

@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: tuple[torch.FloatTensor] | None = None
    logits: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[tuple[torch.FloatTensor]] | None = None
    cross_attentions: tuple[tuple[torch.FloatTensor]] | None = None
    decoder_hidden_states: tuple[tuple[torch.FloatTensor]] | None = None
    past_key_values: Cache | None = None
    draft_token_acceptance_rate: float | None = None
    total_draft_tokens: int | None = None
    total_accepted_tokens: int | None = None

def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        cur_len += 1
        added_len -= cur_len
    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i:i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs

class RawLogitsCandidateGenerator(AssistedCandidateGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assistant_config = getattr(self, 'assistant_generation_config', None)
        if assistant_config is None:
            assistant_config = self.assistant_model.generation_config
        if is_sklearn_available() and hasattr(assistant_config, 'assistant_confidence_threshold') and assistant_config.assistant_confidence_threshold:
            if not hasattr(self, 'probs'):
                self.probs = []
            if not hasattr(self, 'matches'):
                self.matches = []

    def get_candidates(self, input_ids: torch.LongTensor) -> tuple[torch.LongTensor, torch.FloatTensor | None, torch.FloatTensor | None]:
        input_ids = input_ids.to(self.assistant_model.device)
        min_new_tokens, max_new_tokens = self._calculate_new_tokens(input_ids)
        if max_new_tokens == 0:
            return (input_ids, None, None)
        self._update_past_and_masks(input_ids)
        generation_args = self._prepare_generation_args(input_ids, min_new_tokens, max_new_tokens)
        candidate_ids, candidate_logits_processed, candidate_logits_raw = self._generate_candidates(generation_args)
        return (candidate_ids, candidate_logits_processed, candidate_logits_raw)

    def _generate_candidates(self, generation_args: dict) -> tuple[torch.LongTensor, torch.FloatTensor | None, torch.FloatTensor | None]:
        assistant_output = self.assistant_model.generate(**generation_args, **self.assistant_kwargs)
        self.assistant_kwargs['past_key_values'] = assistant_output.past_key_values
        assistant_config = getattr(self, 'assistant_generation_config', None)
        if assistant_config is None:
            assistant_config = self.assistant_model.generation_config
        if is_sklearn_available() and hasattr(assistant_config, 'assistant_confidence_threshold') and assistant_config.assistant_confidence_threshold and (type(self) is RawLogitsCandidateGenerator):
            scores_tensor = torch.cat(assistant_output.scores, dim=0)
            scores_softmax = torch.softmax(scores_tensor, dim=-1)
            ids = assistant_output.sequences[-1, -len(assistant_output.scores):]
            p = scores_softmax[range(len(ids)), ids]
            self.probs.extend(p.tolist())
        candidate_logits_processed = torch.stack(assistant_output.scores, dim=1)
        candidate_ids = assistant_output.sequences
        candidate_logits_raw = None
        if self.generation_config.output_logits and hasattr(assistant_output, 'logits') and (assistant_output.logits is not None):
            candidate_logits_raw = torch.stack(assistant_output.logits, dim=1)
        return (candidate_ids, candidate_logits_processed, candidate_logits_raw)

def _speculative_sampling(candidate_input_ids, candidate_logits, candidate_length, new_logits, next_token_logits, is_done_candidate, candidate_logits_raw, fsd_threshold: float=0.0, fsd_div_type: str='js'):
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i
    target_probs = next_token_logits.softmax(dim=-1)
    cand_probs = candidate_logits_raw.softmax(dim=-1)
    cp, tp = (cand_probs, target_probs[:, :-1, :])
    if cp.shape[-1] != tp.shape[-1]:
        width = min(cp.shape[-1], tp.shape[-1])
        cp = cp[..., :width]
        cp = cp / cp.sum(dim=-1, keepdim=True)
        tp = tp[..., :width]
        tp = tp / tp.sum(dim=-1, keepdim=True)
    if fsd_div_type == 'kl':
        divs = kl_div(cp.log().clamp(min=-10000000000.0), tp, reduction='none').sum(dim=-1)
    elif fsd_div_type == 'js':
        m = 0.5 * (cp + tp)
        divs = (0.5 * torch.nn.functional.kl_div(torch.log(cp), m, reduction='none') + 0.5 * torch.nn.functional.kl_div(torch.log(tp), m, reduction='none')).sum(dim=-1)
    elif fsd_div_type == 'draft_tokens':
        draft_token_ids = new_candidate_input_ids
        draft_token_probs_candidate = cp[:, torch.arange(candidate_length), draft_token_ids].squeeze(0, 1)
        draft_token_probs_target = tp[:, torch.arange(candidate_length), draft_token_ids].squeeze(0, 1)
        divs = (draft_token_probs_candidate - draft_token_probs_target).abs().sum(dim=-1)
    else:
        raise ValueError(f'Invalid fsd_div_type: {fsd_div_type}')
    is_accepted_fsd = divs <= fsd_threshold
    r_i = torch.rand_like(probability_ratio)
    is_accepted_sd = r_i <= probability_ratio
    is_accepted = is_accepted_fsd | is_accepted_sd
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()
    if is_done_candidate and n_matches == candidate_length:
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, :n_matches + 1]
    else:
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp(p_n_plus_1 - q_n_plus_1, min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t
    return (valid_tokens, n_matches)

def _greedy_fuzzy_sampling(candidate_input_ids, candidate_logits_raw, candidate_length, next_token_logits, is_done_candidate, fsd_threshold: float=0.0, fsd_div_type: str='js'):
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    if fsd_div_type == 'exact':
        target_ids = next_token_logits.argmax(dim=-1)
        matches = new_candidate_input_ids == target_ids[:, :candidate_length]
        keep = matches[0].to(torch.int8).cumprod(dim=0)
        n_matches = int(keep.sum().item())
        if is_done_candidate and n_matches == candidate_length:
            n_matches = candidate_length - 1
            return (new_candidate_input_ids[:, :n_matches + 1], n_matches)
        bonus = target_ids[:, n_matches:n_matches + 1]
        return (torch.cat((new_candidate_input_ids[:, :n_matches], bonus), dim=-1), n_matches)
    cand_probs = candidate_logits_raw.softmax(dim=-1)
    target_probs = next_token_logits.softmax(dim=-1)
    cp, tp = (cand_probs, target_probs[:, :-1, :])
    if cp.shape[-1] != tp.shape[-1]:
        width = min(cp.shape[-1], tp.shape[-1])
        cp = cp[..., :width]
        cp = cp / cp.sum(dim=-1, keepdim=True)
        tp = tp[..., :width]
        tp = tp / tp.sum(dim=-1, keepdim=True)
    if fsd_div_type == 'kl':
        divs = kl_div(cp.log().clamp(min=-10000000000.0), tp, reduction='none').sum(dim=-1)
    elif fsd_div_type == 'js':
        m = 0.5 * (cp + tp)
        divs = (0.5 * torch.nn.functional.kl_div(cp.log().clamp(min=-10000000000.0), m, reduction='none') + 0.5 * torch.nn.functional.kl_div(tp.log().clamp(min=-10000000000.0), m, reduction='none')).sum(dim=-1)
    elif fsd_div_type == 'draft_tokens':
        draft_token_ids = new_candidate_input_ids
        draft_token_probs_candidate = cp[:, torch.arange(candidate_length), draft_token_ids].squeeze(0, 1)
        draft_token_probs_target = tp[:, torch.arange(candidate_length), draft_token_ids].squeeze(0, 1)
        divs = (draft_token_probs_candidate - draft_token_probs_target).abs().sum(dim=-1)
    else:
        raise ValueError(f'Invalid fsd_div_type: {fsd_div_type}')
    is_accepted = divs <= fsd_threshold
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()
    if is_done_candidate and n_matches == candidate_length:
        n_matches = n_matches - 1
        valid_tokens = new_candidate_input_ids[:, :n_matches + 1]
    else:
        p_n_plus_1 = target_probs[:, n_matches, :]
        next_tokens = torch.multinomial(p_n_plus_1, num_samples=1).squeeze(1)[None, :]
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], next_tokens), dim=-1)
        else:
            valid_tokens = next_tokens
    return (valid_tokens, n_matches)

def _assisted_decoding(model, input_ids: torch.LongTensor, logits_processor: LogitsProcessorList, stopping_criteria: StoppingCriteriaList, generation_config: GenerationConfig, synced_gpus: bool=False, streamer: Optional['BaseStreamer']=None, inputs_tensor: torch.FloatTensor | None=None, assistant_model: Optional['PreTrainedModel']=None, assistant_tokenizer: Optional['PreTrainedTokenizerBase']=None, tokenizer: Optional['PreTrainedTokenizerBase']=None, fsd_threshold: float=0.0, fsd_div_type: str='js', track_acceptance_metrics: bool=False, **model_kwargs) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor]:
    if not model_kwargs['use_cache']:
        raise ValueError('assisted generate requires `use_cache=True`')
    if generation_config.cache_implementation in ['static', 'hybrid', 'sliding_window'] or ('past_key_values' in model_kwargs and hasattr(model_kwargs['past_key_values'], 'layers') and any((getattr(l, 'is_compileable', False) for l in model_kwargs['past_key_values'].layers))):
        raise ValueError('assisted generate is not supported with Static cache classes`')
    if assistant_model is None:
        raise ValueError('assistant_model is required for assisted generation')
    generation_config.output_logits = True
    candidate_generator = RawLogitsCandidateGenerator(input_ids=input_ids, assistant_model=assistant_model, generation_config=generation_config, model_kwargs=model_kwargs, inputs_tensor=inputs_tensor, logits_processor=logits_processor)
    do_sample = generation_config.do_sample
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    if track_acceptance_metrics:
        total_draft_tokens = 0
        total_accepted_tokens = 0
    else:
        total_draft_tokens = None
        total_accepted_tokens = None
    scores = () if return_dict_in_generate and output_scores else None
    raw_logits = () if return_dict_in_generate and output_logits else None
    decoder_attentions = () if return_dict_in_generate and output_attentions else None
    cross_attentions = () if return_dict_in_generate and output_attentions else None
    decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
        encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
    batch_size, cur_len = input_ids.shape[:2]
    if batch_size > 1:
        raise ValueError('assisted generate is only supported for batch_size = 1')
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
    this_peer_finished = False
    is_first_iteration = True
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        cur_len = input_ids.shape[1]
        candidate_input_ids, candidate_logits, candidate_logits_raw = candidate_generator.get_candidates(input_ids)
        candidate_input_ids = candidate_input_ids.to(model.device)
        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(model.device)
        if candidate_logits_raw is not None:
            candidate_logits_raw = candidate_logits_raw.to(model.device)
        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        is_done_candidate = stopping_criteria(candidate_input_ids, None)
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = _prepare_attention_mask(candidate_kwargs, candidate_input_ids.shape[1], model.config.is_encoder_decoder)
        candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
        if 'cache_position' in candidate_kwargs:
            candidate_kwargs['cache_position'] = torch.cat((candidate_kwargs['cache_position'], torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long)), dim=0)
        model_inputs = model.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
        if 'logits_to_keep' in model_inputs:
            model_inputs['logits_to_keep'] = candidate_length + 1
        outputs = model(**model_inputs)
        new_logits = outputs.logits[:, -candidate_length - 1:].to(dtype=torch.float32, device=input_ids.device)
        next_token_logits = new_logits.clone()
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, :cur_len + i], new_logits[:, i, :])
        if candidate_logits is not None and (do_sample or (fsd_threshold is not None and fsd_threshold > 0.0)):
            if do_sample:
                valid_tokens, n_matches = _speculative_sampling(candidate_input_ids, candidate_logits, candidate_length, new_logits, next_token_logits, is_done_candidate, candidate_logits_raw=candidate_logits_raw, fsd_threshold=fsd_threshold, fsd_div_type=fsd_div_type)
            else:
                valid_tokens, n_matches = _greedy_fuzzy_sampling(candidate_input_ids, candidate_logits_raw, candidate_length, next_token_logits, is_done_candidate, fsd_threshold=fsd_threshold, fsd_div_type=fsd_div_type)
            if track_acceptance_metrics and candidate_length > 0:
                total_draft_tokens += candidate_length
                total_accepted_tokens += n_matches
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)
            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, :n_matches + 1]
            if track_acceptance_metrics and candidate_length > 0:
                total_draft_tokens += candidate_length
                total_accepted_tokens += n_matches
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[1]
        outputs.past_key_values.crop(new_cur_len - 1)
        candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder, num_new_tokens=n_matches + 1)
        if synced_gpus and this_peer_finished:
            continue
        if return_dict_in_generate:
            newly_added_length = n_matches + 1
            if output_scores:
                scores += tuple((new_logits[:, i, :] for i in range(newly_added_length)))
            if output_logits:
                raw_logits += tuple((next_token_logits[:, i, :] for i in range(newly_added_length)))
            newly_added_length = new_cur_len if is_first_iteration else newly_added_length
            if output_attentions:
                if model.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(cross_attentions, outputs.cross_attentions, cur_len, newly_added_length)
                    decoder_attentions = _split_model_outputs(decoder_attentions, outputs.decoder_attentions, cur_len, newly_added_length, is_decoder_attention=True)
                elif outputs.attentions[0] is not None:
                    decoder_attentions = _split_model_outputs(decoder_attentions, outputs.attentions, cur_len, newly_added_length, is_decoder_attention=True)
            if output_hidden_states:
                if model.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(decoder_hidden_states, outputs.decoder_hidden_states, cur_len, newly_added_length)
                else:
                    decoder_hidden_states = _split_model_outputs(decoder_hidden_states, outputs.hidden_states, cur_len, newly_added_length)
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        is_first_iteration = False
    if streamer is not None:
        streamer.end()
    if hasattr(candidate_generator, 'assistant_model') and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == 'heuristic':
        candidate_generator.assistant_model.generation_config.num_assistant_tokens = candidate_generator.num_assistant_tokens
    if track_acceptance_metrics:
        acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    else:
        acceptance_rate = None
        total_draft_tokens = None
        total_accepted_tokens = None
    if return_dict_in_generate:
        cache = None
        if any((cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES)):
            cache_key = next((cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs))
            cache = model_kwargs[cache_key]
        if model.config.is_encoder_decoder:
            base_dict = {'sequences': input_ids, 'scores': scores, 'logits': raw_logits, 'encoder_attentions': encoder_attentions, 'encoder_hidden_states': encoder_hidden_states, 'decoder_attentions': decoder_attentions, 'cross_attentions': cross_attentions, 'decoder_hidden_states': decoder_hidden_states, 'past_key_values': cache}
            output_class = GenerateEncoderDecoderOutput
        else:
            base_dict = {'sequences': input_ids, 'scores': scores, 'logits': raw_logits, 'attentions': decoder_attentions, 'hidden_states': decoder_hidden_states, 'past_key_values': cache}
            output_class = GenerateDecoderOnlyOutput
        if track_acceptance_metrics:
            try:
                return output_class(**base_dict, draft_token_acceptance_rate=acceptance_rate, total_draft_tokens=total_draft_tokens, total_accepted_tokens=total_accepted_tokens)
            except TypeError:
                output = output_class(**base_dict)
                try:
                    output.draft_token_acceptance_rate = acceptance_rate
                    output.total_draft_tokens = total_draft_tokens
                    output.total_accepted_tokens = total_accepted_tokens
                except Exception:
                    pass
                return output
        else:
            return output_class(**base_dict)
    else:
        return input_ids

def generate(model, inputs: torch.Tensor | None=None, generation_config: GenerationConfig | None=None, logits_processor: LogitsProcessorList | None=None, stopping_criteria: StoppingCriteriaList | None=None, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None=None, synced_gpus: bool | None=None, assistant_model: Optional['PreTrainedModel']=None, streamer: Optional['BaseStreamer']=None, negative_prompt_ids: torch.Tensor | None=None, negative_prompt_attention_mask: torch.Tensor | None=None, **kwargs) -> GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput | torch.LongTensor:
    custom_generate = kwargs.pop('custom_generate', None)
    trust_remote_code = kwargs.pop('trust_remote_code', None)
    fsd_threshold = kwargs.pop('fsd_threshold', 0.0)
    fsd_div_type = kwargs.pop('fsd_div_type', 'js')
    track_acceptance_metrics = kwargs.pop('track_acceptance_metrics', False)
    generation_mode_kwargs = model._extract_generation_mode_kwargs(None, kwargs, synced_gpus, assistant_model, streamer)
    generation_mode_kwargs['fsd_threshold'] = fsd_threshold
    generation_mode_kwargs['fsd_div_type'] = fsd_div_type
    generation_mode_kwargs['track_acceptance_metrics'] = track_acceptance_metrics
    has_default_max_length = kwargs.get('max_length') is None and (generation_config is None or generation_config.max_length is None)
    has_default_min_length = kwargs.get('min_length') is None and (generation_config is None or generation_config.min_length is None)
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)
    generation_mode = generation_config.get_generation_mode(assistant_model)
    decoding_method = getattr(type(model), GENERATION_MODES_MAPPING[generation_mode])
    model._validate_model_kwargs(model_kwargs.copy())
    model._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    accepts_attention_mask = 'attention_mask' in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = 'encoder_outputs' not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get('attention_mask', None) is not None
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
    if 'inputs_tensor' in inspect.signature(decoding_method).parameters.keys():
        generation_mode_kwargs['inputs_tensor'] = inputs_tensor
    batch_size = inputs_tensor.shape[0]
    device = inputs_tensor.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
    if not model.config.is_encoder_decoder:
        if generation_config._pad_token_tensor is not None and batch_size > 1 and (len(inputs_tensor.shape) == 2) and (torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0):
            import logging
            logger = logging.get_logger(__name__)
            logger.warning("A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.")
    if not model.config.is_encoder_decoder and model_input_name == 'inputs_embeds':
        generation_config.use_cache = True
    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs['attention_mask'] = model._prepare_attention_mask_for_generation(inputs_tensor, generation_config, model_kwargs)
    elif kwargs_has_attention_mask:
        if model_input_name == 'input_ids' and len(model_kwargs['attention_mask'].shape) > 2:
            raise ValueError('`attention_mask` passed to `generate` must be 2D.')
    if model.config.is_encoder_decoder and 'encoder_outputs' not in model_kwargs:
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, generation_config)
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs, decoder_start_token_id=generation_config._decoder_start_token_tensor, device=inputs_tensor.device)
    else:
        input_ids = inputs_tensor if model_input_name == 'input_ids' else model_kwargs.pop('input_ids')
    input_ids, model_kwargs = model._expand_inputs_for_generation(input_ids=input_ids, expand_size=max(generation_config.num_beams, generation_config.num_return_sequences), is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs)
    if generation_config.token_healing:
        input_ids = model.heal_tokens(input_ids, generation_mode_kwargs.get('tokenizer'))
    if streamer is not None:
        streamer.put(input_ids.cpu())
    input_ids_length = input_ids.shape[1]
    generation_config = model._prepare_generated_length(generation_config=generation_config, has_default_max_length=has_default_max_length, has_default_min_length=has_default_min_length, model_input_name=model_input_name, inputs_tensor=inputs_tensor, input_ids_length=input_ids_length)
    if model._supports_logits_to_keep() and 'logits_to_keep' not in model_kwargs:
        model_kwargs['logits_to_keep'] = 1
    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
    max_cache_length = generation_config.max_length - 1
    if inputs_tensor.shape[1] != input_ids_length and model_input_name == 'inputs_embeds' and (not model.config.is_encoder_decoder):
        max_cache_length += inputs_tensor.shape[1]
    model._prepare_cache_for_generation(generation_config, model_kwargs, generation_mode, batch_size, max_cache_length)
    if model.device.type != input_ids.device.type:
        warnings.warn(f"You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model is on {model.device.type}. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('{model.device.type}') before running `.generate()`.", UserWarning)
    prepared_logits_processor = model._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor, device=inputs_tensor.device, model_kwargs=model_kwargs, negative_prompt_ids=negative_prompt_ids, negative_prompt_attention_mask=negative_prompt_attention_mask)
    prepared_stopping_criteria = model._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=generation_mode_kwargs.get('tokenizer'))
    model_kwargs['use_cache'] = generation_config.use_cache
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        result = _assisted_decoding(model, input_ids, logits_processor=prepared_logits_processor, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, **generation_mode_kwargs, **model_kwargs)
    else:
        result = decoding_method(model, input_ids, logits_processor=prepared_logits_processor, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, **generation_mode_kwargs, **model_kwargs)
    return result
