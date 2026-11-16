from typing import Optional

import os

import torch

import log


class Tree:
    def __init__(
        self,
        prefix_tokens: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        max_len: int,
    ):
        self._logger = log.get_logger()

        self._device = device
        self._dtype = dtype
        self._max_len = max_len
        self._prefix_tokens = prefix_tokens

        self.PROMPT = torch.tensor(0, device=self._device)
        self.GENERATED = torch.tensor(5, device=self._device)
        self.PROCESSED = torch.tensor(10, device=self._device)
        self.CANDIDATE = torch.tensor(15, device=self._device)
        self.POST_CANDIDATE = torch.tensor(20, device=self._device)
        self.POST_PROCESSED = torch.tensor(25, device=self._device)

        self.prefix_len = self._prefix_tokens.size(-1)
        self.end = self.prefix_len

        self._data = torch.zeros(
            (4, self._max_len), dtype=torch.long, device=self._device
        )
        self.logprobs = torch.zeros(
            self._max_len, dtype=torch.float32, device=self._device
        )
        self.amask = torch.zeros(
            (1, 1, self._max_len, self._max_len), dtype=self._dtype, device=self._device
        )

        self._initialize_data()
        self._init_attention_mask()

    def _initialize_data(self):
        self.tokens[: self.prefix_len] = self._prefix_tokens.flatten()
        self.positions[: self.prefix_len] = torch.arange(
            self.prefix_len, device=self._device
        )
        self.parents[1 : self.prefix_len] = torch.arange(
            self.prefix_len - 1, device=self._device
        )
        self.status[: self.prefix_len - 1] = self.PROMPT
        self.status[self.prefix_len - 1] = self.CANDIDATE

    def _init_attention_mask(self, px_len=None):
        px_len = px_len if px_len is not None else self.prefix_len
        self.amask.zero_()
        self.amask[..., :px_len, :px_len] = torch.tril(
            torch.ones((px_len, px_len), dtype=self._dtype, device=self._device)
        )

    def add(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor,
        parent_indices: torch.Tensor,
        logprobs: torch.Tensor,
        token_status: Optional[torch.Tensor] = None,
    ):
        token_status = self.CANDIDATE if token_status is None else token_status
        add_size = token_ids.numel()

        if add_size > self._max_len - self.end:
            raise ValueError(
                "required addition size %d exceeds remaining tree capacity of %d",
                add_size,
                self._max_len - self.end,
            )

        if not (
            add_size
            == token_positions.numel()
            == parent_indices.numel()
            == logprobs.numel()
        ):
            raise ValueError(
                "Inconsistent sizes of input tensors: %d, %d, %d, %d",
                add_size,
                token_positions.numel(),
                parent_indices.numel(),
                logprobs.numel(),
            )

        parent_indices_view = parent_indices.view(-1)

        self.tokens[self.end : self.end + add_size] = token_ids.view(-1)
        self.positions[self.end : self.end + add_size] = token_positions.view(-1)
        self.parents[self.end : self.end + add_size] = parent_indices_view
        self.status[self.end : self.end + add_size] = token_status
        self.logprobs[self.end : self.end + add_size] = logprobs.view(-1)

        amask_draft = self.amask[..., parent_indices_view, : self.end]
        new_amask_eye = torch.eye(add_size, device=self._device)[None, None, ...]
        amask_draft = torch.cat((amask_draft, new_amask_eye), dim=-1)
        self.amask[..., self.end : self.end + add_size, : self.end + add_size] = (
            amask_draft
        )

        self.end += add_size

    def gather(self, src_indices: torch.Tensor, dest_indices: torch.Tensor):
        if src_indices.dtype == torch.bool:
            src_indices = torch.where(src_indices)[0]

        if src_indices.numel() == 0:
            raise ValueError("No source indices provided.")

        interim_indices = torch.arange(self.end, device=self._device)
        interim_indices[src_indices] = dest_indices
        _parents = self.parents[src_indices]
        _top = int(_parents.max().item())
        if _top >= int(self.end) or int(_parents.min().item()) < 0:
            import sys, traceback
            sys.stderr.write('TREE_PARENT_OUT_OF_RANGE top=%d min=%d end=%d src=%d\n'
                             % (_top, int(_parents.min().item()), int(self.end), int(src_indices.numel())))
            traceback.print_stack(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError('TREE_PARENT_OUT_OF_RANGE')
        self.parents[src_indices] = interim_indices[self.parents[src_indices]]

        self._data[:, dest_indices] = self._data[:, src_indices]
        self.logprobs[dest_indices] = self.logprobs[src_indices]
        self.amask[..., dest_indices, :] = self.amask[..., src_indices, :]
        self.amask[..., :, dest_indices] = self.amask[..., :, src_indices]

        self.end = int(dest_indices.max().item()) + 1

        self.amask[..., self.end :, :].zero_()
        self.amask[..., :, self.end :].zero_()

        return interim_indices

    def reorder_by_sequence(self, seq_mask: torch.Tensor, seq_indices: torch.Tensor):
        new_len = torch.sum(seq_mask).item()

        if torch.any(seq_mask[self.prefix_len :]):
            src_indices = seq_indices[seq_indices >= self.prefix_len]
            dest_indices = torch.arange(self.prefix_len, new_len, device=self._device)

            self.tokens[dest_indices] = self.tokens[src_indices]
            self.positions[dest_indices] = dest_indices
            self.parents[dest_indices] = dest_indices - 1
            self.status[dest_indices] = self.GENERATED

        self.end = int(new_len)
        self.prefix_len = self.end
        self.status[: self.prefix_len] = self.PROMPT
        self.logprobs[self.end :].zero_()
        self._data[:, self.end :].zero_()

        self._init_attention_mask()

    def take_snapshot(self, name: str, tokenizer):
        tokens = {}

        for idx in range(self.prefix_len - 1, self.end):
            tokens[idx] = {
                "token": tokenizer.decode(self.tokens[idx].item()),
                "parent": self.parents[idx].item(),
                "status": self.status[idx].item(),
            }

        return {
            "type": "tree",
            "name": name,
            "tokens": tokens,
        }

    @property
    def tokens(self):
        return self._data[0]

    @property
    def positions(self):
        return self._data[1]

    @property
    def parents(self):
        return self._data[2]

    @property
    def status(self):
        return self._data[3]


class BatchTree:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        max_len: int,
        batch_size: int,
    ):
        self._logger = log.get_logger()

        self._device = device
        self._dtype = dtype
        self._max_len = max_len
        self.batch_size = batch_size

        self.PROMPT = torch.tensor(0, device=self._device)
        self.GENERATED = torch.tensor(5, device=self._device)
        self.PROCESSED = torch.tensor(10, device=self._device)
        self.CANDIDATE = torch.tensor(15, device=self._device)


        self.end = torch.zeros(
            (self.batch_size,), dtype=torch.long, device=self._device
        )
        self.prefix_len = torch.zeros(
            (self.batch_size,), dtype=torch.long, device=self._device
        )


        self._data = torch.zeros(
            (self.batch_size, 4, self._max_len), dtype=torch.long, device=self._device
        )

        self.logprobs = torch.zeros(
            (self.batch_size, self._max_len), dtype=torch.float32, device=self._device
        )

        self.amask = torch.zeros(
            (self.batch_size, 1, self._max_len, self._max_len),
            dtype=self._dtype,
            device=self._device,
        )

    def add_request(self, prefix_tokens: torch.Tensor, batch_idx: int):

        prefix_len = prefix_tokens.numel()


        self._data[batch_idx].zero_()
        self.prefix_len[batch_idx].zero_()
        self.end[batch_idx].zero_()
        self.amask[batch_idx].zero_()
        self.logprobs[batch_idx].zero_()


        self.tokens[batch_idx, :prefix_len] = prefix_tokens
        self.positions[batch_idx, :prefix_len] = torch.arange(
            prefix_len, dtype=torch.long, device=self._device
        )
        self.parents[batch_idx, 1:prefix_len] = torch.arange(
            prefix_len - 1, device=self._device
        )
        self.status[batch_idx, : prefix_len - 1] = self.PROMPT
        self.status[batch_idx, prefix_len - 1] = self.CANDIDATE

        self.prefix_len[batch_idx] = prefix_len
        self.end[batch_idx] = prefix_len


        self.amask[batch_idx, 0, :prefix_len, :prefix_len] = torch.ones(
            (prefix_len, prefix_len), device=self._device
        ).tril_()

    def remove_requests(self, batch_indices: torch.Tensor):
        ...

    def add(
        self,
        batch_indices: torch.Tensor,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor,
        parent_indices: torch.Tensor,
        logprobs: torch.Tensor,
        token_status: Optional[torch.Tensor] = None,
    ):
        token_status = self.CANDIDATE if token_status is None else token_status


        if not (
            token_ids.shape
            == token_positions.shape
            == parent_indices.shape
            == logprobs.shape
        ):
            raise ValueError(
                "Inconsistent sizes of input tensors: %s, %s, %s, %s",
                token_ids.shape,
                token_positions.shape,
                parent_indices.shape,
                logprobs.shape,
            )

        seq_indices = torch.zeros(
            batch_indices.shape[-1], device=self._device, dtype=torch.long
        )
        unique_batches, cnts = torch.unique(batch_indices, return_counts=True)

        for b_idx in unique_batches:
            positions = (batch_indices == b_idx).nonzero(as_tuple=True)[0]
            seq_indices[positions] = self.end[b_idx] + torch.arange(
                positions.numel(), device=self._device
            )

        self.tokens[batch_indices, seq_indices] = token_ids
        self.positions[batch_indices, seq_indices] = token_positions
        self.parents[batch_indices, seq_indices] = parent_indices
        self.status[batch_indices, seq_indices] = token_status
        self.logprobs[batch_indices, seq_indices] = logprobs

        self.amask[batch_indices, 0, seq_indices, :] = self.amask[
            batch_indices, 0, parent_indices, :
        ]
        self.amask[batch_indices, 0, seq_indices, seq_indices] = True

        self.end[unique_batches] += cnts

    def gather(
        self,
        batch_idx: int,
        src_indices: torch.Tensor,
        dest_indices: torch.Tensor,
    ):
        interim_indices = torch.arange(self.end[batch_idx], device=self._device)  # type: ignore
        interim_indices[src_indices] = dest_indices
        self.parents[batch_idx, src_indices] = interim_indices[
            self.parents[batch_idx, src_indices]
        ]

        self._data[batch_idx, :, dest_indices] = self._data[batch_idx, :, src_indices]
        self.logprobs[batch_idx, dest_indices] = self.logprobs[batch_idx, src_indices]
        self.amask[batch_idx, :, dest_indices, :] = self.amask[
            batch_idx, :, src_indices, :
        ]
        self.amask[batch_idx, ..., dest_indices] = self.amask[
            batch_idx, ..., src_indices
        ]

        self.end[batch_idx] = dest_indices.max() + 1

        self.amask[batch_idx, :, self.end[batch_idx] :, :].zero_()
        self.amask[batch_idx, ..., self.end[batch_idx] :].zero_()
        self._data[batch_idx, :, self.end[batch_idx] :].zero_()
        self.logprobs[batch_idx, self.end[batch_idx] :].zero_()
        self.status[batch_idx, self.end[batch_idx] :].zero_()

    def take_snapshot(self, name: str, batch_idx: int, tokenizer):
        tokens = {}

        for idx in range(self.prefix_len[batch_idx] - 2, self.end[batch_idx]):
            tokens[idx] = {
                "token": tokenizer.decode(
                    self.tokens[batch_idx, idx].item(), special_tokens=False
                ),
                "id": self.tokens[batch_idx, idx].item(),
                "parent": self.parents[batch_idx, idx].item(),
                "status": self.status[batch_idx, idx].item(),
            }

        return {
            "type": "tree",
            "name": name,
            "batch_idx": batch_idx,
            "prefix": tokenizer.decode(
                self.tokens[batch_idx, : self.prefix_len[batch_idx]]
            ),
            "tokens": tokens,
        }

    def reorder(
        self, batch_idx: int, src_indices: torch.Tensor, dest_indices: torch.Tensor
    ):
        self.gather(batch_idx, src_indices, dest_indices)
        self.end[batch_idx] = dest_indices.size(-1)
        self.prefix_len[batch_idx] = self.end[batch_idx]
        self.status[batch_idx, : self.end[batch_idx]] = self.GENERATED

    @property
    def tokens(self):
        return self._data[:, 0, :]

    @property
    def positions(self):
        return self._data[:, 1, :]

    @property
    def parents(self):
        return self._data[:, 2, :]

    @property
    def status(self):
        return self._data[:, 3, :]
