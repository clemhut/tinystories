from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer

from src.dpnlp_pipeline.io import load_json, save_json

EOT_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class TokenizerSpec:
    pretrained_name: str = "EleutherAI/gpt-neo-125M"
    vocab_size: int = 10_000


class RestrictedTokenizer:
    def __init__(self, backend_tokenizer, kept_token_ids: list[int]):
        self._tokenizer = backend_tokenizer
        self._kept_token_ids = list(kept_token_ids)
        self._limit = len(self._kept_token_ids)
        self._old_to_new_id = {token_id: index for index, token_id in enumerate(self._kept_token_ids)}
        self._new_to_old_id = {index: token_id for index, token_id in enumerate(self._kept_token_ids)}
        self.unk_token_id = self._limit
        self.unk_token = "<unk>"
        self._special_id_map = self._build_special_id_map()
        self._reverse_special_id_map = {target_id: source_id for source_id, target_id in self._special_id_map.items()}

    def _build_special_id_map(self) -> dict[int, int]:
        special_ids = []
        for token_id in (
            getattr(self._tokenizer, "bos_token_id", None),
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._tokenizer, "pad_token_id", None),
        ):
            if token_id is not None and token_id not in special_ids:
                special_ids.append(token_id)

        next_id = self.unk_token_id + 1
        mapping: dict[int, int] = {}
        for token_id in special_ids:
            if token_id in self._old_to_new_id:
                mapping[token_id] = self._old_to_new_id[token_id]
            else:
                mapping[token_id] = next_id
                next_id += 1
        return mapping

    def _restrict_ids(self, input_ids: list[int]) -> list[int]:
        restricted: list[int] = []
        for token_id in input_ids:
            if token_id in self._old_to_new_id:
                restricted.append(self._old_to_new_id[token_id])
            elif token_id in self._special_id_map:
                restricted.append(self._special_id_map[token_id])
            else:
                restricted.append(self.unk_token_id)
        return restricted

    def tokenize_to_ids(self, text: str, return_tensor: bool = False) -> list[int] | Tensor:
        encoded = self._tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]
        restricted = self._restrict_ids(encoded)
        if return_tensor:
            return torch.tensor([restricted], dtype=torch.long)
        return restricted

    def decode(self, input_ids: list[int]) -> str:
        pieces: list[str] = []
        backend_buffer: list[int] = []

        def flush_backend_buffer() -> None:
            if backend_buffer:
                pieces.append(self._tokenizer.decode(backend_buffer))
                backend_buffer.clear()

        for token_id in input_ids:
            if token_id == self.unk_token_id:
                flush_backend_buffer()
                pieces.append(self.unk_token)
            elif token_id in self._reverse_special_id_map:
                backend_buffer.append(self._reverse_special_id_map[token_id])
            elif token_id in self._new_to_old_id:
                backend_buffer.append(self._new_to_old_id[token_id])
            else:
                flush_backend_buffer()
                pieces.append(self.unk_token)

        flush_backend_buffer()
        return "".join(pieces)

    def get_vocab_size(self) -> int:
        if self._special_id_map:
            return max([self.unk_token_id, *self._special_id_map.values()]) + 1
        return self.unk_token_id + 1

    def get_special_tokens(self) -> dict:
        return self._tokenizer.special_tokens_map

    def get_eos_token(self) -> str:
        return self._tokenizer.eos_token

    def get_bos_token(self) -> str:
        return self._tokenizer.bos_token

    def get_pad_token(self) -> str:
        return self._tokenizer.pad_token

    def get_pad_token_id(self) -> int:
        pad_token_id = self._tokenizer.pad_token_id
        return self._special_id_map.get(pad_token_id, pad_token_id)


def iter_story_chunks(path: Path, delimiter: str = EOT_TOKEN, chunk_size: int = 1024 * 1024):
    buffer = ""
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            start = 0
            while True:
                index = buffer.find(delimiter, start)
                if index == -1:
                    buffer = buffer[start:]
                    break
                end = index + len(delimiter)
                yield buffer[start:end]
                start = end
    if buffer:
        yield buffer


def build_restricted_vocab_payload(
    train_path: Path,
    backend_tokenizer,
    pretrained_name: str,
    vocab_size: int,
) -> dict:
    special_token_ids = {
        token_id
        for token_id in (
            getattr(backend_tokenizer, "bos_token_id", None),
            getattr(backend_tokenizer, "eos_token_id", None),
            getattr(backend_tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }

    counts: Counter[int] = Counter()
    for story_text in iter_story_chunks(train_path):
        input_ids = backend_tokenizer(story_text, truncation=False, return_attention_mask=False)["input_ids"]
        counts.update(token_id for token_id in input_ids if token_id not in special_token_ids)

    kept_token_ids = [
        token_id
        for token_id, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:vocab_size]
    ]
    return {
        "pretrained_name": pretrained_name,
        "vocab_size": vocab_size,
        "kept_token_ids": kept_token_ids,
    }


def _payload_matches_spec(payload: dict, spec: TokenizerSpec) -> bool:
    return payload.get("pretrained_name") == spec.pretrained_name and int(payload.get("vocab_size", -1)) == spec.vocab_size


def build_tokenizer(
    spec: TokenizerSpec | None = None,
    train_path: Path | None = None,
    vocab_path: Path | None = None,
) -> RestrictedTokenizer:
    spec = spec or TokenizerSpec()
    tokenizer = AutoTokenizer.from_pretrained(spec.pretrained_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    payload: dict | None = None
    if vocab_path is not None and vocab_path.exists():
        cached_payload = load_json(vocab_path)
        if isinstance(cached_payload, dict) and _payload_matches_spec(cached_payload, spec):
            payload = cached_payload

    if payload is None:
        if train_path is None:
            raise ValueError("train_path is required when no compatible restricted vocab artifact is available")
        payload = build_restricted_vocab_payload(train_path, tokenizer, spec.pretrained_name, spec.vocab_size)
        if vocab_path is not None:
            save_json(vocab_path, payload)

    return RestrictedTokenizer(tokenizer, kept_token_ids=list(payload["kept_token_ids"]))
