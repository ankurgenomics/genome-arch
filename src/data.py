"""Data loading for genome-arch.

GenomicBenchmarks loading uses the same Hugging Face `datasets` hub source as
genome-ft (`katarinagresova/Genomic_Benchmarks_*`), for the sanity-check stage
described in the README. Verified working against the live hub in this
session: `human_nontata_promoters` sequences are fixed at 251bp.

ClinVar and multi-species conservation (UCSC Cactus) loaders for the primary
task are not implemented yet -- this covers the sanity-check stage only.
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.model import PAD_IDX

NUCLEOTIDE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def pad_to_multiple(seq_len: int, block_size: int) -> int:
    """Smallest length >= seq_len that's divisible by block_size."""
    remainder = seq_len % block_size
    return seq_len if remainder == 0 else seq_len + (block_size - remainder)


def encode_sequence(seq: str, target_len: int) -> list[int]:
    """Encode a DNA string to indices, right-padded with PAD_IDX to target_len.

    Unknown bases (N, etc.) fall back to index 0 (A) rather than crashing --
    GenomicBenchmarks sequences are occasionally not pure ACGT.
    """
    seq = seq.upper()
    idx = [NUCLEOTIDE_TO_IDX.get(b, 0) for b in seq[:target_len]]
    idx += [PAD_IDX] * (target_len - len(idx))
    return idx


def load_genomic_benchmark(task_name: str, block_size: int) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int
]:
    """Load a GenomicBenchmarks task, returning ((x_train, y_train), (x_test,
    y_test), seq_len). seq_len is padded up to the nearest multiple of
    block_size so it's compatible with BlockConv1d.
    """
    from datasets import load_dataset

    ds = load_dataset(f"katarinagresova/Genomic_Benchmarks_{task_name}")
    raw_len = max(len(s) for s in ds["train"]["seq"])
    seq_len = pad_to_multiple(raw_len, block_size)

    def encode_split(split) -> Tuple[torch.Tensor, torch.Tensor]:
        x = [encode_sequence(s, seq_len) for s in split["seq"]]
        y = list(split["label"])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    return encode_split(ds["train"]), encode_split(ds["test"]), seq_len
