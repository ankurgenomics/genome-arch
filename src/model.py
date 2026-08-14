"""
genome-arch architecture: multi-scale block convolutions + sparse attention + an
approximate reverse-complement (RC) handling scheme, trained from random
initialization.

This is a from-scratch reimplementation of the ideas described in the README
(inspired by, not a port of, Omnii / StripedHyena2 / Caduceus). Design choices
made where the source architectures aren't public:

- BlockConv1d: each block along the sequence gets its own dense weight matrix
  instead of one kernel shared across the whole sequence (the "remove the
  Toeplitz restriction" property described for Omnii's block convolutions).
- Multi-scale fusion: standard Conv1d at three kernel sizes (short/medium/long),
  combined with the block convolution via a learned per-position gate, rather
  than a fixed concatenation.
- TopKSparseAttention: attention restricted to each query's top-k highest-scoring
  keys, a standard, well-understood way to implement "sparse" attention. This is
  a simplification of Omnii's "dynamic sparse attention" (which is undocumented
  beyond the public preview), not a claim of matching it exactly.
- Reverse-complement handling: the same encoder (shared weights) is run on the
  sequence and on its reverse complement, and the two representations are
  combined. This is a simpler, weaker guarantee than Caduceus's RC-equivariant
  Mamba blocks (which build the symmetry into the operator itself), documented
  honestly as an approximation, not a claim of matching Caduceus's construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Nucleotide encoding: A=0, C=1, G=2, T=3, PAD=4. A dedicated pad index matters
# because real sequence lengths (e.g. GenomicBenchmarks' 251bp promoters) don't
# divide evenly into every block size; padding with a real base like "A" would
# silently bias the model, since 251 is prime and needs padding for any block
# size other than 1 or 251.
VOCAB_SIZE = 5
PAD_IDX = 4

# Complement lookup: A(0)<->T(3), C(1)<->G(2), PAD(4)->PAD(4). Used instead of
# the earlier `3 - i` formula so padding survives reverse-complementing intact.
_COMPLEMENT = torch.tensor([3, 2, 1, 0, 4])


class BlockConv1d(nn.Module):
    """Position-dependent dense block convolution.

    Splits the sequence into non-overlapping blocks and applies a distinct
    dense weight matrix per block position (not shared across positions, unlike
    a standard convolution's shared kernel). Trades translation invariance for
    the ability to specialize by position.
    """

    def __init__(self, in_channels: int, out_channels: int, block_size: int, num_blocks: int):
        super().__init__()
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels

        fan_in = block_size * in_channels
        self.weight = nn.Parameter(torch.randn(num_blocks, fan_in, block_size * out_channels) / fan_in**0.5)
        self.bias = nn.Parameter(torch.zeros(num_blocks, block_size * out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_channels), seq_len must equal num_blocks * block_size
        b, seq_len, c = x.shape
        expected = self.num_blocks * self.block_size
        if seq_len != expected:
            raise ValueError(f"BlockConv1d expected seq_len={expected}, got {seq_len}")

        x = x.reshape(b, self.num_blocks, self.block_size * c)
        out = torch.einsum("bnk,nko->bno", x, self.weight) + self.bias
        out = out.reshape(b, self.num_blocks * self.block_size, self.out_channels)
        return out


class TopKSparseAttention(nn.Module):
    """Multi-head self-attention restricted to each query's top-k keys.

    A standard sparse-attention construction: compute full scaled dot-product
    scores, keep only the top-k per query row, renormalize with softmax over
    just those, zero out the rest. Concentrates long-range compute on the
    positions that actually score highly instead of attending uniformly.
    """

    def __init__(self, dim: int, num_heads: int, topk: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk = topk
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (b, heads, n, head_dim)

        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5  # (b, heads, n, n)

        k_eff = min(self.topk, n)
        topk_vals, topk_idx = scores.topk(k_eff, dim=-1)
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)

        attn = F.softmax(sparse_scores, dim=-1)
        out = attn @ v  # (b, heads, n, head_dim)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class MultiScaleBlock(nn.Module):
    """Short/medium/long convolutions plus a block convolution, gated and fused."""

    def __init__(self, dim: int, seq_len: int, block_size: int):
        super().__init__()
        if seq_len % block_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by block_size ({block_size})")
        num_blocks = seq_len // block_size

        self.conv_short = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(dim, dim, kernel_size=15, padding=7)
        self.conv_long = nn.Conv1d(dim, dim, kernel_size=65, padding=32)
        self.block_conv = BlockConv1d(dim, dim, block_size, num_blocks)
        self.gate = nn.Linear(dim * 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        x_t = x.transpose(1, 2)  # (b, dim, seq_len) for Conv1d
        h_short = self.conv_short(x_t).transpose(1, 2)
        h_medium = self.conv_medium(x_t).transpose(1, 2)
        h_long = self.conv_long(x_t).transpose(1, 2)
        h_block = self.block_conv(x)

        stacked = torch.stack([h_block, h_short, h_medium, h_long], dim=-2)  # (b, n, 4, dim)
        gate_logits = self.gate(torch.cat([h_block, h_short, h_medium, h_long], dim=-1))  # (b, n, 4)
        gate = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (b, n, 4, 1)
        return (stacked * gate).sum(dim=-2)  # (b, n, dim)


class GenomeArchModel(nn.Module):
    """Full model: embedding -> multi-scale block conv -> sparse attention -> head.

    Runs the encoder on both the input sequence and its reverse complement
    (shared weights) and averages the two representations, as an approximate
    reverse-complement handling scheme (see module docstring for the honest
    caveat vs. Caduceus's stronger equivariance guarantee).
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int = 32,
        block_size: int = 16,
        num_heads: int = 4,
        topk: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.multi_scale = MultiScaleBlock(embed_dim, seq_len, block_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, topk)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def reverse_complement(x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) LongTensor of nucleotide indices in {0,1,2,3,4=PAD}.
        comp = _COMPLEMENT.to(x.device)[x]
        return comp.flip(dims=[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.norm1(self.multi_scale(h) + h)
        h = self.norm2(self.attn(h) + h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.encode(x)
        rc = self.encode(self.reverse_complement(x))
        rc_aligned = rc.flip(dims=[1])  # align RC positions back to forward-strand order
        combined = (fwd + rc_aligned) / 2
        pooled = combined.mean(dim=1)
        return self.head(pooled)
