"""A compact Transformer encoder-decoder implemented with PyTorch modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        q = self._shape(self.q_proj(query), batch_size)
        k = self._shape(self.k_proj(key), batch_size)
        v = self._shape(self.v_proj(value), batch_size)

        # Scaled dot-product attention: each head attends over the key sequence.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # Mask is True for positions that are allowed to attend.
            scores = scores.masked_fill(~mask.to(torch.bool), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)

    def _shape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        *,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_position = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.tgt_position = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, self.pad_idx)
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt_input, memory, tgt_mask, src_mask)
        return self.generator(decoded)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_position(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)
        x = self.tgt_position(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)


def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    # Shape: [batch, 1, 1, src_len], broadcast over heads and target positions.
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # Causal mask prevents the decoder from seeing future target tokens.
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return pad_mask.expand(batch_size, 1, tgt_len, tgt_len) & causal_mask


@torch.no_grad()
def greedy_decode(
    model: TransformerNMT,
    src: torch.Tensor,
    *,
    bos_idx: int,
    eos_idx: int,
    max_len: int,
) -> torch.Tensor:
    model.eval()
    src_mask = make_src_mask(src, model.pad_idx)
    memory = model.encode(src, src_mask)
    generated = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        # Feed the generated prefix back into the decoder one token at a time.
        tgt_mask = make_tgt_mask(generated, model.pad_idx)
        decoded = model.decode(generated, memory, tgt_mask, src_mask)
        logits = model.generator(decoded[:, -1])
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_idx), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(eos_idx)
        if finished.all():
            break
    return generated
