import math
import torch
from torch import nn, Tensor


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self._n = n_heads
        self._d = d_model
        self._rsqrt = 1.0 / math.sqrt(d_model // n_heads)

        self.qkv_linear = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.projection = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        bs, sl, dm = x.size()

        q, k, v = self.qkv_linear(x).split(self._d, dim=2)
        q = q.view(bs, sl, self._n, -1).transpose(-3, -2)
        k = k.view(bs, sl, self._n, -1).transpose(-3, -2)
        v = v.view(bs, sl, self._n, -1).transpose(-3, -2)

        attn = (q @ k.transpose(-2, -1)) * self._rsqrt
        attn = attn.masked_fill(self.mask[:, :, :sl, :sl] == 0, float("-inf"))
        attn = attn.softmax(-1)
        attn = self.dropout(attn)

        output = attn @ v
        output = output.transpose(-3, -2).contiguous().view(bs, sl, dm)
        output = self.projection(output)

        return output


class PointwiseFeedForward(nn.Sequential):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(dropout),
        )


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.mha_norm = nn.LayerNorm(d_model, bias=bias)
        self.mha = CausalMultiheadSelfAttention(
            d_model,
            n_heads,
            max_seq_len,
            dropout,
            bias,
        )

        self.mlp_norm = nn.LayerNorm(d_model, bias=bias)
        self.mlp = PointwiseFeedForward(d_model, d_ff, dropout, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.mha_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Gpt(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 6,
        d_model: int = 192,
        n_heads: int = 6,
        d_ff: int = 768,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(
            *[
                Block(
                    d_model,
                    n_heads,
                    d_ff,
                    max_seq_len,
                    dropout,
                    bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)

        self.register_buffer("timesteps", torch.arange(max_seq_len).view(1, -1))
        self.apply(self._initialize_weights)
        for name, param in self.named_parameters():
            if name.endswith("CausalMultiheadSelfAttention.projection.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, seqs: Tensor) -> Tensor:
        pos = self.timesteps.expand(seqs.size(0), -1).to(seqs.device)

        seq_emb = self.token_embedding(seqs)
        pos_emb = self.position_embedding(pos[:, :seqs.size(1)])
        x = self.dropout(seq_emb + pos_emb)

        x = self.blocks(x)
        x = self.norm(x)

        logits = self.lm_head(x)

        return logits

    def _initialize_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
